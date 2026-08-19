"""Assembly of final ms-swift samples from SampleSpec + step images.

Handles: WebP processing with coordinate rebasing, task/history/turn message
construction, token budget fitting, content-gated reasoning attachment, and
POSIX image paths. Action points are ALWAYS rebased to the final image size
of the step's own screenshot.
"""
from __future__ import annotations

import os
import re
import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .coordinates import Action, rebase_action_points
from .native_interface import (coordinate_free_action, interface_layout,
                               render_action_for_contract)
from .images import (DEFAULT_MAX_LONG, ProcessedImage, compute_target_long_side,
                     load_image, process_image, save_image)
from .reasoning import ReasoningGate, compose_assistant_target, detect_category
from .quality import (QualityScore, attach_quality, finish_has_evidence,
                      score_action_step, score_grounding as score_grounding_q)
from .token_budget import estimate_sequence
from .windows import CHUNK, SINGLE, WINDOW, SampleSpec, Step, Trajectory

SYSTEM_PROMPT = "You are a computer-use agent."

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_id(text: str, max_len: int = 96) -> str:
    out = _SANITIZE_RE.sub("_", (text or "x").strip())[:max_len]
    return out or "x"




def stable_image_name(trajectory_id: str, step_ids, suffix: str = ".webp") -> str:
    """Collision-resistant deterministic image artifact name.

    Human-readable prefixes are truncated for portability, while the digest is
    computed from the full untruncated identifiers. This prevents two long
    source IDs with the same first 80/96 characters from overwriting each
    other's image bytes.
    """
    if isinstance(step_ids, str):
        ids = [step_ids]
    else:
        ids = [str(x) for x in (step_ids or [])]
    payload = json.dumps([str(trajectory_id), ids], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return f"{sanitize_id(str(trajectory_id), 56)}__{digest}{suffix}"

def decide_max_long(spec: SampleSpec) -> int:
    # Grounding and action supervision both benefit from preserving genuinely
    # small controls. Source adapters may provide the trusted target bbox on
    # the anchor Step.
    target_w = (spec.current_step.metadata or {}).get("target_width_px")
    target_h = (spec.current_step.metadata or {}).get("target_height_px")
    if target_w is None:
        target_w = spec.metadata.get("target_width_px")
    if target_h is None:
        target_h = spec.metadata.get("target_height_px")
    if spec.task_type == "grounding" or target_w is not None or target_h is not None:
        return compute_target_long_side(spec.current_step.image_size, target_w, target_h)
    return DEFAULT_MAX_LONG


def _save_step_image(step: Step, source: str, trajectory_id: str, ctx,
                     max_long: int, quality: int):
    """Process+save one step screenshot; returns (rel_path, w, h) or None."""
    try:
        processed = process_image(load_image(step.image_bytes),
                                  max_long=max_long, quality=quality)
    except Exception:
        return None
    name = f"{sanitize_id(trajectory_id)}__{sanitize_id(step.step_id)}.webp"
    rel = f"images/{source}/{name}"
    save_image(processed.data, os.path.join(ctx.dataset_root, "images", source, name))
    return rel, processed.width, processed.height


def _rebaser(step: Step, processed_sizes: Dict[str, Tuple[int, int]]):
    def final_action_text(s: Step) -> str:
        if s.action is None:
            return ""
        final_size = processed_sizes.get(s.step_id)
        action = s.action
        if final_size and tuple(s.image_size) != tuple(final_size):
            action = rebase_action_points(
                Action(verb=s.action.verb, args=dict(s.action.args),
                       points=list(s.action.points), original=s.action.original,
                       original_space=s.action.original_space,
                       original_points=list(s.action.original_points)),
                s.image_size, final_size)
        return action.render()
    return final_action_text


def _history_step_max_long(step: Step) -> int:
    md = step.metadata or {}
    tw, th = md.get("target_width_px"), md.get("target_height_px")
    if tw is not None or th is not None:
        return compute_target_long_side(step.image_size, tw, th)
    return DEFAULT_MAX_LONG


def _format_native_user(contract: Optional[dict], *, image: str, task: str,
                        history: str = "", with_task: bool = True,
                        visual: bool = False) -> str:
    if not contract:
        parts = [image]
        if task:
            parts.append(f"Task: {task}")
        if history:
            parts.append("Previous actions:\n" + history)
        return "\n".join(parts)
    layout = contract.get("message_layout") or {}
    if visual:
        key = "visual_user_with_task_template" if with_task else "visual_user_without_task_template"
        template = layout.get(key)
    else:
        template = layout.get("text_user_template")
    if not isinstance(template, str):
        raise ValueError(f"native interface missing {key if visual else 'text_user_template'}")
    return template.format(image=image, task=task if with_task else "", history=history)


def assemble_sample(spec: SampleSpec, ctx, quality: int = 80,
                    trajectory: Optional[Trajectory] = None) -> Optional[dict]:
    budget = ctx.config.get("context_budget", 8192)
    contract = ctx.config.get("_native_interface_contract")
    layout = interface_layout(contract)
    system_prompt = layout["system_prompt"]
    image_placeholder = layout["image_placeholder"]
    max_long = decide_max_long(spec)

    anchor_processed = process_image(load_image(spec.current_step.image_bytes),
                                     max_long=max_long, quality=quality, lossless=True)
    image_dir = os.path.join(ctx.dataset_root, "images", spec.source)
    anchor_name = stable_image_name(spec.trajectory_id, spec.step_ids)
    anchor_rel = f"images/{spec.source}/{anchor_name}"

    # Pending image bytes are committed only after all semantic, token and
    # quality gates pass. Visual history reuses stable per-step paths.
    pending_images: List[Tuple[ProcessedImage, str]] = [(anchor_processed, anchor_name)]
    images: List[str] = []
    image_sizes: List[Tuple[int, int]] = []

    hist = [h for h in spec.history_texts if h and h.strip()]
    plan = None
    if spec.task_type == "action":
        plan = ctx.reasoning_gate.allow(sorted(spec.signals), spec.trajectory_id,
                                        spec.current_step.step_id)

    try:
        target, canonical_target = render_action_for_contract(
            spec.current_step.action, spec.current_step.image_size,
            (anchor_processed.width, anchor_processed.height), contract)
    except Exception:
        ctx.reject(spec.source, "native_action_render_failed")
        return None
    if not target:
        return None

    messages: List[Dict[str, object]] = [
        {"role": "system", "content": system_prompt, "loss": False}
    ]
    history_policy = (contract or {}).get("history_policy") or {"mode": "text_actions"}
    history_mode = history_policy.get("mode", "text_actions")
    visual_prev: List[Step] = []

    if history_mode == "visual_recent_rounds" and contract:
        n = int(history_policy.get("recent_visual_rounds") or 0)
        visual_prev = list(spec.metadata.get("_prev_steps") or [])[-n:]
        recent_texts = [st.action_text for st in visual_prev if st.action_text]
        older_hist = list(hist)
        # Remove matching recent suffix from the text summary so the same action
        # is not repeated both visually and textually.
        for t in reversed(recent_texts):
            if older_hist and older_hist[-1] == t:
                older_hist.pop()
        older_policy = history_policy.get("older_actions", "omit")
        if older_policy == "coordinate_free":
            older_hist = [coordinate_free_action(x) for x in older_hist]
        elif older_policy == "omit":
            older_hist = []
        elif older_policy != "full":
            ctx.reject(spec.source, "unsupported_native_history_policy")
            return None

        older_summary = ""
        if older_hist:
            template = (contract.get("message_layout") or {}).get("older_history_template")
            item_template = (contract.get("message_layout") or {}).get("history_item_template") or "{action}"
            if not isinstance(template, str):
                ctx.reject(spec.source, "native_history_template_missing")
                return None
            rendered_items = "\n".join(item_template.format(action=x) for x in older_hist)
            older_summary = template.format(history=rendered_items)

        task_location = history_policy.get("task_location")
        for i, st in enumerate(visual_prev):
            try:
                proc = process_image(load_image(st.image_bytes),
                                     max_long=_history_step_max_long(st),
                                     quality=quality, lossless=True)
                name = stable_image_name(spec.trajectory_id, [st.step_id])
                rel = f"images/{spec.source}/{name}"
                prev_target, _ = render_action_for_contract(
                    st.action, st.image_size, (proc.width, proc.height), contract)
            except Exception:
                ctx.reject(spec.source, "native_visual_history_render_failed")
                return None
            with_task = task_location in {"first_user", "all_user"} and i == 0 or task_location == "all_user"
            user_content = _format_native_user(
                contract, image=image_placeholder,
                task=spec.task if with_task else "", history="",
                with_task=with_task, visual=True)
            if i == 0 and older_summary:
                # Exact placement is contract-controlled. The template can use
                # {history}; if it cannot, fail rather than inventing layout.
                loc = history_policy.get("older_summary_location", "first_user")
                if loc == "first_user":
                    user_content = user_content + older_summary
                elif loc != "current_user":
                    ctx.reject(spec.source, "unsupported_older_summary_location")
                    return None
            messages.append({"role": "user", "content": user_content, "loss": False})
            messages.append({"role": "assistant", "content": prev_target, "loss": False})
            images.append(rel)
            image_sizes.append((proc.width, proc.height))
            pending_images.append((proc, name))

        current_with_task = task_location in {"current_user", "all_user"} or not visual_prev
        current_user = _format_native_user(
            contract, image=image_placeholder,
            task=spec.task if current_with_task else "", history="",
            with_task=current_with_task, visual=True)
        if older_summary and history_policy.get("older_summary_location") == "current_user":
            current_user = current_user + older_summary
        messages.append({"role": "user", "content": current_user, "loss": False})
        images.append(anchor_rel)
        image_sizes.append((anchor_processed.width, anchor_processed.height))
    else:
        history_cap = 12 if spec.representation == "chunk" else 8
        kept_hist = hist[-history_cap:]
        if contract:
            item_template = (contract.get("message_layout") or {}).get("history_item_template")
            if kept_hist and not isinstance(item_template, str):
                ctx.reject(spec.source, "native_history_item_template_missing")
                return None
            history_text = "\n".join(item_template.format(action=h) for h in kept_hist) if kept_hist else ""
        else:
            history_text = "\n".join(f"- {h}" for h in kept_hist)
        user_content = _format_native_user(
            contract, image=image_placeholder, task=spec.task,
            history=history_text, with_task=True, visual=False)
        messages.append({"role": "user", "content": user_content, "loss": False})
        images.append(anchor_rel)
        image_sizes.append((anchor_processed.width, anchor_processed.height))

    messages.append({"role": "assistant",
                     "content": compose_assistant_target(target, plan), "loss": True})

    est = estimate_sequence(
        user_text="\n".join(str(m["content"]) for m in messages if m["role"] == "user"),
        assistant_target="\n".join(str(m["content"]) for m in messages if m["role"] == "assistant"),
        image_sizes=image_sizes,
        system_text=system_prompt,
    )
    if est > budget:
        ctx.state.add_rejection(spec.source, "over_context_budget")
        return None

    signals = sorted(spec.signals)
    last_assistant = [m for m in messages if m["role"] == "assistant"][-1]["content"]
    metadata = {
        "representation": spec.representation,
        "estimated_tokens": est,
        "image_count": len(images),
        "history_steps": len(hist),
        "visual_history_rounds": len(visual_prev),
        "window_length": len(spec.step_ids),
        "original_image_size": list(spec.current_step.image_size),
        "final_image_size": [anchor_processed.width, anchor_processed.height],
        "signals": signals,
        "app": spec.app,
        "reasoning_category": detect_category(signals) if str(last_assistant).startswith("Plan:") else None,
        "action_original": spec.current_step.action.original if spec.current_step.action else None,
        "action_original_space": (spec.current_step.action.original_space.value
                                  if spec.current_step.action and spec.current_step.action.original_space else None),
        "action_canonical_final_pixels": canonical_target,
        "image_encoding": "webp_lossless",
        "native_interface": bool(contract),
        "interface_coordinate_space": ((contract.get("coordinate_space") or {}).get("type")
                                       if contract else "processed_image_pixels"),
        "interface_history_mode": history_mode,
        "interface_image_placeholder": image_placeholder,
        "assistant_target_sha256": hashlib.sha256(str(messages[-1]["content"]).encode("utf-8")).hexdigest(),
    }
    if "recovery_evidenced" in spec.signals:
        metadata["recovery_verified"] = True

    # Propagate only explicit, non-private source metadata used by selection,
    # splitting and hard verification. Internal keys remain local.
    for _k in ("group_id", "collection_run", "task_family", "content_family",
               "target_width_px", "target_height_px", "target_bbox",
               "bbox_click_validated", "bbox_center_offset_norm",
               "explicit_success", "reliable_final_state", "verifier_evidence"):
        if _k in (spec.current_step.metadata or {}):
            metadata[_k] = spec.current_step.metadata[_k]
        elif _k in spec.metadata:
            metadata[_k] = spec.metadata[_k]
    if spec.current_step.action is not None and spec.current_step.action.verb == "finish":
        step_meta = spec.current_step.metadata or {}
        evidenced = finish_has_evidence(
            explicit_success=bool(step_meta.get("explicit_success") or spec.metadata.get("explicit_success")),
            reliable_final_state=bool(step_meta.get("reliable_final_state") or spec.metadata.get("reliable_final_state")),
            verifier_evidence=bool(step_meta.get("verifier_evidence") or spec.metadata.get("verifier_evidence")))
        metadata["finish_evidence"] = "yes" if evidenced else "no"
        if not evidenced:
            ctx.reject(spec.source, "finish_without_evidence")
            return None

    sample = {
        "messages": messages,
        "images": images,
        "source": spec.source,
        "trajectory_id": spec.trajectory_id,
        "step_id": (spec.step_ids[-1] if len(spec.step_ids) == 1
                    else f"{spec.step_ids[0]}..{spec.step_ids[-1]}"),
        "task_type": spec.task_type,
        "metadata": metadata,
    }
    spec.metadata.pop("_window_steps", None)
    verb = spec.current_step.action.verb if spec.current_step.action else ""
    prev_texts = [traj.action_text for traj in
                  (spec.metadata.get("_prev_steps") or [])]
    qs = score_action_step(
        verb=verb, task_text=spec.task, subgoal=spec.current_step.subgoal,
        signals=sorted(spec.signals), app=spec.app,
        app_counter=ctx.app_counter, total_samples=ctx.total_samples,
        is_first_step=spec.metadata.get("_is_first_step", False),
        repeated_identical=len(prev_texts) >= 1 and prev_texts[-1] == spec.current_step.action_text
        and "no_state_change" not in spec.signals,
        state_change_from_prev=spec.metadata.get("_state_changed", True),
        prev_was_ineffective="recovery_evidenced" in spec.signals,
        target_width_px=(spec.current_step.metadata or {}).get("target_width_px"),
        bbox_center_offset_norm=(spec.current_step.metadata or {}).get("bbox_center_offset_norm"),
        representation=spec.representation)
    if qs.bucket == "Reject":
        ctx.reject(spec.source, f"quality:{qs.reject_reason}")
        return None
    attach_quality(sample, qs, est)
    for processed, name in pending_images:
        save_image(processed.data, os.path.join(image_dir, name))
    if spec.task_type == "action":
        ctx.reasoning_gate.note_action_sample()
        ctx.note_app(spec.app)
    return sample

def _assemble_no_extra(spec: SampleSpec, ctx, quality: int = 80) -> Optional[dict]:
    spec.extra_images = []
    spec.metadata["_window_steps"] = [spec.current_step]
    return assemble_sample(spec, ctx, quality)


def assemble_grounding(*, source: str, trajectory_id: str, step_id: str,
                       image_bytes: bytes, instruction: str, target_xy: Tuple[int, int],
                       image_size: Tuple[int, int], target_width_px: Optional[int],
                       target_height_px: Optional[int] = None,
                       app: str = "", ctx=None, extra_meta: Optional[dict] = None) -> Optional[dict]:
    """Grounding sample with contract-driven coordinate/output serialization."""
    max_long = compute_target_long_side(image_size, target_width_px, target_height_px)
    processed = process_image(load_image(image_bytes), max_long=max_long, lossless=True)
    from .coordinates import Action, CoordSpace
    action = Action("point", points=[target_xy], original_space=CoordSpace.PIXEL,
                    original_points=[(float(target_xy[0]), float(target_xy[1]))])
    contract = ctx.config.get("_native_interface_contract")
    layout = interface_layout(contract)
    try:
        target, canonical_target = render_action_for_contract(
            action, image_size, (processed.width, processed.height), contract)
    except Exception:
        ctx.reject(source, "native_action_render_failed")
        return None
    name = stable_image_name(trajectory_id, [step_id])
    rel = f"images/{source}/{name}"

    task = f"Point to the {instruction} on the screen. Respond with the target coordinates."
    history_mode = ((contract or {}).get("history_policy") or {}).get("mode", "text_actions")
    try:
        if contract and history_mode == "visual_recent_rounds":
            user = _format_native_user(contract, image=layout["image_placeholder"], task=task,
                                       with_task=True, visual=True)
        else:
            user = _format_native_user(contract, image=layout["image_placeholder"], task=task,
                                       history="", with_task=True, visual=False)
    except Exception:
        ctx.reject(source, "native_grounding_user_render_failed")
        return None
    messages = [
        {"role": "system", "content": layout["system_prompt"], "loss": False},
        {"role": "user", "content": user, "loss": False},
        {"role": "assistant", "content": target, "loss": True},
    ]
    est = estimate_sequence(user, target, [(processed.width, processed.height)],
                            system_text=layout["system_prompt"])
    if est > ctx.config.get("context_budget", 8192):
        ctx.state.add_rejection(source, "over_context_budget")
        return None
    meta = {
        "representation": "grounding",
        "estimated_tokens": est,
        "image_count": 1,
        "history_steps": 0,
        "window_length": 1,
        "original_image_size": list(image_size),
        "final_image_size": [processed.width, processed.height],
        "signals": ["small_target"] if min(
            target_width_px if target_width_px is not None else 999,
            target_height_px if target_height_px is not None else 999) < 24 else [],
        "app": app,
        "reasoning_category": None,
        "target_width_px": target_width_px,
        "target_height_px": target_height_px,
        "action_original": f"point({target_xy[0]}, {target_xy[1]})",
        "action_original_space": "pixel",
        "action_canonical_final_pixels": canonical_target,
        "image_encoding": "webp_lossless",
        "native_interface": bool(contract),
        "interface_coordinate_space": ((contract.get("coordinate_space") or {}).get("type")
                                       if contract else "processed_image_pixels"),
        "interface_history_mode": history_mode,
        "interface_image_placeholder": layout["image_placeholder"],
        "assistant_target_sha256": hashlib.sha256(str(messages[-1]["content"]).encode("utf-8")).hexdigest(),
    }
    if extra_meta:
        meta.update(extra_meta)
    sample = {
        "messages": messages,
        "images": [rel],
        "source": source,
        "trajectory_id": trajectory_id,
        "step_id": step_id,
        "task_type": "grounding",
        "metadata": meta,
    }
    qs = score_grounding_q(target_width_px=target_width_px or 32,
                           target_height_px=target_height_px or 16, text=instruction,
                           category=(extra_meta or {}).get("element_category"),
                           app=app, app_counter=ctx.app_counter,
                           total_samples=ctx.total_samples)
    if qs.bucket == "Reject":
        ctx.reject(source, f"quality:{qs.reject_reason}")
        return None
    attach_quality(sample, qs, est)
    save_image(processed.data, os.path.join(ctx.dataset_root, "images", source, name))
    ctx.note_app(app)
    return sample

def assemble_replay(*, messages: List[Dict[str, str]], images_pil: List[Image.Image],
                    source_name: str, sample_id: str, task_type: str,
                    metadata: Optional[dict], ctx) -> Optional[dict]:
    """Text-only or multimodal replay sample with atomic image persistence.

    Images are processed in memory first.  They are written only after the
    sample passes assistant/content/context checks, preventing orphan files
    when a replay candidate is rejected late in assembly.
    """
    # Reserve the content-derived ID before image persistence. Duplicate
    # candidates are rejected and the source keeps scanning to exact quota.
    if hasattr(ctx, "reserve_replay_id") and not ctx.reserve_replay_id(sample_id):
        return None

    md_in = dict(metadata or {})
    lossless_images = bool(md_in.get("lossless_images", False))
    processed_images = [process_image(img, max_long=DEFAULT_MAX_LONG, quality=80,
                                      lossless=lossless_images)
                        for img in images_pil]
    image_sizes = [(im.width, im.height) for im in processed_images]
    messages = [{**m, "loss": (m.get("role") == "assistant")} for m in messages]

    user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
    assistant_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    if not user_text.strip() or not assistant_text.strip():
        ctx.state.add_rejection("replay", "empty_user_or_assistant")
        return None
    est = estimate_sequence(user_text, assistant_text, image_sizes)
    if est > ctx.config.get("context_budget", 8192):
        ctx.state.add_rejection("replay", "over_context_budget")
        return None

    images: List[str] = []
    for i, processed in enumerate(processed_images):
        name = f"{sanitize_id(sample_id)}__{i}.webp"
        save_image(processed.data, os.path.join(ctx.dataset_root, "images", "replay", name))
        images.append(f"images/replay/{name}")

    md = md_in
    md.pop("lossless_images", None)
    md.update({"estimated_tokens": est, "image_count": len(images),
               "representation": "replay",
               "image_encoding": ("webp_lossless" if lossless_images else "webp_q80")})
    return {
        "messages": messages,
        "images": images,
        "source": "replay",
        "trajectory_id": sample_id,
        "step_id": sample_id,
        "task_type": task_type,
        "metadata": md,
    }

