"""GUI-360 Lite adapter (cua-lite/GUI-360; origin vyokky/GUI-360, MIT).

Verified schema (2026-08, dataset-server rows):
    columns: images (list[Image] embedded JPEG), messages (JSON string),
             metadata (JSON string), _folded (grounding/understanding only)
    messages: OpenAI-style; user turns carry {"type":"image","index":k} +
              {"type":"text"}; assistant turns carry content parts AND/OR
              tool_calls [{function:{name, arguments}}].
    COORDINATES ARE NORMALIZED INTEGERS IN [0, 1000] (both `point`/
    `click(coordinate=[x,y])` tool calls and understanding control_rects).
    reference resolution: metadata.others.resolution = [w, h] (e.g. 1040x736)

Cohorts (stats.json):
    desktop.use             train 10,641  + validation 221
    desktop.grounding.point train 77,843  + validation 1,583
    desktop.understanding   train 95,367  + validation 1,968

Documented composition adjustment (spec section 8): the public lite release
exposes NO failure/recovery category, so the requested 3,000 failure/recovery
slots are reallocated to difficult grounding and screen understanding:
    use (action prediction)   ~10,862  (all rows)
    difficult grounding       ~5,000
    screen understanding      ~4,138
No categories were fabricated.

Consumed via HF `datasets` STREAMING (no local mirror; row groups fetched on
demand). Folded rows are unfolded to one sample per member.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Iterator, List, Optional, Tuple

from processing.assemble import SYSTEM_PROMPT, assemble_grounding, assemble_replay
from processing.coordinates import (CoordSpace, Point,
                                      parse_gui360_tool_call)
from processing.dedup import phash
from processing.images import load_image
from processing.sampling import detect_task_signals
from processing.token_budget import estimate_sequence
from processing.windows import Step, Trajectory, build_single
from processing.quality import UNDERSTANDING_MAX_CONTROLS
from .common import BuildContext
from .revisions import source_revision

REPO = "cua-lite/GUI-360"
REVISION = source_revision("gui360")  # immutable Run 1 snapshot
NORM_MAX = 1000

APP_RE = re.compile(r"^[A-Za-z]+")
_SAVE_EXPORT_RE = re.compile(r"\bsave\b|\bexport\b|\bsave as\b|\boverwrite\b|\brename\b|\bformat\b|\bsort\b|\border\b|\brank\b", re.I)
_DIALOG_RE = re.compile(r"dialog|modal|window|popup|confirm", re.I)


def app_from_id(row_id: str) -> str:
    m = APP_RE.match(row_id or "")
    return (m.group(0) if m else "office").lower()


def decode_image(img_feature) -> Optional[bytes]:
    """datasets Image feature -> bytes (streaming gives dict{bytes,path})."""
    if img_feature is None:
        return None
    if isinstance(img_feature, (bytes, bytearray)):
        return bytes(img_feature)
    if isinstance(img_feature, dict):
        b = img_feature.get("bytes")
        if b:
            return b
        path = img_feature.get("path")
        if path and isinstance(path, str):
            import os
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()
    return None


def iter_rows(ctx: BuildContext, config_name: str, split: str = "train") -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset(REPO, config_name, split=split, streaming=True, revision=REVISION)
    for row in ds:
        yield row


def unfold(row: dict) -> List[Tuple[dict, Optional[bytes]]]:
    """Unfold folded rows into (member, image_bytes) pairs. `use` rows pass
    through unchanged (image list shared across turns)."""
    folded = row.get("_folded")
    if folded:
        try:
            members = json.loads(folded) if isinstance(folded, str) else folded
        except json.JSONDecodeError:
            return []
        img = decode_image(row["images"][0]) if row.get("images") else None
        out = []
        for m in members:
            out.append((m, img))
        return out
    return [(row, None)]


def parse_meta(row_or_member) -> dict:
    meta = row_or_member.get("metadata")
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return meta or {}


def parse_messages(row_or_member) -> List[dict]:
    msgs = row_or_member.get("messages")
    if isinstance(msgs, str):
        try:
            return json.loads(msgs)
        except json.JSONDecodeError:
            return []
    return msgs or []


# ---------------------------------------------------------------- quality helpers

_GROUND_QUOTED_CLICK_RE = re.compile(
    r"\b(?:click(?:ing)?|select(?:ing)?|choose|choosing|open(?:ing)?|activate|activating|pick(?:ing)?)\s+(?:on\s+)?(?:the\s+)?['\"]([^'\"]{1,80})['\"](?:\s+(button|tab|item|option|listitem|menu|link|cell))?",
    re.I,
)
_GROUND_CELL_ACTION_RE = re.compile(
    r"\b(?:click|select|choose)\s+(?:on\s+)?(?:the\s+)?(?:cell\s+)?([A-Z]{1,3}\d{1,7}(?::[A-Z]{1,3}\d{1,7})?)\b",
    re.I,
)
_KEYBOARD_ONLY_RE = re.compile(r"\b(?:keyboard shortcut|hotkey|press\s+(?:enter|return|tab|escape|esc|ctrl|alt|shift)|type\b|input\b)", re.I)


def grounding_referent(text: str) -> Optional[str]:
    """Extract a high-precision *visible* referent from verbose GUI-360 text.

    The grounding source contains next-action reasoning, including keyboard
    operations for which the annotated point is only incidental.  Training
    those as ``Point to the <whole paragraph>`` is ambiguous.  Run 1 keeps
    only cases where the text explicitly names a visible click/select target.
    """
    text = " ".join((text or "").split())
    if not text:
        return None
    m = _GROUND_QUOTED_CLICK_RE.search(text)
    if m:
        label = m.group(1).strip()
        kind = (m.group(2) or "").strip().lower()
        # strip source-only label ids: "File Tab' (label 17)" is not a visual
        # instruction the model needs.
        if not label or len(label) > 80:
            return None
        return f"{label} {kind}".strip()
    m = _GROUND_CELL_ACTION_RE.search(text)
    if m:
        return f"cell {m.group(1).upper()}"
    # No explicit visible click/select referent.  In particular, do not turn
    # keyboard-only actions into a guessed point target.
    if _KEYBOARD_ONLY_RE.search(text):
        return None
    return None


UNDERSTANDING_RECT_EPS = 25.0  # audit max edge spill ~21/1000


def _control_kind(c: dict, text: str) -> str:
    raw = str(c.get("control_type") or c.get("category") or c.get("type") or "").strip().lower()
    if raw:
        return raw[:48]
    low = text.lower()
    for word in ("button", "tab", "menu", "checkbox", "radio", "cell", "input", "textbox",
                 "link", "dropdown", "combobox", "toolbar", "icon", "slider"):
        if word in low:
            return word
    return "unknown"


def sanitize_understanding_controls(raw_controls, max_controls: int = UNDERSTANDING_MAX_CONTROLS):
    """Select a compact but information-dense set of visible controls.

    The selector is deterministic and explicitly rewards spatial coverage,
    label diversity, control-type diversity, small targets and rare controls.
    Only after selection are controls sorted top-to-bottom, left-to-right for
    a stable target serialization.
    """
    if not isinstance(raw_controls, list):
        return []
    import math
    from collections import Counter
    cleaned = []
    seen = set()
    for c in raw_controls:
        if not isinstance(c, dict):
            continue
        text = " ".join(str(c.get("control_text") or "").split())
        rect = c.get("control_rect")
        if not text or len(text) > 180 or not isinstance(rect, (list, tuple)) or len(rect) < 4:
            continue
        try:
            vals = [float(v) for v in rect[:4]]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in vals):
            continue
        x1, y1, x2, y2 = vals
        if min(vals) < -UNDERSTANDING_RECT_EPS or max(vals) > 1000.0 + UNDERSTANDING_RECT_EPS:
            continue
        x1, y1, x2, y2 = [max(0.0, min(1000.0, v)) for v in (x1, y1, x2, y2)]
        if x2 <= x1 or y2 <= y1:
            continue
        rect2 = [int(round(v)) for v in (x1, y1, x2, y2)]
        key = (text.casefold(), tuple(rect2))
        if key in seen:
            continue
        seen.add(key)
        kind = _control_kind(c, text)
        cleaned.append({"control_text": text, "control_rect": rect2, "_kind": kind})
    if len(cleaned) <= max_controls:
        picked = cleaned
    else:
        label_freq = Counter(c["control_text"].casefold() for c in cleaned)
        kind_freq = Counter(c["_kind"] for c in cleaned)
        picked = []
        remaining = list(cleaned)
        seen_kinds = set()
        seen_tokens = set()

        def center(c):
            x1, y1, x2, y2 = c["control_rect"]
            return ((x1 + x2) / 2000.0, (y1 + y2) / 2000.0)

        def score(c):
            x1, y1, x2, y2 = c["control_rect"]
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            smaller = min(w, h)
            size_bonus = min(1.6, 70.0 / max(8.0, smaller))
            rarity = 0.65 / math.sqrt(label_freq[c["control_text"].casefold()])
            rarity += 0.55 / math.sqrt(kind_freq[c["_kind"]])
            novelty = 0.75 if c["_kind"] not in seen_kinds else 0.0
            toks = set(re.findall(r"[A-Za-z0-9]+", c["control_text"].casefold()))
            novelty += 0.35 * min(2.0, len(toks - seen_tokens))
            if not picked:
                spatial = 0.8
            else:
                cx, cy = center(c)
                spatial = 1.8 * min(math.hypot(cx - px, cy - py) for px, py in map(center, picked))
            return size_bonus + rarity + novelty + spatial

        while remaining and len(picked) < max_controls:
            best = max(remaining, key=lambda c: (score(c),
                                                  -c["control_rect"][1],
                                                  -c["control_rect"][0],
                                                  c["control_text"].casefold()))
            picked.append(best)
            seen_kinds.add(best["_kind"])
            seen_tokens.update(re.findall(r"[A-Za-z0-9]+", best["control_text"].casefold()))
            remaining.remove(best)

    result = [{"control_text": c["control_text"], "control_rect": c["control_rect"]}
              for c in picked]
    result.sort(key=lambda c: (c["control_rect"][1], c["control_rect"][0],
                               c["control_rect"][3], c["control_rect"][2],
                               c["control_text"].casefold()))
    return result[:max_controls]


# ---------------------------------------------------------------- use cohort

RESOLUTION_ASPECT_TOLERANCE = 0.02  # >2% aspect delta => ambiguous convention


def resolve_conversion_space(metadata_wh, actual_wh):
    """Dimension guard for GUI-360's [0,1000] normalized coordinates.

    The reference space is metadata.others.resolution; the real image may
    disagree. Returns (conv_w, conv_h, status) where status is:
      "ok"                       metadata == actual (normal path, unchanged)
      "mismatch_used_actual"     same aspect within tolerance -> convert in
                                 the ACTUAL image space (uniform rescale of
                                 the reference, coordinates stay proportional)
      "ambiguous"                aspect differs -> convention cannot be
                                 unambiguously converted; caller must reject
    """
    mw, mh = int(metadata_wh[0]), int(metadata_wh[1])
    aw, ah = int(actual_wh[0]), int(actual_wh[1])
    if (mw, mh) == (aw, ah):
        return mw, mh, "ok"
    if mh and ah and abs((aw / ah) - (mw / mh)) / (mw / mh) <= RESOLUTION_ASPECT_TOLERANCE:
        return aw, ah, "mismatch_used_actual"
    return aw, ah, "ambiguous"


def use_row_to_trajectory(row: dict, stats: Optional[dict] = None) -> Optional[Trajectory]:
    meta = parse_meta(row)
    others = meta.get("others", {}) or {}
    row_id = others.get("id", "gui360_unknown")
    resolution = others.get("resolution") or [1040, 736]
    w, h = int(resolution[0]), int(resolution[1])
    msgs = parse_messages(row)
    images: List[Optional[bytes]] = [decode_image(im) for im in (row.get("images") or [])]

    task = ""
    steps: List[Step] = []
    img_idx = 0
    prev_hash = None
    prev_action = None
    continuity_id = 0
    i = 0
    while i < len(msgs):
        m = msgs[i]
        role = m.get("role")
        content = m.get("content") or []
        if role == "user":
            text_parts = [c.get("text", "") for c in content
                          if isinstance(c, dict) and c.get("type") == "text"]
            n_images = sum(1 for c in content
                           if isinstance(c, dict) and c.get("type") == "image")
            if text_parts and any(t.strip() for t in text_parts):
                task = " ".join(t for t in text_parts if t.strip())
            step_img = images[img_idx] if img_idx < len(images) else None
            img_idx += n_images or (1 if step_img else 0)
            assistant = (msgs[i + 1] if i + 1 < len(msgs)
                         and msgs[i + 1].get("role") == "assistant" else None)
            if assistant is None:
                continuity_id += 1
                prev_hash = prev_action = None
                i += 1
                continue
            tool_calls = assistant.get("tool_calls") or []
            pil = load_image(step_img) if step_img else None
            conv_w = conv_h = None
            dim_status = "ok"
            if pil is not None:
                conv_w, conv_h, dim_status = resolve_conversion_space((w, h), pil.size)
                if dim_status == "ambiguous":
                    if stats is not None:
                        stats["ambiguous"] = stats.get("ambiguous", 0) + 1
                    continuity_id += 1
                    prev_hash = prev_action = None
                    i += 2
                    continue

            action = None
            if tool_calls and pil is not None:
                # GUI-360 sometimes batches click+type/key/terminate in one
                # assistant turn.  For a one-action policy the *first* call is
                # a truthful next action; later calls have no intervening
                # screenshot and therefore must never become separate targets.
                first = tool_calls[0].get("function") or {}
                args = first.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = None
                if isinstance(args, dict):
                    action = parse_gui360_tool_call(first.get("name", ""), args,
                                                    conv_w, conv_h)
                if len(tool_calls) > 1 and stats is not None:
                    stats["multi_tool_turns"] = stats.get("multi_tool_turns", 0) + 1
                    stats["batched_tool_calls_ignored"] = stats.get("batched_tool_calls_ignored", 0) + len(tool_calls) - 1
                if action is None and stats is not None:
                    stats["unparseable_first_tool"] = stats.get("unparseable_first_tool", 0) + 1

            if action is not None and pil is not None:
                if dim_status != "ok" and stats is not None:
                    stats["mismatch_used_actual"] = stats.get("mismatch_used_actual", 0) + 1
                hsh = phash(pil)
                from processing.quality import visual_effect_expected
                no_change = (prev_hash is not None
                             and bin(hsh ^ prev_hash).count("1") <= 6
                             and visual_effect_expected(prev_action))
                signals = detect_task_signals(task)
                if no_change:
                    signals.add("no_state_change")
                if _SAVE_EXPORT_RE.search(task):
                    signals.add("save")
                if _DIALOG_RE.search(task):
                    signals.add("modal_dialog")
                steps.append(Step(
                    step_id=f"{row_id}_s{len(steps)}", image_bytes=step_img,
                    image_size=(conv_w, conv_h), action=action, phash=hsh,
                    prev_phash=prev_hash, subgoal="", signals=signals,
                    metadata={"row_id": row_id,
                              "tool_call_count": len(tool_calls),
                              "coordinate_dimension_mismatch": dim_status == "mismatch_used_actual",
                              "continuity_id": continuity_id},
                ))
                prev_hash = hsh
                prev_action = action
            else:
                # Missing image or unparseable first call creates a real
                # causal gap. Never carry action history/recovery evidence
                # across a state transition we did not supervise.
                continuity_id += 1
                prev_hash = prev_action = None
            i += 2
            continue
        i += 1
    if not task or not steps:
        return None
    return Trajectory(trajectory_id=f"gui360_use_{row_id}", task=task, steps=steps,
                      app=app_from_id(row_id), source="gui360",
                      metadata={"cohort": "use", "row_id": row_id,
                                "os": others.get("os", "windows")})


def build_use_samples(ctx: BuildContext) -> List[dict]:
    """The use cohort streams app-contiguous blocks (audit 2026-08: offsets
    0-3500 excel, then word, then ppt). A per-app cap keeps any one Office
    app below ~55% of the use quota."""
    from processing.sampling import AppCap
    out: List[dict] = []
    quota = ctx.quota.get("gui360", 0)
    prior_apps = dict((ctx.config.get("_resume_gui360_app_counts", {}) or {}).get("use", {}))
    # quota here is the remaining use-cohort quota; cap is based on the full
    # cohort size so resume never shrinks below already-selected app counts.
    prior_total = sum(prior_apps.values())
    app_cap = AppCap(cap=max(30, (quota + prior_total) * 3 // 5), counts=prior_apps)
    for split in ("train", "validation"):
        for row in iter_rows(ctx, "desktop.use", split):
            if ctx.remaining("gui360") <= 0:
                return out
            row_id = parse_meta(row).get("others", {}).get("id", "?")
            app = app_from_id(row_id)
            if not app_cap.allow(app):
                continue
            key = f"{REPO}:desktop.use:{split}:{parse_meta(row).get('others', {}).get('id', '')}"
            if ctx.state.is_shard_done("gui360", key):
                continue
            dim_stats: Dict[str, int] = {}
            traj = use_row_to_trajectory(row, stats=dim_stats)
            if traj is None:
                ctx.reject("gui360", "unparseable_use_row")
                ctx.state.mark_shard_done("gui360", key)
                continue
            if dim_stats.get("ambiguous"):
                ctx.reject("gui360", "coordinate_space_ambiguous", dim_stats["ambiguous"])
            if dim_stats.get("mismatch_used_actual"):
                ctx.note_stat("gui360", "resolution_mismatch_used_actual",
                              dim_stats["mismatch_used_actual"])
            if ctx.decontaminate(traj.task, "gui360"):
                ctx.state.mark_shard_done("gui360", key)
                continue
            got = 0
            for idx, step in enumerate(traj.steps):
                if ctx.remaining("gui360") <= 0:
                    break
                dup, reason = ctx.dedup.consider(image_phash=step.phash,
                                                 signals=step.signals,
                                                 task_text=traj.task,
                                                 action_text=step.action_text)
                if dup:
                    ctx.reject("gui360", reason)
                    continue
                spec = build_single(traj, idx)
                spec.task_type = "action"
                # GUI360 bypasses action_representation_specs, so attach the
                # same anchor-local quality context explicitly. Never infer a
                # state transition at the start of a continuity segment.
                seg = (step.metadata or {}).get("continuity_id", 0)
                prev_steps = []
                for prior_step in reversed(traj.steps[:idx]):
                    if (prior_step.metadata or {}).get("continuity_id", 0) != seg:
                        break
                    prev_steps.append(prior_step)
                    if len(prev_steps) >= 4:
                        break
                spec.metadata["_prev_steps"] = list(reversed(prev_steps))
                spec.metadata["_is_first_step"] = not prev_steps
                spec.metadata["_state_changed"] = (None if step.prev_phash is None else
                    bin(step.phash ^ step.prev_phash).count("1") > 6)
                sample = assemble_sample(spec, ctx, trajectory=traj)
                if sample is None:
                    continue
                out.append(sample)
                ctx.consume("gui360")
                got += 1
            ctx.state.add_selected("gui360", got, trajectories=1 if got else 0)
            if got:
                app_cap.record(app, got)
                ctx.persist_samples(out[-got:])
            ctx.state.mark_shard_done("gui360", key)
            ctx.state.save()
    return out


# -------------------------------------------------------- grounding/underst.

def target_rect_pixels(rect, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if not rect or len(rect) < 4:
        return None
    try:
        vals = [float(v) for v in rect[:4]]
    except (TypeError, ValueError):
        return None
    if min(vals) < -UNDERSTANDING_RECT_EPS or max(vals) > 1000.0 + UNDERSTANDING_RECT_EPS:
        return None
    vals = [max(0.0, min(1000.0, v)) for v in vals]
    x1 = Point(vals[0], 0, CoordSpace.NORM_0_1000).to_pixels(w, 1)[0]
    y1 = Point(0, vals[1], CoordSpace.NORM_0_1000).to_pixels(1, h)[1]
    x2 = Point(vals[2], 0, CoordSpace.NORM_0_1000).to_pixels(w, 1)[0]
    y2 = Point(0, vals[3], CoordSpace.NORM_0_1000).to_pixels(1, h)[1]
    if not (0 <= x1 < x2 < w and 0 <= y1 < y2 < h):
        return None
    return x1, y1, x2, y2


def build_grounding_samples(ctx: BuildContext, want: int) -> List[dict]:
    from processing.sampling import AppCap
    out: List[dict] = []
    prior_apps = dict((ctx.config.get("_resume_gui360_app_counts", {}) or {}).get("grounding", {}))
    grd_app_cap = AppCap(cap=max(20, (want + sum(prior_apps.values())) * 3 // 5), counts=prior_apps)
    for split in ("train", "validation"):
        for row in iter_rows(ctx, "desktop.grounding.point", split):
            if len(out) >= want or ctx.remaining("gui360") <= 0:
                return out
            meta = parse_meta(row)
            others = meta.get("others", {}) or {}
            resolution = others.get("resolution") or [1040, 736]
            w, h = int(resolution[0]), int(resolution[1])
            key = f"{REPO}:grounding:{split}:{others.get('id', '')}"
            if ctx.state.is_shard_done("gui360", key):
                continue
            if not grd_app_cap.allow(app_from_id(others.get("id", "?"))):
                continue
            for member, image_bytes in unfold(row):
                if len(out) >= want or ctx.remaining("gui360") <= 0:
                    break
                if image_bytes is None:
                    image_bytes = decode_image((row.get("images") or [None])[0])
                if not image_bytes:
                    ctx.reject("gui360", "missing_image_bytes")
                    continue
                msgs = parse_messages(member)
                user_text = ""
                point_norm = None
                for m in msgs:
                    if m.get("role") == "user":
                        for c in (m.get("content") or []):
                            if isinstance(c, dict) and c.get("type") == "text":
                                user_text = c.get("text", "")
                    tc = (m.get("tool_calls") or [{}])[0]
                    fn = tc.get("function") or {}
                    if fn.get("name") == "point":
                        args = fn.get("arguments")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        coord = (args or {}).get("coordinate")
                        if coord and len(coord) >= 2:
                            point_norm = (float(coord[0]), float(coord[1]))
                if not user_text or point_norm is None:
                    ctx.reject("gui360", "unparseable_grounding_row")
                    continue
                pil = load_image(image_bytes)
                conv_w, conv_h, dim_status = resolve_conversion_space((w, h), pil.size)
                if dim_status == "ambiguous":
                    ctx.reject("gui360", "coordinate_space_ambiguous")
                    continue
                if dim_status == "mismatch_used_actual":
                    ctx.note_stat("gui360", "resolution_mismatch_used_actual")
                point = Point(point_norm[0], point_norm[1],
                              CoordSpace.NORM_0_1000).to_pixels(conv_w, conv_h)
                instruction = grounding_referent(user_text)
                if not instruction:
                    ctx.reject("gui360", "ambiguous_grounding_referent")
                    continue
                if not (0 <= point[0] < conv_w and 0 <= point[1] < conv_h):
                    ctx.reject("gui360", "coordinate_out_of_bounds")
                    continue
                hsh = phash(pil)
                dup, reason = ctx.dedup.consider(image_phash=hsh, signals=["grounding"],
                                                 task_text=instruction,
                                                 action_text=f"point({point[0]},{point[1]})")
                if dup:
                    ctx.reject("gui360", reason)
                    continue
                row_id = others.get("id", "g")
                sample = assemble_grounding(
                    source="gui360", trajectory_id=f"gui360_grd_{row_id}",
                    step_id=f"{row_id}_p", image_bytes=image_bytes,
                    instruction=instruction[:200], target_xy=point,
                    image_size=(conv_w, conv_h), target_width_px=None,
                    app=app_from_id(row_id), ctx=ctx,
                    extra_meta={"cohort": "grounding.point",
                                "coordinate_space_source": "norm_0_1000",
                                "coordinate_dimension_mismatch":
                                    dim_status == "mismatch_used_actual"})
                if sample is not None:
                    out.append(sample)
                    grd_app_cap.record(app_from_id(row_id), 1)
                    ctx.persist_samples([sample])
                    ctx.consume("gui360")
            ctx.state.mark_shard_done("gui360", key)
    return out


def build_understanding_samples(ctx: BuildContext, want: int) -> List[dict]:
    """Canonical screen-understanding samples with observable ordering.

    The source can return hundreds of controls in an internal order.  Asking
    for "the first 60" is unlearnable from pixels alone, so we sanitize and
    spatially order controls before capping to the first 60.
    """
    from processing.sampling import AppCap
    out: List[dict] = []
    prior_apps = dict((ctx.config.get("_resume_gui360_app_counts", {}) or {}).get("understanding", {}))
    app_cap = AppCap(cap=max(20, (want + sum(prior_apps.values())) * 3 // 5), counts=prior_apps)
    canonical_prompt = (
        f"List up to {UNDERSTANDING_MAX_CONTROLS} interactive controls visible on the screen, "
        "ordered from top to bottom and then left to right. Return a JSON list; each item must "
        "contain control_text and control_rect, where control_rect is [x1,y1,x2,y2] in normalized "
        "0-to-1000 screen coordinates."
    )
    for split in ("train", "validation"):
        for row in iter_rows(ctx, "desktop.understanding", split):
            if len(out) >= want or ctx.remaining("gui360") <= 0:
                return out
            meta = parse_meta(row)
            others = meta.get("others", {}) or {}
            row_id = others.get("id", "u")
            app = app_from_id(row_id)
            if not app_cap.allow(app):
                continue
            key = f"{REPO}:understanding:{split}:{row_id}"
            if ctx.state.is_shard_done("gui360", key):
                continue
            image_bytes = decode_image((row.get("images") or [None])[0]) if row.get("images") else None
            for member, img_bytes in unfold(row):
                if len(out) >= want or ctx.remaining("gui360") <= 0:
                    break
                if img_bytes is not None:
                    image_bytes = img_bytes
                if not image_bytes:
                    ctx.reject("gui360", "missing_image_bytes")
                    continue
                msgs = parse_messages(member)
                answer = ""
                for m in msgs:
                    if m.get("role") == "assistant":
                        for c in (m.get("content") or []):
                            if isinstance(c, dict) and c.get("type") == "text":
                                answer = c.get("text", "")
                if not answer:
                    ctx.reject("gui360", "unparseable_understanding_row")
                    continue
                try:
                    controls = json.loads(answer)
                except (json.JSONDecodeError, TypeError):
                    ctx.reject("gui360", "malformed_understanding_json")
                    continue
                cleaned = sanitize_understanding_controls(controls)
                if not cleaned:
                    ctx.reject("gui360", "no_valid_understanding_controls")
                    continue
                answer = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
                pil = load_image(image_bytes)
                hsh = phash(pil)
                dup, reason = ctx.dedup.consider(image_phash=hsh, signals=["understanding"],
                                                 task_text=canonical_prompt,
                                                 action_text=answer)
                if dup:
                    ctx.reject("gui360", reason)
                    continue
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT, "loss": False},
                    {"role": "user", "content": f"<image>\n{canonical_prompt}", "loss": False},
                    {"role": "assistant", "content": answer, "loss": True},
                ]
                est = estimate_sequence(canonical_prompt, answer, [pil.size])
                if est > ctx.config.get("context_budget", 8192):
                    ctx.reject("gui360", "over_context_budget")
                    continue
                from processing.images import process_image, save_image
                from processing.assemble import stable_image_name
                processed = process_image(pil, lossless=True)
                sid = f"gui360_und_{row_id}"
                name = stable_image_name(sid, [sid])
                save_image(processed.data, str(ctx.dataset_root) + "/images/gui360/" + name)
                sample = {
                    "messages": messages,
                    "images": [f"images/gui360/{name}"],
                    "source": "gui360",
                    "trajectory_id": sid,
                    "step_id": sid,
                    "task_type": "screen_understanding",
                    "metadata": {"representation": "understanding",
                                 "estimated_tokens": est, "image_count": 1,
                                 "history_steps": 0, "window_length": 1,
                                 "original_image_size": list(pil.size),
                                 "final_image_size": [processed.width, processed.height],
                                 "signals": ["understanding"], "app": app,
                                 "reasoning_category": None,
                                 "control_count": len(cleaned),
                                 "selection": "diverse_24",
                                 "image_encoding": "webp_lossless",
                                 "group_id": f"gui360_understanding::{app}",
                                 "ordering": "top_to_bottom_then_left_to_right",
                                 "action_original": None,
                                 "action_original_space": "norm_0_1000"},
                }
                out.append(sample)
                app_cap.record(app, 1)
                ctx.persist_samples([sample])
                ctx.consume("gui360")
            ctx.state.mark_shard_done("gui360", key)
    return out


def run(ctx: BuildContext, use_all: bool = True,
        grounding_want: int = 5000, understanding_want: int = 4138) -> List[dict]:
    """Build the three approved GUI-360 cohorts with resume-safe exact quotas.

    A resumed run must continue the original use/grounding/understanding mix,
    not reinterpret the remaining global count as a new smaller dataset.
    """
    remaining_global = ctx.quota.get("gui360", 0)
    already = int(ctx.state.selected_total("gui360"))
    full_target = already + remaining_global
    ctx.state.set_target("gui360", full_target)

    full_targets = {
        "grounding": min(grounding_want, full_target),
        "understanding": min(understanding_want, max(0, full_target - grounding_want)),
    }
    full_targets["use"] = max(0, full_target - full_targets["grounding"] - full_targets["understanding"]) if use_all else 0
    prior = ctx.config.get("_resume_gui360_cohorts", {}) or {}
    todo = {k: max(0, full_targets[k] - int(prior.get(k, 0))) for k in full_targets}

    out: List[dict] = []
    builders = (
        ("use", build_use_samples),
        ("grounding", build_grounding_samples),
        ("understanding", build_understanding_samples),
    )
    for cohort, fn in builders:
        want = todo[cohort]
        if want <= 0:
            continue
        ctx.quota["gui360"] = want
        got = fn(ctx) if cohort == "use" else fn(ctx, want)
        if len(got) != want:
            raise RuntimeError(
                f"mandatory GUI360 cohort '{cohort}' emitted {len(got)} samples "
                f"(wanted exactly {want}); failing closed")
        out.extend(got)

    ctx.quota["gui360"] = max(0, sum(todo.values()) - len(out))
    return out

