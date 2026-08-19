"""PC-Agent-E adapter (henryhe0123/PC-Agent-E, MIT).

Verified layout (2026-08, via HTTP range inspection of data.zip, 1.57 GB):
    data/events/taskNNN.md       - task instruction + verbose human walkthrough
    data/events/taskNNN.jsonl    - one event per step:
        {"action": "click (654, 191)", "screenshot": "screenshot/xxx.png",
         "element": "...", "rect": {left, top, right, bottom},
         "marked_screenshot": "...", "thought": "<verbose, NOT trained>"}
    data/events/screenshot/*.png - 4,503 screenshots (1920x1080)

Policy (spec section 11): use ALL 4,503 public samples; keep the actual
action; never train the verbose source thoughts (they only feed signal
detection); one sample per event (no synthetic multiplication); boost
responses remain metadata only.

Storage: zero local mirroring. Task metadata (jsonl+md) is fetched with
range requests; screenshots are fetched per-task and deleted after
processing. The full zip download happens only when explicitly allowed on a
storage-capable machine.
"""
from __future__ import annotations

import io
import json
import re
from typing import Dict, Iterator, List, Optional, Tuple

from processing.assemble import assemble_sample
from processing.coordinates import (Action, CoordSpace, Point,
                                      parse_pc_agent_e)
from processing.dedup import phash
from processing.images import load_image, png_dimensions
from processing.remote_access import download_file, hf_url, open_remote_zip
from processing.sampling import (deterministic_keep, score_steps,
                                   trajectory_priority)
from processing.windows import (Step, Trajectory, build_chunk, build_single,
                                  build_window, choose_representation,
                                  suggest_window_starts, CHUNK, WINDOW)
from .common import BuildContext
from .revisions import source_revision

REPO = "henryhe0123/PC-Agent-E"
REVISION = source_revision("pcagente")  # immutable Run 1 snapshot
ZIP_PATH = "data.zip"

_MD_DESC_RE = re.compile(r"\*\*Description:\*\*\s*(.+)")
_VERIFY_RE = re.compile(r"verif|check|confirm|make sure|ensure", re.I)
_RECOVERY_RE = re.compile(r"did not work|does not work|failed|failure|unsuccessful|no effect|not responding|retry|try again", re.I)
_SUCCESS_RE = re.compile(r"\b(?:task|operation|save|export)?\s*(?:is|was|has been)?\s*(?:successfully\s+)?(?:complete|completed|finished|done|saved|exported)\b|\bverified\b[^.]{0,60}\b(?:complete|done|saved|exported)\b", re.I)
_CLICK_RECT_VERBS = {"click", "double_click", "right_click", "middle_click"}


def parse_md_description(md_text: str) -> str:
    m = _MD_DESC_RE.search(md_text or "")
    return m.group(1).strip() if m else ""


def extract_signals(task: str, thought: str, action: Action,
                    no_state_change: bool) -> set:
    signals = set()
    if no_state_change:
        signals.add("no_state_change")
    if _VERIFY_RE.search(task or "") or _VERIFY_RE.search(thought or ""):
        signals.add("verification")
    if _RECOVERY_RE.search(thought or ""):
        signals.update({"recovery_hint", "recovery_evidenced"})
    if action is not None:
        if action.verb in ("hotkey", "press"):
            signals.add("keyboard_shortcut")
        if action.verb == "drag":
            signals.add("drag")
        if action.verb == "wait":
            signals.add("wait")
        if action.verb == "finish":
            signals.update({"finish_verification", "verification"})
    from processing.sampling import detect_task_signals
    signals |= detect_task_signals(task)
    return signals


def task_files_from_zip(zf) -> List[Tuple[str, str]]:
    """[(task_id, jsonl_path, md_path)] sorted numerically."""
    import zipfile
    tasks: Dict[str, Dict[str, str]] = {}
    for name in zf.namelist():
        m = re.match(r"data/events/(task\d+)\.(jsonl|md)$", name)
        if m:
            tasks.setdefault(m.group(1), {})[m.group(2)] = name
    out = []
    for tid in sorted(tasks, key=lambda t: int(t[4:])):
        d = tasks[tid]
        if "jsonl" in d:
            out.append((tid, d["jsonl"], d.get("md")))
    return out


def _validated_rect_metadata(rect: dict, action: Action, dims: Tuple[int, int]) -> Optional[dict]:
    """Validate PC-Agent-E element rectangles against the supervised click.

    Missing rectangles are allowed. Malformed or out-of-bounds rectangles are
    ignored for non-click actions, but click-like actions with a present trusted
    rectangle are rejected when the click falls outside it.
    """
    if not isinstance(rect, dict) or not rect:
        return {}
    try:
        x1, y1, x2, y2 = (float(rect.get("left")), float(rect.get("top")),
                          float(rect.get("right")), float(rect.get("bottom")))
    except (TypeError, ValueError):
        return None if action.verb in _CLICK_RECT_VERBS else {}
    w, h = dims
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        return None if action.verb in _CLICK_RECT_VERBS else {}
    md = {"target_bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
          "target_width_px": max(1, int(round(x2 - x1))),
          "target_height_px": max(1, int(round(y2 - y1)))}
    if action.verb in _CLICK_RECT_VERBS and action.points:
        x, y = action.points[0]
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return None
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        diag = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        md["bbox_click_validated"] = True
        md["bbox_center_offset_norm"] = round((((x - cx) ** 2 + (y - cy) ** 2) ** 0.5) / diag, 5)
    return md


def build_trajectory(tid: str, jsonl_text: str, md_text: str,
                     images_by_name: Dict[str, bytes]) -> Optional[Trajectory]:
    task = parse_md_description(md_text)
    if not task:
        task = f"Complete the desktop task shown (task {tid})."
    steps: List[Step] = []
    prev_hash = None
    prev_action = None
    continuity_id = 0
    for i, line in enumerate(jsonl_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continuity_id += 1
            prev_hash = prev_action = None
            continue
        shot = ev.get("screenshot") or ""
        # run() keys fetched screenshots by the full jsonl value ("screenshot/x.png");
        # accept the basename form as well so the two sides can never diverge again.
        shot_key = shot if shot in images_by_name else shot.split("/")[-1]
        if not shot_key or shot_key not in images_by_name:
            continuity_id += 1
            prev_hash = prev_action = None
            continue
        data = images_by_name[shot_key]
        dims = png_dimensions(data)
        if not dims:
            continuity_id += 1
            prev_hash = prev_action = None
            continue
        action = parse_pc_agent_e(ev.get("action", ""), *dims)
        if action is None:
            continuity_id += 1
            prev_hash = prev_action = None
            continue
        h = phash(load_image(data))
        from processing.quality import visual_effect_expected
        no_state_change = (prev_hash is not None
                           and bin(h ^ prev_hash).count("1") <= 6
                           and visual_effect_expected(prev_action))
        thought = ev.get("thought", "") or ""
        signals = extract_signals(task, thought, action, no_state_change)
        rect = ev.get("rect") or {}
        rect_meta = _validated_rect_metadata(rect, action, dims)
        if rect_meta is None:
            # The source claims an element rectangle but the supervised click
            # does not land inside it. This is objective grounding corruption.
            continuity_id += 1
            prev_hash = prev_action = None
            continue
        finish_meta = {}
        if action.verb == "finish" and _SUCCESS_RE.search(thought):
            finish_meta["verifier_evidence"] = True
            finish_meta["explicit_success"] = True
        step_meta = {"element": ev.get("element", ""),
                     "rect": [rect.get("left"), rect.get("top"),
                              rect.get("right"), rect.get("bottom")],
                     "task_family": f"pcagente::{tid}",
                     "continuity_id": continuity_id}
        step_meta.update(rect_meta)
        step_meta.update(finish_meta)
        steps.append(Step(
            step_id=f"{tid}_s{i}", image_bytes=data, image_size=dims,
            action=action, phash=h, prev_phash=prev_hash,
            subgoal=ev.get("element", "") or "", signals=signals,
            metadata=step_meta,
        ))
        prev_hash = h
        prev_action = action
    if not steps:
        return None
    return Trajectory(trajectory_id=f"pcagente_{tid}", task=task, steps=steps,
                      app="windows_desktop", source="pcagente",
                      metadata={"task_id": tid})


def samples_for_trajectory(traj: Trajectory, ctx: BuildContext) -> List[dict]:
    """PC-Agent-E: one sample per event (all 4,503), single-step with text
    history; representation follows the global ratio only as windowing of
    existing steps (never duplicating steps into extra samples)."""
    out = []
    prev_actions: List[str] = []
    for idx, step in enumerate(traj.steps):
        if step.action is None:
            continue
        if ctx.remaining("pcagente") <= 0:
            break
        if ctx.decontaminate(traj.task, "pcagente"):
            ctx.reject("pcagente", "osworld_contaminated", len(traj.steps))
            return []
        dup, reason = ctx.dedup.consider(image_phash=step.phash,
                                         signals=step.signals,
                                         task_text=traj.task,
                                         action_text=step.action_text)
        if dup:
            ctx.reject("pcagente", reason)
            continue
        from processing.quality import wait_sample_allowed
        if step.action.verb == "wait" and not wait_sample_allowed(prev_actions):
            ctx.reject("pcagente", "repeated_wait")
            continue
        spec = build_single(traj, idx)
        spec.task_type = "action"
        spec.metadata["_is_first_step"] = (idx == 0)
        spec.metadata["_prev_steps"] = traj.steps[max(0, idx - 4):idx]
        spec.metadata["_state_changed"] = (None if step.prev_phash is None else
            bin(step.phash ^ step.prev_phash).count("1") > 6)
        sample = assemble_sample(spec, ctx, trajectory=traj)
        if sample is None:
            continue
        out.append(sample)
        prev_actions.append(step.action_text)
        ctx.consume("pcagente")
    return out


def run(ctx: BuildContext) -> List[dict]:
    """Stream the PC-Agent-E zip via range requests; process task by task."""
    ctx.state.set_target("pcagente", int(ctx.state.selected_total("pcagente")) + int(ctx.quota.get("pcagente", 0)))
    out: List[dict] = []
    url = hf_url(REPO, ZIP_PATH, revision=REVISION)
    zf = open_remote_zip(url, ctx.http())
    tasks = task_files_from_zip(zf)
    for tid, jsonl_path, md_path in tasks:
        if ctx.remaining("pcagente") <= 0:
            break
        key = f"{REPO}:{ZIP_PATH}:{tid}"
        if ctx.state.is_shard_done("pcagente", key):
            continue
        jsonl_text = zf.read(jsonl_path).decode("utf-8", "replace")
        md_text = zf.read(md_path).decode("utf-8", "replace") if md_path else ""
        needed = set(re.findall(r'"screenshot":\s*"([^"]+)"', jsonl_text))
        images: Dict[str, bytes] = {}
        for shot in sorted(needed):
            member = f"data/events/{shot}"
            try:
                images[shot] = zf.read(member)
            except KeyError:
                continue
        traj = build_trajectory(tid, jsonl_text, md_text, images)
        del images
        if traj is None:
            ctx.reject("pcagente", "unparseable_task")
            ctx.state.mark_shard_done("pcagente", key)
            ctx.state.save()
            continue
        # spec section 11: ALL public samples are used; no downweighting here
        got = samples_for_trajectory(traj, ctx)
        out.extend(got)
        ctx.persist_samples(got)
        ctx.state.add_selected("pcagente", len(got), trajectories=1)
        ctx.state.mark_shard_done("pcagente", key)
        ctx.state.save()
    return out
