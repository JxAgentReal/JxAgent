"""VideoCUA adapter (ServiceNow/VideoCUA, MIT).

Verified schema (2026-08):
    raw_data/<Application>.zip           - one zip per application (87 total)
      <task_id>/action_log.json          - {task_id, task_instruction, platform,
                                           action_log: [{action_type, timestamp,
                                           action_params{x,y,...}, groundcua_id}]}
      <task_id>/video/video.mp4          - video dimensions decoded per task
    action types: CLICK/MOVE_TO/DRAG_TO/SCROLL/TYPE/PRESS/HOTKEY/KEY_(UP|DOWN)/
                  MOUSE_(UP|DOWN)/TERMINATE_SUCCESS/... (absolute pixels in
                  a task-specific source coordinate frame)

Pipeline per spec section 9: one application zip (or task) at a time; the
video is opened ONCE per task with a persistent PyAV decoder; frames are
decoded only at required timestamps; raw temporary data is deleted
immediately. Tasks are fetched from the remote zip via HTTP range requests,
so even a full application archive is never stored locally.

App caps prevent browser/file-explorer dominance; signals prioritize dialog
handling, wait states, saves, typing+verification, unusual apps, app
switching, drags, keyboard use, error handling.
"""
from __future__ import annotations

import io
import json
import re
from typing import Dict, Iterator, List, Optional, Tuple

from processing.assemble import assemble_sample
from processing.coordinates import parse_videocua_action
from processing.dedup import phash
from processing.quality import visual_effect_expected
from processing.images import load_image
from processing.remote_access import hf_tree, hf_url, open_remote_zip
from processing.sampling import (AppCap, action_representation_specs,
                                   detect_task_signals, deterministic_keep,
                                   trajectory_priority)
from processing.windows import (CHUNK, WINDOW, Step, Trajectory)
from .common import BuildContext
from .revisions import source_revision

REPO = "ServiceNow/VideoCUA"
REVISION = source_revision("videocua")  # immutable Run 1 snapshot

WAIT_GAP_SECONDS = 3.0
MAX_STEPS_PER_TASK = 60

_BROWSER_DOMINANT = {"chromium", "brave", "firefox", "duckduckgo", "edge"}
_EXPLORER_DOMINANT = {"files", "nautilus", "thunar", "pcmanfm", "nemo"}

_PRIORITY_APPS = {"libreoffice", "gimp", "inkscape", "blender", "freecad",
                  "audacity", "darktable", "inkscape", "gnu octave", "gnucash",
                  "gnumeric", "intellij idea", "eclipse", "ardour", "fontforge",
                  "grassgis", "calibre", "joplin", "anki"}


def decode_frames(video_bytes: bytes, timestamps: List[float]) -> Dict[float, bytes]:
    """Open the video ONCE and decode only the required timestamps (PyAV).

    Long gaps are handled by seeking to the preceding keyframe; targets are
    matched by presentation time (frame pts), not by frame counter, so seeking
    cannot misalign. The decoder stays persistent for the whole task.
    Returns {timestamp: PNG-encoded frame bytes}."""
    import av

    out: Dict[float, bytes] = {}
    container = av.open(io.BytesIO(video_bytes))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) or 30.0
        time_base = float(stream.time_base)
        half_frame = 0.5 / fps

        def frame_time(frame) -> float:
            if frame.pts is None:
                return -1.0
            return frame.pts * time_base

        last_t = 0.0
        for target in sorted(set(timestamps)):
            # seek when it saves real decoding work (>3s of frames)
            if target - last_t > 3.0 and last_t >= 0:
                try:
                    container.seek(int(max(0.0, target - 0.2) / time_base),
                                   stream=stream, backward=True)
                except av.error.AVError:
                    pass
            got = None
            try:
                for frame in container.decode(stream):
                    t = frame_time(frame)
                    if t < 0:
                        continue
                    if t >= target - half_frame - 1e-9:
                        got = frame
                        last_t = t
                        break
                    last_t = t
            except av.error.EOFError:
                pass  # stream ended before this target
            if got is None:
                continue
            buf = io.BytesIO()
            got.to_image().save(buf, format="PNG")
            out[target] = buf.getvalue()
    finally:
        container.close()
    return out


def _meta_resolution(task_meta: dict) -> Optional[Tuple[int, int]]:
    """Return an explicit source coordinate frame only when source metadata proves it.

    VideoCUA archives are not guaranteed to use one global desktop resolution.
    We therefore accept only explicit width/height metadata.  Missing metadata is
    handled by using the actual decoded frame dimensions *iff every parsed raw
    point fits that frame*.  We never silently assume 1920x1080.
    """
    candidates = [task_meta]
    for key in ("metadata", "screen", "display", "video_metadata"):
        v = task_meta.get(key)
        if isinstance(v, dict):
            candidates.append(v)
    pairs = (
        ("width", "height"), ("screen_width", "screen_height"),
        ("display_width", "display_height"), ("video_width", "video_height"),
    )
    for obj in candidates:
        for wk, hk in pairs:
            try:
                w, h = int(obj.get(wk, 0)), int(obj.get(hk, 0))
            except Exception:
                continue
            if w > 1 and h > 1:
                return w, h
        res = obj.get("resolution") or obj.get("screen_size") or obj.get("display_size")
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            try:
                w, h = int(res[0]), int(res[1])
            except Exception:
                continue
            if w > 1 and h > 1:
                return w, h
        if isinstance(res, str):
            m = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", res)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
                if w > 1 and h > 1:
                    return w, h
    return None


def _entry_raw_points(entry: dict) -> List[Tuple[float, float]]:
    """Extract absolute raw action-log points without interpreting coordinates."""
    p = entry.get("action_params") or {}
    pts: List[Tuple[float, float]] = []
    if isinstance(p, dict):
        for xk, yk in (("x", "y"), ("start_x", "start_y"), ("end_x", "end_y")):
            if xk in p and yk in p:
                try:
                    pts.append((float(p[xk]), float(p[yk])))
                except Exception:
                    pass
        for key in ("coordinate", "start_coordinate", "end_coordinate"):
            v = p.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    pts.append((float(v[0]), float(v[1])))
                except Exception:
                    pass
    return pts


def _normalize_micro_actions(actions: List[dict]) -> List[dict]:
    """Canonicalize raw pointer micro-actions without inventing visual states.

    VideoCUA represents a drag as MOVE/CLICK -> MOUSE_DOWN -> DRAG_TO ->
    MOUSE_UP, while DRAG_TO itself contains only the destination.  Recover the
    proven drag start from the last pointer location held at MOUSE_DOWN, inject
    it into DRAG_TO, and drop the low-level down/up primitives.  Same-timestamp
    MOVE_TO immediately before CLICK is also redundant and removed.
    """
    import copy
    enriched: List[dict] = []
    last_pointer: Optional[Tuple[float, float]] = None
    held_start: Optional[Tuple[float, float]] = None
    for entry in actions:
        e = copy.deepcopy(entry)
        kind = str(e.get("action_type", "")).upper()
        pts = _entry_raw_points(e)
        if kind == "MOUSE_DOWN":
            held_start = last_pointer
            continue
        if kind == "MOUSE_UP":
            held_start = None
            continue
        if kind == "DRAG_TO":
            p = e.setdefault("action_params", {})
            if p.get("start_x") is None and p.get("start_y") is None and held_start is not None:
                p["start_x"], p["start_y"] = held_start
            # A drag without a proven held start is ambiguous and will be
            # rejected by parse_videocua_action after hardening below.
            if pts:
                last_pointer = pts[-1]
            enriched.append(e)
            continue
        if pts:
            last_pointer = pts[-1]
        enriched.append(e)

    out: List[dict] = []
    for i, entry in enumerate(enriched):
        kind = str(entry.get("action_type", "")).upper()
        if kind == "MOVE_TO" and i + 1 < len(enriched):
            nxt = enriched[i + 1]
            nk = str(nxt.get("action_type", "")).upper()
            try:
                same_ts = abs(float(entry.get("timestamp", 0) or 0) -
                              float(nxt.get("timestamp", 0) or 0)) <= 1e-6
            except Exception:
                same_ts = False
            if same_ts and nk in {"CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"}:
                a, b = _entry_raw_points(entry), _entry_raw_points(nxt)
                if not a or not b or (abs(a[-1][0] - b[0][0]) <= 64 and
                                      abs(a[-1][1] - b[0][1]) <= 64):
                    continue
        out.append(entry)
    return out


def build_trajectory(task_meta: dict, frames: Dict[float, bytes],
                     resolution: Optional[Tuple[int, int]] = None) -> Optional[Trajectory]:
    """Build a trajectory with a proven coordinate-frame mapping.

    Rules:
      * explicit metadata/resolution wins;
      * otherwise the decoded frame itself is the coordinate frame only when
        every raw point used by that task lies inside it;
      * if neither can be proven, reject rather than guess.
    """
    from processing.coordinates import rebase_action_points, validate_point
    from processing.images import png_dimensions

    actions = _normalize_micro_actions(list(task_meta.get("action_log", []) or []))
    task = (task_meta.get("task_instruction") or "").strip()
    if not task or not frames:
        return None
    platform = (task_meta.get("platform") or "unknown").strip()
    task_id = str(task_meta.get("task_id", "na"))
    signals_task = detect_task_signals(task)

    frame_dims_by_ts: Dict[float, Tuple[int, int]] = {}
    for ts, blob in frames.items():
        dims = png_dimensions(blob)
        if dims and dims[0] > 1 and dims[1] > 1:
            frame_dims_by_ts[float(ts)] = (int(dims[0]), int(dims[1]))
    if not frame_dims_by_ts:
        return None

    explicit = resolution or _meta_resolution(task_meta)
    coord_mode = "explicit_metadata" if explicit else "decoded_frame"

    # Without explicit metadata, prove that each raw coordinate fits the actual
    # frame observed for that timestamp. Any outlier makes the task ambiguous.
    if explicit is None:
        for entry in actions:
            ts = float(entry.get("timestamp", 0) or 0)
            dims = frame_dims_by_ts.get(ts)
            if dims is None:
                continue
            for x, y in _entry_raw_points(entry):
                if not (0 <= x <= dims[0] - 1 and 0 <= y <= dims[1] - 1):
                    return None

    steps: List[Step] = []
    prev_ts: Optional[float] = None
    prev_hash: Optional[int] = None
    prev_action = None
    oob_skipped = 0
    transformed = 0
    continuity_id = 0

    for i, entry in enumerate(actions):
        ts = float(entry.get("timestamp", 0) or 0)
        if ts not in frames or ts not in frame_dims_by_ts:
            continuity_id += 1
            prev_hash = prev_action = None
            prev_ts = None
            continue
        frame_dims = frame_dims_by_ts[ts]
        source_dims = explicit or frame_dims
        action = parse_videocua_action(entry, source_dims[0], source_dims[1])
        if action is None:
            continuity_id += 1
            prev_hash = prev_action = None
            prev_ts = None
            continue
        if tuple(frame_dims) != tuple(source_dims):
            action = rebase_action_points(action, source_dims, frame_dims)
            transformed += 1
        if any(not validate_point(x, y, frame_dims[0], frame_dims[1])
               for (x, y) in action.points):
            oob_skipped += 1
            continuity_id += 1
            prev_hash = prev_action = None
            prev_ts = None
            continue

        hsh = phash(load_image(frames[ts]))
        no_change = prev_hash is not None and bin(hsh ^ prev_hash).count("1") <= 6
        signals = set(signals_task)
        # Same-timestamp entries are micro-actions from one UI moment, not
        # evidence that the prior action failed. Require positive elapsed time
        # before inferring a no-state-change recovery signal.
        causal_next_state = prev_ts is not None and ts > prev_ts + 1e-6
        if causal_next_state and no_change and visual_effect_expected(prev_action):
            signals.add("no_state_change")
        if prev_ts is not None and ts - prev_ts >= WAIT_GAP_SECONDS:
            signals.update({"wait", "wait_evidenced"})
        if re.search(r"\bsave\b|\bexport\b|\bdownload\b|\brename\b", task, re.I):
            signals.add("save")
        if re.search(r"dialog|confirm|popup", task, re.I):
            signals.add("modal_dialog")
        steps.append(Step(
            step_id=f"vc_{task_id}_s{i}", image_bytes=frames[ts], image_size=frame_dims,
            action=action, phash=hsh, prev_phash=prev_hash,
            subgoal=platform, signals=signals,
            metadata={"timestamp": ts, "raw_action_type": entry.get("action_type"),
                      "groundcua_id": entry.get("groundcua_id"),
                      "continuity_id": continuity_id},
        ))
        prev_hash, prev_ts, prev_action = hsh, ts, action

    if not steps:
        return None
    coord_meta: dict = {
        "coordinate_mode": coord_mode,
        "source_coordinate_space": list(explicit) if explicit else "per_decoded_frame",
    }
    frame_sizes = sorted({tuple(s.image_size) for s in steps})
    if len(frame_sizes) == 1:
        coord_meta["actual_frame_size"] = list(frame_sizes[0])
    else:
        coord_meta["actual_frame_sizes"] = [list(fs) for fs in frame_sizes]
    if transformed:
        coord_meta["transformed_actions"] = transformed
        # Preserve an explicit, auditable description of the deterministic
        # source->decoded-frame mapping for downstream manifests/tests.
        # The actual point rebasing uses endpoint-preserving scale_point().
        if explicit and len(frame_sizes) == 1:
            coord_meta["transform"] = {
                "from": list(explicit),
                "to": list(frame_sizes[0]),
                "endpoint_preserving": True,
            }
    if oob_skipped:
        coord_meta["coordinate_out_of_bounds_after_transform"] = oob_skipped
    meta = {"task_id": task_id, **coord_meta}
    return Trajectory(trajectory_id=f"videocua_{platform}_{task_id}".replace(" ", "_"),
                      task=task, steps=steps[:MAX_STEPS_PER_TASK], app=platform.lower(),
                      source="videocua", metadata=meta)


def samples_for_trajectory(traj: Trajectory, ctx: BuildContext) -> List[dict]:
    out: List[dict] = []
    if ctx.decontaminate(traj.task, "videocua"):
        ctx.reject("videocua", "osworld_contaminated")
        return []
    cap = ctx.config.get("per_trajectory_cap", 4)
    specs = [(spec, traj) for spec in action_representation_specs(traj, cap=cap)]
    if not specs:
        return []

    from processing.sampling import anchor_eligible
    for spec, traj_like in specs:
        if ctx.remaining("videocua") <= 0:
            break
        step = spec.current_step
        if step.action is None or not anchor_eligible(step):
            continue  # MOVE_TO never anchors (45% of raw actions)
        dup, reason = ctx.dedup.consider(image_phash=step.phash, signals=step.signals,
                                         task_text=traj.task,
                                         action_text=step.action_text)
        if dup:
            ctx.reject("videocua", reason)
            continue
        if spec.representation in (WINDOW, CHUNK):
            spec.metadata["_window_steps"] = traj_like.steps[
                traj_like.steps.index(spec.current_step) - len(spec.step_ids) + 1:
                traj_like.steps.index(spec.current_step) + 1]
        spec.task_type = "action"
        sample = assemble_sample(spec, ctx, trajectory=traj_like)
        if sample is None:
            continue
        out.append(sample)
        ctx.consume("videocua")
    return out


def run(ctx: BuildContext, max_app_zip_bytes: Optional[int] = None) -> List[dict]:
    """Process VideoCUA app-by-app via remote zip range access."""
    remaining = int(ctx.quota.get("videocua", 0))
    full_target = int(ctx.state.selected_total("videocua")) + remaining
    ctx.state.set_target("videocua", full_target)
    out: List[dict] = []
    # Application caps are a property of the full requested source mixture.
    # Recomputing them from only the remaining tail can make a resumed build
    # impossible to fill or distort the application distribution.
    total_quota = full_target
    entries = hf_tree(REPO, "raw_data", revision=REVISION, session=ctx.http())
    apps = []
    for e in entries:
        if e.get("type") == "file" and e["path"].endswith(".zip"):
            apps.append((e["path"], e.get("lfs", {}).get("size", e.get("size", 0))))
    # Cap must still allow the configured quota to be physically reachable.
    # Reserve ~25% headroom for source-specific rejection while preventing one
    # application from dominating the mix.
    import math
    min_reachable = math.ceil(total_quota / max(1, len(apps)))
    prior_apps = dict((ctx.config.get("_resume_app_counts", {}) or {}).get("videocua", {}))
    app_cap = AppCap(cap=max(40, math.ceil(min_reachable * 1.25)), counts=prior_apps)
    if ctx.smoke:
        apps.sort(key=lambda a: a[1])          # smallest archive first
    else:
        # priority: unusual professional apps first, browsers capped later
        apps.sort(key=lambda a: (0 if _is_priority(a[0]) else 1, a[1]))

    for zip_path, size in apps:
        if ctx.remaining("videocua") <= 0:
            break
        app_name = zip_path.rsplit("/", 1)[-1].removesuffix(".zip")
        if not app_cap.allow(app_name):
            continue
        before = ctx.remaining("videocua")
        got = _process_app_zip(ctx, zip_path, app_name, app_cap)
        out.extend(got)
        # _process_app_zip checkpoints selected counts per task. Do not add
        # them a second time here; double accounting made progress/targets
        # drift until the next durable-row reconciliation.
        ctx.state.save()
    return out


def _is_priority(app_path: str) -> bool:
    low = app_path.lower()
    return any(p in low for p in _PRIORITY_APPS)


def _process_app_zip(ctx: BuildContext, zip_path: str, app_name: str,
                     app_cap: AppCap) -> List[dict]:
    out: List[dict] = []
    shard_key = f"{REPO}:{zip_path}"
    if ctx.state.is_shard_done("videocua", shard_key):
        return out
    url = hf_url(REPO, zip_path, revision=REVISION)
    try:
        zf = open_remote_zip(url, ctx.http())
        names = zf.namelist()
    except Exception:
        ctx.reject("videocua", "app_zip_open_failed_retryable")
        return out
    task_ids = sorted({n.split("/")[0] for n in names
                       if n.count("/") >= 1 and n.split("/")[0].isdigit()})
    made = 0
    for tid in task_ids:
        if ctx.remaining("videocua") <= 0 or not app_cap.allow(app_name):
            break
        task_key = f"{shard_key}:{tid}"
        if ctx.state.is_shard_done("videocua", task_key):
            continue
        try:
            log = json.loads(zf.read(f"{tid}/action_log.json").decode("utf-8"))
        except Exception:
            ctx.reject("videocua", "action_log_failed")
            ctx.state.mark_shard_done("videocua", task_key)
            continue
        stamps = [float(a.get("timestamp", 0) or 0) for a in log.get("action_log", [])]
        if not stamps:
            ctx.state.mark_shard_done("videocua", task_key)
            continue
        try:
            video = zf.read(f"{tid}/video/video.mp4")
        except Exception:
            ctx.reject("videocua", "video_fetch_failed_retryable")
            continue
        try:
            frames = decode_frames(video, stamps)
        except Exception:
            ctx.reject("videocua", "video_decode_failed_retryable")
            del video
            continue
        del video
        traj = build_trajectory(log, frames, None)
        del frames
        if traj is None:
            ctx.reject("videocua", "unparseable_or_ambiguous_coordinate_task")
            ctx.state.mark_shard_done("videocua", task_key)
            continue
        oob = (traj.metadata or {}).get("coordinate_out_of_bounds_after_transform", 0)
        if oob:
            ctx.reject("videocua", "coordinate_out_of_bounds_after_transform", oob)
        if not deterministic_keep(traj.trajectory_id, trajectory_priority(traj)):
            ctx.state.mark_shard_done("videocua", task_key)
            continue
        got = samples_for_trajectory(traj, ctx)
        out.extend(got)
        made += len(got)
        app_cap.record(app_name, len(got))
        ctx.persist_samples(got)
        ctx.state.add_selected("videocua", len(got), trajectories=1)
        ctx.state.mark_shard_done("videocua", task_key)
        ctx.state.save()
    ctx.state.mark_shard_done("videocua", shard_key)
    return out
