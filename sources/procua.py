"""ProCUA-SFT adapter (nvidia/ProCUA-SFT, CC BY 4.0).

Verified layout (2026-08):
    shards/procua_sft_NNNNN.tar.zst   - 50 shards, ~18.5 GB each, 853 GB total
    manifest.jsonl                     - per-shard {archive, bytes, idx, parts,
                                         run_count, trajectory_count}
    runs_manifest.jsonl                - run -> shard mapping
    inside each shard: part_*/<run>/<trajectory_id>/trajectory.json + *.png

trajectory.json (from the dataset card):
    {trajectory_id, metadata, goal, steps: [{subgoal, subgoal_intent,
     actions: [{screenshot, pyautogui_command, action_type,
                action_generation{thought,action,code}, raw_reasoning,
                raw_response}]}]}

SAMPLING (spec section 7): stratified; trajectory preference 6-40 actions;
per-trajectory cap 4; temporal bucket coverage; 2x positive weighting on
recovery/verification/save/export/dialog/sort/rank/quantity/multi-target/
multi-app/uncommon/scroll+read; downweight trivial/repeated steps.

STORAGE: shards are NEVER stored locally. Each shard is STREAMED over HTTP
(zstd + sequential tar 'r|'); screenshots for the current trajectory are
buffered in memory only until its trajectory.json passes; chosen images are
written as WebP and everything else is discarded. The HTTP stream is closed
as soon as the quota is filled.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Iterator, List, Optional, Tuple

from processing.assemble import assemble_sample
from processing.coordinates import parse_pyautogui
from processing.dedup import phash
from processing.images import load_image, png_dimensions
from processing.remote_access import fetch_bytes, hf_url, stream_tar_zst
from processing.sampling import (action_representation_specs,
                                   detect_task_signals, deterministic_keep,
                                   trajectory_priority)
from processing.windows import (CHUNK, WINDOW, Step, Trajectory)
from .common import BuildContext
from .revisions import source_revision

REPO = "nvidia/ProCUA-SFT"
REVISION = source_revision("procua")  # immutable Run 1 snapshot
MANIFEST = "manifest.jsonl"
RUNS_MANIFEST = "runs_manifest.jsonl"

_BUFFER_LIMIT_BYTES = 600 << 20  # hard bound on in-flight screenshot bytes

_SAVE_EXPORT_RE = re.compile(r"\bsave\b|\bexport\b|\bsave as\b|\boverwrite\b", re.I)
_DIALOG_RE = re.compile(r"dialog|modal|popup|confirm", re.I)
_SCROLL_RE = re.compile(r"scroll", re.I)
_VERIFY_RE = re.compile(r"verif|check|confirm|ensure|make sure", re.I)


def list_shards(ctx: BuildContext) -> List[Dict]:
    manifest = fetch_bytes(hf_url(REPO, MANIFEST, revision=REVISION), session=ctx.http()).decode("utf-8")
    return [json.loads(l) for l in manifest.splitlines() if l.strip()]


def parse_trajectory(traj_json: dict, pngs: Dict[str, bytes], source_dir: str = "") -> Optional[Trajectory]:
    traj_id = str(traj_json.get("trajectory_id") or "unknown")
    goal = (traj_json.get("goal") or "").strip()
    if not goal:
        return None
    parts = [p for p in source_dir.split("/") if p]
    part = next((p for p in parts if p.startswith("part_")), "unknown_part")
    run_name = "unknown_run"
    if part in parts:
        i_part = parts.index(part)
        if i_part + 1 < len(parts):
            run_name = parts[i_part + 1]
    steps: List[Step] = []
    prev_hash = None
    prev_action = None
    idx = 0
    continuity_id = 0
    for step_group in traj_json.get("steps", []) or []:
        subgoal = step_group.get("subgoal") or ""
        intent = step_group.get("subgoal_intent") or ""
        for act in step_group.get("actions", []) or []:
            shot = act.get("screenshot")
            if not shot or shot not in pngs:
                # Never infer state continuity across a missing visual/action.
                continuity_id += 1
                prev_hash = None
                prev_action = None
                continue
            data = pngs[shot]
            dims = png_dimensions(data)
            if not dims:
                continuity_id += 1
                prev_hash = None
                prev_action = None
                continue
            command = act.get("pyautogui_command") or ""
            if not command and isinstance(act.get("action_generation"), dict):
                command = act["action_generation"].get("code") or ""
            action = parse_pyautogui(command, *dims)
            if action is None:
                # Unsupported/ambiguous source commands create a trajectory
                # gap. The next valid screenshot remains usable as a fresh
                # pre-action state but must not be compared causally to the
                # previous accepted action.
                continuity_id += 1
                prev_hash = None
                prev_action = None
                continue
            hsh = phash(load_image(data))
            from processing.quality import visual_effect_expected
            no_change = (prev_hash is not None
                         and bin(hsh ^ prev_hash).count("1") <= 6
                         and visual_effect_expected(prev_action))
            signals = detect_task_signals(goal, subgoal)
            if no_change:
                signals.add("no_state_change")
            if _SAVE_EXPORT_RE.search(goal) or _SAVE_EXPORT_RE.search(subgoal):
                signals.add("save")
            if _DIALOG_RE.search(goal) or _DIALOG_RE.search(subgoal):
                signals.add("modal_dialog")
            if _VERIFY_RE.search(subgoal):
                signals.add("verification")
            if _SCROLL_RE.search(command or ""):
                signals.add("scroll_read")
            app = _app_of(traj_json)
            steps.append(Step(
                step_id=f"{traj_id}_a{idx}", image_bytes=data, image_size=dims,
                action=action, phash=hsh, prev_phash=prev_hash,
                subgoal=subgoal, signals=signals,
                metadata={"subgoal_intent": intent, "action_type": act.get("action_type"),
                          "source_command": command, "continuity_id": continuity_id,
                          "collection_run": run_name, "part": part,
                          "group_id": f"procua::{run_name}"},
            ))
            prev_hash = hsh
            prev_action = action
            idx += 1
    if not steps:
        return None
    # Future-state evidence is audit metadata only. It is never used to invent
    # reasoning, but allows quality reports to flag actions whose expected
    # visible effect never appears in the next recorded pre-action frame.
    for i, st in enumerate(steps[:-1]):
        nxt = steps[i + 1]
        if st.metadata.get("continuity_id") != nxt.metadata.get("continuity_id"):
            st.metadata["next_state_changed"] = None
        else:
            st.metadata["next_state_changed"] = bin(st.phash ^ nxt.phash).count("1") > 6
    steps[-1].metadata["next_state_changed"] = None
    return Trajectory(trajectory_id=f"procua_{traj_id}", task=goal, steps=steps,
                      app=app, source="procua",
                      metadata={"raw_id": traj_id, "collection_run": run_name,
                                "part": part, "group_id": f"procua::{run_name}"})


_APP_TEXT_RE = [
    (re.compile(r"libreoffice calc|spreadsheet|\bexcel\b|\bcolumn|\bcell", re.I), "libreoffice_calc"),
    (re.compile(r"libreoffice writer|\bdocument\b|\bword\b", re.I), "libreoffice_writer"),
    (re.compile(r"libreoffice impress|\bslide|\bpresentation|powerpoint", re.I), "libreoffice_impress"),
    (re.compile(r"vs code|visual studio code|\bide\b|\bterminal\b|\bcode editor", re.I), "vscode"),
    (re.compile(r"\bchrome\b|\bbrowser\b|\bfirefox\b", re.I), "browser"),
    (re.compile(r"file(s)? manager|\bfolder\b|\bzip\b|\bcompress|\barchive", re.I), "file_manager"),
    (re.compile(r"\bvlc\b|media player", re.I), "vlc"),
    (re.compile(r"\bgimp\b|\bimage editor", re.I), "gimp"),
]


def _app_of(traj_json: dict, shot_path: str = "") -> str:
    """ProCUA metadata has NO application field (audit 2026-08); derive the
    app from the goal text for diversity tracking."""
    meta = (traj_json.get("metadata") or {})
    for key in ("application", "app"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            return v.lower()[:40]
    goal = traj_json.get("goal") or ""
    for pat, name in _APP_TEXT_RE:
        if pat.search(goal):
            return name
    return "desktop"


def samples_for_trajectory(traj: Trajectory, ctx: BuildContext) -> List[dict]:
    out: List[dict] = []
    if ctx.decontaminate(traj.task, "procua"):
        ctx.reject("procua", "osworld_contaminated", len(traj.steps))
        return []
    cap = ctx.config.get("per_trajectory_cap", 4)
    specs = [(spec, traj) for spec in action_representation_specs(traj, cap=cap)]
    if not specs:
        return []

    from processing.sampling import anchor_eligible
    for spec, traj_like in specs:
        if ctx.remaining("procua") <= 0:
            break
        step = spec.current_step
        if step.action is None or not anchor_eligible(step):
            continue  # lone key_down/move stay in history, never anchor
        dup, reason = ctx.dedup.consider(image_phash=step.phash, signals=step.signals,
                                         task_text=traj.task,
                                         action_text=step.action_text)
        if dup:
            ctx.reject("procua", reason)
            continue
        if spec.representation in (WINDOW, CHUNK):
            end = traj_like.steps.index(spec.current_step)
            spec.metadata["_window_steps"] = traj_like.steps[max(0, end - len(spec.step_ids) + 1):end + 1]
        spec.task_type = "action"
        sample = assemble_sample(spec, ctx, trajectory=traj_like)
        if sample is None:
            continue
        out.append(sample)
        ctx.consume("procua")
    return out


def _trajectory_screenshot_refs(traj_json: dict) -> set:
    refs = set()
    for step_group in traj_json.get("steps", []) or []:
        for act in step_group.get("actions", []) or []:
            shot = act.get("screenshot")
            if isinstance(shot, str) and shot:
                refs.add(shot.lstrip("./"))
    return refs


def _process_member_stream(ctx: BuildContext, shard_url: str,
                           max_bytes: Optional[int]) -> List[dict]:
    """Sequential tar.zst stream with order-independent trajectory assembly.

    Real archives are normally image-first, but production correctness must
    not depend on tar member ordering.  We keep a bounded per-directory image
    buffer and a parsed trajectory manifest until all referenced screenshots
    have arrived.  Byte accounting is decremented on *every* removal path so
    the safety valve reflects actual resident memory.
    """
    out: List[dict] = []
    pending: Dict[str, Dict[str, bytes]] = {}
    pending_json: Dict[str, Tuple[dict, set]] = {}
    pending_bytes = 0
    done_dirs = set()

    def drop_dir(d: str) -> None:
        nonlocal pending_bytes
        imgs = pending.pop(d, {})
        pending_bytes = max(0, pending_bytes - sum(len(v) for v in imgs.values()))
        pending_json.pop(d, None)

    def try_finish(d: str, *, allow_partial: bool = False) -> bool:
        nonlocal pending_bytes
        if d not in pending_json or d in done_dirs:
            return False
        traj_json, refs = pending_json[d]
        imgs = pending.get(d, {})
        # Screenshot references are not required to use the exact tar member
        # spelling.  Dataset revisions have used both full member paths and
        # paths relative to the trajectory directory. Resolve conservatively
        # within this trajectory only, never by a global basename search.
        parse_imgs = dict(imgs)
        resolved = set()
        for ref in refs:
            candidates = [ref]
            if not ref.startswith(d + "/"):
                candidates.append(f"{d}/{ref}")
            for cand in candidates:
                if cand in imgs:
                    parse_imgs[ref] = imgs[cand]
                    resolved.add(ref)
                    break
        if not allow_partial and refs and not refs.issubset(resolved):
            return False
        # Parsing with a partial set is only used at EOF; missing screenshots
        # are dropped by parse_trajectory and the row is rejected if no safe
        # action remains.
        resident = sum(len(v) for v in imgs.values())
        traj = parse_trajectory(traj_json, parse_imgs, source_dir=d)
        pending_bytes = max(0, pending_bytes - resident)
        pending.pop(d, None)
        pending_json.pop(d, None)
        done_dirs.add(d)
        if traj is None:
            ctx.reject("procua", "unparseable_trajectory")
            return True
        if not deterministic_keep(traj.trajectory_id, trajectory_priority(traj)):
            return True
        got = samples_for_trajectory(traj, ctx)
        out.extend(got)
        ctx.state.add_selected("procua", len(got), trajectories=1 if got else 0)
        if len(out) % 20 == 0:
            ctx.state.save()
        return True

    for member, tf in stream_tar_zst(shard_url, byte_limit=max_bytes, session=ctx.http()):
        name = member.name.lstrip("./")
        if not member.isfile():
            continue
        if ctx.remaining("procua") <= 0:
            break
        d, _, base = name.rpartition("/")
        if not d or d in done_dirs:
            continue

        if base == "trajectory.json":
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                traj_json = json.loads(f.read().decode("utf-8", "replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                ctx.reject("procua", "trajectory_json_invalid")
                drop_dir(d)
                done_dirs.add(d)
                continue
            refs = _trajectory_screenshot_refs(traj_json)
            if not refs:
                ctx.reject("procua", "trajectory_without_screenshot_refs")
                drop_dir(d)
                done_dirs.add(d)
                continue
            pending_json[d] = (traj_json, refs)
            try_finish(d)
            continue

        if base.lower().endswith(".png"):
            if member.size > 8 << 20:
                ctx.reject("procua", "oversize_png")
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            data = f.read()
            # trajectory.json screenshot refs are full member paths in this
            # dataset; keep both full path and basename alias only when needed.
            bucket = pending.setdefault(d, {})
            old = bucket.get(name)
            if old is not None:
                pending_bytes -= len(old)
            bucket[name] = data
            pending_bytes += len(data)
            if try_finish(d):
                continue

            while pending_bytes > _BUFFER_LIMIT_BYTES and pending:
                oldest = next(iter(pending))
                ctx.reject("procua", "stream_buffer_eviction")
                drop_dir(oldest)
                done_dirs.add(oldest)

    # A byte-limited smoke can end before all referenced screenshots arrive.
    # Only attempt a partial parse at EOF; production full streams should have
    # complete refs and therefore finish during iteration.
    for d in list(pending_json):
        if ctx.remaining("procua") <= 0:
            break
        try_finish(d, allow_partial=True)
    return out


def _seeded_shard_order(ctx: BuildContext, shards: List[Dict]) -> List[Dict]:
    """Globally diversify ProCUA traversal using the official run manifest.

    The exact manifest field names have changed across snapshots, so this
    parser accepts common archive/shard keys and fails safely to a seeded
    archive permutation when the mapping is unavailable. It never returns the
    old numeric 0,1,2... order.
    """
    import hashlib
    try:
        raw = fetch_bytes(hf_url(REPO, RUNS_MANIFEST, revision=REVISION),
                          session=ctx.http()).decode("utf-8")
        rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    except Exception:
        rows = []
    run_rank = {}
    ranked_runs = sorted(rows, key=lambda r: hashlib.sha256(
        ("jxagent-procua-run-v2:" + str(r.get("run") or r.get("run_id") or r.get("name") or r)).encode()).hexdigest())
    for rank, row in enumerate(ranked_runs):
        archive = row.get("archive") or row.get("shard") or row.get("shard_archive")
        if isinstance(archive, str):
            run_rank[archive] = min(rank, run_rank.get(archive, rank))
        archives = row.get("archives") or row.get("shards")
        if isinstance(archives, list):
            for a in archives:
                if isinstance(a, str):
                    run_rank[a] = min(rank, run_rank.get(a, rank))
    def key(shard):
        archive = str(shard.get("archive") or "")
        return (run_rank.get(archive, 10**12),
                hashlib.sha256(("jxagent-procua-shard-v2:" + archive).encode()).hexdigest())
    return sorted(shards, key=key)


def run(ctx: BuildContext, max_shards: Optional[int] = None,
        stream_byte_limit: Optional[int] = None) -> List[dict]:
    ctx.state.set_target("procua", int(ctx.state.selected_total("procua")) + int(ctx.quota.get("procua", 0)))
    out: List[dict] = []
    shards = _seeded_shard_order(ctx, list_shards(ctx))
    if max_shards:
        shards = shards[:max_shards]
    for shard in shards:
        if ctx.remaining("procua") <= 0:
            break
        archive = shard["archive"]
        if ctx.state.is_shard_done("procua", archive):
            continue
        url = hf_url(REPO, archive, revision=REVISION)
        got = _process_member_stream(ctx, url, stream_byte_limit)
        out.extend(got)
        ctx.persist_samples(got)
        ctx.state.mark_shard_done("procua", archive)
        ctx.state.save()
    return out
