#!/usr/bin/env python3
"""JxAgent Run 1 dataset builder.

Builds the ms-swift multimodal JSONL dataset (2026-08 quality pass; the
single source of truth for quotas is SOURCE_TARGETS below, mirrored in
configs/dataset.yaml and PROJECT_STATUS.json):

    ProCUA:        46,000   (nvidia/ProCUA-SFT, streamed .tar.zst shards)
    GUI-360 Lite:  16,000   (cua-lite/GUI-360, parquet streaming)
    VideoCUA:      17,500   (ServiceNow/VideoCUA, per-task remote zip ranges)
    GroundCUA:      4,000   (ServiceNow/GroundCUA, per-file remote fetch)
    PC-Agent-E:     4,503   (henryhe0123/PC-Agent-E, remote zip ranges)
    General Replay: 7,500   (public replay mixture)
    TOTAL quota:   95,503   (realized ~95.3k after the PC-Agent-E quality gate)

Low-storage by design: no source is mirrored locally; large archives are
streamed or range-read; every stage is resumable with atomic state.

Examples:
    tiny smoke tests (one source at a time, network required):
      python build_jxagent_dataset.py --output ./SmokePC --sources pcagente --pcagente-count 20
      python build_jxagent_dataset.py --output ./SmokeGUI360 --sources gui360 --gui360-count 20
      python build_jxagent_dataset.py --output ./SmokeVideo --sources videocua --videocua-count 20
      python build_jxagent_dataset.py --output ./SmokeGround --sources groundcua --groundcua-count 20
      python build_jxagent_dataset.py --output ./SmokeProCUA --sources procua --procua-count 20 --procua-stream-bytes 300000000
      python build_jxagent_dataset.py --output ./SmokeReplay --sources replay --replay-count 20

    full build (counts below are already the CLI defaults; explicit for
    clarity — see HANDOFF §23.4 for the authoritative build-host sequence):
      python build_jxagent_dataset.py \
        --output ./JxAgentData \
        --sources pcagente groundcua gui360 replay videocua procua \
        --pcagente-count 4503 --groundcua-count 4000 --gui360-count 16000 \
        --gui360-grounding 2500 --replay-count 7500 --videocua-count 17500 \
        --procua-count 46000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processing.decontamination import Decontaminator, fetch_reference_instructions
from processing.dedup import DedupIndex
from processing.reasoning import ReasoningGate, TARGET_RATE
from processing.splitting import assign_splits
from processing.selection import (candidate_targets as selection_candidate_targets,
                                  select_best_valid, load_frontier_scores,
                                  gui360_final_subtargets)
from processing.state import BuildLock, BuildState
from processing.validation import (config_hash_of, environment_snapshot,
                                   finalize, quality_audit, summarize_failures)
from sources.common import BuildContext
from sources.revisions import (REVISIONS as SOURCE_REVISIONS,
                               manifest_view as revisions_manifest_view,
                               unresolved_sources)

SOURCE_ORDER = ["pcagente", "groundcua", "gui360", "replay", "videocua", "procua"]
# 2026-08 data-quality pass final mixture (see DATA_QUALITY_REPORT.md):
# better examples, not more: synthetic ProCUA trimmed, human VideoCUA grown,
# GUI-360 grounding reduced (overlaps GroundCUA), GroundCUA bucket-balanced.
SOURCE_TARGETS = {
    "procua": 46000, "gui360": 16000, "videocua": 17500,
    "groundcua": 4000, "pcagente": 4503, "replay": 7500,
}
FINAL_TOTAL = sum(SOURCE_TARGETS.values())  # 95,503 quota (PC-Agent-E quality
# gate realistically yields ~4.3k -> realized total ~95.3k, within 95-105k)

SOURCE_MODULES = {
    "pcagente": "sources.pc_agent_e",
    "groundcua": "sources.groundcua",
    "gui360": "sources.gui360",
    "videocua": "sources.videocua",
    "procua": "sources.procua",
    "replay": "sources.replay",
}

SOURCE_INFO = {
    "procua": {"repo": "nvidia/ProCUA-SFT", "license": "CC-BY-4.0",
               "coords": "absolute pixels (pyautogui commands)"},
    "gui360": {"repo": "cua-lite/GUI-360 (origin vyokky/GUI-360)",
               "license": "MIT (origin)",
               "coords": "normalized integers [0,1000] vs metadata.others.resolution"},
    "videocua": {"repo": "ServiceNow/VideoCUA", "license": "MIT",
                 "coords": "absolute pixels in explicit source metadata when available; otherwise proven against each decoded frame"},
    "groundcua": {"repo": "ServiceNow/GroundCUA", "license": "MIT",
                  "coords": "float pixel bbox [x1,y1,x2,y2]"},
    "pcagente": {"repo": "henryhe0123/PC-Agent-E", "license": "MIT",
                 "coords": "absolute pixels in action strings"},
    "replay": {"repo": "mixed public (Magicoder/orca-math/smoltalk/cauldron/hermes)",
               "license": "per-source, documented in sources/replay.py",
               "coords": "n/a (text-only + VQA images)"},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="JxAgent Run 1 dataset builder")
    p.add_argument("--output", required=True, help="dataset output root (e.g. ./JxAgentData)")
    p.add_argument("--sources", nargs="+", default=SOURCE_ORDER, choices=SOURCE_ORDER,
                   help="sources to build (default: recommended build order)")
    p.add_argument("--config", default=None, help="path to configs/dataset.yaml overrides")
    p.add_argument("--procua-count", type=int, default=SOURCE_TARGETS["procua"])
    p.add_argument("--gui360-count", type=int, default=SOURCE_TARGETS["gui360"])
    p.add_argument("--videocua-count", type=int, default=SOURCE_TARGETS["videocua"])
    p.add_argument("--groundcua-count", type=int, default=SOURCE_TARGETS["groundcua"])
    p.add_argument("--pcagente-count", type=int, default=SOURCE_TARGETS["pcagente"])
    p.add_argument("--replay-count", type=int, default=SOURCE_TARGETS["replay"])
    # Replay sub-counts default to the canonical mixture defined ONCE in
    # sources/replay.py CATEGORIES; any flag set here overrides exactly that
    # category (tests assert the default resolution stays 1600/1500/1700/1400/1300).
    p.add_argument("--replay-coding", type=int, default=None)
    p.add_argument("--replay-math", type=int, default=None)
    p.add_argument("--replay-instruction", type=int, default=None)
    p.add_argument("--replay-vqa", type=int, default=None)
    p.add_argument("--replay-tool", type=int, default=None)
    p.add_argument("--gui360-grounding", type=int, default=2500,
                   help="GUI-360 difficult grounding slice (reduced: overlaps GroundCUA)")
    p.add_argument("--validation-pct", type=float, default=3.0)
    p.add_argument("--no-resume", action="store_true", help="ignore existing state")
    p.add_argument("--no-decontamination", action="store_true")
    p.add_argument("--offline", action="store_true",
                   help="never touch the network (sources requiring it are skipped)")
    p.add_argument("--finalize-only", action="store_true",
                   help="re-finalize from state/selected_samples.jsonl without building")
    p.add_argument("--quality-audit", type=int, default=100)
    p.add_argument("--skip-quality-audit", action="store_true")
    # smoke-test knobs (keep network usage tiny)
    p.add_argument("--smoke", action="store_true", help="tiny real-source smoke build")
    p.add_argument("--procua-stream-bytes", type=int, default=None,
                   help="stop the ProCUA shard HTTP stream after this many uncompressed bytes")
    p.add_argument("--procua-max-shards", type=int, default=None)
    p.add_argument("--videocua-max-app-bytes", type=int, default=None)
    # reproducibility / safety knobs (audit patch 2026-08-16)
    p.add_argument("--clear-lock", action="store_true",
                   help="force-remove state/build.lock before starting (stale-lock recovery)")
    p.add_argument("--no-enforce-quotas", action="store_true",
                   help="skip production quota-acceptance enforcement (debug/smoke only)")
    p.add_argument("--frontier-scores", default=None,
                   help="optional JSONL of objectively verified base-model frontier scores")
    p.add_argument("--interface-manifest", default=os.environ.get("JXAGENT_INTERFACE_MANIFEST"),
                   help="verified Qwen3.8 native interface manifest; mandatory for any >=5k preproduction/full build")
    p.add_argument("--extra-decontamination-reference", action="append", default=[],
                   help="additional benchmark reference JSON/JSONL/TXT; may repeat")
    p.add_argument("--allow-unversioned", action="store_true",
                   help="explicitly permit UNVERSIONED DEVELOPMENT MODE when no git "
                        "commit is available (never valid for a production build)")
    return p.parse_args(argv)


def get_builder_revision(allow_unversioned: bool) -> str:
    """Git commit of this checkout. Production builds REQUIRE a real commit;
    offline development may explicitly opt into UNVERSIONED DEVELOPMENT MODE.
    Never creates a repository or touches any remote."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=15, cwd=os.path.dirname(
                                 os.path.abspath(__file__))).stdout.strip()
        if sha and len(sha) >= 7 and " " not in sha:
            return sha
    except Exception:  # noqa: BLE001 - no git / not a repo / git missing
        pass
    env_opt_in = os.environ.get("JXAGENT_ALLOW_UNVERSIONED", "") == "1"
    if allow_unversioned or env_opt_in:
        print("[builder] WARNING: UNVERSIONED DEVELOPMENT MODE — no git commit; "
              "artifacts from this build are NOT reproducibly attributable. "
              "Never publish a dataset built this way.", flush=True)
        return "UNVERSIONED-DEV"
    raise SystemExit(
        "FATAL: builder revision unavailable — this project is not a git checkout "
        "(git rev-parse HEAD failed). A production build must be attributable to an "
        "exact builder revision.\n"
        "  - owner action: run `git init && git add -A && git commit` in JxAgent/ "
        "(no remote is created or pushed), then rerun;\n"
        "  - offline development only: pass --allow-unversioned (UNVERSIONED "
        "DEVELOPMENT MODE; artifacts must not be published).")


def is_production_build(args, counts: dict) -> bool:
    """Production = full-scale, non-smoke build (or full-scale finalize-only).
    Only production builds enforce revision resolution, builder identity and
    quota acceptance; smokes/tests stay permissive. Scale is judged by the
    quotas of the sources ACTUALLY requested on this run."""
    active = {k: v for k, v in counts.items() if k in args.sources} or dict(counts)
    return (not args.smoke) and sum(active.values()) >= 10000


def build_config_snapshot(args, config: dict, counts: dict) -> dict:
    """COMPLETE effective configuration (defaults + yaml + CLI), canonicalized
    for hashing. Same effective config -> same config hash, always."""
    from processing.quality import (BUCKET_A_MIN, BUCKET_B_MIN, BUCKET_C_MIN)
    from processing.windows import (REPRESENTATION_RATIOS, WINDOW_MIN, WINDOW_MAX,
                                    CHUNK_MIN, CHUNK_MAX)
    replay_mixture = resolve_replay_counts(args)
    return {
        "source_targets": dict(counts),
        "source_revisions": revisions_manifest_view(),
        "gui360_grounding": args.gui360_grounding,
        "replay_mixture": replay_mixture,
        "reasoning": {"rate": config.get("reasoning", {}).get("rate", TARGET_RATE)},
        "representation_ratios": dict(REPRESENTATION_RATIOS),
        "window_steps": [WINDOW_MIN, WINDOW_MAX],
        "chunk_steps": [CHUNK_MIN, CHUNK_MAX],
        "images": config.get("images", {}),
        "context_budget": config.get("context_budget", 8192),
        "per_trajectory_cap": config.get("per_trajectory_cap", 4),
        "dedup": config.get("dedup", {}),
        "split": {"validation_pct": args.validation_pct,
                  "method": "group-aware semantic/provenance md5 bucketing"},
        "quality_thresholds": {"A_min": BUCKET_A_MIN, "B_min": BUCKET_B_MIN,
                               "C_min": BUCKET_C_MIN},
        "decontamination": {"enabled": not args.no_decontamination,
                            "ngram": 8, "jaccard_threshold": 0.5,
                            "reference": "xlang-ai/OSWorld",
                            "reference_revision": SOURCE_REVISIONS["osworld"]["sha"],
                            "reference_cache_sha256": SOURCE_REVISIONS["osworld"]["cache_sha256"],
                            "extra_references": _reference_file_manifest(args.extra_decontamination_reference)},
        "validation_tolerances": config.get("validation_tolerances", {}),
        "selection": config.get("selection", {}),
        "coverage_floors": config.get("coverage_floors", {}),
        "loss_token_gate": config.get("loss_token_gate", {}),
        "selection_policy": "best_valid_v2: quality + token efficiency + diversity + optional verified frontier scoring",
        "seed_note": "all ranking ties, split and replay sampling are deterministic hash functions",
    }


def load_config(path: str | None) -> dict:
    cfg = {
        "context_budget": 8192,
        "per_trajectory_cap": 4,
        "images": {"webp_quality": 80, "max_long": 1600, "grounding_max_long": 1920},
        # Run 1 hardening: synthetic reasoning is disabled by default after
        # the manual factuality audit.  The YAML may only opt in explicitly
        # after an independent >=99.5% reasoning-accuracy gate.
        "reasoning": {"rate": TARGET_RATE},
        "dedup": {"phash_threshold": 6},
        "split": {"validation_pct": 3.0},
        "selection": {"enabled": True,
                      "oversample_factors": {"procua": 1.50, "gui360": 1.25,
                                             "videocua": 1.25, "groundcua": 1.35,
                                             "pcagente": 1.0, "replay": 1.15},
                      "max_per_group": {}},
        "coverage_floors": {},
        "loss_token_gate": {"max_auxiliary_task_share": 0.20},
    }
    if path and os.path.exists(path):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def ensure_layout(root: str):
    for sub in ("final", "state", "manifests", "rlvr",
                "images/procua", "images/gui360", "images/videocua",
                "images/groundcua", "images/pcagente", "images/replay"):
        os.makedirs(os.path.join(root, *sub.split("/")), exist_ok=True)


def _reference_file_manifest(paths) -> list:
    rows = []
    for raw in paths or []:
        p = os.path.abspath(raw)
        if not os.path.isfile(p):
            rows.append({"path": p, "missing": True})
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        rows.append({"path": p, "sha256": h.hexdigest(), "bytes": os.path.getsize(p)})
    return rows


def build_decontaminator(ctx_args, state_dir: str, session) -> tuple[Decontaminator | None, dict]:
    if ctx_args.no_decontamination:
        return None, {"skipped": True, "reason": "disabled by flag"}
    shared = [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".cache", "osworld_instructions.json")]
    try:
        pairs = fetch_reference_instructions(state_dir, session=session,
                                             offline=ctx_args.offline,
                                             shared_cache_paths=shared)
        d = Decontaminator()
        for task_id, instruction in pairs:
            d.add_reference(task_id, instruction)
        extra_count = 0
        if ctx_args.extra_decontamination_reference:
            from processing.decontamination import load_reference_file
            for ref_path in ctx_args.extra_decontamination_reference:
                for task_id, instruction in load_reference_file(ref_path):
                    d.add_reference(task_id, instruction)
                    extra_count += 1
        return d, {"reference_instructions": len(pairs),
                   "extra_reference_instructions": extra_count,
                   "extra_reference_files": _reference_file_manifest(ctx_args.extra_decontamination_reference)}
    except Exception as e:  # noqa: BLE001
        if ctx_args.offline:
            return None, {"skipped": True, "reason": f"offline and no cache: {e}"}
        raise


def resolve_replay_counts(args) -> dict:
    """Canonical Replay mixture resolution (pure; tested).

    Defaults come from sources.replay.CATEGORIES — the single source of
    truth. CLI sub-counts override exactly their category; a non-default
    --replay-count scales the resolved mixture proportionally."""
    from sources.replay import CATEGORIES as REPLAY_CATEGORIES
    overrides = {"coding": args.replay_coding, "math": args.replay_math,
                 "instruction": args.replay_instruction,
                 "vqa": args.replay_vqa, "tool": args.replay_tool}
    counts = {cat: (overrides[cat] if overrides[cat] is not None else default)
              for cat, (_, default) in REPLAY_CATEGORIES.items()}
    if args.replay_count != SOURCE_TARGETS["replay"]:
        total = args.replay_count
        scale = total / SOURCE_TARGETS["replay"]
        counts = {k: max(1, int(round(v * scale))) for k, v in counts.items()}
    if args.smoke:
        counts = {k: max(1, args.replay_count // 5) for k in counts}
    return counts


def run_source(name: str, ctx: BuildContext, args) -> list:
    import importlib
    mod = importlib.import_module(SOURCE_MODULES[name])
    if name == "procua":
        limit = args.procua_stream_bytes
        if limit is None and (args.smoke or args.procua_count <= 500):
            limit = 400 << 20  # tiny builds must not stream a whole 18.5 GB shard
        return mod.run(ctx, max_shards=args.procua_max_shards,
                       stream_byte_limit=limit)
    if name == "gui360":
        q = int((ctx.config.get("_candidate_source_targets") or {}).get("gui360", args.gui360_count))
        if args.smoke:
            gw = max(2, q // 3)
            uw = max(2, q // 3)
        else:
            sub = gui360_final_subtargets(q, args.gui360_grounding, 2650)
            gw, uw = sub["grounding"], sub["understanding"]
        return mod.run(ctx, grounding_want=gw, understanding_want=uw)
    if name == "replay":
        final_mix = resolve_replay_counts(args)
        candidate_total = int((ctx.config.get("_candidate_source_targets") or {}).get("replay", args.replay_count))
        total_final = max(1, sum(final_mix.values()))
        raw = {k: candidate_total * v / total_final for k, v in final_mix.items()}
        requested = {k: int(v) for k, v in raw.items()}
        need = candidate_total - sum(requested.values())
        for k in sorted(raw, key=lambda x: (raw[x] - int(raw[x]), x), reverse=True)[:max(0, need)]:
            requested[k] += 1
        prior = ctx.config.get("_resume_replay_categories", {}) or {}
        remaining = {cat: max(0, n - int(prior.get(cat, 0)))
                     for cat, n in requested.items()}
        return mod.run(ctx, remaining)
    if name == "videocua":
        return mod.run(ctx, max_app_zip_bytes=args.videocua_max_app_bytes)
    return mod.run(ctx)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = os.path.abspath(args.output)
    ensure_layout(root)

    # single-builder lock: a second live builder on the same output corrupts
    # state; stale locks (dead pid) are auto-recovered, --clear-lock forces
    lock = BuildLock(os.path.join(root, "state"))
    lock.acquire(force_clear=args.clear_lock)

    counts = {
        "procua": args.procua_count, "gui360": args.gui360_count,
        "videocua": args.videocua_count, "groundcua": args.groundcua_count,
        "pcagente": args.pcagente_count, "replay": args.replay_count,
    }
    production = is_production_build(args, counts)

    # production preflight: pinned revisions must be resolved, and the build
    # must be attributable to an exact builder revision
    if production:
        try:
            unresolved = unresolved_sources()
            if unresolved:
                raise SystemExit(
                    f"FATAL: unresolved source revisions {unresolved}; resolve the "
                    f"commit SHAs in sources/revisions.py before a production build "
                    f"(never invent a SHA).")
            builder_commit = get_builder_revision(allow_unversioned=False)
        except SystemExit:
            lock.release()
            raise
    else:
        try:
            builder_commit = get_builder_revision(allow_unversioned=True)
        except SystemExit:
            builder_commit = "UNVERSIONED-DEV"

    config = load_config(args.config)
    config["validation_pct"] = args.validation_pct

    # SECOND-STAGE HARD GATE: the 5k preproduction build is forbidden until
    # Qwen3.8's exact native interface has been frozen from official local
    # evidence. Smoke/development builds below 5k remain possible so parser and
    # quality code can be tested without model files.
    active_total = sum(v for k, v in counts.items() if k in args.sources)
    interface_meta = None
    if not args.smoke and active_total >= 5000:
        if not args.interface_manifest:
            lock.release()
            raise SystemExit(
                "FATAL: >=5,000 sample preproduction requires --interface-manifest. "
                "Run tools/freeze_qwen_interface.py on the locally downloaded "
                "Qwen3.8 model with an official hash-pinned --native-contract first.")
        try:
            from processing.native_interface import load_verified_manifest
            native_contract, interface_meta = load_verified_manifest(args.interface_manifest)
        except Exception as e:
            lock.release()
            raise SystemExit(f"FATAL: native interface gate failed: {e}")
        config["_native_interface_contract"] = native_contract
        if float(config.get("reasoning", {}).get("rate", 0.0) or 0.0) != 0.0:
            lock.release()
            raise SystemExit(
                "FATAL: native Computer Use builds require reasoning.rate=0.0. "
                "Synthetic plan text is not permitted in second-stage production supervision.")
    elif args.interface_manifest:
        # Small development builds may voluntarily test the exact contract.
        try:
            from processing.native_interface import load_verified_manifest
            native_contract, interface_meta = load_verified_manifest(args.interface_manifest)
            config["_native_interface_contract"] = native_contract
            if float(config.get("reasoning", {}).get("rate", 0.0) or 0.0) != 0.0:
                lock.release()
                raise SystemExit(
                    "FATAL: supplied native interface contract requires reasoning.rate=0.0.")
        except Exception as e:
            lock.release()
            raise SystemExit(f"FATAL: supplied interface manifest is invalid: {e}")
    candidate_counts = (dict(counts) if args.smoke else
                        selection_candidate_targets(counts, config.get("selection", {})))
    config["_final_source_targets"] = dict(counts)
    config["_candidate_source_targets"] = dict(candidate_counts)

    state = BuildState(os.path.join(root, "state"))
    if args.no_resume:
        # A true fresh build must not merge durable samples or dedup state from
        # an earlier run.  The old implementation reset counters/shards only,
        # then silently reloaded selected_samples.jsonl at finalization.
        for name in ("selected_samples.jsonl", "dedup_index.json",
                     "failures.jsonl", "image_hashes.jsonl"):
            path = os.path.join(state.state_dir, name)
            if os.path.exists(path):
                os.remove(path)
        state.progress["sources"] = {}
        state.processed_shards = {}
        state.save()

    # Durable selected rows are the source of truth for resume accounting. A
    # kill may happen between append_jsonl() and add_selected(); conversely a
    # retry may append the same row twice.  Reconcile by unique sample id.
    durable_rows = state.read_jsonl("selected_samples.jsonl")
    durable_seen = set()
    durable_counts = {k: 0 for k in candidate_counts}
    durable_replay_categories = {}
    durable_replay_ids = set()
    durable_gui360_cohorts = {"use": 0, "grounding": 0, "understanding": 0}
    durable_gui360_apps = {"use": {}, "grounding": {}, "understanding": {}}
    durable_app_counts = {k: {} for k in counts}
    durable_global_apps = {}
    for row in durable_rows:
        key = (row.get("source"), row.get("trajectory_id"), row.get("step_id"))
        if key in durable_seen:
            continue
        durable_seen.add(key)
        src = row.get("source")
        if src in durable_counts:
            durable_counts[src] += 1
        meta = row.get("metadata") or {}
        app = str(meta.get("app") or "unknown").strip().lower()
        if src in durable_app_counts:
            d = durable_app_counts[src]
            d[app] = d.get(app, 0) + 1
        durable_global_apps[app] = durable_global_apps.get(app, 0) + 1
        if src == "replay":
            tt = str(row.get("task_type", ""))
            cat = tt.removeprefix("replay_")
            durable_replay_categories[cat] = durable_replay_categories.get(cat, 0) + 1
            rid = str(row.get("trajectory_id") or "")
            if rid:
                durable_replay_ids.add(rid)
        if src == "gui360":
            tt = str(row.get("task_type") or "")
            cohort = ("use" if tt == "action" else
                      "grounding" if tt == "grounding" else
                      "understanding" if tt == "screen_understanding" else None)
            if cohort:
                durable_gui360_cohorts[cohort] += 1
                d = durable_gui360_apps[cohort]
                d[app] = d.get(app, 0) + 1
    for src, n in durable_counts.items():
        state.source_counts(src)["selected"] = n
        state.source_counts(src)["target"] = candidate_counts[src]

    from processing.remote_access import session_with_headers
    session = session_with_headers()

    decontaminator, decon_meta = build_decontaminator(args, state.state_dir, session)
    dedup = state.load_dedup_index()
    gate = ReasoningGate(rate=config.get("reasoning", {}).get("rate", TARGET_RATE))

    remaining_counts = {src: max(0, target - durable_counts.get(src, 0))
                        for src, target in candidate_counts.items()}
    config["_resume_replay_categories"] = dict(durable_replay_categories)
    config["_resume_gui360_cohorts"] = dict(durable_gui360_cohorts)
    config["_resume_gui360_app_counts"] = durable_gui360_apps
    config["_resume_app_counts"] = durable_app_counts
    ctx = BuildContext(dataset_root=root, state=state, config=config,
                       session=session, dedup=dedup,
                       decontaminator=decontaminator, reasoning_gate=gate,
                       offline=args.offline, smoke=args.smoke, quota=remaining_counts,
                       app_counter=dict(durable_global_apps),
                       total_samples=sum(durable_global_apps.values()),
                       seen_replay_ids=set(durable_replay_ids))

    config_snapshot = build_config_snapshot(args, config, counts)
    config_snapshot["candidate_source_targets"] = dict(candidate_counts)
    config_snapshot["frontier_scores_path"] = os.path.abspath(args.frontier_scores) if args.frontier_scores else None
    config_snapshot["native_interface"] = interface_meta
    cfg_hash = config_hash_of(config_snapshot)
    build_id = hashlib.sha256(
        (cfg_hash + time.strftime("%Y-%m-%dT%H:%M:%S")).encode("utf-8")).hexdigest()[:16]
    started_at = state.progress.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%S")

    all_samples = []
    source_errors: dict[str, str] = {}
    if not args.finalize_only:
        for name in args.sources:
            if ctx.remaining(name) <= 0:
                continue
            if args.offline and name != "replay":
                # replay is the only streaming source usable offline is false
                # for all of them; in offline mode we skip network sources.
                print(f"[skip] {name}: offline mode", flush=True)
                continue
            print(f"[build] {name}: target {ctx.remaining(name)}", flush=True)
            t0 = time.time()
            try:
                samples = run_source(name, ctx, args)
                all_samples.extend(samples)
                # a successful (re)run clears any error flag an earlier
                # interrupted run left on this source
                state.source_counts(name).pop("raised_error", None)
                print(f"[done] {name}: +{len(samples)} samples "
                      f"({time.time() - t0:.1f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[fail] {name}: {e}", flush=True)
                # a raised source can never count as accepted, even when enough
                # samples were selected before the error (audit §9)
                source_errors[name] = str(e)[:200]
                state.source_counts(name)["raised_error"] = str(e)[:200]
                state.append_jsonl("failures.jsonl", [{
                    "source": name, "error": str(e), "traceback": tb,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }])
            finally:
                state.save_dedup_index(dedup)
                state.save()

    # reload previously selected samples when resuming/finalizing; sources
    # persist each unit's samples at their checkpoint (kill-safe resume), so
    # prior is the durable record — dedup it against itself and against this
    # process's in-memory samples.
    prior = state.read_jsonl("selected_samples.jsonl")
    seen_ids = set()
    merged = []
    for s in prior:
        key = (s["source"], s["trajectory_id"], s["step_id"])
        if key not in seen_ids:
            merged.append(s)
            seen_ids.add(key)
    for s in all_samples:
        key = (s["source"], s["trajectory_id"], s["step_id"])
        if key not in seen_ids:
            merged.append(s)
            seen_ids.add(key)

    frontier_scores = load_frontier_scores(args.frontier_scores)
    selected, selection_report = select_best_valid(
        merged, counts, config=config.get("selection", {}),
        frontier_scores=frontier_scores, gui360_grounding=args.gui360_grounding,
        gui360_understanding=2650, coverage_floors=config.get("coverage_floors", {}))
    selected = assign_splits(selected, args.validation_pct)
    state.atomic_write_json(os.path.join(root, "final", "selection_report.json"), selection_report)

    decon_report = decontaminator.report() if decontaminator else decon_meta
    dedup_stats = dict(dedup.stats)

    targets = {k: v for k, v in counts.items() if k in args.sources or args.finalize_only}
    # finalize-only: honor error flags recorded by the run that was interrupted
    for name, sc in state.progress.get("sources", {}).items():
        if sc.get("raised_error") and name not in source_errors:
            source_errors[name] = str(sc["raised_error"])
    build_identity = {
        "build_id": build_id,
        "builder_commit": builder_commit,
        "config_hash": cfg_hash,
        "config_snapshot": config_snapshot,
        "environment": environment_snapshot(),
        "source_revisions": revisions_manifest_view(),
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selection_policy": config_snapshot["selection_policy"],
        "selection_report": selection_report,
        "decontamination": config_snapshot["decontamination"],
        "state_corruption": state.corruption_summary(),
        "failures_rows": state.read_jsonl("failures.jsonl"),
    }
    stats = finalize(root, selected, validation_pct_marker=args.validation_pct,
                     decontamination_report=decon_report, dedup_stats=dedup_stats,
                     targets=targets, source_errors=source_errors,
                     tolerances=config.get("validation_tolerances", {}),
                     build_identity=build_identity, state_dir=state.state_dir,
                     release_gates=({} if args.smoke else {
                         "coverage_floors": config.get("coverage_floors", {}),
                         "loss_token_gate": config.get("loss_token_gate", {})}))

    if not args.skip_quality_audit:
        audit = quality_audit(selected, root, n=args.quality_audit)
        if audit:
            state.atomic_write_json(os.path.join(root, "quality_audit.json"), audit)

    # per-source manifests
    for name in SOURCE_ORDER:
        info = dict(SOURCE_INFO[name])
        info.update({
            "selected": stats.get("samples_per_source", {}).get(name, 0),
            "target": counts.get(name),
            "revisions": revisions_manifest_view().get(name,
                    revisions_manifest_view().get(SOURCE_INFO[name].get("repo", ""), {})),
            "rejections": state.source_counts(name).get("rejected_by_reason", {}),
            "notes": state.source_counts(name).get("notes", {}),
        })
        state.atomic_write_json(os.path.join(root, "manifests", f"{name}.json"), info)

    state.save_dedup_index(dedup)
    state.save()
    lock.release()

    quota_failed = (production and not args.no_enforce_quotas
                    and stats.get("quota_acceptance_passed") is False)
    if quota_failed:
        rejected = {k: v for k, v in (stats.get("quota_acceptance") or {}).items()
                    if not v.get("accepted")}
        print(f"[QUOTA ACCEPTANCE FAILED] {json.dumps(rejected, indent=1)}",
              flush=True)
    print(json.dumps({
        "total": stats.get("total_samples"),
        "train": stats.get("train_samples"),
        "validation": stats.get("validation_samples"),
        "per_source": stats.get("samples_per_source"),
        "fatal_failure": stats.get("fatal_failure"),
        "quota_acceptance_passed": stats.get("quota_acceptance_passed"),
        "config_hash": cfg_hash,
        "builder_commit": builder_commit,
        "images_tree_hash": stats.get("images_tree_hash"),
        "failures": stats.get("failures"),
    }, indent=1))
    return 1 if (stats.get("fatal_failure") or quota_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
