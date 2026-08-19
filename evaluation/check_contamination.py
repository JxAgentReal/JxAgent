#!/usr/bin/env python3
"""Contamination check of a prepared dataset against the EXACT pinned
evaluation task set (original OSWorld or OSWorld Verified).

Unlike the build-time decontaminator (which REMOVES matches from training
data), this evaluation-side tool only FLAGS: nothing is dropped, nothing is
rebuilt. Output is a machine-readable report with per-match detail:

  exact / near (8-gram Jaccard) / containment counts, per-match records with
  source, sample id, benchmark task id, scores and matched fragments.

Usage:
  python evaluation/check_contamination.py \
      --dataset-root ./JxAgentData_Run1_Final \
      --task-list evaluation/benchmark_task_lists/osworld_verified.json \
      --benchmark osworld_verified \
      --out evaluation/contamination_report_verified.json

The report feeds run_agent.py --contamination-report; runs without a clean
report carry a contamination blocker in every claim evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.decontamination import (containment, jaccard, normalize,  # noqa: E402
                                        shingles)

NGRAM = 8
SIMILARITY_THRESHOLD = 0.5
TASK_LINE_RE = None  # compiled lazily


def _task_text_of_sample(sample: dict) -> Optional[str]:
    """Extract the computer-use task text from an assembled final sample
    (user turn 'Task: ...' line). Replay/understanding samples return None."""
    tt = sample.get("task_type", "")
    if tt.startswith("replay") or tt == "screen_understanding":
        return None
    for m in sample.get("messages", []):
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(p) for p in content)
        for line in content.splitlines():
            if line.startswith("Task:"):
                return line[len("Task:"):].strip()
    return None


def _sample_id(sample: dict) -> str:
    meta = sample.get("metadata") or {}
    return str(meta.get("id") or meta.get("sample_id")
               or hashlib.sha256(json.dumps(sample.get("messages", ""),
                                            sort_keys=True).encode()).hexdigest()[:16])


def build_reference(task_list: List[dict]) -> List[dict]:
    refs = []
    for t in task_list:
        instr = t.get("instruction") or ""
        if not instr:
            continue
        refs.append({"task_id": t["task_id"], "instruction": instr,
                     "norm_hash": hashlib.sha256(
                         normalize(instr).encode("utf-8")).hexdigest(),
                     "shingles": shingles(instr),
                     "words": set(normalize(instr).split())})
    return refs


def check_text(text: str, refs: List[dict],
               threshold: float = SIMILARITY_THRESHOLD) -> Optional[dict]:
    """Best match against the reference set, or None below threshold."""
    norm = normalize(text)
    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    for r in refs:
        if r["norm_hash"] == h:
            return {"kind": "exact", "task_id": r["task_id"], "score": 1.0,
                    "fragments": []}
    cand = shingles(text)
    if not cand:
        return None
    words = set(norm.split())
    short = len(words) < NGRAM
    best = None
    for r in refs:
        j = jaccard(cand, r["shingles"])
        c = (len(words & r["words"]) / max(1, len(words)) if short
             else containment(cand, r["shingles"]))
        score = max(j, c)
        if score >= threshold and (best is None or score > best["score"]):
            matched = sorted(cand & r["shingles"])[:5]
            best = {"kind": ("near_jaccard" if j >= c else "containment"),
                    "task_id": r["task_id"], "score": round(score, 4),
                    "fragments": matched}
    return best


def iter_dataset_samples(dataset_root: str):
    for split in ("train", "validation"):
        path = os.path.join(dataset_root, "final", f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield split, json.loads(line)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--task-list", required=True,
                   help="pinned benchmark task list JSON (task_id/instruction)")
    p.add_argument("--benchmark", default="osworld_verified")
    p.add_argument("--out", default=None)
    p.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = p.parse_args(argv)

    with open(args.task_list, "r", encoding="utf-8") as f:
        task_list = json.load(f)
    refs = build_reference(task_list)

    matches: List[dict] = []
    counts = {"scanned": 0, "skipped_non_cu": 0, "exact": 0, "near": 0,
              "containment": 0}
    for split, sample in iter_dataset_samples(args.dataset_root):
        text = _task_text_of_sample(sample)
        if text is None:
            counts["skipped_non_cu"] += 1
            continue
        counts["scanned"] += 1
        hit = check_text(text, refs, threshold=args.threshold)
        if hit:
            counts[hit["kind"] if hit["kind"] in counts else
                   ("near" if hit["kind"] == "near_jaccard" else "containment")] += 1
            matches.append({
                "split": split, "sample_id": _sample_id(sample),
                "source": (sample.get("metadata") or {}).get("source", "unknown"),
                "task_text_sha256": hashlib.sha256(
                    normalize(text).encode("utf-8")).hexdigest(),
                "benchmark": args.benchmark,
                "benchmark_task_id": hit["task_id"],
                "match_kind": hit["kind"], "score": hit["score"],
                "fragments": hit["fragments"],
                "action": "FLAGGED_NOT_REMOVED",
            })

    task_list_sha = hashlib.sha256(
        open(args.task_list, "rb").read()).hexdigest()
    report = {
        "status": "clean" if not matches else "flagged",
        "benchmark": args.benchmark,
        "task_list_path": os.path.abspath(args.task_list),
        "task_list_sha256": task_list_sha,
        "reference_instructions": len(refs),
        "ngram": NGRAM,
        "similarity_threshold": args.threshold,
        "counts": counts,
        "flagged_match_count": len(matches),
        "matches": matches,
        "blockers": ([] if not matches else
                     [{"reason": f"{len(matches)} potential contamination "
                                 f"matches flagged; resolve (rebuild dataset "
                                 f"or justify) before comparative claims"}]),
        "note": "flag-only tool; nothing is removed from training data here",
    }
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"contamination_report_{args.benchmark}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print(json.dumps({k: report[k] for k in
                      ("status", "counts", "flagged_match_count", "blockers")}, indent=1))
    print(f"[contamination] report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
