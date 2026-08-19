#!/usr/bin/env python3
"""Export objectively verifiable JxAgent candidates for base-model frontier scoring.

This tool never calls a model. It produces a bounded JSONL worklist containing
only targets whose correctness can be checked objectively. A separate scorer
may add frontier_score in [0,1] and feed the result back to
build_jxagent_dataset.py --frontier-scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sid(s: dict) -> str:
    return f"{s.get('source','')}::{s.get('trajectory_id','')}::{s.get('step_id','')}"


def last_assistant(s: dict) -> str:
    for m in reversed(s.get("messages") or []):
        if m.get("role") == "assistant":
            return str(m.get("content") or "")
    return ""


def evidence(s: dict):
    meta = s.get("metadata") or {}
    task_type = str(s.get("task_type") or "")
    source = str(s.get("source") or "")
    if task_type == "grounding":
        return "grounding_target_with_known_point"
    if source == "pcagente" and task_type == "action" and meta.get("bbox_click_validated") is True:
        return "pc_agent_e_click_inside_trusted_bbox"
    if task_type == "replay_tool":
        return "structured_tool_arguments"
    if meta.get("finish_evidence") == "yes":
        return "verified_finish_outcome"
    if meta.get("math_verified") is True:
        return "verified_math_reference"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="dataset root or selected_samples.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-samples", type=int, default=30000)
    a = ap.parse_args()
    src = Path(a.dataset)
    if src.is_dir():
        candidates = [src / "state" / "selected_samples.jsonl",
                      src / "final" / "train.jsonl"]
        src = next((p for p in candidates if p.is_file()), candidates[0])
    if not src.is_file():
        raise SystemExit(f"candidate JSONL not found: {src}")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    by_kind = {}
    with out.open("w", encoding="utf-8") as f:
        for s in rows(src):
            ev = evidence(s)
            if not ev:
                continue
            row = {
                "sample_id": sid(s),
                "source": s.get("source"),
                "task_type": s.get("task_type"),
                "verifiable": True,
                "verification_basis": ev,
                "target": last_assistant(s),
                "frontier_score": None,
                "scoring_semantics": "0=base_already_easy, 1=base_uncertain_or_wrong",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            by_kind[ev] = by_kind.get(ev, 0) + 1
            if n >= max(0, a.max_samples):
                break
    print(json.dumps({"output": str(out), "samples": n,
                      "verification_basis": by_kind}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
