#!/usr/bin/env python3
"""Action syntax / coordinate validation for JxAgent datasets and model outputs.

Modes:
  --dataset-root DIR --full   validate every sample in final/*.jsonl
  --file predictions.jsonl    validate model outputs (messages or bare text)

Checks: unified action grammar, coordinate bounds vs metadata image size,
finish/point semantics for grounding, empty targets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.coordinates import action_in_bounds
from processing.validation import extract_action_text

# Canonical parser: the SAME one the runtime harness uses (audit H1 fix).
from evaluation.action_parser import parse_model_output

VERBS = {"click", "double_click", "right_click", "middle_click", "move", "point",
         "drag", "scroll", "type", "press", "hotkey", "key_down", "key_up",
         "mouse_down", "mouse_up", "wait", "finish"}


def check_text(text: str, final_size=None):
    """Returns list of problems for one model output / action text.

    Uses the canonical Plan/Action parser: bare actions and
    'Plan: ...\\nAction: ...' both parse; malformed envelopes
    (missing/multiple Action blocks, trailing content) are errors.
    """
    problems = []
    parsed = parse_model_output(text)
    if not parsed.ok:
        return [parsed.error or "unparseable_action"]
    if parsed.action.verb not in VERBS:
        problems.append(f"bad_verb:{parsed.action.verb}")
    if final_size and parsed.action.points:
        w, h = final_size
        if not action_in_bounds(parsed.action, int(w), int(h)):
            problems.append(f"out_of_bounds:{parsed.action.points}")
    return problems


def iter_dataset_samples(root: str):
    for split in ("train", "validation"):
        path = os.path.join(root, "final", f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield split, json.loads(line)


def validate_dataset(root: str, full: bool = False, limit: int = 0):
    counts = {}
    problems = {}
    n = 0
    for split, sample in iter_dataset_samples(root):
        n += 1
        if limit and n > limit:
            break
        if sample.get("task_type", "").startswith("replay") or \
           sample.get("task_type") == "screen_understanding":
            continue
        size = (sample.get("metadata", {}) or {}).get("final_image_size")
        for problem in check_text(extract_action_text(sample), size):
            counts[problem] = counts.get(problem, 0) + 1
        if full:
            for img in sample.get("images", []):
                full_path = os.path.join(root, *img.split("/"))
                if not os.path.exists(full_path):
                    key = "missing_image"
                    problems.setdefault(key, []).append(img)
    return n, counts, problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root")
    p.add_argument("--full", action="store_true")
    p.add_argument("--file", help="jsonl with model outputs")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    if args.file:
        bad = 0
        total = 0
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("action") or obj.get("output") or ""
                total += 1
                probs = check_text(text)
                if probs:
                    bad += 1
                    if bad <= 10:
                        print("INVALID:", probs, text[:120])
        print(f"validated {total} outputs, invalid: {bad}")
        return 1 if bad else 0

    if not args.dataset_root:
        p.error("need --dataset-root or --file")

    n, counts, problems = validate_dataset(args.dataset_root, args.full, args.limit)
    print(f"validated {n} samples")
    if counts:
        print("action problems:", json.dumps(counts, indent=1))
    if problems:
        print("structural problems:", {k: len(v) for k, v in problems.items()})
    fatal = {k: v for k, v in counts.items()
             if k.startswith("out_of_bounds") or k in ("empty_action", "unparseable_action")}
    return 1 if fatal or problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
