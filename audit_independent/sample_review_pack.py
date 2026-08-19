#!/usr/bin/env python3
"""Stratified manual-review sampler for the final JxAgent dataset (audit Phases 2-3).

Draws a review pack from final/train.jsonl + final/validation.jsonl:
  - per-source quotas (adjustable): 250 procua / 150 gui360 / 150 videocua /
    100 groundcua / 100 pcagente / 150 replay
  - oversamples suspicious categories: finish, wait, recovery, no_state_change,
    save/export, dialogs, multi-app, sorting/ranking, exact quantity,
    multi-target, tiny grounding targets, long windows, reasoning
  - emits review_pack_<ts>.json + a flat checklist for manual grading
    (EXCELLENT / GOOD / QUESTIONABLE / BAD + reason per sample)

Read-only with respect to the dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

SOURCE_QUOTAS = {"procua": 250, "gui360": 150, "videocua": 150,
                 "groundcua": 100, "pcagente": 100, "replay": 150}

SUSPICIOUS_QUOTA = 400   # extra samples drawn from suspicious categories

SUSPICIOUS_MATCHERS = {
    "finish": lambda s, a, m: a and a.startswith("finish"),
    "wait": lambda s, a, m: a and a.startswith("wait"),
    "recovery": lambda s, a, m: "recovery" in (m.get("signals") or []),
    "no_state_change": lambda s, a, m: "no_state_change" in (m.get("signals") or []),
    "save_export": lambda s, a, m: "save" in (m.get("signals") or []) or "export" in (m.get("signals") or []),
    "dialog": lambda s, a, m: any("dialog" in str(x) for x in (m.get("signals") or [])),
    "multi_app": lambda s, a, m: "multi_app" in (m.get("signals") or []),
    "sorting_ranking": lambda s, a, m: bool({"sorting", "ranking"} & set(m.get("signals") or [])),
    "exact_quantity": lambda s, a, m: "exact_quantity" in (m.get("signals") or []),
    "multi_target": lambda s, a, m: "multi_target" in (m.get("signals") or []),
    "tiny_target": lambda s, a, m: (m.get("target_width_px") or 999) < 16,
    "long_window": lambda s, a, m: (m.get("window_length") or 1) >= 8,
    "reasoning": lambda s, a, m: "reasoning_category" in m and m.get("reasoning_category"),
}


def last_assistant(sample):
    for m in reversed(sample.get("messages", [])):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


_ACTION_LINE = re.compile(r"^Action:\s*(.+)$", re.MULTILINE)


def action_of(sample):
    c = last_assistant(sample).strip()
    if not c:
        return ""
    m = None
    for m in _ACTION_LINE.finditer(c):
        pass
    if m:
        return m.group(1).strip()
    lines = [l for l in c.splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def task_of(sample):
    for m in sample.get("messages", []):
        if m.get("role") == "user":
            mm = re.search(r"Task:\s*(.*)", str(m.get("content", "")), re.DOTALL)
            if mm:
                return mm.group(1).strip()[:200]
            c = str(m.get("content", "")).replace("<image>", "").strip()
            return c[:200]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for name in ("train", "validation"):
        p = os.path.join(args.dataset_root, "final", f"{name}.jsonl")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                rows.extend(json.loads(l) for l in f if l.strip())

    by_source = defaultdict(list)
    suspicious = defaultdict(list)
    for s in rows:
        src = s.get("source", "?")
        meta = s.get("metadata", {}) or {}
        a = action_of(s)
        by_source[src].append(s)
        for cat, match in SUSPICIOUS_MATCHERS.items():
            try:
                if match(s, a, meta):
                    suspicious[cat].append(s)
            except Exception:
                pass

    pack = []
    seen = set()

    def key(s):
        return (s.get("source"), s.get("trajectory_id"), s.get("step_id"))

    def take(pool, n, tag):
        pool = [s for s in pool if key(s) not in seen]
        for s in rng.sample(pool, min(n, len(pool))):
            seen.add(key(s))
            pack.append((tag, s))

    # per-source stratified draws (proportional across task types within source)
    for src, quota in SOURCE_QUOTAS.items():
        pool = by_source.get(src, [])
        take(pool, quota, f"source:{src}")

    # suspicious oversampling, spread across categories
    per_cat = max(1, SUSPICIOUS_QUOTA // max(1, len(suspicious)))
    for cat, pool in suspicious.items():
        take(pool, per_cat, f"suspicious:{cat}")

    out_rows = []
    for tag, s in pack:
        out_rows.append({
            "review_tag": tag,
            "source": s.get("source"),
            "trajectory_id": s.get("trajectory_id"),
            "step_id": s.get("step_id"),
            "task_type": s.get("task_type"),
            "representation": (s.get("metadata", {}) or {}).get("representation"),
            "app": (s.get("metadata", {}) or {}).get("app"),
            "signals": (s.get("metadata", {}) or {}).get("signals"),
            "action": action_of(s),
            "task": task_of(s),
            "images": s.get("images"),
            "full_sample": s,
            "grade": "",       # EXCELLENT / GOOD / QUESTIONABLE / BAD
            "reason": "",
        })

    ts = time.strftime("%Y%m%d_%H%M%S")
    root_tag = os.path.basename(os.path.normpath(args.dataset_root))
    jpath = os.path.join(args.out_dir, f"review_pack_{root_tag}_{ts}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "dataset_root": os.path.abspath(args.dataset_root),
                   "counts_by_tag": dict(Counter(r["review_tag"] for r in out_rows)),
                   "samples": out_rows}, f, ensure_ascii=False, indent=1)
    cpath = os.path.join(args.out_dir, f"review_checklist_{root_tag}_{ts}.md")
    with open(cpath, "w", encoding="utf-8") as f:
        f.write("# Manual review checklist\n\n")
        f.write("For each item: open the image(s), read task + action, verify the action\n")
        f.write("makes sense ON THAT SCREEN (target exists, coordinates plausible,\n")
        f.write("typing matches the task, scroll direction sensible, wait/finish\n")
        f.write("justified, history coherent). Grade EXCELLENT/GOOD/QUESTIONABLE/BAD + reason.\n\n")
        f.write("| # | tag | source | app | action | task (first 90 chars) |\n")
        f.write("|---|-----|--------|-----|--------|----------------------|\n")
        for i, r in enumerate(out_rows):
            f.write(f"| {i} | {r['review_tag']} | {r['source']} | {r['app']} | "
                    f"`{r['action'][:60]}` | {str(r['task'])[:90].replace('|', '/')} |\n")
    print(f"[review-pack] {len(out_rows)} samples -> {jpath}")
    print(f"[review-pack] checklist -> {cpath}")
    print("[review-pack] tag counts:", dict(Counter(r["review_tag"] for r in out_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
