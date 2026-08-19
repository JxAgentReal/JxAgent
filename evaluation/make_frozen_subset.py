#!/usr/bin/env python3
"""Deterministic frozen-evaluation-subset generation.

Policy:
  - the subset is chosen from benchmark metadata ONLY (domain + task id),
    never from model performance;
  - selection is a pure function of (task list content, seed, size): the same
    inputs always regenerate the same subset;
  - an existing subset file is never silently regenerated with a different
    task list -- regeneration requires --force (the previous file is kept as
    a timestamped backup);
  - when the canonical task list is unavailable (OSWorld Verified today) the
    artifact is written as an explicit PENDING marker that every consumer
    rejects as "not ready".

Example:
  python evaluation/make_frozen_subset.py --benchmark osworld_verified \
      --task-list evaluation/benchmark_task_lists/osworld_verified.json \
      --size 100 --seed 1337 \
      --output evaluation/osworld_verified_frozen_subset.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks import (load_task_list, hash_task_ids)  # noqa: E402

PENDING_STATUS = "PENDING_VERIFIED_TASK_LIST"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def allocate_seats(domain_counts: Dict[str, int], size: int) -> Dict[str, int]:
    """Largest-remainder proportional allocation across domains."""
    total = sum(domain_counts.values())
    seats: Dict[str, int] = {}
    remainders: List[tuple] = []
    used = 0
    for domain, count in domain_counts.items():
        exact = size * count / float(total)
        base = int(exact)
        seats[domain] = base
        used += base
        remainders.append((exact - base, domain))
    leftover = size - used
    for _, domain in sorted(remainders, key=lambda t: (-t[0], t[1])):
        if leftover <= 0:
            break
        seats[domain] += 1
        leftover -= 1
    return seats


def select_subset(tasks: List[dict], size: int, seed: int,
                  source_sha256: str) -> List[dict]:
    by_domain: Dict[str, List[str]] = {}
    for t in tasks:
        by_domain.setdefault(t["domain"] or "unknown", []).append(t["task_id"])
    for domain in by_domain:
        by_domain[domain].sort()
    seats = allocate_seats({d: len(v) for d, v in by_domain.items()}, size)
    selected: List[str] = []
    for domain in sorted(by_domain):
        pool = by_domain[domain]
        k = min(seats[domain], len(pool))
        rng = random.Random(f"{seed}:{source_sha256}:{domain}")
        # sorted copy so pool order never matters, then deterministic shuffle
        chosen = sorted(rng.sample(sorted(pool), k)) if k else []
        selected.extend(chosen)
    return [t for t in tasks if t["task_id"] in set(selected)]


def build_subset(benchmark: str, task_list_path: str, size: int, seed: int) -> dict:
    tasks = load_task_list(task_list_path)
    source_sha = sha256_file(task_list_path)
    subset = select_subset(tasks, size, seed, source_sha)
    if not subset:
        raise SystemExit("subset selection produced no tasks")
    domain_counts: Dict[str, int] = {}
    for t in subset:
        domain_counts[t["domain"]] = domain_counts.get(t["domain"], 0) + 1
    task_ids = sorted(t["task_id"] for t in subset)
    return {
        "status": "ready",
        "benchmark": benchmark,
        "created_utc": utc_now(),
        "source_task_list": os.path.abspath(task_list_path),
        "source_task_list_sha256": source_sha,
        "source_task_count": len(tasks),
        "selection_seed": seed,
        "size": len(task_ids),
        "requested_size": size,
        "algorithm": "domain_proportional_largest_remainder+seeded_sample_v1",
        "domain_counts": domain_counts,
        "task_ids": task_ids,
        "subset_sha256": hash_task_ids(task_ids),
    }


def build_pending(benchmark: str, reason: str) -> dict:
    return {
        "status": PENDING_STATUS,
        "benchmark": benchmark,
        "created_utc": utc_now(),
        "reason": reason,
        "note": ("Generate the real subset with make_frozen_subset.py once the "
                 "canonical task list is pinned and hashed. Consumers must "
                 "treat this file as NOT READY."),
    }


def load_existing(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default="osworld_verified")
    p.add_argument("--task-list", help="canonical pinned task list JSON")
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "osworld_verified_frozen_subset.json"))
    p.add_argument("--force", action="store_true",
                   help="explicitly regenerate even if a different subset exists")
    p.add_argument("--create-pending", action="store_true",
                   help="write/refresh the PENDING marker (no task list needed)")
    args = p.parse_args(argv)

    if args.create_pending:
        subset = build_pending(args.benchmark, "canonical Verified task list not yet pinned")
    elif args.task_list:
        subset = build_subset(args.benchmark, args.task_list, args.size, args.seed)
    else:
        print("[frozen-subset] no --task-list given and --create-pending not set;\n"
              "              cannot create the real subset without the canonical "
              "task list.", file=sys.stderr)
        return 2

    if os.path.exists(args.output):
        existing = load_existing(args.output)
        same_request = (
            existing.get("status") == subset.get("status")
            and existing.get("benchmark") == subset["benchmark"]
            and existing.get("selection_seed") == subset.get("selection_seed")
            and existing.get("source_task_list_sha256") == subset.get("source_task_list_sha256")
            and existing.get("requested_size") == subset.get("requested_size"))
        if same_request and existing.get("status") == "ready":
            same_ids = existing.get("task_ids") == subset["task_ids"]
            if same_ids:
                print(f"[frozen-subset] identical subset already exists: {args.output}")
                return 0
        if not args.force:
            print(f"[frozen-subset] REFUSING to overwrite {args.output}\n"
                  "  existing subset differs from the requested one.\n"
                  "  Regeneration is intentional-only: re-run with --force.",
                  file=sys.stderr)
            return 3
        backup = f"{args.output}.bak-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.replace(args.output, backup)
        print(f"[frozen-subset] previous subset backed up to {backup}")

    tmp = args.output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=1, sort_keys=True)
    os.replace(tmp, args.output)
    print(f"[frozen-subset] wrote {args.output} (status={subset['status']}"
          + (f", size={subset['size']}, sha256={subset['subset_sha256'][:16]}…)"
             if subset["status"] == "ready" else ")"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
