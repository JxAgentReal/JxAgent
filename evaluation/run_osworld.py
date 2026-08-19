#!/usr/bin/env python3
"""OSWorld evaluation entry point.

Default and only safe mode today: --dry-run (synthetic tasks, scripted
model, full harness path). Real benchmark runs are doubly gated:
  1. --confirm-real-benchmark must be passed explicitly, AND
  2. every benchmark protocol field must be pinned (no
     REQUIRES_EXTERNAL_VERIFICATION markers), AND
  3. the OSWorld environment backend must be importable on the host.
This script never starts VMs or servers itself.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import benchmarks as bm  # noqa: E402


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default="osworld_verified",
                   choices=bm.list_benchmarks())
    p.add_argument("--dry-run", action="store_true",
                   help="offline synthetic run (no OSWorld, no model)")
    p.add_argument("--confirm-real-benchmark", action="store_true")
    p.add_argument("--arm", choices=["base", "adapter"], required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--frozen-subset",
                   default=os.path.join(here, "osworld_verified_frozen_subset.json"))
    p.add_argument("--task-list")
    p.add_argument("--base-run-dir")
    p.add_argument("--allow-without-baseline", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--tasks", type=int, default=12)
    p.add_argument("--adapter-path")
    p.add_argument("--checkpoint-gate", type=int, choices=[20, 55, 100])
    p.add_argument("--train-output-dir")
    p.add_argument("--total-steps", type=int)
    p.add_argument("--model-revision")
    p.add_argument("--contamination-report")
    p.add_argument("--model-backend", choices=["scripted", "openai"],
                   default="openai")
    p.add_argument("--script-file")
    args, passthrough = p.parse_known_args(argv)

    if not args.dry_run and not args.confirm_real_benchmark:
        print("[run_osworld] REFUSING real benchmark run.\n"
              "  Real runs are gated: pass --confirm-real-benchmark AND make sure\n"
              "  the benchmark identity is fully pinned (no "
              "REQUIRES_EXTERNAL_VERIFICATION fields),\n  and run on the "
              "evaluation host with the OSWorld environment provisioned.\n"
              "  For harness validation use --dry-run.", file=sys.stderr)
        return 2

    from evaluation import run_agent
    argv2 = [
        "--arm", args.arm,
        "--benchmark", args.benchmark,
        "--output-dir", args.output_dir,
        "--frozen-subset", args.frozen_subset,
        "--model-backend", args.model_backend,
    ]
    if args.dry_run:
        argv2.append("--dry-run")
        argv2 += ["--tasks", str(args.tasks)]
    if args.confirm_real_benchmark:
        argv2.append("--confirm-real-benchmark")
    if args.task_list:
        argv2 += ["--task-list", args.task_list]
    if args.base_run_dir:
        argv2 += ["--base-run-dir", args.base_run_dir]
    if args.allow_without_baseline:
        argv2.append("--allow-without-baseline")
    if args.force_rerun:
        argv2.append("--force-rerun")
    if args.adapter_path:
        argv2 += ["--adapter-path", args.adapter_path]
    if args.checkpoint_gate:
        argv2 += ["--checkpoint-gate", str(args.checkpoint_gate)]
    if args.train_output_dir:
        argv2 += ["--train-output-dir", args.train_output_dir]
    if args.total_steps:
        argv2 += ["--total-steps", str(args.total_steps)]
    if args.model_revision:
        argv2 += ["--model-revision", args.model_revision]
    if args.contamination_report:
        argv2 += ["--contamination-report", args.contamination_report]
    if args.script_file:
        argv2 += ["--script-file", args.script_file]
    argv2 += passthrough
    return run_agent.main(argv2)


if __name__ == "__main__":
    raise SystemExit(main())
