#!/usr/bin/env python3
"""Timestamped tap for swift/HF Trainer logs + steady-state throughput stats.

tap mode (piped between swift and the console):
    PYTHONUNBUFFERED=1 NPROC_PER_NODE=N swift sft ... 2>&1 |
        python log_tap.py tap --raw run.log --events steps.jsonl
  Mirrors the raw stream to --raw and records the arrival time of every
  per-step Trainer log record ({'loss': ...,'step': N,...}). Because
  timestamps are taken on ARRIVAL, model download/load and dataset
  preprocessing happen BEFORE the first event and never contaminate the
  measured window.

stats mode (after the run):
    python log_tap.py stats --events steps.jsonl --warmup 10 --measure 100 \
        --world 8 --eff-batch 32 --samples 100003 --epoch-steps 3126 \
        --budget 35 --reserve 6 [--vram vram.log] [--mean-tokens 2340]
  Computes the steady-state window strictly AFTER the warmup steps:
  duration = ts(step warmup+measure) - ts(step warmup).

Pure helpers (parse_log_step, window_stats, epoch_hours, parse_vram_peaks)
are unit tested offline.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time

STEP_RE = re.compile(r"'step':\s*(\d+)")


# --------------------------------------------------------------------------
# Pure helpers (unit tested)
# --------------------------------------------------------------------------

def parse_log_step(line: str):
    """Return the step number if the line is a per-step Trainer log record."""
    if "{'loss'" not in line and "'loss':" not in line:
        return None
    m = STEP_RE.search(line)
    return int(m.group(1)) if m else None


def window_stats(events, warmup: int, measure: int) -> dict:
    """events: iterable of {'step': int, 'ts': float} (arrival order).

    Measures the window of `measure` steps that starts after `warmup`
    completed steps. Raises ValueError if the events do not cover it.
    """
    by_step = {}
    for e in events:
        s = int(e["step"])
        if s not in by_step:          # keep the FIRST arrival per step
            by_step[s] = float(e["ts"])
    need_first, need_last = warmup, warmup + measure
    if need_first not in by_step or need_last not in by_step:
        have = max(by_step) if by_step else 0
        raise ValueError(
            f"need step {need_first} and step {need_last} timestamps; "
            f"highest logged step is {have} - run crashed or logs incomplete")

    warmup_duration = by_step[need_first] - by_step[min(by_step)]
    duration = by_step[need_last] - by_step[need_first]

    deltas = [by_step[s] - by_step[s - 1]
              for s in range(need_first + 1, need_last + 1) if s in by_step and (s - 1) in by_step]
    if not deltas:
        raise ValueError("no consecutive step pairs inside the measured window")

    def pct90(xs):
        xs = sorted(xs)
        k = max(0, min(len(xs) - 1, round(0.9 * (len(xs) - 1))))
        return xs[k]

    return {
        "warmup_seconds": warmup_duration,
        "measured_seconds": duration,
        "measured_steps": measure,
        "seconds_per_step_mean": duration / measure,
        "seconds_per_step_median": statistics.median(deltas),
        "seconds_per_step_p90": pct90(deltas),
        "steps_per_second": measure / duration,
    }


def epoch_hours(epoch_steps: int, seconds_per_step: float) -> float:
    return epoch_steps * seconds_per_step / 3600.0


def parse_vram_peaks(path: str) -> dict:
    """Best-effort column maxima from `rocm-smi --csv` samples.

    Returns {header_column: max_numeric_value}. Non-CSV lines (timestamps)
    and non-numeric rows are ignored.
    """
    header = None
    maxima = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            cells = [c.strip() for c in line.strip().split(",")]
            if len(cells) < 2:
                continue
            if header is None and any(c for c in cells) and not any(
                    _is_num(c) for c in cells):
                header = cells
                continue
            if header is None or len(cells) != len(header):
                continue
            for name, cell in zip(header, cells):
                if _is_num(cell):
                    v = float(cell.rstrip("%"))
                    maxima[name] = max(maxima.get(name, v), v)
    return maxima


def _is_num(s: str) -> bool:
    try:
        float(s.rstrip("%"))
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def cmd_tap(args) -> int:
    with open(args.raw, "w", encoding="utf-8", errors="replace") as raw, \
            open(args.events, "w", encoding="utf-8") as ev:
        for line in sys.stdin:
            raw.write(line)
            raw.flush()
            s = parse_log_step(line)
            if s is not None:
                ev.write(json.dumps({"step": s, "ts": time.time()}) + "\n")
                ev.flush()
    return 0


def cmd_stats(args) -> int:
    with open(args.events, "r", encoding="utf-8") as f:
        events = [json.loads(l) for l in f if l.strip()]
    w = window_stats(events, args.warmup, args.measure)

    sps = w["steps_per_second"]
    samples_per_s = sps * args.eff_batch
    est_epoch_h = epoch_hours(args.epoch_steps, w["seconds_per_step_mean"])
    remaining = args.budget - est_epoch_h

    print("---- throughput (steady-state window) ----")
    print(f"GPU count (world size)        : {args.world}")
    print(f"global effective batch        : {args.eff_batch}")
    print(f"warmup duration               : {w['warmup_seconds']:.1f} s ({args.warmup} steps)")
    print(f"measured training duration    : {w['measured_seconds']:.1f} s ({args.measure} steps)")
    print(f"optimizer steps / second      : {sps:.4f}")
    print(f"samples / second              : {samples_per_s:.4f}")
    if args.mean_tokens:
        print(f"tokens / second (est.)        : {samples_per_s * args.mean_tokens:.1f}")
    print(f"median seconds / step         : {w['seconds_per_step_median']:.2f}")
    print(f"p90 seconds / step            : {w['seconds_per_step_p90']:.2f}")
    if args.vram and __import__("os").path.exists(args.vram):
        peaks = parse_vram_peaks(args.vram)
        for k in sorted(peaks):
            print(f"peak {k:<30}: {peaks[k]}")
    print(f"epoch optimizer steps         : {args.epoch_steps}")
    print(f"estimated epoch wall time     : {est_epoch_h:.2f} h")
    print(f"GPU budget                    : {args.budget:.1f} h")
    print(f"remaining after training      : {remaining:.2f} h (reserve {args.reserve:.1f} h)")
    if remaining < args.reserve:
        print("WARNING: training would eat into the evaluation reserve. "
              "Review before train.sh.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("tap")
    p.add_argument("--raw", required=True)
    p.add_argument("--events", required=True)
    p.set_defaults(fn=cmd_tap)

    p = sub.add_parser("stats")
    p.add_argument("--events", required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--measure", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--eff-batch", type=int, required=True)
    p.add_argument("--samples", type=int, required=True)
    p.add_argument("--epoch-steps", type=int, required=True)
    p.add_argument("--budget", type=float, default=35.0)
    p.add_argument("--reserve", type=float, default=6.0)
    p.add_argument("--vram", default=None)
    p.add_argument("--mean-tokens", type=float, default=None,
                   help="mean tokens/sample -> estimated tokens per second")
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
