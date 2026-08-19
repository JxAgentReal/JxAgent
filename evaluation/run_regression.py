#!/usr/bin/env python3
"""Regression panel runner (instruction following / coding / math / VQA /
tool use), paired base-vs-adapter with the same thresholds as the config.

Dry-run mode evaluates scripted per-sample outcomes offline to validate
comparison logic, persistence and thresholds. Real mode delegates each panel
to its provider (lm-eval-harness or equivalent) on the evaluation host --
providers are intentionally NOT executed or downloaded here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "regression_config.yaml")


class RegressionConfigError(RuntimeError):
    pass


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tol = cfg.get("tolerance") or {}
    for key in ("investigation_points", "hard_stop_points", "noise_floor_points"):
        if key not in tol:
            raise RegressionConfigError(f"regression config missing tolerance.{key}")
    required_panels = {"instruction_following", "coding", "math", "general_vqa", "tool_use"}
    missing = required_panels - set((cfg.get("panels") or {}).keys())
    if missing:
        raise RegressionConfigError(f"regression config missing panels: {sorted(missing)}")
    return cfg


def evaluate_thresholds(base_rate: float, adapter_rate: float, cfg: dict) -> dict:
    tol = cfg["tolerance"]
    delta_pp = 100.0 * (adapter_rate - base_rate)
    regression_pp = -delta_pp
    if regression_pp >= tol["hard_stop_points"]:
        verdict = "HARD_STOP"
    elif regression_pp >= tol["investigation_points"]:
        verdict = "INVESTIGATE"
    elif regression_pp >= tol["noise_floor_points"]:
        verdict = "WATCH"           # above noise floor but under investigation threshold
    else:
        verdict = "OK"
    return {"base_rate": base_rate, "adapter_rate": adapter_rate,
            "delta_pp": round(delta_pp, 2), "regression_pp": round(regression_pp, 2),
            "verdict": verdict}


def dry_run_panel(name: str, samples: int, seed: int) -> List[bool]:
    """Deterministic synthetic per-sample outcomes (True = correct)."""
    rng = random.Random(f"{seed}:{name}")
    p = {"instruction_following": 0.82, "coding": 0.55, "math": 0.60,
         "general_vqa": 0.74, "tool_use": 0.66}.get(name, 0.7)
    return [rng.random() < p for _ in range(samples)]


def compare_panels(base: Dict[str, List[bool]], adapter: Dict[str, List[bool]],
                   cfg: dict) -> dict:
    report = {"panels": {}, "tolerance": cfg["tolerance"]}
    worst = "OK"
    for name in sorted(base):
        b, a = base[name], adapter[name]
        if len(b) != len(a):
            raise RegressionConfigError(
                f"panel {name}: paired comparison requires equal sample counts "
                f"(base {len(b)} vs adapter {len(a)})")
        res = evaluate_thresholds(sum(b) / len(b) if b else 0.0,
                                  sum(a) / len(a) if a else 0.0, cfg)
        overlap = bool((cfg["panels"].get(name) or {}).get("overlap_exposed_tasks"))
        res["overlap_exposed"] = overlap
        if overlap:
            res["note"] = ("training replay contains GSM-style data; treat "
                           "with caution (labeled overlap-exposed)")
        report["panels"][name] = res
        order = {"OK": 0, "WATCH": 1, "INVESTIGATE": 2, "HARD_STOP": 3}
        if order[res["verdict"]] > order[worst]:
            worst = res["verdict"]
    report["overall_verdict"] = worst
    report["regression_detected"] = worst in ("INVESTIGATE", "HARD_STOP")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--base-results", help="JSON {panel: [bool,...]} from a base run")
    p.add_argument("--adapter-results")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    cfg = load_config(args.config)

    if args.dry_run:
        base = {n: dry_run_panel(n, (cfg["panels"][n].get("samples") or 100), args.seed)
                for n in cfg["panels"]}
        adapter = {n: dry_run_panel(n, len(base[n]), args.seed + 1)
                   for n in cfg["panels"]}
    else:
        if not (args.base_results and args.adapter_results):
            print("[regression] real mode requires --base-results and "
                  "--adapter-results (or run with --dry-run). Providers "
                  "(lm-eval-harness etc.) run on the evaluation host, not here.",
                  file=sys.stderr)
            return 2
        if not os.path.exists(args.base_results):
            print("[regression] baseline-first: base results file missing - "
                  "run the base arm first through the same panel",
                  file=sys.stderr)
            return 2
        with open(args.base_results, "r", encoding="utf-8") as f:
            base = json.load(f)
        with open(args.adapter_results, "r", encoding="utf-8") as f:
            adapter = json.load(f)

    report = compare_panels(base, adapter, cfg)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "regression_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if not report["regression_detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
