#!/usr/bin/env python3
"""Offline deterministic representation-mixture simulator (no network, no
remote datasets).

Models the SAMPLE-LEVEL single/window/chunk mix produced by the builder's
emission logic for the intended Run 1 quotas:

  procua   46,000   trajectory-rich, new emission logic
  videocua 17,500   trajectory-rich, new emission logic
  gui360   16,000   use ~10,862 singles + grounding 2,500 + understanding 2,650
  pcagente ~4,303   one single per event (single-only by design)
  groundcua 4,000   grounding (not part of the sequence mix)
  replay    7,500   not part of the sequence mix

It simulates BOTH the pre-patch emission (BEFORE: 55/40/5 trajectory draw,
2 windows + 2 singles per window-role trajectory) and the current
action_representation_specs (AFTER) so the delta is measurable, checks the
sample-level band (single 60-68 / window 27-35 / chunk 4-7 over ACTION
samples), and estimates tokens with the per-representation averages from
DATA_QUALITY_REPORT.md section 8 (single 1.8k / window 4.2k / chunk 6.5k /
grounding 1.1k / understanding 2.4k / replay 0.9k).

Deterministic: an LCG drives every random draw, so reruns are identical.
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.coordinates import Action, CoordSpace
from processing.sampling import action_representation_specs, select_step_indices
from processing.windows import (ACTION_SAMPLE_BANDS, ACTION_SOURCE_RATIOS,
                                CHUNK, REPRESENTATION_RATIOS, SINGLE, WINDOW,
                                Step, Trajectory, build_chunk, build_single,
                                build_window, choose_representation,
                                suggest_window_starts)

QUOTAS = {"procua": 46000, "videocua": 17500, "gui360": 16000,
          "groundcua": 4000, "pcagente": 4303, "replay": 7500}
GUI360_USE = 10862
GUI360_GROUNDING = 2500
GUI360_UNDERSTANDING = 2650

TOKENS = {"single": 1800, "window": 4200, "chunk": 6500,
          "grounding": 1100, "understanding": 2400, "replay": 900}
PREV_EPOCH_ESTIMATE = 234_000_000  # documented planning figure
TOKEN_GUARD = 1.20

# audit-derived source behaviour (2026-08): anchor eligibility + dedup +
# quality attrition per emitted spec, and fraction of windows/chunks that pass
# the informativeness gates (fabricated phash distances model this directly)
SURVIVAL = {
    "procua": {"single": 0.90, "window": 0.80, "chunk": 0.80},
    "videocua": {"single": 0.90, "window": 0.55, "chunk": 0.55},
}


class LCG:
    def __init__(self, seed=12345):
        self.s = seed

    def u(self):
        self.s = (1103515245 * self.s + 12345) % (1 << 31)
        return self.s / float(1 << 31)


def sim_trajectory(tid: str, n: int, informative_frac: float, rng: LCG) -> Trajectory:
    """Trajectory with fabricated phashes: consecutive screens differ in >6
    bits with probability informative_frac, else are near-identical."""
    steps = []
    ph = int(rng.u() * (1 << 62))
    for i in range(n):
        if i > 0:
            if rng.u() < informative_frac:
                # flip 8 DISTINCT bits -> hamming distance 8 > threshold 6
                bits = set()
                while len(bits) < 8:
                    bits.add(int(rng.u() * 62))
                for b in bits:
                    ph = ph ^ (1 << b)
            # else: near-identical frame (small delta)
            else:
                ph = ph ^ (1 << int(rng.u() * 4))
        verb = ["click", "type", "scroll", "press"][i % 4]
        action = Action(verb, args={"text": f"v{i}"} if verb == "type" else {},
                        points=[(100 + i * 7, 200 + i * 3)],
                        original_space=CoordSpace.PIXEL)
        steps.append(Step(step_id=f"{tid}_s{i}", image_bytes=b"", image_size=(1920, 1080),
                          action=action, phash=int(ph), prev_phash=steps[-1].phash if steps else None,
                          signals=set(), metadata={}))
    return Trajectory(trajectory_id=tid, task="Sort the files and export the report.",
                      steps=steps, app="excel", source="procua", metadata={})


def old_emission_specs(traj: Trajectory, cap: int = 4):
    """Pre-patch emission (procua/videocua samples_for_trajectory, 2026-08-16
    morning): 55/40/5 trajectory draw; window-role -> 2 windows + 2 singles;
    chunk-role -> 1 chunk + 2 singles."""
    representation = choose_representation(traj.trajectory_id, REPRESENTATION_RATIOS)
    indices = select_step_indices(traj, cap=cap)
    if not indices:
        return []
    specs = []
    if representation == SINGLE:
        specs = [build_single(traj, i) for i in indices]
    elif representation == WINDOW:
        for s in suggest_window_starts(traj, 4, count=2):
            specs.append(build_window(traj, s, 4))
        specs += [build_single(traj, i) for i in indices[:2]]
    else:
        if traj.length >= 8:
            start = suggest_window_starts(traj, 8, count=1)[0]
            specs.append(build_chunk(traj, start, min(12, traj.length - start)))
        specs += [build_single(traj, i) for i in indices[:2]]
    return specs


def simulate_source(source: str, quota: int, length_range, informative_frac: float,
                    emission, ratios=None, seed=1) -> dict:
    rng = LCG(seed)
    counts = {"single": 0, "window": 0, "chunk": 0}
    surv = SURVIVAL[source]
    traj_i = 0
    emitted = 0
    while emitted < quota and traj_i < quota * 20:  # generous trajectory supply
        n = length_range[0] + int(rng.u() * (length_range[1] - length_range[0] + 1))
        traj = sim_trajectory(f"{source}_traj_{traj_i}", max(n, 1), informative_frac, rng)
        specs = emission(traj, cap=4, ratios=ratios) if emission is action_representation_specs \
            else emission(traj, cap=4)
        kept = []
        for spec in specs:
            r = surv.get(spec.representation, 0.8)
            if rng.u() <= r:
                kept.append(spec.representation)
        for rep in kept:
            if emitted >= quota:
                break
            counts[rep if rep in counts else "single"] += 1
            emitted += 1
        traj_i += 1
    return {"counts": counts, "total": emitted, "trajectories": traj_i}


def mix_report(per_source: dict) -> dict:
    tot = {"single": 0, "window": 0, "chunk": 0}
    for src, r in per_source.items():
        for k, v in r["counts"].items():
            tot[k] += v
    n = max(1, sum(tot.values()))
    return {k: round(v / n, 4) for k, v in tot.items()}, tot


def token_estimate(action_mix: dict, gui360_use: int) -> dict:
    procua = action_mix["procua"]["counts"]
    videocua = action_mix["videocua"]["counts"]
    pcae = action_mix["pcagente"]["counts"]
    tokens = 0
    tokens += sum(procua[k] * TOKENS[k] for k in TOKENS if k in procua)
    tokens += sum(videocua[k] * TOKENS[k] for k in TOKENS if k in videocua)
    tokens += pcae["single"] * TOKENS["single"] + gui360_use * TOKENS["single"]
    tokens += QUOTAS["groundcua"] * TOKENS["grounding"]
    tokens += GUI360_GROUNDING * TOKENS["grounding"]
    tokens += GUI360_UNDERSTANDING * TOKENS["understanding"]
    tokens += QUOTAS["replay"] * TOKENS["replay"]
    n_action = sum(sum(r["counts"].values()) for r in action_mix.values()) + gui360_use
    n_total = n_action + QUOTAS["groundcua"] + GUI360_GROUNDING + GUI360_UNDERSTANDING + QUOTAS["replay"]
    return {"estimated_epoch_tokens": tokens,
            "estimated_average_tokens_per_sample": round(tokens / max(1, n_total)),
            "guard_prev_estimate": PREV_EPOCH_ESTIMATE,
            "guard_limit": int(PREV_EPOCH_ESTIMATE * TOKEN_GUARD),
            "guard_ok": tokens <= PREV_EPOCH_ESTIMATE * TOKEN_GUARD}


def run(emit_new=True, ratios=None) -> dict:
    emission = action_representation_specs if emit_new else old_emission_specs
    per_source = {
        "procua": simulate_source("procua", QUOTAS["procua"], (6, 33), 0.75,
                                  emission, ratios=ratios, seed=11),
        "videocua": simulate_source("videocua", QUOTAS["videocua"], (5, 16), 0.70,
                                    emission, ratios=ratios, seed=22),
        "pcagente": {"counts": {"single": QUOTAS["pcagente"], "window": 0, "chunk": 0},
                     "total": QUOTAS["pcagente"], "trajectories": QUOTAS["pcagente"]},
    }
    shares, totals = mix_report(per_source)
    # gui360 use singles join the action mix
    action_totals = dict(totals)
    action_totals["single"] += GUI360_USE
    n = max(1, sum(action_totals.values()))
    action_shares = {k: round(v / n, 4) for k, v in action_totals.items()}
    bands_ok = all(ACTION_SAMPLE_BANDS[k][0] <= action_shares[k] <= ACTION_SAMPLE_BANDS[k][1]
                   for k in ("single", "window", "chunk"))
    return {
        "mode": "AFTER (action_representation_specs)" if emit_new else "BEFORE (old emission)",
        "ratios_used": (ratios or ACTION_SOURCE_RATIOS) if emit_new else REPRESENTATION_RATIOS,
        "per_source": {k: {"counts": v["counts"], "trajectories": v["trajectories"]}
                       for k, v in per_source.items()},
        "action_sample_counts": action_totals,
        "action_sample_shares": action_shares,
        "bands": {k: ACTION_SAMPLE_BANDS[k] for k in ("single", "window", "chunk")},
        "bands_ok": bands_ok,
        "tokens": token_estimate(per_source, GUI360_USE),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true",
                    help="grid-search ACTION_SOURCE_RATIOS candidates that land inside the band")
    args = ap.parse_args()

    before = run(emit_new=False)
    after = run(emit_new=True)

    def show(r):
        print(f"\n== {r['mode']} ==")
        print("  per-source counts:", json.dumps({k: v["counts"] for k, v in r["per_source"].items()}))
        print("  action sample shares:", r["action_sample_shares"],
              "bands_ok:", r["bands_ok"])
        print("  tokens:", json.dumps(r["tokens"]))

    show(before)
    show(after)

    if args.search and not after["bands_ok"]:
        best = None
        for pw in range(30, 60, 2):
            for pc in range(10, 34, 2):
                ps = 100 - pw - pc
                if ps <= 0:
                    continue
                ratios = {SINGLE: ps / 100, WINDOW: pw / 100, CHUNK: pc / 100}
                r = run(emit_new=True, ratios=ratios)
                sh = r["action_sample_shares"]
                if r["bands_ok"]:
                    dist = (abs(sh["single"] - 0.65) + abs(sh["window"] - 0.30)
                            + abs(sh["chunk"] - 0.05))
                    if best is None or dist < best[0]:
                        best = (dist, ratios, sh)
        if best:
            print("\n[search] suggested ratios:", best[1], "-> shares", best[2])
        else:
            print("\n[search] no candidate inside the band — widen the grid")

    print("\nSUMMARY_JSON")
    print(json.dumps({"before": {"shares": before["action_sample_shares"],
                                 "tokens": before["tokens"]},
                      "after": {"shares": after["action_sample_shares"],
                                "bands_ok": after["bands_ok"],
                                "tokens": after["tokens"]}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
