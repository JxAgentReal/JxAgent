#!/usr/bin/env python3
"""Simulate the optimized final sampler WITHOUT downloading datasets.

Uses real audit-derived distributions (2026-08, .audit/*.json where
available) plus synthetic fixtures to verify that the final quotas, strata
and quality gates behave: no app dominance, no per-trajectory flooding, no
infinite search, no duplicate floods, realized reasoning rate, bucket mix.

Run:  python tools/simulate_sampler.py [--scale 0.02]
Writes: .audit/simulation.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from processing.coordinates import Action, CoordSpace, Point, parse_pc_agent_e
from processing.dedup import DedupIndex, phash
from processing.images import load_image
from processing.reasoning import ReasoningGate
from processing.sampling import select_step_indices
from processing.state import BuildState
from sources.common import BuildContext

sys.path.insert(0, str(ROOT / "tests"))
from conftest import make_png_bytes  # noqa: E402

RNG = random.Random(20260816)


def make_step(tid, i, verb, size=(640, 360), marker=None, signals=()):
    from processing.windows import Step
    marker = i if marker is None else marker
    data = make_png_bytes(size[0], size[1], marker=marker)
    act = Action(verb, points=[((100 + i * 7) % (size[0] - 2), (200 + i * 11) % (size[1] - 2))],
                 original_space=CoordSpace.PIXEL)
    h = phash(load_image(data))
    return Step(step_id=f"{tid}_s{i}", image_bytes=data, image_size=size,
                action=act, phash=h, signals=set(signals))


def simulate_procua(ctx, n_traj, lengths, verbs, apps):
    from processing.windows import Trajectory, build_single, build_window, build_chunk, choose_representation, suggest_window_starts
    got = []
    repr_counts = Counter()
    per_traj = Counter()
    for t in range(n_traj):
        if ctx.remaining("procua") <= 0:
            break
        tid = f"pt_{t}"
        n = lengths[t % len(lengths)]
        app = apps[t % len(apps)]
        task = RNG.choice([
            "Sort the rows by revenue and export the sheet as CSV.",
            "In LibreOffice Calc, fill Pass/Fail for all 28 students based on scores.",
            "Compress the five text files into a password zip.",
            "Open the document and enable the Developer tab.",
            "Save the presentation as PDF with speaker notes.",
        ])
        steps = []
        for i in range(n):
            verb = verbs[i % len(verbs)]
            signals = set()
            if i == 4 and n > 6:
                signals |= {"no_state_change", "recovery"}
            if i == n - 1:
                signals |= {"save", "verification"}
            steps.append(make_step(tid, i, verb, signals=signals,
                                   marker=(3 if (i == 5 and n > 6) else i)))
        traj = Trajectory(trajectory_id=tid, task=task, steps=steps, app=app, source="procua")
        from sources.procua import samples_for_trajectory
        samples = samples_for_trajectory(traj, ctx)
        got.extend(samples)
        per_traj[tid] = len(samples)
        for s in samples:
            repr_counts[s["metadata"]["representation"]] += 1
    return got, repr_counts, per_traj


def simulate_videocua_anchors():
    """The MOVE_TO share of selected anchors must collapse to ~0."""
    verbs = (["move"] * 45 + ["click"] * 44 + ["typing"] * 5 + ["press"] * 3 +
             ["drag"] * 1 + ["scroll"] * 2)
    verbs = [v if v != "typing" else "type" for v in verbs]
    verbs = [v if v != "press" else "press" for v in verbs]
    anchors = Counter()
    total = Counter()
    for t in range(300):
        steps = [make_step(f"vt_{t}", i, RNG.choice(verbs)) for i in range(10)]
        from processing.windows import Trajectory
        traj = Trajectory(trajectory_id=f"vt_{t}", task="Do the thing", steps=steps,
                          app="x", source="videocua")
        for s in steps:
            total[s.action.verb] += 1
        for idx in select_step_indices(traj, cap=4):
            anchors[steps[idx].action.verb] += 1
    return total, anchors


def simulate_gui360_balance(quota):
    from processing.sampling import AppCap
    cap = AppCap(cap=max(30, quota * 3 // 5))
    # stream order from the audit: excel block, then word, then ppt (+4s variants)
    stream = (["excel"] * 3600) + (["excel"] * 2000) + (["word"] * 1800) + \
             (["word"] * 1500) + (["ppt"] * 1400) + (["ppt"] * 562)
    accepted = Counter()
    for app in stream:
        if sum(accepted.values()) >= quota:
            break
        if cap.allow(app):
            cap.record(app)
            accepted[app] += 1
    return accepted


def simulate_groundcua_buckets():
    from processing.quality import BucketQuota, grounding_bucket
    from sources.groundcua import select_element
    # real size distribution from the audit: 2% tiny / 37% small / 39% medium / 23% large
    dist = (["tiny"] * 2 + ["small"] * 37 + ["medium"] * 39 + ["large"] * 23)
    quota = BucketQuota()
    picks = Counter()
    screens = 0
    while sum(picks.values()) < 200 and screens < 4000:
        screens += 1
        entries = []
        for k in range(60):
            b = dist[RNG.randrange(len(dist))]
            w = {"tiny": RNG.randint(6, 15), "small": RNG.randint(16, 31),
                 "medium": RNG.randint(32, 63), "large": RNG.randint(64, 200)}[b]
            entries.append({"bbox": [10 * (k % 12), 30 * (k // 12),
                                     10 * (k % 12) + w, 30 * (k // 12) + max(10, w // 2)],
                            "text": f"control {k}", "category": "Button"})
        e = select_element(entries, quota=quota)
        if e:
            w = e["bbox"][2] - e["bbox"][0]
            picks[grounding_bucket(w)] += 1
            quota.record(grounding_bucket(w))
    return picks, screens


def simulate_pcagente_real_actions(ctx):
    """Real 78-task audit actions + synthetic screenshots through the gate."""
    audit = json.load(open(ROOT / ".audit" / "pcae_tasks.json"))
    from sources.pc_agent_e import build_trajectory, samples_for_trajectory
    from sources.pc_agent_e import parse_md_description
    verb_counter = Counter()
    rejections = Counter()
    total = 0
    per_traj = Counter()
    for t in audit[:40]:
        if ctx.remaining("pcagente") <= 0:
            break
        shots = {}
        needed = []
        events = []
        for line in t["jsonl"].splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            name = (ev.get("screenshot") or "").split("/")[-1]
            if not name:
                continue
            if name not in shots:
                shots[name] = make_png_bytes(640, 360, marker=len(shots) % 200)
            events.append(ev)
        jsonl = "\n".join(json.dumps({**ev, "screenshot": ev["screenshot"]}) for ev in events)
        traj = build_trajectory(t["tid"], jsonl, t["md"], shots)
        if traj is None:
            continue
        samples = samples_for_trajectory(traj, ctx)
        per_traj[t["tid"]] = len(samples)
        for s in samples:
            from processing.validation import extract_action_text
            verb_counter[extract_action_text(s).split("(")[0]] += 1
        total += len(samples)
        rejections.update(ctx.rejected.get("pcagente", {}))
    return verb_counter, rejections, total, per_traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.02)
    args = ap.parse_args()
    scale = args.scale

    tmp = tempfile.TemporaryDirectory()
    ctx = BuildContext(
        dataset_root=tmp.name, state=BuildState(os.path.join(tmp.name, "state")),
        config={"context_budget": 8192, "per_trajectory_cap": 4},
        dedup=DedupIndex(), reasoning_gate=ReasoningGate(rate=0.12),
        quota={"procua": int(46000 * scale), "pcagente": int(4503 * scale)})

    report = {}

    # ---- ProCUA
    lengths = [8, 11, 14, 14, 16, 19, 22, 27]      # audit med/avg/max
    verbs = (["click"] * 25 + ["key_down"] * 8 + ["type"] * 7 + ["wait"] * 3 +
             ["move"] * 2 + ["double_click"] * 1 + ["scroll"] * 1)
    apps = (["libreoffice_calc"] * 5 + ["libreoffice_writer"] * 3 +
            ["libreoffice_impress"] * 2 + ["vscode"] * 2 + ["file_manager"] * 2 + ["desktop"])
    got, repr_counts, per_traj = simulate_procua(ctx, n_traj=min(90, int(3000 * scale * 3)),
                                                 lengths=lengths, verbs=verbs, apps=apps)
    report["procua"] = {
        "samples": len(got),
        "max_per_trajectory": max(per_traj.values()) if per_traj else 0,
        "representations": dict(repr_counts),
        "anchor_verbs": dict(Counter(
            s["messages"][-1]["content"].split("(")[0].strip()
            for s in got)),
        "quality_buckets": dict(Counter(s["metadata"]["quality"]["bucket"] for s in got)),
        "avg_est_tokens": round(sum(s["metadata"]["estimated_tokens"] for s in got) / max(1, len(got))),
    }

    # ---- VideoCUA anchors
    total_v, anchor_v = simulate_videocua_anchors()
    move_share_raw = total_v["move"] / sum(total_v.values())
    move_share_anchor = anchor_v.get("move", 0) / max(1, sum(anchor_v.values()))
    report["videocua"] = {
        "move_share_raw": round(move_share_raw, 3),
        "move_share_selected": round(move_share_anchor, 4),
        "selected_verbs": dict(anchor_v),
    }

    # ---- GUI-360 balance
    use_quota = 10862  # full-size: the balance check needs real block sizes
    accepted = simulate_gui360_balance(max(30, use_quota))
    shares = {k: round(v / max(1, sum(accepted.values())), 3) for k, v in accepted.items()}
    report["gui360_use_balance"] = {"counts": dict(accepted), "shares": shares,
                                    "max_app_share": max(shares.values())}

    # ---- GroundCUA buckets
    picks, screens = simulate_groundcua_buckets()
    report["groundcua_buckets"] = {"picked": dict(picks), "screens_scanned": screens,
                                   "shares": {k: round(v / max(1, sum(picks.values())), 3)
                                              for k, v in picks.items()}}

    # ---- PC-Agent-E with real audit actions
    verb_counter, rejections, total_pca, per_traj_pca = simulate_pcagente_real_actions(ctx)
    report["pcagente"] = {
        "samples": total_pca,
        "verbs": dict(verb_counter.most_common()),
        "rejections": dict(rejections),
        "max_per_trajectory": max(per_traj_pca.values()) if per_traj_pca else 0,
        "finish_samples": verb_counter.get("finish", 0),
        "wait_samples": verb_counter.get("wait", 0),
    }

    # ---- reasoning realized rate
    report["reasoning"] = ctx.reasoning_gate.stats()

    out = ROOT / ".audit" / "simulation.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    # ---- invariants
    assert report["procua"]["max_per_trajectory"] <= 4
    assert move_share_anchor < 0.02, "MOVE_TO must not anchor samples"
    assert report["gui360_use_balance"]["max_app_share"] <= 0.61
    assert report["gui360_use_balance"]["shares"].get("ppt", 0) > 0
    gc = report["groundcua_buckets"]["shares"]
    assert gc.get("small", 0) <= 0.55 and gc.get("tiny", 0) >= 0.05
    assert report["pcagente"]["finish_samples"] > 0
    rr = report["reasoning"]["realized_rate"]
    assert 0.08 <= rr <= 0.16, rr
    print("SIMULATION INVARIANTS: ALL PASS")
    tmp.cleanup()


if __name__ == "__main__":
    main()
