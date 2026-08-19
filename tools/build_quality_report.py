#!/usr/bin/env python3
"""Build DATA_QUALITY_REPORT + data_quality_report.json + quality_review/ pack
from the 2026-08 audit artifacts (.audit/*), smoke builds, and pipeline stats.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from processing.validation import extract_action_text  # noqa: E402

# ------------------------------------------------------------------ scores
SOURCE_SCORES = {
    "procua":   {"score": 7.0, "rationale": "Rich OSWorld-style goals, executed "
                 "trajectories, healthy lengths (med 14). Synthetic single-generator, "
                 "app diversity unverifiable (no app metadata), click 50%, 8.4% "
                 "unparseable actions. Trimmed 48k->46k."},
    "gui360":   {"score": 7.5, "rationale": "Office depth is the strongest slice: "
                 "use cohort carries terminate(status) finish actions (15% of steps) "
                 "and balanced excel/word/ppt after stream balancing. Grounding "
                 "intents are verbose trajectory-speak (overlaps GroundCUA -> cut to "
                 "2.5k). Understanding answers averaged 27k chars -> capped to 60 "
                 "controls. 20k->16k."},
    "videocua": {"score": 8.0, "rationale": "Unique HUMAN behavior: 20% of steps "
                 "follow >=3s gaps (wait decisions), real drags/hotkeys, professional "
                 "apps. 45% raw MOVE_TO spam now fully excluded from anchors; no "
                 "finish actions (covered by other sources). 15k->17.5k."},
    "groundcua": {"score": 7.5, "rationale": "Precise human-verified pixel "
                  "annotations, 74.8 elements/screen, 87 platforms incl. Windows. "
                  "Selector was small-biased (63/70) -> bucket quotas 25/40/25/10. "
                  "Some weak labels ('image','click'). 5k->4k."},
    "pcagente": {"score": 7.0, "rationale": "Real Windows-settings tasks (OSWorld "
                 "'os' domain), finish (17%) + wait actions now parsed (were silently "
                 "dropped), 8.5 steps/task. Verbose thoughts excluded from targets; "
                 "trivial first clicks downweighted to C; duplicate actions rejected. "
                 "Quality gate yields ~4.3k of 4,503."},
    "replay":   {"score": 8.0, "rationale": "All licenses verified (Apache-2.0/MIT/"
                 "CC-BY-family), healthy formats audited on real rows, no OSWorld "
                 "contamination path. Rebalanced toward instruction following and "
                 "VQA; math trimmed (GSM-style seed overlap with regression panel)."},
}

CAPABILITY_MATRIX = {
    # rows: capability, columns per source 0-3
    "Recovery":                    {"procua": 2, "gui360": 1, "videocua": 3, "groundcua": 0, "pcagente": 2, "replay": 0},
    "Verification":                {"procua": 2, "gui360": 2, "videocua": 1, "groundcua": 0, "pcagente": 2, "replay": 0},
    "Loop breaking":               {"procua": 1, "gui360": 1, "videocua": 2, "groundcua": 0, "pcagente": 1, "replay": 0},
    "Wait states":                 {"procua": 1, "gui360": 0, "videocua": 3, "groundcua": 0, "pcagente": 2, "replay": 0},
    "Finish decisions":            {"procua": 0, "gui360": 3, "videocua": 0, "groundcua": 0, "pcagente": 3, "replay": 0},
    "Save/export":                 {"procua": 3, "gui360": 3, "videocua": 1, "groundcua": 0, "pcagente": 1, "replay": 0},
    "Dialogs":                     {"procua": 2, "gui360": 2, "videocua": 1, "groundcua": 1, "pcagente": 1, "replay": 0},
    "Sorting":                     {"procua": 2, "gui360": 2, "videocua": 0, "groundcua": 0, "pcagente": 0, "replay": 0},
    "Ranking":                     {"procua": 1, "gui360": 1, "videocua": 0, "groundcua": 0, "pcagente": 0, "replay": 0},
    "Exact quantity":              {"procua": 2, "gui360": 2, "videocua": 0, "groundcua": 0, "pcagente": 0, "replay": 0},
    "Multi target":                {"procua": 2, "gui360": 1, "videocua": 1, "groundcua": 0, "pcagente": 1, "replay": 0},
    "Multi app":                   {"procua": 1, "gui360": 1, "videocua": 1, "groundcua": 0, "pcagente": 1, "replay": 0},
    "Long horizon":                {"procua": 2, "gui360": 1, "videocua": 2, "groundcua": 0, "pcagente": 1, "replay": 0},
    "Professional apps":           {"procua": 1, "gui360": 1, "videocua": 3, "groundcua": 3, "pcagente": 1, "replay": 0},
    "Office":                      {"procua": 3, "gui360": 3, "videocua": 1, "groundcua": 0, "pcagente": 0, "replay": 0},
    "Grounding":                   {"procua": 1, "gui360": 2, "videocua": 1, "groundcua": 3, "pcagente": 1, "replay": 0},
    "Tiny targets":                {"procua": 0, "gui360": 1, "videocua": 0, "groundcua": 3, "pcagente": 0, "replay": 0},
    "Drag":                        {"procua": 1, "gui360": 1, "videocua": 3, "groundcua": 1, "pcagente": 1, "replay": 0},
    "Keyboard":                    {"procua": 2, "gui360": 1, "videocua": 2, "groundcua": 0, "pcagente": 2, "replay": 0},
    "Text entry":                  {"procua": 2, "gui360": 1, "videocua": 2, "groundcua": 0, "pcagente": 1, "replay": 0},
    "State understanding":         {"procua": 1, "gui360": 2, "videocua": 1, "groundcua": 1, "pcagente": 1, "replay": 0},
    "General reasoning preservation": {"procua": 0, "gui360": 0, "videocua": 0, "groundcua": 0, "pcagente": 0, "replay": 3},
    "Coding preservation":         {"procua": 0, "gui360": 0, "videocua": 0, "groundcua": 0, "pcagente": 0, "replay": 3},
    "Math preservation":           {"procua": 0, "gui360": 0, "videocua": 0, "groundcua": 0, "pcagente": 0, "replay": 3},
    "VQA preservation":            {"procua": 0, "gui360": 0, "videocua": 0, "groundcua": 0, "pcagente": 0, "replay": 3},
}

FINAL_MIX = {"procua": 46000, "gui360": 16000, "videocua": 17500,
             "groundcua": 4000, "pcagente": 4503, "replay": 7500}

FAILURE_MODE_COVERAGE = {
    # estimated share of FINAL dataset addressing each mode (actionable samples)
    "Recovery":          {"est_pct": 8.5, "basis": "VideoCUA no-state-change ~10% of steps + ProCUA phash-detected; sim: 2/29 with plans"},
    "Verification":      {"est_pct": 6.0, "basis": "goal-text verify signals ~10% + save/export steps"},
    "Premature finish":  {"est_pct": 2.4, "basis": "~2.3k explicit finish targets (PCAE 17%, GUI360 15% of use) now supervised"},
    "Loops / no progress": {"est_pct": 3.0, "basis": "repeated-identical rejects + recovery-after-no-change samples"},
    "Save/export":       {"est_pct": 9.0, "basis": "ProCUA+GUI360 goal patterns (save as/export/print)"},
    "Dialogs":           {"est_pct": 5.0, "basis": "dialog/confirm/overwrite task patterns"},
    "Wait/loading":      {"est_pct": 2.8, "basis": "VideoCUA >=3s gaps (20.2% of its steps)"},
    "Sorting/ranking":   {"est_pct": 5.5, "basis": "ProCUA sort/arrange goals + GUI360 Office"},
    "Exact quantity":    {"est_pct": 4.0, "basis": "'all 28 students'/'exactly N' goal patterns"},
    "Multi target":      {"est_pct": 5.0, "basis": "multi-clause goals"},
    "Multi app":         {"est_pct": 1.5, "basis": "rare in all sources - known gap, mitigated by Office+settings breadth"},
    "Long horizon":      {"est_pct": 5.0, "basis": "chunks (5% of action) + VideoCUA long tasks + history windows"},
    "Difficult grounding": {"est_pct": 6.8, "basis": "GroundCUA 4k (65% tiny+small) + GUI360 grounding 2.5k"},
}

REDUNDANCY = {
    "gui360_grounding_vs_groundcua": ("both teach point-to-element; GUI-360 intents are "
                                      "verbose trajectory sentences, GroundCUA has clean pixel "
                                      "labels + 87 platforms -> GUI360 grounding cut 5k->2.5k, "
                                      "GroundCUA kept as the specialist (4k)"),
    "procua_office_vs_gui360_office": ("both cover Office; ProCUA is synthetic long-horizon, "
                                       "GUI-360 use is finer-grained Office UI ops incl. finish -> "
                                       "both kept, ProCUA trimmed 48k->46k"),
    "videocua_move_vs_everyone": ("MOVE_TO mouse-motion spam (45% of VideoCUA actions) removed "
                                  "entirely - pure redundancy"),
    "pcae_windows_vs_gui360_windows": ("PC-Agent-E is Windows-OS/settings heavy, GUI-360 is "
                                       "Office-heavy: complementary, not redundant"),
    "replay_vs_cu": ("no functional overlap; replay exists for capability preservation"),
}


def load_smoke_rows():
    rows = []
    for root in ("SmokeVideo2", "SmokeGround2", "SmokeProCUA2", "SmokeVideo",
                 "SmokeGround", "SmokeProCUA"):
        d = ROOT / root / "final"
        if not d.exists():
            continue
        for split in ("train", "validation"):
            f = d / f"{split}.jsonl"
            if f.exists():
                for line in f.open(encoding="utf-8"):
                    if line.strip():
                        r = json.loads(line)
                        r["_root"] = root
                        rows.append(r)
    return rows


def build_review_pack(rows):
    out = ROOT / "quality_review"
    (out / "computer_use").mkdir(parents=True, exist_ok=True)
    (out / "reasoning").mkdir(parents=True, exist_ok=True)
    (out / "grounding").mkdir(parents=True, exist_ok=True)
    (out / "replay_refs").mkdir(parents=True, exist_ok=True)

    index = {"computer_use": [], "reasoning": [], "grounding": [], "replay": []}

    # real smoke samples
    cu = [r for r in rows if r.get("task_type") == "action"]
    for r in cu:
        entry = {"file": f"{r['_root']}/{r['images'][0] if r['images'] else ''}",
                 "source": r["source"], "traj": r["trajectory_id"],
                 "task_rendered": r["messages"][1]["content"][:400] if len(r["messages"]) > 1 else "",
                 "target": r["messages"][-1]["content"][:300],
                 "quality": r["metadata"].get("quality"),
                 "representation": r["metadata"].get("representation")}
        index["computer_use"].append(entry)
        if r["messages"][-1]["content"].startswith("Plan:"):
            index["reasoning"].append({**entry,
                "plan": r["messages"][-1]["content"].split("\nAction:")[0]})
        if r["task_type"] == "grounding":
            index["grounding"].append(entry)

    # GUI-360 real audit rows rendered without images (references)
    try:
        g = json.load(open(ROOT / ".audit" / "gui360_rows.json"))
        n = 0
        for r in g["desktop.use"][:160]:
            meta = json.loads(r["row"]["metadata"]) if isinstance(r["row"]["metadata"], str) else r["row"]["metadata"]
            msgs = json.loads(r["row"]["messages"]) if isinstance(r["row"]["messages"], str) else r["row"]["messages"]
            task = ""
            actions = []
            for m in msgs:
                if m.get("role") == "user":
                    for c in (m.get("content") or []):
                        if isinstance(c, dict) and c.get("type") == "text" and not task:
                            task = c["text"]
                elif m.get("role") == "assistant":
                    tc = (m.get("tool_calls") or [{}])[0].get("function") or {}
                    args = tc.get("arguments") or {}
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except Exception: args = {}
                    co = args.get("coordinate")
                    actions.append(f"{tc.get('name')}({co if co else args})")
            if not task:
                continue
            index["computer_use"].append({
                "source": "gui360", "traj": meta.get("others", {}).get("id"),
                "task_rendered": task[:400], "target_actions": actions[:8],
                "note": "audit row reference; images render in the full build"})
            n += 1
        # grounding refs from grounding.point cohort
        for r in g["desktop.grounding.point"][:50]:
            folded = r["row"].get("_folded")
            members = json.loads(folded) if isinstance(folded, str) and folded else [r["row"]]
            for mem in members[:1]:
                msgs = json.loads(mem["messages"]) if isinstance(mem["messages"], str) else mem["messages"]
                intent, pt = "", None
                for m in msgs:
                    if m.get("role") == "user":
                        for c in (m.get("content") or []):
                            if isinstance(c, dict) and c.get("type") == "text":
                                intent = c["text"]
                    tc = (m.get("tool_calls") or [{}])[0].get("function") or {}
                    if tc.get("name") == "point":
                        args = tc.get("arguments") or {}
                        if isinstance(args, str):
                            try: args = json.loads(args)
                            except Exception: args = {}
                        pt = args.get("coordinate")
                if intent and pt:
                    index["grounding"].append({
                        "source": "gui360", "intent": intent[:300],
                        "point_norm_0_1000": pt,
                        "note": "audit row reference"})
    except FileNotFoundError:
        pass

    # PC-Agent-E real events as references
    try:
        audit = json.load(open(ROOT / ".audit" / "pcae_tasks.json"))
        for t in audit[:80]:
            for line in t["jsonl"].splitlines()[:4]:
                ev = json.loads(line)
                index["computer_use"].append({
                    "source": "pcagente", "traj": t["tid"],
                    "task_rendered": (t["md"].split("**Description:**")[1].split("\n")[0]
                                      if "**Description:**" in t["md"] else "")[:300],
                    "target": ev.get("action", "")[:120],
                    "element": ev.get("element", ""),
                    "note": "audit row reference; images render in the full build"})
    except FileNotFoundError:
        pass

    # replay references
    try:
        rep = json.load(open(ROOT / ".audit" / "replay_rows.json"))
        mc = json.load(open(ROOT / ".audit" / "replay_Magicoder-.json"))
        om = json.load(open(ROOT / ".audit" / "replay_orca-math-.json"))
        for r in mc[:16]:
            index["replay"].append({"source": "coding:magicoder",
                                    "user": r["instruction"][:300],
                                    "assistant": r["response"][:300]})
        for r in om[:12]:
            index["replay"].append({"source": "math:orca",
                                    "user": r["question"][:300],
                                    "assistant": r["answer"][:300]})
        for r in rep["smoltalk"][:8]:
            index["replay"].append({"source": "instruction:smoltalk",
                                    "turns": [m.get("role") + ": " + str(m.get("content"))[:120]
                                              for m in (r.get("messages") or [])[:4]]})
        for r in rep["cauldron_aokvqa"][:8]:
            t = (r.get("texts") or [{}])[0]
            index["replay"].append({"source": "vqa:aokvqa",
                                    "user": str(t.get("user", ""))[:200],
                                    "assistant": str(t.get("assistant", ""))[:120],
                                    "has_image": bool(r.get("images"))})
        for r in rep["hermes"][:6]:
            convs = r.get("conversations") or []
            index["replay"].append({"source": "tool:hermes",
                                    "turns": [f'{c.get("from")}: {str(c.get("value"))[:110]}' for c in convs[:4]],
                                    "tools_present": bool(r.get("tools"))})
    except FileNotFoundError:
        pass

    (out / "index.json").write_text(json.dumps(index, indent=1))
    counts = {k: len(v) for k, v in index.items()}
    print("review pack:", counts)
    return counts


def main():
    rows = load_smoke_rows()
    sim = json.load(open(ROOT / ".audit" / "simulation.json"))

    bucket_counter = Counter(r["metadata"].get("quality", {}).get("bucket", "n/a") for r in rows)
    report = {
        "generated": "2026-08-16",
        "source_quality_scores": {k: v["score"] for k, v in SOURCE_SCORES.items()},
        "source_quality_rationales": {k: v["rationale"] for k, v in SOURCE_SCORES.items()},
        "final_mixture": FINAL_MIX,
        "final_mixture_total": sum(FINAL_MIX.values()),
        "previous_mixture": {"procua": 48000, "gui360": 20000, "videocua": 15000,
                             "groundcua": 5000, "pcagente": 4503, "replay": 7500},
        "reasoning_percentage": 0.12,
        "sequence_mixture": {"single": 0.55, "window": 0.40, "chunk": 0.05},
        "image_policy": {"format": "webp", "quality": 80, "max_long": 1600,
                         "grounding_max_long": 1920,
                         "evidence": "visual q75 readable-but-soft; PSNR(q75,q80)=39.9dB, "
                                     "PSNR(q80,q85)=44.2dB; q85 +10% bytes for no visible gain"},
        "quality_buckets_definition": {"A": "high value hard example (>=7.0)",
                                       "B": "useful normal (>=5.0)",
                                       "C": "useful mainly for diversity (>=3.0)",
                                       "Reject": "insufficient value or incorrect"},
        "observed_buckets_smoke": dict(bucket_counter),
        "simulated_buckets_procua": sim["procua"]["quality_buckets"],
        "capability_coverage_matrix": CAPABILITY_MATRIX,
        "failure_mode_coverage_pct": {k: v["est_pct"] for k, v in FAILURE_MODE_COVERAGE.items()},
        "failure_mode_coverage_basis": {k: v["basis"] for k, v in FAILURE_MODE_COVERAGE.items()},
        "cross_source_redundancy": REDUNDANCY,
        "audit_findings_fixed": [
            "reasoning gate delivered 7.8% instead of 15% (quota deadlock) -> self-regulating gate, category renormalization",
            "double 'Plan: Plan:' prefix in assembled targets",
            "GUI-360 terminate(status) actions (15% of use steps) silently dropped -> finish parsing",
            "PC-Agent-E finish/wait action strings (112/666 in audit) silently dropped -> parsing",
            "VideoCUA MOVE_TO (45% of raw actions) anchoring samples -> anchor-ineligible",
            "ProCUA key_down/move (18%) anchoring samples -> anchor-ineligible",
            "GUI-360 streams app-contiguous blocks (first 300 rows 100% excel) -> per-app caps",
            "GroundCUA selector tiny/small bias (63/70 small) -> bucket quotas 25/40/25/10",
            "GUI-360 understanding answers ~27k chars/422 controls token-unusable -> capped at 24 controls",
            "ProCUA metadata has no application field -> app inferred from goal text",
            "PC-Agent-E repeated identical actions and consecutive waits -> rejected",
            "trivial first clicks auto-Rejected -> floored to C (diversity value)",
        ],
        "real_samples_inspected": {
            "gui360": {"use_rows": 300, "grounding_rows": 300, "understanding_rows": 300,
                       "offset_probes": 12, "limitation": "images via dataset-server only"},
            "pcagente": {"tasks": 78, "events": 666, "limitation": "screenshots not fetched in bulk"},
            "videocua": {"apps": 8, "tasks": 85, "actions": 699, "decoded_samples": 7},
            "groundcua": {"screens": 70, "elements": 5235, "platforms": 3},
            "procua": {"trajectories": 20, "actions": 392},
            "replay": {"rows": 491, "sources": 5},
        },
        "smoke_verification": {
            "videocua": "PASS (4 samples, quality+recovery reasoning live)",
            "groundcua": "PASS (4 samples, bucket A for tiny target)",
            "procua": "PASS (4 samples, app inference live)",
            "pcagente": "PENDING (network stalls on range-body CDN edge; standalone verified)",
            "gui360": "PENDING (needs HF_TOKEN for streaming; rows audited via datasets-server)",
            "replay": "PENDING (needs HF_TOKEN; rows audited via datasets-server/first-rows)",
        },
        "estimated_tokens_per_sample": {
            "single": 1800, "window": 4200, "chunk": 6500, "grounding": 1100,
            "understanding": 2400, "replay": 900,
            "weighted_average": 2450,
        },
        "estimated_storage_gb": 21,
        "tests": "run the current pytest suite; do not rely on historical fixed-count claims",
    }
    (ROOT / "data_quality_report.json").write_text(json.dumps(report, indent=1))
    pack_counts = build_review_pack(rows)
    report["review_pack_counts"] = pack_counts
    (ROOT / "data_quality_report.json").write_text(json.dumps(report, indent=1))
    print("data_quality_report.json written")


if __name__ == "__main__":
    main()
