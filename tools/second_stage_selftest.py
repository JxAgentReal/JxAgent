#!/usr/bin/env python3
"""Offline end-to-end data self-test for JxAgent second-stage hardening.

Creates a synthetic native-interface candidate pool, applies best-valid
selection, group-aware splitting, final validation and release gates, then
re-opens every emitted sample and validates it again. No network, model or GPU
is used.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import sys
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.assemble import assemble_grounding, assemble_sample
from processing.coordinates import Action, CoordSpace
from processing.dedup import DedupIndex
from processing.reasoning import ReasoningGate
from processing.selection import select_best_valid
from processing.splitting import assign_splits, group_overlap, split_samples
from processing.state import BuildState
from processing.validation import finalize, validate_sample
from processing.windows import SampleSpec, Step
from sources.common import BuildContext
from PIL import Image, ImageDraw


def make_png_bytes(w: int = 640, h: int = 480, marker: int = 0) -> bytes:
    img = Image.new("RGB", (w, h), (18 + marker % 40, 24, 30))
    d = ImageDraw.Draw(img)
    bx = (marker * 97) % max(1, w - 120)
    by = (marker * 53) % max(1, h - 120)
    d.rectangle([bx, by, bx + 100, by + 80],
                fill=(30 + (marker * 37) % 220, (marker * 71) % 220, (marker * 13) % 220))
    for i in range(8):
        x0 = (i * w // 8 + marker * 7) % max(1, w - 20)
        y0 = (i * h // 8 + marker * 13) % max(1, h - 20)
        d.rectangle([x0, y0, x0 + 12, y0 + 8], fill=(200 - marker % 100, 90, 40 + i * 5))
        d.text((4, i * 12), f"fixture {marker}", fill=(250, 250, 250))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def contract() -> dict:
    return {
        "schema_version": 1,
        "model_id": "Qwen/Qwen3.8-27B",
        "adapter": {"family": "jxagent_text_action_v1"},
        "coordinate_space": {"type": "normalized_0_1000"},
        "message_layout": {
            "system_prompt": "SELFTEST_NATIVE_SYSTEM",
            "image_placeholder": "<selftest_image>",
            "assistant_action_template": "NATIVE[{action}]",
            "visual_user_with_task_template": "{image}\nTASK={task}",
            "visual_user_without_task_template": "{image}",
            "older_history_template": "\nOLDER:\n{history}",
            "history_item_template": "H:{action}",
        },
        "history_policy": {
            "mode": "visual_recent_rounds",
            "recent_visual_rounds": 4,
            "task_location": "current_user",
            "older_actions": "coordinate_free",
            "older_summary_location": "current_user",
        },
    }


def make_ctx(root: Path) -> BuildContext:
    return BuildContext(
        dataset_root=str(root),
        state=BuildState(str(root / "state")),
        config={"context_budget": 50000, "per_trajectory_cap": 8,
                "_native_interface_contract": contract()},
        dedup=DedupIndex(), reasoning_gate=ReasoningGate(rate=0.0),
        offline=True, smoke=False, quota={"procua": 100, "groundcua": 100},
    )


def action_candidate(ctx: BuildContext, i: int, task: str, *, finish=False,
                     with_history=False) -> dict:
    img = make_png_bytes(640, 480, marker=100 + i)
    if finish:
        action = Action("finish", args={"status": "success"}, original="finish",
                        original_space=CoordSpace.PIXEL)
        md = {"explicit_success": True, "group_id": f"procua_group_{i}"}
    else:
        action = Action("click", points=[(80 + i * 23, 90 + i * 17)],
                        original=f"click {i}", original_space=CoordSpace.PIXEL)
        md = {"group_id": f"procua_group_{i}",
              "target_width_px": 28 + (i % 4) * 5,
              "target_height_px": 20 + (i % 3) * 4}
    step = Step(step_id=f"p{i}", image_bytes=img, image_size=(640, 480),
                action=action, metadata=md)
    prev = []
    hist = []
    if with_history:
        for j in range(5):
            pst = Step(step_id=f"p{i}_h{j}",
                       image_bytes=make_png_bytes(640, 480, marker=200 + i * 10 + j),
                       image_size=(640, 480),
                       action=Action("click", points=[(30 + j * 7, 40 + j * 9)],
                                     original_space=CoordSpace.PIXEL), metadata={})
            prev.append(pst); hist.append(pst.action_text)
    spec = SampleSpec(source="procua", trajectory_id=f"procua_traj_{i}",
                      step_ids=[step.step_id], representation="single", task=task,
                      current_step=step, history_texts=hist, app="selftest_app",
                      task_type="action", metadata={"_prev_steps": prev,
                                                    "group_id": f"procua_group_{i}"})
    out = assemble_sample(spec, ctx)
    if out is None:
        raise RuntimeError(f"action candidate {i} rejected: {ctx.state.source_counts('procua')}")
    return out


def run(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("images/procua", "images/groundcua", "final", "state"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(root)

    tasks = [
        "Save the document and export it as PDF.",
        "Open Settings and choose the requested option.",
        "Set the exact quantity to 3.",
        "Sort the items and rank them by value.",
        "Handle the confirmation dialog and modal window.",
        "Use the file chooser to select the requested file.",
        "Select the two requested targets in the application.",
    ]
    candidates = [action_candidate(ctx, i, task, with_history=(i == 2))
                  for i, task in enumerate(tasks)]
    candidates.append(action_candidate(
        ctx, 7, "Verify that the requested task is complete, then finish.", finish=True))

    for i in range(3):
        g = assemble_grounding(
            source="groundcua", trajectory_id=f"ground_{i}", step_id=f"g{i}",
            image_bytes=make_png_bytes(640, 480, marker=400 + i),
            instruction=f"unique target button {i}", target_xy=(120 + i * 80, 160 + i * 40),
            image_size=(640, 480), target_width_px=18 + i * 3,
            target_height_px=16 + i * 2, app="selftest_app", ctx=ctx,
            extra_meta={"group_id": f"ground_group_{i}", "referent_unique": True,
                        "element_category": "button"})
        if g is None:
            raise RuntimeError(f"grounding candidate {i} rejected")
        candidates.append(g)

    selected, selection_report = select_best_valid(
        candidates, {"procua": 6, "groundcua": 2},
        coverage_floors={"save_export": 1, "finish_verification": 1})
    if len(selected) != 8:
        raise RuntimeError(f"selection count mismatch: {len(selected)}")

    selected = assign_splits(selected, validation_pct=50.0)
    train, val = split_samples(selected)
    if group_overlap(train, val):
        raise RuntimeError("group leakage detected before finalization")

    # Persist the selection report exactly where production finalization hashes it.
    (root / "final" / "selection_report.json").write_text(
        json.dumps(selection_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stats = finalize(
        str(root), selected, validation_pct_marker=50.0,
        targets={"procua": 6, "groundcua": 2}, source_errors={},
        release_gates={
            "loss_token_gate": {"max_auxiliary_task_share": 0.20},
            "coverage_floors": {"save_export": 1, "finish_verification": 1},
        },
    )
    if stats.get("fatal_failure"):
        raise RuntimeError(f"finalization fatal: {stats.get('failures')}")
    if stats.get("quota_acceptance_passed") is not True:
        raise RuntimeError(f"quota acceptance failed: {stats.get('quota_acceptance')}")

    reread = []
    for name in ("train.jsonl", "validation.jsonl"):
        for line in (root / "final" / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                reread.append(json.loads(line))
    errors = []
    for row in reread:
        ok, reason = validate_sample(row, str(root))
        if not ok:
            errors.append({"sample": row.get("step_id"), "reason": reason})
    if errors:
        raise RuntimeError(f"re-read validation failed: {errors}")

    manifest = json.loads((root / "final" / "manifest.json").read_text(encoding="utf-8"))
    loss = json.loads((root / "final" / "loss_token_report_estimated.json").read_text(encoding="utf-8"))
    return {
        "status": "PASS",
        "synthetic_only": True,
        "network_used": False,
        "gpu_used": False,
        "model_used": False,
        "candidate_samples": len(candidates),
        "selected_samples": len(selected),
        "train_samples": len(train),
        "validation_samples": len(val),
        "reread_samples_validated": len(reread),
        "source_counts": stats.get("samples_per_source"),
        "motif_coverage": stats.get("motif_coverage"),
        "motif_coverage_unmet": stats.get("motif_coverage_unmet"),
        "assistant_loss_tokens_estimated": loss.get("total", {}).get("assistant_loss_tokens"),
        "quota_acceptance_passed": stats.get("quota_acceptance_passed"),
        "fatal_failure": stats.get("fatal_failure"),
        "manifest_fatal_failure": manifest.get("fatal_failure"),
        "image_files_hashed": stats.get("image_files_hashed"),
        "images_tree_hash": stats.get("images_tree_hash"),
        "selection_policy": selection_report.get("policy"),
        "frontier_scores_used": selection_report.get("frontier_scores_used"),
        "output_root": str(root),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=None,
                    help="persistent synthetic dataset root; temp dir when omitted")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    temp = None
    if args.output_root:
        root = Path(args.output_root).resolve()
        if root.exists():
            shutil.rmtree(root)
    else:
        temp = tempfile.TemporaryDirectory(prefix="jxagent_second_stage_selftest_")
        root = Path(temp.name)
    report = run(root)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text, end="")
    if temp is not None:
        temp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
