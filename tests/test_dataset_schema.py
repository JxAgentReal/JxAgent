"""End-to-end offline schema test: synthetic samples from two sources ->
assign_splits -> finalize -> validate every output file and fatal check."""
import json
import os

import pytest

from processing.splitting import assign_splits
from processing.state import BuildState
from processing.validation import (compute_stats, finalize, quality_audit,
                                   validate_sample)
from sources.pc_agent_e import build_trajectory, samples_for_trajectory
from sources.gui360 import use_row_to_trajectory
from tests.conftest import gui360_use_row, make_ctx, make_events


def build_synthetic_dataset(tmp_path):
    ctx = make_ctx(tmp_path, quota={"pcagente": 12, "gui360": 12})
    from processing.windows import build_single
    from processing.assemble import assemble_sample

    samples = []
    jsonl, shots = make_events(n=6)
    for t in range(3):
        traj = build_trajectory(f"task{t}", jsonl,
                                f"# T\n**Description:** Save and export file {t} as PDF.\n",
                                {k: v for k, v in shots.items()})
        samples.extend(samples_for_trajectory(traj, ctx))
    from sources.gui360 import use_row_to_trajectory
    from sources.gui360 import build_use_samples  # noqa: F401  (network; not used)
    for t in range(3):
        row = gui360_use_row(n_steps=3, row_id=f"excel_1_{t + 1}",
                             task=f"Save the workbook {t} as read-only.")
        traj = use_row_to_trajectory(row)
        for idx in range(len(traj.steps)):
            spec = build_single(traj, idx)
            spec.task_type = "action"
            s = assemble_sample(spec, ctx, trajectory=traj)
            if s:
                s["source"] = "gui360"
                samples.append(s)
    return samples


@pytest.fixture
def synthetic_dataset(tmp_path):
    return build_synthetic_dataset(tmp_path)


def test_ms_swift_schema(synthetic_dataset, tmp_path):
    for s in synthetic_dataset:
        assert set(s) >= {"messages", "images", "source", "trajectory_id",
                          "step_id", "task_type", "metadata"}
        roles = [m["role"] for m in s["messages"]]
        assert roles[0] in ("system", "user")
        assert roles[-1] == "assistant"
        n_img = sum(m["content"].count("<image>") for m in s["messages"])
        assert n_img == len(s["images"])
        ok, reason = validate_sample(s, str(tmp_path))
        assert ok, reason


def test_finalize_writes_all_outputs(synthetic_dataset, tmp_path):
    root = str(tmp_path)
    samples = assign_splits(synthetic_dataset, 3.0)
    stats = finalize(root, samples, decontamination_report={"total_scanned": 0},
                     dedup_stats={}, targets={"pcagente": 12, "gui360": 12})
    for f in ("train.jsonl", "validation.jsonl", "manifest.json", "stats.json",
              "source_stats.json", "decontamination_report.json"):
        assert os.path.exists(os.path.join(root, "final", f)), f
    assert stats["fatal_failure"] is False
    assert stats["train_samples"] + stats["validation_samples"] == stats["total_samples"]


def test_final_jsonl_is_valid_and_portable(synthetic_dataset, tmp_path):
    root = str(tmp_path)
    finalize(root, assign_splits(synthetic_dataset, 3.0))
    for split in ("train", "validation"):
        path = os.path.join(root, "final", f"{split}.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                assert isinstance(obj["messages"], list)
                for p in obj["images"]:
                    assert "\\" not in p and not p.startswith(("/", "C:"))


def test_stats_contents(synthetic_dataset, tmp_path):
    stats = compute_stats(synthetic_dataset, str(tmp_path))
    for key in ("total_samples", "samples_per_source", "single_step_count",
                "short_window_count", "reasoning_percentage",
                "action_verb_distribution", "average_estimated_tokens",
                "total_image_bytes"):
        assert key in stats


def test_fatal_failure_on_missing_image(synthetic_dataset, tmp_path):
    root = str(tmp_path)
    samples = assign_splits(synthetic_dataset, 3.0)
    samples[0]["images"] = ["images/pcagente/ghost.webp"]
    stats = finalize(root, samples)
    assert stats["fatal_failure"] is True
    assert stats["failures"]["missing_image"] >= 1


def test_fatal_failure_on_out_of_bounds_coordinate(synthetic_dataset, tmp_path):
    root = str(tmp_path)
    samples = assign_splits(synthetic_dataset, 3.0)
    victim = next(s for s in samples if s["task_type"] == "action")
    victim["messages"][-1]["content"] = "click(x=99999, y=99999)"
    stats = finalize(root, samples)
    assert stats["fatal_failure"] is True
    assert stats["failures"]["coordinate_out_of_bounds"] >= 1


def test_fatal_failure_on_empty_target(synthetic_dataset, tmp_path):
    root = str(tmp_path)
    samples = assign_splits(synthetic_dataset, 3.0)
    samples[0]["messages"][-1]["content"] = ""
    stats = finalize(root, samples)
    assert stats["fatal_failure"] is True


def test_zero_trajectory_overlap_in_final_output(synthetic_dataset, tmp_path):
    root = str(tmp_path)
    finalize(root, assign_splits(synthetic_dataset, 3.0))
    from processing.splitting import trajectory_overlap
    train = [json.loads(l) for l in open(os.path.join(root, "final", "train.jsonl"), encoding="utf-8")]
    val = [json.loads(l) for l in open(os.path.join(root, "final", "validation.jsonl"), encoding="utf-8")]
    assert not trajectory_overlap(train, val)


def test_quality_audit_runs(synthetic_dataset, tmp_path):
    audit = quality_audit(assign_splits(synthetic_dataset, 3.0), str(tmp_path), n=10)
    assert audit is not None
    assert audit["inspected"] == 10
    assert set(audit["classification"]) <= {"good", "questionable", "bad"}
    assert audit["classification"].get("bad", 0) == 0


def test_text_only_replay_sample_final_schema(tmp_path):
    ctx = make_ctx(tmp_path, quota={"replay": 1})
    from processing.assemble import assemble_replay
    sample = assemble_replay(
        messages=[{"role": "user", "content": "Q"},
                  {"role": "assistant", "content": "A"}],
        images_pil=[], source_name="replay", sample_id="replay_math_x",
        task_type="replay_math", metadata={}, ctx=ctx)
    sample["split"] = "train"
    ok, reason = validate_sample(sample, str(tmp_path))
    assert ok, reason
