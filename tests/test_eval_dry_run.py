"""End-to-end offline dry run: config parity, Plan parsing, coordinate
transforms, trajectory logging, result persistence, resume behavior,
scoring, failure accounting, manifest generation, baseline-first comparison.
This is the main validation mode for the harness (no model, no OSWorld)."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import run_agent
from evaluation import scoring as sc


def _run(tmp_path, arm, out_name, tasks=12, base_dir=None, extra=None):
    argv = ["--dry-run", "--arm", arm,
            "--output-dir", str(tmp_path / out_name),
            "--tasks", str(tasks)]
    if base_dir:
        argv += ["--base-run-dir", str(tmp_path / base_dir)]
    if extra:
        argv += extra
    rc = run_agent.main(argv)
    return rc, str(tmp_path / out_name)


def test_dry_run_base_arm_full_flow(tmp_path):
    rc, out = _run(tmp_path, "base", "base")
    assert rc == 0
    # manifest written
    manifest = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    assert manifest["arm"] == "base" and manifest["dry_run"] is True
    assert manifest["adapter"]["merged"] is False
    assert manifest["benchmark"]["name"] == "osworld_verified"
    assert manifest["step_budget"] == 50 and manifest["seed"] == 1337
    # per-task records + aggregate + trajectories
    n_tasks = len(os.listdir(os.path.join(out, "tasks")))
    agg = json.load(open(os.path.join(out, "aggregate.json"), encoding="utf-8"))
    assert agg["accounting_complete"] is True
    assert agg["expected_task_count"] == 12 and n_tasks == 12
    # mixed outcomes exist in the synthetic set: success + failures
    counts = agg["status_counts"]
    assert counts[sc.STATUS_SUCCESS] > 0
    assert (counts[sc.STATUS_MODEL_FAILURE] + counts[sc.STATUS_TIMEOUT]
            + counts[sc.STATUS_INVALID_TASK]) > 0
    # trajectories record plan-preserved steps and transforms
    traj_dir = os.path.join(out, "trajectories")
    traj = [json.loads(l) for l in
            open(os.path.join(traj_dir, os.listdir(traj_dir)[0]), encoding="utf-8")]
    steps = [r for r in traj if r["record"] == "step"]
    assert any(r["observation_transform"]["original_size"] == [1920, 1080]
               for r in steps)
    assert any(r["parsed_plan"] for r in steps)  # Plan text preserved in logs
    assert any(r["raw_model_output"] and "Action:" in r["raw_model_output"]
               for r in steps)


def test_dry_run_adapter_arm_and_comparison(tmp_path):
    _run(tmp_path, "base", "base")
    rc, out = _run(tmp_path, "adapter", "adapter", base_dir="base")
    assert rc == 0
    comparison = json.load(open(os.path.join(out, "comparison.json"),
                                encoding="utf-8"))
    # baseline gate passed (manifests compatible) but dry-run base is not a
    # real baseline -> NOT_COMPARABLE, SOTA never claimed
    assert comparison["baseline_gate"]["comparable"] is True
    assert comparison["claim"]["sota_claim_allowed"] == "NO"
    assert "statistics" in comparison
    assert "published_reference_note" in comparison


def test_resume_skips_completed_and_reruns_missing(tmp_path):
    _run(tmp_path, "base", "base")
    out = str(tmp_path / "base")
    results = sc.load_task_results(out)
    assert len(results) == 12
    # simulate an interrupted run: delete 3 task records
    tasks_dir = os.path.join(out, "tasks")
    for name in sorted(os.listdir(tasks_dir))[:3]:
        os.remove(os.path.join(tasks_dir, name))
    before = json.load(open(os.path.join(out, "aggregate.json"), encoding="utf-8"))
    rc, _ = _run(tmp_path, "base", "base")
    assert rc == 0
    after = sc.load_task_results(out)
    assert len(after) == 12                      # no double counting
    agg2 = json.load(open(os.path.join(out, "aggregate.json"), encoding="utf-8"))
    # deterministic reruns reproduce identical scores
    assert agg2["strict_success_rate"] == before["strict_success_rate"]
    assert agg2["status_counts"] == before["status_counts"]


def test_force_rerun_redoes_terminal_results(tmp_path):
    _run(tmp_path, "base", "base")
    out = str(tmp_path / "base")
    first = sorted(os.listdir(os.path.join(out, "tasks")))[0]
    rc, _ = _run(tmp_path, "base", "base", extra=["--force-rerun"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "tasks", first + ".old"))


def test_adapter_without_checkpoint_info_ok_in_dry_run(tmp_path):
    # dry-run adapter arm has no real adapter; manifest records absence
    _run(tmp_path, "base", "base")
    rc, out = _run(tmp_path, "adapter", "adapter2", base_dir="base")
    assert rc == 0
    manifest = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    assert manifest["arm"] == "adapter"


def test_real_run_refused_without_confirmation(tmp_path):
    with pytest.raises(SystemExit):
        run_agent.main(["--arm", "base", "--output-dir",
                        str(tmp_path / "real"), "--tasks", "2"])


def test_parity_lock_active_in_runner(tmp_path):
    # scaffold parity is asserted before any task runs (would raise on
    # tampered arm config); sanity: normal config passes (no exception above)
    from evaluation.scaffold import load_scaffold, assert_arm_parity
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "evaluation", "osworld_config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        arms = yaml.safe_load(f)["defaults"]["arms"]
    assert_arm_parity(load_scaffold(), arms["base"], arms["adapter"])
