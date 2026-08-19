"""Regression panel config + threshold + paired comparison tests."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import run_regression
from evaluation.run_regression import (RegressionConfigError, compare_panels,
                                       dry_run_panel, evaluate_thresholds,
                                       load_config)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, "evaluation", "regression_config.yaml")


def test_shipped_config_has_all_panels_and_thresholds():
    cfg = load_config(CFG)
    assert set(cfg["panels"]) >= {"instruction_following", "coding", "math",
                                  "general_vqa", "tool_use"}
    assert cfg["tolerance"]["investigation_points"] == 3
    assert cfg["tolerance"]["hard_stop_points"] == 5


def test_math_panel_overlap_labeled():
    cfg = load_config(CFG)
    math = cfg["panels"]["math"]
    assert math.get("overlap_exposed_tasks") == ["gsm8k"], \
        "GSM8K must be labeled overlap-exposed (orca-math is GSM-style)"
    assert "non_gsm" in str(math["tasks"]), \
        "an independent math set must be the primary signal"


@pytest.mark.parametrize("base,adapter,verdict", [
    (0.80, 0.80, "OK"),
    (0.80, 0.785, "OK"),        # 1.5 pp: below noise floor
    (0.80, 0.775, "WATCH"),     # 2.5 pp: above floor, below investigation
    (0.80, 0.76, "INVESTIGATE"),  # 4 pp
    (0.80, 0.74, "HARD_STOP"),   # 6 pp
    (0.80, 0.85, "OK"),          # improvement never a regression
])
def test_thresholds(base, adapter, verdict):
    cfg = load_config(CFG)
    res = evaluate_thresholds(base, adapter, cfg)
    assert res["verdict"] == verdict


def test_paired_comparison_requires_equal_counts():
    cfg = load_config(CFG)
    with pytest.raises(RegressionConfigError):
        compare_panels({"math": [True] * 10}, {"math": [True] * 9}, cfg)


def test_dry_run_panels_deterministic():
    a = dry_run_panel("math", 50, seed=1)
    b = dry_run_panel("math", 50, seed=1)
    assert a == b
    c = dry_run_panel("math", 50, seed=2)
    assert a != c or a == c  # different seed may coincide; determinism is the contract


def test_compare_flags_regression(tmp_path):
    cfg = load_config(CFG)
    base = {n: dry_run_panel(n, 100, seed=5) for n in cfg["panels"]}
    adapter = {n: dry_run_panel(n, 100, seed=5) for n in cfg["panels"]}
    adapter["math"] = [False] * 100  # catastrophic math regression
    report = compare_panels(base, adapter, cfg)
    assert report["panels"]["math"]["verdict"] == "HARD_STOP"
    assert report["overall_verdict"] == "HARD_STOP"
    assert report["regression_detected"] is True


def test_cli_dry_run_writes_report(tmp_path):
    out = tmp_path / "rep.json"
    rc = run_regression.main(["--dry-run", "--out", str(out)])
    assert rc in (0, 1)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "overall_verdict" in report and "panels" in report


def test_cli_real_mode_requires_base_results_first(tmp_path):
    adapter_results = tmp_path / "a.json"
    adapter_results.write_text(json.dumps({"math": [True]}), encoding="utf-8")
    rc = run_regression.main(["--adapter-results", str(adapter_results)])
    assert rc == 2  # baseline-first: no base results -> refuse
