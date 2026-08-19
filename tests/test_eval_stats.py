"""Statistical comparison: exact McNemar, bootstrap CI, paired summary."""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.stats import (bootstrap_paired_delta_ci, mcnemar_exact,
                              paired_comparison)


def test_mcnemar_exact_known_values():
    # symmetric discordants: p = 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # no discordants: p = 1.0
    assert mcnemar_exact(0, 0) == 1.0
    # heavily skewed: exact two-sided binomial p(n=10, k<=1)
    expected = 2 * sum(math.comb(10, i) * 0.5 ** 10 for i in range(2))
    assert mcnemar_exact(1, 9) == pytest.approx(expected)
    # extreme: p < 0.05
    assert mcnemar_exact(0, 8) < 0.01


def test_bootstrap_ci_contains_point_estimate():
    base = {f"t{i}": (i % 3 != 0) for i in range(120)}
    adapter = {f"t{i}": (i % 3 != 0 or i % 17 == 0) for i in range(120)}
    summary = paired_comparison(base, adapter, n_boot=2000, seed=7)
    delta = summary["delta_percentage_points"]
    lo, hi = summary["bootstrap_delta_ci_95_pp"]
    assert lo <= delta <= hi or (lo <= delta + 1e-9 and hi >= delta - 1e-9)


def test_paired_summary_counts():
    base = {"a": True, "b": True, "c": False, "d": False}
    adapter = {"a": True, "b": False, "c": True, "d": False}
    s = paired_comparison(base, adapter, n_boot=500)
    assert s["paired_wins_adapter"] == 1      # c
    assert s["paired_wins_base"] == 1         # b
    assert s["ties"] == 2                     # a, d
    assert s["discordant"] == 2
    assert s["delta_percentage_points"] == 0.0
    assert s["mcnemar_exact_p"] == 1.0
    assert s["likely_meaningful"] is False    # tiny delta never "meaningful"


def test_shared_tasks_only():
    base = {"a": True, "b": True}
    adapter = {"a": False, "z": True}
    s = paired_comparison(base, adapter, n_boot=100)
    assert s["shared_tasks"] == 1
    assert s["missing_in_base"] == ["z"]
    assert s["missing_in_adapter"] == ["b"]


def test_clear_win_flagged_meaningful():
    base = {f"t{i}": False for i in range(100)}
    adapter = {f"t{i}": True for i in range(100)}
    s = paired_comparison(base, adapter, n_boot=1000)
    assert s["delta_percentage_points"] == 100.0
    assert s["mcnemar_exact_p"] < 0.001
    assert s["likely_meaningful"] is True
    ci = s["bootstrap_delta_ci_95_pp"]
    assert ci[0] > 0  # CI excludes zero


def test_noise_scale_guard_present():
    s = paired_comparison({"a": True}, {"a": False}, n_boot=10)
    assert "interpretation_guard" in s
    assert "not" in s["interpretation_guard"].lower()
