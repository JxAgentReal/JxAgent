"""Result denominator accounting tests: nothing disappears silently."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import scoring as sc


def R(tid, status):
    return sc.TaskResult(task_id=tid, status=status)


def test_strict_denominator_counts_missing_and_errors():
    results = [R("t1", sc.STATUS_SUCCESS), R("t2", sc.STATUS_HARNESS_FAILURE),
               R("t3", sc.STATUS_ENVIRONMENT_FAILURE)]
    agg = sc.aggregate(results, ["t1", "t2", "t3", "t4"])  # t4 never attempted
    assert agg["expected_task_count"] == 4
    assert agg["successes"] == 1
    assert agg["strict_success_rate"] == 0.25          # 1/4, failures count
    assert agg["status_counts"][sc.STATUS_MISSING] == 1
    assert agg["accounting_complete"] is False


def test_protocol_denominator_excludes_specified_statuses():
    results = [R("t1", sc.STATUS_SUCCESS), R("t2", sc.STATUS_SUCCESS),
               R("t3", sc.STATUS_INVALID_TASK), R("t4", sc.STATUS_ENVIRONMENT_FAILURE),
               R("t5", sc.STATUS_MODEL_FAILURE)]
    agg = sc.aggregate(results, ["t1", "t2", "t3", "t4", "t5"])
    assert agg["strict_success_rate"] == 0.4           # 2/5
    assert agg["protocol_denominator"] == 3            # excludes t3, t4
    assert agg["protocol_success_rate"] == pytest.approx(2 / 3)
    assert "denominator_explanation" in agg


def test_unexpected_results_flagged():
    results = [R("t1", sc.STATUS_SUCCESS), R("ghost", sc.STATUS_SUCCESS)]
    agg = sc.aggregate(results, ["t1"])
    assert agg["unexpected_task_ids"] == ["ghost"]
    assert agg["accounting_complete"] is False


def test_task_result_roundtrip_with_extra(tmp_path):
    r = sc.TaskResult(task_id="chrome/x", status=sc.STATUS_MODEL_FAILURE,
                      failure_category="premature_finish",
                      extra={"env_success_at_budget": True, "arm": "adapter"})
    path = sc.save_task_result(str(tmp_path), r)
    loaded = sc.load_task_results(str(tmp_path))
    assert len(loaded) == 1
    assert loaded[0].task_id == "chrome/x"
    assert loaded[0].extra["env_success_at_budget"] is True
    assert loaded[0].status == sc.STATUS_MODEL_FAILURE


def test_atomic_write_no_partial_files(tmp_path):
    r = R("t1", sc.STATUS_SUCCESS)
    sc.save_task_result(str(tmp_path), r)
    files = os.listdir(os.path.join(str(tmp_path), "tasks"))
    assert files == ["t1.json"]  # no .tmp leftovers


def test_empty_task_list_safe():
    agg = sc.aggregate([], [])
    assert agg["strict_success_rate"] == 0.0
    assert agg["protocol_success_rate"] is None
    assert agg["accounting_complete"] is True
