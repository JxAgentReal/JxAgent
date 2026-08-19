"""Failure taxonomy: mechanical-only assignment, manual annotation, split
attribution (harness vs environment vs model caused)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import failure_taxonomy as ftx


def test_all_fourteen_categories_present():
    assert len(ftx.ALL_CATEGORIES) == 14
    for c in ["grounding_error", "typing_error", "planning_error",
              "recovery_failure", "verification_failure", "premature_finish",
              "failure_to_finish", "navigation_error", "wait_timing_error",
              "state_misunderstanding", "long_horizon_context_failure",
              "tool_harness_error", "environment_error", "wrong_action"]:
        assert c in ftx.ALL_CATEGORIES


def test_mechanical_assignment_only():
    assert ftx.auto_category("parser_failure", False, None) == "tool_harness_error"
    assert ftx.auto_category("harness_failure", False, None) == "tool_harness_error"
    assert ftx.auto_category("environment_failure", False, None) == "environment_error"
    # premature finish: finish emitted but scorer says fail
    assert ftx.auto_category("model_failure", True, False) == "premature_finish"
    # failure to finish: budget ended, no finish, scorer says achieved
    assert ftx.auto_category("timeout", False, None, env_success_at_budget=True) == \
        "failure_to_finish"
    # plain model failure with no mechanical signal: NOT guessed
    assert ftx.auto_category("model_failure", False, False, False) is None
    assert ftx.auto_category("timeout", False, None, False) is None


def test_subjective_never_auto_assigned():
    for status in ("model_failure", "timeout"):
        assert ftx.auto_category(status, False, False, False) is None


def test_manual_annotation_merge_and_validation():
    results = [{"task_id": "t1", "status": "model_failure",
                "failure_category": None, "failure_annotation": "pending"},
               {"task_id": "t2", "status": "model_failure",
                "failure_category": "premature_finish",
                "failure_annotation": "manual:premature_finish"}]
    out = ftx.apply_manual_annotations(
        results, {"t1": {"category": "grounding_error", "note": "clicked 90px off"},
                  "t2": {"category": "planning_error"}})  # t2 not overwritten
    assert out[0]["failure_category"] == "grounding_error"
    assert out[0]["failure_annotation"] == "manual:grounding_error"
    assert out[1]["failure_category"] == "premature_finish"  # kept
    with pytest.raises(ValueError):
        ftx.apply_manual_annotations(results, {"t1": {"category": "not_a_category"}})


def test_attribution_split():
    results = [
        {"task_id": "a", "status": "success"},
        {"task_id": "b", "status": "parser_failure",
         "failure_category": "tool_harness_error", "failure_annotation": "none"},
        {"task_id": "c", "status": "environment_failure",
         "failure_category": "environment_error", "failure_annotation": "none"},
        {"task_id": "d", "status": "model_failure",
         "failure_category": "grounding_error", "failure_annotation": "manual:grounding_error"},
        {"task_id": "e", "status": "model_failure",
         "failure_category": None, "failure_annotation": "pending"},
    ]
    breakdown = ftx.failure_breakdown(results)
    assert breakdown["by_attribution"][ftx.HARNESS_CAUSED] == 1
    assert breakdown["by_attribution"][ftx.ENVIRONMENT_CAUSED] == 1
    assert breakdown["by_attribution"][ftx.MODEL_CAUSED] == 1
    assert breakdown["pending_manual_annotation"] == 1
