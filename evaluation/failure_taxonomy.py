#!/usr/bin/env python3
"""Failure taxonomy (14 categories) with mechanical-only auto assignment.

Rules:
  - only categories provable from harness data are assigned automatically;
  - subjective reasoning failures are NEVER guessed: they stay
    failure_annotation="pending" for later manual annotation;
  - harness-caused, environment-caused and model-caused failures are
    reported separately so model-caused rates are never polluted.
"""
from __future__ import annotations

from typing import List, Optional

# harness / environment side
TOOL_HARNESS_ERROR = "tool_harness_error"
ENVIRONMENT_ERROR = "environment_error"

# model-caused, mechanically detectable combinations
PREMATURE_FINISH = "premature_finish"
FAILURE_TO_FINISH = "failure_to_finish"

# model-caused, require human judgment (manual annotation only)
GROUNDING_ERROR = "grounding_error"
TYPING_ERROR = "typing_error"
NAVIGATION_ERROR = "navigation_error"
WRONG_ACTION = "wrong_action"
PLANNING_ERROR = "planning_error"
RECOVERY_FAILURE = "recovery_failure"
STATE_MISUNDERSTANDING = "state_misunderstanding"
WAIT_TIMING_ERROR = "wait_timing_error"
VERIFICATION_FAILURE = "verification_failure"
LONG_HORIZON_CONTEXT_FAILURE = "long_horizon_context_failure"

ALL_CATEGORIES = [
    TOOL_HARNESS_ERROR, ENVIRONMENT_ERROR, GROUNDING_ERROR, TYPING_ERROR,
    NAVIGATION_ERROR, WRONG_ACTION, PLANNING_ERROR, RECOVERY_FAILURE,
    STATE_MISUNDERSTANDING, WAIT_TIMING_ERROR, VERIFICATION_FAILURE,
    PREMATURE_FINISH, FAILURE_TO_FINISH, LONG_HORIZON_CONTEXT_FAILURE,
]

MANUAL_ONLY_CATEGORIES = [
    GROUNDING_ERROR, TYPING_ERROR, NAVIGATION_ERROR, WRONG_ACTION,
    PLANNING_ERROR, RECOVERY_FAILURE, STATE_MISUNDERSTANDING,
    WAIT_TIMING_ERROR, VERIFICATION_FAILURE, LONG_HORIZON_CONTEXT_FAILURE,
]

NEEDS_MANUAL = "needs_manual_annotation"

HARNESS_CAUSED = "harness_caused"
ENVIRONMENT_CAUSED = "environment_caused"
MODEL_CAUSED = "model_caused"
UNATTRIBUTED = "unattributed"

_CAUSED_BY = {
    TOOL_HARNESS_ERROR: HARNESS_CAUSED,
    ENVIRONMENT_ERROR: ENVIRONMENT_CAUSED,
    PREMATURE_FINISH: MODEL_CAUSED,
    FAILURE_TO_FINISH: MODEL_CAUSED,
    **{c: MODEL_CAUSED for c in MANUAL_ONLY_CATEGORIES},
    NEEDS_MANUAL: UNATTRIBUTED,
}


def caused_by(category: Optional[str]) -> str:
    if category is None:
        return UNATTRIBUTED
    return _CAUSED_BY.get(category, UNATTRIBUTED)


def auto_category(status: str, finished: bool,
                  env_success_at_finish: Optional[bool],
                  env_success_at_budget: Optional[bool] = None) -> Optional[str]:
    """Assign ONLY mechanically provable categories. Returns None when the
    failure is a plain model failure with no mechanical signal (manual pass).

    Mechanical rules:
      parser/harness failure            -> tool_harness_error
      environment failure               -> environment_error
      finish emitted, scorer says fail  -> premature_finish  (mechanical)
      budget end, no finish, scorer says
      goal state was achieved           -> failure_to_finish (mechanical)
    """
    if status in ("parser_failure", "harness_failure"):
        return TOOL_HARNESS_ERROR
    if status == "environment_failure":
        return ENVIRONMENT_ERROR
    if status in ("model_failure", "timeout"):
        if finished and env_success_at_finish is False:
            return PREMATURE_FINISH
        if not finished and env_success_at_budget is True:
            return FAILURE_TO_FINISH
    return None


def apply_manual_annotations(results: List[dict],
                             annotations: dict) -> List[dict]:
    """Merge a manual {task_id: {"category": ..., "note": ...}} map into
    result dicts. Refuses invalid categories; only pending results may be
    re-annotated (existing manual labels are kept unless overwrite=True)."""
    out = []
    for r in results:
        r = dict(r)
        ann = annotations.get(r.get("task_id"))
        if ann:
            cat = ann.get("category")
            if cat not in ALL_CATEGORIES:
                raise ValueError(f"invalid failure category for {r.get('task_id')}: {cat!r}")
            current = r.get("failure_annotation", "none")
            if current == "pending" or ann.get("overwrite", False):
                r["failure_category"] = cat
                r["failure_annotation"] = f"manual:{cat}"
                if ann.get("note"):
                    r["annotation_note"] = ann["note"]
        out.append(r)
    return out


def failure_breakdown(results: List[dict]) -> dict:
    """Counts by category and by attribution (harness/env/model)."""
    by_cat: dict = {}
    by_cause = {HARNESS_CAUSED: 0, ENVIRONMENT_CAUSED: 0, MODEL_CAUSED: 0,
                UNATTRIBUTED: 0}
    pending = 0
    for r in results:
        if r.get("status") == "success":
            continue
        cat = r.get("failure_category") or NEEDS_MANUAL
        if r.get("failure_annotation") == "pending" and cat == NEEDS_MANUAL:
            pending += 1
        by_cat[cat] = by_cat.get(cat, 0) + 1
        by_cause[caused_by(cat)] += 1
    return {"by_category": by_cat, "by_attribution": by_cause,
            "pending_manual_annotation": pending}
