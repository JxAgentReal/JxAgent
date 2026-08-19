"""Canonical Plan/Action parser tests (audit H1 fix).

The parser must accept BOTH bare actions and 'Plan: ...\\nAction: ...'
(the format ~12% of training targets use), reject malformed envelopes,
preserve the Plan text, and be the SAME parser validate_actions uses.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.action_parser import parse_model_output
from evaluation.validate_actions import check_text


def test_plain_action():
    p = parse_model_output("click(x=100, y=200)")
    assert p.ok and p.plan is None
    assert p.action.verb == "click"
    assert p.action.points == [(100, 200)]


def test_plan_plus_action():
    p = parse_model_output("Plan: The screen is unchanged, switching approach.\n"
                           "Action: click(x=100, y=200)")
    assert p.ok
    assert p.plan == "The screen is unchanged, switching approach."
    assert p.action.verb == "click"
    assert p.action_text == "click(x=100, y=200)"


def test_multiline_plan():
    p = parse_model_output("Plan: First wait for the dialog.\n"
                           "It may take a moment to load.\n"
                           "Action: wait(seconds=2)")
    assert p.ok
    assert "dialog" in p.plan and "moment" in p.plan
    assert p.action.verb == "wait"


def test_malformed_plan_no_action():
    p = parse_model_output("Plan: only reasoning, never acts")
    assert not p.ok and p.error == "missing_action"


def test_two_action_lines_rejected():
    p = parse_model_output("Plan: x\nAction: click(x=1, y=1)\nAction: click(x=2, y=2)")
    assert not p.ok and p.error == "multiple_action_blocks"


def test_trailing_content_rejected():
    p = parse_model_output("Action: click(x=1, y=1)\nSome extra commentary")
    assert not p.ok and p.error == "trailing_content"
    p2 = parse_model_output("Action: click(x=1, y=1)\nPlan: after the fact")
    assert not p2.ok and p2.error == "trailing_content"


def test_unknown_verb():
    p = parse_model_output("Action: teleport(x=1, y=2)")
    assert not p.ok and p.error.startswith("unknown_verb:teleport")


def test_invalid_coordinates():
    p = parse_model_output("Action: click(x=abc, y=)")
    assert not p.ok
    assert p.error in ("unparseable_action",)


def test_finish_and_wait_actions():
    assert parse_model_output('finish(status="success")').action.verb == "finish"
    p = parse_model_output("Plan: Confirm before finishing.\nAction: finish()")
    assert p.ok and p.action.verb == "finish"
    assert parse_model_output("Action: wait(seconds=3)").action.verb == "wait"


def test_empty_and_whitespace():
    assert parse_model_output("").error == "empty_output"
    assert parse_model_output("   \n  ").error == "empty_output"


def test_plan_ignored_for_execution_but_preserved():
    p = parse_model_output("Plan: ANY TEXT, even with (parens) and verbs like click.\n"
                           "Action: scroll(clicks=-5)")
    assert p.ok
    assert p.action.verb == "scroll"
    assert "ANY TEXT" in p.plan


def test_validator_uses_canonical_parser():
    # same acceptance in evaluation/validate_actions.check_text
    assert check_text("Plan: x\nAction: click(x=5, y=5)") == []
    assert check_text("Action: click(x=1, y=1)\nAction: click(x=2, y=2)") == \
        ["multiple_action_blocks"]
    assert check_text("click(x=3000, y=10)", final_size=(1920, 1080)) != []


def test_case_insensitive_labels():
    p = parse_model_output("plan: lower case label\naction: click(x=1, y=2)")
    assert p.ok and p.action.verb == "click"
