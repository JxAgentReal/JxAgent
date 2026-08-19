#!/usr/bin/env python3
"""Canonical model-output parser for the JxAgent evaluation harness.

THE one parser used by both the runtime harness and
evaluation/validate_actions.py. Do not implement a second one anywhere.

Accepted forms (whitespace-tolerant):
  1. bare action          click(x=100, y=200)
  2. plan + action        Plan: concise reasoning
                          Action: click(x=100, y=200)
  3. multiline plan       Plan: first line
                          continued on following lines
                          Action: wait(seconds=2)

Rules:
  - the Plan text is preserved for trajectory logs but ignored for execution
  - exactly one Action line is required (0 -> missing_action, >=2 ->
    multiple_action_blocks)
  - no non-empty line may follow the Action line (trailing_content)
  - the action itself must match the canonical rendered-action grammar
    (processing.coordinates._RENDER_RE); unknown verbs are reported as
    unknown_verb, everything else unparseable as unparseable_action
  - coordinate bound checking is NOT done here (it needs the observed image
    size); use processing.coordinates.action_in_bounds separately
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.coordinates import Action, parse_rendered_action  # noqa: E402

_PLAN_RE = re.compile(r"^Plan[ \t]*:(?P<rest>.*)$", re.IGNORECASE)
_ACTION_RE = re.compile(r"^Action[ \t]*:(?P<rest>.*)$", re.IGNORECASE)
_VERB_RE = re.compile(
    r"^(click|double_click|right_click|middle_click|move|point|drag|scroll|type|press|hotkey|"
    r"key_down|key_up|mouse_down|mouse_up|wait|finish)\s*\(")


@dataclass
class ParsedModelOutput:
    plan: Optional[str]          # reasoning text, None when absent
    action: Optional[Action]     # parsed canonical action, None on error
    action_text: Optional[str]   # the raw Action line content
    error: Optional[str]         # None when parseable

    @property
    def ok(self) -> bool:
        return self.error is None and self.action is not None


def parse_model_output(text: str) -> ParsedModelOutput:
    """Parse one model generation into (plan, action). Never raises."""
    raw = (text or "").strip()
    if not raw:
        return ParsedModelOutput(None, None, None, "empty_output")

    lines = raw.splitlines()
    first = lines[0].strip()

    starts_with_plan = _PLAN_RE.match(lines[0]) is not None
    starts_with_action = _ACTION_RE.match(lines[0]) is not None

    if not starts_with_plan and not starts_with_action:
        # bare action form: the whole text must be a single action
        return _finish_plan(None, raw)

    plan_lines: list[str] = []
    action_lines: list[str] = []          # collected Action: line contents
    after_action = False
    trailing = False
    for i, line in enumerate(lines):
        m_a = _ACTION_RE.match(line)
        m_p = _PLAN_RE.match(line)
        if m_a:
            if action_lines:
                return ParsedModelOutput(None, None, None, "multiple_action_blocks")
            action_lines.append(m_a.group("rest").strip())
            after_action = True
        elif m_p:
            if after_action:
                trailing = True  # Plan after Action is arbitrary trailing text
            else:
                plan_lines.append(m_p.group("rest").strip())
        else:
            if after_action and line.strip():
                trailing = True
            elif not after_action:
                plan_lines.append(line.rstrip())
    if trailing:
        return ParsedModelOutput(None, None, None, "trailing_content")
    if not action_lines:
        return ParsedModelOutput(None, None, None, "missing_action")

    plan = "\n".join(plan_lines).strip() if plan_lines else None
    return _finish_plan(plan, action_lines[0])


def _finish_plan(plan: Optional[str], action_text: str) -> ParsedModelOutput:
    action_text = (action_text or "").strip()
    if not action_text:
        return ParsedModelOutput(plan, None, None, "missing_action")
    if not _VERB_RE.match(action_text):
        verb = action_text.split("(", 1)[0].split(" ", 1)[0][:40]
        return ParsedModelOutput(plan, None, action_text, f"unknown_verb:{verb}")
    action = parse_rendered_action(action_text)
    if action is None:
        return ParsedModelOutput(plan, None, action_text, "unparseable_action")
    return ParsedModelOutput(plan, action, action_text, None)
