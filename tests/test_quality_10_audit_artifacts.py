"""Large local-source probes from the checked-in audit artifacts.

These do not replace production validation; they regression-test parser and
quality assumptions against hundreds/thousands of *real* source annotations
already bundled with the project audit pack.
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from processing.coordinates import parse_gui360_tool_call, parse_videocua_action
from sources.replay import (_canonical_tools, _instruction_quality_ok,
                            _tool_row_valid)
from sources.videocua import _normalize_micro_actions

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / ".audit"
pytestmark = pytest.mark.skipif(not AUDIT.exists(), reason="local audit artifacts not bundled")


def _walk_tasks(obj):
    out = []
    def walk(x):
        if isinstance(x, dict):
            if "action_log" in x:
                out.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return out


def test_all_audited_videocua_drags_have_proven_start_after_normalization():
    data = json.loads((AUDIT / "videocua_logs.json").read_text(encoding="utf-8"))
    drags = 0
    for task in _walk_tasks(data):
        for entry in _normalize_micro_actions(task.get("action_log") or []):
            if str(entry.get("action_type")).upper() == "DRAG_TO":
                drags += 1
                action = parse_videocua_action(entry, 4096, 2160)
                assert action is not None and len(action.points) == 2
    assert drags >= 7


def test_real_hermes_probe_structural_acceptance_and_conflict_detection():
    rows = json.loads((AUDIT / "replay_rows.json").read_text(encoding="utf-8"))["hermes"]
    valid = conflicts = invalid = 0
    for row in rows:
        info = _canonical_tools(row.get("tools"))
        if info is None:
            conflicts += 1
            continue
        if _tool_row_valid(row.get("conversations") or [], info[1]):
            valid += 1
        else:
            invalid += 1
    assert len(rows) >= 100
    assert conflicts >= 1  # known duplicate-name incompatible-schema row
    assert valid >= 98
    assert invalid <= 1


def test_real_smoltalk_probe_filters_poor_average_and_domain_duplicates():
    rows = json.loads((AUDIT / "replay_rows.json").read_text(encoding="utf-8"))["smoltalk"]
    accepted = 0
    for row in rows:
        msgs = [{"role": m.get("role"), "content": str(m.get("content", ""))}
                for m in row.get("messages", [])
                if m.get("role") in {"system", "user", "assistant"}
                and str(m.get("content", "")).strip()]
        if _instruction_quality_ok(row, msgs):
            accepted += 1
            assert str(row.get("quality")).lower() in {"good", "excellent"}
            assert str(row.get("category")).lower() not in {"coding", "math"}
    assert len(rows) >= 100 and 30 <= accepted <= 80


def test_real_gui360_tool_schema_probe_counts_are_known():
    rows_by_cohort = json.loads((AUDIT / "gui360_rows.json").read_text(encoding="utf-8"))
    rows = rows_by_cohort.get("desktop.use") or []
    if not rows:
        pytest.skip("desktop.use rows absent")
    total_calls = multi_turns = 0
    parseable_first = 0
    assistant_turns = 0
    # Audit rows use the same raw shape as the adapter.  Import lazily so this
    # test remains a source-format assertion, not a mock.
    from sources.gui360 import parse_messages, parse_meta
    for wrapped in rows:
        row = wrapped.get("row", wrapped) if isinstance(wrapped, dict) else wrapped
        meta = parse_meta(row)
        res = (meta.get("others") or {}).get("resolution") or [1040, 736]
        for member in (row.get("data") or row.get("members") or [row]):
            for m in parse_messages(member):
                tcs = m.get("tool_calls") or []
                if m.get("role") != "assistant" or not tcs:
                    continue
                assistant_turns += 1
                total_calls += len(tcs)
                multi_turns += int(len(tcs) > 1)
                fn = tcs[0].get("function") or {}
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception: args = {}
                parseable_first += int(parse_gui360_tool_call(fn.get("name", ""), args, int(res[0]), int(res[1])) is not None)
    assert assistant_turns > 1000
    assert total_calls >= assistant_turns
    assert multi_turns > 100  # real source has batched no-intermediate-frame calls
    assert parseable_first / assistant_turns > 0.95
