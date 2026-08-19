import json

import pytest

from sources.gui360 import (app_from_id, parse_messages, target_rect_pixels,
                            unfold, use_row_to_trajectory)
from tests.conftest import make_png_bytes


def gui360_use_row(n_steps=3, row_id="excel_1_105", task="Save the file as read-only."):
    images = [{"bytes": make_png_bytes(1040, 736, marker=i), "path": f"{i}.jpg"}
              for i in range(n_steps)]
    messages = []
    for i in range(n_steps):
        user = {"role": "user", "content": (
            [{"type": "image", "index": i}] if i == 0 else [{"type": "image", "index": i}])}
        if i == 0:
            user["content"].append({"type": "text", "text": task})
        coord = [40 + i * 100, 70 + i * 50]
        messages.append(user)
        messages.append({"role": "assistant",
                         "content": [{"type": "inline_reasoning", "text": "verbose..."},
                                     {"type": "action_description", "text": "verbose..."}],
                         "tool_calls": [{"type": "function", "function": {
                             "name": "click", "arguments": {"coordinate": coord}}}]})
    return {
        "images": images,
        "messages": json.dumps(messages),
        "metadata": json.dumps({"platform": "desktop", "task_type": "use",
                                "others": {"id": row_id, "resolution": [1040, 736],
                                           "os": "windows", "source": "vyokky/GUI-360"}}),
    }


def test_use_row_to_trajectory_steps_and_coords():
    traj = use_row_to_trajectory(gui360_use_row(n_steps=3))
    assert traj is not None
    assert len(traj.steps) == 3
    assert traj.task.startswith("Save")
    # [0,1000] normalized coordinate converted to 1040x736 pixels
    x, y = traj.steps[0].action.points[0]
    assert (x, y) == (42, 51)  # round(41.6)=42? -> 40/1000*1040 = 41.6 -> 42
    assert traj.app == "excel"
    # verbose reasoning is NOT carried into the training target
    assert "inline_reasoning" not in traj.steps[0].action_text


def test_use_row_invalid_json_messages():
    row = gui360_use_row()
    row["messages"] = "{broken"
    assert use_row_to_trajectory(row) is None


def test_unfold_folded_row():
    member = {"messages": json.dumps([
        {"role": "user", "content": [{"type": "image", "index": 0},
                                     {"type": "text", "text": "intent"}]},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {
            "name": "point", "arguments": {"coordinate": [606, 327]}}}]}]),
        "metadata": "{}"}
    row = {"images": [{"bytes": make_png_bytes(1040, 736), "path": "x.jpg"}],
           "messages": "[]", "metadata": "{}",
           "_folded": json.dumps([member])}
    members = unfold(row)
    assert len(members) == 1
    m, img = members[0]
    assert img is not None
    msgs = parse_messages(m)
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "point"


def test_unfold_passthrough_use_row():
    row = gui360_use_row(n_steps=1)
    out = unfold(row)
    assert len(out) == 1


def test_target_rect_pixels_norm_1000():
    # [60, 960, 91, 989] on 1040x736 -> pixel rect within bounds
    rect = target_rect_pixels([60, 960, 91, 989], 1040, 736)
    assert rect == (62, 706, 95, 727)
    # full-range rect clamps to the last valid pixel (boxes must stay in bounds)
    assert target_rect_pixels([0, 0, 1000, 1000], 1040, 736) == (0, 0, 1039, 735)
    # invalid rect (x2 <= x1)
    assert target_rect_pixels([500, 100, 100, 200], 1040, 736) is None


def test_app_from_id_patterns():
    assert app_from_id("excel_1_105") == "excel"
    assert app_from_id("excel_4s_1_2") == "excel"
    assert app_from_id("win_word_9") == "win"
    assert app_from_id("") == "office"


def test_recovery_signal_when_frames_identical():
    # same screenshot twice -> second step flagged no_state_change
    row = gui360_use_row(n_steps=2)
    row["images"][1] = {"bytes": row["images"][0]["bytes"], "path": "dup.jpg"}
    traj = use_row_to_trajectory(row)
    assert "no_state_change" in traj.steps[1].signals
