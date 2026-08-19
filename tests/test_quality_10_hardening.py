import json
from pathlib import Path

import pytest
from PIL import Image

from processing.coordinates import (CoordSpace, Point, parse_gui360_tool_call,
                                    parse_pyautogui)
from processing.dedup import DedupIndex, index_from_state, index_to_state
from processing.images import compute_target_long_side
from processing.validation import validate_sample
from sources.gui360 import (grounding_referent, sanitize_understanding_controls,
                            use_row_to_trajectory)
from sources.replay import (_canonical_sample_id, _canonical_tools,
                            _math_difficulty, _tool_row_valid)
from sources.videocua import _normalize_micro_actions, build_trajectory
from tests.conftest import make_png_bytes


def test_procua_triple_quoted_type_is_exact():
    a = parse_pyautogui('pyautogui.typewrite("""Product A""", interval=0.01)', 1920, 1080)
    assert a is not None
    assert a.render() == 'type(text="Product A")'
    b = parse_pyautogui('pyautogui.typewrite("""=B2*C2""")', 1920, 1080)
    assert b is not None and b.args["text"] == "=B2*C2"


def test_procua_multistatement_only_safe_collapses():
    a = parse_pyautogui('pyautogui.keyDown("return")\npyautogui.keyUp("return")', 100, 100)
    assert a is not None and a.render() == 'press(key="return")'
    d = parse_pyautogui('pyautogui.moveTo(10,20)\npyautogui.dragTo(50,60)', 100, 100)
    assert d is not None and d.render().startswith('drag(x1=10, y1=20, x2=50, y2=60')
    assert parse_pyautogui('pyautogui.click(1,2)\npyautogui.click(3,4)', 100, 100) is None
    assert parse_pyautogui('pyautogui.dragTo(50,60)', 100, 100) is None


def test_normalized_endpoints_are_pixel_safe():
    assert Point(0, 0, CoordSpace.NORM_0_1000).to_pixels(1040, 736) == (0, 0)
    assert Point(1000, 1000, CoordSpace.NORM_0_1000).to_pixels(1040, 736) == (1039, 735)


def test_gui360_key_and_drag_schema():
    k = parse_gui360_tool_call("key", {"keys": ["ctrl", "a"]}, 1000, 800)
    assert k is not None and k.render() == 'hotkey("ctrl", "a")'
    assert parse_gui360_tool_call("key", {"keys": ["shift", "left", "down", "shift"]}, 1000, 800) is None
    d = parse_gui360_tool_call("drag", {"start_coordinate": [100, 200], "coordinate": [800, 900]}, 1001, 1001)
    assert d is not None and d.points == [(100, 200), (800, 900)]


def test_gui360_grounding_referent_is_high_precision():
    assert grounding_referent("To proceed, I need to click the 'Yes' button in the dialog") == "Yes button"
    assert grounding_referent("I need to press Enter to confirm the formula input") is None
    assert grounding_referent("Click cell H4 to continue") == "cell H4"


def test_understanding_controls_are_clipped_sorted_and_capped():
    controls = [
        {"control_text": "B", "control_rect": [500, 500, 700, 600]},
        {"control_text": "A", "control_rect": [-2, 100, 100, 150]},
        {"control_text": "bad", "control_rect": [-100, 0, 1, 1]},
    ]
    got = sanitize_understanding_controls(controls)
    assert [x["control_text"] for x in got] == ["A", "B"]
    assert got[0]["control_rect"][0] == 0


def test_video_drag_reconstruction_from_mouse_hold():
    raw = [
        {"action_type": "MOVE_TO", "timestamp": 1.0, "action_params": {"x": 10, "y": 20}},
        {"action_type": "MOUSE_DOWN", "timestamp": 1.0, "action_params": {"text": "Left"}},
        {"action_type": "DRAG_TO", "timestamp": 2.0, "action_params": {"x": 90, "y": 80}},
        {"action_type": "MOUSE_UP", "timestamp": 2.0, "action_params": {"text": "Left"}},
    ]
    got = _normalize_micro_actions(raw)
    drag = next(x for x in got if x["action_type"] == "DRAG_TO")
    assert drag["action_params"]["start_x"] == 10
    assert drag["action_params"]["start_y"] == 20
    assert all(x["action_type"] not in {"MOUSE_DOWN", "MOUSE_UP"} for x in got)


def test_video_dynamic_coordinate_frame_rejects_ambiguous_outlier():
    log = {"task_id": "x", "task_instruction": "Click it", "platform": "app",
           "action_log": [{"action_type": "CLICK", "timestamp": 1.0,
                           "action_params": {"x": 3000, "y": 100}}]}
    frames = {1.0: make_png_bytes(1920, 1080, marker=1)}
    assert build_trajectory(log, frames, None) is None
    # Explicit source metadata makes the mapping provable.
    log["screen_width"], log["screen_height"] = 3840, 2160
    assert build_trajectory(log, frames, None) is not None


def test_replay_id_includes_image_content():
    messages = [{"role": "user", "content": "What is shown?"},
                {"role": "assistant", "content": "A thing"}]
    a = Image.new("RGB", (8, 8), "red")
    b = Image.new("RGB", (8, 8), "blue")
    assert _canonical_sample_id("vqa", messages, [a]) != _canonical_sample_id("vqa", messages, [b])


def test_tool_schema_conflict_and_literal_mismatch_are_rejected():
    tools = [
        {"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}},
        {"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]}}},
    ]
    assert _canonical_tools(tools) is None

    tools = [{"type": "function", "function": {"name": "set_scene", "parameters": {
        "type": "object", "properties": {"lighting_scene": {"type": "string"}},
        "required": ["lighting_scene"]}}}]
    info = _canonical_tools(tools)
    conv = [
        {"from": "human", "value": 'lighting_scene = "dim"'},
        {"from": "gpt", "value": '<tool_call>{"name":"set_scene","arguments":{"lighting_scene":"soft glow"}}</tool_call>'},
    ]
    assert info is not None and not _tool_row_valid(conv, info[1])


def test_math_difficulty_not_all_trivial():
    assert _math_difficulty("Jungkook is 5th. How many are faster?") == "easy"
    assert _math_difficulty("A store discounts a $240 item by 15%, then applies 8% tax. What is the final price and difference from the original?") in {"medium", "medium_hard"}


def test_dedup_supervision_key_includes_visual_state():
    d = DedupIndex(phash_threshold=0)
    assert d.consider(image_phash=1, signals=[], task_text="T", action_text="click(x=1,y=1)")[0] is False
    # Same task/action, different state is legitimate supervision.
    assert d.consider(image_phash=2, signals=[], task_text="T", action_text="click(x=1,y=1)")[0] is False
    # Exact same decision state repeats.
    assert d.consider(image_phash=1, signals=[], task_text="T", action_text="click(x=1,y=1)")[0] is True
    d2 = index_from_state(index_to_state(d))
    assert d2.is_near_duplicate(1)


def test_tiny_target_uses_smaller_dimension_and_correct_formula():
    # 20x5 target on 2560 image: height is the limiting dimension, requiring
    # the maximum 1920 retention rather than using width alone.
    assert compute_target_long_side((2560, 1440), 20, 5) == 1920


def test_validator_rejects_legacy_multi_assistant_action(tmp_path):
    img = tmp_path / "images" / "procua" / "x.webp"
    img.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(img)
    sample = {
        "messages": [
            {"role": "user", "content": "<image>\nTask: x"},
            {"role": "assistant", "content": "click(x=1, y=1)"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "click(x=2, y=2)"},
        ],
        "images": ["images/procua/x.webp"], "source": "procua",
        "trajectory_id": "t", "task_type": "action",
        "metadata": {"final_image_size": [100, 80]},
    }
    ok, reason = validate_sample(sample, str(tmp_path))
    assert not ok and reason == "multi_assistant_action_without_state_alignment"


def test_validator_accepts_spatial_understanding(tmp_path):
    img = tmp_path / "images" / "gui360" / "x.webp"
    img.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(img)
    answer = json.dumps([
        {"control_text": "A", "control_rect": [0, 0, 100, 100]},
        {"control_text": "B", "control_rect": [0, 200, 100, 300]},
    ])
    sample = {"messages": [{"role": "user", "content": "<image>\nList"},
                            {"role": "assistant", "content": answer}],
              "images": ["images/gui360/x.webp"], "source": "gui360",
              "trajectory_id": "u", "task_type": "screen_understanding",
              "metadata": {"final_image_size": [100, 80]}}
    assert validate_sample(sample, str(tmp_path)) == (True, "")


def test_run1_default_config_disables_synthetic_reasoning():
    import build_jxagent_dataset as bj
    assert bj.load_config(None)["reasoning"]["rate"] == 0.0


def test_dedup_near_state_new_action_becomes_canonical_for_future_repeats():
    d = DedupIndex(phash_threshold=1)
    # register state 0b0000 with action A
    assert d.consider(image_phash=0, signals=[], task_text="T", action_text="click(x=1,y=1)")[0] is False
    # near state 0b0001 with new action B is legitimate once
    assert d.consider(image_phash=1, signals=[], task_text="T", action_text="click(x=2,y=2)")[0] is False
    # repeating B on the same near-equivalent visible state must now be rejected
    dup, reason = d.consider(image_phash=1, signals=[], task_text="T", action_text="click(x=2,y=2)")
    assert dup and reason in {"task_action_state_duplicate", "near_duplicate_image_action"}


def test_grounding_bucket_uses_thin_dimension():
    from processing.quality import grounding_bucket, score_grounding
    assert grounding_bucket(200, 5) == "tiny"
    thin = score_grounding(target_width_px=200, target_height_px=5, text="Thin control",
                           category="Button", app="x")
    square = score_grounding(target_width_px=200, target_height_px=200, text="Large control",
                             category="Button", app="x")
    assert thin.components["grounding_difficulty"] > square.components["grounding_difficulty"]


def test_hotkey_no_visible_change_is_not_automatic_failure_evidence():
    from processing.quality import visual_effect_expected
    a = parse_gui360_tool_call("key", {"keys": ["ctrl", "c"]}, 1000, 800)
    assert a is not None and a.verb == "hotkey"
    assert visual_effect_expected(a) is False


def test_video_same_timestamp_does_not_invent_recovery_and_preserves_groundcua_id():
    log = {"task_id": "same", "task_instruction": "Click twice", "platform": "app",
           "action_log": [
               {"action_type": "CLICK", "timestamp": 1.0, "groundcua_id": "g1",
                "action_params": {"x": 10, "y": 10}},
               {"action_type": "CLICK", "timestamp": 1.0, "groundcua_id": "g2",
                "action_params": {"x": 20, "y": 20}},
           ]}
    frame = make_png_bytes(100, 80, marker=1)
    traj = build_trajectory(log, {1.0: frame}, None)
    assert traj is not None and len(traj.steps) == 2
    assert "recovery_evidenced" not in traj.steps[1].signals
    assert traj.steps[0].metadata["groundcua_id"] == "g1"
    assert traj.steps[1].metadata["groundcua_id"] == "g2"


def test_replay_canonical_id_reservation_rejects_duplicate(tmp_path):
    from sources.common import BuildContext
    from processing.state import BuildState
    ctx = BuildContext(dataset_root=str(tmp_path), state=BuildState(str(tmp_path / "state")),
                       config={"context_budget": 8192})
    assert ctx.reserve_replay_id("abc") is True
    assert ctx.reserve_replay_id("abc") is False
    assert ctx.rejected["replay"]["duplicate_canonical_id"] == 1


def test_recursive_tool_schema_validation():
    from sources.replay import _json_type_ok
    schema = {"type": "object", "required": ["items"], "additionalProperties": False,
              "properties": {"items": {"type": "array", "minItems": 1,
                  "items": {"type": "object", "required": ["n"],
                            "properties": {"n": {"type": "integer", "minimum": 1}}}}}}
    assert _json_type_ok({"items": [{"n": 2}]}, schema)
    assert not _json_type_ok({"items": [{"n": 0}]}, schema)
    assert not _json_type_ok({"items": [{"n": "2"}]}, schema)


def test_no_state_change_alone_is_not_recovery_quality_boost():
    from processing.quality import score_action_step
    plain = score_action_step(verb="click", task_text="Open settings", signals={"no_state_change"})
    explicit = score_action_step(verb="click", task_text="Open settings", signals={"no_state_change", "recovery_evidenced"})
    assert plain.components["recovery"] == 0.0
    assert explicit.components["recovery"] == 1.0


def test_pc_agent_recovery_requires_explicit_failure_language():
    from processing.coordinates import Action
    from sources.pc_agent_e import extract_signals
    action = Action("click", points=[(1, 1)])
    ordinary = extract_signals("Go back to settings", "Click the Back button", action, True)
    assert "no_state_change" in ordinary
    assert "recovery_evidenced" not in ordinary
    failed = extract_signals("Open settings", "The previous click did not work, try again", action, True)
    assert "recovery_evidenced" in failed


def test_windows_never_cross_parser_continuity_gap():
    from processing.coordinates import Action
    from processing.windows import Step, Trajectory, build_single, window_is_informative
    img = make_png_bytes(100, 80, marker=1)
    steps = [
        Step("a", img, (100,80), Action("click", points=[(1,1)]), phash=1, prev_phash=None, metadata={"continuity_id":0}),
        Step("b", img, (100,80), Action("click", points=[(2,2)]), phash=2, prev_phash=1, metadata={"continuity_id":0}),
        Step("c", img, (100,80), Action("click", points=[(3,3)]), phash=7, prev_phash=None, metadata={"continuity_id":1}),
    ]
    traj = Trajectory("t", "task", steps=steps, source="procua")
    assert not window_is_informative(steps)
    single = build_single(traj, 2)
    assert single.history_texts == []


def test_window_signals_belong_to_anchor_not_earlier_history():
    from processing.coordinates import Action
    from processing.windows import Step, Trajectory, build_window
    img = make_png_bytes(100, 80, marker=1)
    steps = [
        Step("a", img, (100,80), Action("click", points=[(1,1)]), signals={"recovery_evidenced"}, metadata={"continuity_id":0}),
        Step("b", img, (100,80), Action("click", points=[(2,2)]), signals=set(), metadata={"continuity_id":0}),
        Step("c", img, (100,80), Action("click", points=[(3,3)]), signals={"save"}, metadata={"continuity_id":0}),
    ]
    traj = Trajectory("t", "task", steps=steps, source="procua")
    spec = build_window(traj, 0, 3)
    assert spec.signals == {"save"}
