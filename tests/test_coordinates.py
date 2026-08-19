import pytest

from processing.coordinates import (Action, BBox, CoordSpace, Point,
                                   action_in_bounds, parse_pyautogui,
                                   parse_pc_agent_e, parse_rendered_action,
                                   parse_videocua_action,
                                   parse_gui360_tool_call, scale_bbox,
                                   scale_point, validate_bbox, validate_point)


def test_pixel_space_passthrough():
    assert Point(100, 200, CoordSpace.PIXEL).to_pixels(1920, 1080) == (100, 200)


def test_norm_0_1_space():
    assert Point(0.5, 0.25, CoordSpace.NORM_0_1).to_pixels(1920, 1080) == (960, 270)


def test_norm_0_1000_gui360_space():
    # GUI-360 convention: [38, 96] on 1040x736 reference
    assert Point(38, 96, CoordSpace.NORM_0_1000).to_pixels(1040, 736) == (39, 71)
    # exact multiples round-trip
    assert Point(500, 500, CoordSpace.NORM_0_1000).to_pixels(1000, 1000) == (500, 500)


def test_bbox_center_and_size():
    b = BBox(10, 20, 110, 220, CoordSpace.PIXEL)
    assert b.center_pixels(1920, 1080) == (60, 120)
    assert b.width_height(1920, 1080) == (100, 200)


def test_scale_point_matches_image_resize():
    # 1920x1080 -> 1600x900
    assert scale_point(960, 540, (1920, 1080), (1600, 900)) == (800, 450)
    assert scale_point(0, 0, (1920, 1080), (1600, 900)) == (0, 0)
    assert scale_point(1919, 1079, (1920, 1080), (1600, 900)) == (1599, 899)


def test_scale_bbox_valid_bounds():
    x1, y1, x2, y2 = scale_bbox([960, 540, 1000, 560], (1920, 1080), (1600, 900))
    assert validate_bbox(x1, y1, x2, y2, 1600, 900)
    # tiny target never collapses to zero size
    x1, y1, x2, y2 = scale_bbox([100, 100, 103, 104], (1920, 1080), (1600, 900))
    assert x2 > x1 and y2 > y1


def test_bounds_validation_rejects_out_of_range():
    assert not validate_point(1600, 450, 1600, 900)
    assert not validate_point(-1, 10, 1600, 900)
    assert validate_point(1599, 899, 1600, 900)
    assert not validate_bbox(100, 100, 90, 200, 1600, 900)


# ------------------------------------------------------------------ parsers

@pytest.mark.parametrize("cmd,verb", [
    ("pyautogui.click(x=512, y=384)", "click"),
    ("pyautogui.click(512, 384)", "click"),
    ("pyautogui.doubleClick(x=10, y=20)", "double_click"),
    ("pyautogui.rightClick(x=10, y=20)", "click"),
    ("pyautogui.moveTo(x=5, y=6)", "move"),
    
    ("pyautogui.scroll(-3)", "scroll"),
    ("pyautogui.scroll(clicks=-3, x=100, y=200)", "scroll"),
    ('pyautogui.typewrite(["hello", "world"])', "type"),
    ('pyautogui.typewrite(text="hello world")', "type"),
    ('pyautogui.press("enter")', "press"),
    ('pyautogui.hotkey("ctrl", "shift", "s")', "hotkey"),
    ("pyautogui.sleep(2.5)", "wait"),
])
def test_parse_pyautogui(cmd, verb):
    a = parse_pyautogui(cmd, 1920, 1080)
    assert a is not None and a.verb == verb, cmd


def test_parse_pyautogui_unknown_returns_none():
    assert parse_pyautogui("os.system('rm -rf /')", 1920, 1080) is None
    assert parse_pyautogui("", 1920, 1080) is None


def test_pc_agent_e_action_strings():
    a = parse_pc_agent_e("click (654, 191)", 1920, 1080)
    assert a.verb == "click" and a.points == [(654, 191)]
    a = parse_pc_agent_e("double click (10, 20)", 1920, 1080)
    assert a.verb == "double_click"
    a = parse_pc_agent_e("right-click (10, 20)", 1920, 1080)
    assert a.verb == "right_click"
    a = parse_pc_agent_e("type hello world", 1920, 1080)
    assert a.verb == "type" and a.args["text"] == "hello world"
    a = parse_pc_agent_e("press ctrl+s", 1920, 1080)
    assert a.verb == "hotkey" and a.args["keys"] == ["ctrl", "s"]
    a = parse_pc_agent_e("scroll down", 1920, 1080)
    assert a.verb == "scroll" and a.args["clicks"] > 0
    a = parse_pc_agent_e("scroll up (100, 200)", 1920, 1080)
    assert a.args["clicks"] < 0 and a.points == [(100, 200)]


def test_videocua_actions():
    click = {"action_type": "CLICK", "action_params": {"x": 47, "y": 242, "numClicks": 2}}
    a = parse_videocua_action(click, 1920, 1080)
    assert a.verb == "double_click" and a.points == [(47, 242)]
    drag = {"action_type": "DRAG_TO", "action_params": {"x": 400, "y": 300, "start_x": 100, "start_y": 100}}
    a = parse_videocua_action(drag, 1920, 1080)
    assert a.verb == "drag" and a.points == [(100, 100), (400, 300)]
    term = {"action_type": "TERMINATE_SUCCESS", "action_params": {}}
    a = parse_videocua_action(term, 1920, 1080)
    assert a.verb == "finish"
    key = {"action_type": "HOTKEY", "action_params": {"keys": ["ctrl", "c"]}}
    a = parse_videocua_action(key, 1920, 1080)
    assert a.verb == "hotkey" and a.args["keys"] == ["ctrl", "c"]
    scroll = {"action_type": "SCROLL", "action_params": {"scrollY": -5, "x": 960, "y": 500}}
    a = parse_videocua_action(scroll, 1920, 1080)
    assert a.verb == "scroll" and a.args["clicks"] == -5


def test_gui360_tool_call_normalized_1000():
    a = parse_gui360_tool_call("click", {"coordinate": [38, 96]}, 1040, 736)
    assert a is not None
    assert a.points == [(39, 71)]
    a = parse_gui360_tool_call("point", {"coordinate": [606, 327]}, 1040, 736)
    assert a.verb == "point" and a.original_space == CoordSpace.NORM_0_1000


# ------------------------------------------------------- render / round-trip

def test_render_parse_round_trip():
    cases = [
        Action("click", points=[(512, 384)]),
        Action("double_click", points=[(10, 20)]),
        Action("point", points=[(606, 327)]),
        Action("drag", points=[(1, 2), (3, 4)], args={"button": "left"}),
        Action("scroll", args={"clicks": -3}, points=[(100, 200)]),
        Action("type", args={"text": 'say "hi"'}),
        Action("press", args={"key": "enter"}),
        Action("hotkey", args={"keys": ["ctrl", "shift", "s"]}),
        Action("wait", args={"seconds": 2.5}),
        Action("finish", args={"status": "success"}),
    ]
    for a in cases:
        rendered = a.render()
        parsed = parse_rendered_action(rendered)
        assert parsed is not None, rendered
        assert parsed.verb == a.verb, rendered
        assert parsed.points == a.points, rendered


def test_render_click_variants():
    assert Action("click", points=[(1, 2)], args={"count": 2}).render().startswith("double_click(")
    assert Action("click", points=[(1, 2)], args={"button": "right"}).render().startswith("right_click(")


def test_action_in_bounds():
    a = Action("click", points=[(1599, 899)])
    assert action_in_bounds(a, 1600, 900)
    a = Action("click", points=[(1600, 0)])
    assert not action_in_bounds(a, 1600, 900)


def test_source_spaces_are_distinct():
    # the same numeric coordinate in different spaces must map differently
    p_pixel = Point(500, 500, CoordSpace.PIXEL).to_pixels(1920, 1080)
    p_norm = Point(500, 500, CoordSpace.NORM_0_1000).to_pixels(1920, 1080)
    p_unit = Point(500, 500, CoordSpace.NORM_0_1).to_pixels(1920, 1080)
    assert len({p_pixel, p_norm, p_unit}) == 3
