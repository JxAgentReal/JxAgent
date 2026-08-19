"""Observation preprocessing parity + coordinate transform tests."""
from __future__ import annotations

import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.preprocessing import preprocess_screenshot, point_roundtrip


@pytest.mark.parametrize("w,h,expect_resize", [
    (1920, 1080, True),   # common OSWorld env resolution
    (1600, 900, False),   # exactly at the cap: never upscale, no-op
    (1280, 720, False),   # smaller than cap: unchanged
    (2000, 1000, True),   # non-16:9-ish 2:1
    (1024, 768, False),   # 4:3, small
    (2878, 1646, True),   # large GroundCUA-style screen
])
def test_resize_policy(w, h, expect_resize):
    img, t = preprocess_screenshot(Image.new("RGB", (w, h)))
    assert t.resized == expect_resize
    ow, oh = t.original_size
    pw, ph = t.processed_size
    if expect_resize:
        assert max(pw, ph) == 1600
        # aspect preserved within a pixel
        assert abs(pw / ph - ow / oh) < 0.01
    else:
        assert (pw, ph) == (w, h)
    assert max(pw, ph) <= 1600  # never upscaled


def test_coordinate_roundtrip_subpixel():
    for w, h in [(1920, 1080), (1600, 900), (1280, 720), (2878, 1646)]:
        _, t = preprocess_screenshot(Image.new("RGB", (w, h)))
        for x, y in [(0, 0), (w - 1, h - 1), (w // 2, h // 2), (123, 456)]:
            rx, ry = point_roundtrip(t, x, y)
            assert abs(rx - x) <= 1 and abs(ry - y) <= 1


def test_env_point_clamped_into_bounds():
    _, t = preprocess_screenshot(Image.new("RGB", (1920, 1080)))
    ex, ey = t.to_env_space(1599, 899)  # max model-space point
    assert 0 <= ex < 1920 and 0 <= ey < 1080
    ex2, ey2 = t.to_env_space(-50, 5000)  # defensive clamp
    assert ex2 == 0 and ey2 == 1079


def test_small_grounding_target_still_representable():
    # an 11 px target at 1920 maps to ~9 px at 1600: still addressable and the
    # inverse map lands within 1 px of the original
    img, t = preprocess_screenshot(Image.new("RGB", (1920, 1080)))
    mx, my = t.to_model_space(960, 540)
    assert 0 <= mx <= 1599 and 0 <= my <= 899
    ex, ey = t.to_env_space(mx, my)
    assert abs(ex - 960) <= 1 and abs(ey - 540) <= 1


def test_transform_log_dict_shape():
    _, t = preprocess_screenshot(Image.new("RGB", (1920, 1080)))
    d = t.as_log_dict()
    assert d["original_size"] == [1920, 1080]
    assert d["processed_size"] == [1600, 900]
    assert abs(d["scale"] - 1600 / 1920) < 1e-9
    assert d["resized"] is True


def test_identical_inputs_identical_outputs():
    a = Image.new("RGB", (1920, 1080), (10, 20, 30))
    b = Image.new("RGB", (1920, 1080), (10, 20, 30))
    ia, ta = preprocess_screenshot(a)
    ib, tb = preprocess_screenshot(b)
    assert ta.as_log_dict() == tb.as_log_dict()
    assert list(ia.getdata()) == list(ib.getdata())
