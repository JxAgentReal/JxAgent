import io

import pytest
from PIL import Image

from processing.images import (DEFAULT_MAX_LONG, GROUNDING_MAX_LONG,
                               compute_target_long_side, load_image, png_dimensions,
                               process_image, resize_dimensions)
from processing.coordinates import scale_point
from tests.conftest import make_png_bytes


def test_resize_never_upscales():
    img = Image.new("RGB", (800, 600))
    p = process_image(img)
    assert (p.width, p.height) == (800, 600)


def test_aspect_ratio_preserved():
    p = process_image(load_image(make_png_bytes(1920, 1080)))
    assert max(p.width, p.height) == 1600
    assert abs(p.width / p.height - 1920 / 1080) < 0.01


def test_webp_quality_and_format():
    p = process_image(load_image(make_png_bytes(640, 480)), quality=80)
    assert p.format == "WEBP"
    img = Image.open(io.BytesIO(p.data))
    assert img.format == "WEBP"
    assert len(p.data) < len(make_png_bytes(640, 480))


def test_small_target_preserves_resolution_up_to_1920():
    size = (2560, 1440)
    # target would become < 11px at 1600 -> keep up to 1920
    long_side = compute_target_long_side(size, target_width_px=15)
    assert long_side == 1878
    # large target shrinks normally
    assert compute_target_long_side(size, target_width_px=100) == 1600
    # no target info -> default
    assert compute_target_long_side(size) == 1600
    # never beyond source
    assert compute_target_long_side((1600, 900), target_width_px=2) == 1600


def test_coordinates_track_resize():
    original = (1920, 1080)
    img = load_image(make_png_bytes(*original))
    p = process_image(img)
    assert p.was_resized
    x, y = scale_point(960, 540, original, (p.width, p.height))
    assert (x, y) == (800, 450)
    # every valid original point stays in bounds after scaling
    for ox, oy in [(0, 0), (1, 1), (1919, 1079), (960, 540)]:
        nx, ny = scale_point(ox, oy, original, (p.width, p.height))
        assert 0 <= nx < p.width and 0 <= ny < p.height


def test_png_header_dimensions():
    data = make_png_bytes(123, 45)
    assert png_dimensions(data) == (123, 45)
    assert png_dimensions(b"\x89PNG\r\n\x1a\n" + b"\x00" * 30) is None


def test_resize_dimensions_pairs():
    assert resize_dimensions(1920, 1080, 1600) == (1600, 900)
    assert resize_dimensions(1080, 1920, 1600) == (900, 1600)
    assert resize_dimensions(1600, 900, 1600) == (1600, 900)
    assert resize_dimensions(100, 50, 1600) == (100, 50)
