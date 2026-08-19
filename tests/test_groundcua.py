import pytest

from processing.assemble import assemble_grounding
from processing.quality import GROUNDCUA_SIZE_BUCKETS, grounding_bucket
from sources.groundcua import GIANT_PX, select_element
from tests.conftest import make_ctx, make_png_bytes


def entry(text="OK", x=100, y=100, w=30, h=16, cat="Button"):
    return {"image_path": "p/x.png", "bbox": [x, y, x + w, y + h],
            "text": text, "category": cat, "id": "e1"}


def test_select_element_prefers_small_informative():
    entries = [entry(text="giant button", w=600, h=200),
               entry(text="tiny gear", w=12, h=12, cat="Visual Elements"),
               entry(text="cursor", w=8, h=8, cat="Others")]
    chosen = select_element(entries)
    assert chosen["text"] == "tiny gear"


def test_select_element_skips_giant_buttons():
    entries = [entry(text="huge", w=GIANT_PX + 1, h=100),
               entry(text="ok", w=40, h=20)]
    assert select_element(entries)["text"] == "ok"


def test_select_element_skips_low_info():
    assert select_element([entry(text="cursor", w=10, h=10)]) is None
    assert select_element([entry(text="", w=10, h=10)]) is None


def test_bucket_boundaries_single_definition():
    # one canonical bucket definition lives in processing/quality.py
    # (the contradictory local copy in sources/groundcua.py was removed)
    names = [name for name, _lo, _hi, _share in GROUNDCUA_SIZE_BUCKETS]
    assert names == ["tiny", "small", "medium", "large"]
    assert [share for _n, _lo, _hi, share in GROUNDCUA_SIZE_BUCKETS] == \
        [0.25, 0.40, 0.25, 0.10]
    assert grounding_bucket(8) == "tiny"
    assert grounding_bucket(20) == "small"
    assert grounding_bucket(50) == "medium"
    assert grounding_bucket(500) == "large"


def test_grounding_assembly_small_target_keeps_resolution(tmp_path):
    ctx = make_ctx(tmp_path, quota={"groundcua": 5})
    # 2560x1440 screenshot with a 15px-wide target: must keep up to 1920
    img = make_png_bytes(2560, 1440, marker=5)
    sample = assemble_grounding(
        source="groundcua", trajectory_id="gc_t1", step_id="hash1",
        image_bytes=img, instruction="small gear icon", target_xy=(500, 300),
        image_size=(2560, 1440), target_width_px=15, app="gimp", ctx=ctx)
    assert sample is not None
    w = sample["metadata"]["final_image_size"][0]
    assert w == 1878
    # point rebased into the final space and in bounds
    action = sample["messages"][-1]["content"]
    assert action.startswith("point(x=")
    x = int(action.split("x=")[1].split(",")[0])
    assert 0 <= x < 1920
    assert sample["task_type"] == "grounding"
    assert sample["images"][0].startswith("images/groundcua/")
    assert "\\" not in sample["images"][0]


def test_grounding_assembly_normal_target_resizes(tmp_path):
    ctx = make_ctx(tmp_path, quota={"groundcua": 5})
    img = make_png_bytes(1920, 1080, marker=6)
    sample = assemble_grounding(
        source="groundcua", trajectory_id="gc_t2", step_id="hash2",
        image_bytes=img, instruction="menu item", target_xy=(900, 500),
        image_size=(1920, 1080), target_width_px=140, app="libreoffice", ctx=ctx)
    assert sample["metadata"]["final_image_size"] == [1600, 900]
    action = sample["messages"][-1]["content"]
    x = int(action.split("x=")[1].split(",")[0])
    y = int(action.split("y=")[1].split(")")[0])
    assert (x, y) == (750, 417)  # 900,500 scaled by 1600/1920, 1080->900
