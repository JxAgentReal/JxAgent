import json
import os

import pytest

from processing.assemble import sanitize_id
from processing.validation import validate_sample


def test_sanitize_id_removes_hostile_characters():
    assert sanitize_id("part_1/cpu-0049--20260320_231802/0004") == "part_1_cpu-0049--20260320_231802_0004"
    assert "\\" not in sanitize_id("a\\b\\c")
    assert sanitize_id("") == "x"


def _sample(image_rel="images/procua/ok.webp", dataset_root=None, create=True):
    if create and dataset_root and image_rel and ".." not in image_rel:
        full = os.path.join(dataset_root, image_rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        from tests.conftest import make_png_bytes
        with open(full, "wb") as f:
            f.write(make_png_bytes(64, 64))
    return {
        "messages": [
            {"role": "system", "content": "You are a computer-use agent."},
            {"role": "user", "content": "<image>\nTask: test"},
            {"role": "assistant", "content": "click(x=10, y=10)"},
        ],
        "images": [image_rel],
        "source": "procua",
        "trajectory_id": "t",
        "step_id": "s",
        "task_type": "action",
        "metadata": {"final_image_size": [64, 64]},
    }


def test_posix_paths_accepted(tmp_path):
    s = _sample(dataset_root=str(tmp_path))
    ok, reason = validate_sample(s, str(tmp_path))
    assert ok, reason


def test_windows_backslash_rejected(tmp_path):
    s = _sample(image_rel="images\\procua\\bad.webp", dataset_root=None, create=False)
    ok, reason = validate_sample(s, str(tmp_path))
    assert not ok and reason == "non_portable_path"


def test_absolute_path_rejected(tmp_path):
    s = _sample(image_rel="/abs/path.webp", dataset_root=None, create=False)
    ok, reason = validate_sample(s, str(tmp_path))
    assert not ok and reason == "non_portable_path"


def test_parent_traversal_rejected(tmp_path):
    s = _sample(image_rel="images/../../etc/passwd", dataset_root=None, create=False)
    ok, reason = validate_sample(s, str(tmp_path))
    assert not ok


def test_missing_image_rejected(tmp_path):
    s = _sample(image_rel="images/procua/absent.webp", dataset_root=None, create=False)
    ok, reason = validate_sample(s, str(tmp_path))
    assert reason == "missing_image"


def test_placeholder_count_must_match_images(tmp_path):
    s = _sample(dataset_root=str(tmp_path))
    s["messages"][1]["content"] = "<image>\n<image>\nTask: x"
    ok, reason = validate_sample(s, str(tmp_path))
    assert not ok and reason == "image_placeholder_mismatch"


def test_jsonl_output_uses_posix_paths(tmp_path):
    s = _sample(dataset_root=str(tmp_path))
    line = json.dumps(s, ensure_ascii=False)
    assert "\\\\" not in line.replace("\\n", "")  # no escaped backslashes
    parsed = json.loads(line)
    assert all("\\" not in p for p in parsed["images"])
