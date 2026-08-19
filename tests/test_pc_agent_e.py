import json
import zipfile

import pytest

from sources.pc_agent_e import (build_trajectory, parse_md_description,
                                samples_for_trajectory, task_files_from_zip)
from tests.conftest import make_ctx, make_png_bytes


def make_events(n=3, size=(1920, 1080)):
    """events + screenshots mirroring the verified zip layout."""
    shots = {}
    lines = []
    for i in range(n):
        name = f"abc_{i:04d}_{i + 1}.png"
        shots[name] = make_png_bytes(*size, marker=i)
        lines.append(json.dumps({
            "action": f"click ({100 + i * 50}, {200 + i * 30})",
            "screenshot": f"screenshot/{name}",
            "element": f"Menu item {i}",
            "rect": {"left": 90 + i * 50, "top": 190 + i * 30,
                     "right": 130 + i * 50, "bottom": 210 + i * 30},
            "marked_screenshot": f"screenshot/{name[:-4]}_marked.png",
            "thought": "Verbose human walkthrough text that must never be trained.",
        }))
    return "\n".join(lines), shots


def test_parse_md_description():
    md = "# Task 56\n**Description:** Create a shortcut of the Setup folder.\n\n**Level:** medium"
    assert parse_md_description(md) == "Create a shortcut of the Setup folder."
    assert parse_md_description("no description") == ""


def test_task_files_from_zip(tmp_path):
    zpath = tmp_path / "data.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("data/events/task301.jsonl", "{}")
        zf.writestr("data/events/task301.md", "x")
        zf.writestr("data/events/task2.jsonl", "{}")
        zf.writestr("data/events/screenshot/a.png", b"x")
    with zipfile.ZipFile(zpath) as zf:
        tasks = task_files_from_zip(zf)
    assert [t[0] for t in tasks] == ["task2", "task301"]  # numeric order
    assert tasks[1][2] == "data/events/task301.md"


def test_build_trajectory_full_pipeline():
    jsonl, shots = make_events(n=4)
    traj = build_trajectory("task1", jsonl, "# T\n**Description:** Open the app and save the file.\n",
                            shots)
    assert traj is not None
    assert len(traj.steps) == 4
    assert traj.task == "Open the app and save the file."
    assert all(s.action.verb == "click" for s in traj.steps)
    # verbose thoughts must not leak into training text
    assert all("walkthrough" not in s.action_text for s in traj.steps)


def test_build_trajectory_skips_missing_screenshots():
    jsonl, shots = make_events(n=3)
    traj = build_trajectory("task1", jsonl, "# T\n**Description:** x\n", {})
    assert traj is None


def test_build_trajectory_accepts_run_keyed_screenshots():
    """run() keys fetched images by the full jsonl value ("screenshot/x.png"),
    not the basename — the two lookups must agree (regression: silent 100%
    unparseable_task rejection)."""
    jsonl, shots = make_events(n=3)
    prefixed = {f"screenshot/{k}": v for k, v in shots.items()}
    traj = build_trajectory("task1", jsonl, "# T\n**Description:** x\n", prefixed)
    assert traj is not None
    assert len(traj.steps) == 3


def test_samples_one_per_event_no_multiplication(tmp_path):
    ctx = make_ctx(tmp_path, quota={"pcagente": 10})
    jsonl, shots = make_events(n=5)
    traj = build_trajectory("task7", jsonl, "# T\n**Description:** Save and export the file as PDF.\n", shots)
    samples = samples_for_trajectory(traj, ctx)
    assert len(samples) == 5  # all events, one sample each
    assert ctx.quota["pcagente"] == 5
    for s in samples:
        assert s["source"] == "pcagente"
        assert s["task_type"] == "action"
        assert s["images"], "sample must embed its screenshot"
        assert s["metadata"]["representation"] == "single"
        # Trusted bboxes are validated; resize stays 1600 when target remains >=11px.
        assert s["metadata"]["final_image_size"] == [1600, 900]
        assert s["metadata"]["bbox_click_validated"] is True
        content = s["messages"][-1]["content"]
        assert "Plan:" in content or content.startswith("click(")
        if "Plan:" in content:
            assert "\nAction: click(x=" in content


def test_samples_stop_at_quota(tmp_path):
    ctx = make_ctx(tmp_path, quota={"pcagente": 2})
    jsonl, shots = make_events(n=5)
    traj = build_trajectory("t", jsonl, "# T\n**Description:** x\n", shots)
    samples = samples_for_trajectory(traj, ctx)
    assert len(samples) == 2
    assert ctx.quota["pcagente"] == 0
