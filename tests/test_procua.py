import json

import pytest

from processing.coordinates import parse_pyautogui
from sources.procua import parse_trajectory, samples_for_trajectory
from tests.conftest import make_ctx, make_png_bytes, make_trajectory


def procua_trajectory_json(n=8, goal="Sort the rows by revenue and export the sheet as CSV."):
    steps = []
    shots = {}
    for i in range(n):
        name = f"part_1/run1/traj_{i}/{i}-1.png"
        shots[name] = make_png_bytes(1920, 1080, marker=i)
        steps.append({
            "subgoal": f"phase {i}",
            "subgoal_intent": "do",
            "actions": [{
                "screenshot": name,
                "pyautogui_command": f"pyautogui.click(x={100 + i * 30}, y={200 + i * 10})",
                "action_type": "pyautogui",
                "action_generation": {"thought": "x", "action": "y",
                                      "code": f"pyautogui.click(x={100 + i * 30}, y={200 + i * 10})"},
                "raw_reasoning": "verbose reasoning must not be trained",
            }],
        })
    return {"trajectory_id": "traj_0001", "metadata": {"application": "libreoffice"},
            "goal": goal, "steps": steps}, shots


def test_parse_trajectory_from_stream_artifacts():
    traj_json, shots = procua_trajectory_json()
    traj = parse_trajectory(traj_json, shots)
    assert traj is not None
    assert traj.length == 8
    assert traj.app == "libreoffice"
    assert traj.trajectory_id == "procua_traj_0001"
    assert "sorting" in traj.steps[0].signals or "export" in traj.steps[0].signals


def test_parse_trajectory_requires_goal():
    traj_json, shots = procua_trajectory_json()
    traj_json["goal"] = ""
    assert parse_trajectory(traj_json, shots) is None


def test_parse_trajectory_missing_screenshots_skipped():
    traj_json, shots = procua_trajectory_json(n=4)
    traj = parse_trajectory(traj_json, {})  # no screenshots available
    assert traj is None


def test_unsupported_command_rejected_not_guessed():
    traj_json, shots = procua_trajectory_json(n=2)
    traj_json["steps"][1]["actions"][0]["pyautogui_command"] = "subprocess.run('rm -rf /')"
    traj = parse_trajectory(traj_json, shots)
    assert traj.length == 1  # the unsupported action was dropped, not guessed


def test_per_trajectory_cap_enforced(tmp_path):
    ctx = make_ctx(tmp_path, quota={"procua": 50})
    traj = make_trajectory(n=30)
    samples = samples_for_trajectory(traj, ctx)
    assert len(samples) <= 4  # default cap


def test_window_and_chunk_samples_present_across_trajectories():
    """Representation selection itself should expose single/window modes.

    This deliberately tests the selector without repeatedly re-encoding large
    synthetic screenshots. Image processing is covered by dedicated assembly
    tests; keeping this unit test selector-only makes the full quality suite
    deterministic and fast.
    """
    from processing.sampling import action_representation_specs
    from processing.windows import SINGLE, WINDOW
    seen = set()
    for t in range(60):
        traj = make_trajectory(n=20, tid=f"traj_{t}", size=(64, 36))
        for spec in action_representation_specs(traj, cap=4):
            seen.add(spec.representation)
    assert SINGLE in seen
    assert WINDOW in seen


def test_run1_does_not_attach_unverified_synthetic_reasoning(tmp_path):
    ctx = make_ctx(tmp_path, quota={"procua": 1000})
    total = 0
    for t in range(8):
        traj = make_trajectory(n=12, tid=f"t{t}", with_recovery=True)
        for sample in samples_for_trajectory(traj, ctx):
            total += 1
            assert not sample["messages"][-1]["content"].startswith("Plan:")
    assert total > 0

