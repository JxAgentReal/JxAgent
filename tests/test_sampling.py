import pytest

from processing.sampling import (AppCap, detect_task_signals, deterministic_keep,
                                 score_steps, select_step_indices,
                                 trajectory_priority)
from tests.conftest import make_trajectory


def test_positive_signals_double_weight():
    traj = make_trajectory(n=10, task="Open the application.")
    traj.steps[2].signals.add("recovery_evidenced")
    traj.steps[3].signals.add("trivial_open")
    scores = {s.index: s.weight for s in score_steps(traj)}
    base = scores[0]
    assert scores[2] == pytest.approx(base * 2.0)
    assert scores[3] == pytest.approx(base * 0.5)


def test_task_text_signal_detection():
    sig = detect_task_signals("Sort the files by name and export the list")
    assert "sorting" in sig and "export" in sig
    sig = detect_task_signals("Select exactly 3 rows")
    assert "exact_quantity" in sig
    sig = detect_task_signals("Rank the top 5 songs")
    assert "ranking" in sig
    sig = detect_task_signals("Save the document as report.docx")
    assert "save" in sig
    assert detect_task_signals("Open settings") == set()


def test_trajectory_priority_bands():
    assert trajectory_priority(make_trajectory(n=10)) == 1.0
    assert trajectory_priority(make_trajectory(n=40)) == 1.0
    assert trajectory_priority(make_trajectory(n=2)) == 0.5
    assert trajectory_priority(make_trajectory(n=5)) == 0.8
    assert trajectory_priority(make_trajectory(n=90)) == 0.9


def test_per_trajectory_cap_respected():
    traj = make_trajectory(n=40)
    picked = select_step_indices(traj, cap=4)
    assert len(picked) <= 4


def test_temporal_coverage_not_only_endings():
    traj = make_trajectory(n=40)
    picked = select_step_indices(traj, cap=4)
    assert len(picked) == 4
    # positions must cover different regions of the trajectory
    rel = sorted(p / 40 for p in picked)
    assert rel[0] < 0.5  # something from the first half
    assert rel[-1] >= 0.5  # something from the second half


def test_selection_deterministic():
    traj = make_trajectory(n=30)
    assert select_step_indices(traj) == select_step_indices(traj)


def test_deterministic_keep_is_stable():
    assert deterministic_keep("t1", 0.5) == deterministic_keep("t1", 0.5)
    assert deterministic_keep("t1", 1.0)


def test_app_cap_limits():
    cap = AppCap(cap=10)
    for _ in range(10):
        assert cap.allow("chrome")
        cap.record("chrome")
    assert not cap.allow("chrome")
    assert cap.allow("excel")


def test_signal_propagation_from_steps():
    traj = make_trajectory(n=10, with_recovery=True)
    scores = {s.index: set(s.signals) for s in score_steps(traj)}
    assert "no_state_change" in scores[4]
