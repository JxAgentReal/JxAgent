import pytest

from processing.windows import (CHUNK, SINGLE, WINDOW, REPRESENTATION_RATIOS,
                                build_chunk, build_single, build_window,
                                choose_representation, frames_add_information,
                                suggest_window_starts)
from tests.conftest import make_trajectory


def test_representation_ratios_sum_to_one():
    assert sum(REPRESENTATION_RATIOS.values()) == pytest.approx(1.0)


def test_choose_representation_deterministic_and_valid():
    for i in range(50):
        r = choose_representation(f"traj_{i}")
        assert r in (SINGLE, WINDOW, CHUNK)
        assert r == choose_representation(f"traj_{i}")


def test_single_step_sample_structure():
    traj = make_trajectory(n=10)
    spec = build_single(traj, 5)
    assert spec.representation == SINGLE
    assert spec.step_ids == [traj.steps[5].step_id]
    assert spec.current_step is traj.steps[5]
    assert len(spec.history_texts) == 5
    assert spec.assistant_turns == [traj.steps[5].action_text]


def test_window_covers_consecutive_steps():
    traj = make_trajectory(n=20)
    spec = build_window(traj, 3, 4)
    assert spec.representation == WINDOW
    assert spec.step_ids == [s.step_id for s in traj.steps[3:7]]
    # window reconstruction: ids map back to contiguous trajectory positions
    ids = [s.step_id for s in traj.steps]
    positions = [ids.index(sid) for sid in spec.step_ids]
    assert positions == list(range(3, 7))
    # hardened contract: history is context, exactly one next-action target
    assert len(spec.assistant_turns) == 1
    assert spec.assistant_turns[0] == traj.steps[6].action_text
    assert spec.history_texts[-3:] == [s.action_text for s in traj.steps[3:6]]


def test_window_bounds():
    traj = make_trajectory(n=10)
    spec = build_window(traj, 8, 4)
    assert len(spec.step_ids) == 2  # clipped at trajectory end


def test_chunk_limits_and_sparse_images():
    traj = make_trajectory(n=30)
    spec = build_chunk(traj, 5, 10)
    assert spec.representation == CHUNK
    assert len(spec.step_ids) == 10
    assert spec.extra_images == []
    assert len(spec.assistant_turns) == 1


def test_frames_add_information():
    from processing.dedup import phash
    from processing.images import load_image
    from tests.conftest import make_png_bytes
    h1 = phash(load_image(make_png_bytes(640, 480, marker=1)))
    h1b = phash(load_image(make_png_bytes(640, 480, marker=1)))
    h2 = phash(load_image(make_png_bytes(640, 480, marker=99)))
    assert not frames_add_information(h1, h1b)
    assert frames_add_information(h1, h2)
    assert frames_add_information(None, h1)


def test_window_starts_within_bounds():
    traj = make_trajectory(n=12)
    for s in suggest_window_starts(traj, 4, count=3):
        assert 0 <= s <= 8
    assert suggest_window_starts(make_trajectory(n=2), 4) == []
