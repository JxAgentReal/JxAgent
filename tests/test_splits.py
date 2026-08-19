import pytest

from processing.splitting import (assign_splits, replay_sample_id, split_of,
                                  split_samples, trajectory_overlap)


def test_split_deterministic():
    assert split_of("traj_A") == split_of("traj_A")


def test_split_zero_trajectory_overlap_by_construction():
    samples = [{"trajectory_id": f"t{i}", "source": "procua", "step_id": str(i)}
               for i in range(500)]
    assigned = assign_splits(samples)
    train, val = split_samples(assigned)
    assert not trajectory_overlap(train, val)


def test_validation_share_approximates_target():
    ids = [f"trajectory_{i}" for i in range(5000)]
    val = sum(1 for t in ids if split_of(t) == "validation")
    assert 0.01 < val / len(ids) < 0.06  # ~3%


def test_windows_of_same_trajectory_share_split():
    tid = "shared_traj"
    a = {"trajectory_id": tid, "step_id": "s1..s4"}
    b = {"trajectory_id": tid, "step_id": "s5..s8"}
    assigned = assign_splits([a, b])
    assert assigned[0]["split"] == assigned[1]["split"]


def test_replay_ids_deterministic():
    assert replay_sample_id("coding", 5, "magicoder") == replay_sample_id("coding", 5, "magicoder")
    assert replay_sample_id("coding", 5, "magicoder") != replay_sample_id("math", 5, "magicoder")


def test_assign_splits_preserves_samples():
    samples = [{"trajectory_id": "x", "source": "replay", "step_id": "y"}]
    out = assign_splits(samples)
    assert out[0]["source"] == "replay"
    assert out[0]["split"] in ("train", "validation")
