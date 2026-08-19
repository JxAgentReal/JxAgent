import json
import os

import pytest

from processing.state import BuildState


def test_atomic_write_leaves_valid_json(tmp_path):
    path = str(tmp_path / "x.json")
    BuildState.atomic_write_json(path, {"a": 1})
    BuildState.atomic_write_json(path, {"a": 2})
    assert json.load(open(path))["a"] == 2
    assert not any(f.endswith(".tmp") for f in os.listdir(tmp_path))


def test_shard_tracking_resume(tmp_path):
    st = BuildState(str(tmp_path))
    assert not st.is_shard_done("procua", "shards/procua_sft_00000.tar.zst")
    st.mark_shard_done("procua", "shards/procua_sft_00000.tar.zst")
    st.save()
    # reload from disk = resume
    st2 = BuildState(str(tmp_path))
    assert st2.is_shard_done("procua", "shards/procua_sft_00000.tar.zst")
    assert not st2.is_shard_done("procua", "shards/procua_sft_00001.tar.zst")


def test_counts_and_rejections_persist(tmp_path):
    st = BuildState(str(tmp_path))
    st.set_target("pcagente", 4503)
    st.add_selected("pcagente", 3, trajectories=1)
    st.add_rejection("pcagente", "coordinate_out_of_bounds")
    st.save()
    st2 = BuildState(str(tmp_path))
    c = st2.source_counts("pcagente")
    assert c["selected"] == 3 and c["target"] == 4503
    assert c["rejected_by_reason"]["coordinate_out_of_bounds"] == 1


def test_jsonl_append_and_read(tmp_path):
    st = BuildState(str(tmp_path))
    st.append_jsonl("selected_samples.jsonl", [{"a": 1}, {"a": 2}])
    st.append_jsonl("selected_samples.jsonl", [{"a": 3}])
    rows = st.read_jsonl("selected_samples.jsonl")
    assert [r["a"] for r in rows] == [1, 2, 3]


def test_corrupt_progress_recovers(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    (d / "progress.json").write_text("{corrupt json", encoding="utf-8")
    st = BuildState(str(tmp_path))
    assert st.progress["sources"] == {}


def test_dedup_index_round_trip_through_state(tmp_path):
    from processing.dedup import DedupIndex
    st = BuildState(str(tmp_path))
    idx = DedupIndex()
    idx.consider(image_phash=42, signals=[], task_text="a", action_text="b")
    st.save_dedup_index(idx)
    restored = st.load_dedup_index()
    assert restored._seen_phashs == [42]


def test_rerun_continues_from_37_percent_style_state(tmp_path):
    st = BuildState(str(tmp_path))
    for i in range(37):
        st.mark_shard_done("procua", f"shard_{i}")
        st.add_selected("procua", 1)
    st.save()
    st2 = BuildState(str(tmp_path))
    assert st2.shard_count("procua") == 37
    assert st2.selected_total("procua") == 37
    # simulate resumed build marking more
    st2.mark_shard_done("procua", "shard_37")
    st2.add_selected("procua", 1)
    st2.save()
    assert BuildState(str(tmp_path)).shard_count("procua") == 38


def test_hard_kill_mid_source_keeps_done_unit_samples(tmp_path):
    """A unit marked done must always have its samples on disk: sources call
    ctx.persist_samples() BEFORE mark_shard_done, so a hard kill (no finally,
    no unwind) can never under-fill the build permanently."""
    from sources.common import BuildContext

    ctx = BuildContext(dataset_root=str(tmp_path), state=BuildState(str(tmp_path / "state")),
                       config={}, quota={"procua": 2})
    sample_a = {"source": "procua", "trajectory_id": "t1", "step_id": "t1_s0",
                "messages": [], "images": []}
    # unit A completes the full checkpoint sequence (persist -> mark -> save)
    ctx.persist_samples([sample_a])
    ctx.state.mark_shard_done("procua", "unit_A")
    ctx.state.save()

    # hard kill here: fresh context, nothing in memory survives
    ctx2 = BuildContext(dataset_root=str(tmp_path), state=BuildState(str(tmp_path / "state")),
                        config={}, quota={"procua": 1})
    assert ctx2.state.is_shard_done("procua", "unit_A")  # unit A skipped on rerun
    sample_b = {"source": "procua", "trajectory_id": "t2", "step_id": "t2_s0",
                "messages": [], "images": []}
    ctx2.persist_samples([sample_b])
    ctx2.state.mark_shard_done("procua", "unit_B")
    ctx2.state.save()

    rows = ctx2.state.read_jsonl("selected_samples.jsonl")
    keys = {(r["source"], r["trajectory_id"], r["step_id"]) for r in rows}
    assert keys == {("procua", "t1", "t1_s0"), ("procua", "t2", "t2_s0")}
