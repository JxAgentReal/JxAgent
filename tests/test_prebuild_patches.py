"""Pre-build patch acceptance tests (2026-08-16 surgical quality patch).

Every test here maps to an acceptance criterion in DATASET_PREBUILD_PATCH_REPORT.md:
replay mixture, reasoning rate, GUI-360/VideoCUA coordinate safety, PC-Agent-E
parser, shared-image safety, orphan cleanup, duplicate ids, representation
mixture bands, and finish-evidence recording.
"""
import importlib.util
import json
import os
import sys

import pytest

from build_jxagent_dataset import load_config, parse_args, resolve_replay_counts
from processing.assemble import assemble_grounding, assemble_replay, assemble_sample
from processing.coordinates import Action, CoordSpace, parse_pc_agent_e
from processing.sampling import action_representation_specs
from processing.validation import cleanup_orphan_images, finalize
from processing.windows import (ACTION_SAMPLE_BANDS, ACTION_SOURCE_RATIOS,
                                chunk_is_informative, window_is_informative)
from sources.gui360 import resolve_conversion_space, use_row_to_trajectory
from sources.videocua import build_trajectory
from tests.conftest import (make_ctx, make_png_bytes, make_step,
                            make_trajectory)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _args(argv):
    return parse_args(["--output", "x"] + argv)


# --------------------------------------------------------------- replay mix

def test_replay_default_mixture_is_canonical():
    from sources.replay import CATEGORIES
    counts = resolve_replay_counts(_args([]))
    assert counts == {"coding": 1600, "math": 1500, "instruction": 1700,
                      "vqa": 1400, "tool": 1300}
    assert counts == {cat: default for cat, (_, default) in CATEGORIES.items()}


def test_replay_explicit_override_and_scaling():
    counts = resolve_replay_counts(_args(["--replay-coding", "999"]))
    assert counts["coding"] == 999
    assert counts["vqa"] == 1400  # untouched categories stay canonical
    scaled = resolve_replay_counts(_args(["--replay-count", "750"]))
    assert sum(scaled.values()) == pytest.approx(750, abs=3)
    smoke = resolve_replay_counts(_args(["--replay-count", "20", "--smoke"]))
    assert set(smoke.values()) == {4}  # max(1, 20 // 5)


def test_replay_yaml_matches_categories():
    import yaml
    from sources.replay import CATEGORIES
    with open(os.path.join(REPO_ROOT, "configs", "dataset.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["replay"] == {cat: default for cat, (_, default) in CATEGORIES.items()}


# ------------------------------------------------------------ reasoning rate

def test_reasoning_default_rate_is_disabled_for_run1():
    from processing.reasoning import TARGET_RATE
    assert TARGET_RATE == 0.0
    assert load_config(None)["reasoning"]["rate"] == 0.0


def test_reasoning_config_override_works(tmp_path):
    cfg_file = tmp_path / "override.yaml"
    cfg_file.write_text("reasoning:\n  rate: 0.2\n")
    assert load_config(str(cfg_file))["reasoning"]["rate"] == 0.2


def test_reasoning_never_on_non_action_samples(tmp_path):
    ctx = make_ctx(tmp_path, quota={"groundcua": 5})
    sample = assemble_grounding(
        source="groundcua", trajectory_id="g1", step_id="g1_p",
        image_bytes=make_png_bytes(800, 600), instruction="OK button",
        target_xy=(400, 300), image_size=(800, 600), target_width_px=12,
        app="7-Zip", ctx=ctx)
    assert sample is not None
    assert "Plan:" not in sample["messages"][-1]["content"]
    assert sample["metadata"]["reasoning_category"] is None

    rep = assemble_replay(
        messages=[{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "hello"}],
        images_pil=[], source_name="replay", sample_id="r1",
        task_type="replay_coding", metadata={}, ctx=make_ctx(tmp_path, quota={"replay": 5}))
    assert rep is not None
    assert "Plan:" not in rep["messages"][-1]["content"]


# --------------------------------------------------- GUI-360 dimension guard

def _gui360_row(n_steps=2, meta_res=(1040, 736), img_size=(1040, 736)):
    images = [{"bytes": make_png_bytes(*img_size, marker=i), "path": f"{i}.jpg"}
              for i in range(n_steps)]
    messages = []
    for i in range(n_steps):
        user = {"role": "user", "content": [{"type": "image", "index": i}]}
        if i == 0:
            user["content"].append({"type": "text", "text": "Save the file as read-only."})
        messages.append(user)
        messages.append({"role": "assistant",
                         "content": [],
                         "tool_calls": [{"type": "function", "function": {
                             "name": "click",
                             "arguments": {"coordinate": [40 + i * 100, 70 + i * 50]}}}]})
    return {
        "images": images,
        "messages": json.dumps(messages),
        "metadata": json.dumps({"platform": "desktop", "task_type": "use",
                                "others": {"id": "excel_1_105",
                                           "resolution": list(meta_res)}}),
    }


def test_gui360_dimension_guard_matching():
    stats = {}
    traj = use_row_to_trajectory(_gui360_row(), stats=stats)
    assert traj is not None and len(traj.steps) == 2
    assert traj.steps[0].image_size == (1040, 736)
    x, y = traj.steps[0].action.points[0]
    assert (x, y) == (42, 51)  # closed-domain endpoint-preserving normalized mapping
    assert not traj.steps[0].metadata.get("coordinate_dimension_mismatch")
    assert stats == {}


def test_gui360_dimension_guard_convertible_uses_actual():
    stats = {}
    traj = use_row_to_trajectory(_gui360_row(img_size=(2080, 1472)), stats=stats)
    assert traj is not None and len(traj.steps) == 2
    # converted in the ACTUAL image space (uniform 2x rescale of the reference)
    assert traj.steps[0].image_size == (2080, 1472)
    x, y = traj.steps[0].action.points[0]
    assert (x, y) == (83, 103)  # 40/1000*2080, 70/1000*1472
    assert traj.steps[0].metadata.get("coordinate_dimension_mismatch") is True
    assert stats["mismatch_used_actual"] == 2


def test_gui360_dimension_guard_ambiguous_rejects():
    stats = {}
    # same width, very different aspect -> convention cannot be converted safely
    traj = use_row_to_trajectory(_gui360_row(img_size=(1040, 900)), stats=stats)
    assert traj is None
    assert stats["ambiguous"] == 2


def test_resolve_conversion_space_paths():
    assert resolve_conversion_space((1040, 736), (1040, 736))[2] == "ok"
    assert resolve_conversion_space((1040, 736), (2080, 1472))[2] == "mismatch_used_actual"
    assert resolve_conversion_space((1040, 736), (1040, 900))[2] == "ambiguous"


# -------------------------------------------------- VideoCUA frame dimensions

def _vc_log(entries):
    return {"task_id": 45525, "task_instruction": "Open test.7z and extract it",
            "platform": "7-Zip", "action_log": entries}


def _frames_for(entries, size):
    return {e["timestamp"]: make_png_bytes(*size, marker=i)
            for i, e in enumerate(entries)}


def test_videocua_identity_at_1080p():
    entries = [{"action_type": "CLICK", "timestamp": 1.0,
                "action_params": {"x": 100, "y": 200, "numClicks": 1}}]
    traj = build_trajectory(_vc_log(entries), _frames_for(entries, (1920, 1080)),
                            (1920, 1080))
    assert traj.steps[0].image_size == (1920, 1080)
    assert traj.steps[0].action.points[0] == (100, 200)  # identity transform


def test_videocua_1280x720_transform():
    entries = [{"action_type": "CLICK", "timestamp": 1.0,
                "action_params": {"x": 100, "y": 200, "numClicks": 1}}]
    traj = build_trajectory(_vc_log(entries), _frames_for(entries, (1280, 720)),
                            (1920, 1080))
    assert traj.steps[0].image_size == (1280, 720)
    assert traj.steps[0].action.points[0] == (67, 133)  # x*1280/1920, y*720/1080
    assert traj.metadata["actual_frame_size"] == [1280, 720]
    assert "transform" in traj.metadata


def test_videocua_different_aspect_transform():
    entries = [{"action_type": "CLICK", "timestamp": 1.0,
                "action_params": {"x": 100, "y": 200, "numClicks": 1}}]
    traj = build_trajectory(_vc_log(entries), _frames_for(entries, (800, 1000)),
                            (1920, 1080))
    assert traj.steps[0].action.points[0] == (42, 185)


def test_videocua_invalid_coordinate_dropped_and_counted():
    entries = [
        {"action_type": "CLICK", "timestamp": 1.0,
         "action_params": {"x": 2000, "y": 200, "numClicks": 1}},  # out of 1920
        {"action_type": "CLICK", "timestamp": 2.0,
         "action_params": {"x": 100, "y": 200, "numClicks": 1}},
    ]
    traj = build_trajectory(_vc_log(entries), _frames_for(entries, (1920, 1080)),
                            (1920, 1080))
    assert traj is not None
    assert len(traj.steps) == 1
    assert traj.metadata["coordinate_out_of_bounds_after_transform"] == 1


# ------------------------------------------------------ PC-Agent-E parser

def test_pcae_drag_structural():
    a = parse_pc_agent_e("drag from (383, 299) to (763, 299)", 1920, 1080)
    assert a is not None and a.verb == "drag"
    assert [tuple(p) for p in a.points] == [(383, 299), (763, 299)]
    # unrelated numbers elsewhere must never shift coordinates -> rejected
    assert parse_pc_agent_e("drag 2 items from (383, 299) to (763, 299)", 1920, 1080) is None
    assert parse_pc_agent_e("drag from (1, 2) to (3, 4) (5, 6)", 1920, 1080) is None
    assert parse_pc_agent_e("drag to the Desktop folder", 1920, 1080) is None


def test_pcae_press_strips_key_prefix():
    a = parse_pc_agent_e("press key enter", 1920, 1080)
    assert a.verb == "press" and a.args["key"] == "enter"
    assert parse_pc_agent_e("press enter", 1920, 1080).args["key"] == "enter"
    assert parse_pc_agent_e("press key backspace", 1920, 1080).args["key"] == "backspace"


def test_pcae_hotkey_parenthesized_list():
    a = parse_pc_agent_e("hotkey (Ctrl, A)", 1920, 1080)
    assert a.verb == "hotkey" and a.args["keys"] == ["Ctrl", "A"]
    assert parse_pc_agent_e("hotkey ctrl+s", 1920, 1080).args["keys"] == ["ctrl", "s"]


def test_pcae_scroll_signed_numeric():
    assert parse_pc_agent_e("scroll (-2)", 1920, 1080).args["clicks"] == -2
    assert parse_pc_agent_e("scroll (5)", 1920, 1080).args["clicks"] == 5
    assert parse_pc_agent_e("scroll down", 1920, 1080).args["clicks"] == 3
    assert parse_pc_agent_e("scroll up", 1920, 1080).args["clicks"] == -3


def test_pcae_type_strips_text_prefix():
    assert parse_pc_agent_e("type text: semantic", 1920, 1080).args["text"] == "semantic"
    assert parse_pc_agent_e("type hello world", 1920, 1080).args["text"] == "hello world"


# ------------------------------------------- shared images / orphan cleanup

def test_assemble_rejection_does_not_persist_orphan_images(tmp_path):
    ctx = make_ctx(tmp_path, quota={"procua": 5})
    traj = make_trajectory(n=4, tid="t1")
    spec = action_representation_specs(traj, cap=4)
    single = [s for s in spec if s.representation == "single"][0]
    single.task = "x" * 40000  # forces estimate over the 8192 budget
    sample = assemble_sample(single, ctx, trajectory=traj)
    assert sample is None  # rejected for budget...
    # ...and hardening now persists images only after acceptance.
    imgs = []
    for dirpath, _d, files in os.walk(os.path.join(str(tmp_path), "images")):
        imgs.extend(files)
    assert not imgs, "rejected sample must not leave orphan image files"


def test_orphan_cleanup_removes_only_unreferenced(tmp_path):
    root = str(tmp_path)
    img_dir = os.path.join(root, "images", "replay")
    os.makedirs(img_dir)
    for name in ("a.webp", "b.webp"):
        with open(os.path.join(img_dir, name), "wb") as f:
            f.write(b"x")
    sample = {"images": ["images/replay/a.webp"]}
    stats = cleanup_orphan_images(root, [sample])
    assert stats["deleted_orphans"] == 1 and stats["kept"] == 1
    assert os.path.exists(os.path.join(img_dir, "a.webp"))
    assert not os.path.exists(os.path.join(img_dir, "b.webp"))
    again = cleanup_orphan_images(root, [sample])  # safe to rerun
    assert again["deleted_orphans"] == 0 and again["kept"] == 1


def test_finalize_duplicate_ids_are_fatal_and_deduped(tmp_path):
    ctx = make_ctx(tmp_path, quota={"pcagente": 5})
    traj = make_trajectory(n=1, tid="dup", source="pcagente")
    spec = action_representation_specs(traj, cap=4)[0]
    s1 = assemble_sample(spec, ctx, trajectory=traj)
    assert s1 is not None
    import copy
    s2 = copy.deepcopy(s1)  # same (source, trajectory_id, step_id)
    s1["split"] = s2["split"] = "train"
    stats = finalize(str(tmp_path), [s1, s2])
    assert stats["failures"].get("duplicate_sample_id") == 1
    assert stats["fatal_failure"] is True
    n = 0
    for line in open(os.path.join(str(tmp_path), "final", "train.jsonl"), encoding="utf-8"):
        if line.strip():
            n += 1
    assert n == 1  # deduped, never shipped twice


def test_finalize_missing_image_is_fatal(tmp_path):
    bad = {"messages": [{"role": "user", "content": "<image>\nTask: x"},
                        {"role": "assistant", "content": "click(x=1, y=1)"}],
           "images": ["images/procua/missing.webp"], "source": "procua",
           "trajectory_id": "t", "step_id": "s", "task_type": "action",
           "metadata": {"final_image_size": [100, 100]},
           "split": "train"}
    stats = finalize(str(tmp_path), [bad])
    assert stats["fatal_failure"] is True
    assert stats["failures"].get("missing_image") == 1


# ----------------------------------------------- representation quality gates

def test_window_quality_gate():
    traj = make_trajectory(n=10, tid="wq")
    assert window_is_informative(traj.steps[0:4])  # distinct frames/actions
    flat = make_trajectory(n=10, tid="wq2")
    same = flat.steps[0].phash
    for s in flat.steps:
        s.phash = same
        s.prev_phash = same
        s.action = Action("click", points=[(10, 10)], original_space=CoordSpace.PIXEL)
    assert not window_is_informative(flat.steps[0:4])  # identical screens + same action


def test_chunk_quality_gate():
    traj = make_trajectory(n=12, tid="cq")
    assert chunk_is_informative(traj.steps[0:10])
    waits = make_trajectory(n=12, tid="cq2")
    for s in waits.steps:
        s.action = Action("wait", args={"seconds": 2.0},
                          original_space=CoordSpace.PIXEL)
    assert not chunk_is_informative(waits.steps[0:10])  # wait-dominated


def test_emission_respects_cap_and_gates():
    traj = make_trajectory(n=20, tid="em")
    specs = action_representation_specs(traj, cap=4)
    assert 1 <= len(specs) <= 4
    assert all(s.representation in ("single", "window", "chunk") for s in specs)
    for s in specs:
        if s.representation == "window":
            ids = [st.step_id for st in traj.steps]
            start = ids.index(s.step_ids[0])
            assert window_is_informative(
                traj.steps[start:start + len(s.step_ids)])


def test_representation_simulator_lands_in_band():
    spec = importlib.util.spec_from_file_location(
        "simrep", os.path.join(REPO_ROOT, "tools", "simulate_representation.py"))
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    scale = 0.05
    sim.QUOTAS = {k: max(20, int(v * scale)) for k, v in sim.QUOTAS.items()}
    sim.GUI360_USE = int(sim.GUI360_USE * scale)
    sim.GUI360_GROUNDING = int(sim.GUI360_GROUNDING * scale)
    sim.GUI360_UNDERSTANDING = int(sim.GUI360_UNDERSTANDING * scale)
    r = sim.run(emit_new=True)
    for rep in ("single", "window", "chunk"):
        lo, hi = ACTION_SAMPLE_BANDS[rep]
        assert lo <= r["action_sample_shares"][rep] <= hi, \
            f"{rep} share {r['action_sample_shares'][rep]} outside [{lo}, {hi}]"
    assert r["tokens"]["guard_ok"]
    assert sum(ACTION_SOURCE_RATIOS.values()) == pytest.approx(1.0)


# ------------------------------------------------------- finish evidence

def test_finish_requires_objective_evidence_in_assembly(tmp_path):
    # Human provenance alone is no longer evidence. PC-Agent-E finish is kept
    # only when the source adapter supplies objective success/final-state data.
    ctx = make_ctx(tmp_path, quota={"pcagente": 5})
    finish_step = make_step(0, action=Action("finish", args={"status": "success"},
                                             original="finish",
                                             original_space=CoordSpace.PIXEL))
    finish_step.metadata["explicit_success"] = True
    from processing.windows import Trajectory
    traj = Trajectory(trajectory_id="pcagente_t1", task="do things",
                      steps=[finish_step], app="windows_desktop", source="pcagente")
    spec = action_representation_specs(traj, cap=4)[0]
    sample = assemble_sample(spec, ctx, trajectory=traj)
    assert sample is not None
    assert sample["metadata"]["finish_evidence"] == "yes"

    # No explicit success, reliable final state or verifier evidence means the
    # finish label is removed even if the preceding trajectory remains useful.
    ctx2 = make_ctx(tmp_path, quota={"procua": 5})
    finish_step2 = make_step(0, action=Action("finish", args={"status": "success"},
                                              original="finish",
                                              original_space=CoordSpace.PIXEL))
    traj2 = Trajectory(trajectory_id="procua_t1", task="do things",
                       steps=[finish_step2], app="libreoffice", source="procua")
    spec2 = action_representation_specs(traj2, cap=4)[0]
    sample2 = assemble_sample(spec2, ctx2, trajectory=traj2)
    assert sample2 is None
    assert ctx2.state.source_counts("procua")["rejected_by_reason"].get("finish_without_evidence", 0) >= 1


def test_finalize_reports_extended_mixture(tmp_path):
    ctx = make_ctx(tmp_path, quota={"pcagente": 5})
    traj = make_trajectory(n=2, tid="mx", source="pcagente")
    samples = []
    for spec in action_representation_specs(traj, cap=4):
        s = assemble_sample(spec, ctx, trajectory=traj)
        if s:
            samples.append(s)
    rep = assemble_replay(
        messages=[{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "hello"}],
        images_pil=[], source_name="replay", sample_id="r1",
        task_type="replay_coding",
        metadata={"replay_source": "Magicoder", "license": "apache-2.0"},
        ctx=make_ctx(tmp_path, quota={"replay": 5}))
    samples.append(rep)
    stats = finalize(str(tmp_path), samples)
    for key in ("replay_category_counts", "representation_counts_by_source",
                "finish_sample_count", "finish_evidence",
                "grounding_size_distribution_by_source",
                "coordinate_dimension_mismatches", "estimated_epoch_tokens",
                "orphan_cleanup", "average_messages_per_sample"):
        assert key in stats, f"stats missing {key}"
    assert stats["replay_category_counts"].get("replay_coding") == 1
