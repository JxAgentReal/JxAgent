"""Tests for the 2026-08 data-quality optimization pass."""
import pytest

from processing.quality import (ANCHOR_ELIGIBLE_VERBS, BucketQuota,
                                GROUNDCUA_SIZE_BUCKETS,
                                UNDERSTANDING_MAX_CONTROLS, app_rarity,
                                attach_quality, finish_has_evidence,
                                grounding_bucket, score_action_step,
                                score_grounding, token_efficiency,
                                wait_sample_allowed)
from processing.windows import REPRESENTATION_RATIOS
from tests.conftest import make_ctx, make_png_bytes, make_trajectory


# ---------------------------------------------------------- action scoring

def test_plain_click_is_not_rejected():
    qs = score_action_step(verb="click", task_text="Open the file menu",
                           signals=set(), app="libreoffice_calc",
                           app_counter={}, total_samples=0)
    assert qs.bucket in ("B", "C"), (qs.bucket, qs.score)
    assert qs.reject_reason is None


def test_recovery_step_scores_a():
    plain = score_action_step(verb="click", task_text="Open the file menu",
                              signals=set(), app="excel", app_counter={}, total_samples=0)
    recovery = score_action_step(verb="click", task_text="Open the file menu",
                                 signals={"no_state_change", "recovery"},
                                 prev_was_ineffective=True, app="excel",
                                 app_counter={}, total_samples=0)
    assert recovery.score > plain.score + 1.5
    assert recovery.bucket in ("A", "B")


def test_finish_and_verification_score_high():
    fin = score_action_step(verb="finish", task_text="Save the file as PDF",
                            signals={"save_export"}, app="word",
                            app_counter={}, total_samples=0)
    assert fin.bucket in ("A", "B")
    assert fin.components["verification"] == 1.0


def test_move_and_lone_keydown_never_anchor():
    for verb in ("move", "key_down", "key_up", "mouse_down", "mouse_up"):
        qs = score_action_step(verb=verb, task_text="t", signals=set(), app="a")
        assert qs.bucket == "Reject"
        assert verb not in ANCHOR_ELIGIBLE_VERBS


def test_trivial_first_click_downweighted_not_rejected():
    first = score_action_step(verb="click", task_text="Open the document", signals=set(),
                              is_first_step=True, app="excel")
    later = score_action_step(verb="click", task_text="Open the document", signals=set(),
                              is_first_step=False, app="excel")
    assert first.score < later.score
    assert first.bucket != "Reject"


def test_repeated_identical_without_recovery_rejected():
    qs = score_action_step(verb="click", task_text="t", signals=set(),
                           repeated_identical=True)
    assert qs.bucket == "Reject" and "repeated" in qs.reject_reason
    # but repetition AFTER an ineffective action (recovery) is kept
    # pixel no-change alone is not proof of recovery; explicit causal
    # evidence is required to keep repeated supervision.
    qs2 = score_action_step(verb="click", task_text="t",
                            signals={"no_state_change", "recovery_evidenced"},
                            repeated_identical=True)
    assert qs2.bucket != "Reject"


def test_giant_target_triviality():
    qs = score_action_step(verb="click", task_text="Click OK", signals=set(),
                           target_width_px=520)
    assert qs.components["triviality"] >= 0.8


def test_rare_app_beats_common_app():
    rare = score_action_step(verb="click", task_text="t", signals=set(),
                             app="grassgis", app_counter={"grassgis": 0}, total_samples=100)
    common = score_action_step(verb="click", task_text="t", signals=set(),
                               app="excel", app_counter={"excel": 80}, total_samples=100)
    assert rare.score > common.score


def test_app_rarity_monotone():
    assert app_rarity("blender", {"blender": 0}, 100) > app_rarity("blender", {"blender": 50}, 100)


def test_quality_metadata_attached_with_token_efficiency():
    sample = {"messages": [], "metadata": {}}
    qs = score_action_step(verb="click", task_text="sort the rows and export",
                           signals={"sorting", "export"}, app="excel")
    attach_quality(sample, qs, 2500)
    assert sample["metadata"]["quality"]["bucket"] in ("A", "B", "C")
    assert sample["metadata"]["quality"]["components"]["difficulty"] > 0
    assert sample["metadata"]["quality"]["token_efficiency"] > 0


def test_token_efficiency_prefers_dense_samples():
    dense = token_efficiency(8.0, 1200)
    sparse = token_efficiency(8.0, 7000)
    assert dense > 3 * sparse


# -------------------------------------------------------- grounding scoring

def test_grounding_tiny_labelled_is_a_normal_is_c():
    tiny = score_grounding(target_width_px=12, target_height_px=10,
                           text="zoom in button", category="Button", app="gimp")
    normal = score_grounding(target_width_px=140, target_height_px=32,
                             text="File menu", category="Menu", app="excel")
    assert tiny.score > normal.score
    assert tiny.bucket in ("A", "B")
    assert normal.bucket in ("B", "C")
    assert normal.bucket != "Reject"


def test_grounding_buckets_and_quota():
    assert grounding_bucket(10) == "tiny"
    assert grounding_bucket(24) == "small"
    assert grounding_bucket(50) == "medium"
    assert grounding_bucket(200) == "large"
    quota = BucketQuota()
    shares = {n: s for n, _, _, s in GROUNDCUA_SIZE_BUCKETS}
    assert sum(shares.values()) == 1.0
    for _ in range(20):
        quota.record("small")
    assert not quota.allow("small")  # 100% small violates the mixture
    assert quota.allow("tiny") and quota.allow("medium")


# ------------------------------------------------------------ wait / finish

def test_consecutive_waits_filtered():
    # one wait after a normal action: allowed
    assert wait_sample_allowed(["click(x=1, y=1)"])
    # a wait immediately after another wait: filtered (loop-bait)
    assert not wait_sample_allowed(["wait(seconds=1.0)"])
    assert not wait_sample_allowed(["wait(seconds=1.0)", "wait(seconds=2.0)"])
    assert wait_sample_allowed(["wait(seconds=1.0)", "click(x=1, y=1)"])


def test_finish_requires_objective_evidence_even_for_human_source():
    assert not finish_has_evidence(task_text="anything", prev_state_changed=None,
                                   human_source=True)
    assert not finish_has_evidence(task_text="anything", prev_state_changed=True,
                                   human_source=False)
    assert finish_has_evidence(explicit_success=True)
    assert finish_has_evidence(reliable_final_state=True)
    assert finish_has_evidence(verifier_evidence=True)


# ------------------------------------------------------- mixture consistency

def test_final_source_mixture_totals():
    import build_jxagent_dataset as b
    assert b.SOURCE_TARGETS == {"procua": 46000, "gui360": 16000,
                                "videocua": 17500, "groundcua": 4000,
                                "pcagente": 4503, "replay": 7500}
    assert 95000 <= sum(b.SOURCE_TARGETS.values()) <= 105000


def test_replay_mixture_sums_to_7500():
    from sources.replay import CATEGORIES
    counts = {k: n for k, (_, n) in CATEGORIES.items()}
    assert counts == {"coding": 1600, "math": 1500, "instruction": 1700,
                      "vqa": 1400, "tool": 1300}
    assert sum(counts.values()) == 7500


def test_representation_mixture_updated():
    assert REPRESENTATION_RATIOS["single"] == 0.55
    assert REPRESENTATION_RATIOS["window"] == 0.40
    assert REPRESENTATION_RATIOS["chunk"] == 0.05
    assert sum(REPRESENTATION_RATIOS.values()) == 1.0


def test_reasoning_target_rate_is_disabled_for_run1():
    from processing.reasoning import TARGET_RATE
    assert TARGET_RATE == 0.0


def test_understanding_cap_is_dense_and_bounded():
    assert UNDERSTANDING_MAX_CONTROLS == 24


# ----------------------------------------------- anchor eligibility end-to-end

def test_videocua_move_to_not_selected_as_anchor(tmp_path):
    from processing.coordinates import Action, CoordSpace, Point
    from processing.sampling import select_step_indices
    from processing.windows import Step
    from tests.conftest import make_png_bytes
    from processing.dedup import phash
    from processing.images import load_image
    steps = []
    for i, verb in enumerate(["move", "click", "move", "drag", "key_down", "click"]):
        data = make_png_bytes(640, 480, marker=i)
        act = Action(verb, points=[(10 + i, 20 + i)],
                     original_space=CoordSpace.PIXEL)
        steps.append(Step(step_id=f"s{i}", image_bytes=data, image_size=(640, 480),
                          action=act, phash=phash(load_image(data))))
    traj = make_trajectory(n=1, tid="mv")
    traj.steps = steps
    picked = select_step_indices(traj, cap=6)
    picked_verbs = {steps[i].action.verb for i in picked}
    assert "move" not in picked_verbs
    assert "key_down" not in picked_verbs
    assert "click" in picked_verbs


def test_pc_agent_e_wait_loop_rejected_but_single_wait_kept(tmp_path):
    from sources.pc_agent_e import build_trajectory, samples_for_trajectory
    from tests.conftest import make_events
    import json as J
    jsonl, shots = make_events(n=3)
    # append two consecutive waits
    shot = sorted(shots)[0]
    for k in (3, 4):
        shots[f"w{k}.png"] = make_png_bytes(1920, 1080, marker=9)
        jsonl += "\n" + J.dumps({"action": f"wait {k} seconds",
                                 "screenshot": f"screenshot/w{k}.png",
                                 "element": "", "rect": {}, "thought": ""})
    traj = build_trajectory("twait", jsonl, "# T\n**Description:** Open things.\n", shots)
    verbs = [s.action.verb for s in traj.steps]
    assert verbs.count("wait") == 2
    ctx = make_ctx(tmp_path, quota={"pcagente": 10})
    samples = samples_for_trajectory(traj, ctx)
    from processing.validation import extract_action_text
    wait_samples = [s for s in samples
                    if extract_action_text(s).startswith("wait")]
    assert len(wait_samples) == 1  # the repeated wait was filtered
    for s in samples:  # no double Plan: prefix anywhere
        assert "Plan: Plan:" not in s["messages"][-1]["content"]


def test_pc_agent_e_finish_parsed_and_kept(tmp_path):
    from sources.pc_agent_e import build_trajectory
    from tests.conftest import make_events
    import json as J
    jsonl, shots = make_events(n=2)
    shots["fin.png"] = make_png_bytes(1920, 1080, marker=11)
    jsonl += "\n" + J.dumps({"action": "finish", "screenshot": "screenshot/fin.png",
                             "element": "", "rect": {},
                             "thought": "Task successfully completed and verified."})
    traj = build_trajectory("tfin", jsonl, "# T\n**Description:** Set the clock.\n", shots)
    assert traj.steps[-1].action.verb == "finish"
    assert "finish_verification" in traj.steps[-1].signals or \
           "verification" in traj.steps[-1].signals


def test_gui360_terminate_becomes_finish():
    from sources.gui360 import use_row_to_trajectory
    from tests.conftest import gui360_use_row
    row = gui360_use_row(n_steps=2)
    import json as J
    msgs = J.loads(row["messages"])
    msgs.append({"role": "user", "content": [{"type": "image", "index": 2}]})
    msgs.append({"role": "assistant", "content": [],
                 "tool_calls": [{"type": "function",
                                 "function": {"name": "terminate",
                                              "arguments": {"status": "success"}}}]})
    row["messages"] = J.dumps(msgs)
    row["images"].append({"bytes": make_png_bytes(1040, 736, marker=7), "path": "2.jpg"})
    traj = use_row_to_trajectory(row)
    assert traj.steps[-1].action.verb == "finish"


def test_gui360_app_cap_balances_use_stream():
    from processing.sampling import AppCap
    cap = AppCap(cap=16 * 1000 * 3 // 5)  # the use-cohort balancing rule
    accepted = {"excel": 0, "word": 0, "ppt": 0}
    for app in ["excel"] * 40 + ["word"] * 30 + ["ppt"] * 30:
        if cap.allow(app):
            cap.record(app)
            accepted[app] += 1
    assert accepted["excel"] <= 16 * 1000 * 3 // 5  # no app dominates
    assert accepted["word"] > 0 and accepted["ppt"] > 0


def test_groundcua_bucket_quota_selects_balanced_targets():
    from sources.groundcua import select_element
    from processing.quality import BucketQuota, grounding_bucket
    entries = []
    # a screen dominated by small elements + one tiny + one medium + one large
    for i in range(30):
        entries.append({"bbox": [10 * i, 100, 10 * i + 20, 116], "text": f"item {i}",
                        "category": "Button"})
    entries.append({"bbox": [500, 300, 508, 308], "text": "tiny gear", "category": "Button"})
    entries.append({"bbox": [600, 300, 660, 320], "text": "medium panel", "category": "Menu"})
    entries.append({"bbox": [700, 300, 860, 340], "text": "large area", "category": "Others"})
    quota = BucketQuota()
    picks = []
    for _ in range(12):
        e = select_element(entries, quota=quota)
        assert e is not None
        w = e["bbox"][2] - e["bbox"][0]
        quota.record(grounding_bucket(w))
        picks.append(grounding_bucket(w))
    from collections import Counter
    c = Counter(picks)
    assert c["small"] <= 9              # not everything is small anymore
    assert c["tiny"] >= 1 and c["medium"] + c["large"] >= 2
