from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from processing.coordinates import Action, CoordSpace
from processing.images import load_image, process_image
from processing.native_interface import render_action_for_contract
from processing.selection import load_frontier_scores, select_best_valid
from processing.splitting import assign_splits, group_overlap, split_samples
from processing.token_budget import estimate_loss_token_report
from processing.validation import validate_sample
from processing.windows import SampleSpec, Step
from sources.gui360 import sanitize_understanding_controls
from sources.pc_agent_e import _validated_rect_metadata
from sources.replay import _verify_math_when_possible
from tests.conftest import make_ctx, make_png_bytes

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(*, mode="text_actions", coord="processed_image_pixels", evidence=None):
    history = {"mode": mode}
    layout = {
        "system_prompt": "NATIVE SYS",
        "image_placeholder": "<native_image>",
        "assistant_action_template": "ACT[{action}]",
        "task_template": "Task: {task}",
        "history_heading": "History",
        "text_user_template": "{image}\nTASK={task}\nHISTORY={history}",
        "history_item_template": "H:{action}",
        "visual_user_with_task_template": "{image}\nTASK={task}",
        "visual_user_without_task_template": "{image}",
        "older_history_template": "\nOLDER:\n{history}",
    }
    if mode == "visual_recent_rounds":
        history.update({"recent_visual_rounds": 4, "task_location": "current_user",
                        "older_actions": "coordinate_free",
                        "older_summary_location": "current_user"})
    return {
        "schema_version": 1,
        "model_id": "Qwen/Qwen3.8-27B",
        "adapter": {"family": "jxagent_text_action_v1"},
        "coordinate_space": {"type": coord},
        "message_layout": layout,
        "history_policy": history,
        "source_evidence": evidence or [],
    }


def _selection_sample(i: int, *, score: float, eff: float = 1.0, motif=None, source="procua"):
    task = "Save and export the document" if motif == "save_export" else "Click the requested control"
    return {
        "messages": [
            {"role": "user", "content": task, "loss": False},
            {"role": "assistant", "content": f"click(x={10+i}, y={20+i})", "loss": True},
        ],
        "images": [], "source": source, "trajectory_id": f"t{i}", "step_id": f"s{i}",
        "task_type": "action",
        "metadata": {"quality": {"score": score, "token_efficiency": eff},
                     "group_id": f"g{i}", "app": f"app{i%3}"},
    }


def test_grayscale_training_input_becomes_rgb_and_lossless_roundtrip_exact():
    img = Image.new("L", (48, 32), 117)
    b = io.BytesIO(); img.save(b, format="PNG")
    rgb = load_image(b.getvalue())
    assert rgb.mode == "RGB"
    processed = process_image(rgb, max_long=1600, lossless=True)
    decoded = load_image(processed.data)
    assert decoded.mode == "RGB"
    assert decoded.size == rgb.size
    assert list(decoded.getdata()) == list(rgb.getdata())


def test_understanding_selector_is_deterministic_bounded_and_spatially_sorted():
    controls = []
    kinds = ["button", "tab", "checkbox", "menu", "input"]
    for i in range(70):
        x = (i * 137) % 920
        y = (i * 83) % 920
        controls.append({"control_text": f"Control {i}",
                         "control_type": kinds[i % len(kinds)],
                         "control_rect": [x, y, x + 30 + (i % 7), y + 16 + (i % 5)]})
    a = sanitize_understanding_controls(controls)
    b = sanitize_understanding_controls(controls)
    assert a == b
    assert len(a) == 24
    positions = [(c["control_rect"][1], c["control_rect"][0]) for c in a]
    assert positions == sorted(positions)
    assert all("_kind" not in c for c in a)


def test_pc_agent_e_bbox_is_hard_checked_and_center_offset_recorded():
    inside = Action("click", points=[(110, 120)], original_space=CoordSpace.PIXEL)
    md = _validated_rect_metadata({"left": 100, "top": 100, "right": 140, "bottom": 140},
                                  inside, (500, 400))
    assert md and md["bbox_click_validated"] is True
    assert 0 <= md["bbox_center_offset_norm"] <= 1
    outside = Action("click", points=[(300, 300)], original_space=CoordSpace.PIXEL)
    assert _validated_rect_metadata({"left": 100, "top": 100, "right": 140, "bottom": 140},
                                    outside, (500, 400)) is None


def test_best_valid_selection_prefers_quality_and_preserves_exact_quota():
    rows = [_selection_sample(i, score=float(i), eff=1.0) for i in range(1, 7)]
    picked, report = select_best_valid(rows, {"procua": 3})
    assert len(picked) == 3
    assert report["sources"]["procua"]["selected"] == 3
    picked_scores = sorted(s["metadata"]["quality"]["score"] for s in picked)
    assert picked_scores == [4.0, 5.0, 6.0]


def test_frontier_scores_only_use_explicitly_verifiable_rows(tmp_path):
    p = tmp_path / "frontier.jsonl"
    p.write_text("\n".join([
        json.dumps({"sample_id": "procua::t1::s1", "verifiable": True, "frontier_score": 0.9}),
        json.dumps({"sample_id": "procua::t2::s2", "verifiable": False, "frontier_score": 1.0}),
    ]) + "\n")
    assert load_frontier_scores(str(p)) == {"procua::t1::s1": 0.9}


def test_motif_floor_rebalances_without_changing_source_quota():
    rows = [_selection_sample(i, score=10-i, motif=None) for i in range(5)]
    special = _selection_sample(99, score=0.1, motif="save_export")
    rows.append(special)
    picked, report = select_best_valid(rows, {"procua": 3}, coverage_floors={"save_export": 1})
    assert len(picked) == 3
    assert any("save_export" in s["metadata"].get("motifs", []) for s in picked)
    assert not report["motif_coverage"]["unmet"]


def test_group_aware_split_keeps_family_out_of_both_splits():
    rows = []
    for i in range(20):
        rows.append({"source": "procua", "trajectory_id": f"t{i}", "step_id": str(i),
                     "task_type": "action", "messages": [], "images": [],
                     "metadata": {"collection_run": f"run{i//2}"}})
    assigned = assign_splits(rows, validation_pct=50)
    train, val = split_samples(assigned)
    assert not group_overlap(train, val)
    by_run = {}
    for r in assigned:
        by_run.setdefault(r["metadata"]["collection_run"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_run.values())


def test_loss_token_report_ignores_unsupervised_history_assistant():
    rows = [{
        "source": "procua", "task_type": "action", "images": [], "metadata": {},
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "this is long old history", "loss": False},
            {"role": "assistant", "content": "ok", "loss": True},
        ]
    }]
    rep = estimate_loss_token_report(rows)
    assert rep["total"]["assistant_loss_tokens"] == 1
    assert rep["total"]["input_text_tokens"] > rep["total"]["assistant_loss_tokens"]


def test_native_normalized_action_preserves_canonical_final_pixels():
    c = _contract(coord="normalized_0_1000")
    a = Action("click", points=[(999, 499)], original_space=CoordSpace.PIXEL)
    rendered, canonical = render_action_for_contract(a, (1000, 500), (1000, 500), c)
    assert rendered == "ACT[click(x=1000, y=1000)]"
    assert canonical == "click(x=999, y=499)"


def test_visual_native_history_has_real_images_but_only_final_assistant_loss(tmp_path):
    c = _contract(mode="visual_recent_rounds")
    ctx = make_ctx(tmp_path, quota={"procua": 10},
                   config={"_native_interface_contract": c, "context_budget": 50000})
    prev = []
    for i in range(5):
        prev.append(Step(step_id=f"p{i}", image_bytes=make_png_bytes(640, 480, marker=i),
                         image_size=(640, 480),
                         action=Action("click", points=[(50+i, 60+i)], original_space=CoordSpace.PIXEL),
                         metadata={}))
    current = Step(step_id="cur", image_bytes=make_png_bytes(640, 480, marker=20),
                   image_size=(640, 480),
                   action=Action("click", points=[(220, 180)], original_space=CoordSpace.PIXEL),
                   metadata={})
    spec = SampleSpec(source="procua", trajectory_id="vh", step_ids=["cur"],
                      representation="single", task="Do the task", current_step=current,
                      history_texts=[s.action_text for s in prev], signals=set(), app="writer",
                      metadata={"_prev_steps": prev})
    from processing.assemble import assemble_sample
    sample = assemble_sample(spec, ctx)
    assert sample is not None
    assistants = [m for m in sample["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 5
    assert [m["loss"] for m in assistants] == [False, False, False, False, True]
    assert len(sample["images"]) == 5
    ok, reason = validate_sample(sample, str(tmp_path))
    assert ok, reason


def _fake_model_and_contract(tmp_path: Path):
    model = tmp_path / "model"; model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "fake", "architectures": ["Fake"]}))
    (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}", "tokenizer_class": "Fake"}))
    (model / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "FakeVision"}))
    evidence = model / "OFFICIAL_CUA_EVIDENCE.md"
    evidence.write_text("Official local evidence for click drag scroll coordinate computer tool action interface.\n")
    c = _contract(evidence=[{"path": evidence.name, "sha256": _sha(evidence), "kind": "official"}])
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(c, indent=2))
    return model, evidence, contract


def test_interface_freeze_is_unresolved_without_explicit_contract(tmp_path):
    model, _, _ = _fake_model_and_contract(tmp_path)
    out = tmp_path / "manifest.json"
    r = subprocess.run([sys.executable, str(ROOT / "tools/freeze_qwen_interface.py"),
                        str(model), "--output", str(out)], capture_output=True, text=True)
    assert r.returncode == 2
    data = json.loads(out.read_text())
    assert data["status"] == "unresolved"
    assert "native_contract_missing" in data["unresolved"]


def test_interface_freeze_and_drift_verification(tmp_path):
    model, evidence, contract = _fake_model_and_contract(tmp_path)
    out = tmp_path / "manifest.json"
    r = subprocess.run([sys.executable, str(ROOT / "tools/freeze_qwen_interface.py"), str(model),
                        "--output", str(out), "--native-contract", str(contract)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(out.read_text())["status"] == "verified"
    verify = subprocess.run([sys.executable, str(ROOT / "tools/verify_interface_manifest.py"),
                             str(model), "--manifest", str(out)], capture_output=True, text=True)
    assert verify.returncode == 0, verify.stdout + verify.stderr
    evidence.write_text(evidence.read_text() + "drift\n")
    verify2 = subprocess.run([sys.executable, str(ROOT / "tools/verify_interface_manifest.py"),
                              str(model), "--manifest", str(out)], capture_output=True, text=True)
    assert verify2.returncode == 2
    assert "drift/missing" in verify2.stdout


def test_extra_decontamination_reference_loader_accepts_mixed_json(tmp_path):
    from processing.decontamination import load_reference_file
    p = tmp_path / "refs.json"
    p.write_text(json.dumps([{"id": "a", "instruction": "Do exact benchmark task"},
                             {"task_id": "b", "prompt": "Second task prompt"},
                             "Third task text"]))
    rows = load_reference_file(str(p))
    assert [x[1] for x in rows] == ["Do exact benchmark task", "Second task prompt", "Third task text"]


def test_math_verifier_arithmetic_and_simple_equation():
    assert _verify_math_when_possible("What is 12 / 3?", "The answer is 4") is True
    assert _verify_math_when_possible("What is 12 / 3?", "The answer is 5") is False
    assert _verify_math_when_possible("Solve the equation 2x + 3 = 11", "x = 4") is True
    assert _verify_math_when_possible("Solve the equation 2x + 3 = 11", "x = 5") is False
    assert _verify_math_when_possible("A story problem with no exact parser", "Maybe 7") is None


def test_mi300x_training_command_keeps_default_loss_scale():
    text = (ROOT / "mi300x/common.sh").read_text()
    assert "--loss_scale default" in text


def test_5000_sample_build_hard_fails_without_interface_manifest(tmp_path):
    import build_jxagent_dataset as b
    out = tmp_path / "build"
    try:
        b.main(["--output", str(out), "--sources", "replay", "--replay-count", "5000",
                "--offline", "--no-decontamination", "--allow-unversioned"])
        assert False, "expected interface gate"
    except SystemExit as e:
        assert "interface-manifest" in str(e)


def test_native_target_tamper_is_fatal_schema_failure(tmp_path):
    c = _contract(mode="text_actions")
    ctx = make_ctx(tmp_path, quota={"procua": 2},
                   config={"_native_interface_contract": c, "context_budget": 50000})
    step = Step(step_id="s", image_bytes=make_png_bytes(320, 240, marker=42),
                image_size=(320, 240),
                action=Action("click", points=[(50, 60)], original_space=CoordSpace.PIXEL))
    spec = SampleSpec(source="procua", trajectory_id="tamper", step_ids=["s"],
                      representation="single", task="click", current_step=step)
    from processing.assemble import assemble_sample
    from processing.validation import finalize
    sample = assemble_sample(spec, ctx)
    assert sample is not None
    sample["messages"][-1]["content"] += " CORRUPTED"
    sample["split"] = "train"
    stats = finalize(str(tmp_path), [sample])
    assert stats["fatal_failure"] is True
    assert stats["failures"]["native_target_hash_mismatch"] == 1
