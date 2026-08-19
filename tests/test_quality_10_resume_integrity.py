import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from processing.assemble import assemble_grounding, assemble_replay, assemble_sample
from processing.coordinates import Action
from processing.windows import Step, Trajectory, build_single
from sources.common import BuildContext
from processing.state import BuildState
from processing.dedup import DedupIndex, phash
from processing.reasoning import ReasoningGate
from tests.conftest import make_png_bytes, make_ctx


def _ctx(tmp_path, source, remaining, selected=0, config=None):
    st = BuildState(str(tmp_path / "state"))
    st.source_counts(source)["selected"] = selected
    return BuildContext(dataset_root=str(tmp_path), state=st,
                        config={"context_budget": 8192, "per_trajectory_cap": 4, **(config or {})},
                        dedup=DedupIndex(), reasoning_gate=ReasoningGate(rate=0),
                        quota={source: remaining})


def test_gui360_resume_preserves_original_cohort_targets(tmp_path, monkeypatch):
    import sources.gui360 as g
    ctx = _ctx(tmp_path, "gui360", remaining=40, selected=60,
               config={"_resume_gui360_cohorts": {"use": 30, "grounding": 20, "understanding": 10}})
    calls = []
    def fake_use(c):
        want = c.quota["gui360"]; calls.append(("use", want)); c.consume("gui360", want)
        return [{"cohort":"use"} for _ in range(want)]
    def fake_named(name):
        def f(c, want):
            calls.append((name, want)); c.consume("gui360", want)
            return [{"cohort":name} for _ in range(want)]
        return f
    monkeypatch.setattr(g, "build_use_samples", fake_use)
    monkeypatch.setattr(g, "build_grounding_samples", fake_named("grounding"))
    monkeypatch.setattr(g, "build_understanding_samples", fake_named("understanding"))
    out = g.run(ctx, grounding_want=30, understanding_want=20)
    assert calls == [("use", 20), ("grounding", 10), ("understanding", 10)]
    assert len(out) == 40
    assert ctx.state.source_counts("gui360")["target"] == 100
    assert ctx.quota["gui360"] == 0


def test_groundcua_resume_uses_full_target_for_platform_cap(tmp_path, monkeypatch):
    import sources.groundcua as g
    prior = {x: 20 for x in ("a", "b", "c", "d")}
    ctx = _ctx(tmp_path, "groundcua", remaining=20, selected=80,
               config={"_resume_app_counts": {"groundcua": prior}})
    monkeypatch.setattr(g, "platform_directories", lambda ctx: list(prior))
    called = []
    def no_files(ctx, platform, page_size, cursor):
        called.append(platform); return []
    monkeypatch.setattr(g, "list_annotation_files", no_files)
    out = g.run(ctx)
    assert out == []
    # Old resume logic derived cap from remaining=20, making every prior=20 app blocked.
    assert set(called) == set(prior)
    assert ctx.state.source_counts("groundcua")["target"] == 100


def test_videocua_resume_cap_and_selected_count_not_double_added(tmp_path, monkeypatch):
    import sources.videocua as v
    apps = [f"app{i}" for i in range(20)]
    prior = {a: 40 for a in apps}
    ctx = _ctx(tmp_path, "videocua", remaining=200, selected=800,
               config={"_resume_app_counts": {"videocua": prior}})
    monkeypatch.setattr(v, "hf_tree", lambda *a, **k: [
        {"type":"file", "path":f"raw_data/{app}.zip", "size":1} for app in apps])
    def fake_process(c, zip_path, app_name, cap):
        assert cap.allow(app_name), f"resume cap incorrectly blocks {app_name}"
        n = min(10, c.remaining("videocua"))
        cap.record(app_name, n)
        c.consume("videocua", n)
        c.state.add_selected("videocua", n, trajectories=1)
        return [{"app":app_name} for _ in range(n)]
    monkeypatch.setattr(v, "_process_app_zip", fake_process)
    out = v.run(ctx)
    assert len(out) == 200
    assert ctx.state.selected_total("videocua") == 1000
    assert ctx.state.source_counts("videocua")["target"] == 1000


def test_replay_resume_skips_previously_selected_canonical_id_and_fills_quota(tmp_path, monkeypatch):
    import sources.replay as r
    rows = [
        {"instruction":"Explain how to implement a stable merge sort in Python with clear constraints.",
         "response":"```python\ndef merge_sort(xs):\n    if len(xs) <= 1: return xs\n    m=len(xs)//2\n    a=merge_sort(xs[:m]); b=merge_sort(xs[m:])\n    out=[]\n    while a and b: out.append(a.pop(0) if a[0] <= b[0] else b.pop(0))\n    return out+a+b\n```"},
        {"instruction":"Write a Python function that returns the unique values of a list while preserving order.",
         "response":"```python\ndef unique_ordered(xs):\n    seen=set(); out=[]\n    for x in xs:\n        if x not in seen:\n            seen.add(x); out.append(x)\n    return out\n```"},
    ]
    monkeypatch.setattr(r, "_iter_dataset", lambda *a, **k: iter(rows))
    monkeypatch.setattr(r, "_select", lambda *a, **k: True)
    first_msgs=[{"role":"user","content":rows[0]["instruction"]},{"role":"assistant","content":rows[0]["response"]}]
    first_id=r._canonical_sample_id("replay_coding", first_msgs, extra={
        "repo":"ise-uiuc/Magicoder-Evol-Instruct-110K",
        "revision":r.revision_for("ise-uiuc/Magicoder-Evol-Instruct-110K")})
    ctx = _ctx(tmp_path, "replay", remaining=1)
    ctx.seen_replay_ids.add(first_id)
    got = r.replay_coding(ctx, 1)
    assert len(got) == 1 and got[0]["trajectory_id"] != first_id
    assert ctx.rejected["replay"]["duplicate_canonical_id"] == 1


def test_no_orphan_image_on_rejected_action_grounding_or_replay(tmp_path):
    # Action rejection by tiny token budget must happen before image persistence.
    ctx = make_ctx(tmp_path / "a", quota={"procua": 1}, config={"context_budget": 1})
    img = make_png_bytes(100, 80, marker=1)
    st = Step("s", img, (100,80), Action("click", points=[(20,20)]), phash=phash(Image.open(io.BytesIO(img))))
    traj = Trajectory("t", "Do the thing", [st], app="x", source="procua")
    assert assemble_sample(build_single(traj, 0), ctx, trajectory=traj) is None
    assert not list((tmp_path / "a" / "images").rglob("*.webp"))

    ctx2 = make_ctx(tmp_path / "g", quota={"groundcua": 1}, config={"context_budget": 1})
    assert assemble_grounding(source="groundcua", trajectory_id="t", step_id="s",
        image_bytes=img, instruction="button", target_xy=(20,20), image_size=(100,80),
        target_width_px=10, target_height_px=10, app="x", ctx=ctx2) is None
    assert not list((tmp_path / "g" / "images").rglob("*.webp"))

    ctx3 = make_ctx(tmp_path / "r", quota={"replay": 1}, config={"context_budget": 1})
    assert assemble_replay(messages=[{"role":"user","content":"<image>\nWhat?"},{"role":"assistant","content":"Answer"}],
        images_pil=[Image.open(io.BytesIO(img))], source_name="replay", sample_id="id", task_type="replay_vqa",
        metadata={}, ctx=ctx3) is None
    assert not list((tmp_path / "r" / "images").rglob("*.webp"))


def test_source_specific_continuity_gaps_reset_history():
    import sources.procua as p
    import sources.pc_agent_e as pc
    import sources.videocua as v
    # ProCUA valid -> ambiguous multi-statement -> valid
    imgs={f"{i}.png":make_png_bytes(100,80,marker=i) for i in range(3)}
    pj={"trajectory_id":"x","goal":"Edit the sheet","steps":[{"actions":[
        {"screenshot":"0.png","pyautogui_command":"pyautogui.click(10,10)"},
        {"screenshot":"1.png","pyautogui_command":"pyautogui.click(1,1)\npyautogui.click(2,2)"},
        {"screenshot":"2.png","pyautogui_command":"pyautogui.click(30,30)"},]}]}
    t=p.parse_trajectory(pj, imgs); assert t and [s.metadata["continuity_id"] for s in t.steps]==[0,1]
    assert build_single(t,1).history_texts == []
    # PC-Agent-E same invariant.
    lines=[json.dumps({"action":"click (10,10)","screenshot":"0.png"}),
           json.dumps({"action":"not a real action","screenshot":"1.png"}),
           json.dumps({"action":"click (30,30)","screenshot":"2.png"})]
    t2=pc.build_trajectory("task1", "\n".join(lines), "**Description:** Edit it", imgs)
    assert t2 and [s.metadata["continuity_id"] for s in t2.steps]==[0,1]
    assert build_single(t2,1).history_texts == []
    # VideoCUA valid -> unprovable drag -> valid.
    log={"task_id":"v","task_instruction":"Edit it","platform":"app","action_log":[
        {"action_type":"CLICK","timestamp":1.0,"action_params":{"x":10,"y":10}},
        {"action_type":"DRAG_TO","timestamp":2.0,"action_params":{"x":20,"y":20}},
        {"action_type":"CLICK","timestamp":3.0,"action_params":{"x":30,"y":30}},]}
    frames={float(i+1):make_png_bytes(100,80,marker=i) for i in range(3)}
    t3=v.build_trajectory(log, frames, None)
    assert t3 and [s.metadata["continuity_id"] for s in t3.steps]==[0,1]
    assert build_single(t3,1).history_texts == []


def _tar_entries(order, traj, png):
    bio=io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w") as tf:
        for name,data in order:
            ti=tarfile.TarInfo(name); ti.size=len(data); tf.addfile(ti, io.BytesIO(data))
    bio.seek(0)
    tf=tarfile.open(fileobj=bio, mode="r:")
    try:
        for m in tf: yield m, tf
    finally:
        tf.close()


@pytest.mark.parametrize("json_first", [True, False])
def test_procua_stream_is_order_independent(tmp_path, monkeypatch, json_first):
    import sources.procua as p
    png=make_png_bytes(100,80,marker=1)
    traj={"trajectory_id":"abc","goal":"Click the button","steps":[{"actions":[{
        "screenshot":"0.png","pyautogui_command":"pyautogui.click(10,10)"}]}]}
    j=json.dumps(traj).encode()
    base="part/run/abc"
    order=[(f"{base}/trajectory.json",j),(f"{base}/0.png",png)] if json_first else [(f"{base}/0.png",png),(f"{base}/trajectory.json",j)]
    monkeypatch.setattr(p, "stream_tar_zst", lambda *a, **k: _tar_entries(order,traj,png))
    monkeypatch.setattr(p, "deterministic_keep", lambda *a, **k: True)
    monkeypatch.setattr(p, "samples_for_trajectory", lambda traj, ctx: [{"id":traj.trajectory_id,"n":len(traj.steps)}])
    ctx=_ctx(tmp_path,"procua",remaining=1)
    got=p._process_member_stream(ctx,"x",None)
    assert got == [{"id":"procua_abc","n":1}]


def test_validator_binds_metadata_dimensions_to_decoded_image(tmp_path):
    from processing.validation import validate_sample
    p = tmp_path / "images" / "procua" / "x.webp"; p.parent.mkdir(parents=True)
    Image.new("RGB", (100,80), "white").save(p)
    base={"messages":[{"role":"user","content":"<image>\nTask: click"},{"role":"assistant","content":"click(x=90, y=70)"}],
          "images":["images/procua/x.webp"],"source":"procua","trajectory_id":"t","step_id":"s","task_type":"action",
          "metadata":{"final_image_size":[1000,800]}}
    assert validate_sample(base, str(tmp_path)) == (False, "final_image_size_mismatch")
    base["metadata"]["final_image_size"]=[100,80]
    assert validate_sample(base, str(tmp_path)) == (True, "")
    base["images"].append("images/procua/x.webp")
    base["messages"][0]["content"] = "<image><image>\nTask: click"
    assert validate_sample(base, str(tmp_path))[1] == "action_requires_exactly_one_image"


def test_stable_image_name_distinguishes_long_ids_with_same_prefix():
    from processing.assemble import stable_image_name
    a = "x" * 180 + "A"
    b = "x" * 180 + "B"
    assert stable_image_name(a, ["step"]) != stable_image_name(b, ["step"])


def test_reduced_gui360_build_scales_canonical_cohorts(monkeypatch, tmp_path):
    import build_jxagent_dataset as b
    import sources.gui360 as g
    args = b.parse_args(["--output", str(tmp_path), "--sources", "gui360",
                         "--gui360-count", "838", "--allow-unversioned"])
    ctx = _ctx(tmp_path, "gui360", remaining=838)
    captured = {}
    def fake_run(c, grounding_want, understanding_want):
        captured.update(g=grounding_want, u=understanding_want)
        return []
    monkeypatch.setattr(g, "run", fake_run)
    b.run_source("gui360", ctx, args)
    # 2500/16000 and 2650/16000 of a representative 838-row GUI slice.
    assert captured == {"g": 131, "u": 139}
    assert 838 - captured["g"] - captured["u"] == 568
