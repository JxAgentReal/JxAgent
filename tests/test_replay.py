import json
import pytest
from PIL import Image

from processing.assemble import assemble_replay
from sources.replay import _cauldron_messages, _content_id, _select
from tests.conftest import make_ctx, make_png_bytes


def test_content_id_deterministic():
    assert _content_id("hello") == _content_id("hello")
    assert _content_id("hello") != _content_id("hellp")


def test_select_deterministic():
    assert _select(7, "magicoder", "coding") == _select(7, "magicoder", "coding")


def test_cauldron_messages_structure():
    row = {
        "images": [{"bytes": make_png_bytes(224, 224), "path": "i.jpg"}],
        "texts": [{"user": "What color is the ball?", "assistant": "Red."}],
    }
    messages, images = _cauldron_messages(row)
    assert len(images) == 1
    assert messages[0]["content"].startswith("<image>")
    assert messages[0]["role"] == "user"
    assert messages[1] == {"role": "assistant", "content": "Red."}


def test_replay_text_only_sample(tmp_path):
    ctx = make_ctx(tmp_path, quota={"replay": 10})
    sample = assemble_replay(
        messages=[{"role": "user", "content": "write a function"},
                  {"role": "assistant", "content": "def f(): pass"}],
        images_pil=[], source_name="replay", sample_id="replay_coding_ab",
        task_type="replay_coding",
        metadata={"replay_source": "Magicoder", "license": "apache-2.0"}, ctx=ctx)
    assert sample is not None
    assert sample["images"] == []
    assert sample["source"] == "replay"
    assert sample["task_type"] == "replay_coding"
    ok, reason = __import__("processing.validation", fromlist=["validate_sample"]) \
        .validate_sample(sample, str(tmp_path))
    assert ok, reason


def test_replay_multimodal_sample(tmp_path):
    ctx = make_ctx(tmp_path, quota={"replay": 10})
    img = Image.open(__import__("io", fromlist=["BytesIO"]).BytesIO(
        make_png_bytes(500, 400))).convert("RGB")
    sample = assemble_replay(
        messages=[{"role": "user", "content": "<image>\nWhat is shown?"},
                  {"role": "assistant", "content": "A chart."}],
        images_pil=[img], source_name="replay", sample_id="replay_vqa_xy",
        task_type="replay_vqa", metadata={"replay_source": "cauldron"}, ctx=ctx)
    assert sample is not None
    assert len(sample["images"]) == 1
    assert sample["images"][0].startswith("images/replay/")
    assert "\\" not in sample["images"][0]
    import os
    assert os.path.exists(os.path.join(str(tmp_path), *sample["images"][0].split("/")))


def test_replay_rejects_empty_assistant(tmp_path):
    ctx = make_ctx(tmp_path, quota={"replay": 10})
    sample = assemble_replay(
        messages=[{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "   "}],
        images_pil=[], source_name="replay", sample_id="x",
        task_type="replay_math", metadata={}, ctx=ctx)
    assert sample is None


def test_replay_categories_total_7500():
    from sources.replay import CATEGORIES
    assert sum(n for _, n in CATEGORIES.values()) == 7500
    assert set(CATEGORIES) == {"coding", "math", "instruction", "vqa", "tool"}


def test_replay_vqa_fails_closed_on_stream_creation_error(tmp_path, monkeypatch):
    """A cauldron stream-creation failure must raise, not silently skip."""
    import sources.replay as replay_mod

    def boom(ctx, repo, config, split="train"):
        raise RuntimeError("simulated HF stream creation failure")

    monkeypatch.setattr(replay_mod, "_iter_dataset", boom)
    with pytest.raises(RuntimeError, match="simulated HF stream creation failure"):
        replay_mod.replay_vqa(make_ctx(tmp_path, quota={"replay": 10}), 5)


def test_replay_vqa_fallback_on_primary_failure(tmp_path, monkeypatch):
    """If aokvqa stream creation fails, replay_vqa must fall through to ai2d."""
    import sources.replay as replay_mod

    call_log = []

    def selective_boom(ctx, repo, config, split="train"):
        call_log.append(config)
        if config == "aokvqa":
            raise RuntimeError("aokvqa auth failure")
        return iter([{"texts": [{"user": "Q?", "assistant": "A."}],
                      "images": [make_png_bytes(100, 100)]}])

    monkeypatch.setattr(replay_mod, "_iter_dataset", selective_boom)
    monkeypatch.setattr(replay_mod, "_select", lambda *a, **k: True)
    ctx = make_ctx(tmp_path, quota={"replay": 10})
    # Should NOT raise — ai2d fallback succeeds
    got = replay_mod.replay_vqa(ctx, 5)
    assert "aokvqa" in call_log
    assert "ai2d" in call_log


def test_replay_tool_fails_closed_on_stream_creation_error(tmp_path, monkeypatch):
    """A hermes stream-creation failure must raise, not silently skip."""
    import sources.replay as replay_mod

    def boom(ctx, repo, config, split="train"):
        raise RuntimeError("simulated HF stream creation failure")

    monkeypatch.setattr(replay_mod, "_iter_dataset", boom)
    with pytest.raises(RuntimeError, match="simulated HF stream creation failure"):
        replay_mod.replay_tool(make_ctx(tmp_path, quota={"replay": 10}), 5)


def test_replay_tool_survives_single_config_failure(tmp_path, monkeypatch):
    """If one Hermes config fails but the other succeeds, tool samples load."""
    import sources.replay as replay_mod

    def selective_boom(ctx, repo, config, split="train"):
        if config == "glaive_func_calling":
            raise RuntimeError("glaive auth failure")
        return iter([{"conversations": [
            {"from": "human", "value": "call weather for Paris"},
            {"from": "gpt", "value": '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>'},
        ], "tools": json.dumps([{
            "type": "function", "function": {"name": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}
        }])}])

    monkeypatch.setattr(replay_mod, "_iter_dataset", selective_boom)
    monkeypatch.setattr(replay_mod, "_select", lambda *a, **k: True)
    ctx = make_ctx(tmp_path, quota={"replay": 10})
    got = replay_mod.replay_tool(ctx, 3)
    assert len(got) >= 1  # at least one from func_calling_singleturn


def test_replay_run_fails_closed_on_empty_mandatory_category(tmp_path, monkeypatch):
    """run() must abort if a mandatory category loads zero samples."""
    import sources.replay as replay_mod

    monkeypatch.setitem(
        replay_mod.CATEGORIES, "coding",
        (lambda ctx, want: [], replay_mod.CATEGORIES["coding"][1]))
    ctx = make_ctx(tmp_path, quota={"replay": 10})
    with pytest.raises(RuntimeError, match="mandatory Replay category 'coding'"):
        replay_mod.run(ctx, {"coding": 5})


def test_hermes_style_conversion_shape():
    """The adapter maps from/value rows incl. function replies to user turns."""
    convs = [
        {"from": "system", "value": "You are a function calling model."},
        {"from": "human", "value": "Get the stock price for ACME"},
        {"from": "assistant", "value": "<tool_call>get_stock_price</tool_call>"},
        {"from": "function", "value": "{\"price\": 42}"},
    ]
    roles = []
    for c in convs:
        frm = c["from"].lower()
        role = {"system": "system", "gizmo": "system", "human": "user",
                "function": "user", "tool": "user"}.get(frm, "assistant")
        roles.append(role)
    assert roles == ["system", "user", "assistant", "user"]
