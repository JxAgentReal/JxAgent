"""Offline tests for the MI300X training infrastructure.

No GPU, no network, no model weights required. Covers:
  - world-size aware effective-batch computation (exact / non-divisible /
    unreachable target / per-device > 1)
  - optimizer-step and checkpoint-gate math (20% / 55% / 100%)
  - gate-checkpoint preservation + pruning
  - real train.jsonl sample counting (incl. missing trailing newline)
  - LoRA module selection by FULL path, including synthetic architectures
    that reuse identical leaf names (gate_proj/up_proj/down_proj/qkv) in the
    vision tower and the text decoder
  - target regex behavior (covers text, never vision/aligner/lm_head)
  - adapter-key -> base-path classification (peft naming)
  - train.sh vs resume.sh effective-argument parity via JXAGENT_DRY_RUN
  - throughput window math (startup/preprocessing excluded by construction)
  - bash -n syntax validation of every mi300x shell script
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MI300X = os.path.join(REPO, "mi300x")
sys.path.insert(0, MI300X)

import inspect_modules as im  # noqa: E402
import log_tap as lt  # noqa: E402
import verify_freeze as vf  # noqa: E402

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")

COMMON = os.path.join(MI300X, "common.sh")


def bash_common(expr: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("JXAGENT_GPUS", None)
    env.update(env_extra or {})
    cmd = f'source "{COMMON}"; {expr}'
    return subprocess.run([BASH, "-c", cmd], capture_output=True, text=True,
                          env=env, cwd=MI300X)


# ------------------------------------------------------------------ topology

@pytest.mark.parametrize("world,per_device,ga,eff", [
    (1, 1, 32, 32),
    (2, 1, 16, 32),
    (4, 1, 8, 32),
    (8, 1, 4, 32),
    (2, 2, 8, 32),
    (16, 2, 1, 32),
])
def test_effective_batch_exact(world, per_device, ga, eff):
    r = bash_common(
        'JXAGENT_GPUS="" PER_DEVICE=""; JXAGENT_GPUS=%d JXAGENT_PER_DEVICE_BATCH=%d jx_compute_topology; '
        'echo "$GRAD_ACCUM $EFFECTIVE_BATCH"' % (world, per_device),
        env_extra={"JXAGENT_GPUS": str(world), "JXAGENT_PER_DEVICE_BATCH": str(per_device)},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"{ga} {eff}"


def test_effective_batch_not_divisible_picks_nearest_and_warns():
    r = bash_common(
        'jx_compute_topology; echo "$GRAD_ACCUM $EFFECTIVE_BATCH"',
        env_extra={"JXAGENT_GPUS": "3"},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "11 33"      # nearest to 32
    assert "NOT 32" in r.stderr                              # never silent


def test_effective_batch_unreachable_target_warns():
    r = bash_common(
        'jx_compute_topology; echo "$GRAD_ACCUM $EFFECTIVE_BATCH"',
        env_extra={"JXAGENT_GPUS": "64"},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "1 64"
    assert "unreachable" in r.stderr


def test_plan_block_prints_all_six_fields():
    r = bash_common(
        'jx_compute_topology; jx_print_plan 100003 3126',
        env_extra={"JXAGENT_GPUS": "8"},
    )
    for field in ("GPU count (world size)", "per device batch",
                  "gradient accumulation", "GLOBAL effective batch",
                  "train sample count", "optimizer steps per epoch"):
        assert field in r.stdout
    assert "4" in r.stdout and "32" in r.stdout


# -------------------------------------------------------------- step/gates

def test_optimizer_steps_ceiling():
    r = bash_common('echo "$(jx_optimizer_steps 100003 32)"; '
                    'echo "$(jx_optimizer_steps 96000 32)"; '
                    'echo "$(jx_optimizer_steps 1 32)"')
    assert r.stdout.split() == ["3126", "3000", "1"]


def test_checkpoint_gates():
    r = bash_common('jx_compute_gates 3125; echo "$SAVE_INTERVAL $EARLY_STEP $MID_STEP $FINAL_STEP"')
    # interval = 625 (exactly 20%), middle = nearest multiple to 55% (1718.75 -> 1875)
    assert r.stdout.strip() == "625 625 1875 3125"


def test_checkpoint_gates_odd_total():
    r = bash_common('jx_compute_gates 3126; echo "$SAVE_INTERVAL $EARLY_STEP $MID_STEP $FINAL_STEP"')
    assert r.stdout.strip() == "625 625 1875 3126"


def test_checkpoint_gates_tiny_total():
    r = bash_common('jx_compute_gates 3; echo "$SAVE_INTERVAL $EARLY_STEP $MID_STEP $FINAL_STEP"')
    assert r.stdout.strip() == "1 1 2 3"


def test_preserve_gates_keeps_early_middle_final_and_prunes_rest(tmp_path):
    out = tmp_path / "out"
    steps = [625, 1250, 1875, 2500, 3125]
    for s in steps:
        ck = out / "v0-20260816" / f"checkpoint-{s}"
        ck.mkdir(parents=True)
        (ck / "adapter_model.safetensors").write_text(f"step={s}")
    r = bash_common(
        'EARLY_STEP=625 MID_STEP=1875 FINAL_STEP=3125 jx_preserve_gates "%s"' % out.as_posix()
    )
    assert r.returncode == 0, r.stderr
    gates = out / "gates"
    for name, s in (("early", 625), ("middle", 1875), ("final", 3125)):
        f = gates / name / "adapter_model.safetensors"
        assert f.exists(), name
        assert f.read_text() == f"step={s}"
    # non-gate periodic checkpoints pruned, gate sources + latest kept
    remaining = sorted(p.name for p in (out / "v0-20260816").iterdir())
    assert remaining == ["checkpoint-1875", "checkpoint-3125", "checkpoint-625"]


# ------------------------------------------------------------ dataset count

def test_sample_count_counts_lines_without_trailing_newline(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    (final / "train.jsonl").write_text('{"a": 1}\n{"a": 2}\n{"a": 3}')  # no trailing newline
    r = bash_common('DATA_DIR="%s" jx_train_sample_count' % tmp_path.as_posix())
    assert r.stdout.strip() == "3"


def test_sample_count_override_warns(tmp_path):
    r = bash_common('DATA_DIR="%s" jx_train_sample_count' % tmp_path.as_posix(),
                    env_extra={"JXAGENT_TRAIN_SAMPLES": "7"})
    assert r.stdout.strip() == "7"
    assert "overrides" in r.stderr


def test_sample_count_missing_file_fails(tmp_path):
    r = bash_common('DATA_DIR="%s" jx_train_sample_count' % tmp_path.as_posix())
    assert r.returncode == 1
    assert "train split not found" in r.stderr


# ------------------------------------------------- LoRA module selection

def test_classify_identical_leaf_names_in_vision_and_text():
    # the exact collision from the audit: vision MLP reuses text leaf names
    assert im.classify_path("visual.blocks.0.mlp.gate_proj") == "vision"
    assert im.classify_path("visual.blocks.0.mlp.up_proj") == "vision"
    assert im.classify_path("visual.blocks.0.mlp.down_proj") == "vision"
    assert im.classify_path("visual.blocks.0.attn.qkv") == "vision"
    assert im.classify_path("model.layers.0.mlp.gate_proj") == "text"
    assert im.classify_path("model.layers.0.mlp.up_proj") == "text"
    assert im.classify_path("model.layers.0.mlp.down_proj") == "text"
    assert im.classify_path("model.layers.0.self_attn.q_proj") == "text"
    assert im.classify_path("model.layers.0.self_attn.qkv") == "text"


def test_classify_aligner_variants():
    assert im.classify_path("visual.merger.mlp.0") == "aligner"
    assert im.classify_path("model.multi_modal_projector.linear_1") == "aligner"
    assert im.classify_path("model.vision_merger.mlp.2") == "aligner"


def test_classify_excluded_heads():
    assert im.classify_path("lm_head") == "lm_head"
    assert im.classify_path("model.embed_tokens") == "embedding"


def test_select_lm_linears_keeps_text_mlp_drops_vision():
    paths = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.gate_proj",          # text MLP must survive
        "model.layers.11.mlp.down_proj",
        "visual.blocks.0.mlp.gate_proj",         # vision MLP must be excluded
        "visual.blocks.1.attn.qkv",
        "visual.merger.mlp.0",                   # aligner
        "lm_head",
        "model.embed_tokens",
    ]
    parts = im.select_lm_linears(paths)
    assert "model.layers.0.mlp.gate_proj" in parts["selected"]
    assert "model.layers.11.mlp.down_proj" in parts["selected"]
    assert not any(p.startswith("visual.") for p in parts["selected"])
    assert len(parts["vision"]) == 2
    assert len(parts["aligner"]) == 1
    assert set(parts["excluded"]) == {"lm_head", "model.embed_tokens"}


def test_target_regex_covers_text_not_vision():
    paths = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.7.mlp.gate_proj",
        "visual.blocks.0.mlp.gate_proj",
        "visual.blocks.0.attn.qkv",
        "visual.merger.mlp.0",
        "lm_head",
    ]
    regex = im.build_target_regex(im.select_lm_linears(paths)["selected"])
    assert re.fullmatch(regex, "model.layers.999.mlp.gate_proj")     # any layer
    assert re.fullmatch(regex, "model.layers.999.self_attn.q_proj")
    assert not re.fullmatch(regex, "visual.blocks.999.mlp.gate_proj")
    assert not re.fullmatch(regex, "visual.blocks.0.attn.qkv")
    assert not re.fullmatch(regex, "lm_head")
    assert not re.fullmatch(regex, "model.embed_tokens")


def test_path_to_pattern_escapes_dots():
    assert im.path_to_pattern("model.layers.12.mlp.gate_proj") == \
        r"model\.layers\.\d+\.mlp\.gate_proj"


# ------------------------------------------------- adapter key classification

def test_adapter_key_parsing():
    assert vf.lora_param_to_base(
        "base_model.model.model.layers.3.mlp.up_proj.lora_A.weight"
    ) == "model.layers.3.mlp.up_proj"
    assert vf.lora_param_to_base(
        "model.layers.3.self_attn.q_proj.lora_B.weight"
    ) == "model.layers.3.self_attn.q_proj"
    assert vf.is_lora_param("base_model.model.model.layers.3.mlp.up_proj.lora_A.weight")
    assert not vf.is_lora_param("model.layers.3.mlp.up_proj.weight")


def test_adapter_vision_key_detected_by_full_path():
    base = vf.lora_param_to_base("model.visual.blocks.1.attn.qkv.lora_B.weight")
    assert base == "model.visual.blocks.1.attn.qkv"
    assert im.classify_path(base) == "vision"      # the old substring check missed this


def test_sample_names_deterministic_and_capped():
    names = [f"visual.blocks.{i}.mlp.gate_proj" for i in range(1000)]
    a = vf.sample_names(names, cap=64)
    b = vf.sample_names(names, cap=64)
    assert a == b and len(a) == 64
    assert a[0] == names[0] and a[-1] == names[-1]
    assert len(vf.sample_names(names[:10], cap=64)) == 10


# ------------------------------------------------- train/resume parity

REGEX = r"(?:model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj)|mlp\.(?:gate_proj|up_proj)))"

CRITICAL_ARGS = [
    "--model", "--train_type", "--lora_rank", "--lora_alpha", "--lora_dropout",
    "--target_modules", "--freeze_vit", "--freeze_aligner", "--torch_dtype",
    "--dataset", "--val_dataset", "--max_length", "--num_train_epochs",
    "--per_device_train_batch_size", "--per_device_eval_batch_size",
    "--gradient_accumulation_steps", "--learning_rate", "--lr_scheduler_type",
    "--warmup_ratio", "--gradient_checkpointing", "--save_steps",
    "--save_total_limit", "--logging_steps", "--dataloader_num_workers",
    "--eval_steps",
]


def _dry_run(script: str, env_extra: dict) -> str:
    env = dict(os.environ)
    env.update(env_extra)
    env["JXAGENT_DRY_RUN"] = "1"
    r = subprocess.run([BASH, os.path.join(MI300X, script)],
                       capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, f"{script} dry-run failed:\n{r.stdout}\n{r.stderr}"
    lines = [l for l in r.stdout.splitlines() if l.startswith("NPROC_PER_NODE=")]
    assert len(lines) == 1, f"{script} did not print exactly one command:\n{r.stdout}"
    return lines[0]


def _parse_cmd(line: str) -> tuple[dict, list]:
    tokens = shlex.split(line)
    assert tokens[0].startswith("NPROC_PER_NODE=")
    assert tokens[1:3] == ["swift", "sft"]
    args, i = {}, 3
    while i < len(tokens):
        assert tokens[i].startswith("--"), tokens[i]
        args[tokens[i]] = tokens[i + 1]
        i += 2
    return args, tokens


@pytest.fixture()
def parity_env(tmp_path):
    model_dir = tmp_path / "model"
    data_dir = tmp_path / "data" / "final"
    data_dir.mkdir(parents=True)
    (data_dir / "train.jsonl").write_text('{"x": 1}\n')
    (data_dir / "validation.jsonl").write_text('{"x": 1}\n')
    modules = tmp_path / "lora_modules.json"
    modules.write_text(json.dumps({"target_modules": [REGEX]}))
    out = tmp_path / "out"
    ckpt = out / "v0-smoke" / "checkpoint-5"
    ckpt.mkdir(parents=True)
    return {
        "JXAGENT_MODEL_DIR": model_dir.as_posix(),
        "JXAGENT_DATA_DIR": (tmp_path / "data").as_posix(),
        "JXAGENT_MODULES_JSON": modules.as_posix(),
        "JXAGENT_TRAIN_OUT": out.as_posix(),
        "JXAGENT_TRAIN_SAMPLES": "100000",
        "JXAGENT_GPUS": "8",
    }


def test_train_resume_argument_parity(parity_env):
    train_line = _dry_run("train.sh", parity_env)
    resume_line = _dry_run("resume.sh", parity_env)

    train_args, train_tokens = _parse_cmd(train_line)
    resume_args, resume_tokens = _parse_cmd(resume_line)

    # same launcher / world size
    assert train_tokens[0] == resume_tokens[0] == "NPROC_PER_NODE=8"

    # every critical argument identical
    for flag in CRITICAL_ARGS:
        assert flag in train_args, f"train.sh missing {flag}"
        assert train_args[flag] == resume_args[flag], \
            f"parity broken at {flag}: {train_args[flag]!r} != {resume_args[flag]!r}"

    # evaluation flag (name varies across transformers versions) identical
    t_eval = {k: v for k, v in train_args.items() if k.endswith("_strategy")}
    r_eval = {k: v for k, v in resume_args.items() if k.endswith("_strategy")}
    assert t_eval and t_eval == r_eval

    # attention backend identical (topology parity for the benchmark too)
    assert train_args.get("--attn_implementation") == \
        resume_args.get("--attn_implementation")

    # resume-only argument
    assert resume_args["--resume_from_checkpoint"].endswith("checkpoint-5")
    assert "--resume_from_checkpoint" not in train_args

    # faithful resume: save_only_model must never appear
    assert "--save_only_model" not in train_args
    assert "--save_only_model" not in resume_args

    # world-size aware batch: 8 GPUs -> GA 4 -> global 32
    assert train_args["--gradient_accumulation_steps"] == "4"
    assert train_args["--per_device_train_batch_size"] == "1"

    # target regex comes verbatim from lora_modules.json
    assert train_args["--target_modules"] == REGEX

    # checkpoint interval derived from the real sample count: ceil(100000/32)/5
    assert train_args["--save_steps"] == "625"
    assert train_args["--eval_steps"] == "625"


def test_throughput_dry_run_uses_training_topology(parity_env):
    env = dict(parity_env)
    env["JXAGENT_THROUGHPUT_OUT"] = env["JXAGENT_TRAIN_OUT"].replace("out", "out_tp")
    line = _dry_run("throughput_test.sh", env)
    args, tokens = _parse_cmd(line)
    assert tokens[0] == "NPROC_PER_NODE=8"                      # same GPUs
    assert args["--gradient_accumulation_steps"] == "4"         # same eff batch
    assert args["--max_length"] == "8192"
    assert args["--gradient_checkpointing"] == "true"
    assert args["--max_steps"] == "110"                          # 10 warmup + 100 measured
    assert args["--logging_steps"] == "1"


def test_train_refuses_without_confirmation(parity_env):
    env = dict(os.environ)
    env.update(parity_env)
    env.pop("JXAGENT_DRY_RUN", None)
    env.pop("JXAGENT_CONFIRM_TRAIN", None)
    r = subprocess.run([BASH, os.path.join(MI300X, "train.sh")],
                       capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 1
    assert "JXAGENT_CONFIRM_TRAIN=1" in r.stderr


def test_resume_refuses_without_confirmation(parity_env):
    env = dict(os.environ)
    env.update(parity_env)
    env.pop("JXAGENT_DRY_RUN", None)
    env.pop("JXAGENT_CONFIRM_TRAIN", None)
    r = subprocess.run([BASH, os.path.join(MI300X, "resume.sh")],
                       capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 1
    assert "JXAGENT_CONFIRM_TRAIN=1" in r.stderr


# ------------------------------------------------- throughput window math

def test_parse_log_step():
    assert lt.parse_log_step("{'loss': 2.31, 'epoch': 0.01, 'step': 7}") == 7
    assert lt.parse_log_step("some other line") is None
    assert lt.parse_log_step("{'eval_loss': 1.0, 'step': 9}") is None


def test_window_stats_excludes_startup_and_warmup():
    events = []
    t = 1000.0                       # huge startup/preprocessing time before step 1
    for s in range(1, 111):
        t += 1.0
        if s == 5:
            t += 50.0                # hiccup inside warmup: must not count
        events.append({"step": s, "ts": t})
    w = lt.window_stats(events, warmup=10, measure=100)
    assert w["measured_seconds"] == pytest.approx(100.0)
    assert w["seconds_per_step_mean"] == pytest.approx(1.0)
    assert w["seconds_per_step_median"] == pytest.approx(1.0)
    assert w["seconds_per_step_p90"] == pytest.approx(1.0)
    assert w["steps_per_second"] == pytest.approx(1.0)
    assert w["warmup_seconds"] == pytest.approx(59.0)   # 9 steps + the 50 s hiccup


def test_window_stats_requires_full_window():
    events = [{"step": s, "ts": float(s)} for s in range(1, 50)]
    with pytest.raises(ValueError):
        lt.window_stats(events, warmup=10, measure=100)


def test_epoch_hours():
    assert lt.epoch_hours(3600, 1.0) == pytest.approx(1.0)


def test_parse_vram_peaks():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write("1719000000\n")
        f.write("GPU,VRAM(%),GFX(%)\n")
        f.write("0,70.5,95.0\n")
        f.write("1719000005\n")
        f.write("0,88.0,99.5\n")
        path = f.name
    try:
        peaks = lt.parse_vram_peaks(path)
        assert peaks["VRAM(%)"] == 88.0
        assert peaks["GFX(%)"] == 99.5
    finally:
        os.unlink(path)


# ------------------------------------------------- shell syntax

def test_all_shell_scripts_pass_bash_n():
    scripts = sorted(
        os.path.join(MI300X, f) for f in os.listdir(MI300X) if f.endswith(".sh")
    )
    assert scripts, "no shell scripts found"
    for s in scripts:
        r = subprocess.run([BASH, "-n", s], capture_output=True, text=True)
        assert r.returncode == 0, f"{s}: {r.stderr}"


def test_python_infrastructure_files_compile():
    import py_compile
    for f in ("inspect_modules.py", "verify_freeze.py", "log_tap.py"):
        py_compile.compile(os.path.join(MI300X, f), doraise=True)
