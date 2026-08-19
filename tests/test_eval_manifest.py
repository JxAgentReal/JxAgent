"""Run manifest + baseline-first compatibility tests."""
from __future__ import annotations

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import benchmarks as bm
from evaluation.run_manifest import (BaselineRequiredError, build_manifest,
                                     check_compatibility, load_manifest,
                                     require_baseline, save_manifest)
from evaluation.scaffold import load_scaffold

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAFFOLD = os.path.join(HERE, "evaluation", "scaffold_config.yaml")
BCFG = os.path.join(HERE, "evaluation", "osworld_config.yaml")


def _manifest(arm, model_revision="rev-a", adapter="x" , task_hash="h1"):
    cfg = load_scaffold(SCAFFOLD)
    bench = bm.get_benchmark("osworld_verified", BCFG)
    return build_manifest(arm=arm, scaffold_cfg=cfg, benchmark=bench,
                          task_ids=["t1", "t2"], task_set_hash=task_hash,
                          model_repo="Qwen/Qwen3.8-27B",
                          model_revision=model_revision,
                          adapter_revision=adapter if arm == "adapter" else None)


def test_manifest_roundtrip(tmp_path):
    m = _manifest("base")
    save_manifest(str(tmp_path), m)
    loaded = load_manifest(str(tmp_path / "manifest.json"))
    assert loaded == m
    for key in ("git_commit", "system_prompt_hash", "sampling_settings",
                "step_budget", "seed", "timeout_policy", "retry_policy",
                "runtime_environment", "benchmark", "model", "adapter"):
        assert key in loaded


def test_compatible_arms(tmp_path):
    ok, diffs = check_compatibility(_manifest("base"), _manifest("adapter"))
    assert ok and diffs == []


def test_incompatible_task_set_detected():
    ok, diffs = check_compatibility(_manifest("base", task_hash="h1"),
                                    _manifest("adapter", task_hash="h2"))
    assert not ok and any(d["field"] == "benchmark.task_set_hash" for d in diffs)


def test_incompatible_base_weights_detected():
    # adapter mounted on a DIFFERENT base revision is not comparable
    ok, diffs = check_compatibility(_manifest("base", model_revision="rev-a"),
                                    _manifest("adapter", model_revision="rev-b"))
    assert not ok and any(d["field"] == "model.revision" for d in diffs)


def test_incompatible_sampling_detected():
    b = _manifest("base")
    a = _manifest("adapter")
    a["sampling_settings"]["temperature"] = 0.7
    ok, diffs = check_compatibility(b, a)
    assert not ok and any("sampling" in d["field"] for d in diffs)


def test_baseline_first_enforcement(tmp_path):
    adapter = _manifest("adapter")
    # no base manifest -> comparative scoring refused
    with pytest.raises(BaselineRequiredError):
        require_baseline(adapter, None)
    # allowed explicitly for syntax testing, marked NOT comparable
    gate = require_baseline(adapter, None, allow_without_baseline=True)
    assert gate["comparable"] is False and gate["reason"] == "NO_BASE_MANIFEST"
    # incompatible base -> refused
    with pytest.raises(BaselineRequiredError):
        require_baseline(adapter, _manifest("base", task_hash="different"))
    # compatible base -> allowed
    gate = require_baseline(adapter, _manifest("base"))
    assert gate["comparable"] is True


def test_manifest_records_dry_run_flag():
    assert _manifest("base")["dry_run"] is False
