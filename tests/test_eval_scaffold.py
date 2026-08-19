"""Scaffold config completeness + base/adapter parity enforcement tests."""
from __future__ import annotations

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.scaffold import (ScaffoldError, check_parity,
                                 effective_config, load_scaffold,
                                 scaffold_hash)

SCAFFOLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evaluation", "scaffold_config.yaml")


def test_shipped_scaffold_is_complete():
    cfg = load_scaffold(SCAFFOLD)  # raises on any missing/null key
    assert cfg["agent"]["system_prompt"] == "You are a computer-use agent."
    assert "finish" in cfg["agent"]["action_space"]
    assert cfg["sampling"]["temperature"] == 0.0  # deterministic-first


def test_parity_allows_only_arm_fields():
    cfg = load_scaffold(SCAFFOLD)
    base = effective_config(cfg, {"model": "/m/base", "adapter": None})
    adapter = effective_config(cfg, {"model": "/m/base", "adapter": "/ckpt",
                                     "checkpoint": "gate100"})
    ok, diffs = check_parity(base, adapter, cfg["arm_specific_keys"])
    assert ok and diffs == []


@pytest.mark.parametrize("section,key,value", [
    ("execution", "step_budget", 60),
    ("sampling", "temperature", 0.7),
    ("sampling", "seed", 999),
    ("observation", "resize_max_long_side", 1280),
    ("history", "limit", 10),
])
def test_parity_catches_score_moving_changes(section, key, value):
    cfg = load_scaffold(SCAFFOLD)
    broken = copy.deepcopy(cfg)
    broken[section][key] = value
    base = effective_config(cfg, {"model": "/m"})
    adapter = effective_config(broken, {"model": "/m", "adapter": "/x"})
    ok, diffs = check_parity(base, adapter, cfg["arm_specific_keys"])
    assert not ok
    assert any(d["field"] == f"{section}.{key}" for d in diffs)


def test_parity_catches_system_prompt_change():
    cfg = load_scaffold(SCAFFOLD)
    broken = copy.deepcopy(cfg)
    broken["agent"]["system_prompt"] = "You are a DIFFERENT agent."
    base = effective_config(cfg, {"model": "/m"})
    adapter = effective_config(broken, {"model": "/m", "adapter": "/x"})
    ok, diffs = check_parity(base, adapter, cfg["arm_specific_keys"])
    assert not ok and any("system_prompt" in d["field"] for d in diffs)


def test_missing_required_key_rejected(tmp_path):
    import yaml
    cfg = load_scaffold(SCAFFOLD)
    del cfg["execution"]["step_budget"]
    p = tmp_path / "broken.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ScaffoldError) as e:
        load_scaffold(str(p))
    assert "execution.step_budget" in str(e.value)


def test_null_value_rejected(tmp_path):
    import yaml
    cfg = load_scaffold(SCAFFOLD)
    cfg["sampling"]["seed"] = None
    p = tmp_path / "null.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ScaffoldError):
        load_scaffold(str(p))


def test_scaffold_hash_stable_and_sensitive():
    cfg = load_scaffold(SCAFFOLD)
    other = copy.deepcopy(cfg)
    assert scaffold_hash(cfg) == scaffold_hash(other)
    other["execution"]["step_budget"] = 51
    assert scaffold_hash(cfg) != scaffold_hash(other)
