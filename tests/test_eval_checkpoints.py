"""Checkpoint discovery + gate mapping tests (names never trusted)."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.checkpoints import (CheckpointInfo, discover_checkpoints,
                                    map_to_gates, require_gates)


def _mk_ckpt(root, name, global_step, epoch=None, adapter=True):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    state = {"global_step": global_step}
    if epoch is not None:
        state["epoch"] = epoch
    with open(os.path.join(d, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)
    if adapter:
        open(os.path.join(d, "adapter_model.safetensors"), "wb").write(b"x")
    return d


def test_discovery_reads_trainer_state_not_names(tmp_path):
    root = str(tmp_path)
    # directory NAME says step 9999 but trainer_state says 600 (20% of 3000)
    _mk_ckpt(root, "checkpoint-9999", 600)
    ckpts = discover_checkpoints(root, total_optimizer_steps=3000)
    assert len(ckpts) == 1
    assert ckpts[0].global_step == 600
    assert ckpts[0].epoch_fraction_pct == pytest.approx(20.0)


def test_gates_layout_from_parallel_training_agent(tmp_path):
    # mi300x/common.sh preserves gates as gates/{early,middle,final}
    root = str(tmp_path)
    _mk_ckpt(os.path.join(root, "gates"), "early", 596)
    _mk_ckpt(os.path.join(root, "gates"), "middle", 1639)
    _mk_ckpt(os.path.join(root, "gates"), "final", 2980)
    ckpts = discover_checkpoints(root, total_optimizer_steps=2980)
    by_name = {os.path.basename(c.path): c for c in ckpts}
    assert set(by_name) == {"early", "middle", "final"}
    gates = map_to_gates(ckpts)
    assert gates[20].name == "early" and gates[55].name == "middle"
    assert gates[100].name == "final"


def test_require_gates_fails_loudly_when_missing(tmp_path):
    root = str(tmp_path)
    _mk_ckpt(root, "checkpoint-1490", 1490)  # only ~50%
    with pytest.raises(FileNotFoundError) as e:
        require_gates(discover_checkpoints(root, total_optimizer_steps=2980))
    assert "20" in str(e.value) or "100" in str(e.value)


def test_checkpoints_without_adapter_excluded(tmp_path):
    root = str(tmp_path)
    _mk_ckpt(root, "checkpoint-596", 596, adapter=False)
    _mk_ckpt(root, "checkpoint-2980", 2980)
    gates = map_to_gates(discover_checkpoints(root, total_optimizer_steps=2980))
    assert gates[20] is None      # weights missing -> gate unusable
    assert gates[100] is not None


def test_epoch_fraction_preferred_over_guess(tmp_path):
    root = str(tmp_path)
    _mk_ckpt(root, "checkpoint-x", 500, epoch=0.55)
    ckpts = discover_checkpoints(root, total_optimizer_steps=1000)
    assert ckpts[0].epoch_fraction_pct == pytest.approx(55.0)
