"""SOTA claim guard tests: a higher number is never sufficient."""
from __future__ import annotations

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import benchmarks as bm
from evaluation.sota_guard import evaluate_claim
from evaluation.run_manifest import build_manifest
from evaluation.scaffold import load_scaffold

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAFFOLD = os.path.join(HERE, "evaluation", "scaffold_config.yaml")
BCFG = os.path.join(HERE, "evaluation", "osworld_config.yaml")


def _manifest(arm, task_hash="h", model_revision="r1", dry_run=False):
    cfg = load_scaffold(SCAFFOLD)
    bench = bm.get_benchmark("osworld_verified", BCFG)
    m = build_manifest(arm=arm, scaffold_cfg=cfg, benchmark=bench,
                       task_ids=[f"t{i}" for i in range(10)],
                       task_set_hash=task_hash, model_repo="Qwen/Qwen3.8-27B",
                       model_revision=model_revision,
                       adapter_revision="a1" if arm == "adapter" else None,
                       dry_run=dry_run)
    return m


def _claim(base_out, adapter_out, **kw):
    defaults = dict(base_manifest=_manifest("base"),
                    adapter_manifest=_manifest("adapter"),
                    base_outcomes=base_out, adapter_outcomes=adapter_out,
                    base_aggregate={"accounting_complete": True},
                    adapter_aggregate={"accounting_complete": True},
                    benchmark=bm.get_benchmark("osworld_verified", BCFG))
    defaults.update(kw)
    return evaluate_claim(**defaults)


def test_higher_number_alone_is_not_comparable():
    base = {f"t{i}": False for i in range(10)}
    adapter = {f"t{i}": True for i in range(10)}
    claim = _claim(base, adapter)
    # A perfect statistically-significant win still does not reach SOTA while
    # published-scaffold equivalence is undemonstrated and the Verified
    # protocol is unpinned; it may at most beat the LOCAL baseline.
    assert claim["sota_claim_allowed"] == "NO"
    assert claim["label"] == "BEATS_LOCAL_BASELINE"
    assert "published_scaffold_equivalence_demonstrated" in claim["failed_conditions"]
    assert "POTENTIAL_SOTA" != claim["label"]


def test_dry_run_baseline_rejected():
    base = {f"t{i}": False for i in range(10)}
    adapter = {f"t{i}": True for i in range(10)}
    claim = _claim(base, adapter,
                   base_manifest=_manifest("base", dry_run=True))
    assert "local_base_run_exists" in claim["failed_conditions"]
    assert claim["label"] == "NOT_COMPARABLE"


def test_incomplete_denominator_blocks_claim():
    base = {f"t{i}": False for i in range(10)}
    adapter = {f"t{i}": True for i in range(10)}
    claim = _claim(base, adapter, adapter_aggregate={"accounting_complete": False})
    assert "complete_denominator_accounting" in claim["failed_conditions"]


def test_statistically_supported_win_required():
    base = {f"t{i}": (i % 2 == 0) for i in range(10)}
    adapter = {f"t{i}": (i % 2 == 0) for i in range(10)}
    adapter["t1"] = True  # +1 task = noise-scale delta
    claim = _claim(base, adapter)
    assert "statistically_supported_win" in claim["failed_conditions"]


def test_task_set_mismatch_not_comparable():
    base = {f"t{i}": False for i in range(10)}
    adapter = {f"t{i}": True for i in range(10)}
    claim = _claim(base, adapter,
                   adapter_manifest=_manifest("adapter", task_hash="OTHER"))
    assert "same_benchmark_identity_and_task_set" in claim["failed_conditions"]
    assert claim["label"] == "NOT_COMPARABLE"


def test_potential_sota_requires_everything():
    base = {f"t{i}": False for i in range(10)}
    adapter = {f"t{i}": True for i in range(10)}
    bench = copy.deepcopy(bm.get_benchmark("osworld_verified", BCFG))
    # resolve every Verified protocol field
    for f in ["repository_revision", "task_list_source", "task_count",
              "task_id_hash", "environment_definition", "step_budget",
              "scoring_method"]:
        bench[f] = "pinned"
    claim = _claim(base, adapter, benchmark=bench,
                   contamination_blockers=[],
                   published_scaffold_equivalence={"demonstrated": True})
    assert claim["label"] == "POTENTIAL_SOTA"
    assert claim["sota_claim_allowed"] == "YES"


def test_unresolved_verified_protocol_blocks_sota_forever():
    base = {f"t{i}": False for i in range(10)}
    adapter = {f"t{i}": True for i in range(10)}
    bench = bm.get_benchmark("osworld_verified", BCFG)  # unresolved fields
    claim = _claim(base, adapter, benchmark=bench, contamination_blockers=[],
                   published_scaffold_equivalence={"demonstrated": True})
    assert claim["label"] == "COMPARABLE_TO_PUBLISHED_RESULT"
    assert claim["sota_claim_allowed"] == "NO"
    assert "verified_protocol_fully_pinned" in claim["failed_conditions"]
