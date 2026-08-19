"""Benchmark identity separation + Verified protocol gating tests."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import benchmarks as bm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, "evaluation", "osworld_config.yaml")


def test_two_distinct_identities_exist():
    names = bm.list_benchmarks(CFG)
    assert "osworld_original" in names and "osworld_verified" in names


def test_original_and_verified_are_not_silent_aliases():
    orig = bm.get_benchmark("osworld_original", CFG)
    ver = bm.get_benchmark("osworld_verified", CFG)
    assert orig["task_count"] == 369
    assert bm.is_resolved(ver["task_count"]) is False  # must be pinned, not assumed
    assert bm.is_resolved(orig["repository_revision"]) is False


def test_verified_protocol_resolution_gate():
    ver = bm.get_benchmark("osworld_verified", CFG)
    assert bm.protocol_resolved(ver) is False
    assert set(bm.unresolved_fields(ver)) >= {"task_list_source", "task_count",
                                              "repository_revision"}
    orig = bm.get_benchmark("osworld_original", CFG)
    assert bm.protocol_resolved(orig) is True  # non-verified: gate n/a


def test_unknown_benchmark_rejected():
    with pytest.raises(bm.BenchmarkError):
        bm.get_benchmark("osworld", CFG)  # ambiguous name must not resolve


def test_task_list_formats(tmp_path):
    pairs = [["chrome/a", "instr a"], ["os/b", "instr b"]]
    p1 = tmp_path / "pairs.json"
    p1.write_text(json.dumps(pairs), encoding="utf-8")
    tasks = bm.load_task_list(str(p1))
    assert tasks[0]["domain"] == "chrome"
    dicts = [{"task_id": "chrome/a", "domain": "chrome", "instruction": "i"}]
    p2 = tmp_path / "dicts.json"
    p2.write_text(json.dumps(dicts), encoding="utf-8")
    assert bm.load_task_list(str(p2))[0]["task_id"] == "chrome/a"


def test_pending_task_list_rejected(tmp_path):
    p = tmp_path / "pending.json"
    p.write_text(json.dumps({"status": "PENDING_VERIFIED_TASK_LIST"}), encoding="utf-8")
    with pytest.raises(bm.BenchmarkError):
        bm.load_task_list(str(p))


def test_shipped_frozen_subset_is_explicitly_pending():
    path = os.path.join(HERE, "evaluation", "osworld_verified_frozen_subset.json")
    assert os.path.exists(path), "pending marker must exist until the real subset is generated"
    with open(path, "r", encoding="utf-8") as f:
        subset = json.load(f)
    assert subset["status"] == "PENDING_VERIFIED_TASK_LIST"
    assert subset["benchmark"] == "osworld_verified"


def test_published_reference_is_demoted():
    import yaml
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ref = cfg["published_reference"]
    assert ref["authoritative"] is False
    assert ref["provenance"] == "UNVERIFIED_QUOTE"
    # no field may present the quote as the baseline
    assert "expected_baseline" not in cfg


def test_task_id_hash_order_insensitive():
    assert bm.hash_task_ids(["a", "b"]) == bm.hash_task_ids(["b", "a"])
    assert bm.hash_task_ids(["a"]) != bm.hash_task_ids(["b"])
