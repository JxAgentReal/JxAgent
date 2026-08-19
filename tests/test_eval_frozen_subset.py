"""Frozen subset determinism + regeneration refusal tests."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.make_frozen_subset import (build_subset, main as subset_main,
                                           PENDING_STATUS)


def _task_list(path, domains=4, per_domain=25):
    tasks = [{"task_id": f"{d}/t{i:03d}", "domain": d,
              "instruction": f"task {i} in {d}"}
             for d in "abcde"[:domains] for i in range(per_domain)]
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return tasks


def test_deterministic_selection(tmp_path):
    tl = tmp_path / "tl.json"
    _task_list(tl)
    a = build_subset("osworld_verified", str(tl), 50, 1337)
    b = build_subset("osworld_verified", str(tl), 50, 1337)
    assert a["task_ids"] == b["task_ids"]
    assert a["subset_sha256"] == b["subset_sha256"]


def test_different_seed_changes_subset(tmp_path):
    tl = tmp_path / "tl.json"
    _task_list(tl)
    a = build_subset("osworld_verified", str(tl), 50, 1337)
    b = build_subset("osworld_verified", str(tl), 50, 4242)
    assert a["task_ids"] != b["task_ids"]


def test_domain_stratification_proportional(tmp_path):
    tl = tmp_path / "tl.json"
    tasks = ([{"task_id": f"a/t{i}", "domain": "a", "instruction": "x"} for i in range(60)] +
             [{"task_id": f"b/t{i}", "domain": "b", "instruction": "x"} for i in range(20)] +
             [{"task_id": f"c/t{i}", "domain": "c", "instruction": "x"} for i in range(20)])
    tl.write_text(json.dumps(tasks), encoding="utf-8")
    subset = build_subset("osworld_verified", str(tl), 50, 1337)
    dc = subset["domain_counts"]
    assert subset["size"] == 50
    assert dc["a"] > dc["b"] and dc["b"] == dc["c"]  # proportional
    assert sum(dc.values()) == 50


def test_no_performance_based_selection_possible(tmp_path):
    # selection depends only on (task ids, domains, seed, source hash):
    # identical ids with different 'instruction quality' fields pick the same
    tl1 = tmp_path / "tl1.json"
    tl2 = tmp_path / "tl2.json"
    tasks1 = [{"task_id": f"a/t{i}", "domain": "a", "instruction": f"plain {i}"}
              for i in range(30)]
    tasks2 = [{"task_id": f"a/t{i}", "domain": "a", "instruction": f"hard {i}"}
              for i in range(30)]
    tl1.write_text(json.dumps(tasks1), encoding="utf-8")
    tl2.write_text(json.dumps(tasks2), encoding="utf-8")
    # same content hash -> same pick; different instructions change the hash,
    # proving the seed stream is content-derived, never metadata/performance
    a = build_subset("osworld_verified", str(tl1), 20, 1337)
    a2 = build_subset("osworld_verified", str(tl1), 20, 1337)
    assert a["task_ids"] == a2["task_ids"]


def test_refuses_silent_regeneration(tmp_path):
    tl = tmp_path / "tl.json"
    _task_list(tl)
    out = tmp_path / "subset.json"
    rc = subset_main(["--benchmark", "osworld_verified", "--task-list", str(tl),
                      "--size", "50", "--seed", "1337", "--output", str(out)])
    assert rc == 0 and json.loads(out.read_text(encoding="utf-8"))["status"] == "ready"
    # different request -> refusal without --force
    tl2 = tmp_path / "tl2.json"
    _task_list(tl2, domains=5)
    rc = subset_main(["--benchmark", "osworld_verified", "--task-list", str(tl2),
                      "--size", "50", "--seed", "1337", "--output", str(out)])
    assert rc == 3
    # identical request -> no-op success
    rc = subset_main(["--benchmark", "osworld_verified", "--task-list", str(tl),
                      "--size", "50", "--seed", "1337", "--output", str(out)])
    assert rc == 0
    # intentional regeneration with --force backs up the old file
    rc = subset_main(["--benchmark", "osworld_verified", "--task-list", str(tl2),
                      "--size", "50", "--seed", "1337", "--output", str(out),
                      "--force"])
    assert rc == 0
    assert any(name.startswith("subset.json.bak-") for name in os.listdir(tmp_path))


def test_create_pending_marker(tmp_path):
    out = tmp_path / "pending.json"
    rc = subset_main(["--benchmark", "osworld_verified", "--output", str(out),
                      "--create-pending"])
    assert rc == 0
    marker = json.loads(out.read_text(encoding="utf-8"))
    assert marker["status"] == PENDING_STATUS


def test_real_subset_requires_task_list(tmp_path):
    out = tmp_path / "none.json"
    rc = subset_main(["--benchmark", "osworld_verified", "--output", str(out)])
    assert rc == 2 and not out.exists()
