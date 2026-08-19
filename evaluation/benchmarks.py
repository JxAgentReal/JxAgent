#!/usr/bin/env python3
"""Benchmark identities: osworld_original vs osworld_verified.

Separate identities, never interchangeable. Protocol fields that cannot be
verified from this repository carry the literal marker
REQUIRES_EXTERNAL_VERIFICATION; `protocol_resolved()` is the gate used by the
SOTA guard and by any "Verified" labeling.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRES_EXTERNAL_VERIFICATION = "REQUIRES_EXTERNAL_VERIFICATION"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "osworld_config.yaml")

# Fields that must be resolved (not carry the marker) before any
# "Verified"/"SOTA" label is allowed.
VERIFIED_PROTOCOL_FIELDS = [
    "repository_revision",
    "task_list_source",
    "task_count",
    "task_id_hash",
    "environment_definition",
    "step_budget",
    "scoring_method",
]


class BenchmarkError(RuntimeError):
    pass


def load_benchmark_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    benchmarks = (cfg or {}).get("benchmarks")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise BenchmarkError(f"no benchmarks defined in {path}")
    return cfg


def get_benchmark(name: str, path: str = DEFAULT_CONFIG_PATH) -> dict:
    cfg = load_benchmark_config(path)
    if name not in cfg["benchmarks"]:
        raise BenchmarkError(
            f"unknown benchmark '{name}'; available: {sorted(cfg['benchmarks'])}. "
            "osworld_original and osworld_verified are distinct identities.")
    return {"name": name, **cfg["benchmarks"][name]}


def list_benchmarks(path: str = DEFAULT_CONFIG_PATH) -> List[str]:
    return sorted(load_benchmark_config(path)["benchmarks"].keys())


def is_resolved(value) -> bool:
    return not (isinstance(value, str) and value == REQUIRES_EXTERNAL_VERIFICATION)


def protocol_resolved(benchmark: dict) -> bool:
    if benchmark.get("name") != "osworld_verified":
        return True
    unresolved = [f for f in VERIFIED_PROTOCOL_FIELDS
                  if not is_resolved(benchmark.get(f))]
    return not unresolved


def unresolved_fields(benchmark: dict) -> List[str]:
    return [f for f in VERIFIED_PROTOCOL_FIELDS
            if not is_resolved(benchmark.get(f))]


def hash_task_ids(task_ids: List[str]) -> str:
    canon = json.dumps(sorted(task_ids), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def load_task_list(path: str) -> List[dict]:
    """Task list file: [{"task_id": ..., "domain": ..., "instruction": ...}, ...]
    or the cached [[task_id, instruction], ...] pairs format."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks: List[dict] = []
    if isinstance(data, dict) and data.get("status") != "ready":
        raise BenchmarkError(
            f"task list {path} is not ready (status={data.get('status')!r})")
    if isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
        for tid, instr in data:
            tasks.append({"task_id": tid, "domain": tid.split("/")[0],
                          "instruction": instr})
    elif isinstance(data, list):
        for t in data:
            tasks.append({"task_id": t["task_id"], "domain": t.get("domain", ""),
                          "instruction": t.get("instruction", "")})
    else:
        raise BenchmarkError(f"unrecognized task list format: {path}")
    if not tasks:
        raise BenchmarkError(f"empty task list: {path}")
    return tasks
