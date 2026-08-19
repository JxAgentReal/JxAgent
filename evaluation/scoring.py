#!/usr/bin/env python3
"""Result accounting: every attempted task produces a record; nothing
disappears; the primary (strict) denominator NEVER excludes failed launches
or errors. When a protocol requires a different denominator, BOTH rates are
reported with a machine-readable explanation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Optional

STATUS_SUCCESS = "success"
STATUS_MODEL_FAILURE = "model_failure"
STATUS_PARSER_FAILURE = "parser_failure"
STATUS_TIMEOUT = "timeout"
STATUS_ENVIRONMENT_FAILURE = "environment_failure"
STATUS_HARNESS_FAILURE = "harness_failure"
STATUS_INVALID_TASK = "invalid_task"
STATUS_ABORTED = "aborted"
STATUS_MISSING = "missing"   # expected but never attempted (crashed run)

ALL_STATUSES = [STATUS_SUCCESS, STATUS_MODEL_FAILURE, STATUS_PARSER_FAILURE,
                STATUS_TIMEOUT, STATUS_ENVIRONMENT_FAILURE, STATUS_HARNESS_FAILURE,
                STATUS_INVALID_TASK, STATUS_ABORTED, STATUS_MISSING]

# statuses that are final once written (resume skips them)
TERMINAL_STATUSES = {STATUS_SUCCESS, STATUS_MODEL_FAILURE, STATUS_PARSER_FAILURE,
                     STATUS_TIMEOUT, STATUS_ENVIRONMENT_FAILURE,
                     STATUS_HARNESS_FAILURE, STATUS_INVALID_TASK}

# default protocol exclusions (mirrors scaffold failure_accounting config)
DEFAULT_PROTOCOL_EXCLUDES = [STATUS_INVALID_TASK, STATUS_ENVIRONMENT_FAILURE,
                            STATUS_HARNESS_FAILURE]


@dataclass
class TaskResult:
    task_id: str
    status: str
    steps: int = 0
    latency_s: float = 0.0
    failure_category: Optional[str] = None          # failure_taxonomy category
    failure_annotation: str = "none"                # none|pending|manual:<cat>
    error_detail: Optional[str] = None
    finished: bool = False                          # model emitted finish
    env_success_at_finish: Optional[bool] = None    # scorer value when known
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        d.update(self.extra)
        return d

    @staticmethod
    def from_dict(d: dict) -> "TaskResult":
        known = {f.name for f in fields(TaskResult)}
        extra = {k: v for k, v in d.items() if k not in known}
        base = {k: v for k, v in d.items() if k in known}
        tr = TaskResult(**base)
        tr.extra = extra
        return tr


def is_success(r: TaskResult) -> bool:
    return r.status == STATUS_SUCCESS


def aggregate(results: List[TaskResult], expected_task_ids: List[str],
              protocol_excludes: Optional[List[str]] = None) -> dict:
    """Aggregate with double denominator accounting.

    strict rate:   successes / len(expected_task_ids)
                   (missing tasks, harness errors, env errors all count
                   against the arm)
    protocol rate: successes / tasks attempted under the protocol
                   (default excludes invalid_task, environment_failure,
                   harness_failure)
    """
    protocol_excludes = (DEFAULT_PROTOCOL_EXCLUDES if protocol_excludes is None
                         else list(protocol_excludes))
    by_id = {r.task_id: r for r in results}
    counts: Dict[str, int] = {s: 0 for s in ALL_STATUSES}
    for tid in expected_task_ids:
        r = by_id.get(tid)
        counts[r.status if r else STATUS_MISSING] += 1
    unexpected = [r.task_id for r in results if r.task_id not in set(expected_task_ids)]

    n_expected = len(expected_task_ids)
    successes = counts[STATUS_SUCCESS]
    strict_rate = successes / n_expected if n_expected else 0.0
    excluded = sum(counts[s] for s in protocol_excludes)
    protocol_denom = n_expected - excluded
    protocol_rate = successes / protocol_denom if protocol_denom > 0 else None

    return {
        "expected_task_count": n_expected,
        "attempted": n_expected - counts[STATUS_MISSING],
        "status_counts": counts,
        "skipped_missing": counts[STATUS_MISSING],
        "unexpected_task_ids": unexpected,
        "successes": successes,
        "strict_success_rate": strict_rate,
        "strict_denominator": n_expected,
        "protocol_success_rate": protocol_rate,
        "protocol_denominator": protocol_denom,
        "protocol_excludes": protocol_excludes,
        "denominator_explanation": (
            "strict = successes / all expected tasks (failed launches and "
            "errors count as failure). protocol = successes / expected minus "
            f"{protocol_excludes}; only meaningful for arms compared under "
            "identical exclusion policy."),
        "accounting_complete": (counts[STATUS_MISSING] == 0 and not unexpected),
    }


def save_task_result(out_dir: str, result: TaskResult) -> str:
    """Atomic per-task persistence (resume safety)."""
    tasks_dir = os.path.join(out_dir, "tasks")
    os.makedirs(tasks_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in result.task_id)
    path = os.path.join(tasks_dir, safe + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def load_task_results(out_dir: str) -> List[TaskResult]:
    tasks_dir = os.path.join(out_dir, "tasks")
    if not os.path.isdir(tasks_dir):
        return []
    out: List[TaskResult] = []
    for name in sorted(os.listdir(tasks_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(tasks_dir, name), "r", encoding="utf-8") as f:
            out.append(TaskResult.from_dict(json.load(f)))
    return out


def write_aggregate(out_dir: str, agg: dict, name: str = "aggregate.json") -> str:
    path = os.path.join(out_dir, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path
