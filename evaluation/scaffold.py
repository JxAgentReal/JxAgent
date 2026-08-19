#!/usr/bin/env python3
"""Scaffold loading, completeness validation and base/adapter parity lock.

A scaffold is valid only if every required key below is present and non-null.
Parity check: the resolved effective config of both arms must be identical
outside `arm_specific_keys`; any other difference is a score-moving variable
and aborts the run BEFORE any task is attempted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SCAFFOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "scaffold_config.yaml")

REQUIRED_KEYS: List[str] = [
    "agent.system_prompt",
    "agent.action_space",
    "agent.coordinate_space",
    "agent.history_rendering",
    "agent.user_template",
    "sampling.temperature",
    "sampling.top_p",
    "sampling.top_k",
    "sampling.max_new_tokens",
    "sampling.seed",
    "sampling.seed_policy",
    "observation.source_resolution",
    "observation.resize_max_long_side",
    "observation.never_upscale",
    "observation.preserve_aspect",
    "observation.interpolation",
    "observation.color_mode",
    "observation.format",
    "history.limit",
    "history.truncation_strategy",
    "history.include_step_numbers",
    "execution.step_budget",
    "execution.action_timeout_s",
    "execution.model_timeout_s",
    "execution.retry_count",
    "execution.retry_backoff_s",
    "execution.screenshot_settle_delay_s",
    "finish_behavior.required_to_score_success",
    "finish_behavior.accept_without_status",
    "finish_behavior.premature_finish_scores_failure",
    "failure_accounting.harness_and_environment_errors_count_as_failure_in_strict_rate",
    "failure_accounting.protocol_rate_excludes",
]


class ScaffoldError(RuntimeError):
    pass


def _get_nested(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def load_scaffold(path: str = DEFAULT_SCAFFOLD_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ScaffoldError(f"scaffold config {path} is not a mapping")
    missing = [k for k in REQUIRED_KEYS
               if _get_nested(cfg, k)[1] is False or _get_nested(cfg, k)[0] is None]
    if missing:
        raise ScaffoldError(f"scaffold config incomplete, missing/null: {missing}")
    return cfg


def scaffold_hash(cfg: dict) -> str:
    """Hash of every score-moving field (excludes nothing; arm overrides are
    merged in before hashing when comparing arms)."""
    canon = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def effective_config(scaffold_cfg: dict, arm_cfg: dict) -> dict:
    """Merge an arm definition (runs.base / runs.adapter entries) into the
    scaffold. Arm keys are namespaced under 'arm'."""
    eff = copy.deepcopy(scaffold_cfg)
    eff["arm"] = copy.deepcopy(arm_cfg or {})
    return eff


def check_parity(base_eff: dict, adapter_eff: dict,
                 arm_specific_keys: List[str]) -> Tuple[bool, List[dict]]:
    """True when the two effective configs differ ONLY in arm-specific keys.
    Returns (ok, diffs) where each diff is
    {field, base, adapter, score_moving: true}."""
    allowed = set(k.split(".", 1)[1] if k.startswith("arm.") else k
                  for k in arm_specific_keys) if arm_specific_keys else set()
    # arm_specific_keys are given relative to the arm namespace in configs;
    # accept both 'runs.model' and 'arm.model' spellings.
    allowed |= set(arm_specific_keys or [])
    diffs: List[dict] = []
    keys = sorted(set(_flatten(base_eff).keys()) | set(_flatten(adapter_eff).keys()))
    flat_b, flat_a = _flatten(base_eff), _flatten(adapter_eff)
    for k in keys:
        if k.startswith("arm."):
            short = "runs." + k[len("arm."):]
            if k in allowed or short in allowed:
                continue
        if flat_b.get(k, "<ABSENT>") != flat_a.get(k, "<ABSENT>"):
            diffs.append({"field": k, "base": flat_b.get(k, "<ABSENT>"),
                          "adapter": flat_a.get(k, "<ABSENT>"),
                          "score_moving": True})
    return (not diffs), diffs


def _flatten(cfg: dict, prefix: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in (cfg or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def assert_arm_parity(scaffold_cfg: dict, base_arm: dict, adapter_arm: dict) -> None:
    ok, diffs = check_parity(
        effective_config(scaffold_cfg, base_arm),
        effective_config(scaffold_cfg, adapter_arm),
        scaffold_cfg.get("arm_specific_keys", []))
    if not ok:
        raise ScaffoldError(
            "base/adapter scaffold parity violated (score-moving fields must "
            f"never differ between arms): {json.dumps(diffs, indent=1)}")
