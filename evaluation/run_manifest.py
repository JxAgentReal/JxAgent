#!/usr/bin/env python3
"""Run manifests: machine-written provenance for every evaluation run, and
the base/adapter compatibility gate (baseline-first enforcement).

Manifest compatibility compares exactly the fields that would make an
adapter score non-comparable to a base score. Arms may differ ONLY in:
model/adapter identity, timestamp, run-specific counters.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import List, Optional, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    """Commit hash of the scaffold repo, or None when not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
             "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _version_of(import_name: str) -> Optional[str]:
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def collect_environment() -> dict:
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hardware": {"machine": platform.machine()},
    }
    torch = _version_of("torch")
    if torch:
        env["pytorch_version"] = torch
        try:
            import torch
            env["hardware"]["gpu_count"] = torch.cuda.device_count()
            env["hardware"]["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            env["rocm_version"] = getattr(torch.version, "hip", None)
        except Exception:  # noqa: BLE001
            pass
    for key, mod in (("inference_library_versions", None),):
        env[key] = {
            "transformers": _version_of("transformers"),
            "vllm": _version_of("vllm"),
            "swift": _version_of("swift"),
        }
    return env


def build_manifest(*, arm: str, scaffold_cfg: dict, benchmark: dict,
                   task_ids: List[str], task_set_hash: str,
                   model_repo: Optional[str], model_revision: Optional[str],
                   adapter_revision: Optional[str] = None,
                   checkpoint_gate: Optional[str] = None,
                   dataset_revision: Optional[str] = None,
                   dataset_manifest_hash: Optional[str] = None,
                   environment_revision: Optional[str] = None,
                   vm_identifiers: Optional[dict] = None,
                   scaffold_config_path: Optional[str] = None,
                   dry_run: bool = False) -> dict:
    from evaluation.scaffold import scaffold_hash
    sampling = dict(scaffold_cfg.get("sampling", {}))
    return {
        "manifest_version": 1,
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "arm": arm,
        "dry_run": dry_run,
        "git_commit": git_commit(),
        "scaffold_commit_note": "not a git checkout" if git_commit() is None else None,
        "benchmark": {
            "name": benchmark.get("name"),
            "repository": benchmark.get("repository"),
            "repository_revision": benchmark.get("repository_revision"),
            "task_count": len(task_ids),
            "task_set_hash": task_set_hash,
        },
        "model": {
            "repository": model_repo,
            "revision": model_revision,
        },
        "adapter": {
            "revision": adapter_revision,
            "checkpoint_gate": checkpoint_gate,
            "merged": False,
        },
        "dataset": {
            "revision": dataset_revision,
            "manifest_hash": dataset_manifest_hash,
        },
        "environment_revision": environment_revision,
        "vm_identifiers": vm_identifiers or {},
        "system_prompt_hash": _sha256_str(str(scaffold_cfg.get("agent", {}).get("system_prompt", ""))),
        "action_space": list(scaffold_cfg.get("agent", {}).get("action_space", [])),
        "tool_schema_hash": _sha256_str(json.dumps(
            scaffold_cfg.get("agent", {}).get("action_space", []), sort_keys=True)),
        "observation": dict(scaffold_cfg.get("observation", {})),
        "sampling_settings": sampling,
        "seed": scaffold_cfg.get("sampling", {}).get("seed"),
        "seed_policy": scaffold_cfg.get("sampling", {}).get("seed_policy"),
        "step_budget": scaffold_cfg.get("execution", {}).get("step_budget"),
        "timeout_policy": {
            k: scaffold_cfg.get("execution", {}).get(k)
            for k in ("action_timeout_s", "model_timeout_s", "screenshot_settle_delay_s")},
        "retry_policy": {
            k: scaffold_cfg.get("execution", {}).get(k)
            for k in ("retry_count", "retry_backoff_s")},
        "history_policy": dict(scaffold_cfg.get("history", {})),
        "scaffold_config_hash": scaffold_hash(scaffold_cfg),
        "scaffold_config_path": scaffold_config_path,
        "runtime_environment": collect_environment(),
    }


def save_manifest(out_dir: str, manifest: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Fields that MUST be identical for two arms' scores to be comparable.
COMPAT_FIELDS = [
    "benchmark.name",
    "benchmark.repository",
    "benchmark.repository_revision",
    "benchmark.task_set_hash",
    "model.revision",   # adapter must sit on the SAME base weights
    "system_prompt_hash",
    "action_space",
    "tool_schema_hash",
    "observation",
    "sampling_settings",
    "seed",
    "seed_policy",
    "step_budget",
    "timeout_policy",
    "retry_policy",
    "history_policy",
    "scaffold_config_hash",
    "environment_revision",
    "vm_identifiers",
    "adapter.merged",
]

# Fields allowed to differ (arm identity + run bookkeeping).
ARM_SPECIFIC_FIELDS = [
    "manifest_version", "created_utc", "arm", "dry_run", "git_commit",
    "scaffold_commit_note", "model.repository", "adapter",
    "dataset",
    "scaffold_config_path", "runtime_environment",
    "benchmark.task_count",  # subset screening may differ from full runs only
                             # when task_set_hash differs -> caught above
]


def _get(manifest: dict, dotted: str):
    cur = manifest
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return "<ABSENT>"
        cur = cur[part]
    return cur


def check_compatibility(base_manifest: dict, adapter_manifest: dict) -> Tuple[bool, List[dict]]:
    """True when the adapter run is comparable to the base run."""
    diffs = []
    for field in COMPAT_FIELDS:
        b, a = _get(base_manifest, field), _get(adapter_manifest, field)
        if b != a:
            diffs.append({"field": field, "base": b, "adapter": a})
    return (not diffs), diffs


class BaselineRequiredError(RuntimeError):
    """Raised when comparative scoring is attempted without a compatible,
    locally-produced base run."""


def require_baseline(adapter_manifest: dict, base_manifest: Optional[dict],
                     allow_without_baseline: bool = False) -> dict:
    """Baseline-first enforcement for COMPARATIVE scoring. Basic syntax /
    dry-run testing does not need a base score."""
    if base_manifest is None:
        if allow_without_baseline:
            return {"comparable": False,
                    "reason": "NO_BASE_MANIFEST",
                    "note": "results are NOT comparable to any baseline; "
                            "syntax/dry-run testing only"}
        raise BaselineRequiredError(
            "comparative scoring refused: no local base-run manifest supplied. "
            "Run the untouched base model through this exact harness first "
            "(the quoted published number is never an acceptable baseline).")
    ok, diffs = check_compatibility(base_manifest, adapter_manifest)
    if not ok:
        raise BaselineRequiredError(
            "comparative scoring refused: base and adapter manifests are not "
            f"compatible: {json.dumps(diffs, indent=1)}")
    return {"comparable": True, "diffs": []}
