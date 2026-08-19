#!/usr/bin/env python3
"""Per-task trajectory logging (JSONL, one record per step + task summary).

Everything needed to re-run analysis later: raw model output, parsed Plan
(preserved separately from the Action), parsed action, EXECUTED action (after
coordinate inverse-transform), observation references, transforms, latency,
retries. Secrets are redacted defensively (no env dump, hf_ tokens scrubbed).
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

_TOKEN_RE = re.compile(r"\b(hf_[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,}|Bearer\s+[A-Za-z0-9._\-]{20,})")


def redact(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return _TOKEN_RE.sub("[REDACTED]", text)


class TrajectoryLogger:
    """Writes RUN_DIR/trajectories/<safe_task_id>.jsonl atomically at close."""

    def __init__(self, run_dir: str, task_id: str, arm: str,
                 checkpoint: Optional[str] = None):
        self.run_dir = run_dir
        self.task_id = task_id
        self.arm = arm
        self.checkpoint = checkpoint
        traj_dir = os.path.join(run_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in task_id)
        self.path = os.path.join(traj_dir, safe + ".jsonl")
        self._tmp = self.path + ".tmp"
        self._fh = open(self._tmp, "w", encoding="utf-8")

    def log_header(self, *, instruction: str, system_prompt: str,
                   base_model_revision: Optional[str],
                   adapter_revision: Optional[str]) -> None:
        self._write({
            "record": "header",
            "task_id": self.task_id,
            "instruction": redact(instruction),
            "arm": self.arm,
            "checkpoint": self.checkpoint,
            "base_model_revision": base_model_revision,
            "adapter_revision": adapter_revision,
            "system_prompt": system_prompt,
        })

    def log_step(self, *, step: int, observation_ref: Optional[str],
                 observation_transform: Optional[dict],
                 raw_model_output: Optional[str],
                 parsed_plan: Optional[str],
                 parsed_action: Optional[str],
                 executed_action: Optional[str],
                 parse_error: Optional[str],
                 latency_s: float, retries: int,
                 env_state_meta: Optional[dict] = None) -> None:
        self._write({
            "record": "step",
            "task_id": self.task_id,
            "step": step,
            "observation_ref": observation_ref,
            "observation_transform": observation_transform,
            "raw_model_output": redact(raw_model_output),
            "parsed_plan": redact(parsed_plan),
            "parsed_action": parsed_action,
            "executed_action": executed_action,
            "parse_error": parse_error,
            "latency_s": round(float(latency_s), 4),
            "retries": retries,
            "env_state_meta": env_state_meta,
        })

    def log_summary(self, *, status: str, success: bool,
                    failure_category: Optional[str], total_steps: int,
                    total_latency_s: float) -> None:
        self._write({
            "record": "summary",
            "task_id": self.task_id,
            "arm": self.arm,
            "checkpoint": self.checkpoint,
            "status": status,
            "success": success,
            "failure_category": failure_category,
            "total_steps": total_steps,
            "total_latency_s": round(float(total_latency_s), 4),
        })

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> str:
        """Flush + atomic rename; the trajectory only becomes visible when
        complete (partial files from crashed runs remain as .tmp)."""
        self._fh.close()
        os.replace(self._tmp, self.path)
        return self.path

    def abandon(self) -> None:
        """Discard a partial trajectory (task aborted before completion)."""
        self._fh.close()
        if os.path.exists(self._tmp):
            os.remove(self._tmp)
