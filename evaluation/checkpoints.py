#!/usr/bin/env python3
"""Checkpoint discovery and gate mapping for 20%/55%/100% evaluation.

Never trusts directory names alone: the optimizer step is read from each
checkpoint's trainer_state.json and the epoch fraction is computed from the
REAL total step count (passed in or read from the training args). Missing
required gates fail loudly with an explicit list.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_GATES = [20, 55, 100]
DEFAULT_TOLERANCE_PCT = 10.0  # a checkpoint maps to a gate within +-10% epoch


@dataclass
class CheckpointInfo:
    path: str
    global_step: int
    epoch_fraction_pct: float      # 0..100
    has_adapter: bool
    name: str

    def to_dict(self) -> dict:
        return {"path": self.path, "global_step": self.global_step,
                "epoch_fraction_pct": round(self.epoch_fraction_pct, 2),
                "has_adapter": self.has_adapter, "name": self.name}


def discover_checkpoints(train_output_dir: str,
                         total_optimizer_steps: Optional[int] = None
                         ) -> List[CheckpointInfo]:
    """Scan swift/HF-style checkpoint dirs; read trainer_state.json for the
    true global_step (never the directory name)."""
    found: List[CheckpointInfo] = []
    patterns = [os.path.join(train_output_dir, "checkpoint-*"),
                os.path.join(train_output_dir, "*", "checkpoint-*"),
                # mi300x/common.sh jx_preserve_gates copies gate checkpoints to
                # gates/{early,middle,final}; names are NEVER trusted - the
                # step is still read from trainer_state.json.
                os.path.join(train_output_dir, "gates", "*"),
                os.path.join(train_output_dir, "*", "gates", "*")]
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path in seen or not os.path.isdir(path):
                continue
            seen.add(path)
            ts_path = os.path.join(path, "trainer_state.json")
            has_adapter = (os.path.exists(os.path.join(path, "adapter_model.safetensors"))
                           or os.path.exists(os.path.join(path, "adapter_model.bin")))
            global_step = None
            epoch = None
            if os.path.exists(ts_path):
                try:
                    with open(ts_path, "r", encoding="utf-8") as f:
                        ts = json.load(f)
                    global_step = ts.get("global_step")
                    epoch = ts.get("epoch")
                except (OSError, ValueError):
                    pass
            if global_step is None:
                # fall back to directory number, but FLAG it as unverified
                try:
                    global_step = int(os.path.basename(path).split("-")[-1])
                except ValueError:
                    continue
            if total_optimizer_steps and epoch is None:
                frac = 100.0 * global_step / float(total_optimizer_steps)
            elif epoch is not None:
                frac = 100.0 * float(epoch)
            else:
                frac = -1.0
            has_adapter = (os.path.exists(os.path.join(path, "adapter_model.safetensors"))
                           or os.path.exists(os.path.join(path, "adapter_model.bin")))
            found.append(CheckpointInfo(path=path, global_step=global_step,
                                        epoch_fraction_pct=frac,
                                        has_adapter=has_adapter,
                                        name=os.path.basename(path)))
    found.sort(key=lambda c: c.global_step)
    return found


def map_to_gates(checkpoints: List[CheckpointInfo],
                 gates: List[int] = DEFAULT_GATES,
                 tolerance_pct: float = DEFAULT_TOLERANCE_PCT
                 ) -> Dict[int, Optional[CheckpointInfo]]:
    """Nearest checkpoint per gate (must be within tolerance and carry an
    adapter). A gate with no valid checkpoint maps to None."""
    out: Dict[int, Optional[CheckpointInfo]] = {}
    for gate in gates:
        candidates = [c for c in checkpoints if c.has_adapter
                     and c.epoch_fraction_pct >= 0]
        best = None
        best_dist = None
        for c in candidates:
            dist = abs(c.epoch_fraction_pct - gate)
            if best_dist is None or dist < best_dist:
                best, best_dist = c, dist
        out[gate] = best if (best is not None and
                             best_dist <= tolerance_pct) else None
    return out


def require_gates(checkpoints: List[CheckpointInfo],
                  gates: List[int] = DEFAULT_GATES,
                  tolerance_pct: float = DEFAULT_TOLERANCE_PCT
                  ) -> Dict[int, CheckpointInfo]:
    mapped = map_to_gates(checkpoints, gates, tolerance_pct)
    missing = [g for g, c in mapped.items() if c is None]
    if missing:
        avail = [c.to_dict() for c in checkpoints]
        raise FileNotFoundError(
            f"required checkpoint gates missing: {missing}. "
            f"Available checkpoints (metadata-read, names not trusted): {avail}. "
            "Fix training checkpoint retention or provide the checkpoint explicitly.")
    return {g: c for g, c in mapped.items() if c is not None}
