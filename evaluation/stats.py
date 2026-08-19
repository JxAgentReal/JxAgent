#!/usr/bin/env python3
"""Paired base-vs-adapter statistics over per-task outcomes.

Small-sample exact statistics only (no asymptotic shortcuts):
  - exact two-sided McNemar (binomial sign test on discordant pairs)
  - percentile bootstrap CI over the paired per-task delta
Effect size (raw delta in percentage points) is always reported WITH its
uncertainty; nothing here ever declares a tiny delta meaningful by itself.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


def _binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b = tasks where base succeeded and adapter failed
    c = tasks where adapter succeeded and base failed
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(_binom_pmf(i, n, 0.5) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail)


def bootstrap_paired_delta_ci(base: Sequence[int], adapter: Sequence[int],
                              n_boot: int = 10000, seed: int = 1337,
                              alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI for the paired success-rate delta
    (adapter - base), in percentage points."""
    if len(base) != len(adapter) or not base:
        raise ValueError("paired outcomes must be same non-zero length")
    n = len(base)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        diff = 0
        for _ in range(n):
            i = rng.randrange(n)
            diff += adapter[i] - base[i]
        deltas.append(100.0 * diff / n)
    deltas.sort()
    lo_idx = max(0, int(math.floor((alpha / 2) * n_boot)) - 1)
    hi_idx = min(n_boot - 1, int(math.floor((1 - alpha / 2) * n_boot)))
    return deltas[lo_idx], deltas[hi_idx]


def paired_comparison(base_outcomes: Dict[str, bool],
                      adapter_outcomes: Dict[str, bool],
                      n_boot: int = 10000, seed: int = 1337) -> dict:
    """Full paired summary over the shared task ids."""
    shared = sorted(set(base_outcomes) & set(adapter_outcomes))
    missing_in_base = sorted(set(adapter_outcomes) - set(base_outcomes))
    missing_in_adapter = sorted(set(base_outcomes) - set(adapter_outcomes))
    b = sum(1 for t in shared if base_outcomes[t] and not adapter_outcomes[t])
    c = sum(1 for t in shared if adapter_outcomes[t] and not base_outcomes[t])
    ties = sum(1 for t in shared if base_outcomes[t] == adapter_outcomes[t])
    base_rate = sum(1 for t in shared if base_outcomes[t]) / len(shared) if shared else 0.0
    adapter_rate = sum(1 for t in shared if adapter_outcomes[t]) / len(shared) if shared else 0.0
    delta_pp = 100.0 * (adapter_rate - base_rate)
    base_seq = [1 if base_outcomes[t] else 0 for t in shared]
    adapter_seq = [1 if adapter_outcomes[t] else 0 for t in shared]
    ci = (bootstrap_paired_delta_ci(base_seq, adapter_seq, n_boot, seed)
          if shared else (float("nan"), float("nan")))
    p = mcnemar_exact(b, c) if shared else 1.0
    return {
        "shared_tasks": len(shared),
        "missing_in_base": missing_in_base,
        "missing_in_adapter": missing_in_adapter,
        "base_success_rate": base_rate,
        "adapter_success_rate": adapter_rate,
        "delta_percentage_points": delta_pp,
        "paired_wins_adapter": c,
        "paired_wins_base": b,
        "ties": ties,
        "discordant": b + c,
        "mcnemar_exact_p": p,
        "bootstrap_delta_ci_95_pp": [ci[0], ci[1]],
        "interpretation_guard": (
            "A raw delta is NOT evidence by itself. Require the bootstrap CI "
            "to exclude 0 (or McNemar p < 0.05) before treating the delta as "
            "real; for OSWorld-scale n (369-439) differences under ~2 pp are "
            "typically indistinguishable from noise."),
        "likely_meaningful": bool(shared) and (
            (ci[0] > 0 or ci[1] < 0) and p < 0.05),
    }
