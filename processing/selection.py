"""Best-valid deterministic final selection for JxAgent.

The source adapters produce a bounded candidate pool. This module chooses the
final source quotas using quality, token efficiency, verified frontier scores,
and diversity rather than accepting the first valid rows encountered.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .motifs import attach_motifs, detect_motifs

DEFAULT_OVERSAMPLE_FACTORS = {
    "procua": 1.50,
    "gui360": 1.25,
    "videocua": 1.25,
    "groundcua": 1.35,
    "pcagente": 1.00,
    "replay": 1.15,
}

# Replay and GUI360 cohort targets are preserved exactly in the final mixture.
REPLAY_BASE = {"coding": 1600, "math": 1500, "instruction": 1700, "vqa": 1400, "tool": 1300}
GUI360_BASE = {"grounding": 2500, "understanding": 2650}


def candidate_targets(final_targets: Dict[str, int], config: Optional[dict] = None) -> Dict[str, int]:
    cfg = config or {}
    enabled = bool(cfg.get("enabled", True))
    factors = dict(DEFAULT_OVERSAMPLE_FACTORS)
    factors.update(cfg.get("oversample_factors") or {})
    if not enabled:
        return dict(final_targets)
    out = {}
    for src, n in final_targets.items():
        factor = max(1.0, float(factors.get(src, 1.0)))
        out[src] = max(n, int(math.ceil(n * factor)))
    return out


def _stable_tie(sample: dict) -> str:
    raw = "|".join(str(sample.get(k, "")) for k in ("source", "trajectory_id", "step_id"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sample_id(sample: dict) -> str:
    return f"{sample.get('source','')}::{sample.get('trajectory_id','')}::{sample.get('step_id','')}"


def load_frontier_scores(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    scores: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or "")
            # Only objectively verifiable records are allowed to affect rank.
            if not sid or row.get("verifiable") is not True:
                continue
            try:
                score = float(row.get("frontier_score"))
            except (TypeError, ValueError):
                continue
            scores[sid] = max(0.0, min(1.0, score))
    return scores


def _group(sample: dict) -> str:
    meta = sample.get("metadata", {}) or {}
    return str(meta.get("group_id") or meta.get("collection_run") or meta.get("task_family")
               or meta.get("content_family") or sample.get("trajectory_id") or "unknown")


def _app(sample: dict) -> str:
    return str((sample.get("metadata", {}) or {}).get("app") or "unknown").lower()


def _verb(sample: dict) -> str:
    meta = sample.get("metadata", {}) or {}
    canonical = meta.get("action_canonical_final_pixels")
    if canonical:
        low = str(canonical).lower().strip()
        return low.split("(", 1)[0].strip() or "text"
    content = ""
    for m in reversed(sample.get("messages", []) or []):
        if m.get("role") == "assistant":
            content = str(m.get("content") or "")
            break
    low = content.lower().strip()
    if "action:" in low:
        low = low.rsplit("action:", 1)[-1].strip()
    return low.split("(", 1)[0].strip() or "text"


def _domain_key(sample: dict) -> str:
    meta = sample.get("metadata", {}) or {}
    if sample.get("task_type") == "replay_coding":
        return f"coding::{meta.get('code_language','unknown')}::{meta.get('coding_problem_type','unknown')}"
    if sample.get("task_type") == "replay_math":
        return f"math::{meta.get('difficulty_bucket','unknown')}"
    return "n/a"


def _base_score(sample: dict, frontier_scores: Dict[str, float], freq: dict) -> float:
    meta = sample.get("metadata", {}) or {}
    q = meta.get("quality") or {}
    quality = float(q.get("score") or 5.0)
    efficiency = float(q.get("token_efficiency") or 0.0)
    # Saturate efficiency so very short trivial targets cannot dominate.
    eff_term = min(1.5, math.log1p(max(0.0, efficiency)) / 2.0)
    group = _group(sample)
    app = _app(sample)
    verb = _verb(sample)
    domain = _domain_key(sample)
    motifs = meta.get("motifs") or detect_motifs(sample)
    rarity = 0.0
    rarity += 0.45 / math.sqrt(max(1, freq["group"][group]))
    rarity += 0.30 / math.sqrt(max(1, freq["app"][app]))
    rarity += 0.20 / math.sqrt(max(1, freq["verb"][verb]))
    if domain != "n/a":
        rarity += 0.25 / math.sqrt(max(1, freq["domain"][domain]))
    rarity += sum(0.10 / math.sqrt(max(1, freq["motif"][m])) for m in motifs)
    frontier = frontier_scores.get(_sample_id(sample))
    # frontier_score means target-model uncertainty/error on a verified target.
    frontier_term = 1.6 * frontier if frontier is not None else 0.0
    return quality + 0.9 * eff_term + rarity + frontier_term


def _frequency_tables(samples: Iterable[dict]) -> dict:
    out = {k: Counter() for k in ("group", "app", "verb", "domain", "motif")}
    for s in samples:
        out["group"][_group(s)] += 1
        out["app"][_app(s)] += 1
        out["verb"][_verb(s)] += 1
        out["domain"][_domain_key(s)] += 1
        for m in ((s.get("metadata", {}) or {}).get("motifs") or detect_motifs(s)):
            out["motif"][m] += 1
    return out


def _scaled_subtargets(final_count: int, base_total: int, base: Dict[str, int]) -> Dict[str, int]:
    if final_count <= 0:
        return {k: 0 for k in base}
    raw = {k: final_count * v / base_total for k, v in base.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    need = final_count - sum(out.values())
    if need > 0:
        for k, _ in sorted(raw.items(), key=lambda kv: (kv[1] - math.floor(kv[1]), kv[0]), reverse=True)[:need]:
            out[k] += 1
    return out


def replay_final_subtargets(final_count: int) -> Dict[str, int]:
    return _scaled_subtargets(final_count, sum(REPLAY_BASE.values()), REPLAY_BASE)


def gui360_final_subtargets(final_count: int, grounding_full: int = 2500,
                            understanding_full: int = 2650) -> Dict[str, int]:
    if final_count == 16000:
        g, u = grounding_full, understanding_full
    else:
        g = int(round(final_count * grounding_full / 16000))
        u = int(round(final_count * understanding_full / 16000))
    g = min(final_count, g)
    u = min(max(0, final_count - g), u)
    return {"grounding": g, "understanding": u, "use": max(0, final_count - g - u)}


def _cohort(sample: dict) -> str:
    src = sample.get("source")
    tt = str(sample.get("task_type") or "")
    if src == "replay":
        return tt.removeprefix("replay_")
    if src == "gui360":
        if tt == "grounding": return "grounding"
        if tt == "screen_understanding": return "understanding"
        return "use"
    return "all"


def select_best_valid(samples: List[dict], final_targets: Dict[str, int], *,
                      config: Optional[dict] = None,
                      frontier_scores: Optional[Dict[str, float]] = None,
                      gui360_grounding: int = 2500,
                      gui360_understanding: int = 2650,
                      coverage_floors: Optional[Dict[str, float]] = None) -> Tuple[List[dict], dict]:
    cfg = config or {}
    frontier_scores = frontier_scores or {}
    prepared: List[dict] = []
    for s in samples:
        s = dict(s)
        s["metadata"] = dict(s.get("metadata", {}) or {})
        attach_motifs(s)
        prepared.append(s)
    freq = _frequency_tables(prepared)

    report = {"policy": "best_valid_v2", "candidate_count": len(prepared),
              "sources": {}, "frontier_scores_used": 0}
    selected: List[dict] = []
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for s in prepared:
        by_source[str(s.get("source"))].append(s)

    for src, target in final_targets.items():
        candidates = by_source.get(src, [])
        if src == "replay":
            subtargets = replay_final_subtargets(target)
        elif src == "gui360":
            subtargets = gui360_final_subtargets(target, gui360_grounding, gui360_understanding)
        else:
            subtargets = {"all": target}
        picked_src: List[dict] = []
        cohort_stats = {}
        for cohort, want in subtargets.items():
            pool = [s for s in candidates if _cohort(s) == cohort]
            ranked = sorted(pool,
                key=lambda s: (-_base_score(s, frontier_scores, freq), _stable_tie(s)))
            # Optional hard per-group cap prevents one collection run or task family
            # from saturating the selected set. Cap scales with target and number
            # of observed groups, and can be overridden in YAML.
            group_cap_cfg = int((cfg.get("max_per_group") or {}).get(src, 0))
            groups = max(1, len({_group(s) for s in pool}))
            auto_cap = max(2, int(math.ceil(max(1, want) / groups * 2.5)))
            group_cap = group_cap_cfg or auto_cap
            gc = Counter()
            chosen = []
            deferred = []
            for s in ranked:
                g = _group(s)
                if gc[g] >= group_cap:
                    deferred.append(s)
                    continue
                chosen.append(s); gc[g] += 1
                if len(chosen) >= want:
                    break
            if len(chosen) < want:
                have = {_sample_id(s) for s in chosen}
                for s in deferred + ranked:
                    if _sample_id(s) in have:
                        continue
                    chosen.append(s); have.add(_sample_id(s))
                    if len(chosen) >= want:
                        break
            for s in chosen:
                sid = _sample_id(s)
                if sid in frontier_scores:
                    report["frontier_scores_used"] += 1
                s["metadata"]["selection_score"] = round(_base_score(s, frontier_scores, freq), 5)
                s["metadata"]["selection_policy"] = "best_valid_v2"
            picked_src.extend(chosen)
            cohort_stats[cohort] = {"candidates": len(pool), "target": want, "selected": len(chosen)}
        selected.extend(picked_src)
        report["sources"][src] = {"candidates": len(candidates), "target": target,
                                  "selected": len(picked_src), "cohorts": cohort_stats}
    # Hard motif coverage rebalancing by same-source/same-cohort swaps. This
    # preserves every source and replay/GUI cohort quota exactly.
    floors = coverage_floors or {}
    cu_total = sum(1 for s in selected if s.get("source") != "replay")
    required = {m: (int(math.ceil(float(v) * cu_total)) if float(v) < 1.0 else int(v))
                for m, v in floors.items()}
    def motif_counts(rows):
        c = Counter()
        for x in rows:
            c.update(set((x.get("metadata", {}) or {}).get("motifs") or detect_motifs(x)))
        return c
    counts = motif_counts(selected)
    selected_ids = {_sample_id(s) for s in selected}
    score_map = {_sample_id(s): _base_score(s, frontier_scores, freq) for s in prepared}
    swaps = []
    for motif, need in sorted(required.items(), key=lambda kv: (-kv[1], kv[0])):
        while counts.get(motif, 0) < need:
            pool = [s for s in prepared if _sample_id(s) not in selected_ids and
                    motif in ((s.get("metadata", {}) or {}).get("motifs") or detect_motifs(s))]
            pool.sort(key=lambda s: (-score_map[_sample_id(s)], _stable_tie(s)))
            swapped = False
            for cand in pool:
                csrc, ccohort = cand.get("source"), _cohort(cand)
                victims = [s for s in selected if s.get("source") == csrc and _cohort(s) == ccohort and
                           motif not in ((s.get("metadata", {}) or {}).get("motifs") or detect_motifs(s))]
                victims.sort(key=lambda s: (score_map.get(_sample_id(s), 0.0), _stable_tie(s)))
                for victim in victims:
                    vm = set((victim.get("metadata", {}) or {}).get("motifs") or detect_motifs(victim))
                    cm = set((cand.get("metadata", {}) or {}).get("motifs") or detect_motifs(cand))
                    # Never break an already-satisfied hard floor.
                    if any(counts.get(m, 0) <= required.get(m, 0) and m not in cm
                           for m in vm if m in required):
                        continue
                    idx = selected.index(victim)
                    selected[idx] = cand
                    selected_ids.remove(_sample_id(victim)); selected_ids.add(_sample_id(cand))
                    for m in vm: counts[m] -= 1
                    for m in cm: counts[m] += 1
                    cand["metadata"]["selection_score"] = round(score_map[_sample_id(cand)], 5)
                    cand["metadata"]["selection_policy"] = "best_valid_v2_motif_floor_swap"
                    swaps.append({"motif": motif, "in": _sample_id(cand), "out": _sample_id(victim)})
                    swapped = True
                    break
                if swapped: break
            if not swapped:
                break
    unmet = {m: {"required": n, "selected": counts.get(m, 0)} for m, n in required.items()
             if counts.get(m, 0) < n}
    report["motif_coverage"] = {"required": required, "selected": dict(counts),
                                "unmet": unmet, "swaps": swaps}
    report["selected_count"] = len(selected)
    return selected, report
