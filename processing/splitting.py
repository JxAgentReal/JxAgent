"""Group-aware deterministic train/validation splitting.

The split key is a semantic/provenance family when the source exposes one,
otherwise a stable trajectory/content family. This prevents near-identical
members of one collection run/template/asset family from leaking across the
early-stopping boundary while preserving deterministic resume behavior.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Set, Tuple

DEFAULT_VALIDATION_PCT = 3.0


def split_of(group_id: str, validation_pct: float = DEFAULT_VALIDATION_PCT) -> str:
    h = int(hashlib.md5(group_id.encode("utf-8")).hexdigest()[:12], 16)
    u = h / 0xFFFFFFFFFFFF
    return "validation" if u < (validation_pct / 100.0) else "train"


def _normalized_template(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", text)
    text = re.sub(r"[A-Fa-f0-9]{16,}", "<id>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1200]


def semantic_group_id(sample: dict) -> str:
    src = str(sample.get("source") or "unknown")
    meta = sample.get("metadata", {}) or {}
    if meta.get("split_group_id"):
        return str(meta["split_group_id"])
    if src == "procua" and meta.get("collection_run") and meta.get("collection_run") != "unknown_run":
        return f"procua_run::{meta['collection_run']}"
    if src == "pcagente" and meta.get("task_family"):
        return str(meta["task_family"])
    if src == "videocua":
        for key in ("task_family", "asset_family", "task_id", "video_id", "asset_id"):
            if meta.get(key):
                return f"videocua::{key}::{meta[key]}"
    if src == "replay":
        # Content-template grouping is deliberately finer than category-level
        # grouping. Category-only grouping would put all math/coding in one split.
        user = "\n".join(str(m.get("content") or "") for m in sample.get("messages", [])
                         if m.get("role") == "user")
        templ = _normalized_template(user)
        digest = hashlib.sha256(templ.encode("utf-8")).hexdigest()[:20]
        return f"replay::{sample.get('task_type')}::{digest}"
    # Grounding and GUI assets are already one trajectory/sample per screen in
    # many cohorts; trajectory remains the safest leakage boundary when no
    # stronger provenance family exists.
    return str(sample.get("trajectory_id") or f"{src}:{sample.get('step_id','?')}")


def assign_splits(samples: Iterable[dict], validation_pct: float = DEFAULT_VALIDATION_PCT) -> List[dict]:
    out = []
    for s in samples:
        s = dict(s)
        s["metadata"] = dict(s.get("metadata", {}) or {})
        gid = semantic_group_id(s)
        s["metadata"]["split_group_id"] = gid
        s["split"] = split_of(gid, validation_pct)
        out.append(s)
    return out


def split_samples(samples: Iterable[dict]) -> Tuple[List[dict], List[dict]]:
    train = [s for s in samples if s.get("split") == "train"]
    val = [s for s in samples if s.get("split") == "validation"]
    return train, val


def trajectory_overlap(train: Iterable[dict], validation: Iterable[dict]) -> Set[str]:
    t_ids = {s.get("trajectory_id") for s in train}
    v_ids = {s.get("trajectory_id") for s in validation}
    return t_ids & v_ids


def group_overlap(train: Iterable[dict], validation: Iterable[dict]) -> Set[str]:
    t = {semantic_group_id(s) for s in train}
    v = {semantic_group_id(s) for s in validation}
    return t & v


def replay_sample_id(source: str, row_index: int, dataset_name: str) -> str:
    return f"replay::{dataset_name}::{source}::{row_index}"
