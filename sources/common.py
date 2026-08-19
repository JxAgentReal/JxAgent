"""Shared context passed to every source adapter."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from processing.decontamination import Decontaminator
from processing.dedup import DedupIndex
from processing.reasoning import ReasoningGate
from processing.state import BuildState


@dataclass
class BuildContext:
    dataset_root: str
    state: BuildState
    config: Dict[str, Any]
    session: Optional[requests.Session] = None
    dedup: Optional[DedupIndex] = None
    decontaminator: Optional[Decontaminator] = None
    reasoning_gate: ReasoningGate = field(default_factory=ReasoningGate)
    offline: bool = False
    smoke: bool = False
    quota: Dict[str, int] = field(default_factory=dict)  # source -> remaining count
    rejected: Dict[str, Dict[str, int]] = field(default_factory=dict)
    app_counter: Dict[str, int] = field(default_factory=dict)
    total_samples: int = 0
    # Canonical replay IDs already durably selected in this or a prior resume.
    # Used to skip duplicate content before any image path can collide.
    seen_replay_ids: set[str] = field(default_factory=set)


    def reserve_replay_id(self, sample_id: str) -> bool:
        """Atomically reserve a canonical replay sample ID for this build.

        Returns False for a duplicate so the source can keep scanning until
        its exact category quota is filled. Durable IDs are seeded by the
        builder on resume.
        """
        if sample_id in self.seen_replay_ids:
            self.reject("replay", "duplicate_canonical_id")
            return False
        self.seen_replay_ids.add(sample_id)
        return True

    def note_app(self, app: str):
        key = (app or "unknown").lower()
        self.app_counter[key] = self.app_counter.get(key, 0) + 1
        self.total_samples += 1

    def remaining(self, source: str) -> int:
        return self.quota.get(source, 0)

    def consume(self, source: str, n: int = 1):
        self.quota[source] = max(0, self.quota.get(source, 0) - n)

    def reject(self, source: str, reason: str, n: int = 1):
        self.state.add_rejection(source, reason, n)
        d = self.rejected.setdefault(source, {})
        d[reason] = d.get(reason, 0) + n

    def note_stat(self, source: str, key: str, n: int = 1):
        """Non-rejection bookkeeping (e.g. dimension-mismatch counters) kept in
        state.progress for manifests and the final mixture report."""
        c = self.state.source_counts(source)
        notes = c.setdefault("notes", {})
        notes[key] = notes.get(key, 0) + n

    def persist_samples(self, samples: List[dict]):
        """Durably record assembled samples immediately. Must be called BEFORE
        the unit that produced them is marked done, so a hard kill can never
        leave 'shard done but samples lost' (unrecoverable under-fill)."""
        if samples:
            self.state.append_jsonl("selected_samples.jsonl", samples)

    def http(self) -> requests.Session:
        if self.session is None:
            from processing.remote_access import session_with_headers
            self.session = session_with_headers()
        return self.session

    def decontaminate(self, task_text: str, source: str) -> bool:
        """True when the task text must be REMOVED (contaminated)."""
        if self.decontaminator is None:
            return False
        remove, _ = self.decontaminator.check_sample(task_text, source)
        return remove


def images_root(ctx: BuildContext, source: str) -> str:
    p = os.path.join(ctx.dataset_root, "images", source)
    os.makedirs(p, exist_ok=True)
    return p
