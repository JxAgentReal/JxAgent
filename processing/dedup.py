"""Near-duplicate detection: perceptual hash (DCT-based pHash, pure numpy)
plus context-aware filtering (spec section 18) and cross-source dedup by
task/action/image hashes (spec section 19).

Meaningful repetition with explicit causal evidence (failed action, recovery,
loading or evidenced wait/verification) is preserved even when frames are
near-identical. Pixel no-change alone is deliberately insufficient.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

# Signals under which near-duplicate visuals are KEPT (context aware).
MEANINGFUL_REPETITION_SIGNALS = {
    "failed_action", "recovery_evidenced", "loading",
    "wait_evidenced", "verification_evidenced",
}

DEFAULT_PHASH_THRESHOLD = 6  # hamming distance <= threshold => near duplicate


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    d = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
    d[0, :] = np.sqrt(1.0 / n)
    return d


_D8x32 = _dct_matrix(32)[:8, :]  # 8x32 projection (first 8 DCT-II basis rows)


def phash(img: Image.Image, hash_size: int = 8, img_size: int = 32) -> int:
    """Standard DCT pHash: 64-bit int. Deterministic, pure numpy."""
    gray = np.asarray(img.convert("L").resize((img_size, img_size), Image.LANCZOS), dtype=np.float64)
    dct = _D8x32 @ gray @ _D8x32.T  # 8x8 low-frequency block
    flat = dct.flatten()
    low = flat[1:]  # skip DC term for median
    med = np.median(low)
    bits = flat > med  # includes DC bit; standard implementations differ, this is consistent
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def phash_bytes(data: bytes) -> int:
    import io
    return phash(Image.open(io.BytesIO(data)))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8", "ignore")).hexdigest()


class _BKNode:
    __slots__ = ("value", "children")
    def __init__(self, value: int):
        self.value = int(value)
        self.children: Dict[int, "_BKNode"] = {}


def _bk_insert(root: _BKNode, value: int) -> None:
    node = root
    while True:
        d = hamming(value, node.value)
        if d == 0:
            return
        nxt = node.children.get(d)
        if nxt is None:
            node.children[d] = _BKNode(value)
            return
        node = nxt


def _bk_find(root: Optional[_BKNode], value: int, radius: int) -> Optional[int]:
    if root is None:
        return None
    stack = [root]
    while stack:
        node = stack.pop()
        d = hamming(value, node.value)
        if d <= radius:
            return node.value
        lo, hi = d - radius, d + radius
        stack.extend(child for edge, child in node.children.items() if lo <= edge <= hi)
    return None


@dataclass
class DedupIndex:
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD
    _seen_phashs: List[int] = field(default_factory=list)
    # phash -> set of action-text hashes seen on that (near-)identical screen
    _phash_actions: Dict[int, Set[str]] = field(default_factory=dict)
    # Exact supervision key includes visual state. Same task/action on a
    # genuinely different screen is not a duplicate decision.
    _task_action_hashes: Set[str] = field(default_factory=set)
    _exact_image_hashes: Set[str] = field(default_factory=set)
    _bk_root: Optional[_BKNode] = field(default=None, init=False, repr=False)
    _bk_ready: bool = field(default=False, init=False, repr=False)
    stats: Dict[str, int] = field(default_factory=lambda: {
        "exact_image_duplicates": 0,
        "near_duplicates_removed": 0,
        "near_duplicates_kept_meaningful": 0,
        "task_action_duplicates": 0,
    })

    def _ensure_bk(self) -> None:
        if self._bk_ready:
            return
        self._bk_root = None
        for h in self._seen_phashs:
            if self._bk_root is None:
                self._bk_root = _BKNode(h)
            else:
                _bk_insert(self._bk_root, h)
        self._bk_ready = True

    def _near_hash(self, h: int) -> Optional[int]:
        self._ensure_bk()
        return _bk_find(self._bk_root, h, self.phash_threshold)

    def _register_phash(self, h: int) -> None:
        self._ensure_bk()
        if self._bk_root is None:
            self._bk_root = _BKNode(h)
        else:
            _bk_insert(self._bk_root, h)
        self._seen_phashs.append(h)

    def is_near_duplicate(self, h: int) -> bool:
        return self._near_hash(h) is not None

    def register_image(self, data: bytes) -> bool:
        """Register image bytes; returns True when the exact bytes repeat."""
        eh = sha256_hex(data)
        if eh in self._exact_image_hashes:
            self.stats["exact_image_duplicates"] += 1
            return True
        self._exact_image_hashes.add(eh)
        return False

    def consider(self, *, image_phash: Optional[int], signals: Iterable[str],
                 task_text: str, action_text: str) -> Tuple[bool, str]:
        """Decide whether a candidate step is a practical duplicate.

        Removal rules (context aware):
          1. identical (task, action, visual state) supervision -> duplicate
          2. near-identical screenshot AND the same action already taken on a
             near-identical screenshot -> duplicate, UNLESS the repetition is
             meaningful (evidenced recovery / wait / loading ...)
        Same/near-identical screenshot with a DIFFERENT action is kept: it is
        distinct supervision on the same visible state (e.g. grounding).
        """
        action_h = text_hash(action_text)
        state_key = str(image_phash) if image_phash is not None else "no_visual_state"
        ta = text_hash(task_text + "\n" + action_text + "\nstate=" + state_key)
        if ta in self._task_action_hashes:
            self.stats["task_action_duplicates"] += 1
            return True, "task_action_state_duplicate"
        if image_phash is not None:
            meaningful = any(s in MEANINGFUL_REPETITION_SIGNALS for s in signals)
            near = self._near_hash(image_phash)
            canonical_state = near if near is not None else image_phash
            if near is not None:
                if action_h in self._phash_actions.get(canonical_state, set()):
                    if meaningful:
                        self.stats["near_duplicates_kept_meaningful"] += 1
                    else:
                        self.stats["near_duplicates_removed"] += 1
                        return True, "near_duplicate_image_action"
                elif meaningful:
                    self.stats["near_duplicates_kept_meaningful"] += 1
            else:
                self._register_phash(image_phash)
            # Store supervision against the canonical BK-tree representative.
            # Otherwise a near-duplicate state with a new action would be
            # written under an unindexed pHash and a later exact repeat of
            # that same near-state/action could slip through.
            self._phash_actions.setdefault(canonical_state, set()).add(action_h)
        self._task_action_hashes.add(ta)
        return False, ""


# --------------------------- state (de)serialization ----------------------

def index_to_state(index: DedupIndex) -> dict:
    return {
        "phash_threshold": index.phash_threshold,
        "seen_phashs": index._seen_phashs,
        "phash_actions": {str(k): sorted(v) for k, v in index._phash_actions.items()},
        "task_action_hashes": sorted(index._task_action_hashes),
        "exact_image_hashes": sorted(index._exact_image_hashes),
        "stats": index.stats,
    }


def index_from_state(state: dict) -> DedupIndex:
    idx = DedupIndex(phash_threshold=state.get("phash_threshold", DEFAULT_PHASH_THRESHOLD))
    idx._seen_phashs = list(state.get("seen_phashs", []))
    idx._phash_actions = {int(k): set(v) for k, v in state.get("phash_actions", {}).items()}
    idx._task_action_hashes = set(state.get("task_action_hashes", []))
    idx._exact_image_hashes = set(state.get("exact_image_hashes", []))
    idx.stats.update(state.get("stats", {}))
    return idx
