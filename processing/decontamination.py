"""OSWorld contamination protection (spec section 20).

Reference set: public OSWorld task instructions fetched from the OSWorld
repository (evaluation_examples), cached under state/. Matching:
  - normalized exact hash match
  - word 8-gram Jaccard similarity >= threshold (default 0.5)
  - containment match for short candidate instructions (all shingles present)
Application overlap alone is NOT a reason for removal.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

from sources.revisions import REVISIONS as _SRC_REVISIONS

# Immutable Run 1 snapshot of the OSWorld evaluation examples (GitHub commit
# SHA; the floating 'main' ref would make decontamination non-reproducible).
OSWORLD_REVISION = _SRC_REVISIONS["osworld"]["sha"]
OSWORLD_CACHE_SHA256 = _SRC_REVISIONS["osworld"]["cache_sha256"]
OSWORLD_EXAMPLES_DIR = ("https://api.github.com/repos/xlang-ai/OSWorld/contents/"
                        f"evaluation_examples/examples?ref={OSWORLD_REVISION}")
OSWORLD_RAW = ("https://raw.githubusercontent.com/xlang-ai/OSWorld/"
               f"{OSWORLD_REVISION}/evaluation_examples/examples")
NGRAM = 8
SIMILARITY_THRESHOLD = 0.5

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    words = _WORD_RE.findall((text or "").lower())
    return " ".join(words)


def shingles(text: str, n: int = NGRAM) -> Set[str]:
    words = normalize(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def containment(a: Set[str], b: Set[str]) -> float:
    """Fraction of a's shingles found in b (short-query containment)."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


@dataclass
class Decontaminator:
    references: List[Dict] = field(default_factory=list)  # {task_id, instruction, shingles}
    exact_hashes: Set[str] = field(default_factory=set)
    threshold: float = SIMILARITY_THRESHOLD
    stats: Dict[str, int] = field(default_factory=lambda: {
        "total_scanned": 0, "exact_matches": 0, "high_similarity_matches": 0,
        "removed_examples": 0,
    })
    removed_by_source: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for ref in self.references:
            h = _norm_hash(ref["instruction"])
            self.exact_hashes.add(h)

    def add_reference(self, task_id: str, instruction: str):
        self.references.append({"task_id": task_id, "instruction": instruction,
                                "shingles": shingles(instruction),
                                "words": set(normalize(instruction).split())})
        self.exact_hashes.add(_norm_hash(instruction))

    def check(self, task_text: str) -> Tuple[bool, str, Optional[float]]:
        """Returns (remove, reason, similarity)."""
        self.stats["total_scanned"] += 1
        h = _norm_hash(task_text)
        if h in self.exact_hashes:
            self.stats["exact_matches"] += 1
            return True, "exact_match", 1.0
        cand = shingles(task_text)
        if not cand:
            return False, "", None
        words = set(normalize(task_text).split())
        best = 0.0
        best_contain = 0.0
        short_candidate = len(words) < NGRAM
        for ref in self.references:
            s = jaccard(cand, ref["shingles"])
            if s > best:
                best = s
            if short_candidate:
                c = len(words & ref["words"]) / max(1, len(words))
            else:
                c = containment(cand, ref["shingles"])
            if c > best_contain:
                best_contain = c
        score = max(best, best_contain)
        if score >= self.threshold:
            self.stats["high_similarity_matches"] += 1
            return True, f"high_similarity:{score:.3f}", score
        return False, "", score

    def check_sample(self, task_text: str, source: str) -> Tuple[bool, str]:
        remove, reason, _ = self.check(task_text)
        if remove:
            self.stats["removed_examples"] += 1
            self.removed_by_source[source] = self.removed_by_source.get(source, 0) + 1
        return remove, reason

    def report(self) -> Dict:
        return {
            "total_scanned": self.stats["total_scanned"],
            "exact_matches": self.stats["exact_matches"],
            "high_similarity_matches": self.stats["high_similarity_matches"],
            "removed_examples": self.stats["removed_examples"],
            "source_breakdown": dict(self.removed_by_source),
            "reference_instructions": len(self.references),
            "ngram": NGRAM,
            "similarity_threshold": self.threshold,
        }


def _norm_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def load_osworld_instructions_from_bundle(bundle: dict) -> List[Tuple[str, str]]:
    """OSWorld evaluation_examples.json style: {domain: [ {id, instruction} ]}."""
    out: List[Tuple[str, str]] = []
    for domain, tasks in (bundle or {}).items():
        if isinstance(tasks, dict):
            tasks = tasks.get("tasks", [])
        for t in tasks or []:
            tid = t.get("id") or f"{domain}_{len(out)}"
            instr = t.get("instruction") or ""
            if instr:
                out.append((f"{domain}/{tid}", instr))
    return out


def fetch_reference_instructions(cache_dir: str, session: Optional[requests.Session] = None,
                                 offline: bool = False,
                                 shared_cache_paths: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Fetch + cache OSWorld task instructions (evaluation_examples/examples/
    <domain>/<task_id>.json -> 'instruction'). Raises on failure unless a
    cache already exists. `shared_cache_paths` are checked before fetching so
    repeated smoke builds do not re-download ~369 reference files."""
    cache_path = os.path.join(cache_dir, "osworld_instructions.json")
    candidates = [cache_path] + list(shared_cache_paths or [])
    for cand in candidates:
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as f:
                pairs = [tuple(x) for x in json.load(f)]
            if pairs:
                if cand != cache_path:  # propagate to this build's state dir
                    os.makedirs(cache_dir, exist_ok=True)
                    tmp = cache_path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(pairs, f)
                    os.replace(tmp, cache_path)
                return pairs
    if offline:
        raise RuntimeError("offline mode and no cached OSWorld instructions")
    session = session or requests.Session()
    pairs: List[Tuple[str, str]] = []
    try:
        domains = session.get(OSWORLD_EXAMPLES_DIR, timeout=60)
        domains.raise_for_status()
        domain_names = [d["name"] for d in domains.json()
                        if isinstance(d, dict) and d.get("type") == "dir"]
        for domain in domain_names:
            listing = session.get(f"{OSWORLD_EXAMPLES_DIR}/{domain}", timeout=60)
            listing.raise_for_status()
            for item in listing.json():
                if not isinstance(item, dict) or not item.get("name", "").endswith(".json"):
                    continue
                task_id = item["name"][:-5]
                try:
                    task = session.get(f"{OSWORLD_RAW}/{domain}/{task_id}.json", timeout=60)
                    task.raise_for_status()
                    instruction = (task.json() or {}).get("instruction") or ""
                except Exception:
                    continue
                if instruction:
                    pairs.append((f"{domain}/{task_id}", instruction))
    except Exception as e:  # noqa: BLE001
        if not pairs:
            raise RuntimeError(f"could not fetch OSWorld reference instructions: {e}")
    if not pairs:
        raise RuntimeError("no OSWorld reference instructions fetched")
    for cand in candidates:
        os.makedirs(os.path.dirname(cand) or ".", exist_ok=True)
        tmp = cand + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pairs, f)
        os.replace(tmp, cand)
    return pairs


def load_reference_file(path: str) -> List[Tuple[str, str]]:
    """Load extra benchmark instructions without assuming a benchmark schema.

    Accepted formats:
      * JSONL rows with instruction/prompt/task/question/text
      * JSON lists of strings or objects
      * JSON dictionaries containing task lists or benchmark->task lists
      * plain text, one non-empty instruction per line

    This is decontamination-only input. It never becomes training data.
    """
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(p)

    def text_from_obj(obj) -> str:
        if isinstance(obj, str):
            return obj.strip()
        if not isinstance(obj, dict):
            return ""
        for key in ("instruction", "prompt", "task", "question", "text"):
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def id_from_obj(obj, fallback: str) -> str:
        if isinstance(obj, dict):
            for key in ("id", "task_id", "uuid", "name"):
                if obj.get(key) is not None:
                    return str(obj[key])
        return fallback

    ext = os.path.splitext(p)[1].lower()
    out: List[Tuple[str, str]] = []
    base = os.path.basename(p)
    if ext == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = text_from_obj(obj)
                if text:
                    out.append((f"extra/{base}/{id_from_obj(obj, str(i))}", text))
        return out

    if ext == ".json":
        obj = json.loads(open(p, "r", encoding="utf-8").read())
        queue = [("root", obj)]
        while queue:
            prefix, cur = queue.pop(0)
            text = text_from_obj(cur)
            if text:
                out.append((f"extra/{base}/{id_from_obj(cur, prefix)}", text))
                continue
            if isinstance(cur, list):
                queue.extend((f"{prefix}/{i}", v) for i, v in enumerate(cur))
            elif isinstance(cur, dict):
                for k, v in cur.items():
                    if isinstance(v, (list, dict)):
                        queue.append((f"{prefix}/{k}", v))
        return out

    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            text = line.strip()
            if text:
                out.append((f"extra/{base}/{i}", text))
    return out
