"""Resumable, atomic build state (spec sections 24-25).

All state files are written atomically (tmp file + os.replace). Rerunning a
build continues from recorded progress: processed shards/archives, selected
trajectories/rows, image hashes, failures and per-source counts.

Corrupt state files are recovered from defaults rather than crashing, but are
ALWAYS reported loudly (stderr) and recorded in `corruption_events` so the
final build report can carry `state_corruption_detected: true` — a silently
ignored dedup index would weaken dedup without anyone noticing.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from .dedup import DedupIndex, index_from_state, index_to_state

# credential-like patterns that must never appear in build artifacts
_SECRET_RES = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),           # HF access token
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),    # GitHub token
    re.compile(r"Authorization:\s*\S+", re.I),    # raw auth header
]


def redact_secrets(text: str) -> str:
    """Replace credential-like substrings before anything is logged or written
    to a published artifact. Never prints the secret itself."""
    for pat in _SECRET_RES:
        text = pat.sub("[REDACTED]", text or "")
    return text


def contains_secret(text: str) -> bool:
    return any(pat.search(text or "") for pat in _SECRET_RES)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except (OSError, Exception):
        return False


class BuildLock:
    """Lightweight single-builder lock (state/build.lock).

    Fails clearly when another live builder holds the lock; auto-removes a
    stale lock left by a crashed process (dead pid); `force_clear` removes it
    unconditionally (explicit stale-lock recovery for a hung builder)."""

    def __init__(self, state_dir: str):
        self.path = os.path.join(state_dir, "build.lock")
        self.held = False

    def acquire(self, force_clear: bool = False) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if force_clear and os.path.exists(self.path):
            os.remove(self.path)
            print(f"[lock] cleared {self.path} (forced)", flush=True)
        if os.path.exists(self.path):
            info = {}
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                info = {}
            pid = int(info.get("pid", -1))
            if pid == os.getpid() or not _pid_alive(pid):
                why = "our own pid" if pid == os.getpid() else f"dead pid {pid}"
                print(f"[lock] removing stale lock ({why})", flush=True)
                os.remove(self.path)
            else:
                raise RuntimeError(
                    f"another builder appears active: {self.path} says pid={pid} "
                    f"host={info.get('hostname', '?')} started={info.get('started', '?')}. "
                    f"Stop it, or pass --clear-lock if you are certain it is stale.")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "hostname": socket.gethostname(),
                       "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        self.held = True

    def release(self) -> None:
        if self.held and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        self.held = False


class BuildState:
    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.progress_path = os.path.join(state_dir, "progress.json")
        self.shards_path = os.path.join(state_dir, "processed_shards.json")
        self.lock = threading.RLock()
        self.corruption_events: List[Dict[str, str]] = []
        self.progress: Dict[str, Any] = self._load(self.progress_path, {
            "sources": {},           # source -> {target, selected, rejected, ...}
            "started_at": None,
            "updated_at": None,
        })
        self.processed_shards: Dict[str, List[str]] = self._load(self.shards_path, {})

    # ------------------------------------------------------------------ io
    def _load(self, path: str, default):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                event = {"path": os.path.basename(path), "error": str(e)[:200]}
                self.corruption_events.append(event)
                print(f"[STATE CORRUPTION] {path}: {e}; "
                      f"recovering from defaults — prior progress in this file "
                      f"is LOST and will be redone (dedup state loss weakens "
                      f"dedup!)", file=sys.stderr, flush=True)
                return default
        return default

    def corruption_summary(self) -> Dict[str, Any]:
        return {
            "state_corruption_detected": bool(self.corruption_events),
            "events": list(self.corruption_events),
        }

    @staticmethod
    def atomic_write_json(path: str, obj: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000) % 1000000}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    def save(self) -> None:
        with self.lock:
            self.progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            if not self.progress.get("started_at"):
                self.progress["started_at"] = self.progress["updated_at"]
            self.atomic_write_json(self.progress_path, self.progress)
            self.atomic_write_json(self.shards_path, self.processed_shards)

    # -------------------------------------------------------------- shards
    def is_shard_done(self, source: str, shard: str) -> bool:
        return shard in set(self.processed_shards.get(source, []))

    def mark_shard_done(self, source: str, shard: str) -> None:
        with self.lock:
            lst = self.processed_shards.setdefault(source, [])
            if shard not in lst:
                lst.append(shard)

    def shard_count(self, source: str) -> int:
        return len(self.processed_shards.get(source, []))

    # ------------------------------------------------------------- counts
    def source_counts(self, source: str) -> Dict[str, Any]:
        return self.progress["sources"].setdefault(source, {
            "selected": 0, "rejected": 0, "target": None,
            "rejected_by_reason": {},
            "trajectories": 0,
        })

    def add_selected(self, source: str, n: int = 1, trajectories: int = 0) -> None:
        with self.lock:
            c = self.source_counts(source)
            c["selected"] += n
            c["trajectories"] += trajectories

    def add_rejection(self, source: str, reason: str, n: int = 1) -> None:
        with self.lock:
            c = self.source_counts(source)
            c["rejected"] += n
            c["rejected_by_reason"][reason] = c["rejected_by_reason"].get(reason, 0) + n

    def set_target(self, source: str, target: int) -> None:
        with self.lock:
            self.source_counts(source)["target"] = target

    def selected_total(self, source: str) -> int:
        return self.progress["sources"].get(source, {}).get("selected", 0)

    # -------------------------------------------------------- append jsonl
    def append_jsonl(self, name: str, rows: List[dict]) -> None:
        if not rows:
            return
        path = os.path.join(self.state_dir, name)
        os.makedirs(self.state_dir, exist_ok=True)
        with self.lock:
            with open(path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read_jsonl(self, name: str) -> List[dict]:
        path = os.path.join(self.state_dir, name)
        if not os.path.exists(path):
            return []
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    # ----------------------------------------------------------- dedup idx
    def load_dedup_index(self) -> DedupIndex:
        path = os.path.join(self.state_dir, "dedup_index.json")
        if os.path.exists(path):
            return index_from_state(self._load(path, {}))
        return DedupIndex()

    def save_dedup_index(self, index: DedupIndex) -> None:
        self.atomic_write_json(os.path.join(self.state_dir, "dedup_index.json"),
                               index_to_state(index))
