"""Stratified trajectory/step sampling (spec sections 6-10).

Per trajectory:
  - temporal coverage across 5 position buckets (beginning / early middle /
    middle / late middle / ending)
  - per-trajectory contribution cap (default 4 training items)
  - strong positive weighting (~2x) for difficult-state signals
  - downweighting of trivial/repeated/ordinary-navigation steps

Per source: application caps so one app cannot dominate.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .windows import Step, Trajectory

DEFAULT_PER_TRAJ_CAP = 4
PREFERRED_LENGTH = (6, 40)
SHORT_MEANINGFUL = 4
POSITIVE_WEIGHT = 2.0

POSITIVE_SIGNALS = {
    "recovery_evidenced", "failed_action",
    "verification", "verification_evidenced", "planning_evidenced", "save",
    "export", "modal_dialog", "file_dialog", "overwrite_confirmation",
    "sorting", "ranking", "exact_quantity", "multi_target", "multi_app",
    "uncommon_action", "scroll_read", "state_transition", "wait", "loading",
    "drag", "keyboard_shortcut", "small_target", "dense_ui", "finish_verification",
}
NEGATIVE_SIGNALS = {
    "repeated_identical_click", "trivial_open", "near_duplicate_no_recovery",
    "ordinary_navigation",
}

TASK_TEXT_PATTERNS = [
    (re.compile(r"\bsort(ed|ing)?\b|\barrange\b|\borderr?\b", re.I), "sorting"),
    (re.compile(r"\brank(ed|ing)?\b|\btop\s*\d+\b|\bhighest\b|\blowest\b", re.I), "ranking"),
    (re.compile(r"\bexactly\s+\d+\b|\b\d+\s+(items?|files?|words?|cells?|rows?|slides?)\b", re.I), "exact_quantity"),
    (re.compile(r"\band\b.*\balso\b|,\s*(then|and)\s*(open|create|rename|delete|move)\b", re.I), "multi_target"),
    (re.compile(r"\bsave(d)?\b|\bsave as\b", re.I), "save"),
    (re.compile(r"\bexport\b|\bdownload\b|\bconvert to\b", re.I), "export"),
    (re.compile(r"\bconfirm\b|\boverwrite\b|\bdialog\b|\bmodal\b", re.I), "modal_dialog"),
    (re.compile(r"\bin\s+(word|excel|powerpoint|ppt|browser|chrome|firefox|explorer)\b.*\bthen\b", re.I), "multi_app"),
]

VERB_RARITY = {
    "drag": "uncommon_action",
    "middle_click": "uncommon_action",
    "right_click": "uncommon_action",
    "hotkey": "keyboard_shortcut",
    "wait": "wait",
    "double_click": "uncommon_action",
}


def detect_task_signals(task_text: str, subgoal_text: str = "") -> Set[str]:
    signals: Set[str] = set()
    for pattern, name in TASK_TEXT_PATTERNS:
        if pattern.search(task_text or "") or pattern.search(subgoal_text or ""):
            signals.add(name)
    return signals


def step_base_signals(step: Step) -> Set[str]:
    signals = set(step.signals)
    if step.action is not None:
        rare = VERB_RARITY.get(step.action.verb)
        if rare:
            signals.add(rare)
    return signals


def anchor_eligible(step: Step) -> bool:
    """MOVE_TO / lone key_down / mouse_down ... never anchor a training item
    (VideoCUA audit: 45% MOVE_TO; ProCUA audit: 15% key_down). They remain in
    text history."""
    from processing.quality import ANCHOR_ELIGIBLE_VERBS
    return step.action is not None and step.action.verb in ANCHOR_ELIGIBLE_VERBS


@dataclass
class StepScore:
    index: int
    weight: float
    bucket: int
    signals: Set[str]


def _bucket_of(index: int, n: int) -> int:
    if n <= 1:
        return 2
    return min(4, int(5 * index / n))


def score_steps(traj: Trajectory) -> List[StepScore]:
    n = traj.length
    task_signals = detect_task_signals(traj.task)
    scores: List[StepScore] = []
    for i, step in enumerate(traj.steps):
        signals = step_base_signals(step) | task_signals
        weight = 1.0
        if signals & POSITIVE_SIGNALS:
            weight *= POSITIVE_WEIGHT
        if signals & NEGATIVE_SIGNALS:
            weight *= 0.5
        # Pixel no-change is recorded for audit but does not increase sampling
        # weight by itself; it is not causal evidence of an ineffective action.
        if step.action is None or not anchor_eligible(step):
            weight = 0.0
        scores.append(StepScore(index=i, weight=weight, bucket=_bucket_of(i, n), signals=signals))
    return scores


def select_step_indices(traj: Trajectory, cap: int = DEFAULT_PER_TRAJ_CAP) -> List[int]:
    """Select up to `cap` step indices with temporal bucket coverage, ranked
    by weight within each bucket."""
    scores = [s for s in score_steps(traj) if s.weight > 0]
    if not scores:
        return []
    by_bucket: Dict[int, List[StepScore]] = {}
    for s in scores:
        by_bucket.setdefault(s.bucket, []).append(s)

    picked: List[int] = []
    # round-robin buckets from the middle outward for position diversity
    order = [2, 1, 3, 0, 4]
    round_idx = 0
    while len(picked) < cap:
        added = False
        for b in order:
            bucket = by_bucket.get(b)
            if not bucket:
                continue
            # rank by weight, deterministic tie-break on index
            bucket.sort(key=lambda s: (-s.weight, s.index))
            take = min(round_idx, len(bucket))
            if take >= len(bucket):
                continue
            chosen = bucket[take]
            if chosen.index not in picked:
                picked.append(chosen.index)
                added = True
            if len(picked) >= cap:
                break
        if not added:
            break
        round_idx += 1
    return sorted(picked)


def action_representation_specs(traj: Trajectory, cap: int = DEFAULT_PER_TRAJ_CAP,
                                ratios: Optional[dict] = None) -> list:
    """Representation emission for trajectory-rich action sources (procua,
    videocua), designed for the SAMPLE-LEVEL 65/30/5 target band.

    Roles (drawn per trajectory, deterministic by id):
      single -> up to `cap` singles
      window -> up to 3 non-overlapping INFORMATIVE windows + 1 single
      chunk  -> 1 informative chunk + 1 single
    Windows/chunks that fail the informativeness gates fall back to the next
    cheaper representation, never to junk. No trajectory is duplicated with
    near-identical context: extra singles from window/chunk trajectories were
    removed, windows use distinct non-overlapping starts.
    """
    from .windows import (ACTION_SOURCE_RATIOS, CHUNK, SINGLE, WINDOW,
                          build_chunk, build_single, build_window,
                          chunk_is_informative, choose_representation,
                          suggest_window_starts, window_is_informative)
    role = choose_representation(traj.trajectory_id, ratios or ACTION_SOURCE_RATIOS)
    indices = select_step_indices(traj, cap=cap)
    if not indices:
        return []

    def informative_windows(count: int) -> list:
        """Non-overlapping informative windows; length 4 normally, clipped to
        the trajectory (WINDOW_MIN 3) so short trajectories still yield one."""
        length = min(4, traj.length)
        if length < 3:
            return []
        picked, used = [], []
        # extra candidates so overlap-skip and quality gates can lose some
        for s in suggest_window_starts(traj, length, count=count + 2):
            if any(abs(s - t) < length for t in used):
                continue  # overlap with an already-picked window
            if not window_is_informative(traj.steps[s:s + length]):
                continue
            used.append(s)
            picked.append(build_window(traj, s, length))
            if len(picked) >= count:
                break
        return picked

    specs = []
    if role == SINGLE:
        specs = [build_single(traj, i) for i in indices]
    elif role == WINDOW:
        specs = informative_windows(3)
        specs.append(build_single(traj, indices[0]))
        if len(specs) == 1:  # no informative window survived -> plain singles
            specs = [build_single(traj, i) for i in indices[:2]]
    else:  # CHUNK
        made = False
        starts8 = suggest_window_starts(traj, 8, count=1)
        if traj.length >= 8 and starts8:
            start = starts8[0]
            length = min(12, traj.length - start)
            if chunk_is_informative(traj.steps[start:start + length]):
                specs.append(build_chunk(traj, start, length))
                made = True
        if not made:
            specs.extend(informative_windows(2))
        specs.append(build_single(traj, indices[0]))

    # Populate anchor-local quality context consistently for every action
    # source. This used to be PC-Agent-E-only, making ProCUA/VideoCUA quality
    # scoring silently fall back to optimistic defaults.
    for spec in specs:
        try:
            idx = traj.steps.index(spec.current_step)
        except ValueError:
            continue
        spec.metadata["_is_first_step"] = (idx == 0 or
            (idx > 0 and (traj.steps[idx - 1].metadata or {}).get("continuity_id", 0) !=
             (spec.current_step.metadata or {}).get("continuity_id", 0)))
        seg = (spec.current_step.metadata or {}).get("continuity_id", 0)
        prev = []
        for st in reversed(traj.steps[:idx]):
            if (st.metadata or {}).get("continuity_id", 0) != seg:
                break
            prev.append(st)
            if len(prev) >= 4:
                break
        spec.metadata["_prev_steps"] = list(reversed(prev))
        # A missing previous state means "unknown / segment start", not
        # evidence of a visible state transition.  Treating first steps as
        # changed falsely evidenced synthetic finish labels and inflated the
        # state-value quality component.
        if spec.current_step.prev_phash is None or spec.current_step.phash is None:
            spec.metadata["_state_changed"] = None
        else:
            spec.metadata["_state_changed"] = (
                bin(spec.current_step.phash ^ spec.current_step.prev_phash).count("1") > 6)
    return specs[:cap]


def trajectory_priority(traj: Trajectory) -> float:
    """Preference multiplier for trajectories (6-40 actions preferred; very
    short trajectories downweighted)."""
    n = traj.length
    if n < SHORT_MEANINGFUL:
        return 0.5
    if PREFERRED_LENGTH[0] <= n <= PREFERRED_LENGTH[1]:
        return 1.0
    if n < PREFERRED_LENGTH[0]:
        return 0.8
    return 0.9  # long trajectories still valuable (recovery/verification states)


@dataclass
class AppCap:
    cap: int
    counts: Dict[str, int] = field(default_factory=dict)

    def allow(self, app: str, boost: float = 1.0) -> bool:
        app = (app or "unknown").lower()
        effective = max(1, int(self.cap * boost))
        return self.counts.get(app, 0) < effective

    def record(self, app: str, count: int = 1):
        app = (app or "unknown").lower()
        self.counts[app] = self.counts.get(app, 0) + count


def deterministic_keep(trajectory_id: str, priority: float, seed: str = "jxagent") -> bool:
    """Deterministic downweighting: priority<1 trajectories are skipped with
    probability (1 - priority)."""
    if priority >= 1.0:
        return True
    u = int(hashlib.md5(f"{seed}:{trajectory_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return u < priority
