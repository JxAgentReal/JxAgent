"""Trajectory data model and window construction (spec section 12).

Representations for computer-use data. Run 1 enforces one hard visual
invariant: every supervised target action is conditioned on its own real
pre-action screenshot. Window/chunk variants therefore carry prior observed
actions as bounded text history but still supervise exactly one next action.
The realized SAMPLE-level targets are ACTION_SAMPLE_TARGETS below.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .coordinates import Action
from .dedup import hamming
from .token_budget import HistoryItem, fit_to_budget

SINGLE = "single"
WINDOW = "window"
CHUNK = "chunk"

# 2026-08 audit: windows roughly double image tokens for moderate sequential
# gain; singles carry most unique supervision per token. Below: the trajectory
# draw ratios (generic default) AND the SAMPLE-LEVEL design targets.
REPRESENTATION_RATIOS = {SINGLE: 0.55, WINDOW: 0.40, CHUNK: 0.05}
# Sample-level design target for trajectory-rich action sources (brief
# 2026-08-16: 65/30/5 band 60-68/27-35/4-7). Because window/chunk-role
# trajectories also emit one single each, and gui360-use + pc-agent-e are
# single-only, the per-trajectory draw ratios for procua/videocua are NOT the
# sample-level targets; they are solved offline by tools/simulate_representation.py
# (tests assert the simulated sample-level mix stays inside the band).
ACTION_SAMPLE_TARGETS = {SINGLE: 0.65, WINDOW: 0.30, CHUNK: 0.05}
ACTION_SAMPLE_BANDS = {SINGLE: (0.60, 0.68), WINDOW: (0.27, 0.35), CHUNK: (0.04, 0.07)}
# Verified by tools/simulate_representation.py (2026-08-16): simulated
# sample-level mix 64.1% single / 31.0% window / 4.9% chunk — mid-band.
ACTION_SOURCE_RATIOS = {SINGLE: 0.16, WINDOW: 0.60, CHUNK: 0.24}
WINDOW_MIN, WINDOW_MAX = 3, 5
CHUNK_MIN, CHUNK_MAX = 8, 12


@dataclass
class Step:
    step_id: str
    image_bytes: bytes                    # screenshot BEFORE the action
    image_size: Tuple[int, int]           # original (w, h)
    action: Optional[Action]              # unified action in ORIGINAL pixel space
    phash: Optional[int] = None
    prev_phash: Optional[int] = None      # phash of the previous step's screenshot
    subgoal: str = ""
    signals: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)

    @property
    def action_text(self) -> str:
        return self.action.render() if self.action else ""


@dataclass
class Trajectory:
    trajectory_id: str
    task: str
    steps: List[Step] = field(default_factory=list)
    app: str = ""
    source: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.task:
            raise ValueError("trajectory requires a task")

    @property
    def length(self) -> int:
        return len(self.steps)


@dataclass
class SampleSpec:
    """Intermediate sample before image processing/finalization."""
    source: str
    trajectory_id: str
    step_ids: List[str]
    representation: str            # single | window | chunk | pass_through
    task: str
    current_step: Step             # anchor step (its image is the main input)
    extra_images: List[Tuple[str, Step]] = field(default_factory=list)  # (label, step)
    history_texts: List[str] = field(default_factory=list)
    assistant_turns: List[str] = field(default_factory=list)  # for multi-turn samples
    signals: Set[str] = field(default_factory=set)
    app: str = ""
    task_type: str = "action"
    metadata: Dict = field(default_factory=dict)


def seeded_bucket(text: str) -> float:
    """Deterministic uniform [0,1) from text."""
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF


def choose_representation(trajectory_id: str, ratios: Dict[str, float] = None) -> str:
    r = ratios or REPRESENTATION_RATIOS
    u = seeded_bucket("repr:" + trajectory_id)
    acc = 0.0
    for name in (SINGLE, WINDOW, CHUNK):
        acc += r[name]
        if u < acc:
            return name
    return SINGLE


def frames_add_information(prev_phash: Optional[int], cur_phash: Optional[int],
                           threshold: int = 6) -> bool:
    """A frame adds state information when it is not a near-duplicate of the
    previous frame. Frames that do NOT add information are exactly the
    no-state-change / loading states we may describe compactly in text."""
    if prev_phash is None or cur_phash is None:
        return True
    return hamming(prev_phash, cur_phash) > threshold


def _continuity_id(step: Step):
    return (step.metadata or {}).get("continuity_id", 0)


def _same_continuity(steps: Sequence[Step]) -> bool:
    return len({_continuity_id(s) for s in steps}) <= 1


def _compact_history(steps: Sequence[Step], start: int, max_items: int = 8) -> List[str]:
    """Only include causally contiguous previous actions.

    Parser/missing-image gaps are marked by source adapters with continuity_id;
    history must never jump across an unobserved or unsupported action.
    """
    if start <= 0 or start >= len(steps):
        return []
    seg = _continuity_id(steps[start])
    prior = []
    for s in reversed(list(steps)[:start]):
        if _continuity_id(s) != seg:
            break
        prior.append(s)
        if len(prior) >= max_items:
            break
    return [x.action_text for x in reversed(prior) if x.action_text]


def build_single(traj: Trajectory, idx: int, history_steps: int = 5) -> SampleSpec:
    step = traj.steps[idx]
    hist = _compact_history(traj.steps, idx, history_steps)
    return SampleSpec(
        source=traj.source, trajectory_id=traj.trajectory_id,
        step_ids=[step.step_id], representation=SINGLE, task=traj.task,
        current_step=step, history_texts=hist,
        assistant_turns=[step.action_text], signals=set(step.signals), app=traj.app,
        metadata=dict(traj.metadata),
    )


def build_window(traj: Trajectory, start: int, length: int,
                 max_extra_images: int = 0) -> SampleSpec:
    """History-enriched next-action window with a strict visual invariant.

    Every supervised assistant target must be conditioned on the screenshot
    that exists *before that exact action*.  The old multi-turn representation
    emitted several action targets while omitting intermediate screenshots,
    which could pair an action with a stale visual state.  A window now keeps
    the temporal value as text history but supervises only its final action,
    whose real pre-action screenshot is ``current_step.image_bytes``.
    """
    window = list(traj.steps[start:start + length])
    if not window:
        raise ValueError("window requires at least one step")
    anchor = window[-1]
    hist = _compact_history(traj.steps, start, 5)
    hist.extend(s.action_text for s in window[:-1] if s.action_text)
    return SampleSpec(
        source=traj.source, trajectory_id=traj.trajectory_id,
        step_ids=[s.step_id for s in window], representation=WINDOW, task=traj.task,
        current_step=anchor, extra_images=[], history_texts=hist,
        assistant_turns=[anchor.action_text],
        # Quality/reasoning labels describe the supervised anchor, not an
        # unrelated earlier action that merely appears in text history.
        signals=set(anchor.signals),
        app=traj.app, metadata=dict(traj.metadata),
    )


def build_chunk(traj: Trajectory, start: int, length: int) -> SampleSpec:
    """Long-horizon text history + one visually grounded next action.

    Chunks preserve 8-12-step action context without fabricating intermediate
    visual turns.  Only the final action is a training target and it always
    uses its own real pre-action screenshot.
    """
    window = list(traj.steps[start:start + length])
    if not window:
        raise ValueError("chunk requires at least one step")
    anchor = window[-1]
    hist = _compact_history(traj.steps, start, 12)
    hist.extend(s.action_text for s in window[:-1] if s.action_text)
    return SampleSpec(
        source=traj.source, trajectory_id=traj.trajectory_id,
        step_ids=[s.step_id for s in window], representation=CHUNK, task=traj.task,
        current_step=anchor, extra_images=[], history_texts=hist,
        assistant_turns=[anchor.action_text],
        # Quality/reasoning labels describe the supervised anchor, not an
        # unrelated earlier action that merely appears in text history.
        signals=set(anchor.signals),
        app=traj.app, metadata=dict(traj.metadata),
    )


def suggest_window_starts(traj: Trajectory, target: int, count: int = 1) -> List[int]:
    """Deterministic spread of window start positions across the trajectory."""
    n = traj.length
    if n < target:
        return []
    span = n - target
    starts = []
    for k in range(count):
        u = seeded_bucket(f"win:{traj.trajectory_id}:{k}")
        starts.append(min(span, int(u * (span + 1))))
    return starts


# ---------------------- window / chunk quality gates ------------------------

def window_is_informative(steps: Sequence[Step], threshold: int = 6) -> bool:
    """A window must add temporal information: at least two state-changing
    steps and at least two distinct actions. Windows where every frame is a
    near-duplicate or the same action repeats are noise, not supervision."""
    if len(steps) < 2 or not _same_continuity(steps):
        return False
    state_changes = sum(1 for s in steps
                        if frames_add_information(s.prev_phash, s.phash, threshold))
    unique_actions = len({s.action_text for s in steps})
    return state_changes >= 2 and unique_actions >= 2


def chunk_is_informative(steps: Sequence[Step], threshold: int = 6) -> bool:
    """Chunks are expensive: require genuine long-horizon content — majority
    state-changing steps, >=4 distinct actions, and not dominated by waits."""
    if len(steps) < CHUNK_MIN or not _same_continuity(steps):
        return False
    state_changes = sum(1 for s in steps
                        if frames_add_information(s.prev_phash, s.phash, threshold))
    if state_changes < 0.5 * len(steps):
        return False
    if len({s.action_text for s in steps}) < 4:
        return False
    waits = sum(1 for s in steps if s.action is not None and s.action.verb == "wait")
    if waits > 0.3 * len(steps):
        return False
    return True
