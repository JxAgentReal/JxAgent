"""Content-gated concise reasoning (spec section 14).

- NO random 25% attachment; reasoning is gated on detectable content signals.
- Run 1 default is 0% synthetic reasoning after the manual factuality audit.
  A future run may opt in only after an independent >=99.5% factuality gate.
- If enabled in a future run, only explicit evidence-bearing signals may
  trigger the recovery/verification/planning/wait categories.
- Maximum 2 complete sentences / ~220 chars; no generic narration, no task
  repetition, no verbose chain of thought.
- Format: "Plan: <sentence(s)>\nAction: <action>"
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Sequence, Set

MAX_CHARS = 220
MAX_SENTENCES = 2
TARGET_RATE = 0.0  # Run 1 hardening: synthetic reasoning disabled unless independently evidence-gated
CATEGORY_DISTRIBUTION = {"recovery": 0.40, "verification": 0.25, "planning": 0.20, "wait": 0.15}

BANNED_PATTERNS = [
    r"looking at the (screenshot|screen|image), i",
    r"i can see",
    r"as (an|a) ai",
    r"first, ",
    r"let me",
]

# signal -> reasoning category
SIGNAL_CATEGORY = {
    # Only explicit evidence-bearing signals may trigger a plan.  Broad task
    # keywords (save/export/back) and near-identical screenshots after MOVE_TO
    # are not sufficient causal evidence.
    "recovery_evidenced": "recovery",
    "verification_evidenced": "verification",
    "planning_evidenced": "planning",
    "wait_evidenced": "wait",
}

TEMPLATES: Dict[str, List[str]] = {
    "recovery": [
        "Plan: The previous action produced no visible change, so a different control is used instead.",
        "Plan: The earlier attempt did not reach the target state; this action takes the alternate path.",
        "Plan: The screen is unchanged after the last step, indicating the control was ineffective; switching approach.",
    ],
    "verification": [
        "Plan: The expected result is now visible, so the work is confirmed before finishing.",
        "Plan: This check confirms the saved output matches the requested name and location.",
        "Plan: The dialog closed and the requested state is present, which verifies completion.",
    ],
    "planning": [
        "Plan: This intermediate step is required before the requested final operation can succeed.",
        "Plan: The task spans two applications, so the content is prepared here before switching.",
        "Plan: Among the visible controls, this one matches the required operation for the next stage.",
    ],
    "wait": [
        "Plan: The interface is still loading, so the correct move is to wait rather than click.",
        "Plan: A progress indicator is active; acting now would target a moving control.",
        "Plan: The operation needs time to finish before its result can be verified.",
    ],
}


def detect_category(signals: Sequence[str]) -> Optional[str]:
    for s in signals:
        if s in SIGNAL_CATEGORY:
            return SIGNAL_CATEGORY[s]
    return None


def _sentences_ok(text: str) -> bool:
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    return 0 < len(parts) <= MAX_SENTENCES


def make_reasoning(signals: Sequence[str], trajectory_id: str = "", step_id: str = "") -> Optional[str]:
    """Deterministic concise reasoning for a step, or None when no content
    signal justifies reasoning."""
    category = detect_category(signals)
    if category is None:
        return None
    templates = TEMPLATES[category]
    h = int(hashlib.md5(f"{trajectory_id}:{step_id}:{category}".encode()).hexdigest()[:8], 16)
    text = templates[h % len(templates)]
    if len(text) > MAX_CHARS or not _sentences_ok(text):
        return None
    if any(re.search(p, text.lower()) for p in BANNED_PATTERNS):
        return None
    if not text.strip().endswith((".", "!", "?")):
        return None
    return text


def compose_assistant_target(action_text: str, plan: Optional[str]) -> str:
    if plan:
        plan = plan.strip()
        if plan.startswith("Plan:"):  # templates already carry the prefix
            plan = plan[len("Plan:"):].strip()
        return f"Plan: {plan}\nAction: {action_text}"
    return action_text


class ReasoningGate:
    """Content-gated reasoning with a SELF-REGULATING overall rate.

    Run 1 is deliberately disabled (TARGET_RATE=0).  The self-regulating
    machinery remains for a future evidence-verified run, where it can track
    the requested rate without forcing synthetic plans into the current data.
    """

    def __init__(self, rate: float = TARGET_RATE,
                 distribution: Dict[str, float] = None,
                 seed: str = "jxagent"):
        self.rate = rate
        self.distribution = distribution or dict(CATEGORY_DISTRIBUTION)
        self.counts = {k: 0 for k in self.distribution}
        self.total = 0          # reasoning attachments
        self.action_samples = 0 # denominator: every accepted action sample
        self.eligible = 0
        self.eligible_cats = {k: 0 for k in self.distribution}
        self._seed = seed

    def quota_left(self, category: str) -> bool:
        # Shares renormalized over categories actually seen in the eligible
        # pool (e.g. planning absent -> its budget flows to the others), plus
        # +2 slack so small counts cannot deadlock at rounded-down quotas.
        if self.total == 0:
            return True
        seen = [c for c, n in self.eligible_cats.items() if n > 0]
        denom = sum(self.distribution[c] for c in seen) or 1.0
        target_share = self.distribution[category] / denom
        return self.counts[category] <= target_share * self.total + 2.0

    def note_action_sample(self, n: int = 1):
        """Call once for every action sample actually emitted."""
        self.action_samples += n

    def allow(self, signals: Sequence[str], trajectory_id: str, step_id: str) -> Optional[str]:
        """Return evidence-backed concise reasoning or ``None``.

        Run 1 defaults to rate=0 because the prior canned templates failed a
        manual factuality audit.  A future run may opt in only after the
        external >=99.5% factuality gate is demonstrated.
        """
        if self.rate <= 0:
            return None
        category = detect_category(signals)
        if category is None:
            return None
        self.eligible += 1
        if category in self.eligible_cats:
            self.eligible_cats[category] += 1
        realized = self.total / max(1, self.action_samples)
        key = f"{self._seed}:{trajectory_id}:{step_id}"
        u = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        # deterministic jitter around the target keeps selection stable across
        # resumes while the self-regulating ratio converges on `rate`
        if realized > self.rate or (realized > self.rate * 0.8 and u > 0.5):
            return None
        if not self.quota_left(category):
            return None
        text = make_reasoning(signals, trajectory_id, step_id)
        if text is None:
            return None
        self.counts[category] += 1
        self.total += 1
        return text

    def stats(self) -> Dict[str, object]:
        return {"counts": dict(self.counts), "total": self.total,
                "eligible": self.eligible,
                "action_samples": self.action_samples,
                "realized_rate": round(self.total / max(1, self.action_samples), 4)}
