"""Multi-dimensional sample quality model (Run 1 data-quality pass).

Every computer-use sample gets component scores in [0,1] kept verbatim in
metadata["quality"], an interpretable composite 0..10, a quality bucket
(A high-value hard / B useful normal / C diversity / Reject), and a token
efficiency score (quality per 1k estimated tokens) used for sampling
priorities. Grounded in the 2026-08 audit findings:

  - MOVE_TO / lone key_down / mouse_down are NOT anchor-eligible (VideoCUA
    audit: 45% MOVE_TO spam; ProCUA audit: 15% key_down)
  - recovery / verification / finish-before-verification / repeated-wait
    rules come from the finish & wait audits
  - grounding difficulty ~ inverse target width, capped, with bucket quotas
  - triviality = first trivial click / giant target / no-state-change without
    recovery follow-up
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# actions that may anchor a training sample; others appear only in history
ANCHOR_ELIGIBLE_VERBS = {
    "click", "double_click", "right_click", "middle_click", "drag", "scroll",
    "type", "press", "hotkey", "finish", "wait",
}
HISTORY_ONLY_VERBS = {"move", "key_down", "key_up", "mouse_down", "mouse_up", "point"}

# A near-identical *next* screenshot is evidence that an action may have been
# ineffective only for actions expected to change visible state.  Cursor moves,
# key/mouse down-up micro-actions and waits must never be labelled recovery just
# because the pixels are unchanged.
VISUAL_EFFECT_EXPECTED_VERBS = {
    "click", "double_click", "right_click", "middle_click", "drag",
    "type", "press",
}

def visual_effect_expected(action) -> bool:
    return action is not None and getattr(action, "verb", "") in VISUAL_EFFECT_EXPECTED_VERBS


DIFFICULTY_PATTERNS = [
    (re.compile(r"\bsort(ed|ing)?\b|\barrange\b|\border\b", re.I), "sorting"),
    (re.compile(r"\brank(ed|ing)?\b|\btop\s*\d+\b|\bhighest\b|\blowest\b", re.I), "ranking"),
    (re.compile(r"\bexactly\s+\d+\b|\ball\s+\d+\b|\b\d+\s+(items?|files?|words?|cells?|rows?|slides?|students?)\b", re.I), "exact_quantity"),
    (re.compile(r"\band\b[^.]{2,40}\balso\b|,\s*(then|and)\s+(open|create|rename|delete|move|save)\b", re.I), "multi_target"),
    (re.compile(r"\bsave(d)?( as)?\b|\bexport\b|\bdownload\b|\bprint\b", re.I), "save_export"),
    (re.compile(r"dialog|modal|popup|confirm|overwrite", re.I), "dialog"),
    (re.compile(r"\bformat\b|\bfont\b|\bstyle\b|\btheme\b|\blayout\b", re.I), "formatting"),
    (re.compile(r"\benable\b|\bdisable\b|\btoggle\b|\bsettings?\b|\bpreferences\b", re.I), "settings"),
    (re.compile(r"verif|check|confirm|ensure|make sure", re.I), "verification"),
]

RARE_VERBS = {"drag": 1.0, "hotkey": 0.8, "double_click": 0.5, "right_click": 0.6,
              "middle_click": 1.0, "scroll": 0.4, "type": 0.3, "press": 0.4}
COMMON_APPS = {"excel", "word", "ppt", "powerpoint", "chrome", "chromium", "firefox",
               "edge", "explorer", "files", "desktop", "libreoffice", "unknown"}

# quality bucket thresholds on the 0..10 composite
BUCKET_A_MIN = 7.0
BUCKET_B_MIN = 5.0
BUCKET_C_MIN = 3.0

# GroundCUA target-size buckets: name -> (min_px, max_px, quota_share)
GROUNDCUA_SIZE_BUCKETS = [
    ("tiny", 0, 16, 0.25),
    ("small", 16, 32, 0.40),
    ("medium", 32, 64, 0.25),
    ("large", 64, 10**9, 0.10),
]

# GUI-360 understanding: answers average 27k chars (422 controls) -> unusable;
# cap to a trainable, honest number of controls
UNDERSTANDING_MAX_CONTROLS = 24

_BASE_ACTION_VALUE = 2.0
_BASE_GROUNDING_VALUE = 1.2

_WEIGHTS = {
    "difficulty": 1.6, "recovery": 2.2, "verification": 1.8, "action_rarity": 1.0,
    "app_rarity": 0.8, "dialog": 0.8, "save_export": 1.0, "state_value": 0.8,
    "triviality": -1.8, "redundancy": -1.2, "sequence_value": 0.6,
    "offcenter_click": -0.7,
}


def detect_difficulty_signals(task_text: str, subgoal: str = "") -> Set[str]:
    signals: Set[str] = set()
    for pat, name in DIFFICULTY_PATTERNS:
        if pat.search(task_text or "") or pat.search(subgoal or ""):
            signals.add(name)
    return signals


def app_rarity(app: str, app_counter: Dict[str, int], total: int) -> float:
    """Rarer apps (share of samples so far) score higher."""
    if not app or total == 0:
        return 0.3
    share = app_counter.get(app.lower(), 0) / max(1, total)
    if app.lower() in COMMON_APPS:
        return max(0.0, 0.5 - share)
    return max(0.3, 1.0 - 3 * share)


@dataclass
class QualityScore:
    components: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0            # 0..10
    bucket: str = "C"             # A | B | C | Reject
    reject_reason: Optional[str] = None
    token_efficiency: float = 0.0  # quality per 1k tokens

    def as_meta(self) -> Dict:
        return {"components": {k: round(v, 3) for k, v in self.components.items()},
                "score": round(self.score, 2), "bucket": self.bucket,
                "reject_reason": self.reject_reason}


def score_action_step(*, verb: str, task_text: str, subgoal: str = "",
                      signals: Iterable[str] = (),
                      app: str = "", app_counter: Optional[Dict[str, int]] = None,
                      total_samples: int = 0,
                      is_first_step: bool = False,
                      repeated_identical: bool = False,
                      state_change_from_prev: bool = True,
                      prev_was_ineffective: bool = False,
                      target_width_px: Optional[int] = None,
                      bbox_center_offset_norm: Optional[float] = None,
                      representation: str = "single") -> QualityScore:
    """Score one candidate anchor step. Returns QualityScore with bucket."""
    comp: Dict[str, float] = {}
    sig = set(signals) | detect_difficulty_signals(task_text, subgoal)

    # difficulty from task text
    comp["difficulty"] = min(1.0, 0.15 * len(sig & {"sorting", "ranking", "exact_quantity",
                                                   "multi_target", "formatting", "settings"}))
    # recovery: previous action ineffective and this step responds to it
    # Pixel similarity alone is not proof of failure: copy shortcuts, focus
    # changes and many legitimate actions can leave the frame nearly unchanged.
    # Reward recovery only when the source supplies explicit causal evidence.
    comp["recovery"] = 1.0 if (prev_was_ineffective or
                               "recovery_evidenced" in sig) else 0.0
    # verification / finish discipline
    comp["verification"] = 1.0 if (verb == "finish" or "verification" in sig) else \
        (0.4 if "save_export" in sig else 0.0)
    # action rarity
    comp["action_rarity"] = RARE_VERBS.get(verb, 0.15)
    # application rarity
    comp["app_rarity"] = app_rarity(app, app_counter or {}, total_samples)
    # dialog / save-export value
    comp["dialog"] = 1.0 if "dialog" in sig or "modal_dialog" in sig else 0.0
    comp["save_export"] = 1.0 if "save_export" in sig or "export" in sig else 0.0
    # state reasoning: step follows a real state change (interpretable screen)
    comp["state_value"] = 0.6 if state_change_from_prev else 0.3
    # small-target grounding difficulty for click-like verbs
    if target_width_px is not None and verb in ("click", "double_click", "right_click", "drag"):
        comp["grounding_difficulty"] = max(0.0, min(1.0, 48.0 / max(6, target_width_px) - 0.4))
    else:
        comp["grounding_difficulty"] = 0.0
    # A click inside the trusted bbox is correct, but for small discrete
    # controls a very off-centre target is less robust supervision. This is a
    # soft down-rank only; bbox-outside clicks are already hard-rejected.
    if bbox_center_offset_norm is not None and verb in ("click", "double_click", "right_click", "middle_click"):
        off = float(bbox_center_offset_norm)
        comp["offcenter_click"] = max(0.0, min(1.0, (off - 0.35) / 0.45))
    else:
        comp["offcenter_click"] = 0.0
    # sequence value: windows/chunks teach sequential behavior
    comp["sequence_value"] = {"single": 0.3, "window": 0.8, "chunk": 1.0}.get(representation, 0.3)

    # ---- penalties
    trivial = 0.0
    if is_first_step and verb in ("click", "double_click") and not sig:
        trivial = 0.5          # routine task-opening click
    if verb == "click" and target_width_px is not None and target_width_px > 300:
        trivial = max(trivial, 0.8)   # giant button
    comp["triviality"] = trivial
    comp["redundancy"] = 1.0 if repeated_identical else 0.0

    raw = sum(_WEIGHTS.get(k, 0.0) * v for k, v in comp.items())
    # BASE_VALUE keeps ordinary-but-useful steps out of Reject: a plain click
    # on a real state change is legitimate supervision (B/C), while recovery,
    # verification, rare actions and hard grounding lift samples to A.
    raw = _BASE_ACTION_VALUE + raw
    score = max(0.0, min(10.0, 10.0 * raw / 8.0))  # normalize to interpretable 0..10

    qs = QualityScore(components=comp, score=score)
    # hard rejects
    if verb not in ANCHOR_ELIGIBLE_VERBS:
        qs.bucket, qs.reject_reason = "Reject", f"anchor_ineligible_verb:{verb}"
        return qs
    if repeated_identical and comp["recovery"] == 0.0:
        qs.bucket, qs.reject_reason = "Reject", "repeated_identical_no_recovery"
        return qs
    # consecutive repeated waits are loop-bait
    qs.bucket = ("A" if score >= BUCKET_A_MIN else
                 "B" if score >= BUCKET_B_MIN else
                 "C" if score >= BUCKET_C_MIN else "Reject")
    if qs.bucket == "Reject":
        if comp["triviality"] > 0 and comp["redundancy"] == 0.0 and verb == "click":
            # routine task-initiation click: diversity value, not corruption
            qs.bucket, qs.reject_reason = "C", None
        else:
            qs.reject_reason = "low_quality"
    return qs


def score_grounding(*, target_width_px: int, target_height_px: int,
                    text: str, category: Optional[str],
                    app: str, app_counter: Optional[Dict[str, int]] = None,
                    total_samples: int = 0) -> QualityScore:
    """Score a grounding example (point-to-element)."""
    comp: Dict[str, float] = {}
    # Geometric mean captures both area and thinness: a 200x5 target remains
    # hard, while an ordinary 140x32 menu entry is not mislabeled as tiny.
    extent = max(1.0, math.sqrt(max(1, target_width_px) * max(1, target_height_px)))
    comp["grounding_difficulty"] = max(0.0, min(1.0, 40.0 / extent - 0.3))
    words = [t for t in re.split(r"\W+", text or "") if t]
    comp["text_quality"] = min(1.0, 0.25 * (len(words) - 1)) if len(words) > 1 else 0.0
    comp["app_rarity"] = app_rarity(app, app_counter or {}, total_samples)
    comp["category_value"] = 1.0 if category in ("Button", "Menu", "Input Elements") else 0.5
    raw = (2.5 * comp["grounding_difficulty"] + 1.0 * comp["text_quality"] +
           0.8 * comp["app_rarity"] + 0.7 * comp["category_value"])
    raw = _BASE_GROUNDING_VALUE + raw  # ordinary targets are C/B, tiny+labelled are A
    score = max(0.0, min(10.0, 10.0 * raw / 5.5))
    qs = QualityScore(components=comp, score=score)
    qs.bucket = ("A" if score >= BUCKET_A_MIN else
                 "B" if score >= BUCKET_B_MIN else
                 "C" if score >= BUCKET_C_MIN else "Reject")
    if qs.bucket == "Reject":
        qs.reject_reason = "low_quality_grounding"
    return qs


def grounding_bucket(target_width_px: int, target_height_px: Optional[int] = None) -> str:
    """Bucket by the smaller target dimension so thin controls stay difficult."""
    extent = min(target_width_px, target_height_px) if target_height_px is not None else target_width_px
    for name, lo, hi, _ in GROUNDCUA_SIZE_BUCKETS:
        if lo <= extent < hi:
            return name
    return GROUNDCUA_SIZE_BUCKETS[-1][0]


class BucketQuota:
    """Enforces the tiny/small/medium/large mixture for grounding sources."""

    def __init__(self, quotas: Optional[Dict[str, float]] = None):
        self.shares = quotas or {name: share for name, _, _, share in GROUNDCUA_SIZE_BUCKETS}
        self.counts: Dict[str, int] = {k: 0 for k in self.shares}

    def allow(self, bucket: str) -> bool:
        total = sum(self.counts.values())
        if total == 0:
            return True
        return self.counts.get(bucket, 0) < self.shares.get(bucket, 0.0) * total * 1.25 + 2

    def record(self, bucket: str):
        self.counts[bucket] = self.counts.get(bucket, 0) + 1


def token_efficiency(score: float, estimated_tokens: int) -> float:
    """Quality per 1,000 estimated tokens (training value per GPU second)."""
    return score / max(1.0, estimated_tokens / 1000.0)


def attach_quality(sample: dict, qs: QualityScore, estimated_tokens: int) -> dict:
    """Attach quality metadata to a final sample dict."""
    qs.token_efficiency = token_efficiency(qs.score, estimated_tokens)
    meta = sample.setdefault("metadata", {})
    meta["quality"] = qs.as_meta()
    meta["quality"]["token_efficiency"] = round(qs.token_efficiency, 3)
    return sample


# ---------------------- wait / finish / recovery audit gates --------------

def consecutive_wait_count(prev_actions: Sequence[str]) -> int:
    n = 0
    for a in reversed(prev_actions):
        if a.startswith("wait"):
            n += 1
        else:
            break
    return n


def wait_sample_allowed(prev_actions: Sequence[str]) -> bool:
    """Repeated waits become loops: keep at most one wait in a row."""
    return consecutive_wait_count(prev_actions) == 0


def finish_has_evidence(*, task_text: str = "", prev_state_changed: Optional[bool] = None,
                        explicit_success: bool = False,
                        reliable_final_state: bool = False,
                        verifier_evidence: bool = False,
                        human_source: bool = False) -> bool:
    """Return True only when completion has direct evidence.

    ``human_source`` is retained for API compatibility but intentionally does
    not grant trust. A demonstration being human-authored is not evidence that
    the supervised finish point is actually complete.
    """
    del task_text, prev_state_changed, human_source
    return bool(explicit_success or reliable_final_state or verifier_evidence)

