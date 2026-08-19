"""Multimodal token budget estimation and history shortening (spec section 13).

Qwen-VL family vision token approximation: patches of 14px merged 2x2 ->
effective 28 px per token; tokens ~= ceil(w/28) * ceil(h/28) plus a small
per-image overhead. Text approximated at 4 characters per token. These are
estimates used for filtering; the trainer pads/truncates nothing implicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_CONTEXT_BUDGET = 8192
IMAGE_PATCH = 28          # effective merged patch size in pixels
IMAGE_OVERHEAD_TOKENS = 8
CHARS_PER_TOKEN = 4.0
MESSAGE_OVERHEAD = 4


def estimate_text_tokens(text: str) -> int:
    return int(math.ceil(len(text) / CHARS_PER_TOKEN)) if text else 0


def estimate_image_tokens(width: int, height: int) -> int:
    return int(math.ceil(width / IMAGE_PATCH) * math.ceil(height / IMAGE_PATCH)) + IMAGE_OVERHEAD_TOKENS


@dataclass
class HistoryItem:
    text: str
    image_tokens: int = 0   # tokens if this history item keeps its image
    image: Optional[object] = None
    keep_priority: int = 0  # higher = more important to keep intact


@dataclass
class BudgetReport:
    estimated_tokens: int = 0
    image_count: int = 0
    history_steps: int = 0
    window_length: int = 0
    fits: bool = True
    shortened: bool = False


def estimate_sequence(user_text: str, assistant_target: str,
                      image_sizes: Sequence[Tuple[int, int]],
                      system_text: str = "",
                      history_texts: Sequence[str] = ()) -> int:
    tokens = MESSAGE_OVERHEAD * (3 + len(history_texts))
    tokens += estimate_text_tokens(system_text)
    tokens += estimate_text_tokens(user_text)
    tokens += estimate_text_tokens(assistant_target)
    for t in history_texts:
        tokens += estimate_text_tokens(t)
    tokens += sum(estimate_image_tokens(w, h) for w, h in image_sizes)
    return tokens


def fit_to_budget(*, task_text: str, history: List[HistoryItem], current_image_tokens: int,
                  assistant_target: str, system_text: str = "",
                  budget: int = DEFAULT_CONTEXT_BUDGET) -> Tuple[List[HistoryItem], BudgetReport]:
    """Shorten history first (drop images, then compress text) so the sample
    fits. Never truncates the assistant target. Returns kept history items."""
    base = estimate_text_tokens(system_text) + estimate_text_tokens(task_text) \
        + estimate_text_tokens(assistant_target) + current_image_tokens + 32
    if base > budget:
        return [], BudgetReport(base, 1, 0, 0, fits=False, shortened=False)

    kept = list(history)
    changed = False

    def total(kept_items: List[HistoryItem]) -> int:
        return base + sum(estimate_text_tokens(h.text) + h.image_tokens for h in kept_items)

    # 1) drop old screenshots first (lowest keep_priority images first)
    while total(kept) > budget:
        image_idxs = [i for i, h in enumerate(kept) if h.image_tokens > 0]
        if not image_idxs:
            break
        i = min(image_idxs, key=lambda j: kept[j].keep_priority)
        kept[i] = HistoryItem(text=kept[i].text, image_tokens=0, keep_priority=kept[i].keep_priority)
        changed = True

    # 2) drop whole history items from the oldest side
    while total(kept) > budget and kept:
        kept.pop(0)
        changed = True

    report = BudgetReport(
        estimated_tokens=total(kept),
        image_count=1 + sum(1 for h in kept if h.image_tokens > 0),
        history_steps=len(kept),
        window_length=len(history),
        fits=total(kept) <= budget,
        shortened=changed,
    )
    return kept, report

# ---------------------------------------------------------------------------
# Assistant-loss accounting

def sample_token_components(sample: dict) -> Dict[str, int]:
    """Deterministic prebuild estimate of tokens that contribute to SFT loss.

    Assistant messages with ``loss=False`` are excluded. System/user/tool text
    is counted as input text. Image tokens use the same conservative estimator
    as the context-budget gate.
    """
    input_text = 0
    assistant_loss = 0
    for m in sample.get("messages", []) or []:
        content = str(m.get("content") or "")
        if m.get("role") == "assistant" and m.get("loss", True) is not False:
            assistant_loss += estimate_text_tokens(content)
        else:
            input_text += estimate_text_tokens(content)
    image_tokens = 0
    meta = sample.get("metadata", {}) or {}
    # Most JxAgent CU samples have one final image size. Replay can have multiple
    # images, where exact dimensions are unavailable in metadata. Preserve a
    # conservative estimate from estimated total only in that case.
    fs = meta.get("final_image_size")
    if isinstance(fs, (list, tuple)) and len(fs) >= 2 and sample.get("images"):
        try:
            one = estimate_image_tokens(int(fs[0]), int(fs[1]))
            image_tokens = one * len(sample.get("images") or [])
        except Exception:
            image_tokens = 0
    return {"input_text_tokens": input_text, "image_tokens": image_tokens,
            "assistant_loss_tokens": assistant_loss}


def estimate_loss_token_report(samples: Sequence[dict]) -> Dict:
    """Aggregate estimated assistant loss by source and task type."""
    from collections import defaultdict
    groups = {"source": defaultdict(lambda: {"samples": 0, "input_text_tokens": 0,
                                               "image_tokens": 0, "assistant_loss_tokens": 0}),
              "task_type": defaultdict(lambda: {"samples": 0, "input_text_tokens": 0,
                                                  "image_tokens": 0, "assistant_loss_tokens": 0})}
    total = {"samples": 0, "input_text_tokens": 0, "image_tokens": 0,
             "assistant_loss_tokens": 0}
    for s in samples:
        c = sample_token_components(s)
        total["samples"] += 1
        for k in ("input_text_tokens", "image_tokens", "assistant_loss_tokens"):
            total[k] += c[k]
        for axis, key in (("source", str(s.get("source") or "unknown")),
                          ("task_type", str(s.get("task_type") or "unknown"))):
            g = groups[axis][key]
            g["samples"] += 1
            for k in ("input_text_tokens", "image_tokens", "assistant_loss_tokens"):
                g[k] += c[k]
    denom = max(1, total["assistant_loss_tokens"])
    out_groups = {}
    for axis, vals in groups.items():
        out_groups[axis] = {}
        for key, row in sorted(vals.items()):
            row = dict(row)
            row["assistant_loss_share"] = round(row["assistant_loss_tokens"] / denom, 6)
            out_groups[axis][key] = row
    return {"method": "estimated_chars_and_image_patches", "total": total,
            **out_groups}
