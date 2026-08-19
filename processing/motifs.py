"""Task motif detection and coverage accounting for JxAgent second-stage hardening.

Motifs are deterministic metadata derived only from observable task text,
action text, source metadata and verified signals. They are used for selection
coverage and release reporting, never to synthesize reasoning.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Set

_PATTERNS = {
    "dialog_modal": re.compile(r"\b(dialog|modal|popup|pop-up|confirmation|confirm)\b", re.I),
    "save_export": re.compile(r"\b(save(?: as)?|export|download|print)\b", re.I),
    "file_chooser": re.compile(r"\b(file chooser|open file|choose file|select file|browse(?: for)? file|save as)\b", re.I),
    "overwrite_confirmation": re.compile(r"\b(overwrite|replace (?:the )?(?:existing )?file|already exists)\b", re.I),
    "exact_quantity": re.compile(r"\bexactly\s+\d+\b|\b\d+\s+(?:items?|files?|rows?|cells?|slides?|words?|entries?)\b", re.I),
    "sort_rank": re.compile(r"\b(sort|sorted|sorting|rank|ranking|highest|lowest|top\s*\d+)\b", re.I),
    "multi_target": re.compile(r"\b(?:and then|then also|both|all of the following|multiple)\b", re.I),
    "settings": re.compile(r"\b(settings?|preferences?|options?|configure|configuration|enable|disable|toggle)\b", re.I),
    "cross_app": re.compile(r"\b(?:from|between)\s+(?:excel|word|powerpoint|chrome|firefox|edge|explorer|libreoffice)[^.!?]{0,100}\b(?:to|and)\s+(?:excel|word|powerpoint|chrome|firefox|edge|explorer|libreoffice)\b", re.I),
}

_SIGNAL_TO_MOTIF = {
    "modal_dialog": "dialog_modal",
    "dialog": "dialog_modal",
    "save_export": "save_export",
    "export": "save_export",
    "drag": "drag",
    "keyboard_shortcut": "hotkeys",
    "recovery_evidenced": "verified_recovery",
    "finish_verification": "finish_verification",
}


def detect_motifs(sample: dict) -> List[str]:
    meta = sample.get("metadata", {}) or {}
    parts: List[str] = []
    for m in sample.get("messages", []) or []:
        if m.get("role") in {"user", "assistant"}:
            parts.append(str(m.get("content") or ""))
    text = "\n".join(parts)
    motifs: Set[str] = set()
    for name, pat in _PATTERNS.items():
        if pat.search(text):
            motifs.add(name)
    signals = set(meta.get("signals") or [])
    for sig, motif in _SIGNAL_TO_MOTIF.items():
        if sig in signals:
            motifs.add(motif)
    action_text = ""
    for m in reversed(sample.get("messages", []) or []):
        if m.get("role") == "assistant":
            action_text = str(m.get("content") or "").lower()
            break
    if "drag(" in action_text:
        motifs.add("drag")
    if "hotkey(" in action_text or "press(" in action_text:
        motifs.add("hotkeys")
    if "scroll(" in action_text:
        motifs.add("scroll_read")
    if meta.get("finish_evidence") == "yes":
        motifs.add("finish_verification")
    if meta.get("recovery_verified") is True:
        motifs.add("verified_recovery")
    if meta.get("cross_app") is True:
        motifs.add("cross_app")
    if meta.get("file_chooser") is True:
        motifs.add("file_chooser")
    return sorted(motifs)


def attach_motifs(sample: dict) -> dict:
    sample.setdefault("metadata", {})["motifs"] = detect_motifs(sample)
    return sample


def coverage(samples: Iterable[dict]) -> Dict[str, int]:
    c: Counter = Counter()
    for sample in samples:
        motifs = (sample.get("metadata", {}) or {}).get("motifs")
        if motifs is None:
            motifs = detect_motifs(sample)
        c.update(set(motifs))
    return dict(sorted(c.items()))
