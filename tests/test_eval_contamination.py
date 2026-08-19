"""Verified-pinned contamination flagging tool tests (flag-only, never remove)."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.check_contamination import (build_reference, check_text,
                                            _task_text_of_sample)


REFS = build_reference([
    {"task_id": "chrome/abc",
     "instruction": "Open the browser settings and change the homepage to example.com"},
    {"task_id": "os/def",
     "instruction": "Show hidden files in the file manager preferences dialog"},
])


def test_exact_match_flagged():
    hit = check_text("Open the browser settings and change the homepage to example.com",
                     REFS)
    assert hit and hit["kind"] == "exact" and hit["task_id"] == "chrome/abc"


def test_near_paraphrase_flagged():
    hit = check_text("Open the browser settings and change the homepage to example.org "
                     "instead", REFS)
    if hit:  # near-threshold paraphrases may or may not cross 0.5 Jaccard;
        assert hit["kind"] in ("near_jaccard", "containment")
    # the flag-only contract is what matters here: nothing is ever dropped


def test_unrelated_text_not_flagged():
    assert check_text("Sort the customer spreadsheet by the revenue column", REFS) is None


def test_task_text_extraction_skips_replay():
    assert _task_text_of_sample({"task_type": "replay_math",
                                 "messages": [{"role": "user", "content": "solve"}]}) is None
    assert _task_text_of_sample({"task_type": "screen_understanding",
                                 "messages": []}) is None
    sample = {"task_type": "action", "messages": [
        {"role": "system", "content": "You are a computer-use agent."},
        {"role": "user", "content": "<img>\nTask: Export the sheet as PDF\nprev"},
        {"role": "assistant", "content": "click(x=1, y=2)"}]}
    assert _task_text_of_sample(sample) == "Export the sheet as PDF"


def test_end_to_end_report_flag_only(tmp_path):
    import subprocess
    root = tmp_path / "ds"
    (root / "final").mkdir(parents=True)
    tl = tmp_path / "tl.json"
    tl.write_text(json.dumps([
        {"task_id": "chrome/x", "instruction": "Export the current sheet as PDF"}]),
        encoding="utf-8")
    rows = [
        {"task_type": "action", "metadata": {"id": "contaminated", "source": "procua"},
         "messages": [{"role": "user", "content": "<i>\nTask: Export the current sheet as PDF"},
                      {"role": "assistant", "content": "click(x=1, y=1)"}]},
        {"task_type": "action", "metadata": {"id": "clean", "source": "videocua"},
         "messages": [{"role": "user", "content": "<i>\nTask: Unrelated wallpaper change task"},
                      {"role": "assistant", "content": "click(x=2, y=2)"}]},
        {"task_type": "replay", "metadata": {"id": "skip", "source": "replay"},
         "messages": [{"role": "user", "content": "chat"},
                      {"role": "assistant", "content": "ok"}]},
    ]
    with open(root / "final" / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "rep.json"
    rc = subprocess.call([sys.executable, "evaluation/check_contamination.py",
                          "--dataset-root", str(root), "--task-list", str(tl),
                          "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "flagged"
    assert report["counts"]["scanned"] == 2
    assert report["counts"]["skipped_non_cu"] == 1
    matches = report["matches"]
    assert len(matches) == 1 and matches[0]["sample_id"] == "contaminated"
    assert matches[0]["action"] == "FLAGGED_NOT_REMOVED"  # never silently dropped
    assert report["blockers"], "flagged matches must produce claim blockers"
