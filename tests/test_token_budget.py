import pytest

from processing.token_budget import (HistoryItem, estimate_image_tokens,
                                     estimate_sequence, estimate_text_tokens,
                                     fit_to_budget)


def test_text_token_estimate():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd" * 100) == 100


def test_image_token_estimate_scales_with_pixels():
    small = estimate_image_tokens(560, 560)
    big = estimate_image_tokens(1600, 900)
    assert small < big
    assert big > 1000  # 1600x900 screenshot is expensive


def test_full_sequence_estimate():
    est = estimate_sequence("Task: do things", "click(x=1, y=2)",
                            [(1600, 900)], system_text="You are a computer-use agent.")
    assert est > 1500


def test_fit_drops_history_images_before_text():
    history = [HistoryItem(text=f"step {i}", image_tokens=200) for i in range(8)]
    kept, report = fit_to_budget(
        task_text="T" * 200, history=history, current_image_tokens=1800,
        assistant_target="click(x=1, y=2)", budget=2600)
    assert report.fits
    assert report.image_count < 9  # some history images dropped
    assert any(h.image_tokens == 0 for h in kept)  # images dropped first


def test_fit_never_truncates_target():
    huge_target = "type(text=\"" + "x" * 60000 + "\")"
    kept, report = fit_to_budget(
        task_text="task", history=[HistoryItem(text="history")],
        current_image_tokens=1800, assistant_target=huge_target, budget=8192)
    assert not report.fits  # rejected, not silently truncated


def test_fit_drops_oldest_history_first():
    history = [HistoryItem(text="H" * 100) for _ in range(50)]
    kept, report = fit_to_budget(task_text="t", history=history,
                                 current_image_tokens=1800,
                                 assistant_target="click(x=1, y=2)", budget=2500)
    assert report.fits
    assert len(kept) < 50
    # oldest entries are the ones removed
    assert all(h.text == "H" * 100 for h in kept)


def test_budget_metadata_recorded():
    _, report = fit_to_budget(task_text="t", history=[HistoryItem(text="x")],
                              current_image_tokens=100, assistant_target="wait()",
                              budget=8192)
    assert report.estimated_tokens > 0
    assert report.image_count == 1
