import pytest
from PIL import Image

from processing.dedup import (DedupIndex, hamming, index_from_state,
                              index_to_state, phash, phash_bytes)
from tests.conftest import make_png_bytes


def test_phash_deterministic_and_distinct():
    a = phash_bytes(make_png_bytes(640, 480, marker=1))
    b = phash_bytes(make_png_bytes(640, 480, marker=1))
    c = phash_bytes(make_png_bytes(640, 480, marker=250))
    assert a == b
    assert hamming(a, c) > 6


def test_near_duplicate_removal():
    idx = DedupIndex()
    dup, reason = idx.consider(image_phash=phash_bytes(make_png_bytes(marker=1)),
                               signals=[], task_text="open the file",
                               action_text="click(x=1, y=2)")
    assert not dup
    dup, reason = idx.consider(image_phash=phash_bytes(make_png_bytes(marker=1)),
                               signals=[], task_text="open the file",
                               action_text="click(x=1, y=2)")
    assert dup and reason == "task_action_state_duplicate"


def test_same_screen_different_action_is_new_supervision():
    idx = DedupIndex()
    idx.consider(image_phash=phash_bytes(make_png_bytes(marker=1)),
                 signals=[], task_text="task A", action_text="click(x=1, y=2)")
    dup, reason = idx.consider(image_phash=phash_bytes(make_png_bytes(marker=1)),
                               signals=[], task_text="different task text",
                               action_text="click(x=9, y=9)")
    # same screenshot but different action => kept (it is a new sample)
    assert not dup


def test_meaningful_repetition_is_preserved():
    idx = DedupIndex()
    idx.consider(image_phash=phash_bytes(make_png_bytes(marker=7)),
                 signals=[], task_text="first task", action_text="click(x=1, y=1)")
    # same screen + same action again with explicit recovery evidence is kept
    dup, _ = idx.consider(image_phash=phash_bytes(make_png_bytes(marker=7)),
                          signals=["no_state_change", "recovery_evidenced"],
                          task_text="second task", action_text="click(x=1, y=1)")
    assert not dup
    assert idx.stats["near_duplicates_kept_meaningful"] >= 1


def test_identical_visuals_without_signal_removed():
    idx = DedupIndex()
    idx.consider(image_phash=phash_bytes(make_png_bytes(marker=7)),
                 signals=[], task_text="t1", action_text="click(x=1, y=1)")
    # same screen + SAME action under a different task wording is redundant
    dup, reason = idx.consider(image_phash=phash_bytes(make_png_bytes(marker=7)),
                               signals=[], task_text="t2", action_text="click(x=1, y=1)")
    assert dup and reason == "near_duplicate_image_action"
    assert idx.stats["near_duplicates_removed"] == 1


def test_exact_image_duplicate_detection():
    idx = DedupIndex()
    data = make_png_bytes(marker=3)
    assert not idx.register_image(data)
    assert idx.register_image(data)
    assert idx.stats["exact_image_duplicates"] == 1


def test_state_round_trip(tmp_path):
    idx = DedupIndex()
    idx.consider(image_phash=12345, signals=[], task_text="a", action_text="b")
    idx.consider(image_phash=999, signals=[], task_text="c", action_text="d")
    state = index_to_state(idx)
    restored = index_from_state(state)
    assert restored._seen_phashs == idx._seen_phashs
    assert restored._task_action_hashes == idx._task_action_hashes


def test_hamming():
    assert hamming(0b0000, 0b1111) == 4
    assert hamming(0, 0) == 0
