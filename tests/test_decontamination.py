import pytest

from processing.decontamination import (Decontaminator, containment, jaccard,
                                        load_osworld_instructions_from_bundle,
                                        normalize, shingles)


def make_decon():
    d = Decontaminator()
    d.add_reference("osworld/libreoffice/a1", "Export the current spreadsheet as a PDF file to the Documents folder.")
    d.add_reference("osworld/vlc/b2", "Change the VLC audio output device to Bluetooth headphones.")
    return d


def test_normalize():
    assert normalize("Hello,  World!HELLO") == "hello world hello"


def test_exact_match_removed():
    d = make_decon()
    remove, reason, _ = d.check("Export the current spreadsheet as a PDF file to the Documents folder.")
    assert remove and reason == "exact_match"


def test_case_and_punctuation_insensitive_exact():
    d = make_decon()
    remove, _, _ = d.check("export the CURRENT spreadsheet as a pdf file to the documents folder.")
    assert remove


def test_high_similarity_removed():
    d = make_decon()
    cand = ("Export the current spreadsheet as a PDF file to the Documents folder and print it")
    remove, reason, score = d.check(cand)
    assert remove and score >= 0.5


def test_application_overlap_alone_is_not_removed():
    d = make_decon()
    # same app, entirely different instruction
    remove, _, score = d.check("Insert a new row at the top of the sheet and bold the header.")
    assert not remove


def test_unrelated_kept():
    d = make_decon()
    remove, _, _ = d.check("Sort the playlist by artist name.")
    assert not remove


def test_containment_short_instruction():
    d = make_decon()
    remove, _, _ = d.check("Change the VLC audio output device")
    assert remove  # short candidate fully contained in reference


def test_report_fields():
    d = make_decon()
    for _ in range(3):
        d.check_sample("Export the current spreadsheet as a PDF file to the Documents folder.", "procua")
    d.check_sample("Completely unrelated task text here now", "gui360")
    report = d.report()
    for key in ("total_scanned", "exact_matches", "high_similarity_matches",
                "removed_examples", "source_breakdown"):
        assert key in report
    assert report["removed_examples"] == 3
    assert report["source_breakdown"]["procua"] == 3


def test_bundle_loader():
    bundle = {"os": [{"id": "t1", "instruction": "Open the terminal."},
                     {"id": "t2"}]}
    pairs = load_osworld_instructions_from_bundle(bundle)
    assert pairs == [("os/t1", "Open the terminal.")]


def test_jaccard_and_shingles():
    s1 = shingles("a b c d e f g h i j")
    s2 = shingles("a b c d e f g h i j")
    assert jaccard(s1, s2) == 1.0
    assert containment(s1, s2) == 1.0
    s3 = shingles("z y x w v u t s r q")
    assert jaccard(s1, s3) == 0.0
