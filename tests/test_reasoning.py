import pytest

from processing.reasoning import (CATEGORY_DISTRIBUTION, ReasoningGate,
                                  compose_assistant_target, make_reasoning)


def test_no_reasoning_without_content_signal():
    assert make_reasoning(["ordinary_click"], "t", "s") is None
    assert make_reasoning([], "t", "s") is None


@pytest.mark.parametrize("signals,category_part", [
    (["recovery_evidenced"], "recovery"),
    (["verification_evidenced"], "verification"),
    (["planning_evidenced"], "planning"),
    (["wait_evidenced"], "wait"),
])
def test_reasoning_category_gating(signals, category_part):
    text = make_reasoning(signals, "traj", "step")
    assert text is not None
    assert text.startswith("Plan:")
    assert CATEGORY_DISTRIBUTION
    # the template bank maps these signals to the right category wording
    templates = {
        "recovery": ("no visible change", "did not reach", "unchanged"),
        "verification": ("confirmed", "verifies", "verified", "confirms"),
        "planning": ("intermediate", "two applications", "next stage"),
        "wait": ("loading", "progress", "time to finish"),
    }
    assert any(w in text.lower() for w in templates[category_part]), text


def test_reasoning_is_concise_and_complete():
    for signals in (["recovery_evidenced"], ["verification_evidenced"],
                    ["planning_evidenced"], ["wait_evidenced"]):
        for i in range(10):
            text = make_reasoning(signals, f"t{i}", f"s{i}")
            assert text is not None
            assert len(text) <= 220
            sentences = [p for p in text.split(". ") if p.strip()]
            assert len(sentences) <= 2
            assert text.rstrip().endswith(".")


def test_no_generic_narration():
    banned = ["Looking at the screenshot", "I can see", "As an AI", "Let me"]
    for signals in (["recovery_evidenced"], ["verification_evidenced"],
                    ["planning_evidenced"], ["wait_evidenced"]):
        for i in range(20):
            text = make_reasoning(signals, f"t{i}", f"s{i}") or ""
            for b in banned:
                assert b.lower() not in text.lower()


def test_compose_target_format():
    plain = compose_assistant_target("click(x=1, y=2)", None)
    assert plain == "click(x=1, y=2)"
    with_plan = compose_assistant_target("click(x=1, y=2)", "Switch approach.")
    assert with_plan == "Plan: Switch approach.\nAction: click(x=1, y=2)"


def test_gate_rate_and_distribution():
    """New contract: the gate self-regulates against note_action_sample() so
    the REALIZED overall rate converges on the configured rate."""
    gate = ReasoningGate(rate=0.12)
    n_reasoning = 0
    n_action_samples = 0
    cats = {"recovery": 0, "verification": 0, "planning": 0, "wait": 0}
    for i in range(4000):
        signals = [["recovery_evidenced"], ["verification_evidenced"],
                   ["planning_evidenced"], ["wait_evidenced"]][i % 4]
        text = gate.allow(signals, f"traj_{i}", f"step_{i}")
        if text:
            n_reasoning += 1
            cats[{"recovery_evidenced": "recovery",
                  "verification_evidenced": "verification",
                  "planning_evidenced": "planning",
                  "wait_evidenced": "wait"}[signals[0]]] += 1
        gate.note_action_sample()   # every action sample emitted
        n_action_samples += 1
    realized = gate.stats()["realized_rate"]
    assert 0.10 <= realized <= 0.15, realized
    # recovery is the largest reasoning share
    assert cats["recovery"] >= cats["wait"]


def test_gate_never_adds_reasoning_to_ineligible():
    gate = ReasoningGate()
    for i in range(100):
        assert gate.allow(["scroll"], "t", "s") is None


def test_run1_default_reasoning_is_disabled():
    gate = ReasoningGate()
    assert gate.rate == 0.0
    for i in range(20):
        assert gate.allow(["recovery_evidenced"], "t", str(i)) is None
