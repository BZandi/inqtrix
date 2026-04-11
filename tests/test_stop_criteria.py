"""Direct regression tests for multi-signal stop heuristics."""

from inqtrix.settings import AgentSettings
from inqtrix.strategies import MultiSignalStopCriteria


def _base_state(**overrides):
    state = {
        "round": 2,
        "question": "Sollen GKV-Leistungen privatisiert werden?",
        "uncovered_aspects": [],
        "source_tier_counts": {"primary": 1, "mainstream": 1, "low": 0},
        "claim_quality_score": 0.8,
        "claim_status_counts": {"verified": 2, "contested": 0, "unverified": 0},
        "claim_needs_primary_total": 0,
        "claim_needs_primary_verified": 0,
        "utility_scores": [],
        "done": False,
        "evidence_sufficiency": 5,
        "prev_citation_count": 0,
        "falsification_triggered": False,
        "competing_events": "",
        "prev_competing_events": "",
        "context": ["Block 1", "Block 2", "Block 3"],
    }
    state.update(overrides)
    return state


def test_check_stagnation_forces_stop_after_broad_low_confidence_search():
    strategy = MultiSignalStopCriteria(AgentSettings())

    conf, detected = strategy.check_stagnation(
        _base_state(),
        conf=4,
        prev_conf=4,
        n_citations=30,
        falsification_just_triggered=False,
    )

    assert detected is True
    assert conf == strategy._confidence_stop


def test_compute_utility_suppresses_stop_when_policy_evidence_is_still_weak():
    strategy = MultiSignalStopCriteria(AgentSettings())
    state = _base_state(
        uncovered_aspects=["Status quo mit konkretem Datum"],
        source_tier_counts={"primary": 0, "mainstream": 0, "low": 1},
        claim_quality_score=0.2,
        claim_status_counts={"verified": 0, "contested": 0, "unverified": 2},
        claim_needs_primary_total=1,
        claim_needs_primary_verified=0,
        utility_scores=[0.1],
        evidence_sufficiency=1,
        prev_citation_count=30,
    )

    utility, should_stop = strategy.compute_utility(
        state,
        conf=4,
        prev_conf=4,
        n_citations=31,
    )

    assert utility < 0.15
    assert should_stop is False
    assert state["done"] is False


def test_extract_competing_events_skips_cap_when_same_event_persists_in_round_three():
    strategy = MultiSignalStopCriteria(AgentSettings())
    state = _base_state(
        round=3,
        prev_competing_events="Verwechslung zwischen Entwurf A und Entwurf B",
    )

    conf = strategy.extract_competing_events(
        state,
        "COMPETING_EVENTS: Verwechslung zwischen Entwurf A und Entwurf B\n",
        conf=strategy._confidence_stop,
    )

    assert conf == strategy._confidence_stop
    assert state["competing_events"] == "Verwechslung zwischen Entwurf A und Entwurf B"
