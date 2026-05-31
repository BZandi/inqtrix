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
    assert conf == 4


def test_compute_utility_stops_uniformly_after_two_low_utility_rounds():
    # The former DE-policy utility-stop suppression is removed: two consecutive
    # low-utility rounds stop the loop regardless of question topic or how weak
    # the evidence still is.
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
    assert should_stop is True
    assert state["done"] is True


def test_compute_utility_rewards_new_verified_claims():
    strategy = MultiSignalStopCriteria(AgentSettings())
    state = _base_state(
        utility_scores=[0.1],
        evidence_sufficiency=5,
        prev_citation_count=30,
        prev_verified_claim_count=0,
        claim_status_counts={"verified": 3, "contested": 0, "unverified": 0},
    )

    utility, should_stop = strategy.compute_utility(
        state,
        conf=5,
        prev_conf=5,
        n_citations=30,
    )

    assert utility >= 0.3
    assert should_stop is False
    assert state["prev_verified_claim_count"] == 3


def test_check_plateau_suppressed_when_evidence_depth_gap_active():
    strategy = MultiSignalStopCriteria(AgentSettings(max_rounds=5))
    state = _base_state(
        round=2,
        _evidence_depth_gap_active=True,
        done=False,
    )

    should_stop = strategy.check_plateau(
        state,
        conf=8,
        prev_conf=8,
        stagnation_detected=False,
    )

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


def test_extract_evidence_scores_emits_warning_on_parse_miss():
    """When EVIDENCE_* fields are missing, a warning is surfaced to SSE.

    The iteration_log already records ``_evidence_*_parsed=False`` flags,
    but a UI consumer relying on the live event stream would miss the
    fallback without an explicit ``emit_progress`` call.
    """
    events: list[tuple[str, dict]] = []
    strategy = MultiSignalStopCriteria(AgentSettings())
    state = _base_state(
        language="de",
        _run_event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )

    # Eval text without EVIDENCE_CONSISTENCY / EVIDENCE_SUFFICIENCY tokens.
    strategy.extract_evidence_scores(state, "GAPS: nothing\n", conf=5)

    progress_events = [
        payload for event_type, payload in events
        if event_type == "inqtrix.progress.message"
    ]
    assert progress_events, "expected at least one progress event"
    assert any(p["severity"] == "warning" for p in progress_events)
    assert state["_evidence_consistency_parsed"] is False
    assert state["_evidence_sufficiency_parsed"] is False


def test_extract_evidence_scores_silent_when_both_present():
    """No spurious progress event when both scores parse cleanly."""
    events: list[tuple[str, dict]] = []
    strategy = MultiSignalStopCriteria(AgentSettings())
    state = _base_state(
        language="de",
        _run_event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )

    strategy.extract_evidence_scores(
        state,
        "EVIDENCE_CONSISTENCY: 7\nEVIDENCE_SUFFICIENCY: 6\n",
        conf=5,
    )

    progress_events = [
        payload for event_type, payload in events
        if event_type == "inqtrix.progress.message"
    ]
    assert progress_events == []
    assert state["_evidence_consistency_parsed"] is True
    assert state["_evidence_sufficiency_parsed"] is True
