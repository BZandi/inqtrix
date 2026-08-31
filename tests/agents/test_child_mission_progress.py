"""What a delegating parent learns while its mission child works.

A kernel run that delegates a deep mission used to receive ELEVEN events
in fifty minutes — queued, started, resumed, waiting, completed — and
nothing about the work itself, because the mission reports progress under
``agent.*`` while the projection allowlist only knew the research
vocabulary. The parent surface could therefore not say what was running.
"""

from inqtrix.runs.shared import (
    build_child_progress_payload,
    should_project_child_event,
)


def _project(event_type: str, payload: dict) -> dict:
    return build_child_progress_payload(
        child_run_id="run_child",
        parent_task_id="call_abc",
        run_status="running",
        event_type=event_type,
        payload=payload,
        snapshot={},
        attempt=1,
    )


def test_mission_progress_events_reach_the_parent() -> None:
    """The regression: these three carry a mission's visible progress."""
    for event_type in (
        "inqtrix.agent.phase.changed",
        "inqtrix.agent.task.started",
        "inqtrix.agent.task.finished",
    ):
        assert should_project_child_event(event_type), event_type


def test_a_task_start_names_the_unit_of_work() -> None:
    projected = _project(
        "inqtrix.agent.task.started",
        {"task_id": "t2", "ordinal": 1, "tool_kind": "web_research", "attempt": 1},
    )
    assert projected["ordinal"] == 1
    assert projected["tool_kind"] == "web_research"
    # The projection's task_id is the PARENT's, never the child's own.
    assert projected["task_id"] == "call_abc"


def test_a_finished_task_carries_its_outcome_but_not_its_report() -> None:
    """The child's executive summary must not cross the run boundary."""
    projected = _project(
        "inqtrix.agent.task.finished",
        {
            "task_id": "t1",
            "ordinal": 0,
            "tool_kind": "web_research",
            "status": "completed",
            "child_run_id": "run_grandchild",
            "result_summary": "## Executive Summary " + "x" * 4000,
            "metrics": {"reference_count": 53, "claim_count": 30},
        },
    )
    assert projected["status"] == "completed"
    assert projected["ordinal"] == 0
    assert projected["metrics"]["reference_count"] == 53
    assert "result_summary" not in projected
    # A GRANDCHILD id must never overwrite the child this row is about.
    assert projected["child_run_id"] == "run_child"


def test_a_phase_change_carries_the_childs_phase() -> None:
    projected = _project(
        "inqtrix.agent.phase.changed",
        {
            "phase": "execution",
            "previous_phase": "planning",
            "snapshot": {"current_node": "execution", "phase": "execution"},
        },
    )
    assert projected["snapshot"]["phase"] == "execution"
    assert projected["current_node"] == "execution"


def test_token_deltas_still_never_cross() -> None:
    """The 91%-of-all-events class stays out: this is a status feed, not
    a mirror of the child's stream."""
    assert not should_project_child_event("inqtrix.output_text.delta")
