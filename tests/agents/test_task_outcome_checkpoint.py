"""TaskOutcome durable-checkpoint (de)serialization tolerance.

The outcomes map is threaded through the LangGraph checkpoint, which the
Postgres saver persists across process restarts and deploys — so a value
written by version N is rehydrated by version N+1's code. These tests pin
that a ``TaskOutcome`` schema change (a field added, removed, or renamed)
does NOT break resume of an in-flight run, and that the one unreconcilable
case fails loudly instead of with an opaque ``TypeError``.
"""

from __future__ import annotations

import json

import pytest

from inqtrix.agents.scheduler import CheckpointSchemaError, TaskOutcome


def test_to_state_from_state_round_trips():
    outcome = TaskOutcome(
        status="completed",
        summary="did the thing",
        evidence=[{"label": "src", "url": "https://x"}],
        claims=[{"text": "c"}],
        child_run_id="rn_child",
        failure_reason="",
        usage={"prompt_tokens": 12, "completion_tokens": 4},
        transient=False,
    )
    state = outcome.to_state()
    # Serializable plain data (goes through the JSON checkpoint serializer).
    assert json.loads(json.dumps(state)) == state
    assert TaskOutcome.from_state(state) == outcome


def test_from_state_drops_unknown_field_removed_since_checkpoint():
    # An older checkpoint carried a field this version renamed/removed.
    legacy = {"status": "failed", "summary": "s", "legacy_score": 0.9}
    outcome = TaskOutcome.from_state(legacy)
    assert outcome.status == "failed"
    assert outcome.summary == "s"
    assert not hasattr(outcome, "legacy_score")


def test_from_state_fills_field_added_since_checkpoint():
    # An older checkpoint lacks a field this version added — the default fills.
    old = {"status": "completed"}
    outcome = TaskOutcome.from_state(old)
    assert outcome.status == "completed"
    assert outcome.summary == ""
    assert outcome.evidence == []
    assert outcome.usage == {"prompt_tokens": 0, "completion_tokens": 0}
    assert outcome.transient is False


def test_from_state_raises_on_missing_required_field():
    # ``status`` has no default — the one drift that cannot be reconciled.
    with pytest.raises(CheckpointSchemaError):
        TaskOutcome.from_state({"summary": "no status here"})
