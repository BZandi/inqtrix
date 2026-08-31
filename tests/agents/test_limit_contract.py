"""Durable Agent Desk limit-decision contract."""

from __future__ import annotations

from dataclasses import replace

from inqtrix.agents.control_memory import MemoryAgentControlStore
from inqtrix.agents.discovery import build_probe_plan
from inqtrix.agents.kernel.deps import run_coro
from inqtrix.agents.limit_contract import (
    LIMIT_CHOICE_PARTIAL,
    AgentLimitGate,
    create_or_get_limit_gate,
    effective_extended_limit,
    latest_terminal_limit_choice,
    next_extended_limit,
    parse_limit_gate,
)


def test_extension_is_monotonic_and_operator_bounded() -> None:
    assert next_extended_limit(current=30, ceiling=60) == 60
    assert next_extended_limit(current=30, ceiling=45, required=44) == 45
    assert next_extended_limit(current=60, ceiling=60) == 60


def test_discovery_budget_retains_visible_omission_count() -> None:
    plan = build_probe_plan(
        None,
        question="Aktuelle Preise?",
        collection_ids=[],
        max_calls=1,
        web_preview_allowed=True,
    )
    assert len(plan.probes) == 1
    assert plan.requested_count == 3
    assert plan.limit == 1
    assert plan.omitted_count == 2


def test_gate_creation_is_idempotent_and_persists_exact_reason() -> None:
    store = MemoryAgentControlStore()
    gate = AgentLimitGate(
        kind="tool_calls",
        used=29,
        current=30,
        proposed=33,
        ceiling=60,
    )
    first, first_created = create_or_get_limit_gate(
        store,
        run_id="run_1234567890ab",
        gate=gate,
        run_async=run_coro,
    )
    second, second_created = create_or_get_limit_gate(
        store,
        run_id="run_1234567890ab",
        gate=gate,
        run_async=run_coro,
    )

    assert first_created is True
    assert second_created is False
    assert second == first
    assert "29 Werkzeugaufrufe" in first.question
    assert "Kein Aufruf dieses Batches wurde ausgeführt" in first.question
    assert parse_limit_gate(first) == gate


def test_operator_ceiling_never_offers_a_fake_extension() -> None:
    store = MemoryAgentControlStore()
    gate = AgentLimitGate(
        kind="steps", used=145, current=145, proposed=145, ceiling=145
    )
    record, _ = create_or_get_limit_gate(
        store,
        run_id="run_1234567890ab",
        gate=gate,
        run_async=run_coro,
    )
    assert [option["id"] for option in record.options] == ["partial", "cancel"]
    assert "lässt keine Erweiterung zu" in record.question


def test_answered_extension_folds_idempotently_without_resetting_usage() -> None:
    store = MemoryAgentControlStore()
    gate = AgentLimitGate(
        kind="tool_calls", used=30, current=30, proposed=60, ceiling=60
    )
    record, _ = create_or_get_limit_gate(
        store,
        run_id="run_1234567890ab",
        gate=gate,
        run_async=run_coro,
    )
    run_coro(
        store.create_clarification(
            replace(record, status="answered", option_id="extend")
        )
    )

    first = effective_extended_limit(
        store,
        run_id=record.run_id,
        kind="tool_calls",
        base=30,
        ceiling=60,
        run_async=run_coro,
    )
    second = effective_extended_limit(
        store,
        run_id=record.run_id,
        kind="tool_calls",
        base=30,
        ceiling=60,
        run_async=run_coro,
    )
    assert first == second == 60
    assert parse_limit_gate(record).used == 30  # type: ignore[union-attr]


def test_partial_choice_is_terminal_only_after_answer_is_persisted() -> None:
    store = MemoryAgentControlStore()
    gate = AgentLimitGate(
        kind="steps", used=73, current=73, proposed=145, ceiling=145
    )
    pending, _ = create_or_get_limit_gate(
        store,
        run_id="run_1234567890ab",
        gate=gate,
        run_async=run_coro,
    )
    assert latest_terminal_limit_choice(
        store, run_id=pending.run_id, run_async=run_coro
    ) is None

    run_coro(
        store.create_clarification(
            replace(
                pending,
                status="answered",
                option_id=LIMIT_CHOICE_PARTIAL,
            )
        )
    )
    terminal = latest_terminal_limit_choice(
        store, run_id=pending.run_id, run_async=run_coro
    )
    assert terminal is not None
    parsed, choice, record = terminal
    assert parsed == gate
    assert choice == LIMIT_CHOICE_PARTIAL
    assert record.status == "answered"


def test_effective_tool_grants_folds_only_run_scoped_tool_approves() -> None:
    """P6B fold: only approve + kind=tool + approval_scope=run grants;
    always-gated tools stay excluded even if such a row slipped in."""
    from inqtrix.agents.control_ports import ApprovalRecord
    from inqtrix.agents.limit_contract import effective_tool_grants

    store = MemoryAgentControlStore()
    run_id = "run_grants567890"

    def _row(
        approval_id: str,
        *,
        kind: str = "tool",
        decision: str = "approve",
        scope: str | None = "run",
        tool: str = "web_instant",
    ) -> ApprovalRecord:
        return ApprovalRecord(
            approval_id=approval_id,
            run_id=run_id,
            kind=kind,
            status="approved" if decision else "pending",
            payload={"actions": [{"tool": tool, "args": {}, "summary": ""}]},
            decision=decision,
            decision_payload=(
                {"approval_scope": scope} if scope is not None else {}
            ),
        )

    rows = [
        _row("apr_granted"),
        _row("apr_once", scope=None),
        _row("apr_rejected", decision="reject"),
        _row("apr_plan", kind="plan"),
        _row("apr_skill", tool="load_skill"),
        # Defense in depth: the service refuses this, and the fold
        # refuses to honor one that slipped in anyway.
        _row("apr_patch", tool="propose_editor_patch"),
    ]
    for row in rows:
        run_coro(store.create_approval(row))

    grants = effective_tool_grants(store, run_id=run_id, run_async=run_coro)
    assert grants == frozenset({"web_instant", "load_skill"})
    assert (
        effective_tool_grants(store, run_id="run_other", run_async=run_coro)
        == frozenset()
    )
