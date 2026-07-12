"""K1 session-context builder contract (plan K).

Secures: durable-rows composition (question, Q/A lines, approval notes,
answer body), the deterministic trim policy with VISIBLE markers, the
artifact registry, and the never-raise degradation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from inqtrix.agents.control_memory import MemoryAgentControlStore
from inqtrix.agents.control_ports import ClarificationRecord
from inqtrix.agents.session_context import (
    RECENT_TURNS_VERBATIM,
    TOTAL_HISTORY_CHAR_BUDGET,
    build_session_context,
    SessionContextPack,
    _within_budget,
)


class FakeRunStore:
    """Narrow stand-in for the run store surface the builder touches."""

    def __init__(
        self,
        summaries: list[dict[str, Any]],
        results: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._summaries = summaries
        self._results = results or {}

    def list_session_runs(
        self, session_id: str, *, visible_to: Any = None
    ) -> list[dict[str, Any]]:
        return [
            dict(summary)
            for summary in self._summaries
            if summary.get("session_id") == session_id
        ]

    def result(self, run_id: str, *, visible_to: Any = None) -> dict[str, Any]:
        payload = self._results.get(run_id)
        if payload is None:
            raise KeyError(run_id)
        return dict(payload)


def _summary(
    run_id: str,
    question: str,
    *,
    status: str = "completed",
    session_id: str = "sess-1",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "question": question,
        "status": status,
        "session_id": session_id,
        "agent_overrides": {},
    }


def _pack(
    summaries: list[dict[str, Any]],
    *,
    results: dict[str, dict[str, Any]] | None = None,
    control: MemoryAgentControlStore | None = None,
    current_run_id: str = "run_current",
) -> SessionContextPack:
    return build_session_context(
        "sess-1",
        run_store=FakeRunStore(summaries, results),
        control=control or MemoryAgentControlStore(),
        run_async=asyncio.run,
        visible_to=None,
        current_run_id=current_run_id,
    )


def test_follow_up_history_contains_question_qa_and_answer():
    control = MemoryAgentControlStore()
    asyncio.run(
        control.create_clarification(
            ClarificationRecord(
                clarification_id="clr_1",
                run_id="run_1",
                question="Welcher Markt?",
                questions=(
                    {
                        "id": "q1",
                        "prompt": "Welcher Markt?",
                        "options": [
                            {"id": "q1_o1", "label": "Europa", "description": ""}
                        ],
                        "multi_select": False,
                    },
                ),
                answers={"q1": {"option_ids": ["q1_o1"], "text": ""}},
                status="answered",
            )
        )
    )
    pack = _pack(
        [_summary("run_1", "Erstelle eine Marktanalyse.")],
        results={"run_1": {"answer": "Die Marktlage ist stabil [W1]."}},
        control=control,
    )
    assert "Nutzer: Erstelle eine Marktanalyse." in pack.history_block
    assert (
        "Rueckfrage: Welcher Markt? — Antwort: Europa" in pack.history_block
    )
    assert "Agent: Die Marktlage ist stabil [W1]." in pack.history_block


def test_current_run_and_children_are_excluded():
    summaries = [
        _summary("run_1", "Frage eins."),
        {**_summary("run_child", "Kindlauf."), "kind": "agent_child"},
        _summary("run_current", "Aktuelle Frage."),
    ]
    pack = _pack(summaries, results={"run_1": {"answer": "Antwort eins."}})
    assert "Frage eins." in pack.history_block
    assert "Kindlauf." not in pack.history_block
    assert "Aktuelle Frage." not in pack.history_block


def test_older_turns_collapse_to_one_liners():
    summaries = [
        _summary(f"run_{i}", f"Frage Nummer {i}.") for i in range(6)
    ]
    results = {
        f"run_{i}": {"answer": f"Antwort {i}."} for i in range(6)
    }
    pack = _pack(summaries, results=results)
    # The oldest turns collapse; the newest RECENT_TURNS_VERBATIM keep
    # their full bodies.
    assert 'Frueher: "Frage Nummer 0."' in pack.history_block
    for i in range(6 - RECENT_TURNS_VERBATIM, 6):
        assert f"Agent: Antwort {i}." in pack.history_block
    assert "Agent: Antwort 0." not in pack.history_block


def test_total_budget_trims_visibly():
    # A verbatim turn can legitimately exceed the per-turn body cap via
    # long free-text clarification answers (uncapped by design — they
    # are user input); the TOTAL budget then drops the OLDEST turns and
    # marks the cut visibly.
    control = MemoryAgentControlStore()
    for i in range(4):
        asyncio.run(
            control.create_clarification(
                ClarificationRecord(
                    clarification_id=f"clr_{i}",
                    run_id=f"run_{i}",
                    question="Kontext?",
                    status="answered",
                    answer="Y" * 3000,
                )
            )
        )
    summaries = [_summary(f"run_{i}", f"Frage {i}.") for i in range(4)]
    results = {f"run_{i}": {"answer": "Antwort."} for i in range(4)}
    pack = _pack(summaries, results=results, control=control)
    assert len(pack.history_block) <= TOTAL_HISTORY_CHAR_BUDGET + 100
    assert pack.history_block.startswith(
        "[... aeltere Verlaufsteile gekuerzt]"
    )
    # The newest turn always survives.
    assert "Frage 3." in pack.history_block


def test_one_oversized_latest_turn_does_not_claim_older_turns_were_trimmed():
    block = _within_budget(["Nutzer: " + "X" * 9000])

    assert "juengster Verlaufsturn gekuerzt" in block
    assert "aeltere Verlaufsteile gekuerzt" not in block


def test_pruned_result_degrades_visibly_not_silently():
    pack = _pack([_summary("run_1", "Frage eins.")], results={})
    assert "Agent: (Ergebnis nicht mehr verfuegbar)" in pack.history_block


def test_artifact_registry_lists_session_deliverables():
    control = MemoryAgentControlStore()
    asyncio.run(
        control.upsert_artifact(
            run_id="run_1",
            kind="memo",
            session_id="sess-1",
            title="Marktanalyse",
            status="ready",
            content_markdown="# Memo",
            payload={},
            refs=[],
            updated_by="agent",
            artifact_id="art_test_memo",
        )
    )
    pack = _pack(
        [_summary("run_1", "Frage.")],
        results={"run_1": {"answer": "A."}},
        control=control,
    )
    assert len(pack.artifact_registry) == 1
    entry = pack.artifact_registry[0]
    assert entry["kind"] == "memo"
    assert entry["title"] == "Marktanalyse"
    assert entry["revision"] == 1


def test_prior_evidence_count_deduplicates_canonical_sources():
    control = MemoryAgentControlStore()
    shared = {"url": "https://example.com/report", "title": "Report"}
    for artifact_id in ("art_one", "art_two"):
        asyncio.run(
            control.upsert_artifact(
                run_id="run_1",
                kind="deliverable",
                session_id="sess-1",
                title=artifact_id,
                status="ready",
                content_markdown="Body",
                payload={},
                refs=[shared],
                updated_by="agent",
                artifact_id=artifact_id,
            )
        )

    pack = _pack(
        [_summary("run_1", "Frage.")],
        results={"run_1": {"answer": "A."}},
        control=control,
    )

    assert pack.prior_evidence_count == 1


def test_effective_execution_response_form_precedes_legacy_override():
    summary = _summary("run_1", "Frage.")
    summary["snapshot"] = {"execution": {"response_form": "canvas"}}
    summary["agent_overrides"] = {"response_form": "chat"}

    pack = _pack([summary], results={"run_1": {"answer": "A."}})

    assert pack.last_response_form == "canvas"


def test_control_store_outage_keeps_the_run_history(caplog):
    """Partial degradation: a broken control store costs the registry
    and the Q/A lines, NEVER the run-based history (each loudly)."""

    class BrokenControl:
        async def list_session_artifacts(self, session_id: str):
            raise RuntimeError("control down")

        async def list_clarifications(self, run_id: str):
            raise RuntimeError("control down")

        async def list_approvals(self, run_id: str):
            raise RuntimeError("control down")

    handler_target = logging.getLogger("inqtrix")
    handler_target.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="inqtrix"):
            pack = build_session_context(
                "sess-1",
                run_store=FakeRunStore(
                    [_summary("run_1", "Frage eins.")],
                    {"run_1": {"answer": "Antwort eins."}},
                ),
                control=BrokenControl(),  # type: ignore[arg-type]
                run_async=asyncio.run,
                visible_to=None,
                current_run_id="run_x",
            )
    finally:
        handler_target.removeHandler(caplog.handler)
    assert "Nutzer: Frage eins." in pack.history_block
    assert "Agent: Antwort eins." in pack.history_block
    assert pack.artifact_registry == ()
    assert any(
        "Artefakt-Registry" in record.message for record in caplog.records
    )


def test_builder_never_raises(caplog):
    class BrokenStore:
        def list_session_runs(self, *args: Any, **kwargs: Any):
            raise RuntimeError("kaputt")

    handler_target = logging.getLogger("inqtrix")
    handler_target.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="inqtrix"):
            pack = build_session_context(
                "sess-1",
                run_store=BrokenStore(),
                control=MemoryAgentControlStore(),
                run_async=asyncio.run,
                visible_to=None,
                current_run_id="run_x",
            )
    finally:
        handler_target.removeHandler(caplog.handler)
    assert pack == SessionContextPack()
    assert any(
        "Session-Kontext" in record.message for record in caplog.records
    )
