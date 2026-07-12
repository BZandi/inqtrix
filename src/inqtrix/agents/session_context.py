"""K1 — the server-built session context pack (plan K).

Rows are truth (R1): every follow-up composes durable run/control metadata —
device-independent and reload-safe. Explicit ``request.history`` replaces
only the rendered conversation block; artifact registry, effective output
form, and prior evidence count still come from server rows.

Trim policy is DETERMINISTIC and loud: the newest
:data:`RECENT_TURNS_VERBATIM` turns keep their full Q/A (answer bodies
capped at :data:`TURN_BODY_CHAR_CAP` chars with a visible cut marker),
older turns collapse to one line each, and when the total budget forces
dropping older turns entirely the block STARTS with a visible trim
marker — a shortened context never masquerades as the whole history
(No Silent Fallbacks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from inqtrix.agents.clarification import round_qa_lines

if TYPE_CHECKING:
    from inqtrix.agents.control_ports import AgentControlStore

log = logging.getLogger("inqtrix")

RECENT_TURNS_VERBATIM = 3
"""Newest turns rendered with full question + answer body."""

TURN_BODY_CHAR_CAP = 2000
"""Per-turn answer-body cap (recent turns); cut is visibly marked."""

TOTAL_HISTORY_CHAR_BUDGET = 8000
"""Hard budget of the whole history block."""

_CUT_MARKER = "\n[... gekuerzt]"
_TRIMMED_HEADER = "[... aeltere Verlaufsteile gekuerzt]\n\n"

_TERMINAL_WORDS = {
    "completed": "abgeschlossen",
    "failed": "fehlgeschlagen",
    "cancelled": "abgebrochen",
}


@dataclass(frozen=True)
class SessionContextPack:
    """What a follow-up turn knows about its session (plan K)."""

    history_block: str = ""
    """The composed conversational history ('' for a first turn)."""

    artifact_registry: tuple[dict[str, Any], ...] = field(
        default_factory=tuple
    )
    """Meta of ALL session deliverables ``{artifact_id, kind, title,
    revision, updated_by}`` — the K2 multi-deliverable index."""

    last_response_form: str = ""
    """Output form of the latest completed turn (K3 routing memory);
    '' when unknown (pre-S3 runs carry no form)."""

    prior_evidence_count: int = 0
    """Distinct references retained by session artifacts (K4)."""


def build_session_context(
    session_id: str,
    *,
    run_store: Any,
    control: "AgentControlStore",
    run_async: Callable[[Awaitable[Any]], Any],
    visible_to: Any,
    current_run_id: str,
) -> SessionContextPack:
    """Compose the pack from durable rows (never raises — a context
    failure must not kill the run; it degrades VISIBLY to empty).

    Args:
        run_store: The run store port (sync surface).
        control: The agent control store (async surface).
        run_async: Bridge that executes a coroutine from this sync worker
            thread (the algorithm passes its ``_run_async``).
        visible_to: The owner's resolved visibility (E5) — the context
            can only ever contain what the OWNER may see.
        current_run_id: Excluded (the turn being answered).
    """
    try:
        return _build(
            session_id,
            run_store=run_store,
            control=control,
            run_async=run_async,
            visible_to=visible_to,
            current_run_id=current_run_id,
        )
    except Exception:  # noqa: BLE001 — degrade visibly, never kill intake
        log.warning(
            "Session-Kontext fuer %s konnte nicht gebaut werden — "
            "der Turn laeuft ohne Verlaufskontext.",
            session_id,
            exc_info=True,
        )
        return SessionContextPack()


def _build(
    session_id: str,
    *,
    run_store: Any,
    control: "AgentControlStore",
    run_async: Callable[[Awaitable[Any]], Any],
    visible_to: Any,
    current_run_id: str,
) -> SessionContextPack:
    summaries = [
        summary
        for summary in run_store.list_session_runs(
            session_id, visible_to=visible_to
        )
        if summary.get("run_id") != current_run_id
        and summary.get("kind") != "agent_child"
    ]
    # Partial degradation: a control-store outage costs the artifact
    # registry, never the run-based history (each visibly, per part).
    artifacts: list[Any] = []
    try:
        artifacts = run_async(control.list_session_artifacts(session_id))
    except Exception:  # noqa: BLE001 — degrade visibly, keep the history
        log.warning(
            "Artefakt-Registry fuer %s nicht verfuegbar — der "
            "Verlaufskontext wird ohne Deliverables-Index gebaut.",
            session_id,
            exc_info=True,
        )
    registry = tuple(
        {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "title": artifact.title,
            "revision": artifact.revision,
            "updated_by": artifact.updated_by,
        }
        for artifact in artifacts
    )
    from inqtrix.agents.evidence import dedup_key

    evidence_keys = {
        key
        for artifact in artifacts
        for ref in artifact.refs
        if (key := dedup_key(dict(ref)))
    }
    evidence_count = len(evidence_keys)
    if not summaries:
        return SessionContextPack(
            artifact_registry=registry,
            prior_evidence_count=evidence_count,
        )
    verbatim_ids = {
        summary.get("run_id") for summary in summaries[-RECENT_TURNS_VERBATIM:]
    }
    turns: list[str] = []
    last_response_form = ""
    for summary in summaries:
        run_id = str(summary.get("run_id", ""))
        if summary.get("status") == "completed":
            execution = (summary.get("snapshot") or {}).get("execution") or {}
            overrides = summary.get("agent_overrides") or {}
            last_response_form = str(
                execution.get("response_form")
                or overrides.get("response_form", "")
                or ""
            )
        if run_id in verbatim_ids:
            turns.append(
                _verbatim_turn(
                    summary,
                    run_id=run_id,
                    run_store=run_store,
                    control=control,
                    run_async=run_async,
                    visible_to=visible_to,
                )
            )
        else:
            turns.append(_one_liner(summary))
    return SessionContextPack(
        history_block=_within_budget(turns),
        artifact_registry=registry,
        last_response_form=last_response_form,
        prior_evidence_count=evidence_count,
    )


def _verbatim_turn(
    summary: dict[str, Any],
    *,
    run_id: str,
    run_store: Any,
    control: "AgentControlStore",
    run_async: Callable[[Awaitable[Any]], Any],
    visible_to: Any,
) -> str:
    question = str(summary.get("question", "")).strip()
    if len(question) > 500:
        question = question[:500] + "…"
    lines = [f"Nutzer: {question}"]
    try:
        for record in reversed(
            run_async(control.list_clarifications(run_id))
        ):
            for prompt, answer in round_qa_lines(
                questions=list(record.questions),
                question=record.question,
                options=list(record.options),
                answers=dict(record.answers),
                answer=record.answer,
                option_id=record.option_id,
            ):
                lines.append(f"Rueckfrage: {prompt} — Antwort: {answer}")
        for approval in reversed(run_async(control.list_approvals(run_id))):
            if approval.status == "rejected":
                note = (
                    f" (Begruendung: {approval.note})" if approval.note else ""
                )
                lines.append(
                    f"Hinweis: Nutzer lehnte die {approval.kind}-Freigabe "
                    f"ab{note}."
                )
            elif approval.decision == "edit":
                lines.append(
                    "Hinweis: Nutzer passte den Plan vor Freigabe an."
                )
    except Exception:  # noqa: BLE001 — degrade visibly, keep the turn
        log.warning(
            "Kontrolldaten (Rueckfragen/Freigaben) fuer Run %s nicht "
            "verfuegbar — der Turn erscheint ohne diese Zeilen im "
            "Verlaufskontext.",
            run_id,
            exc_info=True,
        )
    status = str(summary.get("status", ""))
    if status == "completed":
        body = _answer_body(run_id, run_store, visible_to)
        if body:
            lines.append(f"Agent: {body}")
        else:
            lines.append("Agent: (Ergebnis nicht mehr verfuegbar)")
    else:
        word = _TERMINAL_WORDS.get(status, status)
        lines.append(f"Agent: Lauf {word}.")
    return "\n".join(lines)


def _one_liner(summary: dict[str, Any]) -> str:
    question = str(summary.get("question", "")).strip()
    if len(question) > 100:
        question = question[:100] + "…"
    status = str(summary.get("status", ""))
    word = _TERMINAL_WORDS.get(status, status)
    return f'Frueher: "{question}" — Lauf {word}.'


def _answer_body(run_id: str, run_store: Any, visible_to: Any) -> str:
    try:
        payload = run_store.result(run_id, visible_to=visible_to)
    except Exception:  # noqa: BLE001 — pruned result is a normal state
        return ""
    body = str(payload.get("answer", "") or "").strip()
    if len(body) > TURN_BODY_CHAR_CAP:
        body = body[:TURN_BODY_CHAR_CAP] + _CUT_MARKER
    return body


def _within_budget(turns: list[str]) -> str:
    """Newest turns survive; dropping older ones is visibly marked."""
    kept = list(turns)
    trimmed = False
    while (
        len(kept) > 1
        and len("\n\n".join(kept)) > TOTAL_HISTORY_CHAR_BUDGET
    ):
        kept.pop(0)
        trimmed = True
    block = "\n\n".join(kept)
    if len(block) > TOTAL_HISTORY_CHAR_BUDGET:
        marker = "\n[... juengster Verlaufsturn gekuerzt]"
        prefix = _TRIMMED_HEADER if trimmed else ""
        content_budget = (
            TOTAL_HISTORY_CHAR_BUDGET
            - len(prefix)
            - len(marker)
        )
        return prefix + block[:content_budget] + marker
    if trimmed:
        available = TOTAL_HISTORY_CHAR_BUDGET - len(_TRIMMED_HEADER)
        block = _TRIMMED_HEADER + block[:available]
    return block
