"""Pure helpers shared by every run-store backend.

The byte-level wire contract (run summary keys, SSE event envelope,
the auto-emitted ``inqtrix.run.snapshot`` companion BEFORE its carrier
event) is pinned by ``tests/contract/``. Both the in-memory store and
the Postgres store build summaries and expand events through these
functions, so the contract has exactly one implementation.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from inqtrix.runtime_logging import sanitize_event_payload

SNAPSHOT_EVENT = "inqtrix.run.snapshot"


class RunRecordView(Protocol):
    """Attribute surface a record must offer to be summarized.

    Satisfied by the in-memory ``RunRecord`` dataclass and the row
    view the Postgres store builds from a ``runs`` row — ``status``
    may be the ``RunStatus`` enum or its string value.
    """

    run_id: str
    status: Any
    question: str
    stack_name: str
    workspace_id: str | None
    mode: str
    agent_overrides: dict[str, Any]
    created_at: float
    started_at: float | None
    finished_at: float | None
    snapshot: dict[str, Any]
    error: dict[str, Any] | None


def status_value(status: Any) -> str:
    """String form of a status that may be an enum or already a string."""
    return getattr(status, "value", status)


def access_annotation(shared: Any) -> dict[str, Any] | None:
    """The shared-in ``access`` payload for *shared* (a SharePermission).

    Defined once so both store backends emit the identical wire shape;
    ``None`` (an owned run) keeps the summary key absent entirely.
    """
    if shared is None:
        return None
    return {"via": "share", "permission": shared.value}


def build_run_summary(
    record: RunRecordView,
    *,
    queue_position: int | None,
    access: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Public run summary — the 16-key wire shape of ``/v1/runs``.

    *access* is the ADDITIVE shared-in annotation
    (``{"via": "share", "permission": "view"}``); owned runs omit the
    key entirely so the historical wire shape stays byte-identical.
    """
    elapsed = None
    if record.started_at is not None:
        end = record.finished_at or time.time()
        elapsed = round(max(0.0, end - record.started_at), 2)
    return {
        "run_id": record.run_id,
        "status": status_value(record.status),
        "queue_position": queue_position,
        "question": record.question,
        "stack": record.stack_name,
        "workspace_id": record.workspace_id,
        "mode": record.mode,
        "agent_overrides": dict(record.agent_overrides),
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "elapsed_seconds": elapsed,
        "snapshot": dict(record.snapshot),
        "error": dict(record.error) if record.error else None,
        "events_url": f"/v1/runs/{record.run_id}/events",
        "result_url": f"/v1/runs/{record.run_id}/result",
        **({"access": access} if access is not None else {}),
    }


def expand_run_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    status: str,
) -> tuple[dict[str, Any] | None, list[tuple[str, dict[str, Any]]]]:
    """Sanitize one event and expand its snapshot companion.

    Any payload whose ``snapshot`` value is a dict updates the run's
    stored snapshot AND auto-emits an ``inqtrix.run.snapshot``
    companion event BEFORE the carrier event (unless the carrier IS
    the snapshot event) — the order the SSE contract pins.

    Args:
        event_type: Carrier event type.
        payload: Raw event payload (sanitized here, never trusted).
        status: Current run status string for the companion payload.

    Returns:
        ``(new_snapshot, events)`` where ``new_snapshot`` is the dict
        to store on the run (``None`` when the payload carried no
        snapshot) and ``events`` is the ordered list of
        ``(type, clean_payload)`` pairs to append.
    """
    clean_payload = sanitize_event_payload(event_type, dict(payload))
    events: list[tuple[str, dict[str, Any]]] = []
    new_snapshot: dict[str, Any] | None = None
    snapshot = clean_payload.get("snapshot")
    if isinstance(snapshot, dict):
        new_snapshot = dict(snapshot)
        if event_type != SNAPSHOT_EVENT:
            events.append(
                (
                    SNAPSHOT_EVENT,
                    sanitize_event_payload(
                        SNAPSHOT_EVENT,
                        {"status": status, "snapshot": new_snapshot},
                    ),
                )
            )
    events.append((event_type, clean_payload))
    return new_snapshot, events
