"""Pure helpers shared by every run-store backend.

The byte-level wire contract (run summary keys, SSE event envelope,
the auto-emitted ``inqtrix.run.snapshot`` companion BEFORE its carrier
event) is pinned by ``tests/contract/``. Both the in-memory store and
the Postgres store build summaries and expand events through these
functions, so the contract has exactly one implementation.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Protocol

from inqtrix.auth.permissions import SharePermission
from inqtrix.runtime_logging import sanitize_event_payload

SNAPSHOT_EVENT = "inqtrix.run.snapshot"
CHILD_PROGRESS_EVENT = "inqtrix.agent.child.progress"

_CHILD_SNAPSHOT_KEYS = frozenset(
    {
        "current_node",
        "completed_rounds",
        "active_round",
        "max_rounds",
        "total_queries",
        "total_citations",
        "total_sources",
        "confidence",
        "source_tier_counts",
        "claim_status_counts",
        "evidence_record_count",
        "consolidated_claim_count",
        "done",
        "progress_estimate",
        "last_message",
        "phase",
    }
)
_CHILD_METRIC_KEYS = frozenset(
    {
        "rounds",
        "queries",
        "sources",
        "citations",
        "claims",
        "result_count",
        "reference_count",
        "claim_count",
    }
)
_CHILD_PROJECTED_EVENTS = frozenset(
    {
        "inqtrix.run.queued",
        "inqtrix.run.started",
        "inqtrix.run.snapshot",
        "inqtrix.run.cancel_requested",
        "inqtrix.run.waiting",
        "inqtrix.run.completed",
        "inqtrix.run.failed",
        "inqtrix.run.cancelled",
        "inqtrix.node.started",
        "inqtrix.node.finished",
        "inqtrix.node.failed",
        "inqtrix.progress.message",
    }
)


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


def access_permits_edit(access: Mapping[str, Any] | None) -> bool:
    """Whether a run summary's ``access`` annotation permits mutation.

    The consumer twin of :func:`access_annotation` — produce and consume of
    the ``access`` wire shape live in one file so they cannot drift. ``None``
    is an OWNED run (full access). A shared-in annotation permits mutation iff
    its grant is edit-or-higher, parsed back through the ORDERED
    :class:`~inqtrix.auth.permissions.SharePermission` so ``manage`` (the
    highest grant, e.g. a workspace owner's) passes too — matching the
    store-level cancel gate (``server/runs.py`` / ``runs/postgres_store.py``),
    never a raw ``== "edit"`` string compare. An unknown permission string is
    treated as no access.
    """
    if access is None:
        return True
    try:
        return SharePermission(access.get("permission")).at_least(
            SharePermission.EDIT
        )
    except ValueError:
        return False


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

    Agent-tree keys (``kind``, ``parent_run_id``, ``root_run_id``,
    ``session_id``) follow the same rule: emitted ONLY when non-default,
    so a standard run's summary stays byte-identical to the historical
    shape. Read via ``getattr`` with defaults so record views that
    predate the fields keep summarizing.
    """
    elapsed = None
    if record.started_at is not None:
        end = record.finished_at or time.time()
        elapsed = round(max(0.0, end - record.started_at), 2)
    agent_tree: dict[str, Any] = {}
    kind = getattr(record, "kind", "standard") or "standard"
    if kind != "standard":
        agent_tree["kind"] = kind
        agent_tree["children_url"] = f"/v1/runs/{record.run_id}/children"
        agent_tree["plan_url"] = f"/v1/runs/{record.run_id}/plan"
        agent_tree["artifacts_url"] = f"/v1/runs/{record.run_id}/artifacts"
    for key in ("parent_run_id", "root_run_id", "session_id", "origin_key"):
        value = getattr(record, key, None)
        if value:
            agent_tree[key] = value
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
        **agent_tree,
        **({"access": access} if access is not None else {}),
    }


def replay_after(
    replay: list[dict[str, Any]], after: int | None
) -> list[dict[str, Any]]:
    """Filter a subscription's replay to events with ``sequence > after``.

    The ONE implementation of the ``?after=<seq>`` reconnect semantics,
    shared by the SSE route over both store backends (R8): the filter
    applies to the REPLAY set only, the live tail is untouched. ``None``
    replays everything (historical behaviour); an ``after`` at or past
    the newest sequence yields an empty replay. Events without a
    ``sequence`` (never produced by the stores) are kept — dropping
    unmarked events silently would hide data.
    """
    if after is None:
        return replay
    return [
        event
        for event in replay
        if not isinstance(event.get("sequence"), int)
        or event["sequence"] > after
    ]


def build_child_progress_payload(
    *,
    child_run_id: str,
    parent_task_id: str,
    run_status: str,
    event_type: str,
    payload: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
    attempt: int | None = None,
) -> dict[str, Any]:
    """Project one child event into a bounded parent-task progress signal.

    The child stream remains the detailed technical record. This projection
    deliberately carries only stable UI facts, preventing provider payloads
    or a child's full state from being copied into the parent event log.
    """
    clean = sanitize_event_payload(event_type, dict(payload))
    carried_snapshot = clean.get("snapshot")
    source_snapshot = (
        carried_snapshot
        if isinstance(carried_snapshot, dict)
        else dict(snapshot or {})
    )
    compact_snapshot = {
        key: value
        for key, value in source_snapshot.items()
        if key in _CHILD_SNAPSHOT_KEYS
    }
    raw_metrics = clean.get("metrics")
    metrics = (
        {
            key: value
            for key, value in raw_metrics.items()
            if key in _CHILD_METRIC_KEYS
        }
        if isinstance(raw_metrics, dict)
        else {}
    )
    message = next(
        (
            str(value).strip()
            for value in (
                clean.get("message"),
                clean.get("label"),
                clean.get("last_message"),
                compact_snapshot.get("last_message"),
            )
            if value and str(value).strip()
        ),
        "",
    )
    error = clean.get("error")
    projected: dict[str, Any] = {
        "task_id": parent_task_id,
        "child_run_id": child_run_id,
        "run_status": run_status,
        "event_type": event_type,
        "updated_at": time.time(),
    }
    if compact_snapshot:
        projected["snapshot"] = compact_snapshot
    current_node = compact_snapshot.get("current_node")
    if current_node:
        projected["current_node"] = current_node
    if message:
        projected["message"] = message
    if metrics:
        projected["metrics"] = metrics
    if attempt is not None:
        projected["attempt"] = max(1, int(attempt))
    if isinstance(error, dict):
        projected["error"] = {
            key: error[key]
            for key in ("type", "message")
            if error.get(key)
        }
    elif error:
        projected["error"] = {"message": str(error)}
    return sanitize_event_payload(CHILD_PROGRESS_EVENT, projected)


def should_project_child_event(event_type: str) -> bool:
    """Whether a child event belongs in the parent's bounded live feed."""
    return event_type in _CHILD_PROJECTED_EVENTS


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
    if isinstance(snapshot, dict) and event_type != CHILD_PROGRESS_EVENT:
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
