"""The run-store port: the surface routers and services program against.

Both backends — the in-memory :class:`~inqtrix.server.runs.RunStore`
(default) and :class:`~inqtrix.runs.postgres_store.PostgresRunStore`
(opt-in durability) — satisfy this Protocol structurally; the runs
router, :class:`~inqtrix.services.run_service.RunService`, and the
:class:`~inqtrix.server.container.AppContainer` cannot tell them
apart. The byte-level wire behaviour behind these methods is pinned by
``tests/contract/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from inqtrix.auth.permissions import SharePermission
    from inqtrix.auth.principal import UserContext
    from inqtrix.server.runs import RunWork


@dataclass(frozen=True)
class RunStoreMetrics:
    """A scrape-time snapshot of run-queue load (the ``/metrics`` source).

    Backend-neutral so the collector never branches on the store type:
    ``queued`` and ``active`` are the QUEUED and RUNNING counts (parked/
    waiting runs are excluded — they hold no slot, matching the per-user
    cap's accounting); ``capacity`` is the in-process concurrency ceiling
    for the memory backend, or ``None`` for the durable backend, where the
    worker fleet — not this API process — owns the execution slots.
    """

    queued: int
    active: int
    capacity: int | None


class RunSubscriptionPort(Protocol):
    """What the SSE route consumes from a subscription."""

    run_id: str
    replay: list[dict[str, Any]]
    queue: Any

    def close(self) -> None:
        """Detach the subscriber; idempotent."""
        ...


class RunStorePort(Protocol):
    """Public surface shared by every run-store backend."""

    def submit(
        self,
        *,
        question: str,
        stack_name: str,
        work: "RunWork",
        agent_overrides: dict[str, Any] | None = None,
        mode: str = "research",
        workspace_id: str | None = None,
        created_by_sub: str | None = None,
        created_by_tenant_id: str | None = None,
        request_payload: dict[str, Any] | None = None,
        kind: str = "standard",
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
        session_id: str | None = None,
        origin_key: str | None = None,
    ) -> dict[str, Any]:
        """Accept one run; returns the public summary (HTTP 202 body).

        The agent-tree kwargs (``kind``/``parent_run_id``/``root_run_id``/
        ``session_id``) default to the standard-run shape; summaries omit
        them entirely at defaults, so historical callers are untouched.
        """
        ...

    def get(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> dict[str, Any]:
        """Public summary; raises ``RunNotFound`` (denial == absence)."""
        ...

    def list(
        self,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> list[dict[str, Any]]:
        """Visible summaries, newest first (unbounded, internal reads)."""
        ...

    def list_session_runs(
        self,
        session_id: str,
        *,
        visible_to: "UserContext | None" = None,
    ) -> list[dict[str, Any]]:
        """Visible summaries of ONE agent session, OLDEST first.

        The session-context builder (plan K1) reads a session's prior
        turns at run start; oldest-first matches transcript order. No
        share-grant union — an agent session is a per-owner surface, so
        only the owner's own visibility applies (E5).
        """
        ...

    def session_owners(
        self, session_id: str
    ) -> set[tuple[str | None, str | None]]:
        """Recorded ``(tenant_id, subject)`` owners for a session id.

        This raw ownership probe is intentionally not visibility-filtered.
        It is used only to prevent a deleted session registry row from being
        reclaimed by a different principal while historical runs remain.
        """
        ...

    def list_page(
        self,
        *,
        limit: int,
        after: tuple[float, str] | None = None,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """One keyset page of visible summaries + next cursor.

        Newest-first over ``(created_at, run_id)``; ``after`` is the
        decoded cursor of the last row of the previous page. The HTTP
        list endpoint uses this so a long run history is not materialised
        whole on every poll.
        """
        ...

    def metrics_snapshot(self) -> RunStoreMetrics:
        """Cheap read of current queue load for the ``/metrics`` collector.

        Read-only and scrape-time only (no hot-path cost): the memory
        backend reads its in-process counters under the store lock; the
        durable backend runs a single grouped COUNT. See
        :class:`RunStoreMetrics` for the field semantics.
        """
        ...

    def result(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> dict[str, Any]:
        """Stored result payload; raises ``RunNotFound`` when absent."""
        ...

    def cancel(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> dict[str, Any]:
        """Cancel queued immediately / request cancel of running."""
        ...

    def cancel_tree(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Cancel one run tree and return its exact affected run ids.

        The ids are an internal post-transition handoff for reconciling
        agent control rows. The public HTTP response remains the first tuple
        item, byte-compatible with :meth:`cancel`.
        """
        ...

    def delete(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        requester_sub: str | None = None,
    ) -> None:
        """Permanently remove one terminal run; owner-only.

        The gate is creator identity, not share visibility (deletion is
        stronger than cancel). ``RunNotFound`` for unknown, non-owner, or
        cross-namespace ids (indistinct denial); ``RunActive`` for a
        non-terminal run. Not idempotent — a repeat delete raises
        ``RunNotFound``.
        """
        ...

    def subscribe(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> RunSubscriptionPort:
        """Event subscription with stored replay plus live tail."""
        ...

    def owner_sub(self, run_id: str) -> str | None:
        """The run's ``created_by_sub`` regardless of visibility.

        The share layer's owner resolver: authorization happens in the
        ShareService against this fact, so the lookup itself must not
        be visibility-gated. ``None`` for unknown runs AND for legacy
        pre-scoping rows (no recorded creator) alike — both are
        unshareable.
        """
        ...

    def emit(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        fence_attempt: int | None = None,
    ) -> None:
        """Append one event (worker-handle write path).

        Args:
            fence_attempt: Durable claim-attempt fence (see
                :meth:`mark_waiting`); in-process backends ignore it.
        """
        ...

    def complete(
        self,
        run_id: str,
        result: dict[str, Any],
        *,
        snapshot: dict[str, Any] | None = None,
        fence_attempt: int | None = None,
    ) -> None:
        """Store the result and mark completed (absorbing).

        Args:
            fence_attempt: Durable claim-attempt fence (see
                :meth:`mark_waiting`); in-process backends ignore it.
        """
        ...

    def fail(
        self,
        run_id: str,
        message: str,
        *,
        error_type: str = "server_error",
        fence_attempt: int | None = None,
    ) -> None:
        """Mark failed with a sanitized error (absorbing).

        Args:
            fence_attempt: Durable claim-attempt fence (see
                :meth:`mark_waiting`); in-process backends ignore it.
        """
        ...

    def mark_cancelled(
        self, run_id: str, *, reason: str, fence_attempt: int | None = None
    ) -> None:
        """Mark cancelled after the executor observed the cancel.

        Args:
            fence_attempt: Durable claim-attempt fence (see
                :meth:`mark_waiting`); in-process backends ignore it.
        """
        ...

    def mark_waiting(
        self, run_id: str, *, status: Any, fence_attempt: int | None = None
    ) -> None:
        """Park a RUNNING run in a waiting status (agent interrupt).

        Non-terminal: the run keeps its work/payload for a later
        :meth:`resume_run`; sweeps exclude waiting runs, only the
        waiting TTL may auto-cancel them (visible ``approval_timeout``).

        Args:
            fence_attempt: The claim attempt that owns this park (M5
                segments). The durable/queue backend records it so a
                reclaimed zombie cannot park a run the live attempt owns;
                the worker always supplies it. In-process backends have no
                zombie reclaim and ignore it, so it is optional — but it is
                part of the contract because the worker calls every store
                through this port.
        """
        ...

    def resume_run(self, run_id: str) -> dict[str, Any]:
        """Move a waiting run back to ``queued`` and re-dispatch it.

        Raises ``RunNotFound`` for unknown ids and ``RunActive`` when
        the run is not waiting (or nothing is retained to resume).
        Returns the public summary.
        """
        ...

    def children(self, run_id: str) -> list[dict[str, Any]]:
        """Direct-children summaries, newest first.

        NOT visibility-gated — the caller must have resolved the
        PARENT via :meth:`get` first; children inherit that access
        (plan rule R7).
        """
        ...

    def import_completed_run(
        self,
        *,
        run_id: str,
        question: str,
        stack_name: str,
        result: dict[str, Any],
        status: str = "completed",
        mode: str = "research",
        agent_overrides: dict[str, Any] | None = None,
        snapshot: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        created_at: float | None = None,
        workspace_id: str | None = None,
        created_by_sub: str | None = None,
        created_by_tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist an already-terminal run imported from a project file.

        Stores a completed report snapshot directly (no execution), scoped to
        the caller, so a loaded project's reports survive a reload and follow
        the user. The client ``run_id`` is kept when free (idempotent re-import
        of the caller's own run); a foreign-owned id collision allocates a fresh
        id instead of overwriting. ``created_at`` keeps the report's original
        date; the durable-retention clock starts at import time (so an old
        report is not pruned immediately), hence no ``finished_at`` parameter.
        Returns the public run summary.
        """
        ...
