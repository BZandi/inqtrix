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

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from inqtrix.auth.principal import UserContext
    from inqtrix.server.runs import RunWork


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
    ) -> dict[str, Any]:
        """Accept one run; returns the public summary (HTTP 202 body)."""
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
    ) -> list[dict[str, Any]]:
        """Visible summaries, newest first."""
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
    ) -> dict[str, Any]:
        """Cancel queued immediately / request cancel of running."""
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
    ) -> None:
        """Append one event (worker-handle write path)."""
        ...

    def complete(
        self,
        run_id: str,
        result: dict[str, Any],
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Store the result and mark completed (absorbing)."""
        ...

    def fail(
        self,
        run_id: str,
        message: str,
        *,
        error_type: str = "server_error",
    ) -> None:
        """Mark failed with a sanitized error (absorbing)."""
        ...

    def mark_cancelled(self, run_id: str, *, reason: str) -> None:
        """Mark cancelled after the executor observed the cancel."""
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
