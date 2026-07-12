"""Checkpointer wiring for the workspace-agent phase machine (E8/R5).

The checkpoint is ONLY a resumability cache: plans, approvals,
clarifications and artifacts live in the Inqtrix control tables (rule
R5), so wiping the checkpointer loses nothing but the ability to resume
in-flight runs — those fail loudly and are restartable.

Postgres deployments get the langgraph ``PostgresSaver`` on its OWN
psycopg pool (the checkpointer tables are library-owned via ``setup()``,
deliberately NOT managed by Alembic — noted in migration 0030). The plan
named the async saver; the runtime executes in sync worker threads, so
the sync saver over the same tables is the working equivalent
(documented deviation).

The volatile escape (``INQTRIX_AGENT_ALLOW_VOLATILE``) uses langgraph's
``InMemorySaver``: interrupted runs then die with the process — the
container logs a WARNING and capabilities report
``workspace_agent_durable: false`` so nothing degrades silently.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger("inqtrix")


class CheckpointerHandle:
    """One lazily-built checkpointer plus its cleanup surface.

    Attributes:
        durable: Whether checkpoints survive a process restart.
    """

    def __init__(self, *, database_url: str | None) -> None:
        self._database_url = database_url
        self._lock = threading.Lock()
        self._saver: Any = None
        self._pool: Any = None
        self.durable = database_url is not None

    def saver(self) -> Any:
        """The checkpointer instance (built on first use, then shared)."""
        with self._lock:
            if self._saver is not None:
                return self._saver
            if self._database_url is None:
                from langgraph.checkpoint.memory import InMemorySaver

                log.warning(
                    "Workspace-Agent laeuft mit FLUECHTIGEM Checkpointer "
                    "(INQTRIX_AGENT_ALLOW_VOLATILE): unterbrochene Laeufe "
                    "ueberleben keinen Neustart."
                )
                self._saver = InMemorySaver()
                return self._saver
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool

            # langgraph's saver speaks psycopg (sync); the app's asyncpg
            # URL prefix must be stripped. Own pool: the saver runs on
            # worker threads, never on the HTTP loop.
            conninfo = self._database_url.replace(
                "postgresql+asyncpg://", "postgresql://"
            )
            self._pool = ConnectionPool(
                conninfo,
                min_size=0,
                max_size=4,
                open=True,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            saver = PostgresSaver(self._pool)
            saver.setup()
            self._saver = saver
            return self._saver

    def delete_thread(self, thread_id: str) -> None:
        """Drop one run's checkpoints (terminal cleanup; best effort)."""
        try:
            saver = self.saver()
            saver.delete_thread(thread_id)
        except Exception:  # noqa: BLE001 — cleanup must never fail the run
            log.warning(
                "Checkpoint-Aufraeumen fuer Thread %s fehlgeschlagen.",
                thread_id,
                exc_info=True,
            )

    def close(self) -> None:
        """Dispose the pool; idempotent."""
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception:  # noqa: BLE001 — shutdown best effort
                    log.warning("Checkpointer-Pool-Close fehlgeschlagen.")
                self._pool = None
            self._saver = None


def build_checkpointer_handle(settings: Any) -> CheckpointerHandle | None:
    """Settings bridge deciding the E8 gate outcome.

    Returns ``None`` when the workspace agent must NOT register (no
    Postgres and no volatile escape) — the container surfaces that as
    ``features.workspace_agent: false``.
    """
    if not settings.agent_platform.enabled:
        return None
    if settings.storage.backend == "postgres" and settings.storage.database_url.strip():
        try:
            import langgraph.checkpoint.postgres  # noqa: F401
            import psycopg_pool  # noqa: F401
        except ImportError:
            log.warning(
                "Workspace-Agent deaktiviert: das 'agent'-Extra ist "
                "nicht installiert (uv sync --extra agent) — "
                "features.workspace_agent bleibt false."
            )
            return None
        return CheckpointerHandle(database_url=settings.storage.database_url)
    if settings.agent_platform.allow_volatile:
        try:
            import langgraph.checkpoint.memory  # noqa: F401
        except ImportError:
            log.warning(
                "Workspace-Agent deaktiviert: langgraph-Checkpointer "
                "nicht importierbar."
            )
            return None
        return CheckpointerHandle(database_url=None)
    return None
