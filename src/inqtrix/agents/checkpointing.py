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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

log = logging.getLogger("inqtrix")

_ASYNCPG_ONLY_QUERY_PARAMS = frozenset(
    {
        "prepared_statement_cache_size",
        "statement_cache_size",
    }
)
"""URL query parameters only the asyncpg dialect understands. libpq
rejects unknown URI parameters outright ("invalid URI query parameter"),
so they must not survive the scheme swap. The bundled pgbouncer URL
appends ``prepared_statement_cache_size=0`` — asyncpg's transaction-
pooling mitigation; the psycopg pool gets the equivalent through its
``prepare_threshold=0`` connect kwarg instead."""


def _psycopg_conninfo(database_url: str) -> str:
    """Translate the app's SQLAlchemy/asyncpg URL into libpq conninfo.

    Swaps the dialect scheme and drops asyncpg-only query parameters;
    everything else (host, credentials, libpq parameters such as
    ``sslmode``) passes through unchanged. Without the drop, the first
    agent run on a pgbouncer-fronted deployment fails at pool open while
    the Agent Desk stays visible — the gate never connects.
    """
    conninfo = database_url.replace("postgresql+asyncpg://", "postgresql://")
    parts = urlsplit(conninfo)
    if not parts.query:
        return conninfo
    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in _ASYNCPG_ONLY_QUERY_PARAMS
    ]
    return urlunsplit(parts._replace(query=urlencode(kept)))


class CheckpointerHandle:
    """One lazily-built checkpointer plus its cleanup surface.

    Attributes:
        durable: Whether checkpoints survive a process restart.
    """

    def __init__(
        self, *, database_url: str | None, max_connections: int = 4
    ) -> None:
        self._database_url = database_url
        self._lock = threading.Lock()
        self._saver: Any = None
        self._pool: Any = None
        self.durable = database_url is not None
        # Server connections this handle may hold
        # (INQTRIX_AGENT_CHECKPOINTER_POOL_SIZE). Its own pool, on its
        # own driver: no engine count and no engine disposal reaches it,
        # so the number has to be readable from outside to appear in the
        # process connection budget at all. No clamp here: the settings
        # path is ge=1-validated loudly, and the volatile path passes 0
        # deliberately -- an InMemorySaver holds no server connections.
        self.max_connections = int(max_connections)

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
            # URL must be translated (scheme + asyncpg-only parameters).
            # Own pool: the saver runs on worker threads, never on the
            # HTTP loop.
            conninfo = _psycopg_conninfo(self._database_url)
            self._pool = ConnectionPool(
                conninfo,
                min_size=0,
                max_size=self.max_connections,
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

    def delete_thread_strict(self, thread_id: str) -> None:
        """Delete one checkpoint lineage or raise so a durable saga can retry."""

        saver = self.saver()
        saver.delete_thread(thread_id)
        remaining = saver.get_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
        if remaining is not None:
            raise RuntimeError("checkpoint lineage remains after deletion")

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
        log.info(
            "Workspace-Agent deaktiviert: INQTRIX_AGENT_ENABLED=false — "
            "features.workspace_agent bleibt false."
        )
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
        return CheckpointerHandle(
            database_url=settings.storage.database_url,
            max_connections=settings.agent_platform.checkpointer_pool_size,
        )
    if settings.agent_platform.allow_volatile:
        try:
            import langgraph.checkpoint.memory  # noqa: F401
        except ImportError:
            log.warning(
                "Workspace-Agent deaktiviert: langgraph-Checkpointer "
                "nicht importierbar."
            )
            return None
        return CheckpointerHandle(database_url=None, max_connections=0)
    log.warning(
        "Workspace-Agent deaktiviert: kein durabler Checkpointer — "
        "INQTRIX_STORAGE_BACKEND ist nicht 'postgres' (oder "
        "INQTRIX_DATABASE_URL ist leer) und INQTRIX_AGENT_ALLOW_VOLATILE "
        "ist false. Fuer den Agent Desk INQTRIX_STORAGE_BACKEND=postgres "
        "setzen oder das volatile Opt-in aktivieren — "
        "features.workspace_agent bleibt false."
    )
    return None
