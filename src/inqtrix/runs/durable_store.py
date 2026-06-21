"""Shared scaffolding for the durable job stores (runs and reindex).

:class:`~inqtrix.runs.postgres_store.PostgresRunStore` and
:class:`~inqtrix.runs.indexing_postgres.PostgresIndexingJobStore` are
two durable stores over the same shape: a sync public surface (routers
and job handles call from worker threads) over an async asyncpg layer.
The schema-agnostic plumbing is genuinely identical between them, so it
lives here once (Designprinzip 4), the same way
:class:`~inqtrix.runs.stream_queue.StreamJobQueue` and
:class:`~inqtrix.worker.loop.BaseWorkerLoop` share the queue and
worker-loop mechanics:

* the dedicated background event loop + ``_call`` sync->async bridge +
  ``close``/``_session`` (asyncpg pools are event-loop-affine, so each
  store owns ONE loop and funnels every DB op through it);
* the no-queue in-process dispatch loop (``_dispatch_locked`` /
  ``_run_worker``), a template method whose claim, handle, and
  terminal-on-success steps are subclass hooks;
* :class:`PollingJobSubscription`, the DB-polling SSE tail, parameterized
  by the terminal-event set and a thread label.

What stays per-store is everything schema-bearing: the SQL bodies, the
run store's share/audit/owner_sub/result surface, and the indexing
store's history-cap / active-collection / progress semantics.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from queue import Queue
from typing import TYPE_CHECKING, Any, Coroutine

from inqtrix.storage.db import build_session_factory, tenant_session
from inqtrix.urls import sanitize_error

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

log = logging.getLogger("inqtrix")

DEFAULT_TENANT = "default"

_SUBSCRIPTION_POLL_SECONDS = 0.3


@dataclass
class _LocalJob:
    """In-process execution state for one job (no-queue mode only)."""

    work: Any = field(repr=False, default=None)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class PollingJobSubscription:
    """Event subscription backed by short-interval database polling.

    Offers the exact attribute trio the SSE route consumes from the
    in-memory subscription: ``replay`` (already-stored events), ``queue``
    (live events as they land in Postgres — including those written by
    worker processes), and ``close()``. The terminal-event set and the
    poller thread label are the only per-job-kind differences, so they
    are constructor arguments — the seam :class:`StreamJobQueue` uses.

    The backing store must expose
    ``_events_after(entity_id, tenant_id, after_sequence) -> list[dict]``.
    """

    def __init__(
        self,
        store: Any,
        entity_id: str,
        tenant_id: str,
        replay: list[dict[str, Any]],
        *,
        terminal_events: frozenset[str],
        thread_label: str,
    ) -> None:
        self.entity_id = entity_id
        self.replay = replay
        self.queue: Queue = Queue()
        self._store = store
        self._tenant_id = tenant_id
        self._terminal_events = terminal_events
        self._stop = threading.Event()
        last_seq = replay[-1]["sequence"] if replay else 0
        replay_is_terminal = bool(
            replay and replay[-1]["type"] in terminal_events
        )
        if not replay_is_terminal:
            self._thread = threading.Thread(
                target=self._poll,
                args=(last_seq,),
                name=f"{thread_label}-{entity_id}",
                daemon=True,
            )
            self._thread.start()

    def _poll(self, last_seq: int) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                events = self._store._events_after(
                    self.entity_id, self._tenant_id, last_seq
                )
                failures = 0
            except Exception:  # noqa: BLE001 — retry transient outages
                failures += 1
                log.warning(
                    "Event-Poller fuer %s: Datenbankfehler (Versuch %d) — "
                    "naechster Versuch folgt.",
                    self.entity_id,
                    failures,
                    exc_info=failures == 1,
                )
                # A transient blip must not freeze the SSE stream for
                # good; back off and keep polling until the client
                # disconnects (close()) or the job ends.
                self._stop.wait(min(5.0, failures))
                continue
            for event in events:
                last_seq = event["sequence"]
                self.queue.put(event)
                if event["type"] in self._terminal_events:
                    return
            self._stop.wait(_SUBSCRIPTION_POLL_SECONDS)

    def close(self) -> None:
        """Stop the poller; idempotent."""
        self._stop.set()


class DurableJobStoreBase:
    """The schema-agnostic half of a durable Postgres job store.

    Subclasses set three class attributes (``_loop_thread_name``,
    ``_dispatch_thread_prefix``, ``_job_kind``) and implement the
    schema-bearing hooks (``_claim_db``, ``_make_handle``,
    ``_auto_complete``, plus the public ``fail`` the no-queue worker
    calls). The ``__init__`` here owns the engine, session factory,
    background loop, and the in-process dispatch bookkeeping; subclasses
    add their own sizing/retention/audit state after calling ``super()``.
    """

    _loop_thread_name: str = "inqtrix-jobs-db"
    _dispatch_thread_prefix: str = "inqtrix-job"
    _job_kind: str = "Durable job"

    def __init__(
        self,
        *,
        engine: "AsyncEngine",
        app_role: str,
        worker_id: str,
        queue: Any,
        max_concurrent: int,
    ) -> None:
        self._engine = engine
        self._session_factory = build_session_factory(engine)
        self._app_role = app_role
        self._worker_id = worker_id
        self._queue = queue
        self._max_concurrent = max_concurrent
        self._lock = threading.RLock()
        self._local: dict[str, _LocalJob] = {}
        self._pending: deque[str] = deque()
        self._running_count = 0
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name=self._loop_thread_name,
            daemon=True,
        )
        self._loop_thread.start()
        # In-process execution: queued/running rows from a previous
        # process are unrecoverable orphans (their work closures died
        # with it). Swept lazily on the first database touch; queue mode
        # never sweeps — workers own those rows.
        self._sweep_orphans = queue is None

    # -- async bridge ----------------------------------------------------- #

    @property
    def worker_id(self) -> str:
        """Identity this process stamps into ``claimed_by``."""
        return self._worker_id

    def _call(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run one coroutine on the store's loop and wait for it."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        """Dispose the engine and stop the background loop; idempotent."""
        if not self._loop.is_closed() and self._loop_thread.is_alive():
            self._call(self._engine.dispose())
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            if not self._loop_thread.is_alive():
                self._loop.close()

    def _session(self, tenant_id: str):
        return tenant_session(
            self._session_factory,
            tenant_id=tenant_id,
            app_role=self._app_role,
        )

    # -- in-process execution (no-queue mode) ----------------------------- #

    def _dispatch_locked(self) -> None:
        while self._running_count < self._max_concurrent and self._pending:
            entity_id = self._pending.popleft()
            local = self._local.get(entity_id)
            if local is None or local.work is None:
                continue
            claimed = self._call(
                self._claim_db(entity_id, DEFAULT_TENANT, allow_takeover=False)
            )
            if claimed is None:
                self._local.pop(entity_id, None)
                continue
            self._running_count += 1
            thread = threading.Thread(
                target=self._run_worker,
                args=(entity_id, local.work, local.cancel_event),
                name=f"{self._dispatch_thread_prefix}-{entity_id}",
                daemon=True,
            )
            thread.start()

    def _run_worker(
        self, entity_id: str, work: Any, cancel_event: threading.Event
    ) -> None:
        handle = self._make_handle(entity_id, cancel_event)
        try:
            work(handle)
            # Auto-complete safety net: the work body normally completes
            # the job itself, so this is usually a no-op (suppressed
            # warning) — the public complete()/fail() path keeps the
            # genuine fenced-out warning.
            self._auto_complete(entity_id)
        except Exception as exc:  # noqa: BLE001 — workers terminate cleanly
            log.exception("%s %s failed", self._job_kind, entity_id)
            self.fail(entity_id, sanitize_error(exc))
        finally:
            with self._lock:
                local = self._local.pop(entity_id, None)
                if local is not None:
                    local.work = None
                self._running_count = max(0, self._running_count - 1)
                self._dispatch_locked()

    # -- schema-bearing hooks (subclasses implement) ---------------------- #

    def _claim_db(self, entity_id: str, tenant_id: str, *, allow_takeover: bool):
        """Async CAS claim (queued -> running, attempt + 1)."""
        raise NotImplementedError

    def _make_handle(self, entity_id: str, cancel_event: threading.Event) -> Any:
        """Build the in-process (non-fenced) job handle."""
        raise NotImplementedError

    def _auto_complete(self, entity_id: str) -> None:
        """Terminal-on-success write with the no-op warning suppressed."""
        raise NotImplementedError

    def fail(self, entity_id: str, message: str, *, error_type: str = "server_error") -> Any:
        """Mark the job failed (public surface; used by the no-queue worker)."""
        raise NotImplementedError
