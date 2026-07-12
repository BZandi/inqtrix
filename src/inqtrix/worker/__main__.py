"""Worker entry point: ``python -m inqtrix.worker`` / ``inqtrix-worker``.

Builds the same composition as the HTTP server (providers, strategies,
registry, resolver) plus worker-scoped Postgres stores and Valkey queues
bound to this worker's consumer name, then runs the claim loops until
SIGTERM/SIGINT. Requires ``INQTRIX_STORAGE_BACKEND=postgres`` and
``INQTRIX_QUEUE_BACKEND=valkey`` — the zero-infrastructure default
deployment has no worker.

One process runs two claim loops when knowledge is enabled: research
runs and durable reindex jobs share the same fencing/heartbeat/reclaim
machinery (:class:`~inqtrix.worker.loop.BaseWorkerLoop`) over separate
Valkey streams. With knowledge disabled, only the run loop runs (and the
absence is logged, never a silent no-op).

Graceful shutdown: stop claiming, drain in-flight jobs for a bounded
window, then exit. Jobs that did not finish are NOT cancelled — the
worker's heartbeat silence hands their dispatch entries to another
worker (at-least-once redelivery, absorbed by the job-row state machine).
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import signal
import socket
import threading

from inqtrix.logging_config import configure_logging
from inqtrix.providers import create_providers
from inqtrix.runs.postgres_store import PostgresRunStore
from inqtrix.runs.valkey_queue import ValkeyRunQueue
from inqtrix.server.container import build_container
from inqtrix.settings import Settings
from inqtrix.strategies import (
    create_default_strategies,
    resolve_claim_extract_model,
)
from inqtrix.worker.indexing_loop import IndexingWorkerLoop
from inqtrix.worker.loop import BaseWorkerLoop, WorkerLoop

log = logging.getLogger("inqtrix")

_DRAIN_SECONDS = 90.0


def main() -> None:
    """Run one worker process until SIGTERM/SIGINT."""
    configure_logging(
        enabled=os.getenv("INQTRIX_LOG_ENABLED", "").lower() == "true",
        level=os.getenv("INQTRIX_LOG_LEVEL", "INFO"),
        console=os.getenv("INQTRIX_LOG_CONSOLE", "").lower() == "true",
    )
    settings = Settings()
    if settings.storage.backend != "postgres":
        raise RuntimeError(
            "Der Worker verlangt INQTRIX_STORAGE_BACKEND=postgres — "
            "ohne durable Job-Zeilen gibt es nichts zu verarbeiten."
        )
    if settings.queue.backend != "valkey" or not settings.queue.valkey_url.strip():
        raise RuntimeError(
            "Der Worker verlangt INQTRIX_QUEUE_BACKEND=valkey und eine "
            "gesetzte INQTRIX_VALKEY_URL."
        )
    if not settings.storage.database_url.strip():
        raise RuntimeError(
            "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
            "INQTRIX_DATABASE_URL."
        )

    worker_id = (
        f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(4)}"
    )

    from inqtrix.storage.db import build_engine

    queue = ValkeyRunQueue(
        url=settings.queue.valkey_url, consumer=worker_id
    )
    store = PostgresRunStore(
        engine=build_engine(
            settings.storage.database_url,
            **settings.storage.pool_kwargs(),
        ),
        app_role=settings.storage.app_role,
        # queue=None here is CLAIM-MODE wiring (the ValkeyRunQueue goes
        # to the WorkerLoop, not the store), not in-process ownership —
        # without the explicit recover_orphans=False the store would
        # infer no-queue mode and blanket-fail every queued/running run
        # of the deployment on worker start ("Verwaister Run ...").
        # Crash recovery in queue mode is stream reclaim + the TTL
        # sweeps, never the orphan sweep.
        queue=None,
        # Submission and wake dispatch are separate from claim ownership:
        # children created inside a worker and parents woken by terminal
        # children must return to the shared stream, while queue=None above
        # remains the load-bearing claim-mode contract.
        dispatch_queue=queue,
        recover_orphans=False,
        max_concurrent=(
            settings.server.run_max_concurrent
            or settings.server.max_concurrent
        ),
        max_queue_size=settings.server.run_queue_max_size,
        # DURABLE retention (default 90d), NEVER the in-memory replay TTL
        # (run_completed_ttl_seconds, 300s): the worker's lazy cleanup
        # runs on every store access, and with the short TTL it deleted
        # every terminal run — reports, answers, child runs — five
        # minutes after completion (live P0 found in the tier E2E).
        completed_ttl_seconds=settings.server.run_durable_retention_seconds,
        worker_id=worker_id,
        # The per-user cap is an ADMISSION bound and fires only in
        # submit(), which the worker never calls (it claims and executes
        # already-admitted runs). Passed for construction symmetry with
        # the API store; held inertly here — re-checking a per-user cap on
        # an already-admitted run would be the wrong layer.
        max_concurrent_per_user=settings.server.run_max_concurrent_per_user,
    )
    # The reindex consumer runs in the SAME process when knowledge is
    # enabled: its store claims rows itself (queue=None worker-claim
    # mode, like the run store) over a separate reindex stream. Built
    # before the container so it can be injected (avoids the container
    # constructing an unused dispatch-mode store).
    index_store = None
    index_queue = None
    if settings.knowledge.enabled:
        from inqtrix.runs.indexing_postgres import PostgresIndexingJobStore
        from inqtrix.runs.indexing_queue import ValkeyIndexingQueue

        index_store = PostgresIndexingJobStore(
            engine=build_engine(
                settings.storage.database_url,
                **settings.storage.pool_kwargs(),
            ),
            app_role=settings.storage.app_role,
            # Claim-mode wiring like the run store above: never sweep.
            queue=None,
            recover_orphans=False,
            max_concurrent=settings.knowledge.reindex_max_concurrent,
            max_queue_size=settings.knowledge.reindex_queue_max_size,
            completed_ttl_seconds=(
                settings.knowledge.reindex_completed_ttl_seconds
            ),
            history_limit=settings.knowledge.reindex_history_limit,
            worker_id=worker_id,
        )
        index_queue = ValkeyIndexingQueue(
            url=settings.queue.valkey_url, consumer=worker_id
        )

    providers = create_providers(settings)
    strategies = create_default_strategies(
        settings.agent,
        llm=providers.llm,
        claim_extract_model=resolve_claim_extract_model(
            providers.llm,
            fallback=settings.models.effective_claim_extract_model,
        ),
        claim_extract_timeout=settings.agent.claim_extract_timeout,
    )
    container = build_container(
        providers=providers,
        strategies=strategies,
        settings=settings,
        semaphore_factory=lambda: asyncio.Semaphore(
            settings.server.max_concurrent
        ),
        run_store=store,
        indexing_store=index_store,
        # The workspace agent drives the permission/identity repositories
        # from sync worker threads via per-call asyncio.run; a pooled
        # asyncpg connection reused across those short-lived loops crashes
        # ("Future attached to a different loop"). NullPool makes the
        # shared platform engine loop-agnostic here (the API stays pooled).
        platform_persistence_null_pool=True,
    )

    # The worker has no auth provider, so the container's quota service
    # (gated on oidc) is never built here. Build it directly when quotas
    # are enabled: the worker meters by the subject persisted on the job
    # row, not by a live principal, so the auth-mode gate does not apply.
    # It records via the loop-agnostic NullPool store on the executor
    # thread (record_blocking) — never admits (admission stays in the API).
    quota_service = None
    if settings.quota.enabled:
        from inqtrix.services.quota_service import QuotaService
        from inqtrix.storage.quota_postgres import PostgresQuotaStore

        quota_service = QuotaService(
            store=PostgresQuotaStore(
                database_url=settings.storage.database_url,
                app_role=settings.storage.app_role,
            ),
            settings=settings.quota,
        )

    loops: list[BaseWorkerLoop] = [
        WorkerLoop(
            store=store,
            queue=queue,
            resolver=container.resolver,
            registry=container.registry,
            runtime=container.runtime,
            concurrency=settings.queue.worker_concurrency,
            max_attempts=settings.queue.worker_max_attempts,
            heartbeat_seconds=settings.queue.worker_heartbeat_seconds,
            claim_idle_seconds=settings.queue.worker_claim_idle_seconds,
            quota_service=quota_service,
        )
    ]
    if (
        index_store is not None
        and index_queue is not None
        and container.knowledge_service is not None
    ):
        loops.append(
            IndexingWorkerLoop(
                store=index_store,
                queue=index_queue,
                knowledge_service=container.knowledge_service,
                concurrency=settings.queue.worker_concurrency,
                max_attempts=settings.queue.worker_max_attempts,
                heartbeat_seconds=settings.queue.worker_heartbeat_seconds,
                claim_idle_seconds=settings.queue.worker_claim_idle_seconds,
                quota_service=quota_service,
            )
        )
    else:
        # Visibility over a silent no-op: a knowledge-disabled worker
        # simply has no reindex stream to consume.
        log.info(
            "Worker %s: Knowledge-Engine deaktiviert — kein "
            "Reindex-Consumer.",
            worker_id,
        )

    stop_event = threading.Event()

    def _request_stop(signum: int, _frame: object) -> None:
        log.info("Worker %s: Signal %d empfangen.", worker_id, signum)
        for active_loop in loops:
            active_loop.request_stop()
        stop_event.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    log.info(
        "Inqtrix worker starting | worker_id=%s | loops=%d | concurrency=%d "
        "| max_attempts=%d | heartbeat=%.0fs | claim_idle=%.0fs",
        worker_id,
        len(loops),
        settings.queue.worker_concurrency,
        settings.queue.worker_max_attempts,
        settings.queue.worker_heartbeat_seconds,
        settings.queue.worker_claim_idle_seconds,
    )
    # Reviewable connection budget (Sichtbarkeit): run store + optional
    # reindex store are this process's pooled engines.
    pooled_engines = 2 if index_store is not None else 1
    log.info(
        "Postgres-Verbindungsbudget | pool_size=%d max_overflow=%d | %d "
        "gepoolte Engines -> worst case %d persistente Verbindungen pro "
        "Worker-Prozess.",
        settings.storage.pool_size,
        settings.storage.pool_max_overflow,
        pooled_engines,
        pooled_engines
        * (settings.storage.pool_size + settings.storage.pool_max_overflow),
    )
    threads = [
        threading.Thread(
            target=active_loop.run_forever,
            name=f"inqtrix-claim-{index}",
            daemon=True,
        )
        for index, active_loop in enumerate(loops)
    ]
    for thread in threads:
        thread.start()
    # The signal handler must run on the main thread; block here until it
    # fires, then let each loop's run_forever observe its stop flag.
    stop_event.wait()
    for thread in threads:
        thread.join(timeout=_DRAIN_SECONDS)

    drained = all(active_loop.drain(_DRAIN_SECONDS) for active_loop in loops)
    log.info(
        "Inqtrix worker stopping | worker_id=%s | drained=%s",
        worker_id,
        drained,
    )
    if quota_service is not None:
        asyncio.run(quota_service.aclose())
    if not drained:
        # Executor threads are non-daemon and would block interpreter
        # exit for up to the full job duration; the orchestrator's
        # SIGKILL window plus stream redelivery is the designed
        # recovery path, so leave decisively.
        os._exit(0)


if __name__ == "__main__":
    main()
