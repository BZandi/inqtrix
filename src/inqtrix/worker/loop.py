"""The worker loop: claim, execute, heartbeat, reclaim, reconcile.

Liveness/correctness model (sized by the 2026-06 research pass):

* At-least-once delivery + guarded job-row transitions = idempotent
  execution. The claim is a compare-and-set (``queued -> running``,
  ``attempt + 1``); terminal writes are fenced by
  ``(claimed_by, attempt)`` so a reclaimed zombie cannot overwrite the
  second attempt's result.
* The OWNING worker heartbeats its in-flight stream entries
  (``XCLAIM JUSTID``) so the reclaim idle threshold detects crashed
  workers within seconds-of-heartbeats, never the job duration.
* Takeover (claiming a ``running`` row) is allowed ONLY for
  redeliveries — fresh duplicate messages (e.g. from the reconciler)
  must not steal a healthy execution.
* Ack order is load-bearing: terminal state commits to Postgres
  FIRST, the stream entry is acked SECOND; a crash in the gap causes
  one redelivery that the state machine absorbs.
* Cross-process cancellation polls the job row's ``cancel_requested``
  column (the source of truth) and flips a local ``threading.Event``
  the work observes at node/document boundaries.
* Graceful shutdown stops claiming and drains in-flight jobs without
  cancelling them; on drain timeout the process exits and the
  heartbeat silence hands the jobs to another worker.

The claim/heartbeat/reclaim/reconcile/cancel machinery is generic over
the job kind and lives in :class:`BaseWorkerLoop`; the run worker and
the reindex worker (:mod:`inqtrix.worker.indexing_loop`) differ only in
the execution body and a few store/queue method names, supplied via the
subclass hooks. Every lifecycle decision logs a marker (claim, skip,
takeover, dead-letter, cancel observed, fenced write discarded) —
Designprinzip "Sichtbarkeit > Cleverness".
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from inqtrix.auth.principal import (
    ANONYMOUS_PRINCIPAL,
    STATIC_PRINCIPAL,
    Principal,
)
from inqtrix.core.results import RunRequest
from inqtrix.execution_failures import terminate_native_run
from inqtrix.quota.models import QuotaSubject
from inqtrix.server.runs import RunHandle
from inqtrix.services.run_service import execute_run_request

if TYPE_CHECKING:
    from inqtrix.core.algorithms import AlgorithmRegistry
    from inqtrix.core.context import RuntimeContext
    from inqtrix.runs.postgres_store import ClaimedRun, PostgresRunStore
    from inqtrix.runs.valkey_queue import QueuedJob, ValkeyRunQueue
    from inqtrix.services.agent_context import AgentContextResolver
    from inqtrix.services.quota_service import QuotaService

log = logging.getLogger("inqtrix")

_CANCEL_POLL_SECONDS = 2.0
_RECONCILE_SECONDS = 60.0
_RECONCILE_MIN_AGE_SECONDS = 120.0
_RECONCILE_COOLDOWN_SECONDS = 600.0
_CLAIM_BLOCK_MS = 5_000
_ERROR_BACKOFF_SECONDS = 5.0

TJob = TypeVar("TJob")
TClaimed = TypeVar("TClaimed")


@dataclass
class _ActiveJob:
    """Book-keeping for one in-flight execution."""

    job: Any
    cancel_event: threading.Event
    future: Future | None = None
    successor: Any | None = None
    """One queued follow-up dispatch held while a parked segment unwinds."""
    handoff_in_progress: bool = False
    """The successor occupies this slot while its row claim is landing."""


class BaseWorkerLoop(Generic[TJob, TClaimed]):
    """Generic claim-and-execute loop over a Valkey job stream.

    Subclasses supply the execution body and the few store/queue calls
    whose names differ by job kind (the dispatch id field, the stale-row
    feed, the cancel-request poll, the re-enqueue). Everything else —
    fencing, ack ordering, heartbeat, reclaim, self-reclaim handling,
    dead-lettering, reconcile, graceful drain — is shared.

    Args:
        store: Durable job store (claims, events, terminal writes). Must
            expose ``worker_id``, ``claim_for_execution`` and ``fail``.
        queue: Valkey queue bound to this worker's consumer name.
        concurrency: Parallel executions in this process.
        max_attempts: Delivery budget before dead-lettering.
        heartbeat_seconds: Idle-reset interval for in-flight entries.
        claim_idle_seconds: Reclaim threshold for entries whose owner
            stopped heartbeating.
        thread_prefix: Executor thread name prefix (for log clarity).
    """

    def __init__(
        self,
        *,
        store: Any,
        queue: Any,
        concurrency: int,
        max_attempts: int,
        heartbeat_seconds: float,
        claim_idle_seconds: float,
        thread_prefix: str = "inqtrix-job",
    ) -> None:
        self._store = store
        self._queue = queue
        self._concurrency = concurrency
        self._max_attempts = max_attempts
        self._heartbeat_seconds = heartbeat_seconds
        self._claim_idle_seconds = claim_idle_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix=thread_prefix
        )
        self._active: dict[str, _ActiveJob] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._terminated = threading.Event()
        self._reenqueued: dict[str, float] = {}
        self._last_reclaim = 0.0
        self._last_reconcile = 0.0

    # -- subclass hooks --------------------------------------------------- #

    def _entity_id(self, job: TJob) -> str:
        """The durable id carried by *job* (``run_id`` / ``job_id``)."""
        raise NotImplementedError

    def _execute(
        self, job: TJob, claimed: TClaimed, cancel_event: threading.Event
    ) -> None:
        """Run the claimed job to a terminal state and ack on landing."""
        raise NotImplementedError

    def _stale_dispatch(self) -> list[tuple[str, str]]:
        """``(entity_id, tenant_id)`` rows stuck queued (reconcile feed)."""
        raise NotImplementedError

    def _cancel_requested(self, watched: dict[str, str]) -> set[str]:
        """Subset of watched ``{entity_id: tenant_id}`` with a pending cancel."""
        raise NotImplementedError

    def _enqueue_dispatch(self, entity_id: str, tenant_id: str) -> None:
        """Re-enqueue one dispatch message (reconciler)."""
        raise NotImplementedError

    def _is_successor_dispatch(self, job: TJob) -> bool:
        """Whether *job* is a real queued successor of an active segment.

        Durable reindex jobs never park and therefore keep the historical
        duplicate-ACK behaviour. The run worker overrides this hook and reads
        the authoritative row status so a resume/wake message is not mistaken
        for a reconciler duplicate while the previous segment is unwinding.
        """
        del job
        return False

    # -- lifecycle -------------------------------------------------------- #

    def request_stop(self) -> None:
        """Stop claiming new jobs; in-flight jobs keep executing."""
        log.info(
            "Worker %s: Shutdown angefordert — keine neuen Claims.",
            self._store.worker_id,
        )
        self._stop.set()

    def drain(self, timeout: float) -> bool:
        """Wait up to *timeout* seconds for in-flight jobs to finish.

        Returns:
            ``True`` when everything finished; ``False`` when jobs are
            still executing — the caller exits anyway and heartbeat
            silence hands those jobs to another worker.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._active:
                    self._terminated.set()
                    return True
            time.sleep(0.5)
        with self._lock:
            remaining = list(self._active)
        if remaining:
            log.warning(
                "Worker %s: Drain-Timeout — %d Jobs werden nach "
                "Heartbeat-Stille von einem anderen Worker uebernommen: %s",
                self._store.worker_id,
                len(remaining),
                ", ".join(remaining),
            )
        self._terminated.set()
        return not remaining

    def run_forever(self) -> None:
        """Main loop; returns after :meth:`request_stop`."""
        self._queue.ensure_group()
        # Own-PEL drain only finds entries when the consumer name is
        # stable across restarts (container hostname without the pid);
        # with the default boot-unique name, crash recovery runs via
        # the idle-based reclaim instead — this call is then a no-op.
        for job in self._queue.claim_pending():
            self._start(job, takeover=True)

        heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="inqtrix-heartbeat", daemon=True
        )
        cancel_poll = threading.Thread(
            target=self._cancel_loop, name="inqtrix-cancel-poll", daemon=True
        )
        heartbeat.start()
        cancel_poll.start()

        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001 — survive transient outages
                log.warning(
                    "Worker %s: Claim-Schleife stolpert ueber einen "
                    "transienten Fehler — naechster Versuch in %.0fs.",
                    self._store.worker_id,
                    _ERROR_BACKOFF_SECONDS,
                    exc_info=True,
                )
                self._stop.wait(_ERROR_BACKOFF_SECONDS)

    def _tick(self) -> None:
        """One claim-loop iteration (reclaim, reconcile, claim new)."""
        now = time.monotonic()
        with self._lock:
            has_capacity = len(self._active) < self._concurrency
        if (
            now - self._last_reclaim >= self._heartbeat_seconds
            and has_capacity
        ):
            # Reclaim only with free capacity: an over-claimed job
            # would sit "running" in the database while parked in this
            # worker's queue, invisible to other reclaimers.
            self._last_reclaim = now
            for job in self._queue.reclaim(
                min_idle_ms=int(self._claim_idle_seconds * 1000),
                count=1,
            ):
                log.warning(
                    "Worker %s uebernimmt Job %s (Owner ohne "
                    "Heartbeat, Zustellung %d).",
                    self._store.worker_id,
                    self._entity_id(job),
                    job.delivery_count,
                )
                self._start(job, takeover=True)
        if now - self._last_reconcile >= _RECONCILE_SECONDS:
            self._last_reconcile = now
            self._reconcile()
        with self._lock:
            has_capacity = len(self._active) < self._concurrency
        if not has_capacity:
            time.sleep(0.5)
            return
        for job in self._queue.claim_new(block_ms=_CLAIM_BLOCK_MS):
            self._start(job, takeover=job.delivery_count > 1)

    def _reconcile(self) -> None:
        """Re-enqueue queued rows whose dispatch message got lost.

        Re-enqueues each entity at most once per cooldown window: under
        sustained backlog every stale-but-healthy queued row would
        otherwise receive one duplicate per worker per minute (the
        duplicates are absorbed by ack-on-duplicate and the guarded
        claim, but flooding the stream helps nobody).
        """
        stale = self._stale_dispatch()
        now = time.monotonic()
        for entity_id, tenant_id in stale:
            last_sent = self._reenqueued.get(entity_id)
            if (
                last_sent is not None
                and now - last_sent < _RECONCILE_COOLDOWN_SECONDS
            ):
                continue
            self._reenqueued[entity_id] = now
            log.warning(
                "Reconciler: Job %s haengt im Status queued — "
                "Dispatch-Nachricht wird erneut gesendet.",
                entity_id,
            )
            self._enqueue_dispatch(entity_id, tenant_id)
        if len(self._reenqueued) > 1_000:
            cutoff = now - _RECONCILE_COOLDOWN_SECONDS
            self._reenqueued = {
                entity_id: sent
                for entity_id, sent in self._reenqueued.items()
                if sent >= cutoff
            }

    def _start(self, job: TJob, *, takeover: bool) -> None:
        entity_id = self._entity_id(job)
        while True:
            with self._lock:
                active = self._active.get(entity_id)
                active_message_id = active.job.message_id if active else None
                successor_message_id = (
                    active.successor.message_id
                    if active is not None and active.successor is not None
                    else None
                )
                handoff_in_progress = bool(
                    active is not None and active.handoff_in_progress
                )
            if active is None:
                break
            if job.message_id in {
                active_message_id,
                successor_message_id,
            }:
                # Self-reclaim: XAUTOCLAIM does not exclude the calling
                # consumer, so after a heartbeat gap a worker can reclaim its
                # OWN active or held-successor entry. Acking either would
                # destroy the job's crash-recovery entry.
                log.warning(
                    "Worker %s: Self-Reclaim fuer aktiven Job %s nach "
                    "Heartbeat-Stille — Eintrag bleibt gehalten.",
                    self._store.worker_id,
                    entity_id,
                )
                return
            if handoff_in_progress or successor_message_id is not None:
                # One held successor is sufficient. Extra messages are true
                # duplicates and must not idle in the PEL until they can steal
                # the queued/running successor.
                log.info(
                    "Worker %s: zusaetzlicher Duplikat-Dispatch fuer Job %s "
                    "bestaetigt.",
                    self._store.worker_id,
                    entity_id,
                )
                self._queue.ack(job.message_id)
                return
            # A queued row while the previous segment is still active is not
            # a reconciler duplicate: it is the durable resume/child-wake
            # successor. The status read deliberately happens outside the
            # worker lock. If it fails, the exception propagates and the
            # message remains unacked for redelivery — never degrade an
            # uncertain successor into an ACK.
            if not self._is_successor_dispatch(job):
                log.info(
                    "Worker %s: Duplikat-Dispatch fuer aktiven Job %s "
                    "bestaetigt.",
                    self._store.worker_id,
                    entity_id,
                )
                self._queue.ack(job.message_id)
                return
            with self._lock:
                current = self._active.get(entity_id)
                if current is not active:
                    # The old execution completed during the row-status read;
                    # re-evaluate and claim normally instead of losing the
                    # successor in that handoff window.
                    continue
                if current.successor is None and not current.handoff_in_progress:
                    current.successor = job
                    log.info(
                        "Worker %s: Folge-Dispatch fuer geparkten Job %s "
                        "bis zum Segment-Abschluss gehalten.",
                        self._store.worker_id,
                        entity_id,
                    )
                    return
            # A successor won the lock between the two reads. Keep the first
            # one and acknowledge this redundant message.
            self._queue.ack(job.message_id)
            return
        if job.delivery_count > self._max_attempts:
            self._store.fail(
                entity_id,
                "Maximale Anzahl Ausfuehrungsversuche erreicht.",
                error_type="max_retries_exceeded",
            )
            self._queue.dead_letter(job, reason="max_attempts_exceeded")
            return
        claimed = self._store.claim_for_execution(
            entity_id, job.tenant_id, allow_takeover=takeover
        )
        if claimed is None:
            log.info(
                "Worker %s: Job %s nicht uebernehmbar (terminal, "
                "fehlend oder aktiv bei anderem Worker) — Nachricht "
                "bestaetigt.",
                self._store.worker_id,
                entity_id,
            )
            self._queue.ack(job.message_id)
            return
        # Register BEFORE submitting: a fast job's finally-pop must
        # find the entry, or it would linger in the active set forever.
        cancel_event = threading.Event()
        entry = _ActiveJob(job=job, cancel_event=cancel_event)
        with self._lock:
            self._active[entity_id] = entry
        try:
            entry.future = self._executor.submit(
                self._execute, job, claimed, cancel_event
            )
        except BaseException:
            with self._lock:
                self._active.pop(entity_id, None)
            raise

    def _finish_active(self, job: TJob, *, allow_successor: bool) -> None:
        """Release one execution slot and hand off its held successor.

        A successor is activated only after the old stream entry was ACKed.
        Otherwise the old entry could later be redelivered with takeover
        permission and fence out the newly started segment. The successor is
        kept as an active placeholder during its claim so capacity and drain
        never observe a false empty slot.
        """
        entity_id = self._entity_id(job)
        placeholder: _ActiveJob | None = None
        with self._lock:
            active = self._active.get(entity_id)
            if active is None or active.job.message_id != job.message_id:
                return
            successor = active.successor
            if successor is None or not allow_successor:
                self._active.pop(entity_id, None)
                if successor is not None:
                    log.warning(
                        "Worker %s: Folge-Dispatch fuer Job %s bleibt "
                        "unbestaetigt, weil der alte Stream-Eintrag nicht "
                        "sicher freigegeben wurde — Redelivery uebernimmt.",
                        self._store.worker_id,
                        entity_id,
                    )
                return
            placeholder = _ActiveJob(
                job=successor,
                cancel_event=threading.Event(),
                handoff_in_progress=True,
            )
            self._active[entity_id] = placeholder
        self._activate_successor(entity_id, placeholder)

    def _activate_successor(
        self, entity_id: str, placeholder: _ActiveJob
    ) -> None:
        """Claim and submit a pre-registered successor placeholder."""
        job = placeholder.job
        try:
            if job.delivery_count > self._max_attempts:
                self._store.fail(
                    entity_id,
                    "Maximale Anzahl Ausfuehrungsversuche erreicht.",
                    error_type="max_retries_exceeded",
                )
                self._queue.dead_letter(
                    job, reason="max_attempts_exceeded"
                )
                with self._lock:
                    if self._active.get(entity_id) is placeholder:
                        self._active.pop(entity_id, None)
                return
            claimed = self._store.claim_for_execution(
                entity_id,
                job.tenant_id,
                # A successor is a fresh queued segment. A crash during this
                # handoff leaves the message for normal reclaim, whose regular
                # _start call supplies takeover=True later.
                allow_takeover=False,
            )
            if claimed is None:
                self._queue.ack(job.message_id)
                with self._lock:
                    if self._active.get(entity_id) is placeholder:
                        self._active.pop(entity_id, None)
                return
            # The row is RUNNING now. Open the placeholder for the NEXT
            # generation before submitting: ThreadPoolExecutor may start a
            # very fast segment before submit() returns, and that segment can
            # park/enqueue its own successor immediately. Leaving the handoff
            # flag set through submit would misclassify that legitimate next
            # successor as an extra duplicate.
            with self._lock:
                if self._active.get(entity_id) is placeholder:
                    placeholder.handoff_in_progress = False
            future = self._executor.submit(
                self._execute, job, claimed, placeholder.cancel_event
            )
            with self._lock:
                if self._active.get(entity_id) is placeholder:
                    placeholder.future = future
        except BaseException:
            with self._lock:
                if self._active.get(entity_id) is placeholder:
                    self._active.pop(entity_id, None)
            # The successor entry remains unacked. Surface the failure, but do
            # not let an unobserved executor-finally exception disappear.
            log.exception(
                "Worker %s: Folge-Dispatch fuer Job %s konnte nicht "
                "aktiviert werden — Redelivery uebernimmt.",
                self._store.worker_id,
                entity_id,
            )

    def _heartbeat_message_ids(self) -> list[str]:
        """Snapshot active and held-successor stream ids under one lock."""
        with self._lock:
            message_ids: list[str] = []
            for active in self._active.values():
                message_ids.append(active.job.message_id)
                if active.successor is not None:
                    message_ids.append(active.successor.message_id)
            return message_ids

    def _heartbeat_loop(self) -> None:
        # Gated on _terminated, NOT _stop: heartbeats must continue
        # through the drain window or every draining job would be
        # reclaimed and double-executed by another worker.
        while not self._terminated.wait(self._heartbeat_seconds):
            message_ids = self._heartbeat_message_ids()
            if not message_ids:
                continue
            try:
                self._queue.heartbeat(message_ids)
            except Exception:  # noqa: BLE001 — heartbeat loss must be visible
                log.warning(
                    "Worker %s: Heartbeat fehlgeschlagen — laufende Jobs "
                    "koennen von anderen Workern uebernommen werden.",
                    self._store.worker_id,
                    exc_info=True,
                )

    def _cancel_loop(self) -> None:
        # Gated on _terminated like the heartbeat: users must still be
        # able to cancel jobs that are draining.
        while not self._terminated.wait(_CANCEL_POLL_SECONDS):
            with self._lock:
                watched = {
                    self._entity_id(active.job): active.job.tenant_id
                    for active in self._active.values()
                    if not active.cancel_event.is_set()
                }
            if not watched:
                continue
            try:
                cancelled = self._cancel_requested(watched)
            except Exception:  # noqa: BLE001 — poller must not die silently
                log.warning(
                    "Cancel-Poller: Datenbankabfrage fehlgeschlagen.",
                    exc_info=True,
                )
                continue
            for entity_id in cancelled:
                log.info(
                    "Worker %s: Abbruch fuer Job %s beobachtet.",
                    self._store.worker_id,
                    entity_id,
                )
                with self._lock:
                    active = self._active.get(entity_id)
                if active is not None:
                    active.cancel_event.set()


class FencedRunHandle(RunHandle):
    """Run handle whose terminal writes carry the claim fence.

    The fence ``(claimed_by, attempt)`` makes a zombie worker's late
    writes a visible no-op instead of overwriting the result of the
    attempt that superseded it.
    """

    def __init__(
        self,
        store: "PostgresRunStore",
        run_id: str,
        cancel_event: threading.Event,
        attempt: int,
    ) -> None:
        super().__init__(store, run_id, cancel_event)
        self._fence_attempt = attempt
        self.terminal_landed = False
        """Whether this attempt's terminal write actually landed —
        ``False`` means a superseding attempt fenced it out and the
        dispatch message must NOT be acked by this attempt."""

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Emit one event, fenced to this claim attempt."""
        self._store.emit(
            self.run_id,
            event_type,
            payload or {},
            fence_attempt=self._fence_attempt,
        )

    def complete(
        self,
        result: dict[str, Any],
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Mark the run completed, fenced to this claim attempt."""
        self.terminal_landed = self._store.complete(
            self.run_id,
            result,
            snapshot=snapshot,
            fence_attempt=self._fence_attempt,
        )

    def fail(self, message: str, *, error_type: str = "server_error") -> None:
        """Mark the run failed, fenced to this claim attempt."""
        self.terminal_landed = self._store.fail(
            self.run_id,
            message,
            error_type=error_type,
            fence_attempt=self._fence_attempt,
        )

    def cancel(self, reason: str = "cancelled") -> None:
        """Mark the run cancelled, fenced to this claim attempt."""
        self.terminal_landed = self._store.mark_cancelled(
            self.run_id, reason=reason, fence_attempt=self._fence_attempt
        )

    def wait(self, status: Any) -> None:
        """Park the run, fenced to this claim attempt (M5 segments).

        The fence keeps a reclaimed zombie from parking a run the live
        attempt owns. After a successful park the worker loop ACKS the
        dispatch message like a terminal state — the resume re-enqueues
        a FRESH message and any worker continues from the persisted
        payload + checkpoint.
        """
        self._store.mark_waiting(
            self.run_id, status=status, fence_attempt=self._fence_attempt
        )
        self.parked = True


class WorkerLoop(BaseWorkerLoop["QueuedJob", "ClaimedRun"]):
    """Claim-and-execute loop for research runs.

    Args:
        store: Postgres run store (claims, events, terminal writes).
        queue: Valkey queue bound to this worker's consumer name.
        resolver: Agent context resolver — the worker re-resolves the
            persisted request body exactly like the HTTP path, so
            stacks/overrides/mode behave identically.
        registry: Algorithm registry shared with the API surface.
        runtime: App-level runtime context.
        concurrency: Parallel executions in this process.
        max_attempts: Delivery budget before dead-lettering.
        heartbeat_seconds: Idle-reset interval for in-flight entries.
        claim_idle_seconds: Reclaim threshold for entries whose owner
            stopped heartbeating.
    """

    def __init__(
        self,
        *,
        store: "PostgresRunStore",
        queue: "ValkeyRunQueue",
        resolver: "AgentContextResolver",
        registry: "AlgorithmRegistry",
        runtime: "RuntimeContext",
        concurrency: int,
        max_attempts: int,
        heartbeat_seconds: float,
        claim_idle_seconds: float,
        quota_service: "QuotaService | None" = None,
    ) -> None:
        super().__init__(
            store=store,
            queue=queue,
            concurrency=concurrency,
            max_attempts=max_attempts,
            heartbeat_seconds=heartbeat_seconds,
            claim_idle_seconds=claim_idle_seconds,
            thread_prefix="inqtrix-job",
        )
        self._resolver = resolver
        self._registry = registry
        self._runtime = runtime
        # Metering for runs that execute off the API process. The worker
        # has no live principal, so token recording uses the subject
        # reconstructed from the claimed run row (see _execute).
        self._quota_service = quota_service

    def _entity_id(self, job: "QueuedJob") -> str:
        return job.run_id

    def _stale_dispatch(self) -> list[tuple[str, str]]:
        return self._store.stale_queued_runs(
            older_than_seconds=_RECONCILE_MIN_AGE_SECONDS
        )

    def _cancel_requested(self, watched: dict[str, str]) -> set[str]:
        return self._store.cancel_requested_runs(watched)

    def _enqueue_dispatch(self, entity_id: str, tenant_id: str) -> None:
        self._queue.enqueue(run_id=entity_id, tenant_id=tenant_id)

    def _is_successor_dispatch(self, job: "QueuedJob") -> bool:
        """Recognize a queued resume/wake behind an unwinding segment."""
        return (
            self._store.dispatch_status(job.run_id, job.tenant_id)
            == "queued"
        )

    def _execute(
        self,
        job: "QueuedJob",
        claimed: "ClaimedRun",
        cancel_event: threading.Event,
    ) -> None:
        handle = FencedRunHandle(
            self._store, job.run_id, cancel_event, claimed.attempt
        )
        old_message_acked = False
        try:
            try:
                payload = claimed.request_payload
                if not payload:
                    raise RuntimeError(
                        "Run ohne gespeicherten Auftrag (request_payload "
                        "leer) — vor der Queue-Umstellung erstellt?"
                    )
                resolved = self._resolver.resolve(payload["body"])
                requested_stack = payload.get("body", {}).get("stack")
                if requested_stack and not resolved.stack_name:
                    # The API resolved this run against a named stack
                    # the worker does not know — executing on default
                    # providers would silently swap the model/search
                    # configuration underneath the user.
                    raise RuntimeError(
                        f"Stack {requested_stack!r} ist diesem Worker "
                        "nicht konfiguriert — Worker und API muessen "
                        "dieselbe Stack-Konfiguration teilen."
                    )
                algorithm = self._registry.get(resolved.mode)
                body = payload.get("body", {}) or {}
                run_request = RunRequest(
                    mode=resolved.mode,
                    question=payload["question"],
                    history=payload.get("history", ""),
                    messages=payload.get("messages", []),
                    agent_overrides=resolved.agent_overrides,
                    knowledge_filters=resolved.knowledge_filters,
                    autonomy=str(body.get("autonomy", "") or ""),
                    session_id=str(body.get("session_id", "") or ""),
                    document_id=str(body.get("document_id", "") or ""),
                    response_form=str(body.get("response_form", "") or ""),
                    skill_ids=tuple(
                        str(item) for item in (body.get("skill_ids") or [])
                    ),
                    skill_revisions={
                        str(key): float(value)
                        for key, value in (
                            body.get("skill_revisions") or {}
                        ).items()
                    },
                    tool_directives=tuple(
                        str(item)
                        for item in (body.get("tool_directives") or [])
                    ),
                    source_policy=body.get("source_policy") or {},
                    web_recency=body.get("web_recency") or None,
                    execution_directive=str(
                        body.get("execution_directive", "") or ""
                    ),
                )
                # Reconstruct the metered subject from the persisted run
                # attribution — the worker has no live principal, but the
                # run's token spend must still count toward the
                # submitter's monthly quota (the in-process path meters
                # via the principal; this is the worker's equivalent).
                quota_subject = None
                if (
                    self._quota_service is not None
                    and claimed.created_by_sub
                    and claimed.created_by_tenant_id
                ):
                    quota_subject = QuotaSubject(
                        tenant_id=claimed.created_by_tenant_id,
                        sub=claimed.created_by_sub,
                    )
                # Reconstruct the OWNER principal from the persisted run
                # attribution: the workspace agent scopes its tool calls
                # and attributes child runs through it (the quota subject
                # alone cannot carry visibility). None only for legacy
                # rows without a recorded creator.
                principal = None
                if (
                    claimed.created_by_sub
                    and claimed.created_by_tenant_id
                    and claimed.created_by_sub
                    not in (ANONYMOUS_PRINCIPAL.sub, STATIC_PRINCIPAL.sub)
                ):
                    # The sentinel subs of the none/apikey modes stay
                    # principal-less: their historical unscoped view must
                    # not turn into a membership-scoped one queue-side.
                    principal = Principal(
                        sub=claimed.created_by_sub,
                        kind="oidc_session",
                        tenant_id=claimed.created_by_tenant_id,
                        role="member",
                    )
                requested_token_budget = int(
                    body.get("token_budget", 0) or 0
                ) or None
                if claimed.kind == "agent_child" and requested_token_budget:
                    # Pre-0043 mission planners embedded their own tiny caps in
                    # child replay payloads. Children now inherit only the
                    # operator's global run cap; trusted root overrides remain
                    # available for non-child runs.
                    log.warning(
                        "Legacy-Tokenbudget im Agent-Child %s ignoriert; "
                        "Operatorgrenzen bleiben autoritativ.",
                        job.run_id,
                    )
                    handle.emit(
                        "inqtrix.agent.activity",
                        {
                            "activity_id": (
                                f"legacy-child-budget:{job.run_id}"
                            ),
                            "scope": "task",
                            "phase": "execution",
                            "operation": "task.legacy_budget_ignored",
                            "detail": (
                                "Veraltetes Task-Budget wird ignoriert"
                            ),
                            "status": "completed",
                            "task_id": str(
                                body.get("parent_task_id") or ""
                            ),
                            "fallback": True,
                        },
                    )
                    requested_token_budget = None
                execute_run_request(
                    handle,
                    algorithm=algorithm,
                    run_request=run_request,
                    resolved=resolved,
                    runtime=self._runtime,
                    principal=principal,
                    quota_service=self._quota_service,
                    quota_subject=quota_subject,
                    token_budget=requested_token_budget,
                    workspace_id=claimed.workspace_id,
                )
            except Exception as exc:  # noqa: BLE001 — terminal-write then ack
                log.exception("Worker-Run %s fehlgeschlagen", job.run_id)
                terminate_native_run(handle, exc)
            if handle.terminal_landed or handle.parked:
                # Terminal state is committed (or the run is PARKED in a
                # waiting status — its resume re-enqueues a fresh
                # message); only now may the stream forget the job.
                self._queue.ack(job.message_id)
                old_message_acked = True
            else:
                # Fenced out: a superseding attempt owns the run AND
                # this very message id — acking here would strip the
                # new owner's crash-recovery entry. Leave it; the
                # owner acks on completion.
                log.warning(
                    "Worker %s: Run %s wurde waehrend der Ausfuehrung "
                    "von einem anderen Worker uebernommen — Ergebnis "
                    "verworfen, Nachricht bleibt beim neuen Owner.",
                    self._store.worker_id,
                    job.run_id,
                )
        except Exception:  # noqa: BLE001 — Futures here are unobserved
            # A failing terminal write or ack must surface in the log,
            # not vanish inside a never-awaited Future; the message
            # stays pending and redelivery retries the job.
            log.exception(
                "Worker %s: Abschlussphase fuer Run %s fehlgeschlagen — "
                "Redelivery uebernimmt.",
                self._store.worker_id,
                job.run_id,
            )
        finally:
            self._finish_active(
                job, allow_successor=old_message_acked
            )
