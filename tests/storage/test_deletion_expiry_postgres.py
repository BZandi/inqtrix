"""Postgres tests for the terminal guarantee on deletion operations (gated).

A deletion receipt that stays ``queued``/``running`` forever is a dead end:
clients poll it without an exit, ``retry`` is rejected because it requires
``delete_failed``, and a second DELETE answers 409. These tests pin the two
ways out — the dispatch timeout for an operation nobody claimed, and the
restart sweep for one whose in-process work closure died with its process.

Every store gets its OWN engine (it drives its own event loop, and asyncpg
pools are loop-affine), and the test's own assertions use short-lived
engines of their own for the same reason.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import insert, select, text

from inqtrix.runs.deletion_operations import (
    DeletionOperationStatus,
    DeletionTargetKind,
    SessionDeletionContext,
)
from inqtrix.runs.deletion_postgres import PostgresDeletionOperationStore
from inqtrix.storage.agent_sessions_orm import agent_sessions
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.deletions_orm import (
    deletion_operation_events,
    deletion_operations,
)
from inqtrix.storage.migrate import run_migrations
from tests.storage._canonical_users import (
    canonical_user_id,
    ensure_canonical_users,
)

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.postgres

APP_ROLE = "inqtrix_app"
OWNER_ID = canonical_user_id("deletion-expiry-owner")
WORKSPACE = "ws-expiry"


class _SilentQueue:
    """A dispatch channel nobody consumes.

    This is the deployment shape the timeout exists for: the row is
    enqueued and accepted, no worker ever claims it, and — unlike the
    no-queue case — no restart ever comes to clean it up.
    """

    def enqueue(self, *, operation_id: str, tenant_id: str) -> None:
        del operation_id, tenant_id

    def ack(self, message_id: str) -> None:
        del message_id


@pytest.fixture(scope="session", autouse=True)
def deletion_expiry_schema_migrated():
    if TEST_DATABASE_URL:
        run_migrations(TEST_DATABASE_URL)
    yield


@pytest_asyncio.fixture(autouse=True)
async def clean_database():
    engine = build_engine(TEST_DATABASE_URL)
    try:
        factory = build_session_factory(engine)
        async with factory() as session:
            async with session.begin():
                bypasses = (
                    await session.execute(
                        text(
                            "SELECT rolsuper OR rolbypassrls FROM pg_roles "
                            "WHERE rolname = current_user"
                        )
                    )
                ).scalar_one()
                if not bypasses:
                    pytest.fail(
                        "INQTRIX_TEST_DATABASE_URL must connect as "
                        "superuser/BYPASSRLS."
                    )
                await session.execute(deletion_operation_events.delete())
                await session.execute(deletion_operations.delete())
                await session.execute(agent_sessions.delete())
                await ensure_canonical_users(session, (OWNER_ID,))
    finally:
        await engine.dispose()
    yield


def _store(
    *,
    queue,
    dispatch_timeout_seconds: float = 240.0,
    worker_id: str = "expiry-test",
) -> PostgresDeletionOperationStore:
    return PostgresDeletionOperationStore(
        engine=build_engine(TEST_DATABASE_URL),
        app_role=APP_ROLE,
        queue=queue,
        max_concurrent=1,
        completed_ttl_seconds=3600,
        dispatch_timeout_seconds=dispatch_timeout_seconds,
        worker_id=worker_id,
    )


async def _seed_session(session_id: str) -> None:
    engine = build_engine(TEST_DATABASE_URL)
    try:
        factory = build_session_factory(engine)
        async with factory() as session:
            async with session.begin():
                await session.execute(
                    insert(agent_sessions).values(
                        id=session_id,
                        tenant_id="default",
                        created_by_user_id=OWNER_ID,
                        workspace_id=WORKSPACE,
                        title="Expiry",
                        items_json="[]",
                        lifecycle_status="active",
                        created_at=1.0,
                        updated_at=1.0,
                    )
                )
    finally:
        await engine.dispose()


async def _session_row(session_id: str) -> dict:
    engine = build_engine(TEST_DATABASE_URL)
    try:
        factory = build_session_factory(engine)
        async with factory() as session:
            row = (
                (
                    await session.execute(
                        select(
                            agent_sessions.c.lifecycle_status,
                            agent_sessions.c.deletion_error,
                        ).where(agent_sessions.c.id == session_id)
                    )
                )
                .mappings()
                .first()
            )
            return dict(row) if row is not None else {}
    finally:
        await engine.dispose()


def _submit_session_deletion(store, session_id: str) -> str:
    summary = store.submit(
        target_kind=DeletionTargetKind.AGENT_SESSION,
        target_id=session_id,
        manifest=(),
        tenant_id="default",
        created_by_user_id=OWNER_ID,
        workspace_id=WORKSPACE,
        work=lambda handle: None,
        session_context=SessionDeletionContext(
            target_kind=DeletionTargetKind.AGENT_SESSION,
            session_id=session_id,
        ),
        total_items=2,
    )
    return summary["operation_id"]


def _get(store, operation_id: str) -> dict:
    return store.get(
        operation_id,
        tenant_id="default",
        created_by_user_id=OWNER_ID,
        workspace_id=WORKSPACE,
    )


@pytest.mark.asyncio
async def test_unclaimed_operation_expires_and_frees_the_session() -> None:
    """The dead end ends: expiry makes the receipt terminal AND retryable."""

    await _seed_session("as_expire")
    store = _store(queue=_SilentQueue(), dispatch_timeout_seconds=0.0)
    try:
        operation_id = _submit_session_deletion(store, "as_expire")
        assert _get(store, operation_id)["status"] == (
            DeletionOperationStatus.DELETE_FAILED.value
        )
        summary = _get(store, operation_id)
        assert summary["error"]["type"] == "dispatch_timeout"

        row = await _session_row("as_expire")
        assert row["lifecycle_status"] == "delete_failed"
        assert row["deletion_error"]

        # Retry requires delete_failed, so this call is the proof that the
        # operation is reachable again instead of stuck behind a 409.
        retried = store.retry(
            operation_id,
            tenant_id="default",
            created_by_user_id=OWNER_ID,
            workspace_id=WORKSPACE,
            work=lambda handle: None,
        )
        assert retried["status"] == DeletionOperationStatus.QUEUED.value
    finally:
        store.close()


@pytest.mark.asyncio
async def test_fresh_operation_survives_the_expiry_pass() -> None:
    """A healthy operation inside its window is never touched."""

    await _seed_session("as_fresh")
    store = _store(queue=_SilentQueue(), dispatch_timeout_seconds=240.0)
    try:
        operation_id = _submit_session_deletion(store, "as_fresh")
        store._last_expiry_check = None
        assert _get(store, operation_id)["status"] == (
            DeletionOperationStatus.QUEUED.value
        )
        row = await _session_row("as_fresh")
        assert row["lifecycle_status"] == "deleting"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_expiry_is_idempotent() -> None:
    """A second pass over an already-failed operation changes nothing."""

    await _seed_session("as_twice")
    store = _store(queue=_SilentQueue(), dispatch_timeout_seconds=0.0)
    try:
        operation_id = _submit_session_deletion(store, "as_twice")
        first = _get(store, operation_id)
        store._last_expiry_check = None
        second = _get(store, operation_id)

        assert first["status"] == DeletionOperationStatus.DELETE_FAILED.value
        assert second["status"] == first["status"]
        assert second["error"]["type"] == "dispatch_timeout"
        assert second["finished_at"] == first["finished_at"]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_restart_sweep_fails_orphans_of_a_previous_process() -> None:
    """A row whose work closure died with its process is swept once."""

    await _seed_session("as_orphan")
    producer = _store(queue=_SilentQueue(), worker_id="previous-process")
    try:
        operation_id = _submit_session_deletion(producer, "as_orphan")
    finally:
        producer.close()

    # queue=None is the in-process shape, the only one resolve_orphan_sweep
    # lets sweep. The generous timeout keeps the dispatch-timeout path out
    # of this test, so the verdict can only come from the sweep.
    successor = _store(
        queue=None,
        dispatch_timeout_seconds=10_000.0,
        worker_id="new-process",
    )
    try:
        summary = _get(successor, operation_id)
        assert summary["status"] == DeletionOperationStatus.DELETE_FAILED.value
        assert summary["error"]["type"] == "server_restarted"

        row = await _session_row("as_orphan")
        assert row["lifecycle_status"] == "delete_failed"
    finally:
        successor.close()


@pytest.mark.asyncio
async def test_queue_mode_never_sweeps_live_worker_rows() -> None:
    """With a queue the rows belong to the workers, so no sweep may run."""

    await _seed_session("as_queued")
    producer = _store(queue=_SilentQueue(), worker_id="previous-process")
    try:
        operation_id = _submit_session_deletion(producer, "as_queued")
    finally:
        producer.close()

    successor = _store(
        queue=_SilentQueue(),
        dispatch_timeout_seconds=10_000.0,
        worker_id="new-process",
    )
    try:
        assert _get(successor, operation_id)["status"] == (
            DeletionOperationStatus.QUEUED.value
        )
    finally:
        successor.close()
