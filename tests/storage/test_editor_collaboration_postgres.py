"""PostgreSQL integration contracts for editor live collaboration.

The suite is deliberately gated on ``INQTRIX_TEST_DATABASE_URL`` and runs
the production store under the restricted application role. It covers the
transactional guarantees that memory fakes cannot prove: conversion CAS,
serialized sequence allocation, lease and instance fencing, patch metadata
co-commit, snapshot/projection consistency, retention, tombstones, and RLS.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, replace

import pytest
import pytest_asyncio
from sqlalchemy import func, insert, select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

import inqtrix.storage.editor_collaboration_postgres as collaboration_postgres
from inqtrix.project.editor_collaboration_ports import (
    CollaborationConflict,
    CollaborationDecision,
    CollaborationDocumentNotFound,
    CollaborationInstanceFenced,
    CollaborationInstanceLease,
    CollaborationLease,
    CollaborationLeaseInvalid,
    CollaborationPatchState,
    CollaborationPermission,
    CollaborationRateLimited,
    CollaborationSnapshot,
    CollaborationSuggestion,
    PersistCollaborationUpdate,
)
from inqtrix.storage.auth_orm import auth_sessions
from inqtrix.storage.db import (
    build_engine,
    build_session_factory,
    tenant_session,
)
from inqtrix.storage.editor_collaboration_orm import (
    editor_collaboration_instances,
    editor_collaboration_leases,
    editor_collaboration_snapshots,
    editor_collaboration_updates,
)
from inqtrix.storage.editor_collaboration_postgres import (
    PostgresEditorCollaborationStore,
)
from inqtrix.storage.editor_orm import (
    editor_comments,
    editor_documents,
    editor_folders,
)
from inqtrix.storage.editor_patch_orm import editor_patches
from inqtrix.storage.identity_orm import audit_log, resource_shares
from inqtrix.storage.migrate import run_migrations
from inqtrix.storage.user_event_orm import user_events
from tests.storage._canonical_users import (
    canonical_user_id,
    ensure_canonical_users,
)

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.postgres

APP_ROLE = "inqtrix_app"
TENANT_A = "collaboration-postgres-a"
TENANT_B = "collaboration-postgres-b"
TENANTS = (TENANT_A, TENANT_B)
OWNER_A = canonical_user_id("collaboration-postgres-owner-a")
OWNER_B = canonical_user_id("collaboration-postgres-owner-b")
DOCUMENT_REVISION = 7
METADATA_REVISION = 3
SCHEMA_VERSION = 1
SCHEMA_HASH = hashlib.sha256(b"inqtrix-editor-schema-v1").hexdigest()


@dataclass(frozen=True)
class _DatabaseHarness:
    """Real database resources and a deterministic time origin for one test."""

    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    store: PostgresEditorCollaborationStore
    now: float


@pytest.fixture(scope="session", autouse=True)
def collaboration_schema_migrated() -> Iterator[None]:
    """Apply the real migration chain once when the integration DB is enabled."""
    if TEST_DATABASE_URL:
        run_migrations(TEST_DATABASE_URL)
    yield


async def _wipe_test_tenants(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Remove only rows owned by this module's reserved test tenants."""
    async with session_factory() as session:
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
                    "INQTRIX_TEST_DATABASE_URL must connect as a "
                    "superuser/BYPASSRLS user for cross-tenant cleanup."
                )
            for table in (
                editor_collaboration_leases,
                editor_collaboration_updates,
                editor_collaboration_snapshots,
                editor_collaboration_instances,
                editor_patches,
                editor_comments,
                resource_shares,
                auth_sessions,
                editor_documents,
                editor_folders,
                user_events,
                audit_log,
            ):
                await session.execute(
                    table.delete().where(table.c.tenant_id.in_(TENANTS))
                )
            await ensure_canonical_users(
                session,
                (OWNER_A,),
                tenant_id=TENANT_A,
            )
            await ensure_canonical_users(
                session,
                (OWNER_B,),
                tenant_id=TENANT_B,
            )


@pytest_asyncio.fixture()
async def database() -> AsyncIterator[_DatabaseHarness]:
    """Yield the collaboration store over a clean, migrated real database."""
    engine = build_engine(TEST_DATABASE_URL)
    session_factory = build_session_factory(engine)
    await _wipe_test_tenants(session_factory)
    harness = _DatabaseHarness(
        engine=engine,
        session_factory=session_factory,
        store=PostgresEditorCollaborationStore(
            session_factory=session_factory,
            app_role=APP_ROLE,
            restrict_to_workspace_members=False,
        ),
        now=time.time(),
    )
    try:
        yield harness
    finally:
        await _wipe_test_tenants(session_factory)
        await engine.dispose()


def _sha256(payload: bytes) -> str:
    """Return the lowercase digest used by collaboration persistence."""
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.asyncio
async def test_current_policy_cursor_is_tenant_scoped(
    database: _DatabaseHarness,
) -> None:
    """Each tenant sees only its greatest committed content-free event ID."""
    async with database.session_factory() as session:
        async with session.begin():
            tenant_b_cursor = (
                await session.execute(
                    insert(user_events)
                    .values(
                        tenant_id=TENANT_B,
                        target_user_id=OWNER_B,
                        scope="share:accepted",
                        resource_type="editor_document",
                        resource_id="ed_policy_b",
                    )
                    .returning(user_events.c.id)
                )
            ).scalar_one()
            tenant_a_cursor = (
                await session.execute(
                    insert(user_events)
                    .values(
                        tenant_id=TENANT_A,
                        target_user_id=OWNER_A,
                        scope="share:accepted",
                        resource_type="editor_document",
                        resource_id="ed_policy_a",
                    )
                    .returning(user_events.c.id)
                )
            ).scalar_one()

    assert await database.store.current_policy_cursor(
        tenant_id=TENANT_A
    ) == int(tenant_a_cursor)
    assert await database.store.current_policy_cursor(
        tenant_id=TENANT_B
    ) == int(tenant_b_cursor)


def _snapshot(
    *,
    tenant_id: str,
    document_id: str,
    sequence: int,
    state_update: bytes,
    projection: str,
    created_at: float,
) -> CollaborationSnapshot:
    """Build a hash-consistent snapshot accepted by the persistence port."""
    return CollaborationSnapshot(
        document_id=document_id,
        tenant_id=tenant_id,
        generation=1,
        covered_sequence=sequence,
        state_update=state_update,
        state_vector=f"vector-{sequence}".encode(),
        state_hash=_sha256(state_update),
        projection_hash=_sha256(projection.encode()),
        schema_version=SCHEMA_VERSION,
        schema_hash=SCHEMA_HASH,
        created_at=created_at,
    )


async def _seed_markdown_document(
    database: _DatabaseHarness,
    *,
    tenant_id: str,
    document_id: str,
    owner_user_id: uuid.UUID,
    markdown: str = "# Initial\n\nBody.",
) -> None:
    """Insert one legacy Markdown document through the bypass test connection."""
    async with database.session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(editor_documents).values(
                    id=document_id,
                    tenant_id=tenant_id,
                    created_by_user_id=owner_user_id,
                    workspace_id=None,
                    title="Collaboration test",
                    content_markdown=markdown,
                    folder_id=None,
                    source="blank",
                    source_run_id=None,
                    revision=DOCUMENT_REVISION,
                    content_mode="markdown",
                    metadata_revision=METADATA_REVISION,
                    collaboration_generation=0,
                    collaboration_schema_version=None,
                    collaboration_schema_hash=None,
                    persisted_sequence=0,
                    projection_sequence=0,
                    projection_updated_at=None,
                    deleted_at=None,
                    diff_anchor_markdown=None,
                    diff_anchor_updated_at=None,
                    created_at=database.now,
                    updated_at=database.now,
                )
            )


async def _activate_document(
    database: _DatabaseHarness,
    *,
    tenant_id: str,
    document_id: str,
    owner_user_id: uuid.UUID,
    projection: str = "# Initial\n\nBody.",
) -> None:
    """Convert a seeded Markdown document with a valid generation-one snapshot."""
    snapshot = _snapshot(
        tenant_id=tenant_id,
        document_id=document_id,
        sequence=0,
        state_update=f"initial:{tenant_id}:{document_id}".encode(),
        projection=projection,
        created_at=database.now,
    )
    await database.store.enable_document(
        tenant_id=tenant_id,
        document_id=document_id,
        owner_user_id=owner_user_id,
        expected_revision=DOCUMENT_REVISION,
        expected_metadata_revision=METADATA_REVISION,
        schema_version=SCHEMA_VERSION,
        schema_hash=SCHEMA_HASH,
        snapshot=snapshot,
        projection_markdown=projection,
        now=database.now,
    )


async def _seed_browser_session(
    database: _DatabaseHarness,
    *,
    tenant_id: str,
    user_id: uuid.UUID,
    session_id: str,
) -> None:
    """Insert one live cookie session for lease authorization checks."""
    async with database.session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(auth_sessions).values(
                    id=session_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    issuer="https://collaboration-tests.example",
                    subject=f"subject-{user_id.hex}",
                    email=f"{user_id.hex}@example.com",
                    display_name="Collaboration Tester",
                    groups=[],
                    csrf_random="ab" * 16,
                    created_at=database.now,
                    expires_at=database.now + 10_000.0,
                )
            )


async def _authorize_writer(
    database: _DatabaseHarness,
    *,
    tenant_id: str,
    document_id: str,
    user_id: uuid.UUID,
    permission: CollaborationPermission = "edit",
    instance_id: str = "node-primary",
    instance_lease_seconds: float = 5_000.0,
) -> tuple[CollaborationInstanceLease, CollaborationLease]:
    """Acquire the writer epoch and persist one live browser lease."""
    session_id = f"session:{tenant_id}:{document_id}:{user_id.hex}"
    await _seed_browser_session(
        database,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
    )
    instance = await database.store.acquire_instance(
        tenant_id=tenant_id,
        instance_id=instance_id,
        now=database.now,
        lease_seconds=instance_lease_seconds,
    )
    lease_id = uuid.uuid4()
    lease = CollaborationLease(
        lease_id=lease_id,
        token_hash=_sha256(f"token:{lease_id}".encode()),
        tenant_id=tenant_id,
        document_id=document_id,
        generation=1,
        user_id=user_id,
        permission=permission,
        session_id=session_id,
        issued_at=database.now,
        expires_at=database.now + 3_600.0,
        last_validated_at=database.now,
    )
    await database.store.issue_lease(
        lease,
        max_active=5,
        max_issued_per_window=30,
        issued_since=database.now - 60.0,
    )
    return instance, lease


def _direct_update(
    *,
    tenant_id: str,
    document_id: str,
    actor_user_id: uuid.UUID,
    instance: CollaborationInstanceLease,
    lease: CollaborationLease,
    payload: bytes,
    now: float,
    expected_sequence: int | None = None,
) -> PersistCollaborationUpdate:
    """Build one ordinary human edit update."""
    return PersistCollaborationUpdate(
        tenant_id=tenant_id,
        document_id=document_id,
        generation=1,
        instance_id=instance.instance_id,
        instance_epoch=instance.epoch,
        lease_id=lease.lease_id,
        actor_user_id=actor_user_id,
        update_hash=_sha256(payload),
        update_bytes=payload,
        actor_kind="human",
        change_kind="direct",
        expected_sequence=expected_sequence,
        now=now,
    )


def _suggestion_update(
    *,
    tenant_id: str,
    document_id: str,
    actor_user_id: uuid.UUID,
    instance: CollaborationInstanceLease,
    lease: CollaborationLease,
    payload: bytes,
    suggestions: tuple[CollaborationSuggestion, ...],
    patches: tuple[CollaborationPatchState, ...],
    now: float,
    expected_sequence: int | None = None,
) -> PersistCollaborationUpdate:
    """Build one human suggestion update with sidecar-validated descriptors."""
    return PersistCollaborationUpdate(
        tenant_id=tenant_id,
        document_id=document_id,
        generation=1,
        instance_id=instance.instance_id,
        instance_epoch=instance.epoch,
        lease_id=lease.lease_id,
        actor_user_id=actor_user_id,
        update_hash=_sha256(payload),
        update_bytes=payload,
        actor_kind="human",
        change_kind="suggestion",
        suggestion_ids=tuple(item.suggestion_id for item in suggestions),
        suggestions=suggestions,
        patches=patches,
        expected_sequence=expected_sequence,
        now=now,
    )


def _assistant_suggestion_update(
    *,
    tenant_id: str,
    document_id: str,
    actor_user_id: uuid.UUID,
    instance: CollaborationInstanceLease,
    payload: bytes,
    suggestions: tuple[CollaborationSuggestion, ...],
    patches: tuple[CollaborationPatchState, ...],
    command_id: uuid.UUID,
    now: float,
    expected_sequence: int | None = None,
) -> PersistCollaborationUpdate:
    """Build one server-side assistant publish without a browser lease."""
    return PersistCollaborationUpdate(
        tenant_id=tenant_id,
        document_id=document_id,
        generation=1,
        instance_id=instance.instance_id,
        instance_epoch=instance.epoch,
        lease_id=None,
        actor_user_id=actor_user_id,
        update_hash=_sha256(payload),
        update_bytes=payload,
        actor_kind="assistant",
        change_kind="suggestion",
        suggestion_ids=tuple(item.suggestion_id for item in suggestions),
        suggestions=suggestions,
        patches=patches,
        command_id=command_id,
        command_payload_hash=_sha256(f"publish:{command_id}".encode()),
        expected_sequence=expected_sequence,
        now=now,
    )


async def _create_human_patch(
    database: _DatabaseHarness,
    *,
    tenant_id: str,
    document_id: str,
    actor_user_id: uuid.UUID,
    instance: CollaborationInstanceLease,
    lease: CollaborationLease,
    patch_id: str | None = None,
    suggestion_id: str | None = None,
) -> tuple[str, str]:
    """Persist one initial insertion suggestion and return its identities."""
    actual_patch_id = patch_id or str(uuid.uuid4())
    actual_suggestion_id = suggestion_id or str(uuid.uuid4())
    descriptor = CollaborationSuggestion(
        suggestion_id=actual_suggestion_id,
        patch_id=actual_patch_id,
        author_id=actor_user_id,
        created_at=database.now + 1.0,
        kind="insertion",
    )
    patch_state = CollaborationPatchState(
        patch_id=actual_patch_id,
        author_id=actor_user_id,
        created_at=database.now + 1.0,
        active_suggestion_ids=(actual_suggestion_id,),
        kinds=("insertion",),
    )
    await database.store.append_update(
        _suggestion_update(
            tenant_id=tenant_id,
            document_id=document_id,
            actor_user_id=actor_user_id,
            instance=instance,
            lease=lease,
            payload=f"suggestion:{actual_suggestion_id}".encode(),
            suggestions=(descriptor,),
            patches=(patch_state,),
            now=database.now + 1.0,
            expected_sequence=0,
        )
    )
    return actual_patch_id, actual_suggestion_id


@pytest.mark.asyncio
async def test_conversion_is_atomic_and_preserves_legacy_body_revision(
    database: _DatabaseHarness,
) -> None:
    """A failed CAS stores nothing; a valid conversion establishes all invariants."""
    document_id = "ed_collaboration_conversion"
    projection = "# Initial\n\nBody."
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
        markdown=projection,
    )
    snapshot = _snapshot(
        tenant_id=TENANT_A,
        document_id=document_id,
        sequence=0,
        state_update=b"conversion-state",
        projection=projection,
        created_at=database.now,
    )

    with pytest.raises(CollaborationConflict) as conflict:
        await database.store.enable_document(
            tenant_id=TENANT_A,
            document_id=document_id,
            owner_user_id=OWNER_A,
            expected_revision=DOCUMENT_REVISION - 1,
            expected_metadata_revision=METADATA_REVISION,
            schema_version=SCHEMA_VERSION,
            schema_hash=SCHEMA_HASH,
            snapshot=snapshot,
            projection_markdown=projection,
            now=database.now,
        )
    assert conflict.value.reason == "revision_conflict"

    async with database.session_factory() as session:
        document_before = (
            await session.execute(
                select(editor_documents).where(
                    editor_documents.c.id == document_id
                )
            )
        ).one()
        snapshot_count = await session.scalar(
            select(func.count())
            .select_from(editor_collaboration_snapshots)
            .where(editor_collaboration_snapshots.c.document_id == document_id)
        )
    assert document_before.content_mode == "markdown"
    assert document_before.collaboration_generation == 0
    assert snapshot_count == 0

    converted = await database.store.enable_document(
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
        expected_revision=DOCUMENT_REVISION,
        expected_metadata_revision=METADATA_REVISION,
        schema_version=SCHEMA_VERSION,
        schema_hash=SCHEMA_HASH,
        snapshot=snapshot,
        projection_markdown=projection,
        now=database.now,
    )

    assert converted.generation == 1
    assert converted.persisted_sequence == converted.projection_sequence == 0
    async with database.session_factory() as session:
        stored = (
            await session.execute(
                select(editor_documents).where(
                    editor_documents.c.id == document_id
                )
            )
        ).one()
        snapshots = (
            await session.execute(
                select(editor_collaboration_snapshots).where(
                    editor_collaboration_snapshots.c.document_id == document_id
                )
            )
        ).all()
    assert stored.content_mode == "collaboration"
    assert stored.revision == DOCUMENT_REVISION
    assert stored.metadata_revision == METADATA_REVISION + 1
    assert stored.collaboration_schema_version == SCHEMA_VERSION
    assert stored.collaboration_schema_hash == SCHEMA_HASH
    assert stored.content_markdown == projection
    assert len(snapshots) == 1
    assert snapshots[0].covered_sequence == 0


@pytest.mark.asyncio
async def test_concurrent_updates_allocate_once_and_replay_is_idempotent(
    database: _DatabaseHarness,
) -> None:
    """Concurrent writers receive consecutive sequences and a hash replay reuses one."""
    document_id = "ed_collaboration_sequences"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    first = _direct_update(
        tenant_id=TENANT_A,
        document_id=document_id,
        actor_user_id=OWNER_A,
        instance=instance,
        lease=lease,
        payload=b"concurrent-update-a",
        now=database.now + 1.0,
    )
    second = _direct_update(
        tenant_id=TENANT_A,
        document_id=document_id,
        actor_user_id=OWNER_A,
        instance=instance,
        lease=lease,
        payload=b"concurrent-update-b",
        now=database.now + 1.0,
    )

    results = await asyncio.gather(
        database.store.append_update(first),
        database.store.append_update(second),
    )
    assert {result.sequence for result in results} == {1, 2}
    assert all(not result.duplicate for result in results)

    oldest_update = first if results[0].sequence == 1 else second
    replay = await database.store.append_update(oldest_update)
    assert replay.duplicate is True
    assert replay.sequence == 1
    assert replay.persisted_sequence == 2

    async with database.session_factory() as session:
        sequences = list(
            (
                await session.execute(
                    select(editor_collaboration_updates.c.sequence)
                    .where(
                        editor_collaboration_updates.c.document_id == document_id
                    )
                    .order_by(editor_collaboration_updates.c.sequence)
                )
            ).scalars()
        )
        persisted_sequence = await session.scalar(
            select(editor_documents.c.persisted_sequence).where(
                editor_documents.c.id == document_id
            )
        )
    assert sequences == [1, 2]
    assert persisted_sequence == 2


@pytest.mark.asyncio
async def test_activity_selects_optional_guest_actor_identity(
    database: _DatabaseHarness,
) -> None:
    """Activity rows expose the additive guest actor column for user edits too."""
    document_id = "ed_collaboration_activity_actor"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    await database.store.append_update(
        _direct_update(
            tenant_id=TENANT_A,
            document_id=document_id,
            actor_user_id=OWNER_A,
            instance=instance,
            lease=lease,
            payload=b"activity-actor-update",
            now=database.now + 1.0,
        )
    )

    activity = await database.store.list_activity(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        before_sequence=None,
        author_user_id=None,
        change_kind=None,
        limit=10,
    )

    assert len(activity) == 1
    assert activity[0].actor_user_id == OWNER_A
    assert activity[0].actor_guest_identity_id is None


@pytest.mark.asyncio
async def test_update_hash_lookup_is_generation_and_tenant_scoped(
    database: _DatabaseHarness,
) -> None:
    """Reconnect lookup returns persisted coordinates without crossing RLS."""
    document_id = "ed_collaboration_hash_lookup"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    payload = b"lookup-persisted-update"
    update_hash = _sha256(payload)
    missing_hash = _sha256(b"lookup-missing-update")
    await database.store.append_update(
        _direct_update(
            tenant_id=TENANT_A,
            document_id=document_id,
            actor_user_id=OWNER_A,
            instance=instance,
            lease=lease,
            payload=payload,
            now=database.now + 1.0,
        )
    )

    found = await database.store.lookup_updates_by_hashes(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        update_hashes=(update_hash, missing_hash),
        instance_id=instance.instance_id,
        instance_epoch=instance.epoch,
        now=database.now + 2.0,
    )

    assert [(item.update_hash, item.sequence) for item in found] == [
        (update_hash, 1)
    ]
    with pytest.raises(CollaborationInstanceFenced):
        await database.store.lookup_updates_by_hashes(
            tenant_id=TENANT_A,
            document_id=document_id,
            generation=1,
            update_hashes=(update_hash,),
            instance_id=instance.instance_id,
            instance_epoch=instance.epoch + 1,
            now=database.now + 2.0,
        )
    instance_b = await database.store.acquire_instance(
        tenant_id=TENANT_B,
        instance_id="node-hash-lookup-b",
        now=database.now,
        lease_seconds=30.0,
    )
    with pytest.raises(CollaborationDocumentNotFound):
        await database.store.lookup_updates_by_hashes(
            tenant_id=TENANT_B,
            document_id=document_id,
            generation=1,
            update_hashes=(update_hash,),
            instance_id=instance_b.instance_id,
            instance_epoch=instance_b.epoch,
            now=database.now + 2.0,
        )
    with pytest.raises(CollaborationConflict) as conflict:
        await database.store.lookup_updates_by_hashes(
            tenant_id=TENANT_A,
            document_id=document_id,
            generation=2,
            update_hashes=(update_hash,),
            instance_id=instance.instance_id,
            instance_epoch=instance.epoch,
            now=database.now + 2.0,
        )
    assert conflict.value.reason == "generation_conflict"


@pytest.mark.asyncio
async def test_append_requires_live_lease_and_current_instance_epoch(
    database: _DatabaseHarness,
) -> None:
    """Revocation blocks the user and an expired writer epoch fences old Node state."""
    document_id = "ed_collaboration_fencing"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
        instance_id="node-old",
        instance_lease_seconds=10.0,
    )
    accepted = _direct_update(
        tenant_id=TENANT_A,
        document_id=document_id,
        actor_user_id=OWNER_A,
        instance=instance,
        lease=lease,
        payload=b"authorized",
        now=database.now + 1.0,
    )
    assert (await database.store.append_update(accepted)).sequence == 1

    assert (
        await database.store.revoke_leases(
            tenant_id=TENANT_A,
            document_id=document_id,
            user_id=OWNER_A,
            now=database.now + 2.0,
        )
        == 1
    )
    with pytest.raises(CollaborationLeaseInvalid):
        await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=lease,
                payload=b"revoked",
                now=database.now + 3.0,
            )
        )

    replacement = await database.store.acquire_instance(
        tenant_id=TENANT_A,
        instance_id="node-new",
        now=database.now + 11.0,
        lease_seconds=10.0,
    )
    assert replacement.epoch == instance.epoch + 1
    with pytest.raises(CollaborationInstanceFenced):
        await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=lease,
                payload=b"stale-instance",
                now=database.now + 12.0,
            )
        )

    async with database.session_factory() as session:
        update_count = await session.scalar(
            select(func.count())
            .select_from(editor_collaboration_updates)
            .where(editor_collaboration_updates.c.document_id == document_id)
        )
    assert update_count == 1


@pytest.mark.asyncio
async def test_current_instance_probe_is_live_and_tenant_scoped(
    database: _DatabaseHarness,
) -> None:
    """The public-probe read returns only the tenant's unexpired fencing row."""
    first = await database.store.acquire_instance(
        tenant_id=TENANT_A,
        instance_id="node-probe-a",
        now=database.now,
        lease_seconds=10.0,
    )
    tenant_b = await database.store.acquire_instance(
        tenant_id=TENANT_B,
        instance_id="node-probe-b",
        now=database.now,
        lease_seconds=30.0,
    )

    assert await database.store.get_current_instance(
        tenant_id=TENANT_A,
        now=database.now + 5.0,
    ) == first
    assert await database.store.get_current_instance(
        tenant_id=TENANT_A,
        now=database.now + 11.0,
    ) is None
    assert await database.store.get_current_instance(
        tenant_id=TENANT_B,
        now=database.now + 11.0,
    ) == tenant_b

    replacement = await database.store.acquire_instance(
        tenant_id=TENANT_A,
        instance_id="node-probe-a-replacement",
        now=database.now + 11.0,
        lease_seconds=10.0,
    )
    observed = await database.store.get_current_instance(
        tenant_id=TENANT_A,
        now=database.now + 12.0,
    )
    assert observed == replacement
    assert replacement.epoch == first.epoch + 1


@pytest.mark.asyncio
async def test_human_suggestion_co_commits_patch_create_and_membership_update(
    database: _DatabaseHarness,
) -> None:
    """Human suggestion updates create and evolve one patch in the same commits."""
    document_id = "ed_collaboration_human_patch"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
        permission="suggest",
    )
    patch_id, first_id = await _create_human_patch(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        actor_user_id=OWNER_A,
        instance=instance,
        lease=lease,
    )

    second_id = str(uuid.uuid4())
    second_descriptor = CollaborationSuggestion(
        suggestion_id=second_id,
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 2.0,
        kind="deletion",
    )
    expanded_patch = CollaborationPatchState(
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        active_suggestion_ids=(first_id, second_id),
        kinds=("insertion", "deletion"),
    )
    second_result = await database.store.append_update(
        _suggestion_update(
            tenant_id=TENANT_A,
            document_id=document_id,
            actor_user_id=OWNER_A,
            instance=instance,
            lease=lease,
            payload=b"expand-human-patch",
            suggestions=(second_descriptor,),
            patches=(expanded_patch,),
            now=database.now + 2.0,
            expected_sequence=1,
        )
    )
    assert second_result.sequence == 2

    async with database.session_factory() as session:
        patch = (
            await session.execute(
                select(editor_patches).where(editor_patches.c.patch_id == patch_id)
            )
        ).mappings().one()
        updates = (
            await session.execute(
                select(
                    editor_collaboration_updates.c.sequence,
                    editor_collaboration_updates.c.suggestion_ids,
                )
                .where(editor_collaboration_updates.c.document_id == document_id)
                .order_by(editor_collaboration_updates.c.sequence)
            )
        ).all()
        persisted_sequence = await session.scalar(
            select(editor_documents.c.persisted_sequence).where(
                editor_documents.c.id == document_id
            )
        )
    assert patch["source"] == "human"
    assert patch["status"] == "pending"
    assert patch["created_by_user_id"] == OWNER_A
    assert patch["collaboration_generation"] == 1
    assert patch["base_sequence"] == 0
    assert patch["suggestion_ids"] == [first_id, second_id]
    assert [item["suggestion_id"] for item in patch["edits"]] == [
        first_id,
        second_id,
    ]
    assert [row.sequence for row in updates] == [1, 2]
    assert updates[0].suggestion_ids == [first_id]
    assert updates[1].suggestion_ids == [second_id]
    assert persisted_sequence == 2


@pytest.mark.asyncio
async def test_private_assistant_publish_creates_patch_and_clears_draft_atomically(
    database: _DatabaseHarness,
) -> None:
    """A matching private draft authorizes one atomic shared patch publish."""
    document_id = "ed_collaboration_private_draft_publish"
    patch_id = str(uuid.uuid4())
    command_id = uuid.uuid4()
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    async with database.session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(editor_comments).values(
                    id="edc_private_ai",
                    document_id=document_id,
                    tenant_id=TENANT_A,
                    created_by_user_id=OWNER_A,
                    comment_markdown="Private rewrite",
                    anchor={},
                    kind="inline_edit",
                    status="open",
                    evidence_preset=None,
                    suggestion_draft={
                        "anchor_version": 1,
                        "change_summary": [],
                        "created_at": database.now,
                        "evidence": None,
                        "group_id": "editor-suggestion-group-publish",
                        "patch_id": patch_id,
                        "proposed_text": "Private provider output",
                        "publication_command_id": str(command_id),
                        "revision": 1,
                        "revision_history": [],
                        "suggestion_id": "editor-suggestion-publish",
                        "updated_at": database.now,
                        "warnings": [],
                    },
                    created_at=database.now,
                    updated_at=database.now,
                )
            )
    instance, _lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
        permission="suggest",
    )

    suggestion_id = str(uuid.uuid4())
    descriptor = CollaborationSuggestion(
        suggestion_id=suggestion_id,
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        kind="replacement",
    )
    patch_state = CollaborationPatchState(
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        active_suggestion_ids=(suggestion_id,),
        kinds=("replacement",),
    )
    result = await database.store.append_update(
        _assistant_suggestion_update(
            tenant_id=TENANT_A,
            document_id=document_id,
            actor_user_id=OWNER_A,
            instance=instance,
            payload=b"publish-private-assistant-draft",
            suggestions=(descriptor,),
            patches=(patch_state,),
            command_id=command_id,
            now=database.now + 1.0,
            expected_sequence=0,
        )
    )

    async with database.session_factory() as session:
        draft = await session.scalar(
            select(editor_comments.c.suggestion_draft).where(
                editor_comments.c.tenant_id == TENANT_A,
                editor_comments.c.document_id == document_id,
                editor_comments.c.id == "edc_private_ai",
            )
        )
        patch = (
            await session.execute(
                select(editor_patches).where(
                    editor_patches.c.tenant_id == TENANT_A,
                    editor_patches.c.patch_id == patch_id,
                )
            )
        ).mappings().one()
    assert draft is None
    assert result.sequence == 1
    assert patch["source"] == "human"
    assert patch["created_by_user_id"] == OWNER_A
    assert patch["created_by_guest_identity_id"] is None
    assert patch["command_id"] == command_id
    assert patch["suggestion_ids"] == [suggestion_id]
    assert patch["edits"] == [
        {
            "suggestion_id": suggestion_id,
            "patch_id": patch_id,
            "author_id": str(OWNER_A),
            "created_at": database.now + 1.0,
            "kind": "replacement",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_command_id",
    [None, uuid.UUID("99999999-9999-4999-8999-999999999999")],
    ids=["missing-draft", "command-mismatch"],
)
async def test_private_assistant_publish_requires_matching_creator_draft(
    database: _DatabaseHarness,
    stored_command_id: uuid.UUID | None,
) -> None:
    """Assistant actor alone or a mismatched command cannot mint a shared patch."""
    document_id = "ed_collaboration_private_publish_without_draft"
    patch_id = str(uuid.uuid4())
    suggestion_id = str(uuid.uuid4())
    request_command_id = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    if stored_command_id is not None:
        async with database.session_factory() as session:
            async with session.begin():
                await session.execute(
                    insert(editor_comments).values(
                        id="edc_private_ai_mismatch",
                        document_id=document_id,
                        tenant_id=TENANT_A,
                        created_by_user_id=OWNER_A,
                        comment_markdown="Private command mismatch",
                        anchor={},
                        kind="inline_edit",
                        status="open",
                        evidence_preset=None,
                        suggestion_draft={
                            "anchor_version": 1,
                            "change_summary": [],
                            "created_at": database.now,
                            "evidence": None,
                            "group_id": "editor-suggestion-group-mismatch",
                            "patch_id": patch_id,
                            "proposed_text": "Private provider output",
                            "publication_command_id": str(stored_command_id),
                            "revision": 1,
                            "revision_history": [],
                            "suggestion_id": "editor-suggestion-mismatch",
                            "updated_at": database.now,
                            "warnings": [],
                        },
                        created_at=database.now,
                        updated_at=database.now,
                    )
                )
    instance, _lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
        permission="suggest",
    )
    descriptor = CollaborationSuggestion(
        suggestion_id=suggestion_id,
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        kind="replacement",
    )
    patch_state = CollaborationPatchState(
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        active_suggestion_ids=(suggestion_id,),
        kinds=("replacement",),
    )

    with pytest.raises(CollaborationConflict) as conflict:
        await database.store.append_update(
            _assistant_suggestion_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                payload=b"publish-without-private-draft",
                suggestions=(descriptor,),
                patches=(patch_state,),
                command_id=request_command_id,
                now=database.now + 1.0,
                expected_sequence=0,
            )
        )

    assert conflict.value.reason == "patch_not_found"
    async with database.session_factory() as session:
        patch_count = await session.scalar(
            select(func.count()).select_from(editor_patches).where(
                editor_patches.c.tenant_id == TENANT_A,
                editor_patches.c.patch_id == patch_id,
            )
        )
        update_count = await session.scalar(
            select(func.count()).select_from(editor_collaboration_updates).where(
                editor_collaboration_updates.c.tenant_id == TENANT_A,
                editor_collaboration_updates.c.document_id == document_id,
            )
        )
        persisted_sequence = await session.scalar(
            select(editor_documents.c.persisted_sequence).where(
                editor_documents.c.tenant_id == TENANT_A,
                editor_documents.c.id == document_id,
            )
        )
        draft_count = await session.scalar(
            select(func.count()).select_from(editor_comments).where(
                editor_comments.c.tenant_id == TENANT_A,
                editor_comments.c.document_id == document_id,
                editor_comments.c.suggestion_draft.is_not(None),
            )
        )
    assert patch_count == 0
    assert update_count == 0
    assert persisted_sequence == 0
    assert draft_count == (1 if stored_command_id is not None else 0)


@pytest.mark.asyncio
async def test_slash_insertion_can_be_superseded_by_one_structure_suggestion(
    database: _DatabaseHarness,
) -> None:
    """A sidecar-proven slash command may replace its durable patch member."""
    document_id = "ed_collaboration_slash_structure"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
        permission="suggest",
    )
    patch_id, slash_id = await _create_human_patch(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        actor_user_id=OWNER_A,
        instance=instance,
        lease=lease,
    )
    structure_id = str(uuid.uuid4())
    slash_descriptor = CollaborationSuggestion(
        suggestion_id=slash_id,
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        kind="insertion",
    )
    structure_descriptor = CollaborationSuggestion(
        suggestion_id=structure_id,
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        kind="structure",
    )
    replacement = CollaborationPatchState(
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        active_suggestion_ids=(structure_id,),
        kinds=("structure",),
        superseded_suggestion_ids=(slash_id,),
    )

    with pytest.raises(CollaborationConflict) as unproven:
        await database.store.append_update(
            _suggestion_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=lease,
                payload=b"unproven-slash-to-structure",
                suggestions=(slash_descriptor, structure_descriptor),
                patches=(replace(replacement, superseded_suggestion_ids=()),),
                now=database.now + 2.0,
                expected_sequence=1,
            )
        )
    assert unproven.value.reason == "patch_membership_shrink"

    result = await database.store.append_update(
        _suggestion_update(
            tenant_id=TENANT_A,
            document_id=document_id,
            actor_user_id=OWNER_A,
            instance=instance,
            lease=lease,
            payload=b"slash-to-structure",
            suggestions=(slash_descriptor, structure_descriptor),
            patches=(replacement,),
            now=database.now + 2.0,
            expected_sequence=1,
        )
    )

    assert result.sequence == 2
    async with database.session_factory() as session:
        patch = (
            await session.execute(
                select(editor_patches).where(editor_patches.c.patch_id == patch_id)
            )
        ).mappings().one()
        update_row = (
            await session.execute(
                select(editor_collaboration_updates)
                .where(
                    editor_collaboration_updates.c.document_id == document_id,
                    editor_collaboration_updates.c.sequence == 2,
                )
            )
        ).mappings().one()
    assert patch["suggestion_ids"] == [structure_id]
    assert patch["edits"] == [
        {
            "suggestion_id": structure_id,
            "patch_id": patch_id,
            "author_id": str(OWNER_A),
            "created_at": database.now + 1.0,
            "kind": "structure",
        }
    ]
    assert update_row["suggestion_ids"] == [slash_id, structure_id]


@pytest.mark.asyncio
async def test_non_decision_update_cannot_empty_pending_patch_membership(
    database: _DatabaseHarness,
) -> None:
    """Postgres rejects a compromised sidecar's orphaned pending patch."""
    document_id = "ed_collaboration_empty_pending_patch"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
        permission="suggest",
    )
    patch_id, suggestion_id = await _create_human_patch(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        actor_user_id=OWNER_A,
        instance=instance,
        lease=lease,
    )
    emptied_patch = CollaborationPatchState(
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        active_suggestion_ids=(),
        kinds=(),
    )

    with pytest.raises(CollaborationConflict) as conflict:
        await database.store.append_update(
            _suggestion_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=lease,
                payload=b"empty-pending-patch",
                suggestions=(),
                patches=(emptied_patch,),
                now=database.now + 2.0,
                expected_sequence=1,
            )
        )

    assert conflict.value.reason == "patch_suggestions_empty"
    async with database.session_factory() as session:
        patch = (
            await session.execute(
                select(editor_patches).where(editor_patches.c.patch_id == patch_id)
            )
        ).mappings().one()
        update_count = await session.scalar(
            select(func.count())
            .select_from(editor_collaboration_updates)
            .where(editor_collaboration_updates.c.document_id == document_id)
        )
        persisted_sequence = await session.scalar(
            select(editor_documents.c.persisted_sequence).where(
                editor_documents.c.id == document_id
            )
        )
    assert patch["status"] == "pending"
    assert patch["suggestion_ids"] == [suggestion_id]
    assert update_count == persisted_sequence == 1


@pytest.mark.asyncio
async def test_open_patch_preserves_exact_ai_edits_for_preview(
    database: _DatabaseHarness,
) -> None:
    """Publishing suggestion metadata never overwrites anchored AI edits."""
    document_id = "ed_collaboration_exact_preview"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    patch_id = str(uuid.uuid4())
    suggestion_id = str(uuid.uuid4())
    exact_edit = {
        "id": "edit-exact-1",
        "find": "old text",
        "text": "new text",
        "position": "replace",
        "quote_before": "before",
        "quote_after": "after",
    }
    async with database.session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(editor_patches).values(
                    patch_id=patch_id,
                    tenant_id=TENANT_A,
                    document_id=document_id,
                    run_id=None,
                    source="agent",
                    status="pending",
                    edits=[exact_edit],
                    summary="Exact preview",
                    warnings=[],
                    revision_before=DOCUMENT_REVISION,
                    collaboration_generation=1,
                    base_sequence=0,
                    decision_sequence=None,
                    suggestion_ids=[],
                    applied_revision=None,
                    applied_edit_ids=None,
                    note="",
                    created_by_user_id=OWNER_A,
                    decided_by_user_id=None,
                    command_id=None,
                    created_at=database.now + 1.0,
                    decided_at=None,
                )
            )
    private_page = await database.store.list_open_patches(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        before=None,
        author_user_id=None,
        suggestion_kind=None,
        limit=50,
    )
    assert private_page.patches == ()
    descriptor = CollaborationSuggestion(
        suggestion_id=suggestion_id,
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        kind="replacement",
    )
    patch_state = CollaborationPatchState(
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        active_suggestion_ids=(suggestion_id,),
        kinds=("replacement",),
    )
    await database.store.append_update(
        _suggestion_update(
            tenant_id=TENANT_A,
            document_id=document_id,
            actor_user_id=OWNER_A,
            instance=instance,
            lease=lease,
            payload=b"publish-exact-preview",
            suggestions=(descriptor,),
            patches=(patch_state,),
            now=database.now + 2.0,
            expected_sequence=0,
        )
    )

    open_page = await database.store.list_open_patches(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        before=None,
        author_user_id=OWNER_A,
        suggestion_kind="replacement",
        limit=50,
    )
    open_patches = open_page.patches

    assert len(open_patches) == 1
    assert open_patches[0].suggestion_ids == (suggestion_id,)
    assert open_patches[0].kinds == ("replacement",)
    assert open_patches[0].exact_edits == (exact_edit,)


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    (("accept", "accepted"), ("reject", "rejected")),
)
@pytest.mark.asyncio
async def test_decision_update_co_commits_patch_terminal_state(
    database: _DatabaseHarness,
    decision: CollaborationDecision,
    expected_status: str,
) -> None:
    """Accept and reject decisions terminalize their patch at the update sequence."""
    document_id = f"ed_collaboration_decision_{decision}"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    patch_id, suggestion_id = await _create_human_patch(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        actor_user_id=OWNER_A,
        instance=instance,
        lease=lease,
    )
    command_id = uuid.uuid4()
    payload = f"decision:{decision}:{patch_id}".encode()
    command_payload_hash = _sha256(
        f"decision-command:{decision}:{patch_id}".encode()
    )
    decision_update = PersistCollaborationUpdate(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        instance_id=instance.instance_id,
        instance_epoch=instance.epoch,
        lease_id=lease.lease_id,
        actor_user_id=OWNER_A,
        update_hash=_sha256(payload),
        update_bytes=payload,
        actor_kind="human",
        change_kind="decision",
        patches=(
            CollaborationPatchState(
                patch_id=patch_id,
                author_id=OWNER_A,
                created_at=database.now + 1.0,
                active_suggestion_ids=(),
                kinds=(),
            ),
        ),
        decision=decision,
        command_id=command_id,
        command_payload_hash=command_payload_hash,
        expected_sequence=1,
        now=database.now + 2.0,
    )
    decided = await database.store.append_update(decision_update)
    replay = await database.store.append_update(decision_update)
    assert decided.sequence == replay.sequence == 2
    assert decided.duplicate is False
    assert replay.duplicate is True

    async with database.session_factory() as session:
        patch = (
            await session.execute(
                select(editor_patches).where(editor_patches.c.patch_id == patch_id)
            )
        ).mappings().one()
        decision_row = (
            await session.execute(
                select(editor_collaboration_updates).where(
                    editor_collaboration_updates.c.document_id == document_id,
                    editor_collaboration_updates.c.sequence == 2,
                )
            )
        ).one()
        update_count = await session.scalar(
            select(func.count())
            .select_from(editor_collaboration_updates)
            .where(editor_collaboration_updates.c.document_id == document_id)
        )
    assert patch["status"] == expected_status
    assert patch["decision_sequence"] == 2
    assert patch["decided_by_user_id"] == OWNER_A
    assert patch["command_id"] == command_id
    assert patch["applied_edit_ids"] == (
        [suggestion_id] if decision == "accept" else None
    )
    assert decision_row.change_kind == "decision"
    assert decision_row.command_id == command_id
    assert decision_row.command_payload_hash == command_payload_hash
    assert update_count == 2

    lookup = await database.store.lookup_command(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        command_id=command_id,
        command_payload_hash=command_payload_hash,
    )
    assert lookup is not None
    assert lookup.sequence == 2
    assert lookup.decision == decision
    assert lookup.patch_ids == (patch_id,)


@pytest.mark.asyncio
async def test_snapshot_and_projection_commit_together_and_conflict_rolls_back(
    database: _DatabaseHarness,
) -> None:
    """Advance snapshot and projection atomically; roll back a conflict."""
    document_id = "ed_collaboration_snapshot"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    await database.store.append_update(
        _direct_update(
            tenant_id=TENANT_A,
            document_id=document_id,
            actor_user_id=OWNER_A,
            instance=instance,
            lease=lease,
            payload=b"snapshot-tail",
            now=database.now + 1.0,
        )
    )
    projection = "# Projected\n\nDurable body."
    snapshot = _snapshot(
        tenant_id=TENANT_A,
        document_id=document_id,
        sequence=1,
        state_update=b"snapshot-state-one",
        projection=projection,
        created_at=database.now + 2.0,
    )
    await database.store.store_snapshot(
        snapshot,
        projection_markdown=projection,
        instance_id=instance.instance_id,
        instance_epoch=instance.epoch,
        now=database.now + 2.0,
    )

    async with database.session_factory() as session:
        document = (
            await session.execute(
                select(editor_documents).where(editor_documents.c.id == document_id)
            )
        ).one()
        stored_snapshot = (
            await session.execute(
                select(editor_collaboration_snapshots).where(
                    editor_collaboration_snapshots.c.document_id == document_id,
                    editor_collaboration_snapshots.c.covered_sequence == 1,
                )
            )
        ).one()
    assert document.content_markdown == projection
    assert document.projection_sequence == 1
    assert document.projection_updated_at == database.now + 2.0
    assert stored_snapshot.projection_hash == _sha256(projection.encode())

    conflicting_projection = "# Conflicting projection"
    conflict_snapshot = _snapshot(
        tenant_id=TENANT_A,
        document_id=document_id,
        sequence=1,
        state_update=b"different-state-at-one",
        projection=conflicting_projection,
        created_at=database.now + 3.0,
    )
    with pytest.raises(CollaborationConflict) as conflict:
        await database.store.store_snapshot(
            conflict_snapshot,
            projection_markdown=conflicting_projection,
            instance_id=instance.instance_id,
            instance_epoch=instance.epoch,
            now=database.now + 3.0,
        )
    assert conflict.value.reason == "snapshot_conflict"

    async with database.session_factory() as session:
        unchanged = (
            await session.execute(
                select(editor_documents).where(editor_documents.c.id == document_id)
            )
        ).one()
        sequence_one_count = await session.scalar(
            select(func.count())
            .select_from(editor_collaboration_snapshots)
            .where(
                editor_collaboration_snapshots.c.document_id == document_id,
                editor_collaboration_snapshots.c.covered_sequence == 1,
            )
        )
    assert unchanged.content_markdown == projection
    assert unchanged.projection_sequence == 1
    assert sequence_one_count == 1


@pytest.mark.asyncio
async def test_load_state_returns_two_complete_snapshot_candidates(
    database: _DatabaseHarness,
) -> None:
    """A corrupt newest snapshot can be skipped without losing its prior tail."""
    document_id = "ed_collaboration_snapshot_fallback"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    for sequence in (1, 2):
        await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=lease,
                payload=f"fallback-update-{sequence}".encode(),
                now=database.now + sequence,
            )
        )
        if sequence == 1:
            projection = "# Snapshot one"
            await database.store.store_snapshot(
                _snapshot(
                    tenant_id=TENANT_A,
                    document_id=document_id,
                    sequence=1,
                    state_update=b"snapshot-one",
                    projection=projection,
                    created_at=database.now + 1.5,
                ),
                projection_markdown=projection,
                instance_id=instance.instance_id,
                instance_epoch=instance.epoch,
                now=database.now + 1.5,
            )

    loaded = await database.store.load_state(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
    )

    assert loaded.snapshot.covered_sequence == 1
    assert [item.sequence for item in loaded.updates] == [2]
    assert len(loaded.fallback_candidates) == 1
    fallback = loaded.fallback_candidates[0]
    assert fallback.snapshot.covered_sequence == 0
    assert [item.sequence for item in fallback.updates] == [1, 2]


@pytest.mark.asyncio
async def test_compaction_retention_and_tombstone_purge_are_bounded(
    database: _DatabaseHarness,
) -> None:
    """Compaction keeps two snapshots, ages payloads before metadata, then purges."""
    document_id = "ed_collaboration_retention"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
        instance_lease_seconds=10_000.0,
    )
    for sequence in (1, 2):
        await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=lease,
                payload=f"retained-update-{sequence}".encode(),
                now=database.now + sequence,
            )
        )
        projection = f"# Projection {sequence}"
        await database.store.store_snapshot(
            _snapshot(
                tenant_id=TENANT_A,
                document_id=document_id,
                sequence=sequence,
                state_update=f"state-{sequence}".encode(),
                projection=projection,
                created_at=database.now + sequence + 0.25,
            ),
            projection_markdown=projection,
            instance_id=instance.instance_id,
            instance_epoch=instance.epoch,
            now=database.now + sequence + 0.25,
        )

    pruned, deleted = await database.store.compact(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        instance_id=instance.instance_id,
        instance_epoch=instance.epoch,
        now=database.now + 100.0,
        payload_retention_seconds=50.0,
        metadata_retention_seconds=1_000.0,
    )
    assert (pruned, deleted) == (1, 0)
    async with database.session_factory() as session:
        updates = (
            await session.execute(
                select(editor_collaboration_updates)
                .where(editor_collaboration_updates.c.document_id == document_id)
                .order_by(editor_collaboration_updates.c.sequence)
            )
        ).all()
        snapshot_sequences = list(
            (
                await session.execute(
                    select(editor_collaboration_snapshots.c.covered_sequence)
                    .where(
                        editor_collaboration_snapshots.c.document_id == document_id
                    )
                    .order_by(
                        editor_collaboration_snapshots.c.covered_sequence.desc()
                    )
                )
            ).scalars()
        )
    assert [row.update_bytes for row in updates] == [
        None,
        b"retained-update-2",
    ]
    assert updates[0].payload_pruned_at == database.now + 100.0
    assert updates[1].payload_pruned_at is None
    assert snapshot_sequences == [2, 1]

    second_pruned, metadata_deleted = await database.store.compact(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        instance_id=instance.instance_id,
        instance_epoch=instance.epoch,
        now=database.now + 5_000.0,
        payload_retention_seconds=50.0,
        metadata_retention_seconds=1_000.0,
    )
    assert second_pruned == 0
    assert metadata_deleted == 1

    next_generation = await database.store.tombstone_document(
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
        now=database.now + 6_000.0,
    )
    assert next_generation == 2
    assert (
        await database.store.purge_tombstones(
            tenant_id=TENANT_A,
            instance_id=instance.instance_id,
            instance_epoch=instance.epoch,
            now=database.now + 6_050.0,
            retention_seconds=100.0,
        )
        == 0
    )
    assert (
        await database.store.purge_tombstones(
            tenant_id=TENANT_A,
            instance_id=instance.instance_id,
            instance_epoch=instance.epoch,
            now=database.now + 6_200.0,
            retention_seconds=100.0,
        )
        == 1
    )

    async with database.session_factory() as session:
        document_count = await session.scalar(
            select(func.count())
            .select_from(editor_documents)
            .where(editor_documents.c.id == document_id)
        )
        snapshot_count = await session.scalar(
            select(func.count())
            .select_from(editor_collaboration_snapshots)
            .where(editor_collaboration_snapshots.c.document_id == document_id)
        )
        lease_count = await session.scalar(
            select(func.count())
            .select_from(editor_collaboration_leases)
            .where(editor_collaboration_leases.c.document_id == document_id)
        )
    assert document_count == snapshot_count == lease_count == 0


@pytest.mark.asyncio
async def test_rotation_reconstructs_one_replacement_after_a_lost_response(
    database: _DatabaseHarness,
) -> None:
    """A durable rotation command returns its first replacement on retry."""
    document_id = "ed_collaboration_rotation_retry"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    _instance, original = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    rotation_command_id = uuid.uuid4()
    replacement = CollaborationLease(
        lease_id=uuid.uuid4(),
        token_hash=_sha256(b"replacement-token-one"),
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        user_id=OWNER_A,
        permission="edit",
        session_id=original.session_id,
        issued_at=database.now + 1.0,
        expires_at=database.now + 3_601.0,
        last_validated_at=database.now + 1.0,
        rotation_command_id=rotation_command_id,
        rotated_from_lease_id=original.lease_id,
    )

    first = await database.store.rotate_lease(
        previous_lease_id=original.lease_id,
        previous_token_hash=original.token_hash,
        replacement=replacement,
        max_issued_per_window=30,
        issued_since=database.now - 60.0,
    )
    replay_candidate = replace(
        replacement,
        lease_id=uuid.uuid4(),
        token_hash=_sha256(b"replacement-token-two"),
        issued_at=database.now + 2.0,
        expires_at=database.now + 3_602.0,
        last_validated_at=database.now + 2.0,
    )
    replayed = await database.store.rotate_lease(
        previous_lease_id=original.lease_id,
        previous_token_hash=original.token_hash,
        replacement=replay_candidate,
        max_issued_per_window=30,
        issued_since=database.now - 60.0,
    )

    assert replayed == first == replacement
    async with database.session_factory() as session:
        rows = (
            await session.execute(
                select(editor_collaboration_leases)
                .where(
                    editor_collaboration_leases.c.document_id == document_id
                )
                .order_by(editor_collaboration_leases.c.issued_at)
            )
        ).all()
    assert len(rows) == 2
    assert rows[0].revoked_at == database.now + 1.0
    assert rows[1].rotation_command_id == rotation_command_id


@pytest.mark.asyncio
async def test_update_started_before_rotation_uses_immediate_successor_authority(
    database: _DatabaseHarness,
) -> None:
    """An in-flight update survives an equal-authority lease rotation."""
    document_id = "ed_collaboration_inflight_rotation"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, original = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    update_started_before_rotation = _direct_update(
        tenant_id=TENANT_A,
        document_id=document_id,
        actor_user_id=OWNER_A,
        instance=instance,
        lease=original,
        payload=b"inflight-update-across-rotation",
        now=database.now + 1.5,
    )
    rotation_command_id = uuid.uuid4()
    successor = replace(
        original,
        lease_id=uuid.uuid4(),
        token_hash=_sha256(b"inflight-rotation-successor"),
        issued_at=database.now + 1.0,
        expires_at=database.now + 3_601.0,
        last_validated_at=database.now + 1.0,
        rotation_command_id=rotation_command_id,
        rotated_from_lease_id=original.lease_id,
    )
    await database.store.rotate_lease(
        previous_lease_id=original.lease_id,
        previous_token_hash=original.token_hash,
        replacement=successor,
        max_issued_per_window=30,
        issued_since=database.now - 60.0,
    )

    persisted = await database.store.append_update(
        update_started_before_rotation
    )
    replayed = await database.store.append_update(
        update_started_before_rotation
    )

    assert persisted.sequence == 1
    assert persisted.persisted_sequence == 1
    assert persisted.duplicate is False
    assert replayed.sequence == 1
    assert replayed.persisted_sequence == 1
    assert replayed.duplicate is True


@pytest.mark.asyncio
async def test_update_started_before_rotation_obeys_successor_permission(
    database: _DatabaseHarness,
) -> None:
    """A rotated read-only authority cannot complete an earlier edit."""
    document_id = "ed_collaboration_inflight_rotation_downgrade"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, original = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    successor = replace(
        original,
        lease_id=uuid.uuid4(),
        token_hash=_sha256(b"inflight-downgraded-successor"),
        permission="view",
        issued_at=database.now + 1.0,
        expires_at=database.now + 3_601.0,
        last_validated_at=database.now + 1.0,
        rotation_command_id=uuid.uuid4(),
        rotated_from_lease_id=original.lease_id,
    )
    await database.store.rotate_lease(
        previous_lease_id=original.lease_id,
        previous_token_hash=original.token_hash,
        replacement=successor,
        max_issued_per_window=30,
        issued_since=database.now - 60.0,
    )

    with pytest.raises(CollaborationLeaseInvalid) as denied:
        await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=original,
                payload=b"inflight-edit-after-downgrade",
                now=database.now + 1.5,
            )
        )
    assert str(denied.value) == "permission_denied"


@pytest.mark.asyncio
async def test_update_started_before_rotation_does_not_follow_a_lease_chain(
    database: _DatabaseHarness,
) -> None:
    """Only an active direct successor can authorize an in-flight update."""
    document_id = "ed_collaboration_inflight_rotation_chain"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, original = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    first_successor = replace(
        original,
        lease_id=uuid.uuid4(),
        token_hash=_sha256(b"inflight-first-successor"),
        issued_at=database.now + 1.0,
        expires_at=database.now + 3_601.0,
        last_validated_at=database.now + 1.0,
        rotation_command_id=uuid.uuid4(),
        rotated_from_lease_id=original.lease_id,
    )
    await database.store.rotate_lease(
        previous_lease_id=original.lease_id,
        previous_token_hash=original.token_hash,
        replacement=first_successor,
        max_issued_per_window=30,
        issued_since=database.now - 60.0,
    )
    second_successor = replace(
        first_successor,
        lease_id=uuid.uuid4(),
        token_hash=_sha256(b"inflight-second-successor"),
        issued_at=database.now + 2.0,
        expires_at=database.now + 3_602.0,
        last_validated_at=database.now + 2.0,
        rotation_command_id=uuid.uuid4(),
        rotated_from_lease_id=first_successor.lease_id,
    )
    await database.store.rotate_lease(
        previous_lease_id=first_successor.lease_id,
        previous_token_hash=first_successor.token_hash,
        replacement=second_successor,
        max_issued_per_window=30,
        issued_since=database.now - 60.0,
    )

    with pytest.raises(CollaborationLeaseInvalid):
        await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=original,
                payload=b"inflight-update-after-second-rotation",
                now=database.now + 2.5,
            )
        )


@pytest.mark.asyncio
async def test_update_started_before_rotation_requires_live_original_lifetime(
    database: _DatabaseHarness,
) -> None:
    """A successor cannot extend the lifetime of a stale captured lease."""
    document_id = "ed_collaboration_inflight_rotation_expired"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, original = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    successor = replace(
        original,
        lease_id=uuid.uuid4(),
        token_hash=_sha256(b"inflight-expired-original-successor"),
        issued_at=database.now + 3_599.0,
        expires_at=database.now + 7_199.0,
        last_validated_at=database.now + 3_599.0,
        rotation_command_id=uuid.uuid4(),
        rotated_from_lease_id=original.lease_id,
    )
    await database.store.rotate_lease(
        previous_lease_id=original.lease_id,
        previous_token_hash=original.token_hash,
        replacement=successor,
        max_issued_per_window=30,
        issued_since=database.now - 60.0,
    )

    with pytest.raises(CollaborationLeaseInvalid):
        await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=original,
                payload=b"inflight-update-after-original-expiry",
                now=database.now + 3_600.5,
            )
        )


@pytest.mark.asyncio
async def test_update_started_before_rotation_rejects_revoked_successor(
    database: _DatabaseHarness,
) -> None:
    """Explicit revocation still blocks a captured predecessor lease."""
    document_id = "ed_collaboration_inflight_rotation_revoked"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, original = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    successor = replace(
        original,
        lease_id=uuid.uuid4(),
        token_hash=_sha256(b"inflight-revoked-successor"),
        issued_at=database.now + 1.0,
        expires_at=database.now + 3_601.0,
        last_validated_at=database.now + 1.0,
        rotation_command_id=uuid.uuid4(),
        rotated_from_lease_id=original.lease_id,
    )
    await database.store.rotate_lease(
        previous_lease_id=original.lease_id,
        previous_token_hash=original.token_hash,
        replacement=successor,
        max_issued_per_window=30,
        issued_since=database.now - 60.0,
    )
    assert (
        await database.store.revoke_leases(
            tenant_id=TENANT_A,
            document_id=document_id,
            user_id=OWNER_A,
            now=database.now + 1.25,
        )
        == 1
    )

    with pytest.raises(CollaborationLeaseInvalid):
        await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=original,
                payload=b"inflight-update-after-explicit-revocation",
                now=database.now + 1.5,
            )
        )


@pytest.mark.asyncio
async def test_update_started_before_rotation_requires_live_session(
    database: _DatabaseHarness,
) -> None:
    """A direct successor never bypasses current account-session checks."""
    document_id = "ed_collaboration_inflight_rotation_session"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, original = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
    )
    successor = replace(
        original,
        lease_id=uuid.uuid4(),
        token_hash=_sha256(b"inflight-session-successor"),
        issued_at=database.now + 1.0,
        expires_at=database.now + 3_601.0,
        last_validated_at=database.now + 1.0,
        rotation_command_id=uuid.uuid4(),
        rotated_from_lease_id=original.lease_id,
    )
    await database.store.rotate_lease(
        previous_lease_id=original.lease_id,
        previous_token_hash=original.token_hash,
        replacement=successor,
        max_issued_per_window=30,
        issued_since=database.now - 60.0,
    )
    async with database.session_factory() as session:
        async with session.begin():
            await session.execute(
                update(auth_sessions)
                .where(auth_sessions.c.id == original.session_id)
                .values(expires_at=database.now + 1.25)
            )

    with pytest.raises(CollaborationLeaseInvalid) as denied:
        await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=original,
                payload=b"inflight-update-after-session-expiry",
                now=database.now + 1.5,
            )
        )
    assert str(denied.value) == "session_invalid"


@pytest.mark.asyncio
async def test_session_rate_limit_serializes_one_user_across_documents(
    database: _DatabaseHarness,
) -> None:
    """Concurrent cold requests cannot bypass the user-wide issuance window."""
    document_ids = (
        "ed_collaboration_rate_a",
        "ed_collaboration_rate_b",
    )
    for document_id in document_ids:
        await _seed_markdown_document(
            database,
            tenant_id=TENANT_A,
            document_id=document_id,
            owner_user_id=OWNER_A,
        )
        await _activate_document(
            database,
            tenant_id=TENANT_A,
            document_id=document_id,
            owner_user_id=OWNER_A,
        )
    session_id = "session:collaboration-rate-across-documents"
    await _seed_browser_session(
        database,
        tenant_id=TENANT_A,
        user_id=OWNER_A,
        session_id=session_id,
    )
    leases = tuple(
        CollaborationLease(
            lease_id=uuid.uuid4(),
            token_hash=_sha256(f"rate-token:{document_id}".encode()),
            tenant_id=TENANT_A,
            document_id=document_id,
            generation=1,
            user_id=OWNER_A,
            permission="edit",
            session_id=session_id,
            issued_at=database.now,
            expires_at=database.now + 3_600.0,
            last_validated_at=database.now,
        )
        for document_id in document_ids
    )

    results = await asyncio.gather(
        *(
            database.store.issue_lease(
                lease,
                max_active=5,
                max_issued_per_window=1,
                issued_since=database.now - 60.0,
            )
            for lease in leases
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, CollaborationLease) for result in results) == 1
    rate_errors = [
        result for result in results if isinstance(result, CollaborationRateLimited)
    ]
    assert [error.reason for error in rate_errors] == ["session_rate_limited"]
    async with database.session_factory() as session:
        lease_count = await session.scalar(
            select(func.count())
            .select_from(editor_collaboration_leases)
            .where(editor_collaboration_leases.c.tenant_id == TENANT_A)
        )
    assert lease_count == 1


@pytest.mark.asyncio
async def test_simultaneous_cold_instance_acquire_has_one_fenced_loser(
    database: _DatabaseHarness,
) -> None:
    """The first primary-slot insert is stable under a true cold-start race."""
    results = await asyncio.gather(
        database.store.acquire_instance(
            tenant_id=TENANT_A,
            instance_id="node-cold-a",
            now=database.now,
            lease_seconds=15.0,
        ),
        database.store.acquire_instance(
            tenant_id=TENANT_A,
            instance_id="node-cold-b",
            now=database.now,
            lease_seconds=15.0,
        ),
        return_exceptions=True,
    )

    assert sum(
        isinstance(result, CollaborationInstanceLease) for result in results
    ) == 1
    assert sum(
        isinstance(result, CollaborationInstanceFenced) for result in results
    ) == 1


@pytest.mark.asyncio
async def test_takeover_fences_snapshot_compaction_and_tombstone_purge(
    database: _DatabaseHarness,
) -> None:
    """A takeover committed first prevents every stale maintenance mutation."""
    document_id = "ed_collaboration_atomic_fence"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    stale_instance, _lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
        instance_id="node-stale",
        instance_lease_seconds=10.0,
    )
    await database.store.tombstone_document(
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
        now=database.now + 2.0,
    )
    current_instance = await database.store.acquire_instance(
        tenant_id=TENANT_A,
        instance_id="node-current",
        now=database.now + 11.0,
        lease_seconds=15.0,
    )
    assert current_instance.epoch == stale_instance.epoch + 1

    with pytest.raises(CollaborationInstanceFenced):
        await database.store.store_snapshot(
            _snapshot(
                tenant_id=TENANT_A,
                document_id=document_id,
                sequence=0,
                state_update=b"stale-snapshot",
                projection="# Stale",
                created_at=database.now + 12.0,
            ),
            projection_markdown="# Stale",
            instance_id=stale_instance.instance_id,
            instance_epoch=stale_instance.epoch,
            now=database.now + 12.0,
        )
    with pytest.raises(CollaborationInstanceFenced):
        await database.store.compact(
            tenant_id=TENANT_A,
            document_id=document_id,
            generation=1,
            instance_id=stale_instance.instance_id,
            instance_epoch=stale_instance.epoch,
            now=database.now + 12.0,
            payload_retention_seconds=0.0,
            metadata_retention_seconds=1.0,
        )
    with pytest.raises(CollaborationInstanceFenced):
        await database.store.purge_tombstones(
            tenant_id=TENANT_A,
            instance_id=stale_instance.instance_id,
            instance_epoch=stale_instance.epoch,
            now=database.now + 12.0,
            retention_seconds=1.0,
        )

    async with database.session_factory() as session:
        document_count = await session.scalar(
            select(func.count())
            .select_from(editor_documents)
            .where(editor_documents.c.id == document_id)
        )
        stale_snapshot_count = await session.scalar(
            select(func.count())
            .select_from(editor_collaboration_snapshots)
            .where(
                editor_collaboration_snapshots.c.document_id == document_id,
                editor_collaboration_snapshots.c.state_hash
                == _sha256(b"stale-snapshot"),
            )
        )
    assert document_count == 1
    assert stale_snapshot_count == 0


@pytest.mark.parametrize("mutation", ["snapshot", "compact", "purge"])
@pytest.mark.asyncio
async def test_takeover_waits_for_inflight_fenced_mutation_transaction(
    database: _DatabaseHarness,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Fence validation remains locked through each maintenance mutation."""
    document_id = f"ed_collaboration_fence_race_{mutation}"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    stale_instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
        instance_id="node-race-stale",
        instance_lease_seconds=10.0,
    )
    if mutation in {"snapshot", "compact"}:
        persisted = await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=stale_instance,
                lease=lease,
                payload=f"fence-race-{mutation}".encode(),
                now=database.now + 1.0,
            )
        )
        assert persisted.sequence == 1
    if mutation == "compact":
        await database.store.store_snapshot(
            _snapshot(
                tenant_id=TENANT_A,
                document_id=document_id,
                sequence=1,
                state_update=b"compact-race-snapshot-one",
                projection="# Compact one",
                created_at=database.now + 1.5,
            ),
            projection_markdown="# Compact one",
            instance_id=stale_instance.instance_id,
            instance_epoch=stale_instance.epoch,
            now=database.now + 1.5,
        )
        second = await database.store.append_update(
            _direct_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=stale_instance,
                lease=lease,
                payload=b"fence-race-compact-second",
                now=database.now + 2.0,
            )
        )
        assert second.sequence == 2
        await database.store.store_snapshot(
            _snapshot(
                tenant_id=TENANT_A,
                document_id=document_id,
                sequence=2,
                state_update=b"compact-race-snapshot-two",
                projection="# Compact two",
                created_at=database.now + 2.5,
            ),
            projection_markdown="# Compact two",
            instance_id=stale_instance.instance_id,
            instance_epoch=stale_instance.epoch,
            now=database.now + 2.5,
        )
    if mutation == "purge":
        await database.store.tombstone_document(
            tenant_id=TENANT_A,
            document_id=document_id,
            owner_user_id=OWNER_A,
            now=database.now + 2.0,
        )

    fence_locked = asyncio.Event()
    release_mutation = asyncio.Event()
    lock_instance_fence = collaboration_postgres._lock_instance_fence

    async def pause_after_fence_lock(
        session: AsyncSession,
        *,
        tenant_id: str,
        instance_id: str,
        instance_epoch: int,
        now: float,
    ) -> object:
        locked = await lock_instance_fence(
            session,
            tenant_id=tenant_id,
            instance_id=instance_id,
            instance_epoch=instance_epoch,
            now=now,
        )
        fence_locked.set()
        await release_mutation.wait()
        return locked

    monkeypatch.setattr(
        collaboration_postgres,
        "_lock_instance_fence",
        pause_after_fence_lock,
    )
    if mutation == "snapshot":
        mutation_call = database.store.store_snapshot(
            _snapshot(
                tenant_id=TENANT_A,
                document_id=document_id,
                sequence=1,
                state_update=b"race-snapshot",
                projection="# Race snapshot",
                created_at=database.now + 3.0,
            ),
            projection_markdown="# Race snapshot",
            instance_id=stale_instance.instance_id,
            instance_epoch=stale_instance.epoch,
            now=database.now + 3.0,
        )
    elif mutation == "compact":
        mutation_call = database.store.compact(
            tenant_id=TENANT_A,
            document_id=document_id,
            generation=1,
            instance_id=stale_instance.instance_id,
            instance_epoch=stale_instance.epoch,
            now=database.now + 3.0,
            payload_retention_seconds=0.0,
            metadata_retention_seconds=3_600.0,
        )
    else:
        mutation_call = database.store.purge_tombstones(
            tenant_id=TENANT_A,
            instance_id=stale_instance.instance_id,
            instance_epoch=stale_instance.epoch,
            now=database.now + 4.0,
            retention_seconds=1.0,
        )

    mutation_task = asyncio.create_task(mutation_call)
    await asyncio.wait_for(fence_locked.wait(), timeout=1.0)
    takeover_task = asyncio.create_task(
        database.store.acquire_instance(
            tenant_id=TENANT_A,
            instance_id="node-race-current",
            now=database.now + 11.0,
            lease_seconds=15.0,
        )
    )
    await asyncio.sleep(0.05)
    takeover_was_blocked = not takeover_task.done()
    release_mutation.set()
    await mutation_task
    current_instance = await takeover_task

    assert takeover_was_blocked
    assert current_instance.epoch == stale_instance.epoch + 1
    async with database.session_factory() as session:
        if mutation == "snapshot":
            stored = await session.scalar(
                select(func.count())
                .select_from(editor_collaboration_snapshots)
                .where(
                    editor_collaboration_snapshots.c.document_id == document_id,
                    editor_collaboration_snapshots.c.state_hash
                    == _sha256(b"race-snapshot"),
                )
            )
            assert stored == 1
        elif mutation == "compact":
            payload = await session.scalar(
                select(editor_collaboration_updates.c.update_bytes).where(
                    editor_collaboration_updates.c.document_id == document_id,
                    editor_collaboration_updates.c.sequence == 1,
                )
            )
            assert payload is None
        else:
            remaining = await session.scalar(
                select(func.count())
                .select_from(editor_documents)
                .where(editor_documents.c.id == document_id)
            )
            assert remaining == 0


@pytest.mark.asyncio
async def test_patch_membership_cannot_shrink_or_change_under_another_actor(
    database: _DatabaseHarness,
) -> None:
    """Only the patch author may monotonically add active suggestion IDs."""
    document_id = "ed_collaboration_patch_membership"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    instance, lease = await _authorize_writer(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        user_id=OWNER_A,
        permission="suggest",
    )
    patch_id, first_id = await _create_human_patch(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        actor_user_id=OWNER_A,
        instance=instance,
        lease=lease,
    )
    second_id = str(uuid.uuid4())
    second_descriptor = CollaborationSuggestion(
        suggestion_id=second_id,
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 2.0,
        kind="insertion",
    )
    expanded = CollaborationPatchState(
        patch_id=patch_id,
        author_id=OWNER_A,
        created_at=database.now + 1.0,
        active_suggestion_ids=(first_id, second_id),
        kinds=("insertion",),
    )
    await database.store.append_update(
        _suggestion_update(
            tenant_id=TENANT_A,
            document_id=document_id,
            actor_user_id=OWNER_A,
            instance=instance,
            lease=lease,
            payload=b"expand-membership",
            suggestions=(second_descriptor,),
            patches=(expanded,),
            now=database.now + 2.0,
            expected_sequence=1,
        )
    )
    shrunken = replace(expanded, active_suggestion_ids=(second_id,))

    with pytest.raises(CollaborationConflict) as shrink_conflict:
        await database.store.append_update(
            _suggestion_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=lease,
                payload=b"shrink-membership",
                suggestions=(),
                patches=(shrunken,),
                now=database.now + 3.0,
                expected_sequence=2,
            )
        )
    assert shrink_conflict.value.reason == "patch_membership_shrink"

    async with database.session_factory() as session:
        async with session.begin():
            await session.execute(
                update(editor_patches)
                .where(editor_patches.c.patch_id == patch_id)
                .values(created_by_user_id=OWNER_B)
            )
    foreign_authored = replace(expanded, author_id=OWNER_B)
    with pytest.raises(CollaborationConflict) as author_conflict:
        await database.store.append_update(
            _suggestion_update(
                tenant_id=TENANT_A,
                document_id=document_id,
                actor_user_id=OWNER_A,
                instance=instance,
                lease=lease,
                payload=b"foreign-author-membership",
                suggestions=(),
                patches=(foreign_authored,),
                now=database.now + 4.0,
                expected_sequence=2,
            )
        )
    assert author_conflict.value.reason == "patch_author_conflict"

    async with database.session_factory() as session:
        patch = (
            await session.execute(
                select(editor_patches).where(editor_patches.c.patch_id == patch_id)
            )
        ).mappings().one()
        persisted_sequence = await session.scalar(
            select(editor_documents.c.persisted_sequence).where(
                editor_documents.c.id == document_id
            )
        )
    assert patch["suggestion_ids"] == [first_id, second_id]
    assert persisted_sequence == 2


@pytest.mark.asyncio
async def test_open_patch_keyset_and_all_open_selection_exceed_two_hundred(
    database: _DatabaseHarness,
) -> None:
    """Published patches paginate completely while private AI rows stay hidden."""
    document_id = "ed_collaboration_many_open_patches"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    published_ids = tuple(str(uuid.uuid4()) for _ in range(205))
    private_ids = tuple(str(uuid.uuid4()) for _ in range(3))
    rows: list[dict[str, object]] = []
    for index, patch_id in enumerate(published_ids):
        suggestion_id = str(uuid.uuid4())
        created_at = database.now + index
        rows.append(
            {
                "patch_id": patch_id,
                "tenant_id": TENANT_A,
                "document_id": document_id,
                "run_id": None,
                "source": "human",
                "status": "pending",
                "edits": [
                    {
                        "suggestion_id": suggestion_id,
                        "patch_id": patch_id,
                        "author_id": str(OWNER_A),
                        "created_at": created_at,
                        "kind": "insertion",
                    }
                ],
                "summary": "",
                "warnings": [],
                "revision_before": DOCUMENT_REVISION,
                "collaboration_generation": 1,
                "base_sequence": 0,
                "decision_sequence": None,
                "suggestion_ids": [suggestion_id],
                "applied_revision": None,
                "applied_edit_ids": None,
                "note": "",
                "created_by_user_id": OWNER_A,
                "decided_by_user_id": None,
                "command_id": None,
                "created_at": created_at,
                "decided_at": None,
            }
        )
    for index, patch_id in enumerate(private_ids, start=len(published_ids)):
        rows.append(
            {
                "patch_id": patch_id,
                "tenant_id": TENANT_A,
                "document_id": document_id,
                "run_id": None,
                "source": "agent",
                "status": "pending",
                "edits": [{"id": f"private-{index}", "text": "private"}],
                "summary": "Private AI work",
                "warnings": [],
                "revision_before": DOCUMENT_REVISION,
                "collaboration_generation": 1,
                "base_sequence": 0,
                "decision_sequence": None,
                "suggestion_ids": [],
                "applied_revision": None,
                "applied_edit_ids": None,
                "note": "",
                "created_by_user_id": OWNER_A,
                "decided_by_user_id": None,
                "command_id": None,
                "created_at": database.now + index,
                "decided_at": None,
            }
        )
    async with database.session_factory() as session:
        async with session.begin():
            await session.execute(insert(editor_patches), rows)

    first = await database.store.list_open_patches(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        before=None,
        author_user_id=None,
        suggestion_kind=None,
        limit=200,
    )
    assert len(first.patches) == 200
    assert first.next_cursor is not None
    second = await database.store.list_open_patches(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        before=first.next_cursor,
        author_user_id=None,
        suggestion_kind=None,
        limit=200,
    )
    observed_ids = tuple(
        patch.patch_id for patch in (*first.patches, *second.patches)
    )
    assert len(second.patches) == 5
    assert second.next_cursor is None
    assert set(observed_ids) == set(published_ids)
    assert set(observed_ids).isdisjoint(private_ids)

    all_open = await database.store.list_open_patch_ids_at_sequence(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        expected_sequence=0,
        limit=5_000,
    )
    assert set(all_open) == set(published_ids)
    with pytest.raises(CollaborationConflict) as stale:
        await database.store.list_open_patch_ids_at_sequence(
            tenant_id=TENANT_A,
            document_id=document_id,
            generation=1,
            expected_sequence=1,
            limit=5_000,
        )
    assert stale.value.reason == "sequence_conflict"
    assert stale.value.current_sequence == 0


@pytest.mark.asyncio
async def test_open_patch_kind_filter_precedes_keyset_limit(
    database: _DatabaseHarness,
) -> None:
    """Interleaved kinds yield complete matching pages beyond the raw row cap."""
    document_id = "ed_collaboration_open_patch_kind_pages"
    await _seed_markdown_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    await _activate_document(
        database,
        tenant_id=TENANT_A,
        document_id=document_id,
        owner_user_id=OWNER_A,
    )
    rows: list[dict[str, object]] = []
    expected_ids: list[str] = []
    for index in range(410):
        patch_id = str(uuid.uuid4())
        suggestion_id = str(uuid.uuid4())
        created_at = database.now + index
        is_modification = index % 2 == 0
        is_human = index % 4 in {0, 1}
        if is_modification:
            expected_ids.append(patch_id)
        if is_human:
            edits = [
                {
                    "suggestion_id": suggestion_id,
                    "patch_id": patch_id,
                    "author_id": str(OWNER_A),
                    "created_at": created_at,
                    "kind": (
                        "modification" if is_modification else "insertion"
                    ),
                }
            ]
        else:
            edits = [
                {
                    "id": f"edit-{index}",
                    "find": "old" if is_modification else "",
                    "text": "new",
                    "position": "replace" if is_modification else "append",
                }
            ]
        rows.append(
            {
                "patch_id": patch_id,
                "tenant_id": TENANT_A,
                "document_id": document_id,
                "run_id": None,
                "source": "human" if is_human else "agent",
                "status": "pending",
                "edits": edits,
                "summary": "",
                "warnings": [],
                "revision_before": DOCUMENT_REVISION,
                "collaboration_generation": 1,
                "base_sequence": 0,
                "decision_sequence": None,
                "suggestion_ids": [suggestion_id],
                "applied_revision": None,
                "applied_edit_ids": None,
                "note": "",
                "created_by_user_id": OWNER_A,
                "decided_by_user_id": None,
                "command_id": None,
                "created_at": created_at,
                "decided_at": None,
            }
        )
    async with database.session_factory() as session:
        async with session.begin():
            await session.execute(insert(editor_patches), rows)

    first = await database.store.list_open_patches(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        before=None,
        author_user_id=None,
        suggestion_kind="replacement",
        limit=200,
    )
    assert len(first.patches) == 200
    assert first.next_cursor is not None
    second = await database.store.list_open_patches(
        tenant_id=TENANT_A,
        document_id=document_id,
        generation=1,
        before=first.next_cursor,
        author_user_id=None,
        suggestion_kind="replacement",
        limit=200,
    )

    observed = (*first.patches, *second.patches)
    assert len(second.patches) == 5
    assert second.next_cursor is None
    assert all("replacement" in patch.kinds for patch in observed)
    assert {patch.patch_id for patch in observed} == set(expected_ids)


@pytest.mark.asyncio
async def test_restricted_role_enforces_tenant_rls_for_reads_and_writes(
    database: _DatabaseHarness,
) -> None:
    """The app role sees its tenant only and cannot forge another tenant row."""
    document_a = "ed_collaboration_rls_a"
    document_b = "ed_collaboration_rls_b"
    for tenant_id, document_id, owner_user_id in (
        (TENANT_A, document_a, OWNER_A),
        (TENANT_B, document_b, OWNER_B),
    ):
        await _seed_markdown_document(
            database,
            tenant_id=tenant_id,
            document_id=document_id,
            owner_user_id=owner_user_id,
        )
        await _activate_document(
            database,
            tenant_id=tenant_id,
            document_id=document_id,
            owner_user_id=owner_user_id,
        )

    with pytest.raises(CollaborationDocumentNotFound):
        await database.store.load_state(
            tenant_id=TENANT_A,
            document_id=document_b,
        )

    async with tenant_session(
        database.session_factory,
        tenant_id=TENANT_A,
        app_role=APP_ROLE,
    ) as session:
        visible_snapshots = (
            await session.execute(
                select(
                    editor_collaboration_snapshots.c.tenant_id,
                    editor_collaboration_snapshots.c.document_id,
                )
            )
        ).all()
    assert [(row.tenant_id, row.document_id) for row in visible_snapshots] == [
        (TENANT_A, document_a)
    ]

    with pytest.raises(DBAPIError):
        async with tenant_session(
            database.session_factory,
            tenant_id=TENANT_A,
            app_role=APP_ROLE,
        ) as session:
            await session.execute(
                insert(editor_collaboration_instances).values(
                    slot="primary",
                    tenant_id=TENANT_B,
                    instance_id="forged-cross-tenant-instance",
                    epoch=1,
                    lease_expires_at=database.now + 10.0,
                    updated_at=database.now,
                )
            )

    loaded_b = await database.store.load_state(
        tenant_id=TENANT_B,
        document_id=document_b,
    )
    assert loaded_b.document.tenant_id == TENANT_B
    assert loaded_b.snapshot.covered_sequence == 0
