"""Postgres integration tests for the file-asset-record store (gated, M6c)."""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from inqtrix.content.ports import FileRecord
from inqtrix.pagination import decode_cursor
from inqtrix.project.asset_records_postgres import PostgresAssetStore
from inqtrix.project.asset_records_ports import (
    AssetDeletionInProgress,
    AssetNotFound,
    GroupNotFound,
)
from inqtrix.project.scoped_upsert import ResourceScope
from inqtrix.runs.deletion_operations import DeletionTargetKind
from inqtrix.runs.deletion_postgres import PostgresDeletionOperationStore
from inqtrix.storage.asset_records_orm import (
    asset_groups,
    asset_records,
    asset_sections,
)
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.deletions_orm import (
    deletion_operation_assets,
    deletion_operation_events,
    deletion_operations,
)
from inqtrix.storage.content_orm import files
from inqtrix.storage.content_postgres import PostgresFileRegistry
from inqtrix.storage.migrate import run_migrations
from tests.storage._canonical_users import (
    canonical_user_id,
    ensure_canonical_users,
)

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.postgres

APP_ROLE = "inqtrix_app"
USER_ID = canonical_user_id("asset-user")
USER_1_ID = canonical_user_id("asset-user-1")
USER_2_ID = canonical_user_id("asset-user-2")
OTHER_USER_ID = canonical_user_id("asset-other-user")


@pytest.fixture(scope="session", autouse=True)
def asset_schema_migrated():
    if TEST_DATABASE_URL:
        run_migrations(TEST_DATABASE_URL)
    yield


@pytest_asyncio.fixture()
async def store():
    engine = build_engine(TEST_DATABASE_URL)
    factory = build_session_factory(engine)
    async with factory() as session:
        async with session.begin():
            bypasses = (
                await session.execute(
                    text("SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user")
                )
            ).scalar_one()
            if not bypasses:
                pytest.fail("INQTRIX_TEST_DATABASE_URL must connect as superuser/BYPASSRLS.")
            await session.execute(asset_records.delete())
            await session.execute(files.delete())
            await session.execute(asset_groups.delete())
            await session.execute(asset_sections.delete())
            await ensure_canonical_users(
                session,
                (USER_ID, USER_1_ID, USER_2_ID, OTHER_USER_ID),
            )
    asset_store = PostgresAssetStore(engine=engine, app_role=APP_ROLE)
    yield asset_store
    await asset_store.aclose()


async def _section(
    store,
    sid,
    *,
    owner: uuid.UUID = USER_ID,
    created_at=1.0,
):
    return await store.upsert_section(
        id=sid, kind="custom", title="S", created_at=created_at, updated_at=created_at,
        created_by_user_id=owner, workspace_id=None,
    )


async def _asset(
    store,
    aid,
    *,
    owner: uuid.UUID = USER_ID,
    section_id="fsec_1",
    group_id=None,
    created_at=1.0,
    text="body",
):
    return await store.upsert_asset(
        id=aid, section_id=section_id, group_id=group_id, title="A", label="A",
        file_name="a.pdf", mime_type="application/pdf", origin="library",
        page_count=2, parse_status="parsed", parse_warning=None, text_truncated=True,
        size_bytes=10, server_file_id=None, extracted_text=text,
        created_at=created_at, updated_at=created_at, created_by_user_id=owner,
        workspace_id=None,
    )


@pytest.mark.asyncio
async def test_prepared_section_identity_converges_concurrent_scopes(store) -> None:
    first, second = await asyncio.gather(
        store.ensure_default_sections(
            created_by_user_id=USER_1_ID,
            workspace_id="workspace-a",
        ),
        store.ensure_default_sections(
            created_by_user_id=USER_1_ID,
            workspace_id="workspace-a",
        ),
    )
    assert [section.id for section in first] == [section.id for section in second]
    assert [section.semantic_role for section in first] == [
        "temporary",
        "library",
        "project_sources",
    ]

    same_owner_other_workspace = await store.ensure_default_sections(
        created_by_user_id=USER_1_ID,
        workspace_id="workspace-b",
    )
    other_owner_same_workspace = await store.ensure_default_sections(
        created_by_user_id=USER_2_ID,
        workspace_id="workspace-a",
    )
    assert {section.id for section in first}.isdisjoint(
        section.id for section in same_owner_other_workspace
    )
    assert {section.id for section in first}.isdisjoint(
        section.id for section in other_owner_same_workspace
    )


@pytest.mark.asyncio
async def test_prepared_role_unique_but_equal_custom_titles_are_allowed(store) -> None:
    prepared = await store.ensure_default_sections(
        created_by_user_id=USER_ID,
        workspace_id=None,
    )
    library = next(
        section for section in prepared if section.semantic_role == "library"
    )
    custom_a = await store.upsert_section(
        id="custom_same_a",
        kind="custom",
        title="Bibliothek",
        created_at=2.0,
        updated_at=2.0,
        created_by_user_id=USER_ID,
        workspace_id=None,
    )
    custom_b = await store.upsert_section(
        id="custom_same_b",
        kind="custom",
        title="Bibliothek",
        created_at=3.0,
        updated_at=3.0,
        created_by_user_id=USER_ID,
        workspace_id=None,
    )
    assert custom_a.semantic_role == custom_b.semantic_role == "custom"

    renamed = await store.upsert_section(
        id=library.id,
        kind=library.kind,
        title="Meine Ablage",
        created_at=library.created_at,
        updated_at=4.0,
        created_by_user_id=USER_ID,
        workspace_id=None,
    )
    assert renamed.semantic_role == "custom"
    replacement = next(
        section
        for section in await store.ensure_default_sections(
            created_by_user_id=USER_ID,
            workspace_id=None,
        )
        if section.semantic_role == "library"
    )
    assert replacement.id != library.id


@pytest.mark.asyncio
async def test_asset_list_excludes_body_get_includes_it(store) -> None:
    await _section(store, "fsec_1")
    await _asset(store, "fa_1", text="HEAVY")
    page, _ = await store.list_assets_page(
        created_by_user_id=USER_ID,
        workspace_id=None,
        limit=50,
        after=None,
    )
    assert page[0].extracted_text == ""
    assert page[0].text_truncated is True  # int 1 round-trips to bool
    full = await store.get_asset("fa_1")
    assert full.extracted_text == "HEAVY"


@pytest.mark.asyncio
async def test_asset_deletion_detaches_bound_file_without_losing_retry_anchor(
    store,
) -> None:
    await _section(store, "fsec_bound_delete")
    asset = await _asset(
        store,
        "fa_bound_delete",
        section_id="fsec_bound_delete",
    )
    file_record = FileRecord(
        id="fl_bound_delete",
        tenant_id="default",
        owner_user_id=USER_ID,
        workspace_id=None,
        file_name="bound.txt",
        content_type="text/plain",
        size_bytes=12,
        sha256="a" * 64,
        object_key="tenants/default/files/fl_bound_delete",
        created_at=1.0,
    )
    registry_engine = build_engine(TEST_DATABASE_URL)
    registry = PostgresFileRegistry(
        session_factory=build_session_factory(registry_engine),
        app_role=APP_ROLE,
    )
    try:
        await registry.create(file_record)
        bound = await store.finalize_asset_upload(
            id=asset.id,
            section_id=asset.section_id,
            group_id=asset.group_id,
            title=asset.title,
            label=asset.label,
            file_name=file_record.file_name,
            mime_type=file_record.content_type,
            origin=asset.origin,
            page_count=asset.page_count,
            parse_status=asset.parse_status,
            parse_warning=asset.parse_warning,
            text_truncated=asset.text_truncated,
            size_bytes=file_record.size_bytes,
            server_file_id=file_record.id,
            parser_id="markitdown:test",
            created_at=asset.created_at,
            updated_at=2.0,
            scope=ResourceScope.from_record(asset),
        )
        await store.set_asset_deletion_state(
            asset.id,
            scope=ResourceScope.from_record(asset),
            lifecycle_status="deleting",
            deletion_operation_id="del_bound_delete",
            deletion_stage="knowledge_removed",
            deletion_error=None,
        )

        with pytest.raises(IntegrityError):
            await registry.delete(file_record.id, tenant_id="default")

        with pytest.raises(
            RuntimeError,
            match="asset file binding changed during deletion",
        ):
            await store.detach_server_file_for_deletion(
                asset.id,
                scope=ResourceScope.from_record(asset),
                operation_id="del_bound_delete",
                expected_server_file_id="fl_other",
            )
        with pytest.raises(AssetDeletionInProgress):
            await store.detach_server_file_for_deletion(
                asset.id,
                scope=ResourceScope.from_record(asset),
                operation_id="del_other",
                expected_server_file_id=file_record.id,
            )

        detached = await store.detach_server_file_for_deletion(
            asset.id,
            scope=ResourceScope.from_record(asset),
            operation_id="del_bound_delete",
            expected_server_file_id=file_record.id,
        )
        repeated = await store.detach_server_file_for_deletion(
            asset.id,
            scope=ResourceScope.from_record(asset),
            operation_id="del_bound_delete",
            expected_server_file_id=file_record.id,
        )

        assert bound.server_file_id == file_record.id
        assert detached.server_file_id is None
        assert repeated.server_file_id is None
        assert repeated.lifecycle_status == "deleting"
        assert repeated.deletion_operation_id == "del_bound_delete"
        assert await registry.delete(file_record.id, tenant_id="default") == file_record
        assert (await store.get_asset(asset.id)).server_file_id is None
    finally:
        await registry_engine.dispose()


@pytest.mark.asyncio
async def test_asset_keyset_walk_and_owner_scope(store) -> None:
    await _section(store, "fsec_1", owner=USER_1_ID)
    await _section(store, "fsec_other", owner=USER_2_ID)
    for n, ts in enumerate([10.0, 10.0, 20.0, 30.0, 30.0]):
        await _asset(store, f"fa_{n}", owner=USER_1_ID, created_at=ts)
    await _asset(
        store, "fa_other", owner=USER_2_ID, section_id="fsec_other",
        created_at=5.0,
    )
    seen, cursor = [], None
    for _ in range(10):
        pg, nxt = await store.list_assets_page(
            created_by_user_id=USER_1_ID,
            workspace_id=None,
            limit=2,
            after=cursor,
        )
        seen.extend(a.id for a in pg)
        if nxt is None:
            break
        cursor = decode_cursor(nxt)
    assert len(seen) == len(set(seen)) == 5
    assert "fa_other" not in seen


@pytest.mark.asyncio
async def test_asset_upsert_preserves_created_at_and_owner(store) -> None:
    await _section(store, "fsec_1", owner=USER_1_ID)
    await _asset(
        store,
        "fa_1",
        owner=USER_1_ID,
        created_at=100.0,
        text="v1",
    )
    await store.upsert_asset(
        id="fa_1", section_id="fsec_1", group_id=None, title="renamed", label="A",
        file_name="a.pdf", mime_type="application/pdf", origin="library",
        page_count=2, parse_status="parsed", parse_warning=None, text_truncated=False,
        size_bytes=10, server_file_id=None, extracted_text="v2",
        created_at=999.0, updated_at=200.0,
        created_by_user_id=USER_1_ID, workspace_id=None,
    )
    asset = await store.get_asset("fa_1")
    assert asset.extracted_text == "v2"
    assert asset.created_at == 100.0
    assert asset.created_by_user_id == USER_1_ID


@pytest.mark.asyncio
async def test_asset_cross_owner_id_collision_is_not_found(store) -> None:
    await _section(store, "fsec_1", owner=USER_1_ID)
    await _section(store, "fsec_other", owner=OTHER_USER_ID)
    await _asset(store, "fa_1", owner=USER_1_ID, text="Alice")

    with pytest.raises(AssetNotFound):
        await _asset(
            store,
            "fa_1",
            owner=OTHER_USER_ID,
            section_id="fsec_other",
            text="Bob",
        )

    assert (await store.get_asset("fa_1")).extracted_text == "Alice"


@pytest.mark.asyncio
async def test_section_cascade_and_group_orphan(store) -> None:
    section = await _section(store, "fsec_1")
    group = await store.upsert_group(
        id="fg_1", section_id="fsec_1", title="G", created_at=1.0, updated_at=1.0,
        created_by_user_id=USER_ID, workspace_id=None,
    )
    await _asset(store, "fa_1", section_id="fsec_1", group_id="fg_1")
    # Group delete -> asset orphans to ungrouped (SET NULL).
    await store.delete_group(
        "fg_1",
        scope=ResourceScope.from_record(group),
    )
    assert (await store.get_asset("fa_1")).group_id is None
    # Section delete -> cascades its assets (FK CASCADE).
    await store.delete_section(
        "fsec_1",
        scope=ResourceScope.from_record(section),
    )
    page, _ = await store.list_assets_page(
        created_by_user_id=USER_ID,
        workspace_id=None,
        limit=50,
        after=None,
    )
    assert page == []


@pytest.mark.asyncio
async def test_group_receipt_fences_upload_and_commits_orphaning_atomically(
    store,
) -> None:
    """A retained group receipt is the lock and the terminal DB truth.

    The group must remain addressable while its worker is running, upload
    finalisation must not enter it, and the FK orphaning must land in the same
    commit that makes the receipt terminal.
    """

    await _section(store, "fsec_group_receipt")
    group = await store.upsert_group(
        id="fg_group_receipt",
        section_id="fsec_group_receipt",
        title="G",
        created_at=1.0,
        updated_at=1.0,
        created_by_user_id=USER_ID,
        workspace_id=None,
    )
    asset = await _asset(
        store,
        "fa_group_receipt",
        section_id="fsec_group_receipt",
        group_id=group.id,
    )

    operation_engine = build_engine(TEST_DATABASE_URL)
    deletion_store = PostgresDeletionOperationStore(
        engine=operation_engine,
        app_role=APP_ROLE,
        queue=None,
        max_concurrent=1,
        completed_ttl_seconds=3600,
        worker_id="asset-group-receipt-test",
        recover_orphans=False,
    )
    worker_started = threading.Event()
    release_worker = threading.Event()
    operation_id: str | None = None

    def _work(handle) -> None:
        worker_started.set()
        if not release_worker.wait(timeout=10):
            raise RuntimeError("test did not release group deletion worker")
        handle.complete()

    try:
        summary = deletion_store.submit(
            target_kind=DeletionTargetKind.GROUP,
            target_id=group.id,
            manifest=(),
            tenant_id="default",
            created_by_user_id=USER_ID,
            workspace_id=None,
            work=_work,
            total_items=1,
        )
        operation_id = str(summary["operation_id"])
        assert await asyncio.to_thread(worker_started.wait, 5)

        assert any(
            item.id == group.id
            for item in await store.list_groups(
                created_by_user_id=USER_ID,
                workspace_id=None,
            )
        )
        with pytest.raises(GroupNotFound):
            await store.finalize_asset_upload(
                id=asset.id,
                section_id=asset.section_id,
                group_id=asset.group_id,
                title=asset.title,
                label=asset.label,
                file_name=asset.file_name,
                mime_type=asset.mime_type,
                origin=asset.origin,
                page_count=asset.page_count,
                parse_status=asset.parse_status,
                parse_warning=asset.parse_warning,
                text_truncated=asset.text_truncated,
                size_bytes=asset.size_bytes,
                server_file_id="file_group_receipt",
                parser_id="markitdown:test",
                created_at=asset.created_at,
                updated_at=2.0,
                scope=ResourceScope.from_record(asset),
                upload_operation_id="upload_group_receipt",
            )

        release_worker.set()
        deadline = time.monotonic() + 10
        while True:
            receipt = deletion_store.get(
                operation_id,
                tenant_id="default",
                created_by_user_id=USER_ID,
                workspace_id=None,
            )
            if receipt["status"] == "deleted":
                break
            if time.monotonic() >= deadline:
                pytest.fail(f"group deletion did not finish: {receipt}")
            await asyncio.sleep(0.05)

        assert all(
            item.id != group.id
            for item in await store.list_groups(
                created_by_user_id=USER_ID,
                workspace_id=None,
            )
        )
        orphaned = await store.get_asset(asset.id)
        assert orphaned.group_id is None
        assert orphaned.server_file_id is None
    finally:
        release_worker.set()
        deletion_store.close()
        if operation_id is not None:
            cleanup_engine = build_engine(TEST_DATABASE_URL)
            cleanup_factory = build_session_factory(cleanup_engine)
            async with cleanup_factory() as session:
                async with session.begin():
                    await session.execute(
                        deletion_operation_events.delete().where(
                            deletion_operation_events.c.operation_id
                            == operation_id
                        )
                    )
                    await session.execute(
                        deletion_operation_assets.delete().where(
                            deletion_operation_assets.c.operation_id
                            == operation_id
                        )
                    )
                    await session.execute(
                        deletion_operations.delete().where(
                            deletion_operations.c.operation_id == operation_id
                        )
                    )
            await cleanup_engine.dispose()


@pytest.mark.asyncio
async def test_insert_carries_the_callers_upload_intent(store) -> None:
    """The intent lands in the INSERT itself, not in a follow-up write.

    The column default is 'ready', and before this a reservation was born
    ready and only a SECOND transaction corrected it -- an asyncpg
    connection timeout in between left an asset that looked like a
    complete file and had no bytes. Only the Postgres path exercises the
    real column default and the `mutable` asymmetry, so this is the
    backend the property must be proven on.
    """
    await _section(store, "fsec_intent", owner=USER_1_ID)
    row = await store.upsert_asset(
        id="fa_intent", section_id="fsec_intent", group_id=None, title="I",
        label="I", file_name="i.pdf", mime_type="application/pdf",
        origin="library", page_count=None, parse_status="parsed",
        parse_warning=None, text_truncated=False, size_bytes=10,
        server_file_id=None, extracted_text="", created_at=1.0,
        updated_at=1.0, created_by_user_id=USER_1_ID, workspace_id=None,
        initial_upload_status="awaiting_upload",
    )
    assert row.upload_status == "awaiting_upload"

    # And the exact observed failure, replayed: no follow-up write happens
    # at all (the connection died) -- the STORED row still tells the truth.
    stored = await store.get_asset("fa_intent")
    assert stored.upload_status == "awaiting_upload", (
        "before the fix this row was 'ready' with no bytes"
    )
    assert stored.server_file_id is None


@pytest.mark.asyncio
async def test_default_intent_keeps_local_assets_ready(store) -> None:
    """A plain local asset (server_file_id=None) is ready immediately."""
    await _section(store, "fsec_local2", owner=USER_1_ID)
    row = await _asset(
        store, "fa_local2", owner=USER_1_ID, section_id="fsec_local2"
    )
    assert row.upload_status == "ready"


@pytest.mark.asyncio
async def test_intent_is_insert_only_on_postgres(store) -> None:
    """A repeated reservation upsert cannot reset an existing row.

    upload_status is deliberately absent from the mutable column set;
    this is the test that goes red if anyone adds it there.
    """
    await _section(store, "fsec_keep2", owner=USER_1_ID)
    await _asset(
        store, "fa_keep2", owner=USER_1_ID, section_id="fsec_keep2"
    )  # born ready
    row = await store.upsert_asset(
        id="fa_keep2", section_id="fsec_keep2", group_id=None, title="K",
        label="K", file_name="a.pdf", mime_type="application/pdf",
        origin="library", page_count=2, parse_status="parsed",
        parse_warning=None, text_truncated=True, size_bytes=10,
        server_file_id=None, extracted_text="body", created_at=1.0,
        updated_at=2.0, created_by_user_id=USER_1_ID, workspace_id=None,
        initial_upload_status="awaiting_upload",
    )
    assert row.upload_status == "ready", (
        "the stored status must survive; the intent acts only at INSERT"
    )


@pytest.mark.asyncio
async def test_reservation_survives_a_failing_follow_up_write_on_postgres(
    store, monkeypatch
) -> None:
    """The observed failure, replayed against the REAL column default.

    The memory twin pins the service pass-through; only this backend
    exercises the actual Postgres default ('ready') that the INSERT must
    override. The follow-up write dies mid-reservation -- the stored row
    must already carry awaiting_upload, or the asset reappears as a
    complete-looking file with no bytes.
    """
    from inqtrix.auth.principal import Principal, UserContext
    from inqtrix.services.asset_records_service import AssetRecordsService

    service = AssetRecordsService(store=store, durable=True)
    visible = UserContext(
        principal=Principal(
            user_id=USER_1_ID,
            kind="oidc_session",
            tenant_id="default",
            role="member",
        )
    )

    async def _dies(*args, **kwargs):
        raise ConnectionError("verbindung weg zwischen den beiden writes")

    await _section(store, "fsec_fault", owner=USER_1_ID)
    monkeypatch.setattr(store, "set_asset_upload_state", _dies)

    with pytest.raises(ConnectionError):
        await service.reserve_upload(
            id="fa_fault", section_id="fsec_fault", group_id=None, title="F",
            label="F", file_name="f.pdf", mime_type="application/pdf",
            origin="library", page_count=None, parse_status="parsed",
            parse_warning=None, text_truncated=False, size_bytes=10,
            created_at=1.0, updated_at=1.0, caller_user_id=USER_1_ID,
            workspace_id=None, visible_to=visible,
        )

    stored = await store.get_asset("fa_fault")
    assert stored.upload_status == "awaiting_upload", (
        "the INSERT itself must carry the intent on the real column default"
    )
    assert stored.server_file_id is None


@pytest.mark.asyncio
async def test_an_unknown_insert_intent_is_rejected_on_postgres(store) -> None:
    """The runtime guard must be wired in THIS backend, not only in memory."""
    await _section(store, "fsec_guard_pg", owner=USER_1_ID)
    with pytest.raises(ValueError, match="initial_upload_status"):
        await store.upsert_asset(
            id="fa_guard_pg", section_id="fsec_guard_pg", group_id=None,
            title="G", label="G", file_name="g.pdf",
            mime_type="application/pdf", origin="library", page_count=None,
            parse_status="parsed", parse_warning=None, text_truncated=False,
            size_bytes=10, server_file_id=None, extracted_text="",
            created_at=1.0, updated_at=1.0, created_by_user_id=USER_1_ID,
            workspace_id=None, initial_upload_status="awaiting-upload",
        )
