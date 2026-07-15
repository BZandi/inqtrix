"""Postgres integration tests for the file-asset-record store (gated, M6c)."""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.pagination import decode_cursor
from inqtrix.project.asset_records_postgres import PostgresAssetStore
from inqtrix.project.asset_records_ports import AssetNotFound
from inqtrix.project.scoped_upsert import ResourceScope
from inqtrix.storage.asset_records_orm import (
    asset_groups,
    asset_records,
    asset_sections,
)
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations
from tests.storage._canonical_users import (
    canonical_user_id,
    ensure_canonical_users,
)

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

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
        size_bytes=10, server_file_id="fl_1", extracted_text=text,
        created_at=created_at, updated_at=created_at, created_by_user_id=owner,
        workspace_id=None,
    )


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
        size_bytes=10, server_file_id="fl_1", extracted_text="v2",
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
    await _section(store, "fsec_1")
    await store.upsert_group(
        id="fg_1", section_id="fsec_1", title="G", created_at=1.0, updated_at=1.0,
        created_by_user_id=USER_ID, workspace_id=None,
    )
    await _asset(store, "fa_1", section_id="fsec_1", group_id="fg_1")
    # Group delete -> asset orphans to ungrouped (SET NULL).
    await store.delete_group(
        "fg_1",
        scope=ResourceScope.from_record(await store.get_group("fg_1")),
    )
    assert (await store.get_asset("fa_1")).group_id is None
    # Section delete -> cascades its assets (FK CASCADE).
    await store.delete_section(
        "fsec_1",
        scope=ResourceScope.from_record(await store.get_section("fsec_1")),
    )
    page, _ = await store.list_assets_page(
        created_by_user_id=USER_ID,
        workspace_id=None,
        limit=50,
        after=None,
    )
    assert page == []
