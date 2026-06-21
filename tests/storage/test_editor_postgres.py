"""Postgres integration tests for the editor-persistence store (gated, M6b).

Same gating/conventions as the other storage suites. Verifies the half of
the :class:`~inqtrix.project.editor_ports.EditorStore` contract only a real
database exercises: the SQL keyset page (tuple comparison + id tiebreaker),
the body-excluding list vs the body-loading get, the ON CONFLICT autosave
upsert, owner/workspace scoping, comment composite-PK isolation, FK cascade
on document delete, and the ON DELETE SET NULL folder orphan. Owner/share
access rules live in the service (covered offline in
``test_editor_persistence.py``).
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.pagination import decode_cursor
from inqtrix.project.editor_postgres import PostgresEditorStore
from inqtrix.project.editor_ports import EditorComment
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.editor_orm import (
    editor_comments,
    editor_documents,
    editor_folders,
)
from inqtrix.storage.migrate import run_migrations

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"


@pytest.fixture(scope="session", autouse=True)
def editor_schema_migrated():
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
                    text(
                        "SELECT rolsuper OR rolbypassrls FROM pg_roles "
                        "WHERE rolname = current_user"
                    )
                )
            ).scalar_one()
            if not bypasses:
                pytest.fail(
                    "INQTRIX_TEST_DATABASE_URL must connect as a "
                    "superuser/BYPASSRLS user (cross-tenant cleanup)."
                )
            await session.execute(editor_comments.delete())
            await session.execute(editor_documents.delete())
            await session.execute(editor_folders.delete())
    editor_store = PostgresEditorStore(engine=engine, app_role=APP_ROLE)
    yield editor_store
    await editor_store.aclose()


async def _save_doc(store, document_id, *, owner="u", workspace=None,
                    created_at=1.0, body="body", folder_id=None):
    return await store.upsert_document(
        id=document_id, title="D", content_markdown=body, folder_id=folder_id,
        source="blank", source_run_id=None, revision=1,
        diff_anchor_markdown=None, diff_anchor_updated_at=None,
        created_at=created_at, updated_at=created_at,
        created_by_sub=owner, workspace_id=workspace,
    )


def _comment(comment_id, document_id, *, created_at=1.0, body="c"):
    return EditorComment(
        id=comment_id, document_id=document_id, comment_markdown=body,
        anchor={"from": 0, "to": 1, "selectedText": "x"},
        kind="collect", status="open", created_at=created_at, updated_at=created_at,
    )


@pytest.mark.asyncio
async def test_document_list_excludes_body_get_includes_it(store) -> None:
    await _save_doc(store, "ed_1", owner="u", created_at=1.0, body="HEAVY BODY")
    page, _ = await store.list_documents_page(
        created_by_sub="u", workspace_id=None, limit=50, after=None
    )
    assert page[0].content_markdown == ""  # list never transfers the body
    full = await store.get_document("ed_1")
    assert full.content_markdown == "HEAVY BODY"


@pytest.mark.asyncio
async def test_document_keyset_walks_with_tiebreaker_and_scopes(store) -> None:
    for n, stamp in enumerate([10.0, 10.0, 20.0, 30.0, 30.0]):
        await _save_doc(store, f"ed_{n}", owner="u1", workspace="w1", created_at=stamp)
    await _save_doc(store, "ed_other", owner="u2", workspace="w1", created_at=5.0)

    seen: list[str] = []
    cursor = None
    for _ in range(10):
        page, next_cursor = await store.list_documents_page(
            created_by_sub="u1", workspace_id="w1", limit=2, after=cursor
        )
        seen.extend(d.id for d in page)
        if next_cursor is None:
            break
        cursor = decode_cursor(next_cursor)
    assert len(seen) == len(set(seen)) == 5
    assert "ed_other" not in seen  # owner-scoped


@pytest.mark.asyncio
async def test_upsert_preserves_created_at_and_owner(store) -> None:
    await _save_doc(store, "ed_1", owner="u1", created_at=100.0, body="v1")
    await store.upsert_document(
        id="ed_1", title="second", content_markdown="v2", folder_id=None,
        source="pasted", source_run_id=None, revision=9,
        diff_anchor_markdown=None, diff_anchor_updated_at=None,
        created_at=999.0, updated_at=200.0,
        created_by_sub="someone-else", workspace_id=None,
    )
    doc = await store.get_document("ed_1")
    assert doc.content_markdown == "v2"
    assert doc.revision == 9
    assert doc.created_at == 100.0
    assert doc.created_by_sub == "u1"


@pytest.mark.asyncio
async def test_comment_composite_pk_isolation_and_cascade(store) -> None:
    await _save_doc(store, "ed_a", owner="u", created_at=1.0)
    await _save_doc(store, "ed_b", owner="u", created_at=1.0)
    await store.upsert_comments([_comment("edc_dup", "ed_a", body="A-text")])
    await store.upsert_comments([_comment("edc_dup", "ed_b", body="B-text")])
    a_page, _ = await store.list_comments_page("ed_a", limit=50, after=None)
    assert next(c for c in a_page if c.id == "edc_dup").comment_markdown == "A-text"
    b_page, _ = await store.list_comments_page("ed_b", limit=50, after=None)
    assert next(c for c in b_page if c.id == "edc_dup").comment_markdown == "B-text"
    # Deleting the document cascades its comments.
    await store.delete_document("ed_a")
    gone, _ = await store.list_comments_page("ed_a", limit=50, after=None)
    assert gone == []


@pytest.mark.asyncio
async def test_folder_delete_orphans_documents(store) -> None:
    await store.upsert_folder(
        id="edf_1", title="F", created_at=1.0, updated_at=1.0,
        created_by_sub="u", workspace_id=None,
    )
    await _save_doc(store, "ed_1", owner="u", created_at=2.0, folder_id="edf_1")
    await store.delete_folder("edf_1")
    doc = await store.get_document("ed_1")
    assert doc.folder_id is None
