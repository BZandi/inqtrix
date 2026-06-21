"""Postgres integration tests for the chat-history store (gated, M6a).

Same gating/conventions as the other storage suites: a disposable
database via ``INQTRIX_TEST_DATABASE_URL``, operations under the
restricted app role, RLS as the second defense layer. Verifies the
half of the :class:`~inqtrix.project.chat_ports.ChatStore` contract that
only a real database exercises: the SQL keyset page (tuple comparison +
the float-epoch id tiebreaker), the ``ON CONFLICT`` autosave upsert,
owner/workspace scoping in the query, FK cascade on thread delete, and
the ``ON DELETE SET NULL`` group orphan. The owner/share ACCESS rules
live in the service and are covered offline in ``test_chat_history.py``.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.pagination import decode_cursor
from inqtrix.project.chat_postgres import PostgresChatStore
from inqtrix.project.chat_ports import ChatMessage
from inqtrix.storage.chat_orm import (
    chat_messages,
    chat_thread_groups,
    chat_threads,
)
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"


@pytest.fixture(scope="session", autouse=True)
def chat_schema_migrated():
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
            # Child -> parent order (FK), though CASCADE would handle it.
            await session.execute(chat_messages.delete())
            await session.execute(chat_threads.delete())
            await session.execute(chat_thread_groups.delete())
    chat_store = PostgresChatStore(engine=engine, app_role=APP_ROLE)
    yield chat_store
    await chat_store.aclose()


async def _save(store, thread_id, *, owner="u", workspace=None, created_at=1.0,
                group_id=None):
    return await store.upsert_thread(
        id=thread_id, title="T", preview="", source="api",
        group_id=group_id, created_at=created_at, updated_at=created_at,
        created_by_sub=owner, workspace_id=workspace,
    )


@pytest.mark.asyncio
async def test_thread_keyset_walks_with_tiebreaker(store) -> None:
    """The DB keyset (tuple_(created_at, id) < cursor) pages the owner's
    threads with no gaps or duplicates, including across a created_at tie."""
    stamps = [10.0, 10.0, 20.0, 30.0, 30.0]
    for n, stamp in enumerate(stamps):
        await _save(store, f"ct_{n}", owner="u", created_at=stamp)

    seen: list[str] = []
    cursor = None
    for _ in range(10):
        page, next_cursor = await store.list_threads_page(
            created_by_sub="u", workspace_id=None, limit=2, after=cursor,
        )
        assert len(page) <= 2
        seen.extend(t.id for t in page)
        if next_cursor is None:
            break
        cursor = decode_cursor(next_cursor)

    assert len(seen) == len(set(seen)) == 5
    assert {seen[0], seen[1]} == {"ct_3", "ct_4"}  # the 30.0 pair, first


@pytest.mark.asyncio
async def test_thread_list_scopes_to_owner_and_workspace(store) -> None:
    await _save(store, "ct_1", owner="u1", workspace="w1", created_at=1.0)
    await _save(store, "ct_2", owner="u2", workspace="w1", created_at=2.0)
    await _save(store, "ct_3", owner="u1", workspace="w2", created_at=3.0)

    page, _ = await store.list_threads_page(
        created_by_sub="u1", workspace_id="w1", limit=50, after=None,
    )
    assert [t.id for t in page] == ["ct_1"]


@pytest.mark.asyncio
async def test_upsert_preserves_created_at_and_owner(store) -> None:
    await _save(store, "ct_1", owner="u1", created_at=100.0)
    # Conflicting re-save with a different created_at + owner must not win.
    await store.upsert_thread(
        id="ct_1", title="second", preview="p", source="imported",
        group_id=None, created_at=999.0, updated_at=200.0,
        created_by_sub="someone-else", workspace_id=None,
    )
    thread = await store.get_thread("ct_1")
    assert thread.title == "second"
    assert thread.source == "imported"
    assert thread.created_at == 100.0
    assert thread.updated_at == 200.0
    assert thread.created_by_sub == "u1"


@pytest.mark.asyncio
async def test_message_append_idempotent_keyset_and_cascade(store) -> None:
    await _save(store, "ct_1", owner="u", created_at=1.0)
    messages = [
        ChatMessage(id=f"cm_{n}", thread_id="ct_1", role="user",
                    content_markdown=f"m{n}", created_at=float(n))
        for n in range(5)
    ]
    await store.append_messages(messages)
    # Re-append cm_0 with edited content: upsert, not a duplicate.
    await store.append_messages(
        [ChatMessage(id="cm_0", thread_id="ct_1", role="user",
                     content_markdown="edited", created_at=0.0)]
    )

    seen: list[str] = []
    cursor = None
    for _ in range(10):
        page, next_cursor = await store.list_messages_page(
            "ct_1", limit=2, after=cursor
        )
        seen.extend(m.id for m in page)
        if next_cursor is None:
            break
        cursor = decode_cursor(next_cursor)
    assert len(seen) == len(set(seen)) == 5
    assert seen[0] == "cm_4"  # newest first

    edited_page, _ = await store.list_messages_page("ct_1", limit=50, after=None)
    edited = next(m for m in edited_page if m.id == "cm_0")
    assert edited.content_markdown == "edited"

    # Deleting the thread cascades its messages.
    await store.delete_thread("ct_1")
    gone, _ = await store.list_messages_page("ct_1", limit=50, after=None)
    assert gone == []


@pytest.mark.asyncio
async def test_message_id_reuse_across_threads_is_isolated(store) -> None:
    """The composite (thread_id, id) PK keeps a re-used message id in one
    thread from overwriting the same id in another thread."""
    await _save(store, "ct_a", owner="ua", created_at=1.0)
    await _save(store, "ct_b", owner="ub", created_at=1.0)
    await store.append_messages([
        ChatMessage(id="cm_dup", thread_id="ct_a", role="assistant",
                    content_markdown="A-secret", created_at=1.0)
    ])
    await store.append_messages([
        ChatMessage(id="cm_dup", thread_id="ct_b", role="user",
                    content_markdown="B-content", created_at=1.0)
    ])
    a_page, _ = await store.list_messages_page("ct_a", limit=50, after=None)
    assert next(m for m in a_page if m.id == "cm_dup").content_markdown == (
        "A-secret"
    )
    b_page, _ = await store.list_messages_page("ct_b", limit=50, after=None)
    assert next(m for m in b_page if m.id == "cm_dup").content_markdown == (
        "B-content"
    )


@pytest.mark.asyncio
async def test_group_delete_orphans_thread(store) -> None:
    """The FK ``ON DELETE SET NULL`` orphans a group's threads instead of
    deleting them."""
    await store.upsert_group(
        id="ctg_1", title="G", created_at=1.0, updated_at=1.0,
        created_by_sub="u", workspace_id=None,
    )
    await _save(store, "ct_1", owner="u", created_at=2.0, group_id="ctg_1")
    await store.delete_group("ctg_1")
    thread = await store.get_thread("ct_1")
    assert thread.group_id is None
