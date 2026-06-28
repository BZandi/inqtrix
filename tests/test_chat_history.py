"""Behavior tests for the chat-history persistence service (M6a).

Runs against the in-memory tier (the offline backend) but asserts the
observable contract both tiers must honour: owner isolation, the
idempotent autosave upsert, the keyset page (including the float-epoch
tiebreaker), message append idempotency, and the cascade/orphan rules.
The gated Postgres suite (``tests/storage/test_chat_postgres.py``)
re-runs the keyset/RLS half against a real database.
"""

from __future__ import annotations

import pytest

from inqtrix.auth.permissions import SharePermission
from inqtrix.auth.principal import Principal, UserContext
from inqtrix.pagination import decode_cursor
from inqtrix.project.chat_memory import MemoryChatStore
from inqtrix.project.chat_ports import ThreadGroupNotFound, ThreadNotFound
from inqtrix.services.chat_history_service import (
    ChatHistoryService,
    ChatValidationError,
)


def _scoped(sub: str) -> UserContext:
    return UserContext(
        principal=Principal(
            sub=sub, kind="oidc_session", tenant_id="default", role="member"
        )
    )


@pytest.fixture()
def service() -> ChatHistoryService:
    return ChatHistoryService(store=MemoryChatStore(), durable=False)


async def _save_thread(
    service: ChatHistoryService,
    *,
    thread_id: str,
    caller_sub: str | None,
    created_at: float,
    title: str = "T",
    group_id: str | None = None,
) -> None:
    await service.save_thread(
        id=thread_id,
        title=title,
        preview="",
        source="api",
        group_id=group_id,
        created_at=created_at,
        updated_at=created_at,
        caller_sub=caller_sub,
        workspace_id=None,
        visible_to=_scoped(caller_sub) if caller_sub else None,
    )


@pytest.mark.asyncio
async def test_thread_list_keyset_walks_without_skip_or_repeat(service) -> None:
    """The owner's threads page newest-first with no gaps or duplicates,
    INCLUDING across a created_at tie (the float-epoch tiebreaker)."""
    # Two pairs share a created_at to force the id tiebreaker.
    stamps = [10.0, 10.0, 20.0, 20.0, 30.0]
    for n, stamp in enumerate(stamps):
        await _save_thread(
            service, thread_id=f"ct_{n}", caller_sub="user-a", created_at=stamp
        )

    seen: list[str] = []
    cursor = None
    for _ in range(10):  # generous bound
        page, next_cursor = await service.list_threads(
            caller_sub="user-a", workspace_id=None, limit=2, after=cursor
        )
        assert len(page) <= 2
        seen.extend(t.id for t in page)
        if next_cursor is None:
            break
        cursor = decode_cursor(next_cursor)

    assert len(seen) == len(set(seen)) == 5
    # Newest-first: highest created_at first.
    assert seen[0] == "ct_4"  # the lone 30.0
    assert set(seen[1:3]) == {"ct_2", "ct_3"}  # the 20.0 pair
    assert set(seen[3:5]) == {"ct_0", "ct_1"}  # the 10.0 pair


@pytest.mark.asyncio
async def test_thread_upsert_preserves_owner_and_created_at(service) -> None:
    """Re-saving a thread updates mutable metadata but never re-homes it
    or shifts its creation time (idempotent autosave)."""
    await _save_thread(
        service, thread_id="ct_1", caller_sub="user-a", created_at=100.0,
        title="first",
    )
    # The owner re-saves with a new title and a later updated_at.
    await service.save_thread(
        id="ct_1", title="second", preview="p", source="imported",
        group_id=None, created_at=999.0, updated_at=200.0,
        caller_sub="user-a", workspace_id=None, visible_to=_scoped("user-a"),
    )
    thread = await service.get_thread("ct_1", visible_to=_scoped("user-a"))
    assert thread.title == "second"
    assert thread.source == "imported"
    assert thread.created_at == 100.0  # NOT 999.0 — creation time is stable
    assert thread.updated_at == 200.0
    assert thread.created_by_sub == "user-a"


@pytest.mark.asyncio
async def test_scoped_owner_isolation(service) -> None:
    """A scoped caller never sees or hijacks another user's thread."""
    await _save_thread(
        service, thread_id="ct_a", caller_sub="user-a", created_at=1.0
    )
    # User B cannot read it, and it is absent from B's list.
    with pytest.raises(ThreadNotFound):
        await service.get_thread("ct_a", visible_to=_scoped("user-b"))
    page, _ = await service.list_threads(
        caller_sub="user-b", workspace_id=None, limit=50, after=None
    )
    assert page == []
    # User B cannot hijack A's thread id via a save.
    with pytest.raises(ThreadNotFound):
        await service.save_thread(
            id="ct_a", title="hijack", preview="", source="api",
            group_id=None, created_at=1.0, updated_at=2.0,
            caller_sub="user-b", workspace_id=None,
            visible_to=_scoped("user-b"),
        )
    # The owner still owns it unchanged.
    thread = await service.get_thread("ct_a", visible_to=_scoped("user-a"))
    assert thread.title == "T"


@pytest.mark.asyncio
async def test_unscoped_caller_sees_everything(service) -> None:
    """The anonymous/static principal (visible_to None) keeps the legacy
    full-visibility behaviour."""
    await _save_thread(
        service, thread_id="ct_x", caller_sub=None, created_at=1.0
    )
    thread = await service.get_thread("ct_x", visible_to=None)
    assert thread.id == "ct_x"
    page, _ = await service.list_threads(
        caller_sub=None, workspace_id=None, limit=50, after=None
    )
    assert [t.id for t in page] == ["ct_x"]


@pytest.mark.asyncio
async def test_message_append_idempotent_and_paginates(service) -> None:
    """Appending the same message id twice does not duplicate it, and the
    message page walks newest-first."""
    await _save_thread(
        service, thread_id="ct_1", caller_sub="user-a", created_at=1.0
    )
    payload = [
        {"id": f"cm_{n}", "role": "user", "content_markdown": f"m{n}",
         "created_at": float(n)}
        for n in range(3)
    ]
    await service.append_messages(
        "ct_1", messages=payload, visible_to=_scoped("user-a")
    )
    # Re-append cm_0 with edited content: upsert, not duplicate.
    await service.append_messages(
        "ct_1",
        messages=[{"id": "cm_0", "role": "user", "content_markdown": "edited",
                   "created_at": 0.0}],
        visible_to=_scoped("user-a"),
    )
    page, next_cursor = await service.list_messages(
        "ct_1", limit=50, after=None, visible_to=_scoped("user-a")
    )
    assert [m.id for m in page] == ["cm_2", "cm_1", "cm_0"]  # newest first
    assert next_cursor is None
    edited = next(m for m in page if m.id == "cm_0")
    assert edited.content_markdown == "edited"


@pytest.mark.asyncio
async def test_message_id_reuse_across_threads_does_not_overwrite(service) -> None:
    """A message id colliding with another thread's message must NOT
    overwrite it — identity is thread-scoped (the isolation invariant)."""
    await _save_thread(
        service, thread_id="ct_a", caller_sub="user-a", created_at=1.0
    )
    await _save_thread(
        service, thread_id="ct_b", caller_sub="user-b", created_at=1.0
    )
    await service.append_messages(
        "ct_a",
        messages=[{"id": "cm_dup", "role": "assistant",
                   "content_markdown": "A-secret", "created_at": 1.0}],
        visible_to=_scoped("user-a"),
    )
    # user-b appends the SAME id into their OWN thread.
    await service.append_messages(
        "ct_b",
        messages=[{"id": "cm_dup", "role": "user",
                   "content_markdown": "B-content", "created_at": 1.0}],
        visible_to=_scoped("user-b"),
    )
    a_page, _ = await service.list_messages(
        "ct_a", limit=50, after=None, visible_to=_scoped("user-a")
    )
    a_msg = next(m for m in a_page if m.id == "cm_dup")
    assert a_msg.content_markdown == "A-secret"  # NOT clobbered
    assert a_msg.thread_id == "ct_a"
    b_page, _ = await service.list_messages(
        "ct_b", limit=50, after=None, visible_to=_scoped("user-b")
    )
    b_msg = next(m for m in b_page if m.id == "cm_dup")
    assert b_msg.content_markdown == "B-content"
    assert b_msg.thread_id == "ct_b"


@pytest.mark.asyncio
async def test_append_to_foreign_thread_denied(service) -> None:
    """A scoped non-owner cannot append into another user's thread."""
    await _save_thread(
        service, thread_id="ct_a", caller_sub="user-a", created_at=1.0
    )
    with pytest.raises(ThreadNotFound):
        await service.append_messages(
            "ct_a",
            messages=[{"id": "cm_x", "role": "user",
                       "content_markdown": "x", "created_at": 1.0}],
            visible_to=_scoped("user-b"),
        )


async def _append(
    service: ChatHistoryService,
    *,
    thread_id: str,
    caller_sub: str,
    message_ids: list[str],
) -> None:
    await service.append_messages(
        thread_id,
        messages=[
            {"id": mid, "role": "user", "content_markdown": mid,
             "created_at": float(n)}
            for n, mid in enumerate(message_ids)
        ],
        visible_to=_scoped(caller_sub),
    )


async def _message_ids(
    service: ChatHistoryService, *, thread_id: str, caller_sub: str
) -> list[str]:
    page, _ = await service.list_messages(
        thread_id, limit=50, after=None, visible_to=_scoped(caller_sub)
    )
    return [m.id for m in page]


@pytest.mark.asyncio
async def test_delete_message_removes_only_that_message_for_the_owner(service) -> None:
    """The core fix: deleting one message drops it from the thread and
    leaves its siblings — the durable counterpart the append-only push
    lacked, so a reload no longer resurrects it."""
    await _save_thread(
        service, thread_id="ct_1", caller_sub="user-a", created_at=1.0
    )
    await _append(
        service, thread_id="ct_1", caller_sub="user-a",
        message_ids=["cm_0", "cm_1", "cm_2"],
    )
    await service.delete_message(
        "ct_1", "cm_1", visible_to=_scoped("user-a")
    )
    assert await _message_ids(
        service, thread_id="ct_1", caller_sub="user-a"
    ) == ["cm_2", "cm_0"]  # cm_1 gone, order otherwise intact (newest-first)


@pytest.mark.asyncio
async def test_delete_message_is_idempotent(service) -> None:
    """Deleting an already-gone (or never-present) message is a no-op, not
    an error — a coalesced-burst or multi-device re-issue must not wedge
    the autosave retry loop."""
    await _save_thread(
        service, thread_id="ct_1", caller_sub="user-a", created_at=1.0
    )
    await _append(
        service, thread_id="ct_1", caller_sub="user-a", message_ids=["cm_0"]
    )
    await service.delete_message("ct_1", "cm_0", visible_to=_scoped("user-a"))
    # Second delete of the same id, and a delete of an unknown id: both quiet.
    await service.delete_message("ct_1", "cm_0", visible_to=_scoped("user-a"))
    await service.delete_message(
        "ct_1", "cm_never", visible_to=_scoped("user-a")
    )
    assert await _message_ids(
        service, thread_id="ct_1", caller_sub="user-a"
    ) == []


@pytest.mark.asyncio
async def test_delete_message_denied_for_a_foreign_caller(service) -> None:
    """A scoped non-owner cannot delete another user's message, and the
    denial is the indistinct ThreadNotFound (existence undisclosed); the
    message survives."""
    await _save_thread(
        service, thread_id="ct_a", caller_sub="user-a", created_at=1.0
    )
    await _append(
        service, thread_id="ct_a", caller_sub="user-a", message_ids=["cm_0"]
    )
    with pytest.raises(ThreadNotFound):
        await service.delete_message(
            "ct_a", "cm_0", visible_to=_scoped("user-b")
        )
    assert await _message_ids(
        service, thread_id="ct_a", caller_sub="user-a"
    ) == ["cm_0"]


@pytest.mark.asyncio
async def test_delete_message_needs_an_edit_share_not_merely_view(service) -> None:
    """Deleting a message is the inverse of appending, so it takes editing
    access (like append_messages), not the owner-only thread delete: an
    EDIT share may delete, a VIEW share may not."""
    await _save_thread(
        service, thread_id="ct_a", caller_sub="user-a", created_at=1.0
    )
    await _append(
        service, thread_id="ct_a", caller_sub="user-a",
        message_ids=["cm_0", "cm_1"],
    )
    # A view share cannot delete.
    with pytest.raises(ThreadNotFound):
        await service.delete_message(
            "ct_a", "cm_0", visible_to=_scoped("user-b"),
            also_visible={"ct_a": SharePermission.VIEW},
        )
    # An edit share can.
    await service.delete_message(
        "ct_a", "cm_0", visible_to=_scoped("user-b"),
        also_visible={"ct_a": SharePermission.EDIT},
    )
    assert await _message_ids(
        service, thread_id="ct_a", caller_sub="user-a"
    ) == ["cm_1"]


@pytest.mark.asyncio
async def test_delete_message_blocked_across_a_different_workspace(service) -> None:
    """Defense-in-depth (mirrors the thread delete): a delete carrying a
    different project's workspace namespace is denied and leaves the row,
    never a silent cross-project drop."""
    await service.save_thread(
        id="ct_1", title="T", preview="", source="api", group_id=None,
        created_at=1.0, updated_at=1.0, caller_sub="user-a",
        workspace_id="ws_a", visible_to=_scoped("user-a"),
    )
    await service.append_messages(
        "ct_1",
        messages=[{"id": "cm_0", "role": "user", "content_markdown": "x",
                   "created_at": 0.0}],
        visible_to=_scoped("user-a"),
    )
    with pytest.raises(ThreadNotFound):
        await service.delete_message(
            "ct_1", "cm_0", visible_to=_scoped("user-a"),
            request_workspace_id="ws_b",
        )
    assert await _message_ids(
        service, thread_id="ct_1", caller_sub="user-a"
    ) == ["cm_0"]


@pytest.mark.asyncio
async def test_delete_thread_cascades_and_is_owner_only(service) -> None:
    await _save_thread(
        service, thread_id="ct_1", caller_sub="user-a", created_at=1.0
    )
    await service.append_messages(
        "ct_1",
        messages=[{"id": "cm_1", "role": "user", "content_markdown": "x",
                   "created_at": 1.0}],
        visible_to=_scoped("user-a"),
    )
    # A foreign user cannot delete it.
    with pytest.raises(ThreadNotFound):
        await service.delete_thread("ct_1", visible_to=_scoped("user-b"))
    # The owner can; the messages go with it.
    await service.delete_thread("ct_1", visible_to=_scoped("user-a"))
    with pytest.raises(ThreadNotFound):
        await service.get_thread("ct_1", visible_to=_scoped("user-a"))


@pytest.mark.asyncio
async def test_delete_thread_is_blocked_across_a_different_workspace(service) -> None:
    """Defense-in-depth: a delete issued from another project's UI namespace
    must not reach this project's rows, even for the same owner. The thread is
    owned by user-a in workspace ws_a; a delete carrying ws_b (the namespace of
    a different synced project) is denied as not-found and leaves the row."""
    await service.save_thread(
        id="ct_1", title="T", preview="", source="api", group_id=None,
        created_at=1.0, updated_at=1.0, caller_sub="user-a",
        workspace_id="ws_a", visible_to=_scoped("user-a"),
    )
    with pytest.raises(ThreadNotFound):
        await service.delete_thread(
            "ct_1", visible_to=_scoped("user-a"), request_workspace_id="ws_b"
        )
    # Still present: the cross-workspace delete was a no-op, not a silent drop.
    assert (await service.get_thread("ct_1", visible_to=_scoped("user-a"))).id == "ct_1"
    # The owner in the SAME workspace deletes normally.
    await service.delete_thread(
        "ct_1", visible_to=_scoped("user-a"), request_workspace_id="ws_a"
    )
    with pytest.raises(ThreadNotFound):
        await service.get_thread("ct_1", visible_to=_scoped("user-a"))


@pytest.mark.asyncio
async def test_group_delete_orphans_threads(service) -> None:
    """Deleting a group ungroups its threads (never deletes them)."""
    await service.save_group(
        id="ctg_1", title="G", created_at=1.0, updated_at=1.0,
        caller_sub="user-a", workspace_id=None, visible_to=_scoped("user-a"),
    )
    await _save_thread(
        service, thread_id="ct_1", caller_sub="user-a", created_at=2.0,
        group_id="ctg_1",
    )
    await service.delete_group("ctg_1", visible_to=_scoped("user-a"))
    thread = await service.get_thread("ct_1", visible_to=_scoped("user-a"))
    assert thread.group_id is None
    with pytest.raises(ThreadGroupNotFound):
        await service.delete_group("ctg_1", visible_to=_scoped("user-a"))


@pytest.mark.asyncio
async def test_invalid_role_and_source_rejected(service) -> None:
    """Out-of-domain role/source fail at the service (clean 400), before
    the database CHECK constraint."""
    with pytest.raises(ChatValidationError):
        await service.save_thread(
            id="ct_1", title="T", preview="", source="bogus",
            group_id=None, created_at=1.0, updated_at=1.0,
            caller_sub="user-a", workspace_id=None,
            visible_to=_scoped("user-a"),
        )
    await _save_thread(
        service, thread_id="ct_2", caller_sub="user-a", created_at=1.0
    )
    with pytest.raises(ChatValidationError):
        await service.append_messages(
            "ct_2",
            messages=[{"id": "cm_1", "role": "system",
                       "content_markdown": "x", "created_at": 1.0}],
            visible_to=_scoped("user-a"),
        )
