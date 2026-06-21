"""Behavior tests for the knowledge-session service (Wissensmodus, offline tier).

The durable tier behind the Ask view's saved sessions. Asserts the observable
contract: load-on-open (list = metadata, get = items body), per-owner scoping,
newest-first ordering, owner-only mutation, and that an update never reassigns
``created_at`` / ownership.
"""

from __future__ import annotations

import pytest

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.project.knowledge_sessions_memory import MemoryKnowledgeSessionStore
from inqtrix.project.knowledge_sessions_ports import (
    KnowledgeSessionGroupNotFound,
    KnowledgeSessionNotFound,
)
from inqtrix.services.knowledge_sessions_service import KnowledgeSessionsService


def _scoped(sub: str) -> UserContext:
    return UserContext(
        principal=Principal(sub=sub, kind="oidc_session", tenant_id="default", role="member")
    )


@pytest.fixture()
def service() -> KnowledgeSessionsService:
    return KnowledgeSessionsService(store=MemoryKnowledgeSessionStore(), durable=False)


async def _save(
    service, sid, *, owner, title="S", items="[]", group_id=None,
    created_at=1.0, updated_at=1.0, workspace_id=None,
):
    return await service.save_session(
        id=sid, title=title, items_json=items, group_id=group_id,
        created_at=created_at, updated_at=updated_at, caller_sub=owner,
        workspace_id=workspace_id,
        visible_to=_scoped(owner),
    )


async def _group(
    service, gid, *, owner, title="G", created_at=1.0, updated_at=1.0,
    workspace_id=None,
):
    return await service.save_group(
        id=gid, title=title, created_at=created_at, updated_at=updated_at,
        caller_sub=owner, workspace_id=workspace_id, visible_to=_scoped(owner),
    )


@pytest.mark.asyncio
async def test_get_returns_items_body_list_returns_metadata_only(service) -> None:
    await _save(service, "ks_1", owner="u", items='[{"q":"hi"}]')
    listed = await service.list_sessions(caller_sub="u", workspace_id=None)
    assert [s.id for s in listed] == ["ks_1"]
    assert listed[0].items_json == "[]"  # heavy body excluded from list
    assert listed[0].group_id is None
    full = await service.get_session("ks_1", visible_to=_scoped("u"))
    assert full.items_json == '[{"q":"hi"}]'


@pytest.mark.asyncio
async def test_list_scoped_to_owner(service) -> None:
    await _save(service, "ks_a", owner="alice")
    await _save(service, "ks_b", owner="bob")
    alice = await service.list_sessions(caller_sub="alice", workspace_id=None)
    assert [s.id for s in alice] == ["ks_a"]


@pytest.mark.asyncio
async def test_list_orders_newest_updated_first(service) -> None:
    await _save(service, "ks_old", owner="u", created_at=1.0, updated_at=1.0)
    await _save(service, "ks_new", owner="u", created_at=2.0, updated_at=5.0)
    listed = await service.list_sessions(caller_sub="u", workspace_id=None)
    assert [s.id for s in listed] == ["ks_new", "ks_old"]


@pytest.mark.asyncio
async def test_get_denies_non_owner(service) -> None:
    await _save(service, "ks_1", owner="u")
    with pytest.raises(KnowledgeSessionNotFound):
        await service.get_session("ks_1", visible_to=_scoped("stranger"))


@pytest.mark.asyncio
async def test_save_existing_by_non_owner_denied(service) -> None:
    await _save(service, "ks_1", owner="u", title="orig")
    with pytest.raises(KnowledgeSessionNotFound):
        await _save(service, "ks_1", owner="stranger", title="hijack")
    full = await service.get_session("ks_1", visible_to=_scoped("u"))
    assert full.title == "orig"  # untouched


@pytest.mark.asyncio
async def test_update_preserves_created_at_and_owner(service) -> None:
    await _save(service, "ks_1", owner="u", title="v1", created_at=1.0, updated_at=1.0)
    await _save(
        service, "ks_1", owner="u", title="v2", items='[{"q":"x"}]',
        created_at=999.0, updated_at=9.0,
    )
    full = await service.get_session("ks_1", visible_to=_scoped("u"))
    assert (full.title, full.items_json) == ("v2", '[{"q":"x"}]')
    assert full.created_at == 1.0  # never reassigned on update
    assert full.updated_at == 9.0


@pytest.mark.asyncio
async def test_delete_removes_session(service) -> None:
    await _save(service, "ks_1", owner="u")
    await service.delete_session("ks_1", visible_to=_scoped("u"))
    with pytest.raises(KnowledgeSessionNotFound):
        await service.get_session("ks_1", visible_to=_scoped("u"))


@pytest.mark.asyncio
async def test_delete_by_non_owner_denied(service) -> None:
    await _save(service, "ks_1", owner="u")
    with pytest.raises(KnowledgeSessionNotFound):
        await service.delete_session("ks_1", visible_to=_scoped("stranger"))
    assert (await service.get_session("ks_1", visible_to=_scoped("u"))).id == "ks_1"


@pytest.mark.asyncio
async def test_group_crud_and_session_group_id_round_trip(service) -> None:
    await _group(service, "kg_1", owner="u", title="Folder")
    groups = await service.list_groups(caller_sub="u", workspace_id=None)
    assert [(g.id, g.title) for g in groups] == [("kg_1", "Folder")]

    saved = await _save(service, "ks_1", owner="u", group_id="kg_1")
    assert saved.group_id == "kg_1"
    listed = await service.list_sessions(caller_sub="u", workspace_id=None)
    assert listed[0].group_id == "kg_1"
    full = await service.get_session("ks_1", visible_to=_scoped("u"))
    assert full.group_id == "kg_1"


@pytest.mark.asyncio
async def test_delete_group_orphans_sessions(service) -> None:
    await _group(service, "kg_1", owner="u")
    await _save(service, "ks_1", owner="u", group_id="kg_1")

    await service.delete_group("kg_1", visible_to=_scoped("u"))

    full = await service.get_session("ks_1", visible_to=_scoped("u"))
    assert full.group_id is None
    with pytest.raises(KnowledgeSessionGroupNotFound):
        await service.delete_group("kg_1", visible_to=_scoped("u"))


@pytest.mark.asyncio
async def test_session_group_must_belong_to_same_owner_and_workspace(service) -> None:
    await _group(service, "kg_1", owner="u", workspace_id="ws_a")

    with pytest.raises(KnowledgeSessionGroupNotFound):
        await _save(
            service, "ks_foreign_owner", owner="stranger", group_id="kg_1",
            workspace_id="ws_a",
        )
    with pytest.raises(KnowledgeSessionGroupNotFound):
        await _save(
            service, "ks_foreign_workspace", owner="u", group_id="kg_1",
            workspace_id="ws_b",
        )

    saved = await _save(
        service, "ks_1", owner="u", group_id="kg_1", workspace_id="ws_a"
    )
    assert saved.group_id == "kg_1"


@pytest.mark.asyncio
async def test_group_access_mirrors_session_owner_and_workspace_rules(service) -> None:
    await _group(service, "kg_1", owner="u", workspace_id="ws_a")
    assert [g.id for g in await service.list_groups(caller_sub="u", workspace_id="ws_a")] == ["kg_1"]
    assert await service.list_groups(caller_sub="u", workspace_id="ws_b") == []

    with pytest.raises(KnowledgeSessionGroupNotFound):
        await service.delete_group("kg_1", visible_to=_scoped("stranger"))
    with pytest.raises(KnowledgeSessionGroupNotFound):
        await service.delete_group(
            "kg_1", visible_to=_scoped("u"), request_workspace_id="ws_b"
        )
    await service.delete_group(
        "kg_1", visible_to=_scoped("u"), request_workspace_id="ws_a"
    )
