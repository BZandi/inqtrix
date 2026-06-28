"""WP-C-E: knowledge-collection ownership, sharing, and admission.

The visibility matrix over the full oidc container with the memory
knowledge store: owners see and manage their collections, strangers
get byte-identical 404s, legacy collections (``created_by_sub None``)
stay open to everyone, view grants admit reads but not writes, edit
grants admit document writes but never collection deletion, deleting
a collection revokes its shares, and the ask paths (chat + native
runs) deny submissions naming an invisible collection.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.auth.directory import MemoryUserDirectory
from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import PermissionService
from inqtrix.knowledge.stores.memory import MemoryKnowledgeStore
from inqtrix.knowledge.stores.ports import KnowledgeProviderContext
from inqtrix.providers.base import ProviderContext
from inqtrix.server.container import build_container
from inqtrix.server.routers import chat, knowledge, runs
from inqtrix.server.routers.shares import build_router as build_shares_router
from inqtrix.settings import ServerSettings, Settings, StorageSettings

from tests.contract._app import StubSearch
from tests.test_knowledge_routes import KnowledgeStubLLM, StubEmbeddings
from tests.test_runs_sharing import OWNER, RECIPIENT, SUB_HEADER, OidcHeaderProvider


def make_world() -> tuple[TestClient, MemoryKnowledgeStore]:
    identity = MemoryIdentityStore()
    users = MemoryUserDirectory()

    async def mirror() -> None:
        for sub, name in ((OWNER, "Olga Owner"), (RECIPIENT, "Rita Recipient")):
            await users.record_login(
                tenant_id="default",
                issuer="http://idp.example",
                subject=sub,
                email=f"{sub}@example.com",
                email_verified=True,
                display_name=name,
            )

    asyncio.run(mirror())
    store = MemoryKnowledgeStore()
    container = build_container(
        providers=ProviderContext(llm=KnowledgeStubLLM(), search=StubSearch()),
        strategies=None,
        settings=Settings(
            server=ServerSettings(public_base_url=""),
            storage=StorageSettings(backend="memory", database_url=""),
        ),
        semaphore_factory=lambda: asyncio.Semaphore(1),
        auth_provider=OidcHeaderProvider(users),
        permissions=PermissionService(
            members=identity, groups=identity, shares=identity, audit=identity
        ),
        workspace_admin=identity,
        knowledge=KnowledgeProviderContext(
            embeddings=StubEmbeddings(),
            store=store,
            default_top_k=4,
        ),
    )
    assert container.share_service is not None
    assert "knowledge_collection" in container.share_service.resource_types
    app = FastAPI()
    app.include_router(knowledge.build_router(container))
    app.include_router(runs.build_router(container))
    app.include_router(chat.build_router(container))
    app.include_router(build_shares_router(container))
    return TestClient(app), store


@pytest.fixture()
def world():
    return make_world()


def as_user(sub: str) -> dict[str, str]:
    return {SUB_HEADER: sub}


def create_collection(client: TestClient, *, sub: str, name: str = "Meine Sammlung") -> str:
    response = client.post(
        "/v1/knowledge/collections", json={"name": name}, headers=as_user(sub)
    )
    assert response.status_code == 201
    return response.json()["id"]


def add_document(client: TestClient, collection_id: str, *, sub: str):
    return client.post(
        f"/v1/knowledge/collections/{collection_id}/documents",
        json={"title": "Notiz", "text": "Die Frist betraegt 24 Stunden."},
        headers=as_user(sub),
    )


def grant(client: TestClient, collection_id: str, *, permission: str = "view"):
    response = client.post(
        "/v1/shares",
        json={
            "resource_type": "knowledge_collection",
            "resource_id": collection_id,
            "invitees": [{"subject_id": RECIPIENT, "permission": permission}],
        },
        headers=as_user(OWNER),
    )
    # These tests assert post-acceptance access, so the recipient consents
    # here. A re-grant carries consent forward (already-accepted -> 404), so
    # both outcomes are valid; the pending/consent flow itself is pinned in
    # tests/test_runs_sharing.py and tests/test_shares.py.
    if response.status_code == 201:
        share_id = response.json()["data"][0]["id"]
        accepted = client.post(
            f"/v1/shares/{share_id}/accept", headers=as_user(RECIPIENT)
        )
        assert accepted.status_code in (200, 404)
    return response


def test_owner_sees_collection_stranger_does_not(world):
    client, _store = world
    collection_id = create_collection(client, sub=OWNER)
    add_document(client, collection_id, sub=OWNER)

    owner_listed = client.get(
        "/v1/knowledge/collections", headers=as_user(OWNER)
    ).json()["data"]
    assert [item["id"] for item in owner_listed] == [collection_id]
    assert "access" not in owner_listed[0]

    stranger_listed = client.get(
        "/v1/knowledge/collections", headers=as_user(RECIPIENT)
    ).json()["data"]
    assert stranger_listed == []

    denied_docs = client.get(
        f"/v1/knowledge/collections/{collection_id}/documents",
        headers=as_user(RECIPIENT),
    )
    assert denied_docs.status_code == 404
    denied_delete = client.delete(
        f"/v1/knowledge/collections/{collection_id}",
        headers=as_user(RECIPIENT),
    )
    assert denied_delete.status_code == 404
    # The collection survives the stranger's delete attempt.
    assert client.get(
        "/v1/knowledge/collections", headers=as_user(OWNER)
    ).json()["data"]


def test_document_reads_deny_via_parent(world):
    client, _store = world
    collection_id = create_collection(client, sub=OWNER)
    document_id = add_document(client, collection_id, sub=OWNER).json()["id"]

    denied_text = client.get(
        f"/v1/knowledge/documents/{document_id}/text",
        headers=as_user(RECIPIENT),
    )
    assert denied_text.status_code == 404
    missing = client.get(
        "/v1/knowledge/documents/kd_does_not_exist/text",
        headers=as_user(RECIPIENT),
    )
    # Denial and absence are byte-identical.
    assert denied_text.json() == missing.json()


def test_legacy_collections_stay_open_to_everyone(world):
    client, store = world
    legacy = asyncio.run(
        store.create_collection(
            name="Bestand", embedding_model="stub-embed-8", embedding_dim=8
        )
    )
    assert legacy.created_by_sub is None

    for sub in (OWNER, RECIPIENT):
        listed = client.get(
            "/v1/knowledge/collections", headers=as_user(sub)
        ).json()["data"]
        assert [item["id"] for item in listed] == [legacy.id]
        assert "access" not in listed[0]
    # Legacy means full access: writes included.
    assert add_document(client, legacy.id, sub=RECIPIENT).status_code == 201


def test_view_grant_admits_reads_not_writes(world):
    client, _store = world
    collection_id = create_collection(client, sub=OWNER)
    document_id = add_document(client, collection_id, sub=OWNER).json()["id"]
    assert grant(client, collection_id).status_code == 201

    listed = client.get(
        "/v1/knowledge/collections", headers=as_user(RECIPIENT)
    ).json()["data"]
    assert [item["id"] for item in listed] == [collection_id]
    assert listed[0]["access"] == {"via": "share", "permission": "view"}

    docs = client.get(
        f"/v1/knowledge/collections/{collection_id}/documents",
        headers=as_user(RECIPIENT),
    )
    assert docs.status_code == 200
    text = client.get(
        f"/v1/knowledge/documents/{document_id}/text",
        headers=as_user(RECIPIENT),
    )
    assert text.status_code == 200

    assert add_document(client, collection_id, sub=RECIPIENT).status_code == 404
    assert (
        client.delete(
            f"/v1/knowledge/documents/{document_id}",
            headers=as_user(RECIPIENT),
        ).status_code
        == 404
    )
    assert (
        client.delete(
            f"/v1/knowledge/collections/{collection_id}",
            headers=as_user(RECIPIENT),
        ).status_code
        == 404
    )


def test_edit_grant_admits_document_writes_not_deletion(world):
    client, _store = world
    collection_id = create_collection(client, sub=OWNER)
    assert grant(client, collection_id, permission="edit").status_code == 201

    created = add_document(client, collection_id, sub=RECIPIENT)
    assert created.status_code == 201
    deleted = client.delete(
        f"/v1/knowledge/documents/{created.json()['id']}",
        headers=as_user(RECIPIENT),
    )
    assert deleted.status_code == 204
    # Deleting the COLLECTION stays owner-only even with edit.
    assert (
        client.delete(
            f"/v1/knowledge/collections/{collection_id}",
            headers=as_user(RECIPIENT),
        ).status_code
        == 404
    )


def test_search_filters_to_visible_collections(world):
    client, store = world
    owned = create_collection(client, sub=OWNER)
    add_document(client, owned, sub=OWNER)
    legacy = asyncio.run(
        store.create_collection(
            name="Bestand", embedding_model="stub-embed-8", embedding_dim=8
        )
    )

    # Explicitly asking for owned + legacy: the invisible owned id is
    # filtered, legacy results come back.
    mixed = client.post(
        "/v1/knowledge/search",
        json={"query": "Frist", "collection_ids": [owned, legacy.id]},
        headers=as_user(RECIPIENT),
    )
    assert mixed.status_code == 200
    assert all(
        hit["collection_id"] == legacy.id for hit in mixed.json()["data"]
    )

    # Asking ONLY for the invisible collection is the indistinct 404.
    denied = client.post(
        "/v1/knowledge/search",
        json={"query": "Frist", "collection_ids": [owned]},
        headers=as_user(RECIPIENT),
    )
    assert denied.status_code == 404

    # An unscoped search ranges over the visible set only.
    unscoped = client.post(
        "/v1/knowledge/search",
        json={"query": "Frist"},
        headers=as_user(RECIPIENT),
    )
    assert unscoped.status_code == 200
    assert all(
        hit["collection_id"] == legacy.id for hit in unscoped.json()["data"]
    )


def test_collection_deletion_revokes_its_shares(world):
    client, _store = world
    collection_id = create_collection(client, sub=OWNER)
    assert grant(client, collection_id).status_code == 201
    before = client.get(
        "/v1/shares/shared-with-me",
        params={"resource_type": "knowledge_collection"},
        headers=as_user(RECIPIENT),
    ).json()["data"]
    assert [item["resource_id"] for item in before] == [collection_id]

    deleted = client.delete(
        f"/v1/knowledge/collections/{collection_id}", headers=as_user(OWNER)
    )
    assert deleted.status_code == 204
    after = client.get(
        "/v1/shares/shared-with-me",
        params={"resource_type": "knowledge_collection"},
        headers=as_user(RECIPIENT),
    ).json()["data"]
    assert after == []


def test_run_submission_denies_invisible_collections(world):
    client, _store = world
    collection_id = create_collection(client, sub=OWNER)

    denied = client.post(
        "/v1/runs",
        json={
            "question": "Wie lange ist die Frist?",
            "mode": "knowledge",
            "knowledge_filters": {"collection_ids": [collection_id]},
        },
        headers=as_user(RECIPIENT),
    )
    assert denied.status_code == 404

    assert grant(client, collection_id).status_code == 201
    admitted = client.post(
        "/v1/runs",
        json={
            "question": "Wie lange ist die Frist?",
            "mode": "knowledge",
            "knowledge_filters": {"collection_ids": [collection_id]},
        },
        headers=as_user(RECIPIENT),
    )
    assert admitted.status_code == 202


def test_chat_submission_denies_invisible_collections(world):
    client, _store = world
    collection_id = create_collection(client, sub=OWNER)

    denied = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Wie lange ist die Frist?"}
            ],
            "mode": "knowledge",
            "knowledge_filters": {"collection_ids": [collection_id]},
        },
        headers=as_user(RECIPIENT),
    )
    assert denied.status_code == 404


def test_legacy_collections_are_not_shareable(world):
    client, store = world
    legacy = asyncio.run(
        store.create_collection(
            name="Bestand", embedding_model="stub-embed-8", embedding_dim=8
        )
    )
    # No owner means no grant authority — and none is needed, the
    # collection is visible to everyone already.
    assert grant(client, legacy.id).status_code == 404
