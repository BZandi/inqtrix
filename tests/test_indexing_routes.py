"""HTTP integration for the background reindex endpoints.

Exercises the real router + container wiring (the in-memory knowledge
store, the indexing service, route registration) end-to-end over a
FastAPI ``TestClient``: create a collection, ingest documents, start a
reindex, poll it to completion, list it, and stream its events.
"""

from __future__ import annotations

import asyncio
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.auth.directory import MemoryUserDirectory
from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import AuthorizationService
from inqtrix.knowledge.stores.memory import MemoryKnowledgeStore
from inqtrix.knowledge.stores.ports import KnowledgeProviderContext
from inqtrix.providers.base import ProviderContext
from inqtrix.server.container import build_container
from inqtrix.server.routers import indexing, knowledge
from inqtrix.server.routers.shares import build_router as build_shares_router
from inqtrix.settings import ServerSettings, Settings, StorageSettings

from tests.test_knowledge_routes import KnowledgeStubLLM, StubEmbeddings, _StubSearch
from tests.test_runs_sharing import (
    OWNER,
    RECIPIENT,
    OidcHeaderProvider,
    user_headers,
)


def make_indexing_client(
    *,
    embeddings: StubEmbeddings | None = None,
) -> TestClient:
    """Build a TestClient exposing the knowledge + reindex routers."""
    identity = MemoryIdentityStore()
    users = MemoryUserDirectory()

    async def mirror_users() -> None:
        for user_id, subject, display_name in (
            (OWNER, "user-owner", "Olga Owner"),
            (RECIPIENT, "user-recipient", "Rita Recipient"),
        ):
            await users.record_login(
                tenant_id="default",
                issuer="http://idp.example",
                subject=subject,
                email=f"{subject}@example.com",
                email_verified=True,
                display_name=display_name,
                canonical_user_id=user_id,
            )

    asyncio.run(mirror_users())
    auth_provider = OidcHeaderProvider(users)
    container = build_container(
        providers=ProviderContext(llm=KnowledgeStubLLM(), search=_StubSearch()),
        strategies=None,
        settings=Settings(
            server=ServerSettings(public_base_url=""),
            storage=StorageSettings(backend="memory", database_url=""),
        ),
        semaphore_factory=lambda: asyncio.Semaphore(1),
        auth_provider=auth_provider,
        permissions=AuthorizationService(
            members=identity,
            shares=identity,
            audit=identity,
        ),
        workspace_admin=identity,
        knowledge=KnowledgeProviderContext(
            embeddings=embeddings or StubEmbeddings(),
            store=MemoryKnowledgeStore(),
            default_top_k=4,
        ),
    )
    assert container.indexing_service is not None
    app = FastAPI()
    app.state.container = container
    app.include_router(knowledge.build_router(container))
    app.include_router(indexing.build_router(container))
    app.include_router(build_shares_router(container))
    client = TestClient(app, headers=user_headers(OWNER))
    client.auth_provider = auth_provider  # type: ignore[attr-defined]
    return client


def _collection_with_two_documents(client: TestClient) -> str:
    created = client.post("/v1/knowledge/collections", json={"name": "Vertraege"})
    assert created.status_code == 201
    collection_id = created.json()["id"]
    for index in range(2):
        ingested = client.post(
            f"/v1/knowledge/collections/{collection_id}/documents",
            json={"title": f"Doc {index}", "text": f"alpha beta gamma {index}"},
        )
        assert ingested.status_code == 201
    return collection_id


def _poll_until_terminal(client: TestClient, job_id: str, *, timeout: float = 2.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/v1/knowledge/indexing-jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.02)
    raise AssertionError("reindex job did not reach a terminal status in time")


def test_reindex_completes_over_http() -> None:
    client = make_indexing_client()
    with client:
        collection_id = _collection_with_two_documents(client)

        started = client.post(
            f"/v1/knowledge/collections/{collection_id}/reindex",
            json={"index_id": "vix_1"},
        )
        assert started.status_code == 202
        summary = started.json()
        assert summary["collection_id"] == collection_id
        assert summary["index_id"] == "vix_1"
        job_id = summary["job_id"]

        final = _poll_until_terminal(client, job_id)
        assert final["status"] == "completed"
        assert final["total_documents"] == 2
        assert final["completed_documents"] == 2
        assert final["percent"] == 100


def test_list_jobs_filters_by_collection() -> None:
    client = make_indexing_client()
    with client:
        collection_id = _collection_with_two_documents(client)
        started = client.post(
            f"/v1/knowledge/collections/{collection_id}/reindex", json={}
        )
        job_id = started.json()["job_id"]
        _poll_until_terminal(client, job_id)

        listed = client.get("/v1/knowledge/indexing-jobs")
        assert listed.status_code == 200
        ids = {job["job_id"] for job in listed.json()["data"]}
        assert job_id in ids

        filtered = client.get(
            "/v1/knowledge/indexing-jobs",
            params={"collection_id": collection_id},
        )
        assert {job["job_id"] for job in filtered.json()["data"]} == {job_id}

        other = client.get(
            "/v1/knowledge/indexing-jobs",
            params={"collection_id": "kc_other"},
        )
        assert other.json()["data"] == []


def test_reindex_unknown_collection_returns_404() -> None:
    client = make_indexing_client()
    with client:
        response = client.post(
            "/v1/knowledge/collections/kc_unknown/reindex", json={}
        )
    assert response.status_code == 404
    assert response.json() == {
        "error": {"message": "Collection nicht gefunden", "type": "not_found"}
    }


def test_reindex_events_stream_replays_completion() -> None:
    client = make_indexing_client()
    with client:
        collection_id = _collection_with_two_documents(client)
        started = client.post(
            f"/v1/knowledge/collections/{collection_id}/reindex", json={}
        )
        job_id = started.json()["job_id"]
        _poll_until_terminal(client, job_id)

        events = client.get(f"/v1/knowledge/indexing-jobs/{job_id}/events")
        assert events.status_code == 200
        body = events.text
        assert "inqtrix.index.started" in body
        assert "inqtrix.index.completed" in body


def test_reindex_stream_stops_before_replay_when_credential_is_revoked() -> None:
    """A job stream emits no replay frame after live credential revocation."""
    client = make_indexing_client()
    with client:
        collection_id = _collection_with_two_documents(client)
        started = client.post(
            f"/v1/knowledge/collections/{collection_id}/reindex", json={}
        )
        job_id = started.json()["job_id"]
        _poll_until_terminal(client, job_id)
        client.auth_provider.revoke_live = True  # type: ignore[attr-defined]

        events = client.get(f"/v1/knowledge/indexing-jobs/{job_id}/events")

    assert events.status_code == 200
    assert events.text == ""


def test_get_unknown_job_returns_404() -> None:
    client = make_indexing_client()
    with client:
        response = client.get("/v1/knowledge/indexing-jobs/ix_missing")
    assert response.status_code == 404


def test_active_reindex_blocks_collection_mutations_over_http(monkeypatch) -> None:
    client = make_indexing_client()
    container = client.app.state.container
    service = container.knowledge_service
    assert service is not None
    started = threading.Event()
    release = threading.Event()
    real_reembed = service.reembed_document

    async def blocking_reembed(
        *,
        document,
        embedding_model,
        authority_check=None,
        actor_user_id=None,
    ):
        started.set()
        await asyncio.to_thread(release.wait, 2)
        return await real_reembed(
            document=document,
            embedding_model=embedding_model,
            authority_check=authority_check,
            actor_user_id=actor_user_id,
        )

    monkeypatch.setattr(service, "reembed_document", blocking_reembed)
    with client:
        collection_id = _collection_with_two_documents(client)
        documents = client.get(
            f"/v1/knowledge/collections/{collection_id}/documents"
        ).json()["data"]
        document_id = documents[0]["id"]
        queued = client.post(
            f"/v1/knowledge/collections/{collection_id}/reindex",
            json={},
        )
        assert queued.status_code == 202
        job_id = queued.json()["job_id"]
        assert started.wait(timeout=2)

        blocked = (
            client.post(
                f"/v1/knowledge/collections/{collection_id}/documents",
                json={"title": "blocked", "text": "blocked"},
            ),
            client.delete(f"/v1/knowledge/documents/{document_id}"),
            client.delete(f"/v1/knowledge/collections/{collection_id}"),
        )
        assert {response.status_code for response in blocked} == {409}
        assert {
            response.json()["error"]["type"] for response in blocked
        } == {"collection_maintenance"}

        cancelling = client.post(
            f"/v1/knowledge/indexing-jobs/{job_id}/cancel"
        )
        assert cancelling.status_code == 200
        assert cancelling.json()["status"] == "cancelling"
        still_blocked = client.delete(
            f"/v1/knowledge/documents/{document_id}"
        )
        assert still_blocked.status_code == 409
        assert still_blocked.json()["error"]["type"] == (
            "collection_maintenance"
        )

        release.set()
        final = _poll_until_terminal(client, job_id)
        assert final["status"] == "cancelled"
        assert client.delete(
            f"/v1/knowledge/documents/{document_id}"
        ).status_code == 204


def test_revoking_edit_share_during_embedding_fails_reindex() -> None:
    class BlockingEmbeddings(StubEmbeddings):
        def __init__(self) -> None:
            super().__init__()
            self.block_reindex = False
            self.started = threading.Event()
            self.release = threading.Event()

        def embed_documents(self, texts, *, model=None):
            if self.block_reindex:
                self.started.set()
                self.release.wait(timeout=3)
            return super().embed_documents(texts, model=model)

    embeddings = BlockingEmbeddings()
    client = make_indexing_client(embeddings=embeddings)
    with client:
        collection_id = _collection_with_two_documents(client)
        granted = client.post(
            "/v1/shares",
            json={
                "resource_type": "knowledge_collection",
                "resource_id": collection_id,
                "invitees": [
                    {"user_id": str(RECIPIENT), "permission": "edit"}
                ],
            },
        )
        assert granted.status_code == 201
        share_id = granted.json()["data"][0]["id"]
        accepted = client.post(
            f"/v1/shares/{share_id}/accept",
            headers=user_headers(RECIPIENT),
        )
        assert accepted.status_code == 200

        embeddings.block_reindex = True
        queued = client.post(
            f"/v1/knowledge/collections/{collection_id}/reindex",
            json={},
            headers=user_headers(RECIPIENT),
        )
        assert queued.status_code == 202
        job_id = queued.json()["job_id"]
        try:
            assert embeddings.started.wait(timeout=2)
            revoked = client.delete(f"/v1/shares/{share_id}")
            assert revoked.status_code == 204
        finally:
            embeddings.release.set()

        final = _poll_until_terminal(client, job_id)
        assert final["status"] == "failed"
        assert final["error"]["type"] == "authorization_revoked"
        assert client.get(
            f"/v1/knowledge/indexing-jobs/{job_id}",
            headers=user_headers(RECIPIENT),
        ).status_code == 404

        events = client.get(f"/v1/knowledge/indexing-jobs/{job_id}/events")
        assert "inqtrix.index.failed" in events.text
        assert "inqtrix.index.document_completed" not in events.text
