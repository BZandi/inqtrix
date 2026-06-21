"""HTTP integration for the background reindex endpoints.

Exercises the real router + container wiring (the in-memory knowledge
store, the indexing service, route registration) end-to-end over a
FastAPI ``TestClient``: create a collection, ingest documents, start a
reindex, poll it to completion, list it, and stream its events.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.knowledge.stores.memory import MemoryKnowledgeStore
from inqtrix.knowledge.stores.ports import KnowledgeProviderContext
from inqtrix.providers.base import ProviderContext
from inqtrix.server.container import build_container
from inqtrix.server.routers import indexing, knowledge
from inqtrix.settings import ServerSettings, Settings, StorageSettings

from tests.test_knowledge_routes import KnowledgeStubLLM, StubEmbeddings, _StubSearch


def make_indexing_client() -> TestClient:
    """Build a TestClient exposing the knowledge + reindex routers."""
    container = build_container(
        providers=ProviderContext(llm=KnowledgeStubLLM(), search=_StubSearch()),
        strategies=None,
        settings=Settings(
            server=ServerSettings(public_base_url=""),
            storage=StorageSettings(backend="memory", database_url=""),
        ),
        semaphore_factory=lambda: asyncio.Semaphore(1),
        knowledge=KnowledgeProviderContext(
            embeddings=StubEmbeddings(),
            store=MemoryKnowledgeStore(),
            default_top_k=4,
        ),
    )
    assert container.indexing_service is not None
    app = FastAPI()
    app.include_router(knowledge.build_router(container))
    app.include_router(indexing.build_router(container))
    return TestClient(app)


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


def test_get_unknown_job_returns_404() -> None:
    client = make_indexing_client()
    with client:
        response = client.get("/v1/knowledge/indexing-jobs/ix_missing")
    assert response.status_code == 404
