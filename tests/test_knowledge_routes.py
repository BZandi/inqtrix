"""HTTP tests for the knowledge surface and the knowledge algorithm.

Wires a stub embedding provider and a stub LLM through
``build_container(knowledge=...)`` — the same Baukasten injection seam
deployments use — and exercises collections, ingestion, search,
``mode=knowledge`` through chat completions and native runs, the
streaming rejection, and the capability manifest.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.knowledge.stores.memory import MemoryKnowledgeStore
from inqtrix.knowledge.stores.ports import KnowledgeProviderContext
from inqtrix.providers.base import LLMResponse, ProviderContext
from inqtrix.search_result import GroundedSearchResult
from inqtrix.server.container import build_container
from inqtrix.server.routers import capabilities, chat, knowledge, runs, sources
from inqtrix.settings import ServerSettings, Settings, StorageSettings

from tests.contract._app import wait_for_run_status
from tests.test_knowledge_engine import StubEmbeddings


class KnowledgeStubLLM:
    """LLM stub answering knowledge prompts with a cited sentence."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def complete(self, *args: Any, **kwargs: Any) -> str:
        return "ok"

    def complete_with_metadata(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        return LLMResponse(
            content="Die Haftung ist auf den Auftragswert begrenzt [K1].",
            prompt_tokens=42,
            completion_tokens=11,
            model="stub-answer-model",
            finish_reason="stop",
        )

    def is_available(self) -> bool:
        return True


class _StubSearch:
    def search(self, *args: Any, **kwargs: Any) -> GroundedSearchResult:
        return GroundedSearchResult()

    def is_available(self) -> bool:
        return True


def make_knowledge_client(
    *,
    llm: KnowledgeStubLLM | None = None,
    with_knowledge: bool = True,
    settings: Settings | None = None,
) -> tuple[TestClient, KnowledgeStubLLM]:
    active_llm = llm or KnowledgeStubLLM()
    knowledge_context = (
        KnowledgeProviderContext(
            embeddings=StubEmbeddings(),
            store=MemoryKnowledgeStore(),
            default_top_k=4,
        )
        if with_knowledge
        else None
    )
    container = build_container(
        providers=ProviderContext(llm=active_llm, search=_StubSearch()),
        strategies=None,
        # Pinned defaults keep the suite hermetic against developer
        # .env files (citation scheme + storage backend).
        settings=settings
        or Settings(
            server=ServerSettings(public_base_url=""),
            storage=StorageSettings(backend="memory", database_url=""),
        ),
        semaphore_factory=lambda: asyncio.Semaphore(1),
        knowledge=knowledge_context,
    )
    app = FastAPI()
    app.include_router(capabilities.build_router(container))
    app.include_router(chat.build_router(container))
    app.include_router(runs.build_router(container))
    if container.knowledge_service is not None:
        app.include_router(knowledge.build_router(container))
        app.include_router(sources.build_router(container))
    return TestClient(app), active_llm


def _create_collection_with_document(client: TestClient) -> str:
    created = client.post(
        "/v1/knowledge/collections", json={"name": "Vertraege"}
    )
    assert created.status_code == 201
    collection_id = created.json()["id"]
    ingested = client.post(
        f"/v1/knowledge/collections/{collection_id}/documents",
        json={
            "title": "Rahmenvertrag Kunde X",
            "text": "Die Haftung ist auf den Auftragswert begrenzt.",
            "metadata": {"source": "vertrag.pdf"},
        },
    )
    assert ingested.status_code == 201
    return collection_id


# ------------------------------------------------------------------ #
# Collection/document/search surface
# ------------------------------------------------------------------ #


def test_collection_and_document_lifecycle_over_http():
    client, _ = make_knowledge_client()
    with client:
        collection_id = _create_collection_with_document(client)

        listing = client.get("/v1/knowledge/collections")
        assert listing.json()["data"][0]["document_count"] == 1
        assert listing.json()["data"][0]["embedding_model"] == "stub-embed-8"

        documents = client.get(
            f"/v1/knowledge/collections/{collection_id}/documents"
        )
        payload = documents.json()["data"][0]
        assert payload["title"] == "Rahmenvertrag Kunde X"
        assert payload["chunk_count"] == 1
        assert payload["metadata"] == {"source": "vertrag.pdf"}

        deleted = client.delete(f"/v1/knowledge/documents/{payload['id']}")
        assert deleted.status_code == 204
        gone = client.delete(f"/v1/knowledge/collections/{collection_id}")
        assert gone.status_code == 204


def test_search_returns_scored_candidates():
    client, _ = make_knowledge_client()
    with client:
        _create_collection_with_document(client)

        response = client.post(
            "/v1/knowledge/search",
            json={"query": "Haftung Auftragswert begrenzt", "top_k": 3},
        )

    assert response.status_code == 200
    hit = response.json()["data"][0]
    assert hit["document_title"] == "Rahmenvertrag Kunde X"
    assert hit["score"] > 0
    assert "Haftung" in hit["text"]


def test_unknown_collection_returns_404_envelope():
    client, _ = make_knowledge_client()
    with client:
        response = client.get(
            "/v1/knowledge/collections/kc_unknown/documents"
        )
    assert response.status_code == 404
    assert response.json() == {
        "error": {"message": "Collection nicht gefunden", "type": "not_found"}
    }


def test_validation_errors_return_400_envelope():
    client, _ = make_knowledge_client()
    with client:
        empty_name = client.post("/v1/knowledge/collections", json={"name": " "})
        bad_top_k = client.post(
            "/v1/knowledge/search", json={"query": "x", "top_k": 0}
        )
    assert empty_name.status_code == 400
    assert empty_name.json()["error"]["type"] == "invalid_request_error"
    assert bad_top_k.status_code == 400


# ------------------------------------------------------------------ #
# mode=knowledge execution
# ------------------------------------------------------------------ #


def test_chat_completion_in_knowledge_mode_answers_with_references(monkeypatch):
    client, llm = make_knowledge_client()
    with client:
        collection_id = _create_collection_with_document(client)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Wie ist die Haftung geregelt?"}
                ],
                "mode": "knowledge",
                "stream": False,
                "knowledge_filters": {"collection_ids": [collection_id]},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == (
        "Die Haftung ist auf den Auftragswert begrenzt [K1]."
    )
    # Two LLM calls: the sufficiency gate (fast tier) + the answer —
    # usage accounts for BOTH (the stub bills 42/11 per call).
    assert payload["usage"] == {
        "prompt_tokens": 84,
        "completion_tokens": 22,
        "total_tokens": 106,
    }
    # Call 0 is the gate, call 1 the answer; both carry the evidence.
    assert "[K1] Rahmenvertrag Kunde X" in llm.prompts[0]
    assert "[K1] Rahmenvertrag Kunde X" in llm.prompts[1]
    assert "AUSSCHLIESSLICH" in llm.prompts[1]


def test_native_run_in_knowledge_mode_completes_with_references():
    client, _ = make_knowledge_client()
    with client:
        collection_id = _create_collection_with_document(client)

        created = client.post(
            "/v1/runs",
            json={
                "question": "Wie ist die Haftung geregelt?",
                "mode": "knowledge",
                "knowledge_filters": {"collection_ids": [collection_id]},
            },
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        wait_for_run_status(client, run_id, "completed")

        result = client.get(f"/v1/runs/{run_id}/result")

    assert result.status_code == 200
    payload = result.json()
    assert payload["answer"].endswith("[K1].")
    reference = payload["references"][0]
    assert reference["label"] == "K1"
    # The export pipeline normalizes URLs (fragments are stripped);
    # the stable contract is the internal document URI scheme.
    assert reference["url"].startswith("inqtrix://documents/kd_")
    assert payload["usage"]["total_tokens"] == 106  # gate + answer


def test_streaming_knowledge_mode_is_rejected_loudly():
    client, _ = make_knowledge_client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Frage"}],
                "mode": "knowledge",
                "stream": True,
            },
        )
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "mode='knowledge' unterstuetzt stream=true noch nicht",
            "type": "invalid_request_error",
        }
    }


def test_knowledge_mode_with_empty_store_says_no_evidence():
    client, _ = make_knowledge_client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Frage ohne Doku?"}],
                "mode": "knowledge",
                "stream": False,
            },
        )
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "keine relevanten" in content


# ------------------------------------------------------------------ #
# Capability manifest
# ------------------------------------------------------------------ #


def test_capabilities_with_knowledge_enabled():
    client, _ = make_knowledge_client()
    with client:
        payload = client.get("/v1/capabilities").json()

    algorithm_ids = [entry["id"] for entry in payload["algorithms"]]
    assert algorithm_ids == ["research", "direct_llm", "knowledge"]
    assert payload["features"]["knowledge"] is True
    assert payload["knowledge"]["default_embedding_model"] == "stub-embed-8"
    assert payload["knowledge"]["default_top_k"] == 4
    catalog = payload["knowledge"]["embedding_catalog"]
    assert catalog == [{"model_id": "stub-embed-8", "card": None}]


def test_capabilities_without_knowledge():
    client, _ = make_knowledge_client(with_knowledge=False)
    with client:
        payload = client.get("/v1/capabilities").json()

    algorithm_ids = [entry["id"] for entry in payload["algorithms"]]
    assert algorithm_ids == ["research", "direct_llm"]
    assert payload["features"]["knowledge"] is False
    assert "knowledge" not in payload


def test_knowledge_mode_unregistered_when_disabled():
    client, _ = make_knowledge_client(with_knowledge=False)
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Frage"}],
                "mode": "knowledge",
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "mode muss 'research' oder 'direct_llm' sein"
    )


# ------------------------------------------------------------------ #
# Settings bridge
# ------------------------------------------------------------------ #


def test_build_knowledge_context_disabled_by_default():
    from inqtrix.server.container import build_knowledge_context

    assert build_knowledge_context(Settings()) is None


def test_build_knowledge_context_constructs_from_settings():
    from inqtrix.providers.embeddings import LiteLLMEmbeddings
    from inqtrix.server.container import build_knowledge_context
    from inqtrix.settings import KnowledgeSettings

    settings = Settings(
        knowledge=KnowledgeSettings(
            enabled=True,
            embedding_model="text-embedding-3-small",
            selectable_embedding_models=(
                "text-embedding-3-small, text-embedding-3-large"
            ),
            default_top_k=5,
        ),
    )
    context = build_knowledge_context(settings)

    assert context is not None
    assert isinstance(context.embeddings, LiteLLMEmbeddings)
    assert context.embeddings.default_model == "text-embedding-3-small"
    assert context.embeddings.selectable_embedding_models == [
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]
    assert context.default_top_k == 5

# ------------------------------------------------------------------ #
# Sources view + citation URLs
# ------------------------------------------------------------------ #


def test_source_view_serves_the_cited_document():
    client, _ = make_knowledge_client()
    with client:
        collection_id = _create_collection_with_document(client)
        document_id = client.get(
            f"/v1/knowledge/collections/{collection_id}/documents"
        ).json()["data"][0]["id"]

        response = client.get(f"/v1/sources/{document_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "Rahmenvertrag Kunde X"
    assert "Haftung" in payload["text"]
    assert payload["collection_id"] == collection_id


def test_source_view_unknown_document_is_a_404_envelope():
    client, _ = make_knowledge_client()
    with client:
        response = client.get("/v1/sources/kd_unknown")
    assert response.status_code == 404
    assert response.json() == {
        "error": {"message": "Quelle nicht gefunden", "type": "not_found"}
    }


def test_citations_switch_to_http_when_public_base_url_is_set():
    settings = Settings(
        server=ServerSettings(public_base_url="https://inqtrix.example/")
    )
    client, _ = make_knowledge_client(settings=settings)
    with client:
        collection_id = _create_collection_with_document(client)
        created = client.post(
            "/v1/runs",
            json={
                "question": "Wie ist die Haftung geregelt?",
                "mode": "knowledge",
                "knowledge_filters": {"collection_ids": [collection_id]},
            },
        )
        run_id = created.json()["run_id"]
        wait_for_run_status(client, run_id, "completed")
        result = client.get(f"/v1/runs/{run_id}/result").json()

    reference = result["references"][0]
    assert reference["url"].startswith(
        "https://inqtrix.example/v1/sources/kd_"
    )
    assert "chunk=" in reference["url"]

def test_source_view_is_deliberately_unscoped_in_v1():
    """Knowledge documents carry no owner yet — every authenticated
    principal may read sources, exactly like the collection/document
    endpoints. Pinned so a future ownership model changes this test
    deliberately, not accidentally."""
    client, _ = make_knowledge_client()
    with client:
        collection_id = _create_collection_with_document(client)
        document_id = client.get(
            f"/v1/knowledge/collections/{collection_id}/documents"
        ).json()["data"][0]["id"]
        response = client.get(f"/v1/sources/{document_id}")
    assert response.status_code == 200


def test_capabilities_lists_retrieval_profiles_with_degradation():
    """All profiles are always listed; operator degradation (no
    reranker wired here) is shown per profile, never hidden."""
    client, _ = make_knowledge_client()
    with client:
        payload = client.get("/v1/capabilities").json()
    knowledge = payload["knowledge"]
    assert knowledge["default_profile"] == "standard"
    profiles = {entry["id"]: entry for entry in knowledge["profiles"]}
    assert set(profiles) == {
        "schnell", "standard", "gruendlich", "tief", "auto",
    }
    assert profiles["standard"]["stages"]["gate_rounds"] == 1
    assert profiles["tief"]["stages"]["decompose"] is True
    # The fixture wires no reranker: every rerank-wanting profile
    # must report the degradation.
    assert profiles["standard"]["stages"]["rerank"] is False
    assert "rerank" in profiles["standard"]["degraded"]
    assert profiles["schnell"]["degraded"] == []
    assert profiles["auto"]["delegates_to"] == [
        "schnell", "standard", "gruendlich",
    ]


def test_capabilities_without_knowledge_has_no_profiles():
    client, _ = make_knowledge_client(with_knowledge=False)
    with client:
        payload = client.get("/v1/capabilities").json()
    assert "knowledge" not in payload


def test_document_text_serves_the_reader_view():
    client, _ = make_knowledge_client()
    with client:
        collection_id = _create_collection_with_document(client)
        documents = client.get(
            f"/v1/knowledge/collections/{collection_id}/documents"
        ).json()["data"]
        document_id = documents[0]["id"]
        payload = client.get(
            f"/v1/knowledge/documents/{document_id}/text"
        ).json()
        assert payload["id"] == document_id
        assert "Haftung" in payload["text"]
        assert payload["title"] == "Rahmenvertrag Kunde X"
        missing = client.get("/v1/knowledge/documents/kd_missing/text")
        assert missing.status_code == 404
