"""Qdrant store integration tests (gated on INQTRIX_TEST_QDRANT_URL).

Port-parity with the memory store plus the hybrid branch, for the legacy
``QdrantKnowledgeStore`` sole-store (the ``qdrant`` vector backend
WITHOUT Postgres). The full-stack path (Postgres + ``QdrantVectorIndex``)
is verified separately. Start the dev stack and run with:

    INQTRIX_TEST_QDRANT_URL=http://127.0.0.1:6333 \\
    INQTRIX_TEST_QDRANT_API_KEY=inqtrix-dev-qdrant-key \\
    uv run pytest tests/test_qdrant_store.py -v
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from inqtrix.knowledge.stores.ports import (
    CollectionNotFound,
    DocumentNotFound,
    EmbeddingDimensionMismatch,
    KnowledgeProviderContext,
)
from inqtrix.services.knowledge_service import KnowledgeService

from tests.test_knowledge_engine import StubEmbeddings

QDRANT_URL = os.environ.get("INQTRIX_TEST_QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("INQTRIX_TEST_QDRANT_API_KEY", "")

pytestmark = pytest.mark.skipif(
    not QDRANT_URL,
    reason="INQTRIX_TEST_QDRANT_URL not set (Qdrant integration)",
)


@pytest_asyncio.fixture()
async def store():
    from inqtrix.knowledge.stores.qdrant_store import QdrantKnowledgeStore

    instance = QdrantKnowledgeStore(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    created: list[str] = []
    original_create = instance.create_collection

    async def tracking_create(**kwargs):
        collection = await original_create(**kwargs)
        created.append(collection.id)
        return collection

    instance.create_collection = tracking_create  # type: ignore[method-assign]
    yield instance
    for collection_id in created:
        try:
            await instance.delete_collection(collection_id)
        except CollectionNotFound:
            pass


@pytest.fixture()
def service(store) -> KnowledgeService:
    return KnowledgeService(
        knowledge=KnowledgeProviderContext(
            embeddings=StubEmbeddings(),
            store=store,
            default_top_k=4,
        ),
        chunk_max_chars=2_000,
        max_document_chars=100_000,
    )


@pytest.mark.asyncio
async def test_collection_and_document_lifecycle(service, store):
    collection = await service.create_collection(name="Vertraege")
    assert collection.embedding_model == "stub-embed-8"
    assert collection.embedding_dim == 8

    document = await service.add_document(
        collection_id=collection.id,
        title="Rahmenvertrag",
        text="Die Haftung ist auf den Auftragswert begrenzt.",
        metadata={"source": "vertrag.pdf"},
    )
    fetched = await store.get_collection(collection.id)
    assert fetched.document_count == 1

    documents = await store.list_documents(collection.id)
    assert [doc.title for doc in documents] == ["Rahmenvertrag"]
    assert documents[0].metadata == {"source": "vertrag.pdf"}

    full = await store.get_document(document.id)
    assert "Haftung" in full.text

    await store.delete_document(document.id)
    assert (await store.get_collection(collection.id)).document_count == 0
    with pytest.raises(DocumentNotFound):
        await store.get_document(document.id)

    await store.delete_collection(collection.id)
    with pytest.raises(CollectionNotFound):
        await store.get_collection(collection.id)


@pytest.mark.asyncio
async def test_dimension_mismatch_is_rejected_loudly(store):
    collection = await store.create_collection(
        name="K", embedding_model="stub-embed-8", embedding_dim=8
    )
    with pytest.raises(EmbeddingDimensionMismatch, match="dimension 4"):
        await store.add_document(
            collection_id=collection.id,
            title="Falsch",
            text="x",
            metadata={},
            chunks=["x"],
            embeddings=[[1.0] * 4],
        )


@pytest.mark.asyncio
async def test_search_scopes_to_the_requested_collections(service):
    legal = await service.create_collection(name="Recht")
    tech = await service.create_collection(name="Technik")
    await service.add_document(
        collection_id=legal.id,
        title="Haftungsvertrag",
        text="Die Haftung ist auf den Auftragswert begrenzt.",
    )
    await service.add_document(
        collection_id=tech.id,
        title="Serverhandbuch",
        text="Der Server startet mit systemctl start inqtrix.",
    )

    scoped = await service.search(
        query="Haftung Auftragswert begrenzt", collection_ids=[tech.id]
    )
    assert all(hit.document_title == "Serverhandbuch" for hit in scoped)

    everything = await service.search(
        query="Haftung Auftragswert begrenzt",
        collection_ids=[legal.id, tech.id],
    )
    assert everything[0].document_title == "Haftungsvertrag"


@pytest.mark.asyncio
async def test_hybrid_lexical_branch_finds_exact_terms(service, store):
    """BM25 must surface an exact German term even when the stub dense
    vectors are uninformative — the hybrid value proposition."""
    assert store.supports_hybrid is True
    collection = await service.create_collection(name="Hybrid")
    await service.add_document(
        collection_id=collection.id,
        title="Reisekosten",
        text="Die Verpflegungspauschale betraegt 28 Euro pro Reisetag.",
    )
    await service.add_document(
        collection_id=collection.id,
        title="Onboarding",
        text="Neue Mitarbeitende erhalten einen Buddy fuer drei Monate.",
    )

    hits = await service.search(
        query="Verpflegungspauschale", collection_ids=[collection.id]
    )
    assert hits[0].document_title == "Reisekosten"


@pytest.mark.asyncio
async def test_unknown_collection_raises(store):
    with pytest.raises(CollectionNotFound):
        await store.get_collection("kc_does_not_exist")
    with pytest.raises(CollectionNotFound):
        await store.list_documents("kc_does_not_exist")
