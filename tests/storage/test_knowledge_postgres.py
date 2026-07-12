"""Postgres integration tests for the canonical knowledge store (gated).

Same gating/conventions as the other storage suites: a disposable
database via ``INQTRIX_TEST_DATABASE_URL``, operations under the
restricted app role, RLS as the second defense layer. Uses the
in-process :class:`MemoryVectorIndex` for vectors so the suite needs
only Postgres (no Qdrant). Verifies the full async ``KnowledgeStore``
contract against Postgres: collection/document lifecycle, document
counts, chunk persistence, retrieval hydration, in-place reembed, cascade
delete, and the dimension/absence error paths.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.knowledge.stores.ports import (
    CollectionNotFound,
    DocumentNotFound,
    EmbeddingDimensionMismatch,
)
from inqtrix.knowledge.stores.postgres_store import PostgresKnowledgeStore
from inqtrix.knowledge.stores.vector_index import ChunkVector, MemoryVectorIndex
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.knowledge_orm import (
    knowledge_chunks,
    knowledge_collections,
    knowledge_documents,
)
from inqtrix.storage.migrate import run_migrations

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"


@pytest.fixture(scope="session", autouse=True)
def knowledge_schema_migrated():
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
            # Child→parent order (FK), though CASCADE would handle it.
            await session.execute(knowledge_chunks.delete())
            await session.execute(knowledge_documents.delete())
            await session.execute(knowledge_collections.delete())
    knowledge_store = PostgresKnowledgeStore(
        engine=engine, app_role=APP_ROLE, vector_index=MemoryVectorIndex()
    )
    yield knowledge_store
    await knowledge_store.aclose()


@pytest.mark.asyncio
async def test_collection_lifecycle_and_document_count(store) -> None:
    collection = await store.create_collection(
        name="Vertraege", embedding_model="m", embedding_dim=2,
        created_by_sub="user-1",
    )
    assert collection.embedding_model == "m"
    assert collection.document_count == 0

    fetched = await store.get_collection(collection.id)
    assert fetched.created_by_sub == "user-1"
    assert (await store.list_collections())[0].id == collection.id

    document = await store.add_document(
        collection_id=collection.id, title="Doc A", text="alpha beta",
        metadata={"source": "a.pdf"},
        chunks=["alpha", "beta"], embeddings=[[1.0, 0.0], [0.0, 1.0]],
        source_chunks=["alpha", "beta"],
    )
    assert document.chunk_count == 2
    assert (await store.get_collection(collection.id)).document_count == 1

    docs = await store.list_documents(collection.id)
    assert [d.id for d in docs] == [document.id]
    assert (await store.get_document(document.id)).text == "alpha beta"
    assert (await store.get_document(document.id)).metadata == {"source": "a.pdf"}


@pytest.mark.asyncio
async def test_list_documents_page_keyset_walks_without_skip_or_repeat(store) -> None:
    """The DB keyset (tuple_(created_at, id) < cursor) pages a collection's
    documents with no gaps or duplicates, including across a created_at tie."""
    collection = await store.create_collection(
        name="Paged", embedding_model="m", embedding_dim=2, created_by_sub="u",
    )
    added = []
    for n in range(5):
        doc = await store.add_document(
            collection_id=collection.id, title=f"Doc {n}", text="x",
            metadata={}, chunks=["x"], embeddings=[[1.0, 0.0]],
            source_chunks=["x"],
        )
        added.append(doc.id)

    full = [d.id for d in await store.list_documents(collection.id)]
    seen: list[str] = []
    cursor = None
    for _ in range(10):  # generous bound
        page, next_cursor = await store.list_documents_page(
            collection.id, limit=2, after=cursor,
        )
        seen.extend(d.id for d in page)
        assert len(page) <= 2
        if next_cursor is None:
            break
        from inqtrix.pagination import decode_cursor
        cursor = decode_cursor(next_cursor)

    assert seen == full  # same order as the unpaginated list
    assert len(seen) == len(set(seen)) == 5


@pytest.mark.asyncio
async def test_search_hydrates_from_postgres(store) -> None:
    collection = await store.create_collection(
        name="C", embedding_model="m", embedding_dim=2
    )
    await store.add_document(
        collection_id=collection.id, title="Haftung", text="full text",
        metadata={}, chunks=["limit clause"], embeddings=[[1.0, 0.0]],
        source_chunks=["limit clause"],
    )
    candidates = await store.search(
        query_embedding=[1.0, 0.0], collection_ids=[collection.id],
        top_k=5, embedding_model="m",
    )
    assert len(candidates) == 1
    hit = candidates[0]
    # Text + title were hydrated from Postgres, not the vector index.
    assert hit.chunk.text == "limit clause"
    assert hit.chunk.source_text == "limit clause"
    assert hit.document_title == "Haftung"
    assert hit.score > 0


@pytest.mark.asyncio
async def test_reembed_replaces_chunks_in_place(store) -> None:
    collection = await store.create_collection(
        name="C", embedding_model="m", embedding_dim=2
    )
    document = await store.add_document(
        collection_id=collection.id, title="D", text="t",
        metadata={}, chunks=["one"], embeddings=[[1.0, 0.0]],
    )
    updated = await store.reembed_document(
        document_id=document.id,
        chunks=["one", "two"], embeddings=[[1.0, 0.0], [0.0, 1.0]],
        source_chunks=["one", "two"],
    )
    assert updated.id == document.id
    assert updated.chunk_count == 2
    hits = await store.search(
        query_embedding=[0.0, 1.0], collection_ids=[collection.id],
        top_k=5, embedding_model="m",
    )
    assert hits and hits[0].chunk.text == "two"


@pytest.mark.asyncio
async def test_reembed_preserves_chunk_ids_by_position(store) -> None:
    """Reindex keeps chunk ids stable by position so citations survive:
    the first chunk keeps its id, growth appends a fresh id, and a shrink
    drops the tail — verified through the public ``get_chunks`` read."""
    collection = await store.create_collection(
        name="C", embedding_model="m", embedding_dim=2
    )
    document = await store.add_document(
        collection_id=collection.id, title="D", text="t",
        metadata={}, chunks=["one", "two"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        source_chunks=["one", "two"],
    )
    before = await store.get_chunks(document.id)
    assert [c.chunk_index for c in before] == [0, 1]
    id0, id1 = before[0].id, before[1].id

    # Grow to three chunks: positions 0/1 keep their ids, 2 is fresh.
    await store.reembed_document(
        document_id=document.id,
        chunks=["one", "two", "three"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        source_chunks=["one", "two", "three"],
    )
    grown = await store.get_chunks(document.id)
    assert [c.id for c in grown[:2]] == [id0, id1]
    assert grown[2].id not in {id0, id1}

    # Shrink back to one chunk: position 0 keeps its original id.
    await store.reembed_document(
        document_id=document.id,
        chunks=["one"], embeddings=[[1.0, 0.0]], source_chunks=["one"],
    )
    shrunk = await store.get_chunks(document.id)
    assert [c.id for c in shrunk] == [id0]


@pytest.mark.asyncio
async def test_get_chunks_orders_and_404s(store) -> None:
    collection = await store.create_collection(
        name="C", embedding_model="m", embedding_dim=2
    )
    document = await store.add_document(
        collection_id=collection.id, title="D", text="t",
        metadata={}, chunks=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    chunks = await store.get_chunks(document.id)
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    with pytest.raises(DocumentNotFound):
        await store.get_chunks("kd_unknown")


@pytest.mark.asyncio
async def test_delete_document_and_collection_cascade(store) -> None:
    collection = await store.create_collection(
        name="C", embedding_model="m", embedding_dim=2
    )
    document = await store.add_document(
        collection_id=collection.id, title="D", text="t",
        metadata={}, chunks=["one"], embeddings=[[1.0, 0.0]],
    )
    await store.delete_document(document.id)
    assert (await store.get_collection(collection.id)).document_count == 0
    with pytest.raises(DocumentNotFound):
        await store.get_document(document.id)

    await store.delete_collection(collection.id)
    with pytest.raises(CollectionNotFound):
        await store.get_collection(collection.id)


@pytest.mark.asyncio
async def test_reconcile_deletes_orphan_vectors_keeps_canonical(store) -> None:
    """The reverse reconcile sweep removes vectors whose canonical Postgres
    document is gone (non-atomic cross-store delete residue) and leaves the
    documents that still exist untouched."""
    collection = await store.create_collection(
        name="C", embedding_model="m", embedding_dim=2
    )
    document = await store.add_document(
        collection_id=collection.id, title="D", text="t",
        metadata={}, chunks=["one"], embeddings=[[1.0, 0.0]],
    )
    # Strand vectors for a document that has NO canonical Postgres row,
    # exactly the drift a crash between the PG commit and the vector delete
    # leaves behind.
    await store._vectors.upsert(
        embedding_model="m", collection_id="kc_ghost", document_id="kd_ghost",
        vectors=[ChunkVector(chunk_id="kch_ghost", dense=(0.0, 1.0), text="x")],
    )
    before = await store._vectors.scroll_chunk_groups(embedding_model="m")
    assert ("kc_ghost", "kd_ghost") in before
    assert (collection.id, document.id) in before

    report = await store.reconcile_orphans()
    assert report["deleted_documents"] == 1
    assert report["details"] == [
        {"document_id": "kd_ghost", "embedding_model": "m"}
    ]

    after = await store._vectors.scroll_chunk_groups(embedding_model="m")
    assert ("kc_ghost", "kd_ghost") not in after  # orphan swept
    assert (collection.id, document.id) in after  # canonical untouched


@pytest.mark.asyncio
async def test_error_paths(store) -> None:
    with pytest.raises(CollectionNotFound):
        await store.get_collection("kc_missing")
    with pytest.raises(DocumentNotFound):
        await store.get_document("kd_missing")
    collection = await store.create_collection(
        name="C", embedding_model="m", embedding_dim=2
    )
    with pytest.raises(EmbeddingDimensionMismatch):
        await store.add_document(
            collection_id=collection.id, title="D", text="t", metadata={},
            chunks=["one"], embeddings=[[1.0, 0.0, 0.0]],  # wrong dim
        )
    with pytest.raises(EmbeddingDimensionMismatch):
        await store.add_document(
            collection_id=collection.id, title="D", text="t", metadata={},
            chunks=["one", "two"], embeddings=[[1.0, 0.0]],  # count mismatch
        )
