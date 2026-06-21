"""Unit tests for the in-process vector index (offline, no infra).

Covers the behaviour the Postgres-canonical store relies on: dense
ranking, collection + embedding-model scoping, deletion, top_k capping,
and the dense-only hybrid guard.
"""

from __future__ import annotations

import pytest

from inqtrix.knowledge.stores.vector_index import (
    ChunkVector,
    MemoryVectorIndex,
    VectorHit,
)


def _cv(chunk_id: str, dense: list[float]) -> ChunkVector:
    return ChunkVector(chunk_id=chunk_id, dense=tuple(dense), text=chunk_id)


@pytest.mark.asyncio
async def test_search_ranks_by_cosine_within_scope() -> None:
    index = MemoryVectorIndex()
    await index.upsert(
        embedding_model="m",
        collection_id="c1",
        document_id="d1",
        vectors=[_cv("a", [1.0, 0.0]), _cv("b", [0.0, 1.0])],
    )
    hits = await index.search(
        embedding_model="m",
        query_embedding=[1.0, 0.0],
        collection_ids=["c1"],
        top_k=2,
    )
    assert [hit.chunk_id for hit in hits] == ["a", "b"]
    assert hits[0].score > hits[1].score


@pytest.mark.asyncio
async def test_search_scopes_by_collection_and_model() -> None:
    index = MemoryVectorIndex()
    await index.upsert(
        embedding_model="m", collection_id="c1", document_id="d1",
        vectors=[_cv("a", [1.0, 0.0])],
    )
    await index.upsert(
        embedding_model="m", collection_id="c2", document_id="d2",
        vectors=[_cv("b", [1.0, 0.0])],
    )
    await index.upsert(
        embedding_model="other", collection_id="c1", document_id="d3",
        vectors=[_cv("c", [1.0, 0.0])],
    )
    hits = await index.search(
        embedding_model="m", query_embedding=[1.0, 0.0],
        collection_ids=["c1"], top_k=10,
    )
    # Only the c1 + model "m" chunk; c2 (other collection) and the
    # "other"-model chunk are out of scope.
    assert [hit.chunk_id for hit in hits] == ["a"]


@pytest.mark.asyncio
async def test_top_k_caps_results() -> None:
    index = MemoryVectorIndex()
    await index.upsert(
        embedding_model="m", collection_id="c1", document_id="d1",
        vectors=[_cv(str(i), [float(i), 1.0]) for i in range(5)],
    )
    hits = await index.search(
        embedding_model="m", query_embedding=[1.0, 1.0],
        collection_ids=["c1"], top_k=2,
    )
    assert len(hits) == 2


@pytest.mark.asyncio
async def test_delete_document_and_collection() -> None:
    index = MemoryVectorIndex()
    await index.upsert(
        embedding_model="m", collection_id="c1", document_id="d1",
        vectors=[_cv("a", [1.0, 0.0])],
    )
    await index.upsert(
        embedding_model="m", collection_id="c1", document_id="d2",
        vectors=[_cv("b", [0.0, 1.0])],
    )
    await index.delete_document(embedding_model="m", document_id="d1")
    hits = await index.search(
        embedding_model="m", query_embedding=[1.0, 0.0],
        collection_ids=["c1"], top_k=10,
    )
    assert {hit.chunk_id for hit in hits} == {"b"}
    await index.delete_collection(embedding_model="m", collection_id="c1")
    hits = await index.search(
        embedding_model="m", query_embedding=[1.0, 0.0],
        collection_ids=["c1"], top_k=10,
    )
    assert hits == []


@pytest.mark.asyncio
async def test_hybrid_search_is_unsupported() -> None:
    index = MemoryVectorIndex()
    assert index.supports_hybrid is False
    with pytest.raises(NotImplementedError):
        await index.hybrid_search(
            embedding_model="m", query_text="q", query_embedding=[1.0],
            collection_ids=["c1"], top_k=1,
        )


def test_vector_hit_shape() -> None:
    hit = VectorHit(chunk_id="kch_x", score=0.5)
    assert hit.chunk_id == "kch_x"
    assert hit.score == 0.5
