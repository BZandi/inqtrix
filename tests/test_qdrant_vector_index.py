"""Offline contract tests for the Postgres-facing Qdrant vector half."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from inqtrix.knowledge.stores.qdrant_store import QdrantVectorIndex
from inqtrix.knowledge.stores.vector_index import ChunkVector, VectorSearchScope


class _SparseStub:
    """Deterministic sparse vectors without model downloads."""

    @staticmethod
    def documents(texts: list[str]):
        return [SimpleNamespace(indices=[0], values=[1.0]) for _text in texts]

    @staticmethod
    def query(_text: str):
        return SimpleNamespace(indices=[0], values=[1.0])


def _index(*, sparse: bool = False) -> QdrantVectorIndex:
    index = object.__new__(QdrantVectorIndex)
    index._client = qdrant_client.QdrantClient(location=":memory:")
    index._sparse_enabled = sparse
    index._bm25 = _SparseStub()
    return index


def _active_scope() -> VectorSearchScope:
    return VectorSearchScope(
        collection_id="kc_active",
        generation_id="gen_active",
        active_revision_ids=("rev_active",),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("hybrid", [False, True])
async def test_active_generation_and_revision_filter_before_ranking(
    hybrid: bool,
) -> None:
    index = _index(sparse=hybrid)
    await index.ensure_model(embedding_model="m", embedding_dim=2)
    await index.upsert(
        embedding_model="m",
        collection_id="kc_active",
        document_id="kd_active",
        vectors=[
            ChunkVector(
                "kch_active",
                (0.8, 0.2),
                text="active",
                generation_id="gen_active",
                revision_id="rev_active",
            ),
            ChunkVector(
                "kch_staged_generation",
                (1.0, 0.0),
                text="staged",
                generation_id="gen_staged",
                revision_id="rev_active",
            ),
            ChunkVector(
                "kch_staged_revision",
                (1.0, 0.0),
                text="staged",
                generation_id="gen_active",
                revision_id="rev_staged",
            ),
        ],
    )

    if hybrid:
        hits = await index.hybrid_search(
            embedding_model="m",
            query_text="active",
            query_embedding=[1.0, 0.0],
            scopes=[_active_scope()],
            top_k=3,
        )
    else:
        hits = await index.search(
            embedding_model="m",
            query_embedding=[1.0, 0.0],
            scopes=[_active_scope()],
            top_k=3,
        )

    assert [hit.chunk_id for hit in hits] == ["kch_active"]


@pytest.mark.asyncio
async def test_legacy_none_fields_are_exact_inside_mixed_scope() -> None:
    index = _index()
    await index.ensure_model(embedding_model="m", embedding_dim=2)
    await index.upsert(
        embedding_model="m",
        collection_id="kc_legacy",
        document_id="kd_legacy",
        vectors=[ChunkVector("kch_legacy", (1.0, 0.0))],
    )
    await index.upsert(
        embedding_model="m",
        collection_id="kc_active",
        document_id="kd_active",
        vectors=[
            ChunkVector(
                "kch_active",
                (0.8, 0.2),
                generation_id="gen_active",
                revision_id="rev_active",
            ),
            # Missing payload fields in a modern collection must not leak
            # through the OR branch created for the separate legacy scope.
            ChunkVector("kch_missing_modern_scope", (1.0, 0.0)),
        ],
    )
    await index.upsert(
        embedding_model="m",
        collection_id="kc_active",
        document_id="kd_compat",
        vectors=[
            ChunkVector(
                "kch_compat",
                (0.7, 0.3),
                generation_id="gen_active",
                revision_id=None,
            )
        ],
    )

    hits = await index.search(
        embedding_model="m",
        query_embedding=[1.0, 0.0],
        scopes=[
            VectorSearchScope(
                collection_id="kc_legacy",
                generation_id=None,
                legacy_document_ids=("kd_legacy",),
            ),
            VectorSearchScope(
                collection_id="kc_active",
                generation_id="gen_active",
                active_revision_ids=("rev_active",),
                legacy_document_ids=("kd_compat",),
            ),
        ],
        top_k=5,
    )

    assert {hit.chunk_id for hit in hits} == {
        "kch_legacy",
        "kch_active",
        "kch_compat",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("hybrid", [False, True])
async def test_migrated_payload_compatibility_is_chunk_exact(
    hybrid: bool,
) -> None:
    """Only migration-verified chunk ids admit pre-lineage payloads.

    Missing-lineage points in the same document and another document of the
    modern collection must remain excluded. Canonical Postgres hydration is
    responsible for the subsequent revision/span verification.
    """

    index = _index(sparse=hybrid)
    await index.ensure_model(embedding_model="m", embedding_dim=2)
    await index.upsert(
        embedding_model="m",
        collection_id="kc_active",
        document_id="kd_migrated",
        vectors=[ChunkVector("kch_migrated", (0.9, 0.1), text="migrated")],
    )
    await index.upsert(
        embedding_model="m",
        collection_id="kc_active",
        document_id="kd_migrated",
        vectors=[
            ChunkVector(
                "kch_same_document_unverified",
                (0.95, 0.05),
                text="unverified",
            )
        ],
    )
    await index.upsert(
        embedding_model="m",
        collection_id="kc_active",
        document_id="kd_other",
        vectors=[ChunkVector("kch_missing_lineage", (1.0, 0.0), text="other")],
    )
    await index.upsert(
        embedding_model="m",
        collection_id="kc_active",
        document_id="kd_active",
        vectors=[
            ChunkVector(
                "kch_active",
                (0.8, 0.2),
                text="active",
                generation_id="gen_active",
                revision_id="rev_active",
            )
        ],
    )
    scope = VectorSearchScope(
        collection_id="kc_active",
        generation_id="gen_active",
        active_revision_ids=("rev_active", "rev_migrated"),
        legacy_payload_chunk_ids=("kch_migrated",),
    )

    if hybrid:
        hits = await index.hybrid_search(
            embedding_model="m",
            query_text="active migrated",
            query_embedding=[1.0, 0.0],
            scopes=[scope],
            top_k=5,
        )
    else:
        hits = await index.search(
            embedding_model="m",
            query_embedding=[1.0, 0.0],
            scopes=[scope],
            top_k=5,
        )

    assert {hit.chunk_id for hit in hits} == {"kch_active", "kch_migrated"}


@pytest.mark.asyncio
async def test_scroll_returns_exact_chunk_scope_payload() -> None:
    index = _index()
    await index.ensure_model(embedding_model="m", embedding_dim=2)
    await index.upsert(
        embedding_model="m",
        collection_id="kc_active",
        document_id="kd_active",
        vectors=[
            ChunkVector(
                "kch_active",
                (1.0, 0.0),
                generation_id="gen_active",
                revision_id="rev_active",
            )
        ],
    )

    refs = await index.scroll_chunk_points(embedding_model="m")

    assert [(ref.chunk_id, ref.generation_id, ref.revision_id) for ref in refs] == [
        ("kch_active", "gen_active", "rev_active")
    ]


@pytest.mark.asyncio
async def test_generation_count_is_scoped_by_collection_and_generation() -> None:
    index = _index()
    await index.ensure_model(embedding_model="m", embedding_dim=2)
    for collection_id, document_id, chunk_id, generation_id in (
        ("kc_a", "kd_a", "kch_a1", "gen_target"),
        ("kc_a", "kd_a", "kch_a2", "gen_target"),
        ("kc_a", "kd_a", "kch_other_generation", "gen_other"),
        ("kc_b", "kd_b", "kch_other_collection", "gen_target"),
    ):
        await index.upsert(
            embedding_model="m",
            collection_id=collection_id,
            document_id=document_id,
            vectors=[
                ChunkVector(
                    chunk_id,
                    (1.0, 0.0),
                    generation_id=generation_id,
                    revision_id="rev",
                )
            ],
        )

    assert await index.count_generation(
        embedding_model="m",
        collection_id="kc_a",
        generation_id="gen_target",
    ) == 2
    await index.delete_generation(
        embedding_model="m",
        collection_id="kc_a",
        generation_id="gen_target",
    )
    assert await index.count_generation(
        embedding_model="m",
        collection_id="kc_a",
        generation_id="gen_target",
    ) == 0
    assert await index.count_generation(
        embedding_model="m",
        collection_id="kc_a",
        generation_id="gen_other",
    ) == 1
    assert await index.count_generation(
        embedding_model="m",
        collection_id="kc_b",
        generation_id="gen_target",
    ) == 1


@pytest.mark.asyncio
async def test_generation_document_cleanup_removes_unknown_points_only_in_scope(
) -> None:
    index = _index()
    await index.ensure_model(embedding_model="m", embedding_dim=2)
    for document_id, chunk_id, generation_id in (
        ("kd_retry", "kch_canonical", "gen_shadow"),
        ("kd_retry", "kch_unknown_orphan", "gen_shadow"),
        ("kd_other", "kch_other_document", "gen_shadow"),
        ("kd_retry", "kch_active", "gen_active"),
    ):
        await index.upsert(
            embedding_model="m",
            collection_id="kc_a",
            document_id=document_id,
            vectors=[
                ChunkVector(
                    chunk_id,
                    (1.0, 0.0),
                    generation_id=generation_id,
                    revision_id="rev",
                )
            ],
        )

    assert await index.count_generation_document(
        embedding_model="m",
        collection_id="kc_a",
        generation_id="gen_shadow",
        document_id="kd_retry",
    ) == 2
    await index.delete_generation_document(
        embedding_model="m",
        collection_id="kc_a",
        generation_id="gen_shadow",
        document_id="kd_retry",
    )
    assert await index.count_generation_document(
        embedding_model="m",
        collection_id="kc_a",
        generation_id="gen_shadow",
        document_id="kd_retry",
    ) == 0
    assert await index.count_generation(
        embedding_model="m",
        collection_id="kc_a",
        generation_id="gen_shadow",
    ) == 1
    assert await index.count_generation(
        embedding_model="m",
        collection_id="kc_a",
        generation_id="gen_active",
    ) == 1
