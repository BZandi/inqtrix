"""Vector-index port: chunk vectors only, keyed by chunk id.

The Postgres-canonical knowledge store keeps collections/documents/chunks
relationally (the source of truth) and delegates ONLY the embedding
vectors to a :class:`VectorIndex`. This is the standard split — Postgres
is the canonical store, the vector DB is a derived index holding vectors
plus a lean payload (just the keys needed to filter and to join back to
the canonical rows).

Two implementations:

* :class:`MemoryVectorIndex` — in-process dense cosine, zero
  infrastructure (dev/test, and the ``postgres`` storage tier without a
  Qdrant service). Lost on restart; rebuildable from the canonical text
  via reindex. Dense-only (no hybrid).
* ``QdrantVectorIndex`` (in :mod:`inqtrix.knowledge.stores.qdrant_store`)
  — dense + optional BM25 sparse, persistent.

The port is async to match the knowledge store; the Qdrant client calls
run off the event loop via ``asyncio.to_thread``.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ChunkVector:
    """One chunk's dense vector plus the text the sparse branch needs.

    Attributes:
        chunk_id: Canonical chunk id (``kch_...``) — the join key back to
            the Postgres ``knowledge_chunks`` row.
        dense: The dense embedding for the chunk.
        text: The embedded chunk text; a hybrid index recomputes its BM25
            sparse vector from this. Dense-only indexes ignore it.
    """

    chunk_id: str
    dense: tuple[float, ...]
    text: str = field(default="", repr=False)


@dataclass(frozen=True)
class VectorHit:
    """One scored vector match (the chunk id + similarity score).

    The knowledge store hydrates the chunk text/source_text and document
    title from Postgres for these ids — the vector index never carries
    that content.
    """

    chunk_id: str
    score: float


def _cosine(a: tuple[float, ...], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@runtime_checkable
class VectorIndex(Protocol):
    """Dense (+ optional sparse) vector store keyed by canonical chunk id."""

    @property
    def supports_hybrid(self) -> bool:
        """Whether a lexical (BM25) branch is configured for hybrid search."""
        ...

    async def is_available(self) -> bool:
        """Return whether the backing vector service is reachable now."""
        ...

    async def ensure_model(
        self, *, embedding_model: str, embedding_dim: int
    ) -> None:
        """Prepare the per-model index space (no-op if already present)."""
        ...

    async def upsert(
        self,
        *,
        embedding_model: str,
        collection_id: str,
        document_id: str,
        vectors: list[ChunkVector],
    ) -> None:
        """Insert/replace the vectors for one document's chunks."""
        ...

    async def delete_document(
        self, *, embedding_model: str, document_id: str
    ) -> None:
        """Remove all chunk vectors of one document."""
        ...

    async def delete_collection(
        self, *, embedding_model: str, collection_id: str
    ) -> None:
        """Remove all chunk vectors of one logical collection."""
        ...

    async def search(
        self,
        *,
        embedding_model: str,
        query_embedding: list[float],
        collection_ids: list[str],
        top_k: int,
    ) -> list[VectorHit]:
        """Dense nearest-neighbour search scoped to *collection_ids*."""
        ...

    async def hybrid_search(
        self,
        *,
        embedding_model: str,
        query_text: str,
        query_embedding: list[float],
        collection_ids: list[str],
        top_k: int,
    ) -> list[VectorHit]:
        """Fused dense + sparse search (only when ``supports_hybrid``)."""
        ...

    async def scroll_chunk_groups(
        self, *, embedding_model: str
    ) -> set[tuple[str, str]]:
        """Distinct ``(collection_id, document_id)`` groups present in the
        model's vector space. The reverse-reconcile input: the canonical store
        diffs this against its Postgres rows to find vectors whose document was
        deleted (cross-store deletes are non-atomic, so a crash can strand
        them)."""
        ...


@dataclass
class _MemoryEntry:
    dense: tuple[float, ...]
    collection_id: str
    document_id: str
    embedding_model: str


class MemoryVectorIndex:
    """In-process dense cosine vector index (zero infrastructure).

    Vectors live in a thread-safe dict keyed by chunk id; search is exact
    O(vectors) cosine over the scoped collections, mirroring the
    in-memory knowledge store's honesty about scale. Lost on restart and
    rebuildable from the canonical text via reindex. Dense-only — it does
    not advertise the hybrid capability.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _MemoryEntry] = {}
        self._lock = threading.RLock()

    @property
    def supports_hybrid(self) -> bool:
        return False

    async def is_available(self) -> bool:
        """In-memory vectors are available whenever the process is alive."""
        return True

    async def ensure_model(
        self, *, embedding_model: str, embedding_dim: int
    ) -> None:
        # Nothing to prepare for an in-process dict.
        return None

    async def upsert(
        self,
        *,
        embedding_model: str,
        collection_id: str,
        document_id: str,
        vectors: list[ChunkVector],
    ) -> None:
        with self._lock:
            for vector in vectors:
                self._entries[vector.chunk_id] = _MemoryEntry(
                    dense=tuple(vector.dense),
                    collection_id=collection_id,
                    document_id=document_id,
                    embedding_model=embedding_model,
                )

    async def delete_document(
        self, *, embedding_model: str, document_id: str
    ) -> None:
        with self._lock:
            doomed = [
                chunk_id
                for chunk_id, entry in self._entries.items()
                if entry.document_id == document_id
            ]
            for chunk_id in doomed:
                del self._entries[chunk_id]

    async def delete_collection(
        self, *, embedding_model: str, collection_id: str
    ) -> None:
        with self._lock:
            doomed = [
                chunk_id
                for chunk_id, entry in self._entries.items()
                if entry.collection_id == collection_id
            ]
            for chunk_id in doomed:
                del self._entries[chunk_id]

    async def search(
        self,
        *,
        embedding_model: str,
        query_embedding: list[float],
        collection_ids: list[str],
        top_k: int,
    ) -> list[VectorHit]:
        scoped = set(collection_ids)
        with self._lock:
            hits = [
                VectorHit(
                    chunk_id=chunk_id,
                    score=_cosine(entry.dense, query_embedding),
                )
                for chunk_id, entry in self._entries.items()
                if entry.embedding_model == embedding_model
                and entry.collection_id in scoped
            ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(0, top_k)]

    async def hybrid_search(
        self,
        *,
        embedding_model: str,
        query_text: str,
        query_embedding: list[float],
        collection_ids: list[str],
        top_k: int,
    ) -> list[VectorHit]:
        # Dense-only index: callers dispatch on ``supports_hybrid`` and
        # never reach this. Failing loud beats a silent dense fallback.
        raise NotImplementedError(
            "MemoryVectorIndex is dense-only (supports_hybrid is False)"
        )

    async def scroll_chunk_groups(
        self, *, embedding_model: str
    ) -> set[tuple[str, str]]:
        with self._lock:
            return {
                (entry.collection_id, entry.document_id)
                for entry in self._entries.values()
                if entry.embedding_model == embedding_model
            }
