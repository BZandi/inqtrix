"""In-memory knowledge store: the zero-infrastructure default.

Same philosophy as the in-memory run store: a thread-safe, in-process
implementation that keeps every knowledge feature fully functional
without a database or vector service. Contents are lost on restart —
acceptable for the first cut and for development; durable stores land
behind the same :class:`~inqtrix.knowledge.stores.ports.KnowledgeStore`
port.

The port is async (uniform with the Postgres-backed store), but this
store does no I/O: its public coroutines wrap synchronous,
lock-guarded work over in-process dicts. The work never awaits while
holding the lock, so a ``threading.RLock`` still gives correct
cross-thread safety (the synchronous research graph and the reindex
worker reach it via ``asyncio.run`` on their own threads).

Retrieval is exact cosine similarity over all candidate chunks. That
is O(chunks) per query, which is the honest and correct choice at the
document volumes an in-process store holds; approximate indexes are a
property of the later vector-store backends, not of this port.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import replace
from typing import Any

from inqtrix.pagination import keyset_page
from inqtrix.knowledge.stores.ports import (
    CollectionNotFound,
    DocumentChunk,
    DocumentNotFound,
    EmbeddingDimensionMismatch,
    KnowledgeCollection,
    KnowledgeDocument,
    RetrievalCandidate,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def _cosine(a: tuple[float, ...], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryKnowledgeStore:
    """Thread-safe in-process implementation of the knowledge store port."""

    def __init__(self) -> None:
        self._collections: dict[str, KnowledgeCollection] = {}
        self._documents: dict[str, KnowledgeDocument] = {}
        self._chunks: dict[str, list[DocumentChunk]] = {}
        self._lock = threading.RLock()

    async def is_available(self) -> bool:
        """In-memory knowledge is available whenever the process is alive."""
        return True

    def _get_collection(self, collection_id: str) -> KnowledgeCollection:
        """Locked lookup shared by the public coroutine and internal callers."""
        with self._lock:
            collection = self._collections.get(collection_id)
            if collection is None:
                raise CollectionNotFound(collection_id)
            return collection

    # -- collections ------------------------------------------------------ #

    async def create_collection(
        self,
        *,
        name: str,
        embedding_model: str,
        embedding_dim: int,
        created_by_sub: str | None = None,
    ) -> KnowledgeCollection:
        """Create a collection with its immutable embedding identity."""
        if embedding_dim <= 0:
            raise EmbeddingDimensionMismatch(
                f"embedding_dim must be positive, got {embedding_dim}"
            )
        with self._lock:
            collection = KnowledgeCollection(
                id=_new_id("kc"),
                name=name,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                created_at=time.time(),
                created_by_sub=created_by_sub,
            )
            self._collections[collection.id] = collection
            return collection

    async def list_collections(self) -> list[KnowledgeCollection]:
        """Return all collections, newest first."""
        with self._lock:
            return sorted(
                self._collections.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )

    async def get_collection(self, collection_id: str) -> KnowledgeCollection:
        """Return one collection or raise :class:`CollectionNotFound`."""
        return self._get_collection(collection_id)

    async def delete_collection(self, collection_id: str) -> None:
        """Delete a collection and every document/chunk inside it."""
        with self._lock:
            if collection_id not in self._collections:
                raise CollectionNotFound(collection_id)
            del self._collections[collection_id]
            doomed = [
                document_id
                for document_id, document in self._documents.items()
                if document.collection_id == collection_id
            ]
            for document_id in doomed:
                self._documents.pop(document_id, None)
                self._chunks.pop(document_id, None)

    # -- documents --------------------------------------------------------- #

    async def add_document(
        self,
        *,
        collection_id: str,
        title: str,
        text: str,
        metadata: dict[str, Any],
        chunks: list[str],
        embeddings: list[list[float]],
        source_chunks: list[str] | None = None,
        page_numbers: list[int | None] | None = None,
    ) -> KnowledgeDocument:
        """Store a document with its pre-chunked, pre-embedded content.

        Raises:
            CollectionNotFound: Unknown *collection_id*.
            EmbeddingDimensionMismatch: When any embedding's dimension
                contradicts the collection, or the chunk/embedding
                counts differ (both indicate a broken ingestion call,
                never something to reconcile silently).
        """
        if len(chunks) != len(embeddings):
            raise EmbeddingDimensionMismatch(
                f"chunk/embedding count mismatch: {len(chunks)} chunks vs "
                f"{len(embeddings)} embeddings"
            )
        with self._lock:
            collection = self._get_collection(collection_id)
            for index, embedding in enumerate(embeddings):
                if len(embedding) != collection.embedding_dim:
                    raise EmbeddingDimensionMismatch(
                        f"chunk {index} has dimension {len(embedding)}, "
                        f"collection {collection_id} requires "
                        f"{collection.embedding_dim} "
                        f"(model {collection.embedding_model})"
                    )
            document = KnowledgeDocument(
                id=_new_id("kd"),
                collection_id=collection_id,
                title=title,
                text=text,
                metadata=dict(metadata),
                chunk_count=len(chunks),
                created_at=time.time(),
            )
            self._documents[document.id] = document
            sources = source_chunks or []
            pages = page_numbers or []
            self._chunks[document.id] = [
                DocumentChunk(
                    id=_new_id("kch"),
                    document_id=document.id,
                    collection_id=collection_id,
                    chunk_index=index,
                    text=chunk_text,
                    embedding=tuple(embedding),
                    source_text=(
                        sources[index] if index < len(sources) else ""
                    ),
                    page_number=pages[index] if index < len(pages) else None,
                )
                for index, (chunk_text, embedding) in enumerate(
                    zip(chunks, embeddings)
                )
            ]
            # dataclasses.replace, NOT field-by-field reconstruction:
            # a manual copy silently drops every field added later
            # (created_by_sub was lost exactly that way).
            self._collections[collection_id] = replace(
                collection, document_count=collection.document_count + 1
            )
            return document

    async def list_documents(self, collection_id: str) -> list[KnowledgeDocument]:
        """Return a collection's documents, newest first."""
        with self._lock:
            self._get_collection(collection_id)
            return sorted(
                (
                    document
                    for document in self._documents.values()
                    if document.collection_id == collection_id
                ),
                key=lambda item: item.created_at,
                reverse=True,
            )

    async def list_documents_page(
        self,
        collection_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[KnowledgeDocument], str | None]:
        """One keyset page (newest first, ``(created_at, id)`` tiebreak)."""
        with self._lock:
            self._get_collection(collection_id)
            ordered = sorted(
                (
                    document
                    for document in self._documents.values()
                    if document.collection_id == collection_id
                ),
                key=lambda item: (item.created_at, item.id),
                reverse=True,
            )
        return keyset_page(
            ordered,
            limit=limit,
            after=after,
            created_at_of=lambda d: d.created_at,
            id_of=lambda d: d.id,
        )

    async def get_document(self, document_id: str) -> KnowledgeDocument:
        """Return one document or raise :class:`DocumentNotFound`."""
        with self._lock:
            document = self._documents.get(document_id)
            if document is None:
                raise DocumentNotFound(document_id)
            return document

    async def delete_document(self, document_id: str) -> None:
        """Delete one document and its chunks."""
        with self._lock:
            document = self._documents.pop(document_id, None)
            if document is None:
                raise DocumentNotFound(document_id)
            self._chunks.pop(document_id, None)
            collection = self._collections.get(document.collection_id)
            if collection is not None:
                self._collections[collection.id] = replace(
                    collection,
                    document_count=max(0, collection.document_count - 1),
                )

    async def reembed_document(
        self,
        *,
        document_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_chunks: list[str] | None = None,
        page_numbers: list[int | None] | None = None,
    ) -> KnowledgeDocument:
        """Rebuild one document's chunks/vectors in place (keep its id).

        Raises:
            DocumentNotFound: Unknown *document_id*.
            EmbeddingDimensionMismatch: When chunk/embedding counts
                differ or any vector contradicts the collection's
                dimension (both indicate a broken re-embed call, never
                something to reconcile silently).
        """
        if len(chunks) != len(embeddings):
            raise EmbeddingDimensionMismatch(
                f"chunk/embedding count mismatch: {len(chunks)} chunks vs "
                f"{len(embeddings)} embeddings"
            )
        with self._lock:
            document = self._documents.get(document_id)
            if document is None:
                raise DocumentNotFound(document_id)
            collection = self._get_collection(document.collection_id)
            for index, embedding in enumerate(embeddings):
                if len(embedding) != collection.embedding_dim:
                    raise EmbeddingDimensionMismatch(
                        f"chunk {index} has dimension {len(embedding)}, "
                        f"collection {document.collection_id} requires "
                        f"{collection.embedding_dim} "
                        f"(model {collection.embedding_model})"
                    )
            sources = source_chunks or []
            pages = page_numbers or []
            self._chunks[document_id] = [
                DocumentChunk(
                    id=_new_id("kch"),
                    document_id=document_id,
                    collection_id=document.collection_id,
                    chunk_index=index,
                    text=chunk_text,
                    embedding=tuple(embedding),
                    source_text=(
                        sources[index] if index < len(sources) else ""
                    ),
                    page_number=pages[index] if index < len(pages) else None,
                )
                for index, (chunk_text, embedding) in enumerate(
                    zip(chunks, embeddings)
                )
            ]
            # dataclasses.replace, NOT field-by-field reconstruction:
            # a manual copy silently drops every field added later.
            updated = replace(document, chunk_count=len(chunks))
            self._documents[document_id] = updated
            return updated

    # -- retrieval ----------------------------------------------------------- #

    async def search(
        self,
        *,
        query_embedding: list[float],
        collection_ids: list[str] | None,
        top_k: int,
        embedding_model: str | None = None,
    ) -> list[RetrievalCandidate]:
        """Exact cosine search over the scoped collections.

        Args:
            query_embedding: The embedded query vector.
            collection_ids: Collections to search; ``None`` searches
                every collection WHOSE DIMENSION MATCHES the query
                vector (collections embedded with a different model
                are skipped — comparing vectors across models would be
                meaningless, and an explicit id with a mismatching
                dimension raises instead).
            top_k: Maximum number of candidates to return.

        Raises:
            CollectionNotFound: An explicitly requested collection id
                is unknown.
            EmbeddingDimensionMismatch: An explicitly requested
                collection has a different embedding dimension than
                the query vector.
        """
        with self._lock:
            if collection_ids is None:
                scoped = [
                    collection
                    for collection in self._collections.values()
                    if collection.embedding_dim == len(query_embedding)
                ]
            else:
                scoped = []
                for collection_id in collection_ids:
                    collection = self._get_collection(collection_id)
                    if collection.embedding_dim != len(query_embedding):
                        raise EmbeddingDimensionMismatch(
                            f"query embedding has dimension "
                            f"{len(query_embedding)}, collection "
                            f"{collection_id} requires {collection.embedding_dim}"
                        )
                    scoped.append(collection)
            scoped_ids = {collection.id for collection in scoped}

            candidates: list[RetrievalCandidate] = []
            for document_id, chunks in self._chunks.items():
                document = self._documents.get(document_id)
                if document is None or document.collection_id not in scoped_ids:
                    continue
                for chunk in chunks:
                    candidates.append(
                        RetrievalCandidate(
                            chunk=chunk,
                            score=_cosine(chunk.embedding, query_embedding),
                            document_title=document.title,
                        )
                    )
            candidates.sort(key=lambda item: item.score, reverse=True)
            return candidates[: max(0, top_k)]
