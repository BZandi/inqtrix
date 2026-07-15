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
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from inqtrix.auth.permissions import SharePermission
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

if TYPE_CHECKING:
    from inqtrix.auth.memory_authority import MemoryAuthorityCoordinator


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
        self._resource_access_guard: Callable[..., Any] | None = None
        self._authority: MemoryAuthorityCoordinator | None = None

    @property
    def atomic_resource_effects(self) -> bool:
        """Whether resource writes include audit and invalidations atomically."""
        return self._authority is not None

    def bind_authority_coordinator(
        self, coordinator: "MemoryAuthorityCoordinator"
    ) -> None:
        """Join the single process-local identity/resource boundary."""
        self._authority = coordinator
        self._lock = coordinator.lock
        self._resource_access_guard = coordinator.resource_access_guard
        coordinator.register_resource(
            "knowledge_collection", self._resource_snapshot
        )

    def _resource_snapshot(self, tenant_id: str, collection_id: str):
        """Return existence and owner while the shared lock is held."""
        from inqtrix.auth.memory_authority import MemoryResourceSnapshot

        collection = self._collections.get(collection_id)
        return MemoryResourceSnapshot(
            exists=(
                collection is not None and collection.tenant_id == tenant_id
            ),
            owner_user_id=(
                collection.created_by_user_id
                if collection is not None and collection.tenant_id == tenant_id
                else None
            ),
        )

    def bind_authorization(
        self,
        *,
        resource_access_guard: Callable[..., Any],
    ) -> None:
        """Bind the identity-store lock used by shared collection writes.

        The collection lock is acquired before this guard. That order matches
        the durable resource-then-share transaction order and turns a revoke
        racing the final in-memory write into one observable outcome.
        """
        self._resource_access_guard = resource_access_guard

    @contextmanager
    def _collection_edit_guard(
        self,
        collection: KnowledgeCollection,
        *,
        actor_user_id: uuid.UUID | None,
        denied_resource_id: str,
        denied_error: type[KeyError],
        owner_only: bool = False,
    ) -> Iterator[None]:
        """Hold live edit authority across one already-locked mutation."""
        if self._resource_access_guard is None:
            if actor_user_id == collection.created_by_user_id:
                yield
                return
            raise denied_error(denied_resource_id)

        from inqtrix.execution_authority import AuthorizationRevoked

        try:
            guard = (
                self._authority.resource_access_guard
                if self._authority is not None
                else self._resource_access_guard
            )
            assert guard is not None
            kwargs = {"owner_only": owner_only} if self._authority is not None else {}
            with guard(
                tenant_id=collection.tenant_id,
                owner_user_id=collection.created_by_user_id,
                actor_user_id=actor_user_id,
                resource_type="knowledge_collection",
                resource_id=collection.id,
                minimum=SharePermission.EDIT,
                **kwargs,
            ):
                yield
        except AuthorizationRevoked as exc:
            raise denied_error(denied_resource_id) from exc

    @property
    def supports_safe_reindex(self) -> bool:
        """In-process jobs and mutations share one serialized boundary."""
        return True

    @property
    def supports_collection_sharing(self) -> bool:
        """The in-process collection and identity stores share one process."""
        return True

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
        created_by_user_id: uuid.UUID | None = None,
    ) -> KnowledgeCollection:
        """Create a collection with its immutable embedding identity."""
        if embedding_dim <= 0:
            raise EmbeddingDimensionMismatch(
                f"embedding_dim must be positive, got {embedding_dim}"
            )
        with self._lock:
            guard = (
                self._authority.creation_guard(
                    tenant_id="default",
                    actor_user_id=created_by_user_id,
                )
                if self._authority is not None
                else nullcontext()
            )
            with guard:
                collection = KnowledgeCollection(
                    id=_new_id("kc"),
                    name=name,
                    embedding_model=embedding_model,
                    embedding_dim=embedding_dim,
                    created_at=time.time(),
                    created_by_user_id=created_by_user_id,
                )
                self._collections[collection.id] = collection
                if self._authority is not None:
                    self._authority.append_resource_effects(
                        tenant_id=collection.tenant_id,
                        actor_user_id=created_by_user_id,
                        owner_user_id=created_by_user_id,
                        action="knowledge_collection.created",
                        resource_type="knowledge_collection",
                        resource_id=collection.id,
                        scope="knowledge_collections",
                    )
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

    async def delete_collection(
        self,
        collection_id: str,
        *,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        """Delete a collection and every document/chunk inside it."""
        with self._lock:
            collection = self._collections.get(collection_id)
            if collection is None:
                raise CollectionNotFound(collection_id)
            with self._collection_edit_guard(
                collection,
                actor_user_id=actor_user_id,
                denied_resource_id=collection_id,
                denied_error=CollectionNotFound,
                owner_only=True,
            ):
                if collection.created_by_user_id != actor_user_id:
                    raise CollectionNotFound(collection_id)
                if self._authority is not None:
                    self._authority.revoke_deleted_resource(
                        tenant_id=collection.tenant_id,
                        actor_user_id=actor_user_id,
                        owner_user_id=collection.created_by_user_id,
                        action="knowledge_collection.deleted",
                        resource_type="knowledge_collection",
                        resource_id=collection.id,
                        scope="knowledge_collections",
                    )
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
        actor_user_id: uuid.UUID | None = None,
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
            with self._collection_edit_guard(
                collection,
                actor_user_id=actor_user_id,
                denied_resource_id=collection_id,
                denied_error=CollectionNotFound,
            ):
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
                        page_number=(
                            pages[index] if index < len(pages) else None
                        ),
                    )
                    for index, (chunk_text, embedding) in enumerate(
                        zip(chunks, embeddings)
                    )
                ]
                # dataclasses.replace, NOT field-by-field reconstruction:
                # a manual copy silently drops every field added later
                # (created_by_user_id was lost exactly that way).
                self._collections[collection_id] = replace(
                    collection, document_count=collection.document_count + 1
                )
                if self._authority is not None:
                    self._authority.append_resource_effects(
                        tenant_id=collection.tenant_id,
                        actor_user_id=actor_user_id,
                        owner_user_id=collection.created_by_user_id,
                        action="knowledge_document.added",
                        resource_type="knowledge_collection",
                        resource_id=collection.id,
                        scope="knowledge_collections",
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

    async def get_chunks(self, document_id: str) -> list[DocumentChunk]:
        """One document's chunks ordered by ``chunk_index`` (no vectors).

        The embedding is stripped so this read surface is byte-identical
        to the Postgres store, which never hydrates vectors here — the
        two backends must not diverge on an observable field.
        """
        with self._lock:
            if document_id not in self._documents:
                raise DocumentNotFound(document_id)
            ordered = sorted(
                self._chunks.get(document_id, []),
                key=lambda chunk: chunk.chunk_index,
            )
            return [replace(chunk, embedding=()) for chunk in ordered]

    async def delete_document(
        self,
        document_id: str,
        *,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        """Delete one document and its chunks."""
        with self._lock:
            document = self._documents.get(document_id)
            if document is None:
                raise DocumentNotFound(document_id)
            collection = self._collections.get(document.collection_id)
            if collection is None:
                raise DocumentNotFound(document_id)
            with self._collection_edit_guard(
                collection,
                actor_user_id=actor_user_id,
                denied_resource_id=document_id,
                denied_error=DocumentNotFound,
            ):
                self._documents.pop(document_id, None)
                self._chunks.pop(document_id, None)
                self._collections[collection.id] = replace(
                    collection,
                    document_count=max(0, collection.document_count - 1),
                )
                if self._authority is not None:
                    self._authority.append_resource_effects(
                        tenant_id=collection.tenant_id,
                        actor_user_id=actor_user_id,
                        owner_user_id=collection.created_by_user_id,
                        action="knowledge_document.deleted",
                        resource_type="knowledge_collection",
                        resource_id=collection.id,
                        scope="knowledge_collections",
                    )

    async def reembed_document(
        self,
        *,
        document_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_chunks: list[str] | None = None,
        page_numbers: list[int | None] | None = None,
        actor_user_id: uuid.UUID | None = None,
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
            with self._collection_edit_guard(
                collection,
                actor_user_id=actor_user_id,
                denied_resource_id=document_id,
                denied_error=DocumentNotFound,
            ):
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
                # Reuse chunk ids by position so a re-embed does not orphan
                # existing citations: (document_id, chunk_index) is the
                # citation key, but a stable chunk_id lets exact provenance
                # links survive reindex too. Positions beyond the previous
                # chunk count get fresh ids; a shrunk document drops the tail.
                previous_ids = [
                    chunk.id for chunk in self._chunks.get(document_id, [])
                ]
                self._chunks[document_id] = [
                    DocumentChunk(
                        id=(
                            previous_ids[index]
                            if index < len(previous_ids)
                            else _new_id("kch")
                        ),
                        document_id=document_id,
                        collection_id=document.collection_id,
                        chunk_index=index,
                        text=chunk_text,
                        embedding=tuple(embedding),
                        source_text=(
                            sources[index] if index < len(sources) else ""
                        ),
                        page_number=(
                            pages[index] if index < len(pages) else None
                        ),
                    )
                    for index, (chunk_text, embedding) in enumerate(
                        zip(chunks, embeddings)
                    )
                ]
                # dataclasses.replace, NOT field-by-field reconstruction:
                # a manual copy silently drops every field added later.
                updated = replace(document, chunk_count=len(chunks))
                self._documents[document_id] = updated
                if self._authority is not None:
                    self._authority.append_resource_effects(
                        tenant_id=collection.tenant_id,
                        actor_user_id=actor_user_id,
                        owner_user_id=collection.created_by_user_id,
                        action="knowledge_document.reindexed",
                        resource_type="knowledge_collection",
                        resource_id=collection.id,
                        scope="knowledge_collections",
                    )
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
