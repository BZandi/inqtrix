"""Qdrant-backed knowledge: a vector index, plus a legacy sole-store.

Two consumers of one Qdrant topology:

* :class:`QdrantVectorIndex` — the vector half of the Postgres-canonical
  store (the production full-stack path). It holds ONLY vectors plus a
  lean payload (``collection_id``/``document_id``/``chunk_id`` — the keys
  to filter and to join back to Postgres), keyed by the canonical chunk
  id. No document text lives in Qdrant.
* :class:`QdrantKnowledgeStore` — the legacy sole-store for the rare
  ``qdrant`` vector backend WITHOUT Postgres. It keeps a small registry
  collection for the canonical records (text in payload) and reuses the
  same chunk topology. Retained for backward compatibility; the
  full-stack path uses ``QdrantVectorIndex`` behind the Postgres store.

Topology (Qdrant's multi-tenant guidance): ONE physical Qdrant
collection per embedding-model configuration, with the logical
collection as an indexed payload field (``is_tenant`` partitioning).
Hybrid retrieval fuses a dense branch with a client-side BM25 sparse
branch (fastembed; a tokenizer + Snowball stemmer ALGORITHM, German,
paired with Qdrant's IDF modifier) via reciprocal rank fusion.

The port is async; the synchronous ``qdrant_client`` calls run off the
event loop via ``asyncio.to_thread`` (proven sync bodies wrapped, not
rewritten). Security note: payload filters are a performance boundary,
not the authorization truth — access stays with the AuthorizationService.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any

from inqtrix.pagination import keyset_page
from inqtrix.knowledge.stores.ports import (
    CollectionNotFound,
    DocumentChunk,
    DocumentNotFound,
    EmbeddingDimensionMismatch,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeError,
    RetrievalCandidate,
)
from inqtrix.knowledge.stores.vector_index import ChunkVector, VectorHit

log = logging.getLogger("inqtrix")

REGISTRY_COLLECTION = "inqtrix_registry"
CHUNKS_PREFIX = "inqtrix_chunks__"

_IMPORT_HINT = (
    "Qdrant support requires the 'knowledge-qdrant' extra "
    "(uv sync --extra knowledge-qdrant)."
)


def _model_slug(embedding_model: str) -> str:
    return CHUNKS_PREFIX + "".join(
        ch if ch.isalnum() else "_" for ch in embedding_model.lower()
    )


def _optional_int(value: Any) -> int | None:
    """Coerce a payload value to ``int`` or ``None`` (best-effort page number)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _point_uuid(stable_id: str) -> str:
    """Deterministic Qdrant point id for a stable record/chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"inqtrix://{stable_id}"))


def _payload_user_id(value: Any) -> uuid.UUID | None:
    """Restore a canonical user UUID from Qdrant's JSON payload boundary."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _require_qdrant():
    """Import the qdrant client + models, or raise the install hint."""
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(_IMPORT_HINT) from exc
    return QdrantClient, models


# The lexical (BM25) branch is monolingual. Its tokenizer language and the
# ISO-639-1 code are the SAME fact in two representations (fastembed wants the
# English name "german"; capability manifests and the algorithm compare against
# detect_ui_language's "de"). Defined ONCE here so the encoder init and the
# `sparse_language` property can never drift apart (Designprinzip 4).
_BM25_TOKENIZER_LANGUAGE = "german"
_BM25_LANGUAGE_CODE = "de"


class _Bm25:
    """Lazy client-side BM25 sparse encoder (German), thread-safe init."""

    def __init__(self) -> None:
        self._encoder = None
        self._lock = threading.Lock()

    def _ensure(self):
        # Double-checked init: the encoder is reached from many
        # to_thread worker threads concurrently; the lock prevents a
        # wasteful double-construction on first use.
        if self._encoder is None:
            with self._lock:
                if self._encoder is None:
                    try:
                        from fastembed import SparseTextEmbedding
                    except ImportError as exc:  # pragma: no cover - env-dependent
                        raise RuntimeError(_IMPORT_HINT) from exc
                    self._encoder = SparseTextEmbedding(
                        model_name="Qdrant/bm25",
                        language=_BM25_TOKENIZER_LANGUAGE,
                    )
        return self._encoder

    def documents(self, texts: list[str]):
        return list(self._ensure().embed(texts))

    def query(self, text: str):
        return list(self._ensure().query_embed(text))[0]


def _ensure_chunks_collection(
    client, models, *, name: str, embedding_dim: int, sparse_enabled: bool
) -> None:
    """Create the per-model chunk collection if absent (idempotent)."""
    if client.collection_exists(name):
        return
    sparse_config = (
        {"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
        if sparse_enabled
        else None
    )
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=embedding_dim, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config=sparse_config,
    )
    # Logical-collection partitioning index (is_tenant pattern).
    client.create_payload_index(
        collection_name=name,
        field_name="collection_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, is_tenant=True
        ),
    )
    client.create_payload_index(
        collection_name=name,
        field_name="document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _scope_filter(models, collection_ids: list[str]):
    return models.Filter(
        must=[
            models.FieldCondition(
                key="collection_id",
                match=models.MatchAny(any=list(collection_ids)),
            )
        ]
    )


def _hits_from_points(points) -> list[VectorHit]:
    hits: list[VectorHit] = []
    for point in points:
        payload = dict(point.payload or {})
        chunk_id = payload.get("chunk_id")
        if not chunk_id:
            continue
        hits.append(VectorHit(chunk_id=str(chunk_id), score=float(point.score or 0.0)))
    return hits


class QdrantVectorIndex:
    """Vector-only Qdrant index (lean payload, keyed by canonical chunk id).

    Implements :class:`~inqtrix.knowledge.stores.vector_index.VectorIndex`.
    The Postgres-canonical store owns documents/chunks; this index owns
    only the dense (+ optional BM25 sparse) vectors and the keys needed to
    filter and to join back.

    Args:
        url: Qdrant REST endpoint.
        api_key: Qdrant API key (empty accepted for loopback dev, logged).
        sparse: ``"bm25_german"`` enables hybrid; ``"off"`` is dense-only.
        timeout: Per-call timeout in seconds.
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str = "",
        sparse: str = "bm25_german",
        timeout: float = 30.0,
    ) -> None:
        QdrantClient, _models = _require_qdrant()
        if not api_key:
            log.warning(
                "QdrantVectorIndex ohne API-Key — nur fuer Loopback-Dev "
                "akzeptabel (self-hosted Qdrant ist standardmaessig ohne Auth)."
            )
        if sparse not in ("bm25_german", "off"):
            raise ValueError(f"unknown sparse mode: {sparse!r}")
        self._client = QdrantClient(url=url, api_key=api_key or None, timeout=timeout)
        self._sparse_enabled = sparse == "bm25_german"
        self._bm25 = _Bm25()

    @property
    def supports_hybrid(self) -> bool:
        return self._sparse_enabled

    @property
    def sparse_language(self) -> str | None:
        """ISO 639-1 code of the BM25 tokenizer language, ``None`` when off.

        The lexical branch tokenizes/stems in exactly one language ("de"
        today, hardcoded ``language="german"``). Read-only: surfaced so the
        capability manifest and the knowledge algorithm can make the
        monolingual limitation of keyword retrieval visible — BM25 is
        language-bound and never cross-lingual. Optional by convention;
        consumers read it via ``getattr(store, "sparse_language", None)``.
        """
        return _BM25_LANGUAGE_CODE if self._sparse_enabled else None

    async def is_available(self) -> bool:
        """Return whether the Qdrant endpoint responds to a read-only call."""
        try:
            await asyncio.to_thread(self._client.get_collections)
            return True
        except Exception:
            return False

    async def ensure_model(
        self, *, embedding_model: str, embedding_dim: int
    ) -> None:
        _client, models = _require_qdrant()
        await asyncio.to_thread(
            _ensure_chunks_collection,
            self._client,
            models,
            name=_model_slug(embedding_model),
            embedding_dim=embedding_dim,
            sparse_enabled=self._sparse_enabled,
        )

    async def upsert(
        self,
        *,
        embedding_model: str,
        collection_id: str,
        document_id: str,
        vectors: list[ChunkVector],
    ) -> None:
        if not vectors:
            return
        await asyncio.to_thread(
            self._sync_upsert, embedding_model, collection_id, document_id, vectors
        )

    def _sync_upsert(
        self,
        embedding_model: str,
        collection_id: str,
        document_id: str,
        vectors: list[ChunkVector],
    ) -> None:
        _client, models = _require_qdrant()
        name = _model_slug(embedding_model)
        sparse_vectors = (
            self._bm25.documents([v.text for v in vectors])
            if self._sparse_enabled
            else None
        )
        points = []
        for index, chunk in enumerate(vectors):
            vector: dict[str, Any] = {"dense": list(chunk.dense)}
            if sparse_vectors is not None:
                sparse = sparse_vectors[index]
                vector["sparse"] = models.SparseVector(
                    indices=list(sparse.indices), values=list(sparse.values)
                )
            points.append(
                models.PointStruct(
                    id=_point_uuid(chunk.chunk_id),
                    vector=vector,
                    payload={
                        "collection_id": collection_id,
                        "document_id": document_id,
                        "chunk_id": chunk.chunk_id,
                    },
                )
            )
        self._client.upsert(collection_name=name, points=points, wait=True)

    async def delete_document(
        self, *, embedding_model: str, document_id: str
    ) -> None:
        await asyncio.to_thread(
            self._sync_delete_by, embedding_model, "document_id", document_id
        )

    async def delete_collection(
        self, *, embedding_model: str, collection_id: str
    ) -> None:
        await asyncio.to_thread(
            self._sync_delete_by, embedding_model, "collection_id", collection_id
        )

    def _sync_delete_by(self, embedding_model: str, key: str, value: str) -> None:
        _client, models = _require_qdrant()
        name = _model_slug(embedding_model)
        if not self._client.collection_exists(name):
            return
        self._client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key, match=models.MatchValue(value=value)
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def search(
        self,
        *,
        embedding_model: str,
        query_embedding: list[float],
        collection_ids: list[str],
        top_k: int,
    ) -> list[VectorHit]:
        return await asyncio.to_thread(
            self._sync_search, embedding_model, query_embedding, collection_ids, top_k
        )

    def _sync_search(
        self,
        embedding_model: str,
        query_embedding: list[float],
        collection_ids: list[str],
        top_k: int,
    ) -> list[VectorHit]:
        _client, models = _require_qdrant()
        name = _model_slug(embedding_model)
        if not self._client.collection_exists(name):
            return []
        result = self._client.query_points(
            collection_name=name,
            query=query_embedding,
            using="dense",
            query_filter=_scope_filter(models, collection_ids),
            limit=top_k,
            with_payload=True,
        )
        return _hits_from_points(result.points)

    async def hybrid_search(
        self,
        *,
        embedding_model: str,
        query_text: str,
        query_embedding: list[float],
        collection_ids: list[str],
        top_k: int,
    ) -> list[VectorHit]:
        if not self._sparse_enabled:
            raise RuntimeError(
                "hybrid_search requires sparse retrieval "
                "(INQTRIX_KNOWLEDGE_SPARSE=bm25_german)"
            )
        return await asyncio.to_thread(
            self._sync_hybrid_search,
            embedding_model,
            query_text,
            query_embedding,
            collection_ids,
            top_k,
        )

    def _sync_hybrid_search(
        self,
        embedding_model: str,
        query_text: str,
        query_embedding: list[float],
        collection_ids: list[str],
        top_k: int,
    ) -> list[VectorHit]:
        _client, models = _require_qdrant()
        name = _model_slug(embedding_model)
        if not self._client.collection_exists(name):
            return []
        scope = _scope_filter(models, collection_ids)
        sparse = self._bm25.query(query_text)
        prefetch_depth = max(top_k * 4, 20)
        result = self._client.query_points(
            collection_name=name,
            prefetch=[
                models.Prefetch(
                    query=query_embedding,
                    using="dense",
                    filter=scope,
                    limit=prefetch_depth,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(sparse.indices), values=list(sparse.values)
                    ),
                    using="sparse",
                    filter=scope,
                    limit=prefetch_depth,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return _hits_from_points(result.points)

    async def scroll_chunk_groups(
        self, *, embedding_model: str
    ) -> set[tuple[str, str]]:
        return await asyncio.to_thread(self._sync_scroll_groups, embedding_model)

    def _sync_scroll_groups(self, embedding_model: str) -> set[tuple[str, str]]:
        name = _model_slug(embedding_model)
        if not self._client.collection_exists(name):
            return set()
        groups: set[tuple[str, str]] = set()
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                collection_id = payload.get("collection_id")
                document_id = payload.get("document_id")
                if collection_id and document_id:
                    groups.add((str(collection_id), str(document_id)))
            if offset is None:
                break
        return groups


class QdrantKnowledgeStore:
    """Legacy sole-store: canonical registry + chunk vectors in Qdrant.

    Implements the async ``KnowledgeStore`` port for the ``qdrant`` vector
    backend WITHOUT Postgres. The full-stack path uses the Postgres store
    with :class:`QdrantVectorIndex` instead. The proven synchronous bodies
    are wrapped with ``asyncio.to_thread`` so the now-async port never
    blocks the event loop.
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str = "",
        sparse: str = "bm25_german",
        timeout: float = 30.0,
    ) -> None:
        QdrantClient, _models = _require_qdrant()
        if not api_key:
            log.warning(
                "QdrantKnowledgeStore ohne API-Key — nur fuer Loopback-Dev "
                "akzeptabel (self-hosted Qdrant ist standardmaessig ohne Auth)."
            )
        if sparse not in ("bm25_german", "off"):
            raise ValueError(f"unknown sparse mode: {sparse!r}")
        self._client = QdrantClient(url=url, api_key=api_key or None, timeout=timeout)
        self._sparse_enabled = sparse == "bm25_german"
        self._bm25 = _Bm25()
        self._registry_ready = False

    @property
    def supports_safe_reindex(self) -> bool:
        """Legacy Qdrant-only storage has no cross-process mutation fence."""
        return False

    @property
    def supports_collection_sharing(self) -> bool:
        """Vector-only metadata cannot form an atomic sharing boundary."""
        return False

    @property
    def supports_hybrid(self) -> bool:
        return self._sparse_enabled

    @property
    def sparse_language(self) -> str | None:
        """ISO 639-1 code of the BM25 tokenizer language, ``None`` when off.

        Same contract as :attr:`QdrantVectorIndex.sparse_language`: the lexical
        branch is monolingual ("de" today). Read-only, consumed via ``getattr``.
        """
        return _BM25_LANGUAGE_CODE if self._sparse_enabled else None

    async def is_available(self) -> bool:
        """Return whether the Qdrant endpoint responds to a read-only call."""
        try:
            await asyncio.to_thread(self._client.get_collections)
            return True
        except Exception:
            return False

    # -- async port (thin wrappers over the proven sync bodies) ---------- #

    async def create_collection(
        self,
        *,
        name,
        embedding_model,
        embedding_dim,
        created_by_user_id: uuid.UUID | None = None,
    ) -> KnowledgeCollection:
        return await asyncio.to_thread(
            self._sync_create_collection,
            name,
            embedding_model,
            embedding_dim,
            created_by_user_id,
        )

    async def list_collections(self) -> list[KnowledgeCollection]:
        return await asyncio.to_thread(self._sync_list_collections)

    async def get_collection(self, collection_id: str) -> KnowledgeCollection:
        return await asyncio.to_thread(self._sync_get_collection, collection_id)

    async def delete_collection(
        self,
        collection_id: str,
        *,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        await asyncio.to_thread(self._sync_delete_collection, collection_id)

    async def add_document(
        self,
        *,
        collection_id,
        title,
        text,
        metadata,
        chunks,
        embeddings,
        source_chunks=None,
        page_numbers=None,
        actor_user_id: uuid.UUID | None = None,
    ) -> KnowledgeDocument:
        return await asyncio.to_thread(
            self._sync_add_document,
            collection_id,
            title,
            text,
            metadata,
            chunks,
            embeddings,
            source_chunks,
            page_numbers,
        )

    async def list_documents(self, collection_id: str) -> list[KnowledgeDocument]:
        return await asyncio.to_thread(self._sync_list_documents, collection_id)

    async def list_documents_page(
        self,
        collection_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[KnowledgeDocument], str | None]:
        """One keyset page (newest first, ``(created_at, id)`` tiebreak).

        Qdrant is the lean vector tier; document listing is not the
        efficiency-critical path, so this slices the full list in memory to
        keep byte-identical paging with the other tiers."""
        documents = await self.list_documents(collection_id)
        documents.sort(key=lambda d: (d.created_at, d.id), reverse=True)
        return keyset_page(
            documents,
            limit=limit,
            after=after,
            created_at_of=lambda d: d.created_at,
            id_of=lambda d: d.id,
        )

    async def get_document(self, document_id: str) -> KnowledgeDocument:
        return await asyncio.to_thread(self._sync_get_document, document_id)

    async def delete_document(
        self,
        document_id: str,
        *,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        await asyncio.to_thread(self._sync_delete_document, document_id)

    async def reembed_document(
        self,
        *,
        document_id,
        chunks,
        embeddings,
        source_chunks=None,
        page_numbers=None,
        actor_user_id: uuid.UUID | None = None,
    ) -> KnowledgeDocument:
        return await asyncio.to_thread(
            self._sync_reembed_document,
            document_id,
            chunks,
            embeddings,
            source_chunks,
            page_numbers,
        )

    async def search(
        self, *, query_embedding, collection_ids, top_k, embedding_model=None
    ) -> list[RetrievalCandidate]:
        return await asyncio.to_thread(
            self._sync_search, query_embedding, collection_ids, top_k, embedding_model
        )

    async def hybrid_search(
        self,
        *,
        query_text,
        query_embedding,
        collection_ids,
        top_k,
        embedding_model=None,
    ) -> list[RetrievalCandidate]:
        if not self._sparse_enabled:
            raise RuntimeError(
                "hybrid_search requires sparse retrieval "
                "(INQTRIX_KNOWLEDGE_SPARSE=bm25_german)"
            )
        return await asyncio.to_thread(
            self._sync_hybrid_search,
            query_text,
            query_embedding,
            collection_ids,
            top_k,
            embedding_model,
        )

    # -- registry --------------------------------------------------------- #

    def _ensure_registry(self) -> None:
        if self._registry_ready:
            return
        _client, models = _require_qdrant()
        if not self._client.collection_exists(REGISTRY_COLLECTION):
            self._client.create_collection(
                collection_name=REGISTRY_COLLECTION,
                vectors_config={
                    "dummy": models.VectorParams(
                        size=1, distance=models.Distance.COSINE
                    )
                },
            )
            for field_name, schema in (
                ("kind", models.PayloadSchemaType.KEYWORD),
                ("record_id", models.PayloadSchemaType.KEYWORD),
                ("collection_id", models.PayloadSchemaType.KEYWORD),
            ):
                self._client.create_payload_index(
                    collection_name=REGISTRY_COLLECTION,
                    field_name=field_name,
                    field_schema=schema,
                )
        self._registry_ready = True

    def _registry_get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        _client, models = _require_qdrant()
        self._ensure_registry()
        points, _offset = self._client.scroll(
            collection_name=REGISTRY_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="kind", match=models.MatchValue(value=kind)),
                    models.FieldCondition(
                        key="record_id", match=models.MatchValue(value=record_id)
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return dict(points[0].payload) if points else None

    def _registry_upsert(self, payload: dict[str, Any]) -> None:
        _client, models = _require_qdrant()
        self._ensure_registry()
        self._client.upsert(
            collection_name=REGISTRY_COLLECTION,
            points=[
                models.PointStruct(
                    id=_point_uuid(payload["record_id"]),
                    vector={"dummy": [0.0]},
                    payload=payload,
                )
            ],
            wait=True,
        )

    def _registry_delete(self, record_id: str) -> None:
        self._client.delete(
            collection_name=REGISTRY_COLLECTION,
            points_selector=[_point_uuid(record_id)],
            wait=True,
        )

    def _collection_payload(self, payload: dict[str, Any]) -> KnowledgeCollection:
        return KnowledgeCollection(
            id=payload["record_id"],
            name=payload["name"],
            embedding_model=payload["embedding_model"],
            embedding_dim=payload["embedding_dim"],
            created_at=payload["created_at"],
            document_count=self._count_documents(payload["record_id"]),
            tenant_id=payload.get("tenant_id", "default"),
            created_by_user_id=_payload_user_id(
                payload.get("created_by_user_id")
            ),
        )

    def _document_payload(self, payload: dict[str, Any]) -> KnowledgeDocument:
        return KnowledgeDocument(
            id=payload["record_id"],
            collection_id=payload["collection_id"],
            title=payload["title"],
            text=payload.get("text", ""),
            metadata=dict(payload.get("metadata", {})),
            chunk_count=payload.get("chunk_count", 0),
            created_at=payload.get("created_at", 0.0),
        )

    def _count_documents(self, collection_id: str) -> int:
        _client, models = _require_qdrant()
        result = self._client.count(
            collection_name=REGISTRY_COLLECTION,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="kind", match=models.MatchValue(value="document")
                    ),
                    models.FieldCondition(
                        key="collection_id",
                        match=models.MatchValue(value=collection_id),
                    ),
                ]
            ),
            exact=True,
        )
        return int(result.count)

    # -- sync cores ------------------------------------------------------- #

    def _sync_create_collection(
        self,
        name,
        embedding_model,
        embedding_dim,
        created_by_user_id: uuid.UUID | None,
    ) -> KnowledgeCollection:
        _client, models = _require_qdrant()
        collection_id = f"kc_{uuid.uuid4().hex[:20]}"
        _ensure_chunks_collection(
            self._client,
            models,
            name=_model_slug(embedding_model),
            embedding_dim=embedding_dim,
            sparse_enabled=self._sparse_enabled,
        )
        self._registry_upsert(
            {
                "kind": "collection",
                "record_id": collection_id,
                "name": name,
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
                "created_at": time.time(),
                "tenant_id": "default",
                "created_by_user_id": created_by_user_id,
            }
        )
        return self._sync_get_collection(collection_id)

    def _sync_list_collections(self) -> list[KnowledgeCollection]:
        _client, models = _require_qdrant()
        self._ensure_registry()
        collections: list[KnowledgeCollection] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=REGISTRY_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="kind", match=models.MatchValue(value="collection")
                        )
                    ]
                ),
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            collections.extend(
                self._collection_payload(dict(point.payload)) for point in points
            )
            if offset is None:
                break
        return sorted(collections, key=lambda item: item.created_at, reverse=True)

    def _sync_get_collection(self, collection_id: str) -> KnowledgeCollection:
        payload = self._registry_get("collection", collection_id)
        if payload is None:
            raise CollectionNotFound(collection_id)
        return self._collection_payload(payload)

    def _sync_delete_collection(self, collection_id: str) -> None:
        _client, models = _require_qdrant()
        collection = self._sync_get_collection(collection_id)
        chunks_name = _model_slug(collection.embedding_model)
        if self._client.collection_exists(chunks_name):
            self._client.delete(
                collection_name=chunks_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="collection_id",
                                match=models.MatchValue(value=collection_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        self._client.delete(
            collection_name=REGISTRY_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="kind", match=models.MatchValue(value="document")
                        ),
                        models.FieldCondition(
                            key="collection_id",
                            match=models.MatchValue(value=collection_id),
                        ),
                    ]
                )
            ),
            wait=True,
        )
        self._registry_delete(collection_id)

    def _validate(self, chunks, embeddings, embedding_dim) -> None:
        if len(chunks) != len(embeddings):
            raise EmbeddingDimensionMismatch(
                f"chunk/embedding count mismatch: {len(chunks)} chunks, "
                f"{len(embeddings)} embeddings"
            )
        for vector in embeddings:
            if len(vector) != embedding_dim:
                raise EmbeddingDimensionMismatch(
                    f"embedding dimension {len(vector)} does not match "
                    f"collection dimension {embedding_dim}"
                )

    def _upsert_chunk_points(
        self, collection, document_id, title, chunks, embeddings, source_chunks,
        page_numbers=None,
    ) -> None:
        _client, models = _require_qdrant()
        chunks_name = _model_slug(collection.embedding_model)
        sparse_vectors = (
            self._bm25.documents(chunks) if self._sparse_enabled else None
        )
        points = []
        for index, (chunk_text, dense) in enumerate(zip(chunks, embeddings)):
            vector: dict[str, Any] = {"dense": dense}
            if sparse_vectors is not None:
                sparse = sparse_vectors[index]
                vector["sparse"] = models.SparseVector(
                    indices=list(sparse.indices), values=list(sparse.values)
                )
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "collection_id": collection.id,
                        "document_id": document_id,
                        "document_title": title,
                        "chunk_index": index,
                        "text": chunk_text,
                        "source_text": (
                            source_chunks[index]
                            if source_chunks and index < len(source_chunks)
                            else ""
                        ),
                        "page_number": (
                            page_numbers[index]
                            if page_numbers and index < len(page_numbers)
                            else None
                        ),
                    },
                )
            )
        if points:
            self._client.upsert(collection_name=chunks_name, points=points, wait=True)

    def _sync_add_document(
        self, collection_id, title, text, metadata, chunks, embeddings, source_chunks,
        page_numbers=None,
    ) -> KnowledgeDocument:
        collection = self._sync_get_collection(collection_id)
        self._validate(chunks, embeddings, collection.embedding_dim)
        document_id = f"kd_{uuid.uuid4().hex[:20]}"
        self._upsert_chunk_points(
            collection, document_id, title, chunks, embeddings, source_chunks,
            page_numbers,
        )
        self._registry_upsert(
            {
                "kind": "document",
                "record_id": document_id,
                "collection_id": collection_id,
                "title": title,
                "text": text,
                "metadata": dict(metadata),
                "chunk_count": len(chunks),
                "created_at": time.time(),
            }
        )
        return self._sync_get_document(document_id)

    def _sync_list_documents(self, collection_id: str) -> list[KnowledgeDocument]:
        _client, models = _require_qdrant()
        self._sync_get_collection(collection_id)
        documents: list[KnowledgeDocument] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=REGISTRY_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="kind", match=models.MatchValue(value="document")
                        ),
                        models.FieldCondition(
                            key="collection_id",
                            match=models.MatchValue(value=collection_id),
                        ),
                    ]
                ),
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            documents.extend(
                self._document_payload(dict(point.payload)) for point in points
            )
            if offset is None:
                break
        return sorted(documents, key=lambda item: item.created_at, reverse=True)

    def _sync_get_document(self, document_id: str) -> KnowledgeDocument:
        payload = self._registry_get("document", document_id)
        if payload is None:
            raise DocumentNotFound(document_id)
        return self._document_payload(payload)

    def _sync_delete_document(self, document_id: str) -> None:
        _client, models = _require_qdrant()
        document = self._sync_get_document(document_id)
        collection = self._sync_get_collection(document.collection_id)
        chunks_name = _model_slug(collection.embedding_model)
        if self._client.collection_exists(chunks_name):
            self._client.delete(
                collection_name=chunks_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        self._registry_delete(document_id)

    def _sync_reembed_document(
        self, document_id, chunks, embeddings, source_chunks, page_numbers=None
    ) -> KnowledgeDocument:
        _client, models = _require_qdrant()
        document = self._sync_get_document(document_id)
        collection = self._sync_get_collection(document.collection_id)
        self._validate(chunks, embeddings, collection.embedding_dim)
        chunks_name = _model_slug(collection.embedding_model)
        if self._client.collection_exists(chunks_name):
            self._client.delete(
                collection_name=chunks_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        self._upsert_chunk_points(
            collection, document_id, document.title, chunks, embeddings, source_chunks,
            page_numbers,
        )
        self._registry_upsert(
            {
                "kind": "document",
                "record_id": document_id,
                "collection_id": document.collection_id,
                "title": document.title,
                "text": document.text,
                "metadata": dict(document.metadata),
                "chunk_count": len(chunks),
                "created_at": document.created_at,
            }
        )
        return self._sync_get_document(document_id)

    def _resolve_target(self, collection_ids, embedding_model):
        _client, models = _require_qdrant()
        explicit = bool(collection_ids)
        if explicit:
            collections = [self._sync_get_collection(cid) for cid in collection_ids]
        else:
            collections = self._sync_list_collections()
            if not collections:
                raise KnowledgeError("no knowledge collections exist")
        models_in_scope = {c.embedding_model for c in collections}
        active_model = embedding_model or collections[0].embedding_model
        if explicit:
            # Canonical contract (Postgres + memory parity): an explicit
            # multi-model selection is a HARD error, never a silently narrowed
            # result set (Designprinzip 1). The narrowing below belongs only to
            # the implicit search-all path, where per-model narrowing is
            # required because the vector index is per embedding model.
            if len(models_in_scope) > 1:
                raise KnowledgeError(
                    "scoped collections use different embedding models "
                    f"({sorted(models_in_scope)}); query one model scope at a time"
                )
        elif embedding_model is not None and models_in_scope - {embedding_model}:
            scoped = [c for c in collections if c.embedding_model == embedding_model]
            if not scoped:
                raise KnowledgeError(
                    f"no scoped collection uses embedding model {embedding_model!r}"
                )
            collections = scoped
        return _model_slug(active_model), _scope_filter(
            models, [c.id for c in collections]
        )

    def _candidates_from_points(self, points) -> list[RetrievalCandidate]:
        candidates = []
        for point in points:
            payload = dict(point.payload or {})
            candidates.append(
                RetrievalCandidate(
                    chunk=DocumentChunk(
                        id=str(point.id),
                        document_id=payload.get("document_id", ""),
                        collection_id=payload.get("collection_id", ""),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        text=payload.get("text", ""),
                        source_text=payload.get("source_text", ""),
                        page_number=_optional_int(payload.get("page_number")),
                    ),
                    score=float(point.score or 0.0),
                    document_title=payload.get("document_title", ""),
                )
            )
        return candidates

    def _sync_search(
        self, query_embedding, collection_ids, top_k, embedding_model
    ) -> list[RetrievalCandidate]:
        chunks_name, scope_filter = self._resolve_target(collection_ids, embedding_model)
        result = self._client.query_points(
            collection_name=chunks_name,
            query=query_embedding,
            using="dense",
            query_filter=scope_filter,
            limit=top_k,
            with_payload=True,
        )
        return self._candidates_from_points(result.points)

    def _sync_hybrid_search(
        self, query_text, query_embedding, collection_ids, top_k, embedding_model
    ) -> list[RetrievalCandidate]:
        _client, models = _require_qdrant()
        chunks_name, scope_filter = self._resolve_target(collection_ids, embedding_model)
        sparse = self._bm25.query(query_text)
        prefetch_depth = max(top_k * 4, 20)
        result = self._client.query_points(
            collection_name=chunks_name,
            prefetch=[
                models.Prefetch(
                    query=query_embedding,
                    using="dense",
                    filter=scope_filter,
                    limit=prefetch_depth,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(sparse.indices), values=list(sparse.values)
                    ),
                    using="sparse",
                    filter=scope_filter,
                    limit=prefetch_depth,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return self._candidates_from_points(result.points)
