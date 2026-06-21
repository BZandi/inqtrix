"""Contracts of the knowledge engine (Baukasten ports).

The knowledge engine deliberately does NOT extend ``ProviderContext``
or ``LLMProvider``: embeddings, vector search, and document storage
are different capabilities with different lifecycles, so they get
their own ports bundled into :class:`KnowledgeProviderContext` — a
sibling of the web provider bundle, wired by the composition root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from inqtrix.knowledge.contextualize import ChunkContextualizer
    from inqtrix.providers.rerankers import RerankerProvider
    from inqtrix.providers.embeddings import EmbeddingProvider


class KnowledgeError(RuntimeError):
    """Base error for knowledge-store failures."""


class CollectionNotFound(KeyError):
    """Raised when a collection id is unknown to the store."""


class DocumentNotFound(KeyError):
    """Raised when a document id is unknown to the store."""


class EmbeddingDimensionMismatch(KnowledgeError):
    """Raised when a vector's dimension contradicts its collection.

    A collection's embedding model and dimension are immutable after
    creation; a mismatch means the wrong model produced the vector.
    Padded or truncated vectors would silently corrupt retrieval
    quality, so the mismatch is always a hard, visible failure.
    """


@dataclass(frozen=True)
class KnowledgeCollection:
    """One logical document collection with a fixed embedding model.

    Attributes:
        id: Server-assigned stable identifier (``kc_...``).
        name: Operator/user-facing label.
        embedding_model: The embedding model id every chunk in this
            collection was embedded with. Immutable after creation.
        embedding_dim: Vector dimension of ``embedding_model``. Set at
            creation (from the embedding catalog or the first embed
            call) and enforced on every subsequent upsert.
        created_at: Unix timestamp of creation.
        document_count: Number of live documents in the collection.
        tenant_id: Tenant scope carried from day one (v1 runs one
            tenant per deployment).
        created_by_sub: OIDC subject of the creator. ``None`` marks
            pre-ownership collections (and everything created by the
            anonymous/static principals) — those stay visible to every
            caller, the deliberate compatibility rule that keeps
            existing deployments working unchanged.
    """

    id: str
    name: str
    embedding_model: str
    embedding_dim: int
    created_at: float
    document_count: int = 0
    tenant_id: str = "default"
    created_by_sub: str | None = None


@dataclass(frozen=True)
class KnowledgeDocument:
    """One ingested document inside a collection.

    Attributes:
        id: Server-assigned stable identifier (``kd_...``).
        collection_id: Owning collection.
        title: Display title used in citations and references.
        text: Full extracted text as provided by the client.
        metadata: Free-form client metadata (source filename, page
            counts, tags). Stored verbatim, never interpreted.
        chunk_count: Number of chunks the document was split into.
        created_at: Unix timestamp of ingestion.
    """

    id: str
    collection_id: str
    title: str
    text: str = field(repr=False, default="")
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    created_at: float = 0.0


@dataclass(frozen=True)
class DocumentChunk:
    """One embedded retrieval unit of a document.

    Attributes:
        id: Stable chunk identifier (``kch_...``).
        document_id: Owning document.
        collection_id: Owning collection (denormalized for scoping).
        chunk_index: Zero-based position within the document.
        text: The chunk text handed to the embedding model and later
            into the answer prompt. May carry an ingestion-time
            contextualization prefix.
        embedding: Dense vector for the chunk. Dimension must equal
            the collection's ``embedding_dim``.
        source_text: The chunk's ORIGINAL document text without any
            synthetic prefix — the corpus quote verification runs
            against (a quote must exist in the cited source, not in
            machine-generated scaffolding). Empty for chunks ingested
            before the field existed; consumers fall back to ``text``.
        page_number: Best-effort 1-based source page this chunk maps to
            (PDFs only), captured at ingest by overlapping the chunk text
            against per-page text. ``None`` when the source carries no
            page concept, the mapping was inconclusive, or the chunk was
            ingested before the field existed — never a guessed value
            (No Silent Fallbacks). Enables a page-level "open PDF at page
            N" jump; NOT an exact bounding box.
    """

    id: str
    document_id: str
    collection_id: str
    chunk_index: int
    text: str
    embedding: tuple[float, ...] = field(repr=False, default=())
    source_text: str = ""
    page_number: int | None = None


@dataclass(frozen=True)
class RetrievalCandidate:
    """One scored retrieval hit.

    Attributes:
        chunk: The matched chunk.
        score: Similarity score (cosine, higher is better).
        document_title: Title of the owning document, resolved by the
            store so consumers need no second lookup.
    """

    chunk: DocumentChunk
    score: float
    document_title: str


@runtime_checkable
class KnowledgeStore(Protocol):
    """Persistence and retrieval port for collections/documents/chunks.

    The first implementation is in-memory
    (:class:`~inqtrix.knowledge.stores.memory.MemoryKnowledgeStore`);
    Postgres-canonical and Qdrant-backed implementations are
    drop-ins. **All methods are async**: the platform persistence layer
    is async end-to-end (asyncpg), and a uniform async port lets the
    HTTP routes ``await`` directly while the synchronous research graph
    and the reindex worker bridge through ``asyncio.run`` in their own
    threads (the established sync-bridge pattern).
    """

    async def create_collection(
        self,
        *,
        name: str,
        embedding_model: str,
        embedding_dim: int,
        created_by_sub: str | None = None,
    ) -> KnowledgeCollection: ...

    async def list_collections(self) -> list[KnowledgeCollection]: ...

    async def get_collection(self, collection_id: str) -> KnowledgeCollection: ...

    async def delete_collection(self, collection_id: str) -> None: ...

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
    ) -> KnowledgeDocument: ...

    async def list_documents(self, collection_id: str) -> list[KnowledgeDocument]: ...

    async def list_documents_page(
        self,
        collection_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[KnowledgeDocument], str | None]:
        """One keyset page of a collection's documents (newest first).

        Separate from :meth:`list_documents` (which returns ALL documents —
        the reindex worker re-embeds the whole collection): this is the
        bounded HTTP-listing path. Returns the page and the ``next_cursor``
        (``None`` on the last page). Ordering is ``(created_at, id)``
        descending; the id is the tiebreaker for the float-epoch
        ``created_at``."""
        ...

    async def get_document(self, document_id: str) -> KnowledgeDocument: ...

    async def delete_document(self, document_id: str) -> None: ...

    async def reembed_document(
        self,
        *,
        document_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_chunks: list[str] | None = None,
        page_numbers: list[int | None] | None = None,
    ) -> KnowledgeDocument:
        """Replace a document's chunks/vectors in place, keeping its id.

        The re-embed primitive behind background reindexing: the
        document's identity, title, text, metadata, and creation time
        are preserved (citations and client references stay valid),
        only the chunk set and embeddings are rebuilt. ``chunk_count``
        is updated to ``len(chunks)``.

        Optional because durable stores adopt it incrementally; the
        :class:`~inqtrix.services.indexing_service.IndexingService`
        checks for support and fails the job visibly when a store
        lacks it, never silently no-ops.

        Raises:
            DocumentNotFound: Unknown *document_id*.
            EmbeddingDimensionMismatch: A vector's dimension contradicts
                the collection, or chunk/embedding counts differ.
        """
        ...

    async def search(
        self,
        *,
        query_embedding: list[float],
        collection_ids: list[str] | None,
        top_k: int,
        embedding_model: str | None = None,
    ) -> list[RetrievalCandidate]: ...


@runtime_checkable
class HybridKnowledgeStore(KnowledgeStore, Protocol):
    """Optional store capability: fused dense + sparse retrieval.

    Stores advertising this protocol fuse a semantic (dense) and a
    lexical (sparse/BM25) branch server-side; the service dispatches
    here automatically and ``/v1/capabilities`` reports
    ``features.hybrid_retrieval`` so the upgrade is visible, never
    silent.
    """

    @property
    def supports_hybrid(self) -> bool:
        """Whether the lexical branch is actually configured.

        ``runtime_checkable`` protocols only verify method EXISTENCE —
        this flag carries the real capability (a store may implement
        the method but run dense-only by configuration)."""
        ...

    async def hybrid_search(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        collection_ids: list[str] | None,
        top_k: int,
        embedding_model: str | None = None,
    ) -> list[RetrievalCandidate]: ...


@dataclass(frozen=True)
class KnowledgeProviderContext:
    """Bundle of knowledge capabilities wired by the composition root.

    Attributes:
        embeddings: The dense embedding provider (query + document
            embedding plus the selectable-model catalog surface).
        store: The collection/document/vector store.
        default_top_k: Default number of evidence chunks retrieved per
            question when the request does not override it.
        reranker: Optional cross-encoder re-scoring stage applied
            after retrieval. ``None`` skips the stage — visible via
            ``features.reranker`` in the capability manifest.
        rerank_candidate_depth: Candidate pool retrieved before
            reranking when a reranker is wired (the reranker reduces
            this pool to the requested top_k).
        contextualizer: Optional ingestion-time chunk
            contextualization (contextual retrieval). ``None`` ingests
            raw chunks — visible via ``features.contextual_retrieval``.
    """

    embeddings: "EmbeddingProvider"
    store: KnowledgeStore
    default_top_k: int = 8
    reranker: "RerankerProvider | None" = None
    rerank_candidate_depth: int = 40
    contextualizer: "ChunkContextualizer | None" = None
