"""SQLAlchemy Core definitions of the canonical knowledge schema.

Separate ``MetaData`` from the other domain schemas on purpose (its
migration is an immutable snapshot, like ``runs_metadata``). This is the
**relational source of truth** for the knowledge engine once
``INQTRIX_STORAGE_BACKEND=postgres`` is active: collections, documents
(full canonical text), and chunk metadata live here; Qdrant holds ONLY
the vectors plus a lean payload (the chunk id and filter keys). The
split follows the standard "Postgres = source of truth, vector DB =
derived index" topology.

Type decisions (mirroring ``runs_orm.py``):

* Ids are text with the existing public prefixes (``kc_``/``kd_``/
  ``kch_``), unchanged from the in-memory and Qdrant stores so the wire
  surface stays byte-identical.
* Timestamps are unix-seconds doubles (``Float``), matching the
  :class:`~inqtrix.knowledge.stores.ports.KnowledgeCollection` /
  ``KnowledgeDocument`` dataclasses and every other table here.
* ``metadata`` uses the text-preserving ``JSON`` type (NOT ``JSONB``) so
  a loaded dict re-serializes to the document the client supplied —
  client metadata is stored verbatim, never reinterpreted.
* ``embedding_dim`` is immutable per collection (set at creation,
  enforced on every upsert); it lives on the collection row.
* ``vector_synced`` on the document is the durable vector-sync
  diagnostic: the canonical text commits to Postgres first, then the
  vectors upsert to Qdrant and the flag flips ``true``; a failed vector
  sync leaves it ``false`` (a queryable, operator-visible "vectors out
  of sync" signal — No Silent Fallbacks), and re-running reindex
  re-embeds from the canonical text and clears it. No outbox table and
  no vector duplication in Postgres. An automatic reconcile sweep keyed
  on the flag is a follow-up. Chunk vectors are never stored here.
* Every table carries ``tenant_id`` for the established row-level
  security layering (GRANT + ENABLE/FORCE RLS + fail-closed tenant
  policy added in the migration, like ``0007_quota``).
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID

knowledge_metadata = MetaData()

knowledge_collections = Table(
    "knowledge_collections",
    knowledge_metadata,
    Column("id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("name", Text, nullable=False),
    Column("embedding_model", Text, nullable=False),
    Column("embedding_dim", Integer, nullable=False),
    Column("created_by_user_id", UUID(as_uuid=True), nullable=True),
    Column("created_at", Float, nullable=False),
    Index("ix_knowledge_collections_tenant_created", "tenant_id", "created_at"),
)
"""Logical knowledge collections. ``embedding_model``/``embedding_dim``
are immutable after creation; ``created_by_user_id`` is the ownership anchor
(``None`` = legacy/unscoped, visible to all — the established rule). Knowledge
is the cross-workspace SHARING surface (owner + ACL via resource_shares), NOT
per-workspace project data, so there is deliberately NO workspace_id dimension
here (see ADR-AUTH-6 and docs/architecture/data-architecture.md). A dead
always-NULL workspace_id column was dropped in migration 0018."""

knowledge_documents = Table(
    "knowledge_documents",
    knowledge_metadata,
    Column("id", Text, primary_key=True),
    Column(
        "collection_id",
        Text,
        ForeignKey("knowledge_collections.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("title", Text, nullable=False),
    Column("text", Text, nullable=False, server_default=text("''")),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
    Column("chunk_count", Integer, nullable=False, server_default=text("0")),
    Column(
        "vector_synced",
        Boolean,
        nullable=False,
        server_default=text("false"),
    ),
    Column("created_at", Float, nullable=False),
    # Keyset-pagination index with the id tiebreaker (created_at is a float
    # epoch that can collide); supersedes (collection_id, created_at).
    Index(
        "ix_knowledge_documents_collection_created_id",
        "collection_id",
        "created_at",
        "id",
    ),
)
"""Ingested documents with their full canonical ``text`` (the source of
truth retrieval hydrates from and reindex re-embeds from). ``metadata``
is stored verbatim. ``vector_synced`` is the reconcile flag (see module
docstring)."""

knowledge_chunks = Table(
    "knowledge_chunks",
    knowledge_metadata,
    Column("id", Text, primary_key=True),
    Column(
        "document_id",
        Text,
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("collection_id", Text, nullable=False),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("chunk_index", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("source_text", Text, nullable=False, server_default=text("''")),
    Column("page_number", Integer, nullable=True),
    Column("created_at", Float, nullable=False),
    Index("ix_knowledge_chunks_document_index", "document_id", "chunk_index"),
)
"""Per-chunk metadata: the embedded ``text`` (may carry a
contextualization prefix) and the pre-contextualization ``source_text``
quote verification runs against. The DENSE/sparse VECTORS are NOT stored
here — they live in Qdrant, keyed by this chunk ``id``. Retrieval returns
chunk ids from Qdrant and hydrates text/source_text + document title from
these rows."""
