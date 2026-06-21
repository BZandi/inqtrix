"""SQLAlchemy Core definitions of the durable reindex-job schema.

The durable twin of the in-memory
:class:`~inqtrix.server.indexing.IndexingJobStore`: it makes a
background re-embed survive not just closing the browser (the in-memory
store already does that) but closing the *server* — the job row is the
source of truth and an :mod:`inqtrix.worker` process executes it.

The shape mirrors :mod:`inqtrix.storage.runs_orm` deliberately (same
``claimed_by``/``attempt`` worker fencing, ``event_seq`` allocator,
JSON-not-JSONB for byte-level SSE replay parity, composite
``(job_id, sequence)`` event PK with CASCADE) so the run-worker
durability machinery can be reused rather than re-derived. The
differences from runs are the reindex domain itself:

* progress columns runs lack (``total_documents``,
  ``completed_documents``, ``current_document_title``) — the snapshot is
  derived from these, so there is no opaque ``snapshot`` column;
* a ``collection_id`` referencing ``knowledge_collections`` (no hard FK,
  matching the run schema's cross-domain foreign-key-free design);
* a partial unique index enforcing one active job per collection — the
  in-memory store serializes this under a lock; the durable store needs
  a DB constraint so two API/worker processes cannot race two re-embed
  passes over the same documents (``IndexingJobConflict`` / HTTP 409).

``status`` is text + a CHECK constraint added in the migration; the
lifecycle ordering authority stays
:class:`~inqtrix.server.indexing.IndexingJobStatus` (no native PG enum).
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
from sqlalchemy.dialects.postgresql import JSON

indexing_metadata = MetaData()

indexing_jobs = Table(
    "indexing_jobs",
    indexing_metadata,
    Column("job_id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    # References ``knowledge_collections.id`` logically, but with no hard
    # FK: that table lives in a different MetaData, and a cross-metadata
    # FK breaks ``create_all`` (it cannot resolve the referent). The run
    # schema is likewise foreign-key-free across domains. An orphan job
    # for a deleted collection is harmless — terminal ones TTL out, an
    # active one re-embeds zero documents and completes.
    Column("collection_id", Text, nullable=False),
    Column("collection_name", Text, nullable=False, server_default=text("''")),
    Column("embedding_model", Text, nullable=False, server_default=text("''")),
    Column("index_id", Text, nullable=True),
    Column(
        "status",
        Text,
        nullable=False,
        server_default=text("'queued'"),
    ),
    Column("workspace_id", Text, nullable=True),
    Column("created_by_sub", Text, nullable=True),
    Column("created_by_tenant_id", Text, nullable=True),
    Column("total_documents", Integer, nullable=False, server_default=text("0")),
    Column(
        "completed_documents", Integer, nullable=False, server_default=text("0")
    ),
    Column(
        "current_document_title", Text, nullable=False, server_default=text("''")
    ),
    Column("error", JSON, nullable=True),
    Column(
        "cancel_requested",
        Boolean,
        nullable=False,
        server_default=text("false"),
    ),
    Column("claimed_by", Text, nullable=True),
    Column("attempt", Integer, nullable=False, server_default=text("0")),
    Column("event_seq", Integer, nullable=False, server_default=text("0")),
    Column("created_at", Float, nullable=False),
    Column("started_at", Float, nullable=True),
    Column("finished_at", Float, nullable=True),
    Index(
        "ix_indexing_jobs_collection_created", "collection_id", "created_at"
    ),
    Index("ix_indexing_jobs_tenant_status", "tenant_id", "status"),
    # One active reindex per collection, enforced at the database so two
    # processes cannot race two re-embed passes (IndexingJobConflict).
    Index(
        "uq_indexing_jobs_active_collection",
        "collection_id",
        unique=True,
        postgresql_where=text("status IN ('queued', 'running')"),
    ),
)
"""Durable reindex-job records — the source of truth once
``INQTRIX_STORAGE_BACKEND=postgres`` is active. The status CHECK
constraint is added in migration 0011 (values from
:class:`~inqtrix.server.indexing.IndexingJobStatus`); the progress
snapshot is derived from the count columns, not stored, so it cannot
drift from the wire shape."""

indexing_job_events = Table(
    "indexing_job_events",
    indexing_metadata,
    Column(
        "job_id",
        Text,
        ForeignKey("indexing_jobs.job_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("sequence", Integer, primary_key=True),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("type", Text, nullable=False),
    Column("created_at", Float, nullable=False),
    Column("data", JSON, nullable=False, server_default=text("'{}'")),
)
"""Per-job event log in emission order; ``(job_id, sequence)`` is the
primary key and ``sequence`` is allocated from
``indexing_jobs.event_seq`` so SSE replay reproduces the in-memory
stream byte-compatibly."""
