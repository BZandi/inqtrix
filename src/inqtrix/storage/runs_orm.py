"""SQLAlchemy Core definitions of the durable run schema.

Separate ``MetaData`` from the identity and content schemas on
purpose: each migration's metadata is an immutable snapshot — run
tables arrive with their own revision (0003).

Type decisions:

* ``run_id`` is text (``run_...``) — the existing public run
  identifier, unchanged from the in-memory store.
* All timestamps are unix-seconds doubles, mirroring the in-memory
  :class:`~inqtrix.server.runs.RunRecord` and the pinned wire format
  (``created_at``/``started_at``/``finished_at`` are float epochs in
  run summaries and event envelopes).
* JSON columns use the text-preserving ``JSON`` type, NOT ``JSONB``:
  JSONB normalizes key order, which would break the byte-level SSE
  replay parity with the in-memory store (``json.dumps`` of the
  loaded dict must reproduce the original frame).
* ``status`` is text + CHECK; the lifecycle ordering lives only in
  :class:`~inqtrix.server.runs.RunStatus` (single ordering authority,
  no native PG enums).
* ``event_seq`` on the run row is the per-run sequence allocator:
  ``UPDATE ... SET event_seq = event_seq + 1 RETURNING event_seq``
  serializes on the row lock, which keeps sequences gap-free and
  strictly increasing across processes — the property the byte-level
  SSE contract pins.
* ``claimed_by`` + ``attempt`` are the worker fencing pair: a claim
  increments ``attempt`` and every terminal write is guarded by both,
  so a zombie worker whose job was reclaimed cannot overwrite the
  second attempt's result.
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

runs_metadata = MetaData()

runs = Table(
    "runs",
    runs_metadata,
    Column("run_id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column(
        "status",
        Text,
        nullable=False,
        server_default=text("'queued'"),
    ),
    Column("mode", Text, nullable=False, server_default=text("'research'")),
    Column("question", Text, nullable=False),
    Column("stack_name", Text, nullable=False, server_default=text("'default'")),
    Column("workspace_id", Text, nullable=True),
    Column("created_by_sub", Text, nullable=True),
    Column("created_by_tenant_id", Text, nullable=True),
    Column("agent_overrides", JSON, nullable=False, server_default=text("'{}'")),
    Column("request_payload", JSON, nullable=True),
    Column("snapshot", JSON, nullable=False, server_default=text("'{}'")),
    Column("result", JSON, nullable=True),
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
    # Keyset-pagination index: the id tiebreaker after created_at is
    # mandatory — created_at is a float epoch and collides on bulk inserts,
    # so a (created_at, run_id) cursor needs both columns to page without
    # skipping or repeating rows. Supersedes the old (tenant_id, created_at).
    Index("ix_runs_tenant_created_id", "tenant_id", "created_at", "run_id"),
    Index("ix_runs_tenant_status", "tenant_id", "status"),
)
"""Durable run records — the source of truth once
``INQTRIX_STORAGE_BACKEND=postgres`` is active; the Valkey stream only
carries dispatch messages. The status CHECK constraint is added in
migration 0003 (values from :class:`~inqtrix.server.runs.RunStatus`)."""

run_events = Table(
    "run_events",
    runs_metadata,
    Column(
        "run_id",
        Text,
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("sequence", Integer, primary_key=True),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("type", Text, nullable=False),
    Column("created_at", Float, nullable=False),
    Column("data", JSON, nullable=False, server_default=text("'{}'")),
)
"""Per-run event log in emission order; ``(run_id, sequence)`` is the
primary key and ``sequence`` is allocated from ``runs.event_seq`` so
replay reproduces the in-memory SSE stream byte-compatibly."""
