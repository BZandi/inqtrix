"""SQLAlchemy Core definitions of the editor-patch schema (M7).

Own ``MetaData`` snapshot (migration 0031), separate from the editor and
runs schemas on purpose — each migration's metadata is immutable.

Design decisions (the 0030 pattern):

* ``document_id`` is a raw-DDL FK to ``editor_documents(id)`` with
  ``ON DELETE CASCADE`` and ``run_id`` a raw-DDL FK to ``runs(run_id)``
  with ``ON DELETE SET NULL`` — both parents live in other MetaData
  snapshots, so the constraints are added by migration 0031, not here.
  The document owns the patch's lifetime; run retention only detaches
  the back-reference.
* Timestamps are unix-seconds floats, JSON columns are text-preserving
  ``JSON`` (not JSONB) — wire payloads round-trip byte-identically
  between the memory and Postgres stores.
* ``source``/``status`` are text + CHECK built from the port's canonical
  tuples in :mod:`inqtrix.project.editor_patch_ports` (single vocabulary
  authority, no native PG enums).
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSON

from inqtrix.project.editor_patch_ports import PATCH_SOURCES, PATCH_STATUSES

editor_patch_metadata = MetaData()


def _values(options: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in options)


editor_patches = Table(
    "editor_patches",
    editor_patch_metadata,
    Column("patch_id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    # Both FKs (document_id -> editor_documents(id) ON DELETE CASCADE,
    # run_id -> runs(run_id) ON DELETE SET NULL) are raw DDL in migration
    # 0031 — the parents live in other MetaData snapshots (module docstring).
    Column("document_id", Text, nullable=False),
    Column("run_id", Text, nullable=True),
    Column("source", Text, nullable=False),
    Column("status", Text, nullable=False, server_default=text("'pending'")),
    Column("edits", JSON, nullable=False, server_default=text("'[]'")),
    Column("summary", Text, nullable=False, server_default=text("''")),
    Column("warnings", JSON, nullable=False, server_default=text("'[]'")),
    Column("revision_before", Integer, nullable=False),
    Column("applied_revision", Integer, nullable=True),
    Column("applied_edit_ids", JSON, nullable=True),
    Column("note", Text, nullable=False, server_default=text("''")),
    Column("created_by_sub", Text, nullable=True),
    Column("created_at", Float, nullable=False),
    Column("decided_at", Float, nullable=True),
    CheckConstraint(
        f"source IN ({_values(PATCH_SOURCES)})", name="ck_editor_patches_source"
    ),
    CheckConstraint(
        f"status IN ({_values(PATCH_STATUSES)})", name="ck_editor_patches_status"
    ),
    Index(
        "ix_editor_patches_tenant_document",
        "tenant_id",
        "document_id",
        "created_at",
        "patch_id",
    ),
)
"""Proposed anchored document edits with their apply/reject lifecycle;
the ``pending -> accepted/rejected`` CAS happens in the store's mark
writes."""
