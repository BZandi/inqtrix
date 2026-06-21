"""SQLAlchemy Core definitions of the editor-persistence schema (M6b).

The second slice of the project-persistence tier, mirroring the chat
schema (``chat_orm.py``): when the platform runs with a Postgres backend,
a user's editor documents, their folders, and their comments become
server-persistent instead of living only in the local markdown project.
Scoped per ``(tenant_id, created_by_sub, workspace_id)`` exactly like
chat / runs / files / knowledge — a user's project is "their data in the
current workspace" (the one-project-per-(user, workspace) model). No
separate ``projects`` table.

Mapping to the chat analogue:

* ``editor_folders``  ~ ``chat_thread_groups`` (grouping; metadata only).
* ``editor_documents`` ~ ``chat_threads``, PLUS a heavy ``content_markdown``
  body column (the document text — lazy-loaded on open, never returned by
  the list endpoint) and editor-specific metadata (revision, source,
  diff-anchor snapshot, folder membership as a nullable ``folder_id``).
* ``editor_comments`` ~ ``chat_messages`` (a composite-PK child of the
  document, cascade-deleted), but UNLIKE messages they are independently
  mutated (resolve / edit / re-tag), so they carry their own ``updated_at``
  and the autosave diffs them per comment.

Type decisions match ``chat_orm.py``: client-supplied prefixed ids as the
primary key (``ed_``/``edf_``/``edc_``), unix-seconds ``Float`` timestamps,
text-preserving ``JSON`` (not ``JSONB``) for the verbatim comment anchor,
per-table ``tenant_id`` for the RLS layering, and keyset indexes with the
``id`` tiebreaker. CHECK constraints pin the source/kind/status/preset
domains to the frontend unions so an out-of-domain write fails loudly.
"""

from __future__ import annotations

from sqlalchemy import (
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

editor_metadata = MetaData()

editor_folders = Table(
    "editor_folders",
    editor_metadata,
    Column("id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("created_by_sub", Text, nullable=True),
    Column("workspace_id", Text, nullable=True),
    Column("title", Text, nullable=False),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Index(
        "ix_editor_folders_owner_created",
        "tenant_id",
        "created_by_sub",
        "created_at",
        "id",
    ),
)
"""Optional grouping of a user's editor documents (the document tree).
``created_by_sub`` is the ownership anchor (``None`` = unscoped/anonymous
deployments). Deleting a folder orphans its documents to ungrouped
(``ON DELETE SET NULL`` on ``editor_documents.folder_id``)."""

editor_documents = Table(
    "editor_documents",
    editor_metadata,
    Column("id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("created_by_sub", Text, nullable=True),
    Column("workspace_id", Text, nullable=True),
    Column("title", Text, nullable=False, server_default=text("''")),
    # The heavy body. Excluded from the list endpoint (metadata only) and
    # loaded on open, the documents-equivalent of the lazy chat messages.
    Column("content_markdown", Text, nullable=False, server_default=text("''")),
    Column(
        "folder_id",
        Text,
        ForeignKey("editor_folders.id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("source", Text, nullable=False, server_default=text("'blank'")),
    Column("source_run_id", Text, nullable=True),
    Column("revision", Integer, nullable=False, server_default=text("1")),
    Column("diff_anchor_markdown", Text, nullable=True),
    Column("diff_anchor_updated_at", Float, nullable=True),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    # Keyset-pagination index for the owner-scoped document list (newest
    # first) with the id tiebreaker; sort is by the stable created_at.
    Index(
        "ix_editor_documents_owner_created",
        "tenant_id",
        "created_by_sub",
        "created_at",
        "id",
    ),
)
"""One editor document. ``content_markdown`` is the heavy body (lazy,
PUT with the document on autosave); ``revision``/``source``/
``source_run_id``/``diff_anchor_*`` round-trip the local
``EditorDocumentRecord``. ``folder_id`` is the (nullable) tree membership."""

editor_comments = Table(
    "editor_comments",
    editor_metadata,
    # COMPOSITE primary key (document_id, id): a comment's identity is
    # scoped to its document, never global — the same isolation rule the
    # chat_messages composite PK encodes (an autosave into document B can
    # never overwrite a same-id comment living in document A).
    Column("id", Text, primary_key=True),
    Column(
        "document_id",
        Text,
        ForeignKey("editor_documents.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("comment_markdown", Text, nullable=False, server_default=text("''")),
    # The positional anchor (block id, char range, surrounding quotes),
    # stored verbatim so a round-trip reconstructs the exact anchor.
    Column("anchor", JSON, nullable=False, server_default=text("'{}'")),
    Column("kind", Text, nullable=False),
    Column("status", Text, nullable=False, server_default=text("'open'")),
    Column("evidence_preset", Text, nullable=True),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Index("ix_editor_comments_document_created", "document_id", "created_at", "id"),
)
"""One anchored comment on a document. Unlike chat messages, comments are
independently mutated (resolve / edit / re-tag), so ``updated_at`` is the
autosave diff key. Visibility inherits from the parent document. Cascades
on document delete."""
