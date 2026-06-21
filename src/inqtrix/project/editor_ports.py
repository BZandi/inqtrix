"""Contracts of the editor-persistence store (M6b).

Mirrors ``chat_ports.py``: the store owns persistence only; scoping
(owner/share visibility) lives in
:class:`~inqtrix.services.editor_persistence_service.EditorPersistenceService`
and the wire shape in the router. Two implementations behind the same
port: :class:`~inqtrix.project.editor_memory.MemoryEditorStore` (the tier
without Postgres, also the offline test backend) and
:class:`~inqtrix.project.editor_postgres.PostgresEditorStore`.

Differences from the chat store:

* A document carries a heavy ``content_markdown`` body. ``list_documents_page``
  returns metadata WITHOUT the body (``content_markdown=""``); ``get_document``
  returns the full document with the body (the load-on-open path).
* Comments are independently mutated (resolve / edit / re-tag), so they are
  a diffed collection with delete propagation — ``upsert_comments`` +
  ``delete_comment`` + ``list_comments_page`` — not append-only like chat
  messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class DocumentNotFound(KeyError):
    """Raised when a document id is unknown to the store (maps to HTTP 404)."""


class FolderNotFound(KeyError):
    """Raised when a folder id is unknown to the store (maps to HTTP 404)."""


@dataclass(frozen=True)
class EditorFolder:
    """One grouping of a user's editor documents (a tree folder).

    Attributes mirror :class:`~inqtrix.project.chat_ports.ChatThreadGroup`:
    a client-supplied prefixed id (``edf_``), unix-seconds timestamps, and
    the ``(tenant_id, created_by_sub, workspace_id)`` scope.
    """

    id: str
    title: str
    created_at: float
    updated_at: float
    tenant_id: str = "default"
    created_by_sub: str | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class EditorDocument:
    """One editor document. ``content_markdown`` is the heavy body — empty
    on records returned by ``list_documents_page`` (metadata only) and
    populated by ``get_document`` (load-on-open).

    Attributes:
        id: Client-supplied id (``ed_``), the primary key.
        title: Document title.
        content_markdown: The document body (heavy; lazy).
        folder_id: Owning folder id, or ``None`` when at the tree root.
        source: Origin (``blank``/``imported-research-report``/``pasted``).
        source_run_id: The research run a document was imported from.
        revision: Client mutation counter (round-tripped verbatim).
        diff_anchor_markdown: Snapshot body for the in-editor diff, or None.
        diff_anchor_updated_at: Unix timestamp of the diff snapshot, or None.
        created_at: Unix timestamp of creation (the stable list-sort key).
        updated_at: Unix timestamp of the last change (display + autosave key).
    """

    id: str
    title: str
    content_markdown: str = field(repr=False, default="")
    folder_id: str | None = None
    source: str = "blank"
    source_run_id: str | None = None
    revision: int = 1
    diff_anchor_markdown: str | None = None
    diff_anchor_updated_at: float | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    tenant_id: str = "default"
    created_by_sub: str | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class EditorComment:
    """One anchored comment on a document.

    Attributes:
        id: Client-supplied id (``edc_``); PK together with ``document_id``.
        document_id: Owning document.
        comment_markdown: The comment body text.
        anchor: The verbatim positional anchor (block id, char range,
            surrounding quotes) — stored as-is, never reinterpreted.
        kind: ``collect``/``inline_edit``/``evidence_review``.
        status: ``open``/``resolved``/``stale``.
        evidence_preset: Optional evidence preset, or ``None``.
        created_at: Unix timestamp (stable; the keyset key within a doc).
        updated_at: Unix timestamp of the last mutation (autosave diff key).
    """

    id: str
    document_id: str
    comment_markdown: str
    anchor: dict[str, Any] = field(default_factory=dict)
    kind: str = "collect"
    status: str = "open"
    evidence_preset: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    tenant_id: str = "default"


@runtime_checkable
class EditorStore(Protocol):
    """Persistence port for editor documents, folders, and comments.

    Scoping note (same as the chat store): ``list_documents_page`` /
    ``list_folders`` take the resolved ``created_by_sub`` + ``workspace_id``
    and filter in the query so the DB ``LIMIT`` is never under-filled.
    ``get_document`` returns the row unscoped — the service applies the
    owner/share access check on top.
    """

    # -- documents -------------------------------------------------------- #

    async def upsert_document(
        self,
        *,
        id: str,
        title: str,
        content_markdown: str,
        folder_id: str | None,
        source: str,
        source_run_id: str | None,
        revision: int,
        diff_anchor_markdown: str | None,
        diff_anchor_updated_at: float | None,
        created_at: float,
        updated_at: float,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> EditorDocument:
        """Create or idempotently update a document by id (autosave upsert).

        An existing id keeps its ``created_at`` and ownership columns; only
        the mutable fields (title, body, folder, source metadata, revision,
        diff anchor, updated_at) are overwritten."""
        ...

    async def list_documents_page(
        self,
        *,
        created_by_sub: str | None,
        workspace_id: str | None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[EditorDocument], str | None]:
        """One keyset page of the caller's documents (newest first),
        METADATA ONLY (``content_markdown=""`` — the body loads on open)."""
        ...

    async def get_document(self, document_id: str) -> EditorDocument:
        """One document WITH its body (load-on-open), or
        :class:`DocumentNotFound`."""
        ...

    async def delete_document(self, document_id: str) -> None:
        """Delete a document and its comments (cascade)."""
        ...

    # -- folders ---------------------------------------------------------- #

    async def upsert_folder(
        self,
        *,
        id: str,
        title: str,
        created_at: float,
        updated_at: float,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> EditorFolder:
        """Create or idempotently update a folder by id."""
        ...

    async def list_folders(
        self,
        *,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> list[EditorFolder]:
        """All of the caller's folders, newest first (few — no keyset page)."""
        ...

    async def delete_folder(self, folder_id: str) -> None:
        """Delete a folder; its documents orphan to ungrouped (SET NULL)."""
        ...

    # -- comments --------------------------------------------------------- #

    async def upsert_comments(
        self, comments: list[EditorComment]
    ) -> list[EditorComment]:
        """Idempotently upsert comments by (document_id, id). An existing
        comment overwrites its mutable fields (body/anchor/kind/status/
        preset/updated_at), never ``created_at``."""
        ...

    async def list_comments_page(
        self,
        document_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[EditorComment], str | None]:
        """One keyset page of a document's comments (newest first)."""
        ...

    async def delete_comment(self, *, document_id: str, comment_id: str) -> None:
        """Delete one comment from a document."""
        ...

    async def aclose(self) -> None:
        """Release backing resources at application shutdown."""
        ...
