"""Contracts of the editor-persistence store (M6b).

Mirrors ``chat_ports.py``: the service resolves owner/share visibility and the
store owns persistence. Child mutations additionally receive the resolved
parent identity so the durable store can lock the parent and revalidate live
authority in its transaction. The wire shape remains in the router. Two
implementations sit behind the same port:
:class:`~inqtrix.project.editor_memory.MemoryEditorStore` (the tier without
Postgres, also the offline test backend) and
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

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from inqtrix.project.scoped_upsert import ResourceScope

from inqtrix.auth.permissions import SharePermission

if TYPE_CHECKING:
    from inqtrix.auth.permissions import ResourceAccess


class DocumentNotFound(KeyError):
    """Raised when a document id is unknown to the store (maps to HTTP 404)."""


class DocumentRevisionConflict(RuntimeError):
    """Raised when a document save loses the revision guard (HTTP 409).

    The store accepts a save only when it carries a STRICTLY newer
    revision than the stored row (monotonic guard, not base+1-exact —
    debounced autosave legitimately jumps several revisions per flush).
    A concurrent writer — human autosave vs. agent patch apply — whose
    counter is stale-or-equal gets this instead of silently overwriting
    the newer content or moving the revision backwards. The carried
    ``current_revision`` lets the client refetch and rebase precisely.

    Attributes:
        current_revision: The revision actually stored right now.
        expected_revision: The base the losing writer assumed
            (its ``revision - 1``).
    """

    def __init__(self, *, current_revision: int, expected_revision: int) -> None:
        super().__init__(
            f"document revision moved to {current_revision}, "
            f"writer assumed base {expected_revision}"
        )
        self.current_revision = current_revision
        self.expected_revision = expected_revision


class DocumentContentModeConflict(RuntimeError):
    """Raised when a legacy body write targets a collaboration document."""

    def __init__(self, document_id: str) -> None:
        super().__init__(f"document {document_id} is no longer markdown-owned")
        self.document_id = document_id


class DocumentMetadataConflict(RuntimeError):
    """Raised when an owner metadata PATCH loses its revision guard."""

    def __init__(self, *, current_revision: int) -> None:
        super().__init__(f"document metadata moved to {current_revision}")
        self.current_revision = current_revision


class FolderNotFound(KeyError):
    """Raised when a folder id is unknown to the store (maps to HTTP 404)."""


@dataclass(frozen=True)
class EditorFolder:
    """One grouping of a user's editor documents (a tree folder).

    Attributes mirror :class:`~inqtrix.project.chat_ports.ChatThreadGroup`:
    a client-supplied prefixed id (``edf_``), unix-seconds timestamps, and
    the ``(tenant_id, created_by_user_id, workspace_id)`` scope.
    """

    id: str
    title: str
    created_at: float
    updated_at: float
    tenant_id: str = "default"
    created_by_user_id: uuid.UUID | None = None
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
    created_by_user_id: uuid.UUID | None = None
    workspace_id: str | None = None
    content_mode: str = "markdown"
    metadata_revision: int = 1
    collaboration_generation: int = 0
    collaboration_schema_version: int | None = None
    collaboration_schema_hash: str | None = None
    persisted_sequence: int = 0
    projection_sequence: int = 0
    projection_updated_at: float | None = None
    deleted_at: float | None = None


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
    created_by_user_id: uuid.UUID | None = None


def comment_write_permission(content_mode: str) -> SharePermission:
    """Return the live permission required to mutate document comments.

    Collaboration comments are a suggestion surface; legacy Markdown
    comments remain an edit surface. Keeping this mapping in the persistence
    contract lets the service preflight and the transactional PostgreSQL
    authority check use one policy definition.
    """
    if content_mode == "collaboration":
        return SharePermission.SUGGEST
    if content_mode == "markdown":
        return SharePermission.EDIT
    raise ValueError(f"unsupported editor content mode: {content_mode!r}")


@runtime_checkable
class EditorStore(Protocol):
    """Persistence port for editor documents, folders, and comments.

    Scoping note (same as the chat store): ``list_documents_page`` /
    ``list_folders`` take the resolved ``created_by_user_id`` + ``workspace_id``
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
        created_by_user_id: uuid.UUID | None,
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
        created_by_user_id: uuid.UUID | None,
        workspace_id: str | None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[EditorDocument], str | None]:
        """One keyset page of the caller's documents (newest first),
        METADATA ONLY (``content_markdown=""`` — the body loads on open)."""
        ...

    async def list_visible_documents_page(
        self,
        *,
        actor_user_id: uuid.UUID | None,
        workspace_id: str | None,
        scope: Literal["owned", "shared", "all"],
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[tuple[EditorDocument, "ResourceAccess"]], str | None]:
        """List owned and/or accepted-shared documents with live access."""
        ...

    async def get_document(self, document_id: str) -> EditorDocument:
        """One document WITH its body (load-on-open), or
        :class:`DocumentNotFound`."""
        ...

    async def patch_document_metadata(
        self,
        *,
        document_id: str,
        expected_metadata_revision: int,
        title: str | None,
        folder_id: str | None,
        set_folder_id: bool,
        diff_anchor_markdown: str | None,
        set_diff_anchor_markdown: bool,
        diff_anchor_updated_at: float | None,
        set_diff_anchor_updated_at: bool,
        updated_at: float,
        scope: ResourceScope,
    ) -> EditorDocument:
        """CAS-update owner-managed metadata without touching the body."""
        ...

    async def delete_document(
        self, document_id: str, *, scope: ResourceScope
    ) -> None:
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
        created_by_user_id: uuid.UUID | None,
        workspace_id: str | None,
    ) -> EditorFolder:
        """Create or idempotently update a folder by id."""
        ...

    async def list_folders(
        self,
        *,
        created_by_user_id: uuid.UUID | None,
        workspace_id: str | None,
    ) -> list[EditorFolder]:
        """All of the caller's folders, newest first (few — no keyset page)."""
        ...

    async def delete_folder(
        self, folder_id: str, *, scope: ResourceScope
    ) -> None:
        """Delete a folder; its documents orphan to ungrouped (SET NULL)."""
        ...

    # -- comments --------------------------------------------------------- #

    async def upsert_comments(
        self,
        comments: list[EditorComment],
        *,
        expected_document_id: str,
        expected_document_owner_user_id: uuid.UUID | None,
        expected_document_workspace_id: str | None,
        expected_document_content_mode: str,
        actor_user_id: uuid.UUID | None,
    ) -> list[EditorComment]:
        """Idempotently upsert comments by (document_id, id). An existing
        comment overwrites its mutable fields (body/anchor/kind/status/
        preset/updated_at), never ``created_at``. The expected document
        identity and current actor authority are revalidated in the same
        transaction as the child write."""
        ...

    async def list_comments_page(
        self,
        document_id: str,
        *,
        created_by_user_id: uuid.UUID | None = None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[EditorComment], str | None]:
        """One keyset page of a document's comments (newest first)."""
        ...

    async def delete_comment(
        self,
        *,
        document_id: str,
        comment_id: str,
        created_by_user_id: uuid.UUID | None = None,
        expected_document_owner_user_id: uuid.UUID | None,
        expected_document_workspace_id: str | None,
        expected_document_content_mode: str,
        actor_user_id: uuid.UUID | None,
    ) -> None:
        """Delete one comment after transactional parent authorization."""
        ...

    async def aclose(self) -> None:
        """Release backing resources at application shutdown."""
        ...
