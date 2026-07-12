"""Editor-persistence service (M6b project tier).

The editor counterpart of
:class:`~inqtrix.services.chat_history_service.ChatHistoryService`: payload
validation, the owner/share access rule (the shared
:func:`~inqtrix.auth.permissions.resolve_owned_access`), and the
"which documents belong to this caller" resolution before the store's
keyset query. Persistence only — it never calls a model.

Ownership model (identical to chat): documents/folders carry
``created_by_sub``; ``None`` (anonymous/static principals) stays visible
to all. Documents are private per-user in M6b (no sharing surface);
comments inherit the parent document's visibility. Saving through a share
needs at least an edit grant; deleting a document/folder stays owner-only.
Every denial is the indistinct :class:`DocumentNotFound`/
:class:`FolderNotFound`.
"""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

from inqtrix.auth.permissions import SharePermission, resolve_owned_access
from inqtrix.services.workspace_guard import deny_cross_workspace
from inqtrix.project.editor_ports import (
    DocumentNotFound,
    EditorComment,
    EditorDocument,
    EditorFolder,
    EditorStore,
    FolderNotFound,
)

if TYPE_CHECKING:
    from inqtrix.auth.principal import UserContext

_VALID_SOURCES = frozenset(
    {"blank", "imported-research-report", "pasted", "agent-artifact"}
)
_VALID_COMMENT_KINDS = frozenset({"collect", "inline_edit", "evidence_review"})
_VALID_COMMENT_STATUSES = frozenset({"open", "resolved", "stale"})
_VALID_EVIDENCE_PRESETS = frozenset({"add_sources", "fact_check", "verify_citations"})


class EditorValidationError(ValueError):
    """Raised for client-payload problems (maps to HTTP 400).

    The source / comment kind / status / evidence-preset domains are
    rejected here, before the database CHECK constraint, so a bad value is
    a clean 400 instead of an opaque 500 (No Silent Fallbacks)."""


class EditorPersistenceService:
    """Application service over an :class:`EditorStore`."""

    def __init__(self, *, store: EditorStore, durable: bool = False) -> None:
        self._store = store
        self._durable = durable

    @property
    def store(self) -> EditorStore:
        """The wired editor store (shutdown disposes its engine)."""
        return self._store

    @property
    def durable(self) -> bool:
        """Whether the backing store survives a restart."""
        return self._durable

    # -- documents -------------------------------------------------------- #

    async def save_document(
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
        caller_sub: str | None,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> EditorDocument:
        """Create or update a document (idempotent autosave).

        A new id is owned by the caller; an existing id requires at least
        an edit grant and keeps its original owner/workspace.
        """
        if source not in _VALID_SOURCES:
            raise EditorValidationError(f"unknown document source: {source!r}")
        try:
            existing = await self._store.get_document(id)
        except DocumentNotFound:
            existing = None
        if existing is not None:
            shared = resolve_owned_access(
                owner_sub=existing.created_by_sub,
                resource_tenant_id=existing.tenant_id,
                resource_id=existing.id,
                visible_to=visible_to,
                also_visible=also_visible,
                not_found=DocumentNotFound,
            )
            if shared is not None and not shared.at_least(SharePermission.EDIT):
                raise DocumentNotFound(id)
            owner_sub = existing.created_by_sub
            owner_workspace = existing.workspace_id
        else:
            owner_sub = caller_sub
            owner_workspace = workspace_id
        return await self._store.upsert_document(
            id=id,
            title=title,
            content_markdown=content_markdown,
            folder_id=folder_id,
            source=source,
            source_run_id=source_run_id,
            revision=revision,
            diff_anchor_markdown=diff_anchor_markdown,
            diff_anchor_updated_at=diff_anchor_updated_at,
            created_at=created_at,
            updated_at=updated_at,
            created_by_sub=owner_sub,
            workspace_id=owner_workspace,
        )

    async def list_documents(
        self,
        *,
        caller_sub: str | None,
        workspace_id: str | None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[EditorDocument], str | None]:
        """One keyset page of the caller's documents (metadata only)."""
        return await self._store.list_documents_page(
            created_by_sub=caller_sub,
            workspace_id=workspace_id,
            limit=limit,
            after=after,
        )

    async def get_document(
        self,
        document_id: str,
        *,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> EditorDocument:
        """One document (with body) the caller may see, or not-found."""
        document = await self._store.get_document(document_id)
        resolve_owned_access(
            owner_sub=document.created_by_sub,
            resource_tenant_id=document.tenant_id,
            resource_id=document.id,
            visible_to=visible_to,
            also_visible=also_visible,
            not_found=DocumentNotFound,
        )
        return document

    async def delete_document(
        self,
        document_id: str,
        *,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
        request_workspace_id: str | None = None,
    ) -> None:
        """Delete a document (owner-only) with its comments (cascade)."""
        document = await self._store.get_document(document_id)
        shared = resolve_owned_access(
            owner_sub=document.created_by_sub,
            resource_tenant_id=document.tenant_id,
            resource_id=document.id,
            visible_to=visible_to,
            also_visible=also_visible,
            not_found=DocumentNotFound,
        )
        if shared is not None:
            raise DocumentNotFound(document_id)
        deny_cross_workspace(
            resource_workspace_id=document.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: DocumentNotFound(document_id),
        )
        await self._store.delete_document(document_id)

    # -- folders ---------------------------------------------------------- #

    async def save_folder(
        self,
        *,
        id: str,
        title: str,
        created_at: float,
        updated_at: float,
        caller_sub: str | None,
        workspace_id: str | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> EditorFolder:
        """Create or update a folder (idempotent)."""
        existing = None
        for folder in await self._store.list_folders(
            created_by_sub=None, workspace_id=None
        ):
            if folder.id == id:
                existing = folder
                break
        if existing is not None:
            resolve_owned_access(
                owner_sub=existing.created_by_sub,
                resource_tenant_id=existing.tenant_id,
                resource_id=existing.id,
                visible_to=visible_to,
                also_visible=also_visible,
                not_found=FolderNotFound,
            )
            owner_sub = existing.created_by_sub
            owner_workspace = existing.workspace_id
        else:
            owner_sub = caller_sub
            owner_workspace = workspace_id
        return await self._store.upsert_folder(
            id=id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            created_by_sub=owner_sub,
            workspace_id=owner_workspace,
        )

    async def list_folders(
        self,
        *,
        caller_sub: str | None,
        workspace_id: str | None,
    ) -> list[EditorFolder]:
        """All of the caller's folders (newest first)."""
        return await self._store.list_folders(
            created_by_sub=caller_sub, workspace_id=workspace_id
        )

    async def delete_folder(
        self,
        folder_id: str,
        *,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
        request_workspace_id: str | None = None,
    ) -> None:
        """Delete a folder (its documents orphan to ungrouped)."""
        existing = None
        for folder in await self._store.list_folders(
            created_by_sub=None, workspace_id=None
        ):
            if folder.id == folder_id:
                existing = folder
                break
        if existing is None:
            raise FolderNotFound(folder_id)
        shared = resolve_owned_access(
            owner_sub=existing.created_by_sub,
            resource_tenant_id=existing.tenant_id,
            resource_id=existing.id,
            visible_to=visible_to,
            also_visible=also_visible,
            not_found=FolderNotFound,
        )
        if shared is not None:
            raise FolderNotFound(folder_id)
        deny_cross_workspace(
            resource_workspace_id=existing.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: FolderNotFound(folder_id),
        )
        await self._store.delete_folder(folder_id)

    # -- comments --------------------------------------------------------- #

    async def save_comments(
        self,
        document_id: str,
        *,
        comments: list[dict[str, Any]],
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> list[EditorComment]:
        """Upsert comments into a document the caller may edit."""
        await self._require_document_edit(
            document_id, visible_to=visible_to, also_visible=also_visible
        )
        parsed = [self._parse_comment(document_id, raw) for raw in comments]
        return await self._store.upsert_comments(parsed)

    async def list_comments(
        self,
        document_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[list[EditorComment], str | None]:
        """One keyset page of a readable document's comments."""
        await self.get_document(
            document_id, visible_to=visible_to, also_visible=also_visible
        )
        return await self._store.list_comments_page(
            document_id, limit=limit, after=after
        )

    async def get_document_context(
        self,
        document_id: str,
        *,
        comment_limit: int = 200,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> tuple[EditorDocument, list[EditorComment]]:
        """One document plus its comments in a single access-checked call.

        The read bundle an agent needs to understand a document before
        proposing changes: it saves the two-round-trip dance of
        ``get_document`` + ``list_comments`` and applies the same
        owner/share visibility once. Comments are the newest page up to
        *comment_limit* (documents carry few comments in practice); the
        cursor is intentionally not surfaced here — a caller needing
        pagination uses :meth:`list_comments` directly.
        """
        document = await self.get_document(
            document_id, visible_to=visible_to, also_visible=also_visible
        )
        comments, _cursor = await self._store.list_comments_page(
            document_id, limit=comment_limit, after=None
        )
        return document, comments

    async def delete_comment(
        self,
        document_id: str,
        comment_id: str,
        *,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> None:
        """Delete one comment from a document the caller may edit."""
        await self._require_document_edit(
            document_id, visible_to=visible_to, also_visible=also_visible
        )
        await self._store.delete_comment(
            document_id=document_id, comment_id=comment_id
        )

    # -- helpers ---------------------------------------------------------- #

    async def _require_document_edit(
        self,
        document_id: str,
        *,
        visible_to: "UserContext | None",
        also_visible: "Mapping[str, SharePermission] | None",
    ) -> None:
        document = await self._store.get_document(document_id)
        shared = resolve_owned_access(
            owner_sub=document.created_by_sub,
            resource_tenant_id=document.tenant_id,
            resource_id=document.id,
            visible_to=visible_to,
            also_visible=also_visible,
            not_found=DocumentNotFound,
        )
        if shared is not None and not shared.at_least(SharePermission.EDIT):
            raise DocumentNotFound(document_id)

    @staticmethod
    def _parse_comment(document_id: str, raw: dict[str, Any]) -> EditorComment:
        comment_id = raw.get("id")
        if not isinstance(comment_id, str) or not comment_id:
            raise EditorValidationError("comment id is required")
        kind = raw.get("kind")
        if kind not in _VALID_COMMENT_KINDS:
            raise EditorValidationError(f"unknown comment kind: {kind!r}")
        status = raw.get("status")
        if status not in _VALID_COMMENT_STATUSES:
            raise EditorValidationError(f"unknown comment status: {status!r}")
        evidence_preset = raw.get("evidence_preset")
        if evidence_preset is not None and evidence_preset not in _VALID_EVIDENCE_PRESETS:
            raise EditorValidationError(
                f"unknown evidence preset: {evidence_preset!r}"
            )
        anchor = raw.get("anchor", {})
        if not isinstance(anchor, dict):
            raise EditorValidationError("comment anchor must be an object")
        content = raw.get("comment_markdown", "")
        if not isinstance(content, str):
            raise EditorValidationError("comment_markdown must be a string")
        created_at = raw.get("created_at")
        updated_at = raw.get("updated_at")
        if not isinstance(created_at, (int, float)) or not isinstance(
            updated_at, (int, float)
        ):
            raise EditorValidationError(
                "comment created_at and updated_at must be numbers"
            )
        return EditorComment(
            id=comment_id,
            document_id=document_id,
            comment_markdown=content,
            anchor=dict(anchor),
            kind=kind,
            status=status,
            evidence_preset=evidence_preset,
            created_at=float(created_at),
            updated_at=float(updated_at),
        )
