"""In-memory editor-persistence store (the tier without Postgres).

The fallback when ``INQTRIX_STORAGE_BACKEND`` is not ``postgres`` and the
offline test backend for the port contract. Mirrors the visibility and
keyset semantics of :class:`~inqtrix.project.editor_postgres.PostgresEditorStore`
byte-for-byte (filter before slice; reuse :func:`~inqtrix.pagination.keyset_page`),
INCLUDING that ``list_documents_page`` returns documents with an empty body
(the body is loaded only via :meth:`get_document`). Process-local and not
durable.
"""

from __future__ import annotations

from dataclasses import replace

from inqtrix.pagination import keyset_page
from inqtrix.project.editor_ports import (
    DocumentNotFound,
    DocumentRevisionConflict,
    EditorComment,
    EditorDocument,
    EditorFolder,
)


class MemoryEditorStore:
    """Process-local :class:`~inqtrix.project.editor_ports.EditorStore`."""

    def __init__(self) -> None:
        self._documents: dict[str, EditorDocument] = {}
        self._folders: dict[str, EditorFolder] = {}
        # Keyed by the COMPOSITE (document_id, id), mirroring the Postgres PK.
        self._comments: dict[tuple[str, str], EditorComment] = {}

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
        existing = self._documents.get(id)
        if existing is not None:
            if existing.revision != revision - 1:
                # Revision CAS, wire-identical to the Postgres store (A2):
                # the stored revision must be EXACTLY the writer's base
                # (incoming is base+1). A stale base — a writer that never
                # saw a concurrent agent patch or peer edit — conflicts and
                # rebases instead of clobbering with a higher counter.
                raise DocumentRevisionConflict(
                    current_revision=existing.revision,
                    expected_revision=revision - 1,
                )
            document = replace(
                existing,
                title=title,
                content_markdown=content_markdown,
                folder_id=folder_id,
                source=source,
                source_run_id=source_run_id,
                revision=revision,
                diff_anchor_markdown=diff_anchor_markdown,
                diff_anchor_updated_at=diff_anchor_updated_at,
                updated_at=updated_at,
            )
        else:
            document = EditorDocument(
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
                created_by_sub=created_by_sub,
                workspace_id=workspace_id,
            )
        self._documents[id] = document
        return document

    async def list_documents_page(
        self,
        *,
        created_by_sub: str | None,
        workspace_id: str | None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[EditorDocument], str | None]:
        items = list(self._documents.values())
        if created_by_sub is not None:
            items = [d for d in items if d.created_by_sub == created_by_sub]
        if workspace_id is not None:
            items = [d for d in items if d.workspace_id == workspace_id]
        items.sort(key=lambda d: (d.created_at, d.id), reverse=True)
        page, cursor = keyset_page(
            items,
            limit=limit,
            after=after,
            created_at_of=lambda d: d.created_at,
            id_of=lambda d: d.id,
        )
        # Metadata only — the body loads via get_document.
        return [replace(d, content_markdown="") for d in page], cursor

    async def get_document(self, document_id: str) -> EditorDocument:
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise DocumentNotFound(document_id) from exc

    async def delete_document(self, document_id: str) -> None:
        self._documents.pop(document_id, None)
        self._comments = {
            key: comment
            for key, comment in self._comments.items()
            if comment.document_id != document_id
        }

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
        existing = self._folders.get(id)
        if existing is not None:
            folder = replace(existing, title=title, updated_at=updated_at)
        else:
            folder = EditorFolder(
                id=id,
                title=title,
                created_at=created_at,
                updated_at=updated_at,
                created_by_sub=created_by_sub,
                workspace_id=workspace_id,
            )
        self._folders[id] = folder
        return folder

    async def list_folders(
        self,
        *,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> list[EditorFolder]:
        items = list(self._folders.values())
        if created_by_sub is not None:
            items = [f for f in items if f.created_by_sub == created_by_sub]
        if workspace_id is not None:
            items = [f for f in items if f.workspace_id == workspace_id]
        items.sort(key=lambda f: (f.created_at, f.id), reverse=True)
        return items

    async def delete_folder(self, folder_id: str) -> None:
        self._folders.pop(folder_id, None)
        for did, document in list(self._documents.items()):
            if document.folder_id == folder_id:
                self._documents[did] = replace(document, folder_id=None)

    # -- comments --------------------------------------------------------- #

    async def upsert_comments(
        self, comments: list[EditorComment]
    ) -> list[EditorComment]:
        stored: list[EditorComment] = []
        for comment in comments:
            key = (comment.document_id, comment.id)
            existing = self._comments.get(key)
            if existing is not None:
                merged = replace(
                    existing,
                    comment_markdown=comment.comment_markdown,
                    anchor=dict(comment.anchor),
                    kind=comment.kind,
                    status=comment.status,
                    evidence_preset=comment.evidence_preset,
                    updated_at=comment.updated_at,
                )
            else:
                merged = replace(comment, anchor=dict(comment.anchor))
            self._comments[key] = merged
            stored.append(merged)
        return stored

    async def list_comments_page(
        self,
        document_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[EditorComment], str | None]:
        items = [c for c in self._comments.values() if c.document_id == document_id]
        items.sort(key=lambda c: (c.created_at, c.id), reverse=True)
        return keyset_page(
            items,
            limit=limit,
            after=after,
            created_at_of=lambda c: c.created_at,
            id_of=lambda c: c.id,
        )

    async def delete_comment(self, *, document_id: str, comment_id: str) -> None:
        self._comments.pop((document_id, comment_id), None)

    async def aclose(self) -> None:
        return None
