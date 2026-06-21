"""Postgres-backed editor-persistence store (M6b durable project tier).

Documents, folders, and comments persist relationally, scoped per
``(tenant_id, created_by_sub, workspace_id)`` with RLS + ``tenant_session``
+ ON CONFLICT autosave — identical discipline to ``chat_postgres.py``.

Editor specifics:

* ``list_documents_page`` SELECTs metadata columns only (NOT the heavy
  ``content_markdown`` body); ``get_document`` SELECTs the full row with the
  body (load-on-open).
* Comments are a diffed collection (upsert + delete + list), not append-only.

The engine is its own NullPool engine (loop-agnostic), awaited from the
HTTP loop — the same rule the chat/knowledge stores document.
"""

from __future__ import annotations

from sqlalchemy import delete, select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from inqtrix.pagination import encode_cursor
from inqtrix.project.base_session_store import (
    BaseSessionStore,
    DEFAULT_TENANT as _DEFAULT_TENANT,
)
from inqtrix.project.editor_ports import (
    DocumentNotFound,
    EditorComment,
    EditorDocument,
    EditorFolder,
)
from inqtrix.storage.editor_orm import (
    editor_comments,
    editor_documents,
    editor_folders,
)

# Document metadata columns (everything EXCEPT the heavy content_markdown
# body) for the list path — the body is transferred only on get_document.
_DOC_META_COLUMNS = (
    editor_documents.c.id,
    editor_documents.c.tenant_id,
    editor_documents.c.created_by_sub,
    editor_documents.c.workspace_id,
    editor_documents.c.title,
    editor_documents.c.folder_id,
    editor_documents.c.source,
    editor_documents.c.source_run_id,
    editor_documents.c.revision,
    editor_documents.c.diff_anchor_markdown,
    editor_documents.c.diff_anchor_updated_at,
    editor_documents.c.created_at,
    editor_documents.c.updated_at,
)


class PostgresEditorStore(BaseSessionStore):
    """Durable :class:`~inqtrix.project.editor_ports.EditorStore` over Postgres.

    Inherits the dedicated engine + tenant-scoped session lifecycle from
    :class:`~inqtrix.project.base_session_store.BaseSessionStore`.
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
        stmt = pg_insert(editor_documents).values(
            id=id,
            tenant_id=_DEFAULT_TENANT,
            created_by_sub=created_by_sub,
            workspace_id=workspace_id,
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
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[editor_documents.c.id],
            set_={
                "title": stmt.excluded.title,
                "content_markdown": stmt.excluded.content_markdown,
                "folder_id": stmt.excluded.folder_id,
                "source": stmt.excluded.source,
                "source_run_id": stmt.excluded.source_run_id,
                "revision": stmt.excluded.revision,
                "diff_anchor_markdown": stmt.excluded.diff_anchor_markdown,
                "diff_anchor_updated_at": stmt.excluded.diff_anchor_updated_at,
                "updated_at": stmt.excluded.updated_at,
            },
        ).returning(editor_documents)
        async with self._session() as session:
            row = (await session.execute(stmt)).one()
        return self._document_from_row(row)

    async def list_documents_page(
        self,
        *,
        created_by_sub: str | None,
        workspace_id: str | None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[EditorDocument], str | None]:
        query = select(*_DOC_META_COLUMNS).where(
            editor_documents.c.tenant_id == _DEFAULT_TENANT
        )
        if created_by_sub is not None:
            query = query.where(editor_documents.c.created_by_sub == created_by_sub)
        if workspace_id is not None:
            query = query.where(editor_documents.c.workspace_id == workspace_id)
        if after is not None:
            query = query.where(
                tuple_(editor_documents.c.created_at, editor_documents.c.id)
                < tuple_(after[0], after[1])
            )
        query = query.order_by(
            editor_documents.c.created_at.desc(),
            editor_documents.c.id.desc(),
        ).limit(limit + 1)
        async with self._session() as session:
            rows = (await session.execute(query)).all()
        documents = [self._document_from_row(row) for row in rows[:limit]]
        next_cursor = (
            encode_cursor(documents[-1].created_at, documents[-1].id)
            if len(rows) > limit and documents
            else None
        )
        return documents, next_cursor

    async def get_document(self, document_id: str) -> EditorDocument:
        async with self._session() as session:
            row = (
                await session.execute(
                    select(editor_documents).where(
                        editor_documents.c.tenant_id == _DEFAULT_TENANT,
                        editor_documents.c.id == document_id,
                    )
                )
            ).first()
        if row is None:
            raise DocumentNotFound(document_id)
        return self._document_from_row(row)

    async def delete_document(self, document_id: str) -> None:
        async with self._session() as session:
            await session.execute(
                delete(editor_documents).where(
                    editor_documents.c.tenant_id == _DEFAULT_TENANT,
                    editor_documents.c.id == document_id,
                )
            )

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
        stmt = pg_insert(editor_folders).values(
            id=id,
            tenant_id=_DEFAULT_TENANT,
            created_by_sub=created_by_sub,
            workspace_id=workspace_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[editor_folders.c.id],
            set_={
                "title": stmt.excluded.title,
                "updated_at": stmt.excluded.updated_at,
            },
        ).returning(editor_folders)
        async with self._session() as session:
            row = (await session.execute(stmt)).one()
        return self._folder_from_row(row)

    async def list_folders(
        self,
        *,
        created_by_sub: str | None,
        workspace_id: str | None,
    ) -> list[EditorFolder]:
        query = select(editor_folders).where(
            editor_folders.c.tenant_id == _DEFAULT_TENANT
        )
        if created_by_sub is not None:
            query = query.where(editor_folders.c.created_by_sub == created_by_sub)
        if workspace_id is not None:
            query = query.where(editor_folders.c.workspace_id == workspace_id)
        query = query.order_by(
            editor_folders.c.created_at.desc(),
            editor_folders.c.id.desc(),
        )
        async with self._session() as session:
            rows = (await session.execute(query)).all()
        return [self._folder_from_row(row) for row in rows]

    async def delete_folder(self, folder_id: str) -> None:
        async with self._session() as session:
            await session.execute(
                delete(editor_folders).where(
                    editor_folders.c.tenant_id == _DEFAULT_TENANT,
                    editor_folders.c.id == folder_id,
                )
            )

    # -- comments --------------------------------------------------------- #

    async def upsert_comments(
        self, comments: list[EditorComment]
    ) -> list[EditorComment]:
        if not comments:
            return []
        values = [
            {
                "id": comment.id,
                "document_id": comment.document_id,
                "tenant_id": _DEFAULT_TENANT,
                "comment_markdown": comment.comment_markdown,
                "anchor": dict(comment.anchor),
                "kind": comment.kind,
                "status": comment.status,
                "evidence_preset": comment.evidence_preset,
                "created_at": comment.created_at,
                "updated_at": comment.updated_at,
            }
            for comment in comments
        ]
        stmt = pg_insert(editor_comments).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[editor_comments.c.document_id, editor_comments.c.id],
            set_={
                "comment_markdown": stmt.excluded.comment_markdown,
                "anchor": stmt.excluded.anchor,
                "kind": stmt.excluded.kind,
                "status": stmt.excluded.status,
                "evidence_preset": stmt.excluded.evidence_preset,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        async with self._session() as session:
            await session.execute(stmt)
        return comments

    async def list_comments_page(
        self,
        document_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[EditorComment], str | None]:
        query = select(editor_comments).where(
            editor_comments.c.tenant_id == _DEFAULT_TENANT,
            editor_comments.c.document_id == document_id,
        )
        if after is not None:
            query = query.where(
                tuple_(editor_comments.c.created_at, editor_comments.c.id)
                < tuple_(after[0], after[1])
            )
        query = query.order_by(
            editor_comments.c.created_at.desc(),
            editor_comments.c.id.desc(),
        ).limit(limit + 1)
        async with self._session() as session:
            rows = (await session.execute(query)).all()
        items = [self._comment_from_row(row) for row in rows[:limit]]
        next_cursor = (
            encode_cursor(items[-1].created_at, items[-1].id)
            if len(rows) > limit and items
            else None
        )
        return items, next_cursor

    async def delete_comment(self, *, document_id: str, comment_id: str) -> None:
        async with self._session() as session:
            await session.execute(
                delete(editor_comments).where(
                    editor_comments.c.tenant_id == _DEFAULT_TENANT,
                    editor_comments.c.document_id == document_id,
                    editor_comments.c.id == comment_id,
                )
            )

    # -- row mapping ------------------------------------------------------ #

    @staticmethod
    def _document_from_row(row) -> EditorDocument:
        # content_markdown is absent on metadata-only (list) rows -> "".
        return EditorDocument(
            id=row.id,
            title=row.title,
            content_markdown=getattr(row, "content_markdown", "") or "",
            folder_id=row.folder_id,
            source=row.source,
            source_run_id=row.source_run_id,
            revision=row.revision,
            diff_anchor_markdown=row.diff_anchor_markdown,
            diff_anchor_updated_at=row.diff_anchor_updated_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
            tenant_id=row.tenant_id,
            created_by_sub=row.created_by_sub,
            workspace_id=row.workspace_id,
        )

    @staticmethod
    def _folder_from_row(row) -> EditorFolder:
        return EditorFolder(
            id=row.id,
            title=row.title,
            created_at=row.created_at,
            updated_at=row.updated_at,
            tenant_id=row.tenant_id,
            created_by_sub=row.created_by_sub,
            workspace_id=row.workspace_id,
        )

    @staticmethod
    def _comment_from_row(row) -> EditorComment:
        return EditorComment(
            id=row.id,
            document_id=row.document_id,
            comment_markdown=row.comment_markdown,
            anchor=dict(row.anchor or {}),
            kind=row.kind,
            status=row.status,
            evidence_preset=row.evidence_preset,
            created_at=row.created_at,
            updated_at=row.updated_at,
            tenant_id=row.tenant_id,
        )
