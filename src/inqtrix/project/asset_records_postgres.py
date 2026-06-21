"""Postgres-backed file-asset-record store (M6c durable project tier).

Sections, groups, and asset records persist relationally, scoped per
``(tenant_id, created_by_sub, workspace_id)`` with RLS + the inherited
tenant-session lifecycle (:class:`BaseSessionStore`). Like editor
documents, ``list_assets_page`` SELECTs metadata columns only (NOT the
heavy ``extracted_text``); ``get_asset`` SELECTs the full row.
"""

from __future__ import annotations

from sqlalchemy import delete, select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from inqtrix.pagination import encode_cursor
from inqtrix.project.base_session_store import (
    BaseSessionStore,
    DEFAULT_TENANT as _DEFAULT_TENANT,
)
from inqtrix.project.asset_records_ports import (
    AssetGroup,
    AssetNotFound,
    AssetRecord,
    AssetSection,
)
from inqtrix.storage.asset_records_orm import (
    asset_groups,
    asset_records,
    asset_sections,
)

# Asset metadata columns (everything EXCEPT the heavy extracted_text) for
# the list path — the text is transferred only on get_asset.
_ASSET_META_COLUMNS = (
    asset_records.c.id,
    asset_records.c.tenant_id,
    asset_records.c.created_by_sub,
    asset_records.c.workspace_id,
    asset_records.c.section_id,
    asset_records.c.group_id,
    asset_records.c.title,
    asset_records.c.label,
    asset_records.c.file_name,
    asset_records.c.mime_type,
    asset_records.c.origin,
    asset_records.c.page_count,
    asset_records.c.parse_status,
    asset_records.c.parse_warning,
    asset_records.c.text_truncated,
    asset_records.c.size_bytes,
    asset_records.c.server_file_id,
    asset_records.c.parser_id,
    asset_records.c.created_at,
    asset_records.c.updated_at,
)


class PostgresAssetStore(BaseSessionStore):
    """Durable :class:`~inqtrix.project.asset_records_ports.AssetStore`.

    Inherits the engine + tenant-session lifecycle from
    :class:`~inqtrix.project.base_session_store.BaseSessionStore`.
    """

    # -- sections --------------------------------------------------------- #

    async def upsert_section(
        self, *, id, kind, title, created_at, updated_at, created_by_sub, workspace_id
    ) -> AssetSection:
        stmt = _with_set(pg_insert(asset_sections), asset_sections, dict(
            id=id, tenant_id=_DEFAULT_TENANT, created_by_sub=created_by_sub,
            workspace_id=workspace_id, kind=kind, title=title,
            created_at=created_at, updated_at=updated_at,
        ), ["kind", "title", "updated_at"]).returning(asset_sections)
        async with self._session() as session:
            row = (await session.execute(stmt)).one()
        return self._section_from_row(row)

    async def list_sections(self, *, created_by_sub, workspace_id) -> list[AssetSection]:
        query = _scoped_query(select(asset_sections), asset_sections, created_by_sub, workspace_id)
        query = query.order_by(asset_sections.c.created_at.desc(), asset_sections.c.id.desc())
        async with self._session() as session:
            rows = (await session.execute(query)).all()
        return [self._section_from_row(row) for row in rows]

    async def delete_section(self, section_id: str) -> None:
        async with self._session() as session:
            await session.execute(delete(asset_sections).where(
                asset_sections.c.tenant_id == _DEFAULT_TENANT,
                asset_sections.c.id == section_id,
            ))

    # -- groups ----------------------------------------------------------- #

    async def upsert_group(
        self, *, id, section_id, title, created_at, updated_at, created_by_sub, workspace_id
    ) -> AssetGroup:
        stmt = _with_set(pg_insert(asset_groups), asset_groups, dict(
            id=id, tenant_id=_DEFAULT_TENANT, created_by_sub=created_by_sub,
            workspace_id=workspace_id, section_id=section_id, title=title,
            created_at=created_at, updated_at=updated_at,
        ), ["section_id", "title", "updated_at"]).returning(asset_groups)
        async with self._session() as session:
            row = (await session.execute(stmt)).one()
        return self._group_from_row(row)

    async def list_groups(self, *, created_by_sub, workspace_id) -> list[AssetGroup]:
        query = _scoped_query(select(asset_groups), asset_groups, created_by_sub, workspace_id)
        query = query.order_by(asset_groups.c.created_at.desc(), asset_groups.c.id.desc())
        async with self._session() as session:
            rows = (await session.execute(query)).all()
        return [self._group_from_row(row) for row in rows]

    async def delete_group(self, group_id: str) -> None:
        async with self._session() as session:
            await session.execute(delete(asset_groups).where(
                asset_groups.c.tenant_id == _DEFAULT_TENANT,
                asset_groups.c.id == group_id,
            ))

    # -- assets ----------------------------------------------------------- #

    async def upsert_asset(
        self, *, id, section_id, group_id, title, label, file_name, mime_type,
        origin, page_count, parse_status, parse_warning, text_truncated,
        size_bytes, server_file_id, parser_id=None, extracted_text, created_at,
        updated_at, created_by_sub, workspace_id,
    ) -> AssetRecord:
        values = dict(
            id=id, tenant_id=_DEFAULT_TENANT, created_by_sub=created_by_sub,
            workspace_id=workspace_id, section_id=section_id, group_id=group_id,
            title=title, label=label, file_name=file_name, mime_type=mime_type,
            origin=origin, page_count=page_count, parse_status=parse_status,
            parse_warning=parse_warning, text_truncated=1 if text_truncated else 0,
            size_bytes=size_bytes, server_file_id=server_file_id,
            parser_id=parser_id, extracted_text=extracted_text,
            created_at=created_at, updated_at=updated_at,
        )
        mutable = ["section_id", "group_id", "title", "label", "file_name",
                   "mime_type", "origin", "page_count", "parse_status",
                   "parse_warning", "text_truncated", "size_bytes",
                   "server_file_id", "parser_id", "extracted_text", "updated_at"]
        stmt = _with_set(pg_insert(asset_records), asset_records, values, mutable).returning(asset_records)
        async with self._session() as session:
            row = (await session.execute(stmt)).one()
        return self._asset_from_row(row)

    async def list_assets_page(
        self, *, created_by_sub, workspace_id, limit, after
    ) -> tuple[list[AssetRecord], str | None]:
        query = _scoped_query(select(*_ASSET_META_COLUMNS), asset_records, created_by_sub, workspace_id)
        if after is not None:
            query = query.where(
                tuple_(asset_records.c.created_at, asset_records.c.id)
                < tuple_(after[0], after[1])
            )
        query = query.order_by(
            asset_records.c.created_at.desc(), asset_records.c.id.desc()
        ).limit(limit + 1)
        async with self._session() as session:
            rows = (await session.execute(query)).all()
        assets = [self._asset_from_row(row) for row in rows[:limit]]
        next_cursor = (
            encode_cursor(assets[-1].created_at, assets[-1].id)
            if len(rows) > limit and assets else None
        )
        return assets, next_cursor

    async def get_asset(self, asset_id: str) -> AssetRecord:
        async with self._session() as session:
            row = (await session.execute(select(asset_records).where(
                asset_records.c.tenant_id == _DEFAULT_TENANT,
                asset_records.c.id == asset_id,
            ))).first()
        if row is None:
            raise AssetNotFound(asset_id)
        return self._asset_from_row(row)

    async def delete_asset(self, asset_id: str) -> None:
        async with self._session() as session:
            await session.execute(delete(asset_records).where(
                asset_records.c.tenant_id == _DEFAULT_TENANT,
                asset_records.c.id == asset_id,
            ))

    # -- row mapping ------------------------------------------------------ #

    @staticmethod
    def _section_from_row(row) -> AssetSection:
        return AssetSection(
            id=row.id, kind=row.kind, title=row.title, created_at=row.created_at,
            updated_at=row.updated_at, tenant_id=row.tenant_id,
            created_by_sub=row.created_by_sub, workspace_id=row.workspace_id,
        )

    @staticmethod
    def _group_from_row(row) -> AssetGroup:
        return AssetGroup(
            id=row.id, section_id=row.section_id, title=row.title,
            created_at=row.created_at, updated_at=row.updated_at,
            tenant_id=row.tenant_id, created_by_sub=row.created_by_sub,
            workspace_id=row.workspace_id,
        )

    @staticmethod
    def _asset_from_row(row) -> AssetRecord:
        return AssetRecord(
            id=row.id, section_id=row.section_id, group_id=row.group_id,
            title=row.title, label=row.label, file_name=row.file_name,
            mime_type=row.mime_type, origin=row.origin, page_count=row.page_count,
            parse_status=row.parse_status, parse_warning=row.parse_warning,
            text_truncated=bool(row.text_truncated), size_bytes=row.size_bytes,
            server_file_id=row.server_file_id,
            parser_id=row.parser_id,
            extracted_text=getattr(row, "extracted_text", "") or "",
            created_at=row.created_at, updated_at=row.updated_at,
            tenant_id=row.tenant_id, created_by_sub=row.created_by_sub,
            workspace_id=row.workspace_id,
        )


def _scoped_query(query, table, created_by_sub, workspace_id):
    query = query.where(table.c.tenant_id == _DEFAULT_TENANT)
    if created_by_sub is not None:
        query = query.where(table.c.created_by_sub == created_by_sub)
    if workspace_id is not None:
        query = query.where(table.c.workspace_id == workspace_id)
    return query


def _with_set(stmt, table, values, mutable_columns):
    """A values()+on_conflict_do_update(set_=mutable) upsert keyed on the PK,
    never reassigning created_at / ownership."""
    stmt = stmt.values(**values)
    return stmt.on_conflict_do_update(
        index_elements=[table.c.id],
        set_={col: getattr(stmt.excluded, col) for col in mutable_columns},
    )
