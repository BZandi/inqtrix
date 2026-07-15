"""In-memory file-asset-record store (tier without Postgres + offline test).

Mirrors :class:`~inqtrix.project.asset_records_postgres.PostgresAssetStore`
byte-for-byte (filter before slice; reuse keyset_page; ``list_assets_page``
returns an empty body). Process-local, not durable.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

from inqtrix.pagination import keyset_page
from inqtrix.project.asset_records_ports import (
    AssetGroup,
    AssetNotFound,
    AssetRecord,
    AssetSection,
    GroupNotFound,
    SectionNotFound,
)
from inqtrix.project.scoped_upsert import ResourceScope, require_memory_scope


class MemoryAssetStore:
    """Process-local :class:`~inqtrix.project.asset_records_ports.AssetStore`."""

    def __init__(self) -> None:
        self._sections: dict[str, AssetSection] = {}
        self._groups: dict[str, AssetGroup] = {}
        self._assets: dict[str, AssetRecord] = {}

    # -- sections --------------------------------------------------------- #

    async def upsert_section(
        self, *, id, kind, title, created_at, updated_at,
        created_by_user_id: uuid.UUID | None, workspace_id
    ) -> AssetSection:
        existing = self._sections.get(id)
        if existing is not None:
            require_memory_scope(
                existing,
                created_by_user_id=created_by_user_id,
                workspace_id=workspace_id,
                resource_id=id,
                not_found=SectionNotFound,
            )
            section = replace(existing, kind=kind, title=title, updated_at=updated_at)
        else:
            section = AssetSection(
                id=id, kind=kind, title=title, created_at=created_at,
                updated_at=updated_at, created_by_user_id=created_by_user_id,
                workspace_id=workspace_id,
            )
        self._sections[id] = section
        return section

    async def list_sections(
        self, *, created_by_user_id: uuid.UUID | None, workspace_id
    ) -> list[AssetSection]:
        items = _scoped(self._sections.values(), created_by_user_id, workspace_id)
        items.sort(key=lambda s: (s.created_at, s.id), reverse=True)
        return items

    async def delete_section(
        self, section_id: str, *, scope: ResourceScope
    ) -> None:
        require_memory_scope(
            self._sections.get(section_id),
            created_by_user_id=scope.created_by_user_id,
            workspace_id=scope.workspace_id,
            resource_id=section_id,
            not_found=SectionNotFound,
        )
        self._sections.pop(section_id, None)
        self._groups = {
            gid: g for gid, g in self._groups.items() if g.section_id != section_id
        }
        self._assets = {
            aid: a for aid, a in self._assets.items() if a.section_id != section_id
        }

    # -- groups ----------------------------------------------------------- #

    async def upsert_group(
        self, *, id, section_id, title, created_at, updated_at,
        created_by_user_id: uuid.UUID | None, workspace_id
    ) -> AssetGroup:
        require_memory_scope(
            self._sections.get(section_id),
            created_by_user_id=created_by_user_id,
            workspace_id=workspace_id,
            resource_id=section_id,
            not_found=SectionNotFound,
        )
        existing = self._groups.get(id)
        if existing is not None:
            require_memory_scope(
                existing,
                created_by_user_id=created_by_user_id,
                workspace_id=workspace_id,
                resource_id=id,
                not_found=GroupNotFound,
            )
            group = replace(existing, section_id=section_id, title=title, updated_at=updated_at)
        else:
            group = AssetGroup(
                id=id, section_id=section_id, title=title, created_at=created_at,
                updated_at=updated_at, created_by_user_id=created_by_user_id,
                workspace_id=workspace_id,
            )
        self._groups[id] = group
        return group

    async def list_groups(
        self, *, created_by_user_id: uuid.UUID | None, workspace_id
    ) -> list[AssetGroup]:
        items = _scoped(self._groups.values(), created_by_user_id, workspace_id)
        items.sort(key=lambda g: (g.created_at, g.id), reverse=True)
        return items

    async def delete_group(
        self, group_id: str, *, scope: ResourceScope
    ) -> None:
        require_memory_scope(
            self._groups.get(group_id),
            created_by_user_id=scope.created_by_user_id,
            workspace_id=scope.workspace_id,
            resource_id=group_id,
            not_found=GroupNotFound,
        )
        self._groups.pop(group_id, None)
        for aid, asset in list(self._assets.items()):
            if asset.group_id == group_id:
                self._assets[aid] = replace(asset, group_id=None)

    # -- assets ----------------------------------------------------------- #

    async def upsert_asset(
        self, *, id, section_id, group_id, title, label, file_name, mime_type,
        origin, page_count, parse_status, parse_warning, text_truncated,
        size_bytes, server_file_id, parser_id=None, extracted_text, created_at,
        updated_at, created_by_user_id: uuid.UUID | None, workspace_id,
    ) -> AssetRecord:
        require_memory_scope(
            self._sections.get(section_id),
            created_by_user_id=created_by_user_id,
            workspace_id=workspace_id,
            resource_id=section_id,
            not_found=SectionNotFound,
        )
        if group_id is not None:
            group = require_memory_scope(
                self._groups.get(group_id),
                created_by_user_id=created_by_user_id,
                workspace_id=workspace_id,
                resource_id=group_id,
                not_found=GroupNotFound,
            )
            if group.section_id != section_id:
                raise GroupNotFound(group_id)
        existing = self._assets.get(id)
        mutable = dict(
            section_id=section_id, group_id=group_id, title=title, label=label,
            file_name=file_name, mime_type=mime_type, origin=origin,
            page_count=page_count, parse_status=parse_status,
            parse_warning=parse_warning, text_truncated=text_truncated,
            size_bytes=size_bytes, server_file_id=server_file_id,
            parser_id=parser_id, extracted_text=extracted_text,
            updated_at=updated_at,
        )
        if existing is not None:
            require_memory_scope(
                existing,
                created_by_user_id=created_by_user_id,
                workspace_id=workspace_id,
                resource_id=id,
                not_found=AssetNotFound,
            )
            asset = replace(existing, **mutable)
        else:
            asset = AssetRecord(
                id=id, created_at=created_at, created_by_user_id=created_by_user_id,
                workspace_id=workspace_id, **mutable,
            )
        self._assets[id] = asset
        return asset

    async def list_assets_page(
        self, *, created_by_user_id: uuid.UUID | None, workspace_id, limit, after
    ) -> tuple[list[AssetRecord], str | None]:
        items = _scoped(self._assets.values(), created_by_user_id, workspace_id)
        items.sort(key=lambda a: (a.created_at, a.id), reverse=True)
        page, cursor = keyset_page(
            items, limit=limit, after=after,
            created_at_of=lambda a: a.created_at, id_of=lambda a: a.id,
        )
        return [replace(a, extracted_text="") for a in page], cursor

    async def get_asset(self, asset_id: str) -> AssetRecord:
        try:
            return self._assets[asset_id]
        except KeyError as exc:
            raise AssetNotFound(asset_id) from exc

    async def delete_asset(
        self, asset_id: str, *, scope: ResourceScope
    ) -> None:
        require_memory_scope(
            self._assets.get(asset_id),
            created_by_user_id=scope.created_by_user_id,
            workspace_id=scope.workspace_id,
            resource_id=asset_id,
            not_found=AssetNotFound,
        )
        self._assets.pop(asset_id, None)

    async def aclose(self) -> None:
        return None


def _scoped(values, created_by_user_id: uuid.UUID | None, workspace_id):
    items = list(values)
    if created_by_user_id is not None:
        items = [i for i in items if i.created_by_user_id == created_by_user_id]
    if workspace_id is not None:
        items = [i for i in items if i.workspace_id == workspace_id]
    return items
