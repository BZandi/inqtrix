"""File-asset-record persistence service (M6c project tier).

The asset-library counterpart of the chat/editor services: payload
validation, the shared owner/share access rule
(:func:`~inqtrix.auth.permissions.resolve_owned_access`), and the
"which records belong to this caller" resolution before the store query.
Sections/groups/assets are private per-user in M6c (no sharing surface);
all denials are the indistinct not-found.
"""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

from inqtrix.auth.permissions import SharePermission, resolve_owned_access
from inqtrix.services.workspace_guard import deny_cross_workspace
from inqtrix.project.asset_records_ports import (
    AssetGroup,
    AssetNotFound,
    AssetRecord,
    AssetSection,
    AssetStore,
    GroupNotFound,
    SectionNotFound,
)

if TYPE_CHECKING:
    from inqtrix.auth.principal import UserContext

_VALID_KINDS = frozenset({"temporary", "custom"})
_VALID_ORIGINS = frozenset({"chat", "editor", "library"})
_VALID_PARSE = frozenset({"parsed", "partial", "unsupported", "error"})


class AssetValidationError(ValueError):
    """Raised for client-payload problems (maps to HTTP 400)."""


class AssetRecordsService:
    """Application service over an :class:`AssetStore`."""

    def __init__(self, *, store: AssetStore, durable: bool = False) -> None:
        self._store = store
        self._durable = durable

    @property
    def store(self) -> AssetStore:
        return self._store

    @property
    def durable(self) -> bool:
        return self._durable

    # -- sections --------------------------------------------------------- #

    async def save_section(
        self, *, id, kind, title, created_at, updated_at, caller_sub,
        workspace_id, visible_to, also_visible=None,
    ) -> AssetSection:
        if kind not in _VALID_KINDS:
            raise AssetValidationError(f"unknown section kind: {kind!r}")
        existing = _find(await self._store.list_sections(created_by_sub=None, workspace_id=None), id)
        owner_sub, owner_ws = self._owner_for(
            existing, caller_sub, workspace_id, visible_to, also_visible, SectionNotFound
        )
        return await self._store.upsert_section(
            id=id, kind=kind, title=title, created_at=created_at,
            updated_at=updated_at, created_by_sub=owner_sub, workspace_id=owner_ws,
        )

    async def list_sections(self, *, caller_sub, workspace_id) -> list[AssetSection]:
        return await self._store.list_sections(created_by_sub=caller_sub, workspace_id=workspace_id)

    async def delete_section(
        self, section_id, *, visible_to, also_visible=None, request_workspace_id=None
    ) -> None:
        existing = _find(await self._store.list_sections(created_by_sub=None, workspace_id=None), section_id)
        if existing is None:
            raise SectionNotFound(section_id)
        self._require_owner(existing, visible_to, also_visible, SectionNotFound)
        deny_cross_workspace(
            resource_workspace_id=existing.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: SectionNotFound(section_id),
        )
        await self._store.delete_section(section_id)

    # -- groups ----------------------------------------------------------- #

    async def save_group(
        self, *, id, section_id, title, created_at, updated_at, caller_sub,
        workspace_id, visible_to, also_visible=None,
    ) -> AssetGroup:
        existing = _find(await self._store.list_groups(created_by_sub=None, workspace_id=None), id)
        owner_sub, owner_ws = self._owner_for(
            existing, caller_sub, workspace_id, visible_to, also_visible, GroupNotFound
        )
        return await self._store.upsert_group(
            id=id, section_id=section_id, title=title, created_at=created_at,
            updated_at=updated_at, created_by_sub=owner_sub, workspace_id=owner_ws,
        )

    async def list_groups(self, *, caller_sub, workspace_id) -> list[AssetGroup]:
        return await self._store.list_groups(created_by_sub=caller_sub, workspace_id=workspace_id)

    async def delete_group(
        self, group_id, *, visible_to, also_visible=None, request_workspace_id=None
    ) -> None:
        existing = _find(await self._store.list_groups(created_by_sub=None, workspace_id=None), group_id)
        if existing is None:
            raise GroupNotFound(group_id)
        self._require_owner(existing, visible_to, also_visible, GroupNotFound)
        deny_cross_workspace(
            resource_workspace_id=existing.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: GroupNotFound(group_id),
        )
        await self._store.delete_group(group_id)

    # -- assets ----------------------------------------------------------- #

    async def save_asset(
        self, *, id, section_id, group_id, title, label, file_name, mime_type,
        origin, page_count, parse_status, parse_warning, text_truncated,
        size_bytes, server_file_id, extracted_text, created_at,
        updated_at, caller_sub, workspace_id, visible_to, also_visible=None,
        parser_id=None,
    ) -> AssetRecord:
        if origin not in _VALID_ORIGINS:
            raise AssetValidationError(f"unknown asset origin: {origin!r}")
        if parse_status not in _VALID_PARSE:
            raise AssetValidationError(f"unknown parse status: {parse_status!r}")
        try:
            existing = await self._store.get_asset(id)
        except AssetNotFound:
            existing = None
        if existing is not None:
            shared = resolve_owned_access(
                owner_sub=existing.created_by_sub, resource_tenant_id=existing.tenant_id,
                resource_id=existing.id, visible_to=visible_to,
                also_visible=also_visible, not_found=AssetNotFound,
            )
            if shared is not None and not shared.at_least(SharePermission.EDIT):
                raise AssetNotFound(id)
            owner_sub, owner_ws = existing.created_by_sub, existing.workspace_id
        else:
            owner_sub, owner_ws = caller_sub, workspace_id
        return await self._store.upsert_asset(
            id=id, section_id=section_id, group_id=group_id, title=title,
            label=label, file_name=file_name, mime_type=mime_type, origin=origin,
            page_count=page_count, parse_status=parse_status,
            parse_warning=parse_warning, text_truncated=text_truncated,
            size_bytes=size_bytes, server_file_id=server_file_id,
            parser_id=parser_id, extracted_text=extracted_text,
            created_at=created_at, updated_at=updated_at,
            created_by_sub=owner_sub, workspace_id=owner_ws,
        )

    async def list_assets(self, *, caller_sub, workspace_id, limit, after):
        return await self._store.list_assets_page(
            created_by_sub=caller_sub, workspace_id=workspace_id, limit=limit, after=after
        )

    async def get_asset(self, asset_id, *, visible_to, also_visible=None) -> AssetRecord:
        asset = await self._store.get_asset(asset_id)
        resolve_owned_access(
            owner_sub=asset.created_by_sub, resource_tenant_id=asset.tenant_id,
            resource_id=asset.id, visible_to=visible_to, also_visible=also_visible,
            not_found=AssetNotFound,
        )
        return asset

    async def delete_asset(
        self, asset_id, *, visible_to, also_visible=None, request_workspace_id=None
    ) -> None:
        asset = await self._store.get_asset(asset_id)
        shared = resolve_owned_access(
            owner_sub=asset.created_by_sub, resource_tenant_id=asset.tenant_id,
            resource_id=asset.id, visible_to=visible_to, also_visible=also_visible,
            not_found=AssetNotFound,
        )
        if shared is not None:
            raise AssetNotFound(asset_id)
        deny_cross_workspace(
            resource_workspace_id=asset.workspace_id,
            request_workspace_id=request_workspace_id,
            not_found=lambda: AssetNotFound(asset_id),
        )
        await self._store.delete_asset(asset_id)

    # -- helpers ---------------------------------------------------------- #

    def _owner_for(self, existing, caller_sub, workspace_id, visible_to, also_visible, not_found):
        if existing is not None:
            resolve_owned_access(
                owner_sub=existing.created_by_sub, resource_tenant_id=existing.tenant_id,
                resource_id=existing.id, visible_to=visible_to,
                also_visible=also_visible, not_found=not_found,
            )
            return existing.created_by_sub, existing.workspace_id
        return caller_sub, workspace_id

    def _require_owner(self, existing, visible_to, also_visible, not_found):
        shared = resolve_owned_access(
            owner_sub=existing.created_by_sub, resource_tenant_id=existing.tenant_id,
            resource_id=existing.id, visible_to=visible_to,
            also_visible=also_visible, not_found=not_found,
        )
        if shared is not None:
            raise not_found(existing.id)


def _find(items, item_id):
    for item in items:
        if item.id == item_id:
            return item
    return None
