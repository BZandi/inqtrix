"""Contracts of the file-asset-record store (M6c).

Mirrors editor_ports: the store owns persistence only; scoping lives in
:class:`~inqtrix.services.asset_records_service.AssetRecordsService` and
the wire shape in the router. Two implementations behind one port:
:class:`~inqtrix.project.asset_records_memory.MemoryAssetStore` (offline/
test) and :class:`~inqtrix.project.asset_records_postgres.PostgresAssetStore`.

Like editor documents, an asset carries a heavy ``extracted_text`` body:
``list_assets_page`` returns metadata only (``extracted_text=""``);
``get_asset`` returns the full record with the text (load-on-open).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from inqtrix.project.scoped_upsert import ResourceScope


class SectionNotFound(KeyError):
    """Raised when a section id is unknown to the store (HTTP 404)."""


class GroupNotFound(KeyError):
    """Raised when a group id is unknown to the store (HTTP 404)."""


class AssetNotFound(KeyError):
    """Raised when an asset id is unknown to the store (HTTP 404)."""


@dataclass(frozen=True)
class AssetSection:
    """A top-level file-library section.

    Attributes:
        id: Client-supplied id (``fsec_...``), the primary key.
        kind: ``temporary`` or ``custom``.
        title: Section label.
        created_at/updated_at: Unix timestamps.
        tenant_id/created_by_user_id/workspace_id: The scope.
    """

    id: str
    kind: str
    title: str
    created_at: float
    updated_at: float
    tenant_id: str = "default"
    created_by_user_id: uuid.UUID | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class AssetGroup:
    """A group within a section (``section_id``)."""

    id: str
    section_id: str
    title: str
    created_at: float
    updated_at: float
    tenant_id: str = "default"
    created_by_user_id: uuid.UUID | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class AssetRecord:
    """One file-asset record (library metadata + extracted text).

    Attributes:
        id: Client-supplied id (``fa_...``), the primary key.
        section_id: Owning section.
        group_id: Owning group, or ``None`` when ungrouped.
        title/label/file_name/mime_type/origin: Display + provenance.
        page_count: Page count (``None`` when not applicable).
        parse_status: ``parsed``/``partial``/``unsupported``/``error``.
        parse_warning: Optional parse warning text.
        text_truncated: Whether ``extracted_text`` was truncated at ingest.
        size_bytes: Original binary size.
        server_file_id: The /v1/files blob reference, or ``None`` (local-only).
        parser_id: Which parser produced ``extracted_text`` — ``markitdown``
            for the server-side parser ladder, ``client`` for the browser
            fallback, ``None`` when unknown (legacy/local-only rows).
        extracted_text: The heavy body — empty on list rows, full on get.
        created_at/updated_at: Unix timestamps.
    """

    id: str
    section_id: str
    group_id: str | None
    title: str
    label: str
    file_name: str
    mime_type: str
    origin: str
    page_count: int | None
    parse_status: str
    parse_warning: str | None
    text_truncated: bool
    size_bytes: int
    server_file_id: str | None
    extracted_text: str = field(repr=False, default="")
    parser_id: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    tenant_id: str = "default"
    created_by_user_id: uuid.UUID | None = None
    workspace_id: str | None = None


@runtime_checkable
class AssetStore(Protocol):
    """Persistence port for file-library sections, groups, and asset records."""

    # -- sections --------------------------------------------------------- #

    async def upsert_section(
        self,
        *,
        id: str,
        kind: str,
        title: str,
        created_at: float,
        updated_at: float,
        created_by_user_id: uuid.UUID | None,
        workspace_id: str | None,
    ) -> AssetSection: ...

    async def list_sections(
        self, *, created_by_user_id: uuid.UUID | None, workspace_id: str | None
    ) -> list[AssetSection]: ...

    async def delete_section(
        self, section_id: str, *, scope: ResourceScope
    ) -> None: ...

    # -- groups ----------------------------------------------------------- #

    async def upsert_group(
        self,
        *,
        id: str,
        section_id: str,
        title: str,
        created_at: float,
        updated_at: float,
        created_by_user_id: uuid.UUID | None,
        workspace_id: str | None,
    ) -> AssetGroup: ...

    async def list_groups(
        self, *, created_by_user_id: uuid.UUID | None, workspace_id: str | None
    ) -> list[AssetGroup]: ...

    async def delete_group(
        self, group_id: str, *, scope: ResourceScope
    ) -> None: ...

    # -- assets ----------------------------------------------------------- #

    async def upsert_asset(
        self,
        *,
        id: str,
        section_id: str,
        group_id: str | None,
        title: str,
        label: str,
        file_name: str,
        mime_type: str,
        origin: str,
        page_count: int | None,
        parse_status: str,
        parse_warning: str | None,
        text_truncated: bool,
        size_bytes: int,
        server_file_id: str | None,
        parser_id: str | None = None,
        extracted_text: str,
        created_at: float,
        updated_at: float,
        created_by_user_id: uuid.UUID | None,
        workspace_id: str | None,
    ) -> AssetRecord: ...

    async def list_assets_page(
        self,
        *,
        created_by_user_id: uuid.UUID | None,
        workspace_id: str | None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[AssetRecord], str | None]:
        """One keyset page of the caller's assets (newest first), METADATA
        ONLY (``extracted_text=""`` — the text loads via get_asset)."""
        ...

    async def get_asset(self, asset_id: str) -> AssetRecord:
        """One asset WITH its extracted text (load-on-open), or
        :class:`AssetNotFound`."""
        ...

    async def delete_asset(
        self, asset_id: str, *, scope: ResourceScope
    ) -> None: ...

    async def aclose(self) -> None: ...
