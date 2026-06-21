"""Behavior tests for the file-asset-record service (M6c, offline tier)."""

from __future__ import annotations

import pytest

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.pagination import decode_cursor
from inqtrix.project.asset_records_memory import MemoryAssetStore
from inqtrix.project.asset_records_ports import (
    AssetNotFound,
    SectionNotFound,
)
from inqtrix.services.asset_records_service import (
    AssetRecordsService,
    AssetValidationError,
)


def _scoped(sub: str) -> UserContext:
    return UserContext(
        principal=Principal(sub=sub, kind="oidc_session", tenant_id="default", role="member")
    )


@pytest.fixture()
def service() -> AssetRecordsService:
    return AssetRecordsService(store=MemoryAssetStore(), durable=False)


async def _section(service, sid, *, owner, created_at=1.0):
    await service.save_section(
        id=sid, kind="custom", title="S", created_at=created_at, updated_at=created_at,
        caller_sub=owner, workspace_id=None, visible_to=_scoped(owner),
    )


async def _asset(service, aid, *, owner, section_id="fsec_1", group_id=None,
                 created_at=1.0, text="body"):
    await service.save_asset(
        id=aid, section_id=section_id, group_id=group_id, title="A", label="A",
        file_name="a.pdf", mime_type="application/pdf", origin="library",
        page_count=2, parse_status="parsed", parse_warning=None,
        text_truncated=False, size_bytes=10, server_file_id="fl_1",
        extracted_text=text, created_at=created_at, updated_at=created_at,
        caller_sub=owner, workspace_id=None, visible_to=_scoped(owner),
    )


@pytest.mark.asyncio
async def test_asset_body_lazy_listed_loaded_on_get(service) -> None:
    await _section(service, "fsec_1", owner="u")
    await _asset(service, "fa_1", owner="u", text="the heavy extracted text")
    page, _ = await service.list_assets(caller_sub="u", workspace_id=None, limit=50, after=None)
    assert page[0].extracted_text == ""  # body excluded from list
    full = await service.get_asset("fa_1", visible_to=_scoped("u"))
    assert full.extracted_text == "the heavy extracted text"


@pytest.mark.asyncio
async def test_asset_keyset_walks(service) -> None:
    await _section(service, "fsec_1", owner="u")
    for n, ts in enumerate([10.0, 10.0, 20.0, 30.0, 30.0]):
        await _asset(service, f"fa_{n}", owner="u", created_at=ts)
    seen, cursor = [], None
    for _ in range(10):
        pg, nxt = await service.list_assets(caller_sub="u", workspace_id=None, limit=2, after=cursor)
        seen.extend(a.id for a in pg)
        if nxt is None:
            break
        cursor = decode_cursor(nxt)
    assert len(seen) == len(set(seen)) == 5


@pytest.mark.asyncio
async def test_asset_upsert_preserves_owner_and_created_at(service) -> None:
    await _section(service, "fsec_1", owner="u")
    await _asset(service, "fa_1", owner="u", created_at=100.0, text="v1")
    await service.save_asset(
        id="fa_1", section_id="fsec_1", group_id=None, title="renamed", label="A",
        file_name="a.pdf", mime_type="application/pdf", origin="library",
        page_count=2, parse_status="parsed", parse_warning=None, text_truncated=False,
        size_bytes=10, server_file_id="fl_1", extracted_text="v2",
        created_at=999.0, updated_at=200.0, caller_sub="u", workspace_id=None,
        visible_to=_scoped("u"),
    )
    asset = await service.get_asset("fa_1", visible_to=_scoped("u"))
    assert asset.title == "renamed"
    assert asset.extracted_text == "v2"
    assert asset.created_at == 100.0
    assert asset.created_by_sub == "u"


@pytest.mark.asyncio
async def test_parser_id_round_trips_as_metadata(service) -> None:
    """parser_id (parse provenance) survives save and rides the metadata
    list (not just the body GET), so the UI badge is correct after a
    keyset-paged reload. Defaults to None when the caller omits it."""
    await _section(service, "fsec_1", owner="u")
    await service.save_asset(
        id="fa_srv", section_id="fsec_1", group_id=None, title="A", label="A",
        file_name="a.pdf", mime_type="application/pdf", origin="library",
        page_count=2, parse_status="parsed", parse_warning=None,
        text_truncated=False, size_bytes=10, server_file_id="fl_1",
        parser_id="markitdown", extracted_text="body", created_at=1.0,
        updated_at=1.0, caller_sub="u", workspace_id=None, visible_to=_scoped("u"),
    )
    await _asset(service, "fa_local", owner="u", created_at=2.0)  # no parser_id
    by_id = {
        a.id: a
        for a, _ in [
            (await service.get_asset("fa_srv", visible_to=_scoped("u")), None),
            (await service.get_asset("fa_local", visible_to=_scoped("u")), None),
        ]
    }
    assert by_id["fa_srv"].parser_id == "markitdown"
    assert by_id["fa_local"].parser_id is None
    page, _ = await service.list_assets(
        caller_sub="u", workspace_id=None, limit=50, after=None
    )
    assert {a.id: a.parser_id for a in page} == {
        "fa_srv": "markitdown",
        "fa_local": None,
    }


@pytest.mark.asyncio
async def test_owner_isolation(service) -> None:
    await _section(service, "fsec_1", owner="u-a")
    await _asset(service, "fa_a", owner="u-a")
    with pytest.raises(AssetNotFound):
        await service.get_asset("fa_a", visible_to=_scoped("u-b"))
    page, _ = await service.list_assets(caller_sub="u-b", workspace_id=None, limit=50, after=None)
    assert page == []


@pytest.mark.asyncio
async def test_section_delete_cascades_group_delete_orphans(service) -> None:
    await _section(service, "fsec_1", owner="u")
    await service.save_group(
        id="fg_1", section_id="fsec_1", title="G", created_at=1.0, updated_at=1.0,
        caller_sub="u", workspace_id=None, visible_to=_scoped("u"),
    )
    await _asset(service, "fa_1", owner="u", group_id="fg_1")
    # Delete the group -> asset orphans to ungrouped (group_id null).
    await service.delete_group("fg_1", visible_to=_scoped("u"))
    assert (await service.get_asset("fa_1", visible_to=_scoped("u"))).group_id is None
    # Delete the section -> its assets + groups cascade away.
    await service.delete_section("fsec_1", visible_to=_scoped("u"))
    with pytest.raises(AssetNotFound):
        await service.get_asset("fa_1", visible_to=_scoped("u"))


@pytest.mark.asyncio
async def test_validation(service) -> None:
    with pytest.raises(AssetValidationError):
        await service.save_section(
            id="fsec_x", kind="bogus", title="S", created_at=1.0, updated_at=1.0,
            caller_sub="u", workspace_id=None, visible_to=_scoped("u"),
        )
    await _section(service, "fsec_1", owner="u")
    with pytest.raises(AssetValidationError):
        await _asset_bad_origin(service)


async def _asset_bad_origin(service):
    await service.save_asset(
        id="fa_bad", section_id="fsec_1", group_id=None, title="A", label="A",
        file_name="a", mime_type="x", origin="bogus", page_count=None,
        parse_status="parsed", parse_warning=None, text_truncated=False, size_bytes=0,
        server_file_id=None, extracted_text="", created_at=1.0, updated_at=1.0,
        caller_sub="u", workspace_id=None, visible_to=_scoped("u"),
    )
