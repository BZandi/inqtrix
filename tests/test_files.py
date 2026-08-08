"""HTTP tests for the private file surface: upload, visibility, and limits.

Memory registry + a tmp-path local object store — fully offline. The
auth provider maps a request header to scoped principals (same pattern
as the run-visibility suite) so creator-only access and the ownerless
legacy view are exercised over the real routers.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import AuthorizationService
from inqtrix.knowledge.parsing import DocumentParseError, DocumentParser
from inqtrix.providers.base import ProviderContext
from inqtrix.server.container import build_container
from inqtrix.server.routers import asset_records as asset_records_router
from inqtrix.server.routers import capabilities as capabilities_router
from inqtrix.server.routers import files as files_router
from inqtrix.services.file_service import FileService
from inqtrix.settings import Settings, StorageSettings
from inqtrix.storage.object_store import LocalFSObjectStore, ObjectStoreError

from tests.contract._app import StubLLM, StubSearch
from tests.test_runs_visibility import SUB_HEADER, HeaderSubAuthProvider

PAYLOAD = b"Inqtrix Testdatei: Haftung ist begrenzt.\n" * 64


class _StubParser(DocumentParser):
    """Deterministic stand-in for the parser ladder: returns fixed text, or
    raises like a scanned-PDF-without-text-layer when ``fail`` is set."""

    def __init__(self, *, text: str = "SERVER-PARSED TEXT", fail: bool = False) -> None:
        self._text = text
        self._fail = fail

    @property
    def parser_id(self) -> str:
        return "stub"

    def parse(self, *, file_name: str, content: bytes) -> str:
        if self._fail:
            raise DocumentParseError(f"Datei {file_name!r} ergab keinen Text")
        return self._text


def make_files_client(
    tmp_path: Path,
    *,
    max_file_bytes: int = 10_000_000,
    document_parser: DocumentParser | None = None,
) -> tuple[TestClient, MemoryIdentityStore]:
    identity = MemoryIdentityStore()
    permissions = AuthorizationService(
        members=identity, shares=identity, audit=identity
    )
    container = build_container(
        providers=ProviderContext(llm=StubLLM(), search=StubSearch()),
        strategies=None,
        settings=Settings(
            storage=StorageSettings(
                backend="memory",
                database_url="",
                object_store_backend="local",
                object_store_path=str(tmp_path / "blobs"),
                max_file_bytes=max_file_bytes,
            )
        ),
        semaphore_factory=lambda: asyncio.Semaphore(1),
        auth_provider=HeaderSubAuthProvider(),
        permissions=permissions,
        # The default offline container wires no parser; inject one so it
        # reaches the FileService (the route and the file.text.read
        # capability both parse through the service).
        document_parser=document_parser,
    )
    app = FastAPI()
    app.include_router(files_router.build_router(container))
    app.include_router(capabilities_router.build_router(container))
    app.include_router(asset_records_router.build_router(container))
    return TestClient(app), identity


def upload(
    client: TestClient,
    *,
    sub: str | None = None,
    content: bytes = PAYLOAD,
    file_name: str = "vertrag.pdf",
) -> dict:
    headers = {SUB_HEADER: sub} if sub else {}
    response = client.post(
        "/v1/files",
        files={"file": (file_name, content, "application/pdf")},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


# ------------------------------------------------------------------ #
# Upload + integrity
# ------------------------------------------------------------------ #


def test_upload_reports_server_measured_hash_and_size(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        payload = upload(client)

    assert payload["id"].startswith("fl_")
    assert payload["size_bytes"] == len(PAYLOAD)
    assert payload["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert payload["content_type"] == "application/pdf"
    # Wire shape carries no storage internals.
    assert "object_key" not in payload


def test_download_roundtrip_preserves_bytes(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        file_id = upload(client)["id"]
        response = client.get(f"/v1/files/{file_id}/content")

    assert response.status_code == 200
    assert response.content == PAYLOAD
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["content-length"] == str(len(PAYLOAD))


def test_oversize_upload_is_rejected_with_413(tmp_path):
    client, _ = make_files_client(tmp_path, max_file_bytes=64)
    with client:
        response = client.post(
            "/v1/files",
            files={"file": ("big.bin", b"x" * 65, "application/octet-stream")},
        )
        listed = client.get("/v1/files").json()

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "invalid_request_error"
    # Nothing half-registered.
    assert listed["data"] == []


def test_content_disposition_is_injection_safe(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        file_id = upload(
            client, file_name='evil"\r\nX-Injected: 1.pdf'
        )["id"]
        response = client.get(f"/v1/files/{file_id}/content")

    disposition = response.headers["content-disposition"]
    assert "\r" not in disposition and "\n" not in disposition
    assert "X-Injected" not in response.headers
    assert 'filename="' in disposition


# ------------------------------------------------------------------ #
# Server-side text extraction (/v1/files/{id}/text) — background parse
# ------------------------------------------------------------------ #


def test_file_text_returns_server_parsed_text_and_parser_id(tmp_path):
    client, _ = make_files_client(
        tmp_path, document_parser=_StubParser(text="Vertragstext")
    )
    with client:
        file_id = upload(client)["id"]
        response = client.get(f"/v1/files/{file_id}/text")

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Vertragstext"
    assert body["parser_id"] == "stub"
    assert body["file_id"] == file_id


def test_file_text_501_when_no_parser_configured(tmp_path):
    # Default offline container wires no parser — the client keeps its parse.
    client, _ = make_files_client(tmp_path)
    with client:
        file_id = upload(client)["id"]
        response = client.get(f"/v1/files/{file_id}/text")

    assert response.status_code == 501


def test_file_text_422_when_file_cannot_be_parsed(tmp_path):
    # A scanned PDF without a text layer surfaces a visible error, never a
    # silent empty body (Designprinzip 1).
    client, _ = make_files_client(tmp_path, document_parser=_StubParser(fail=True))
    with client:
        file_id = upload(client)["id"]
        response = client.get(f"/v1/files/{file_id}/text")

    assert response.status_code == 422


def test_file_text_enforces_access_check(tmp_path):
    client, _ = make_files_client(tmp_path, document_parser=_StubParser())
    with client:
        file_id = upload(client, sub="user-a")["id"]
        as_other = client.get(
            f"/v1/files/{file_id}/text", headers={SUB_HEADER: "user-b"}
        )

    assert as_other.status_code == 404


# ------------------------------------------------------------------ #
# Visibility (mirrors the run rules)
# ------------------------------------------------------------------ #


def test_scoped_principal_cannot_see_anothers_file(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        file_id = upload(client, sub="user-a")["id"]

        as_b_meta = client.get(
            f"/v1/files/{file_id}", headers={SUB_HEADER: "user-b"}
        )
        as_b_content = client.get(
            f"/v1/files/{file_id}/content", headers={SUB_HEADER: "user-b"}
        )
        missing = client.get(
            "/v1/files/fl_does_not_exist", headers={SUB_HEADER: "user-b"}
        )

    assert as_b_meta.status_code == 404
    assert as_b_content.status_code == 404
    # Denial and absence are byte-identical.
    assert as_b_meta.json() == missing.json()


def test_listing_is_creator_scoped_and_legacy_unscoped(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        file_a = upload(client, sub="user-a")["id"]
        file_b = upload(client, sub="user-b")["id"]

        listed_a = client.get(
            "/v1/files", headers={SUB_HEADER: "user-a"}
        ).json()
        listed_anonymous = client.get("/v1/files").json()

    assert [item["id"] for item in listed_a["data"]] == [file_a]
    assert listed_anonymous["data"] == []


def test_owner_delete_removes_metadata_and_blob(tmp_path):
    client, _ = make_files_client(tmp_path)
    blob_root = tmp_path / "blobs"
    with client:
        file_id = upload(client, sub="user-a")["id"]
        assert any(blob_root.rglob("fl_*"))

        deleted = client.delete(
            f"/v1/files/{file_id}", headers={SUB_HEADER: "user-a"}
        )
        gone = client.get(
            f"/v1/files/{file_id}", headers={SUB_HEADER: "user-a"}
        )

    assert deleted.status_code == 204
    assert gone.status_code == 404
    assert not any(blob_root.rglob("fl_*"))


def test_delete_keeps_metadata_and_quota_anchor_when_blob_delete_fails(
    tmp_path, monkeypatch
):
    client, _ = make_files_client(tmp_path)
    with client:
        file_id = upload(client, sub="user-a")["id"]

        def fail_delete(self, key):
            raise ObjectStoreError("temporary object-store failure")

        monkeypatch.setattr(LocalFSObjectStore, "delete", fail_delete)
        failed = client.delete(
            f"/v1/files/{file_id}", headers={SUB_HEADER: "user-a"}
        )
        metadata = client.get(
            f"/v1/files/{file_id}", headers={SUB_HEADER: "user-a"}
        )

    assert failed.status_code == 503
    assert failed.json()["error"]["type"] == "object_store_unavailable"
    assert metadata.status_code == 200


@pytest.mark.asyncio
async def test_object_store_probe_is_single_flight_across_caller_timeouts() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0

    class BlockingStore:
        def is_available(self) -> bool:
            nonlocal calls
            calls += 1
            started.set()
            release.wait(timeout=2)
            return True

    service = FileService(
        registry=Mock(),
        object_store=BlockingStore(),  # type: ignore[arg-type]
        permissions=Mock(),
        max_file_bytes=1024,
    )
    first = asyncio.create_task(service.object_store_available())
    assert await asyncio.to_thread(started.wait, 1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(service.object_store_available(), timeout=0.01)
    assert calls == 1

    release.set()
    assert await service.object_store_available() is True
    assert calls == 1


def test_workspace_namespace_filters_listing(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        tagged = client.post(
            "/v1/files",
            files={"file": ("a.txt", b"a", "text/plain")},
            headers={"X-Inqtrix-Workspace-Id": "ws-ui-0001"},
        ).json()["id"]
        upload(client, file_name="b.txt")

        filtered = client.get(
            "/v1/files", headers={"X-Inqtrix-Workspace-Id": "ws-ui-0001"}
        ).json()

    assert [item["id"] for item in filtered["data"]] == [tagged]


# ------------------------------------------------------------------ #
# Capability manifest
# ------------------------------------------------------------------ #


def test_capabilities_advertise_files_feature(tmp_path):
    client, _ = make_files_client(tmp_path, max_file_bytes=12_345)
    with client:
        payload = client.get("/v1/capabilities").json()

    assert payload["features"]["files"] is True
    assert payload["files"] == {"max_file_bytes": 12_345}


# ------------------------------------------------------------------ #
# Failure paths and limits (review findings)
# ------------------------------------------------------------------ #


def test_missing_blob_is_a_loud_503_not_an_empty_200(tmp_path):
    """A registry row whose blob vanished must answer 503, never a
    200 with an empty body (the stream opens eagerly)."""
    client, _ = make_files_client(tmp_path)
    blob_root = tmp_path / "blobs"
    with client:
        file_id = upload(client)["id"]
        for blob in blob_root.rglob("fl_*"):
            blob.unlink()

        content = client.get(f"/v1/files/{file_id}/content")
        metadata = client.get(f"/v1/files/{file_id}")

    assert content.status_code == 503
    assert content.json()["error"]["type"] == "object_store_unavailable"
    assert metadata.status_code == 200


def test_oversize_content_length_is_rejected_before_parsing(
    tmp_path, monkeypatch
):
    """The Content-Length precheck answers 413 without ever reaching
    the service (no parse, no spool)."""

    async def _must_not_run(self, **kwargs):
        raise AssertionError("upload must be rejected before the service")

    monkeypatch.setattr(
        "inqtrix.services.file_service.FileService.upload", _must_not_run
    )
    client, _ = make_files_client(tmp_path, max_file_bytes=1024)
    with client:
        response = client.post(
            "/v1/files",
            files={
                "file": (
                    "big.bin",
                    b"x" * (256 * 1024),
                    "application/octet-stream",
                )
            },
        )
    assert response.status_code == 413


def test_multi_chunk_413_leaves_no_spool_files(tmp_path, monkeypatch):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(spool_dir))
    client, _ = make_files_client(tmp_path, max_file_bytes=2 * 1024 * 1024)
    with client:
        response = client.post(
            "/v1/files",
            files={
                "file": (
                    "big.bin",
                    b"x" * (3 * 1024 * 1024),
                    "application/octet-stream",
                )
            },
        )
    assert response.status_code == 413
    assert not list(spool_dir.glob("inqtrix-upload-*"))


def test_legacy_anonymous_reads_scoped_files_and_tenants_stay_separate(
    tmp_path,
):
    client, _ = make_files_client(tmp_path)
    with client:
        file_id = upload(client, sub="user-a")["id"]

        as_anonymous = client.get(f"/v1/files/{file_id}/content")
        as_other_tenant = client.get(
            f"/v1/files/{file_id}",
            headers={SUB_HEADER: "user-a", "X-Test-Tenant": "tenant-x"},
        )
        missing = client.get(
            "/v1/files/fl_missing", headers={SUB_HEADER: "user-a"}
        )

    # Anonymous/static modes can read only genuinely ownerless legacy rows.
    assert as_anonymous.status_code == 404
    # Same sub in another tenant: hidden, byte-identical to absence.
    assert as_other_tenant.status_code == 404
    assert as_other_tenant.json() == missing.json()


# ------------------------------------------------------------------ #
# Upload binding (file + section-bound asset record in one request)
# ------------------------------------------------------------------ #


def test_default_asset_sections_are_idempotent_per_owner_and_workspace(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        headers = {
            SUB_HEADER: "user-a",
            "X-Inqtrix-Workspace-Id": "workspace-a",
        }
        first = client.put("/v1/assets/default-sections", headers=headers)
        second = client.put("/v1/assets/default-sections", headers=headers)
        other_workspace = client.put(
            "/v1/assets/default-sections",
            headers={
                SUB_HEADER: "user-a",
                "X-Inqtrix-Workspace-Id": "workspace-b",
            },
        )
        other_owner = client.put(
            "/v1/assets/default-sections",
            headers={
                SUB_HEADER: "user-b",
                "X-Inqtrix-Workspace-Id": "workspace-a",
            },
        )

    assert first.status_code == second.status_code == 200
    assert [row["semantic_role"] for row in first.json()["data"]] == [
        "temporary",
        "library",
        "project_sources",
    ]
    assert [row["id"] for row in first.json()["data"]] == [
        row["id"] for row in second.json()["data"]
    ]
    first_ids = {row["id"] for row in first.json()["data"]}
    assert first_ids.isdisjoint(row["id"] for row in other_workspace.json()["data"])
    assert first_ids.isdisjoint(row["id"] for row in other_owner.json()["data"])


def _create_section(
    client: TestClient, section_id: str, *, sub: str
) -> None:
    response = client.put(
        f"/v1/assets/sections/{section_id}",
        json={"kind": "custom", "title": "S", "created_at": 1.0, "updated_at": 1.0},
        headers={SUB_HEADER: sub},
    )
    assert response.status_code == 200, response.text


def _upload_bound(
    client: TestClient,
    *,
    sub: str,
    asset_id: str = "file-up-1",
    section_id: str = "fsec_up",
    extra: dict[str, str] | None = None,
):
    data = {
        "asset_id": asset_id,
        "section_id": section_id,
        "title": "Vertrag",
        "label": "vertrag",
        "origin": "library",
        "created_at": "5.0",
        "updated_at": "5.0",
        **(extra or {}),
    }
    return client.post(
        "/v1/files",
        files={"file": ("vertrag.pdf", PAYLOAD, "application/pdf")},
        data=data,
        headers={SUB_HEADER: sub},
    )


def test_bound_upload_persists_file_and_section_placement(tmp_path):
    """A 201 with binding means the collection placement is durable: the
    asset row exists in the target section before the client hears back,
    so a page reload cannot strand the file outside its collection."""
    client, _ = make_files_client(tmp_path)
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        response = _upload_bound(client, sub="user-a")
        assert response.status_code == 201, response.text
        payload = response.json()
        listed = client.get(
            "/v1/assets", headers={SUB_HEADER: "user-a"}
        ).json()["data"]
        detail = client.get(
            "/v1/assets/file-up-1", headers={SUB_HEADER: "user-a"}
        ).json()

    asset = payload["asset"]
    assert asset["id"] == "file-up-1"
    assert asset["section_id"] == "fsec_up"
    assert asset["server_file_id"] == payload["id"]
    # Server-measured facts win over client hints.
    assert asset["size_bytes"] == len(PAYLOAD)
    assert asset["file_name"] == "vertrag.pdf"
    assert [a["id"] for a in listed] == ["file-up-1"]
    assert "prepared_parser_id" in listed[0]
    assert "prepared_content_hash" in listed[0]
    assert "prepared_at" in listed[0]
    # The upload path never writes a body; text follows via asset PUT.
    assert detail["extracted_text"] == ""
    assert detail["prepared_text"] == ""


def test_bound_upload_returns_202_while_canonical_parse_is_queued(tmp_path):
    client, _ = make_files_client(
        tmp_path,
        document_parser=_StubParser(text="KANONISCHER SERVER-TEXT"),
    )
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        response = _upload_bound(client, sub="user-a")

    assert response.status_code == 202
    payload = response.json()
    assert payload["object"] == "upload_operation"
    assert payload["asset"]["server_file_id"]
    assert payload["asset"]["upload_status"] == "parsing"
    assert payload["upload_operation"]["status"] == "queued"
    assert payload["upload_operation"]["stage"] == "parsing"


def test_explicit_reservation_precedes_bytes_and_finalizes_same_asset(tmp_path):
    client, _ = make_files_client(tmp_path)
    reservation_body = {
        "section_id": "fsec_up",
        "group_id": None,
        "title": "vertrag.pdf",
        "label": "vertrag",
        "file_name": "vertrag.pdf",
        "mime_type": "application/pdf",
        "origin": "library",
        "page_count": None,
        "parse_status": "parsed",
        "parse_warning": None,
        "text_truncated": False,
        "size_bytes": len(PAYLOAD),
        "parser_id": None,
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        reserved = client.post(
            "/v1/assets/file-up-1/upload-reservation",
            json=reservation_body,
            headers={SUB_HEADER: "user-a"},
        )
        before = client.get(
            "/v1/assets/file-up-1", headers={SUB_HEADER: "user-a"}
        ).json()
        uploaded = _upload_bound(client, sub="user-a")

    assert reserved.status_code == 201
    assert before["upload_status"] == "awaiting_upload"
    assert before["server_file_id"] is None
    assert uploaded.status_code == 201
    assert uploaded.json()["asset"]["id"] == "file-up-1"
    assert uploaded.json()["asset"]["upload_status"] == "ready"


def test_bound_upload_replay_reuses_operation_file_and_quota_identity(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        first = _upload_bound(client, sub="user-a")
        replay = _upload_bound(
            client,
            sub="user-a",
            extra={"created_at": "999.0", "updated_at": "999.0"},
        )
        files = client.get("/v1/files", headers={SUB_HEADER: "user-a"}).json()["data"]

    assert first.status_code == 201
    assert replay.status_code == 200
    assert replay.json()["id"] == first.json()["id"]
    assert (
        replay.json()["upload_operation"]["operation_id"]
        == first.json()["upload_operation"]["operation_id"]
    )
    assert replay.json()["upload_operation"]["status"] == "ready"
    assert [item["id"] for item in files] == [first.json()["id"]]


def test_upload_operation_endpoints_are_owner_scoped(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        uploaded = _upload_bound(client, sub="user-a").json()
        operation_id = uploaded["upload_operation"]["operation_id"]
        detail = client.get(
            f"/v1/uploads/{operation_id}", headers={SUB_HEADER: "user-a"}
        )
        listed = client.get("/v1/uploads", headers={SUB_HEADER: "user-a"})
        hidden = client.get(
            f"/v1/uploads/{operation_id}", headers={SUB_HEADER: "user-b"}
        )
        invalid_retry = client.post(
            f"/v1/uploads/{operation_id}/retry",
            headers={SUB_HEADER: "user-a"},
        )

    assert detail.status_code == 200
    assert detail.json()["status"] == "ready"
    assert [item["operation_id"] for item in listed.json()["data"]] == [operation_id]
    assert hidden.status_code == 404
    assert invalid_retry.status_code == 409
    assert invalid_retry.json()["error"]["type"] == "upload_operation_conflict"


def test_bound_original_cannot_bypass_asset_aggregate_delete(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        uploaded = _upload_bound(client, sub="user-a").json()
        blocked = client.delete(
            f"/v1/files/{uploaded['id']}", headers={SUB_HEADER: "user-a"}
        )
        asset = client.get("/v1/assets/file-up-1", headers={SUB_HEADER: "user-a"})
        original = client.get(
            f"/v1/files/{uploaded['id']}", headers={SUB_HEADER: "user-a"}
        )

    assert blocked.status_code == 409
    assert blocked.json()["error"]["type"] == "asset_aggregate_required"
    assert asset.status_code == 200
    assert original.status_code == 200


def test_bound_upload_into_foreign_section_leaves_no_file(tmp_path):
    """Cross-user binding is the indistinct not-found AND fully undone:
    neither a file row nor an asset row survives the rejection."""
    client, _ = make_files_client(tmp_path)
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        response = _upload_bound(client, sub="user-b")
        files_b = client.get("/v1/files", headers={SUB_HEADER: "user-b"}).json()
        assets_b = client.get("/v1/assets", headers={SUB_HEADER: "user-b"}).json()

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found"
    assert files_b["data"] == []
    assert assets_b["data"] == []


def test_bound_upload_unknown_section_leaves_no_file(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        response = _upload_bound(client, sub="user-a", section_id="fsec_missing")
        files_a = client.get("/v1/files", headers={SUB_HEADER: "user-a"}).json()

    assert response.status_code == 404
    assert files_a["data"] == []


def test_bound_upload_invalid_origin_is_rejected_and_undone(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        response = _upload_bound(client, sub="user-a", extra={"origin": "bogus"})
        files_a = client.get("/v1/files", headers={SUB_HEADER: "user-a"}).json()

    assert response.status_code == 400
    assert files_a["data"] == []


def test_binding_requires_both_asset_and_section_id(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        response = client.post(
            "/v1/files",
            files={"file": ("a.pdf", PAYLOAD, "application/pdf")},
            data={"asset_id": "file-up-1"},
            headers={SUB_HEADER: "user-a"},
        )
        files_a = client.get("/v1/files", headers={SUB_HEADER: "user-a"}).json()

    assert response.status_code == 400
    # Rejected before any bytes were stored.
    assert files_a["data"] == []


def test_overlong_binding_field_is_rejected_before_storage(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        response = _upload_bound(
            client, sub="user-a", extra={"label": "x" * 2000}
        )
        files_a = client.get("/v1/files", headers={SUB_HEADER: "user-a"}).json()

    assert response.status_code == 400
    assert "label" in response.json()["error"]["message"]
    assert files_a["data"] == []


def test_non_finite_binding_timestamps_are_rejected_before_storage(tmp_path):
    """NaN/Infinity would persist into Float columns and then poison every
    JSON render of the record (json.dumps refuses non-finite floats),
    bricking the caller's asset listing — reject at the door instead."""
    client, _ = make_files_client(tmp_path)
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        for bad in ("nan", "inf"):
            response = _upload_bound(
                client, sub="user-a", extra={"created_at": bad}
            )
            assert response.status_code == 400, response.text
        files_a = client.get("/v1/files", headers={SUB_HEADER: "user-a"}).json()

    assert files_a["data"] == []


def test_nul_byte_in_binding_field_is_rejected_before_storage(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        response = _upload_bound(client, sub="user-a", extra={"title": "abc\x00def"})
        files_a = client.get("/v1/files", headers={SUB_HEADER: "user-a"}).json()

    assert response.status_code == 400
    assert files_a["data"] == []


def test_out_of_range_page_count_is_rejected_before_storage(tmp_path):
    client, _ = make_files_client(tmp_path)
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        too_big = _upload_bound(
            client, sub="user-a", extra={"page_count": "3000000000"}
        )
        negative = _upload_bound(client, sub="user-a", extra={"page_count": "-1"})
        files_a = client.get("/v1/files", headers={SUB_HEADER: "user-a"}).json()

    assert too_big.status_code == 400
    assert negative.status_code == 400
    assert files_a["data"] == []


def test_unexpected_binding_failure_is_visible_and_recoverable(tmp_path, monkeypatch):
    """A transient bind failure keeps one inspectable operation and file.

    Deleting the bytes would make a server restart unable to continue, while
    hiding them would create untracked storage.  The durable operation instead
    exposes its retry state and exact registered file as one aggregate.
    """
    from inqtrix.services.asset_records_service import AssetRecordsService

    async def explode(self, **kwargs):
        raise RuntimeError("unexpected persistence failure")

    monkeypatch.setattr(AssetRecordsService, "bind_uploaded_file", explode)
    client, _ = make_files_client(tmp_path)
    with client:
        _create_section(client, "fsec_up", sub="user-a")
        response = _upload_bound(client, sub="user-a")
        files_a = client.get("/v1/files", headers={SUB_HEADER: "user-a"}).json()
        asset_a = client.get(
            "/v1/assets/file-up-1", headers={SUB_HEADER: "user-a"}
        ).json()
        operation = client.get(
            f"/v1/uploads/{response.json()['upload_operation']['operation_id']}",
            headers={SUB_HEADER: "user-a"},
        ).json()

    assert response.status_code == 202
    assert len(files_a["data"]) == 1
    assert asset_a["upload_status"] == "retrying"
    assert asset_a["server_file_id"] is None
    assert operation["status"] == "queued"
    assert operation["error"]["type"] == "dependency_error"


def test_upload_without_binding_stores_no_asset_record(tmp_path):
    """The plain upload contract is untouched: no binding fields, no
    asset row — exactly the pre-binding wire behavior."""
    client, _ = make_files_client(tmp_path)
    with client:
        payload = upload(client, sub="user-a")
        assets_a = client.get("/v1/assets", headers={SUB_HEADER: "user-a"}).json()

    assert "asset" not in payload
    assert assets_a["data"] == []


def test_s3_backend_without_credentials_fails_loudly():
    with pytest.raises(ValueError, match="INQTRIX_S3_AUTH_MODE=static"):
        StorageSettings(
            object_store_backend="s3",
            s3_endpoint_url="http://127.0.0.1:8333",
            s3_access_key="key",
            s3_secret_key="",
        )


def test_files_routes_register_through_create_app(tmp_path):
    """One smoke through the public factory: the routes exist there."""
    from inqtrix.providers.base import ProviderContext
    from inqtrix.server.app import create_app

    app = create_app(
        settings=Settings(
            storage=StorageSettings(
                backend="memory",
                database_url="",
                object_store_path=str(tmp_path / "blobs"),
            )
        ),
        providers=ProviderContext(llm=StubLLM(), search=StubSearch()),
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/files",
            files={"file": ("a.txt", b"hello", "text/plain")},
        )
        fetched = client.get(f"/v1/files/{created.json()['id']}/content")

    assert created.status_code == 201
    assert fetched.content == b"hello"
