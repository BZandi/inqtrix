"""HTTP tests for the file surface: upload, visibility, shares, limits.

Memory registry + a tmp-path local object store — fully offline. The
auth provider maps a request header to scoped principals (same pattern
as the run-visibility suite) so creator-only access, share grants, and
the legacy unscoped view are all exercised over the real routers.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import PermissionService, SharePermission
from inqtrix.knowledge.parsing import DocumentParseError, DocumentParser
from inqtrix.providers.base import ProviderContext
from inqtrix.server.container import build_container
from inqtrix.server.routers import capabilities as capabilities_router
from inqtrix.server.routers import files as files_router
from inqtrix.services.file_service import FILE_RESOURCE_TYPE
from inqtrix.settings import Settings, StorageSettings

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
    permissions = PermissionService(
        members=identity, groups=identity, shares=identity, audit=identity
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
    )
    # The default offline container wires no parser; inject one to exercise the
    # server-side text extraction route (the frozen container is replaced).
    if document_parser is not None:
        container = replace(container, document_parser=document_parser)
    app = FastAPI()
    app.include_router(files_router.build_router(container))
    app.include_router(capabilities_router.build_router(container))
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
    assert {item["id"] for item in listed_anonymous["data"]} == {
        file_a,
        file_b,
    }


def test_share_grant_opens_read_but_not_delete(tmp_path):
    client, identity = make_files_client(tmp_path)
    with client:
        file_id = upload(client, sub="user-a")["id"]
        identity.add_share(
            subject_type="user",
            subject_id="user-b",
            resource_type=FILE_RESOURCE_TYPE,
            resource_id=file_id,
            permission=SharePermission.VIEW,
        )

        shared_read = client.get(
            f"/v1/files/{file_id}/content", headers={SUB_HEADER: "user-b"}
        )
        shared_delete = client.delete(
            f"/v1/files/{file_id}", headers={SUB_HEADER: "user-b"}
        )

    assert shared_read.status_code == 200
    assert shared_read.content == PAYLOAD
    # view does not imply manage; the denial stays a 404.
    assert shared_delete.status_code == 404


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


def test_missing_blob_is_a_loud_502_not_an_empty_200(tmp_path):
    """A registry row whose blob vanished must answer 502, never a
    200 with an empty body (the stream opens eagerly)."""
    client, _ = make_files_client(tmp_path)
    blob_root = tmp_path / "blobs"
    with client:
        file_id = upload(client)["id"]
        for blob in blob_root.rglob("fl_*"):
            blob.unlink()

        content = client.get(f"/v1/files/{file_id}/content")
        metadata = client.get(f"/v1/files/{file_id}")

    assert content.status_code == 502
    assert content.json()["error"]["type"] == "server_error"
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


def test_manage_share_grants_read_and_delete(tmp_path):
    client, identity = make_files_client(tmp_path)
    with client:
        file_id = upload(client, sub="user-a")["id"]
        identity.add_share(
            subject_type="user",
            subject_id="user-b",
            resource_type=FILE_RESOURCE_TYPE,
            resource_id=file_id,
            permission=SharePermission.MANAGE,
        )

        read = client.get(
            f"/v1/files/{file_id}/content", headers={SUB_HEADER: "user-b"}
        )
        deleted = client.delete(
            f"/v1/files/{file_id}", headers={SUB_HEADER: "user-b"}
        )

    assert read.status_code == 200
    assert deleted.status_code == 204
    assert not any((tmp_path / "blobs").rglob("fl_*"))


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

    # Legacy unscoped principals keep full access (historical mode).
    assert as_anonymous.status_code == 200
    # Same sub in another tenant: hidden, byte-identical to absence.
    assert as_other_tenant.status_code == 404
    assert as_other_tenant.json() == missing.json()


def test_s3_backend_without_credentials_fails_loudly():
    from inqtrix.server.container import build_object_store

    with pytest.raises(RuntimeError, match="INQTRIX_S3_ENDPOINT_URL"):
        build_object_store(
            Settings(
                storage=StorageSettings(
                    object_store_backend="s3",
                    s3_endpoint_url="http://127.0.0.1:8333",
                    s3_access_key="key",
                    s3_secret_key="",
                )
            )
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
