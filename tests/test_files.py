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
