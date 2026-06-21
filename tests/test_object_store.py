"""Unit tests for the object-store implementations.

LocalFS runs offline against tmp_path; the S3 implementation has a
gated integration suite (``INQTRIX_TEST_S3_ENDPOINT``) against the
SeaweedFS dev stack — see ``docs/development/local-infrastructure.md``.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from inqtrix.storage.object_store import (
    LocalFSObjectStore,
    ObjectStoreError,
    S3ObjectStore,
)


def write_source(tmp_path: Path, content: bytes) -> Path:
    source = tmp_path / "source.bin"
    source.write_bytes(content)
    return source


# ------------------------------------------------------------------ #
# LocalFS (offline)
# ------------------------------------------------------------------ #


def test_localfs_roundtrip_and_delete(tmp_path):
    store = LocalFSObjectStore(root=tmp_path / "store")
    source = write_source(tmp_path, b"blob-bytes" * 100)

    store.put("tenants/default/files/fl_1", source)
    streamed = b"".join(store.stream("tenants/default/files/fl_1"))
    assert streamed == b"blob-bytes" * 100

    store.delete("tenants/default/files/fl_1")
    with pytest.raises(ObjectStoreError, match="not found"):
        list(store.stream("tenants/default/files/fl_1"))
    # Deleting again stays a tolerated no-op.
    store.delete("tenants/default/files/fl_1")


def test_localfs_put_overwrites_atomically(tmp_path):
    store = LocalFSObjectStore(root=tmp_path / "store")
    store.put("k", write_source(tmp_path, b"first"))
    store.put("k", write_source(tmp_path, b"second"))
    assert b"".join(store.stream("k")) == b"second"
    # No stray .part staging files left behind.
    assert not list((tmp_path / "store").rglob("*.part"))


def test_localfs_rejects_keys_escaping_the_root(tmp_path):
    store = LocalFSObjectStore(root=tmp_path / "store")
    with pytest.raises(ObjectStoreError, match="escapes"):
        list(store.stream("../outside"))
    with pytest.raises(ObjectStoreError, match="escapes"):
        store.put("../../etc/passwd", write_source(tmp_path, b"x"))


# ------------------------------------------------------------------ #
# S3 / SeaweedFS (gated)
# ------------------------------------------------------------------ #

S3_ENDPOINT = os.environ.get("INQTRIX_TEST_S3_ENDPOINT", "")

s3_gated = pytest.mark.skipif(
    not S3_ENDPOINT,
    reason="INQTRIX_TEST_S3_ENDPOINT not set (SeaweedFS integration)",
)


@pytest.fixture()
def s3_store() -> S3ObjectStore:
    return S3ObjectStore(
        endpoint_url=S3_ENDPOINT,
        bucket="inqtrix-files-test",
        access_key=os.environ.get(
            "INQTRIX_TEST_S3_ACCESS_KEY", "inqtrix-dev-access"
        ),
        secret_key=os.environ.get(
            "INQTRIX_TEST_S3_SECRET_KEY", "inqtrix-dev-secret"
        ),
    )


@s3_gated
def test_s3_roundtrip_and_delete(tmp_path, s3_store):
    key = f"tenants/default/files/fl_{uuid.uuid4().hex}"
    payload = b"seaweed-blob" * 1000
    s3_store.put(key, write_source(tmp_path, payload))
    try:
        assert b"".join(s3_store.stream(key)) == payload
    finally:
        s3_store.delete(key)
    with pytest.raises(ObjectStoreError):
        list(s3_store.stream(key))


@s3_gated
def test_s3_missing_key_raises_loudly(s3_store):
    with pytest.raises(ObjectStoreError, match="download failed"):
        list(s3_store.stream("tenants/default/files/fl_does_not_exist"))


# ------------------------------------------------------------------ #
# S3 offline (botocore Stubber — no network)
# ------------------------------------------------------------------ #


def make_stubbed_store():
    from botocore.stub import Stubber

    store = S3ObjectStore(
        endpoint_url="http://stub.invalid",
        bucket="inqtrix-files",
        access_key="stub-access",
        secret_key="stub-secret",
    )
    stubber = Stubber(store._s3())
    return store, stubber


def test_s3_stream_raises_eagerly_on_missing_key_offline():
    store, stubber = make_stubbed_store()
    stubber.add_client_error(
        "get_object", service_error_code="NoSuchKey", http_status_code=404
    )
    with stubber:
        # The error must surface at stream() call time, BEFORE any
        # iteration — that is what keeps the HTTP 502 path alive.
        with pytest.raises(ObjectStoreError, match="download failed"):
            store.stream("tenants/default/files/fl_missing")


def test_s3_ensure_bucket_creates_only_on_definite_404_offline(tmp_path):
    store, stubber = make_stubbed_store()
    stubber.add_client_error(
        "head_bucket", service_error_code="404", http_status_code=404
    )
    stubber.add_response("create_bucket", {}, {"Bucket": "inqtrix-files"})
    with stubber:
        store.ensure_bucket()
    assert store._bucket_ensured is True

    # A non-404 head failure (auth, network) must re-raise, never
    # silently fall through to create_bucket.
    failing_store, failing_stubber = make_stubbed_store()
    failing_stubber.add_client_error(
        "head_bucket", service_error_code="403", http_status_code=403
    )
    with failing_stubber:
        with pytest.raises(ObjectStoreError, match="head_bucket failed"):
            failing_store.ensure_bucket()
    assert failing_store._bucket_ensured is False


def test_s3_delete_failure_raises_loudly_offline():
    store, stubber = make_stubbed_store()
    stubber.add_client_error(
        "delete_object", service_error_code="InternalError", http_status_code=500
    )
    with stubber:
        with pytest.raises(ObjectStoreError, match="delete failed"):
            store.delete("tenants/default/files/fl_1")
