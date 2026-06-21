"""Binary blob storage behind a Baukasten port (files feature).

The store holds opaque blobs under content-neutral keys
(``tenants/<tenant>/files/<uuid>``); every fact ABOUT a blob — owner,
filename, hash, permissions — lives in the file registry. The store is
never reachable by clients: Inqtrix checks permissions and streams the
bytes itself, so credentials and topology stay inside the backend.

Two implementations:

* :class:`LocalFSObjectStore` — directory tree on local disk, the
  zero-infrastructure default.
* :class:`S3ObjectStore` — any S3-compatible endpoint via boto3 with
  path-style addressing (SeaweedFS is the reference dev stack;
  virtual-host bucket addressing does not exist on self-hosted
  stores). boto3 is synchronous by design here — callers running on
  the event loop wrap calls in a thread (FastAPI does this for sync
  routes automatically).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Iterator

log = logging.getLogger("inqtrix")

_STREAM_CHUNK_BYTES = 1024 * 1024


class ObjectStoreError(RuntimeError):
    """Raised when a blob operation fails (missing key, backend down).

    Deliberately loud: a registry row whose blob cannot be served is
    an inconsistency operators must see, never an empty download.
    """


class ObjectStore(ABC):
    """Port for opaque blob storage keyed by caller-supplied strings."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the backing blob service is reachable now.

        This is a read-only operator probe for health/capability surfaces.
        It must not create buckets, write sentinel objects, or expose
        endpoints/credentials; callers only receive a boolean.
        """

    @abstractmethod
    def put(self, key: str, source_path: Path) -> None:
        """Store the file at *source_path* under *key* (overwrite)."""

    @abstractmethod
    def stream(self, key: str) -> Iterator[bytes]:
        """Return a chunk iterator over the blob's bytes.

        Implementations open the blob EAGERLY — a missing key or an
        unreachable backend raises here, before any byte is handed
        out, so HTTP callers can still answer with a clean error
        status instead of aborting a started 200 response.

        Raises:
            ObjectStoreError: When the key does not exist or the
                backend is unreachable.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove the blob. Missing keys are tolerated — the registry
        row is the source of truth and may outlive a crashed upload."""


def _iter_file(handle: IO[bytes]) -> Iterator[bytes]:
    """Yield chunks from an already-open handle, closing it at the end."""
    with handle:
        while chunk := handle.read(_STREAM_CHUNK_BYTES):
            yield chunk


def _iter_s3_body(body) -> Iterator[bytes]:
    """Yield chunks from an already-fetched S3 body, closing it at the end."""
    try:
        while chunk := body.read(_STREAM_CHUNK_BYTES):
            yield chunk
    finally:
        body.close()


class LocalFSObjectStore(ObjectStore):
    """Directory-tree blob store (zero-infrastructure default).

    Args:
        root: Base directory; keys become paths below it. Created on
            first write. Key segments are server-generated (tenant id,
            UUID), never client-controlled — path traversal is
            structurally impossible, but the guard in :meth:`_path`
            keeps that invariant explicit.
    """

    def __init__(self, *, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root.resolve()):
            raise ObjectStoreError(f"object key escapes the store root: {key!r}")
        return path

    def is_available(self) -> bool:
        """Return whether the local directory backend is usable."""
        target = self._root
        while not target.exists() and target != target.parent:
            target = target.parent
        return target.is_dir() and os.access(target, os.W_OK | os.X_OK)

    def put(self, key: str, source_path: Path) -> None:
        """Copy *source_path* into the tree (atomic via temp + rename)."""
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(target.suffix + ".part")
        shutil.copyfile(source_path, staging)
        staging.replace(target)

    def stream(self, key: str) -> Iterator[bytes]:
        """Open the blob eagerly, then return the chunk iterator."""
        path = self._path(key)
        try:
            handle = path.open("rb")
        except OSError as exc:
            raise ObjectStoreError(
                f"blob not found in local store: {key!r}"
            ) from exc
        return _iter_file(handle)

    def delete(self, key: str) -> None:
        """Remove the blob file; a missing file is a tolerated no-op."""
        self._path(key).unlink(missing_ok=True)


class S3ObjectStore(ObjectStore):
    """S3-compatible blob store (SeaweedFS, AWS S3, ...).

    Args:
        endpoint_url: Service endpoint, e.g. ``http://127.0.0.1:8333``.
        bucket: Bucket for every blob; created on first use when the
            service allows it.
        access_key: S3 access key (constructor-first — never read from
            the environment here).
        secret_key: S3 secret key.
        region: Region name boto3 requires; self-hosted stores ignore
            the value.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ) -> None:
        self._endpoint_url = endpoint_url
        self._bucket = bucket
        self._access_key = access_key
        self._secret_key = secret_key
        self._region = region
        self._client = None
        self._bucket_ensured = False
        self._bucket_lock = threading.Lock()

    def _s3(self):
        """Lazily build the boto3 client (path-style addressing)."""
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                region_name=self._region,
                config=Config(s3={"addressing_style": "path"}),
            )
        return self._client

    def is_available(self) -> bool:
        """Return whether the S3 endpoint is reachable without mutating it.

        A missing bucket still counts as reachable because first upload is
        allowed to create it through :meth:`ensure_bucket`; connection,
        transport, and authorization failures do not.
        """
        from botocore.exceptions import ClientError

        try:
            self._s3().head_bucket(Bucket=self._bucket)
            return True
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            )
            return status == 404
        except Exception:
            return False

    def ensure_bucket(self) -> None:
        """Create the bucket when missing (idempotent, thread-safe).

        Only a definite "bucket does not exist" answer triggers
        creation; auth failures, network errors, and everything else
        re-raise so misconfiguration never hides behind a create
        attempt. Concurrent first uploads are serialized by a lock and
        racing creators tolerate the already-exists answers.
        """
        from botocore.exceptions import ClientError

        with self._bucket_lock:
            if self._bucket_ensured:
                return
            client = self._s3()
            try:
                client.head_bucket(Bucket=self._bucket)
                self._bucket_ensured = True
                return
            except ClientError as exc:
                status = exc.response.get("ResponseMetadata", {}).get(
                    "HTTPStatusCode"
                )
                if status != 404:
                    raise ObjectStoreError(
                        f"head_bucket failed for {self._bucket!r}: {exc}"
                    ) from exc
                log.warning(
                    "object store bucket %s missing; creating it",
                    self._bucket,
                )
            create_kwargs: dict = {"Bucket": self._bucket}
            if self._region and self._region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._region
                }
            try:
                client.create_bucket(**create_kwargs)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in (
                    "BucketAlreadyOwnedByYou",
                    "BucketAlreadyExists",
                ):
                    raise ObjectStoreError(
                        f"create_bucket failed for {self._bucket!r}: {exc}"
                    ) from exc
            self._bucket_ensured = True

    def put(self, key: str, source_path: Path) -> None:
        """Upload the file (multipart handled by boto3 transfer).

        The bucket is ensured once per process before the first
        upload — startup stays network-free, the first write fails
        loudly when the store is unreachable.
        """
        if not self._bucket_ensured:
            self.ensure_bucket()
        try:
            self._s3().upload_file(str(source_path), self._bucket, key)
        except ObjectStoreError:
            raise
        except Exception as exc:
            raise ObjectStoreError(
                f"S3 upload failed for key {key!r}: {exc}"
            ) from exc

    def stream(self, key: str) -> Iterator[bytes]:
        """Fetch the object eagerly, then return the body iterator."""
        try:
            response = self._s3().get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            raise ObjectStoreError(
                f"S3 download failed for key {key!r}: {exc}"
            ) from exc
        return _iter_s3_body(response["Body"])

    def delete(self, key: str) -> None:
        """Delete the object (S3 delete is idempotent by contract)."""
        try:
            self._s3().delete_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            raise ObjectStoreError(
                f"S3 delete failed for key {key!r}: {exc}"
            ) from exc
