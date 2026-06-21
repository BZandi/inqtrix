"""File upload/download orchestration (registry + object store + authz).

The only place that combines the three collaborators:

* metadata and listing facts come from the
  :class:`~inqtrix.content.ports.FileRegistry`,
* bytes go to / come from the
  :class:`~inqtrix.storage.object_store.ObjectStore`,
* access decisions go through the
  :class:`~inqtrix.auth.permissions.PermissionService`.

Access rules (uniform with run visibility): legacy unscoped principals
keep full access; scoped principals own what they uploaded; non-owners
need a share grant resolved by the permission service. Denied access
is indistinguishable from absence, but every denial is logged and
audited by the permission layer.

Uploads stream through a spool file with running SHA-256 and size
accounting — the limit is enforced without ever holding the file in
memory, and the registry row is written only after the blob landed in
the store (an aborted upload leaves nothing behind but a temp file).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterator

from inqtrix.auth.permissions import (
    PermissionService,
    ResourceNotFound,
    SharePermission,
)
from inqtrix.auth.principal import Principal
from inqtrix.content.ports import FileNotFound, FileRecord, FileRegistry
from inqtrix.storage.object_store import ObjectStore

log = logging.getLogger("inqtrix")

FILE_RESOURCE_TYPE = "file"
"""``resource_shares.resource_type`` value for uploaded files."""

_SAFE_KEY_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
"""Allowed shape for server-generated object-key segments. Tenant ids
and file ids are server-controlled, but the guard keeps the
"keys are shape-safe" invariant explicit instead of assumed."""


class FileTooLarge(ValueError):
    """Raised when an upload exceeds the configured size limit."""

    def __init__(self, *, limit_bytes: int) -> None:
        super().__init__(f"file exceeds the limit of {limit_bytes} bytes")
        self.limit_bytes = limit_bytes


@dataclass(frozen=True)
class SpooledUpload:
    """Result of spooling one upload to disk (pre-registry facts)."""

    path: Path
    size_bytes: int
    sha256: str


class FileService:
    """Upload, list, stream, and delete user files.

    Args:
        registry: Metadata persistence port (memory or Postgres).
        object_store: Blob storage port (local directory or S3).
        permissions: The authorization chokepoint; consulted for every
            non-owner access.
        max_file_bytes: Upload size limit enforced while spooling.
    """

    def __init__(
        self,
        *,
        registry: FileRegistry,
        object_store: ObjectStore,
        permissions: PermissionService,
        max_file_bytes: int,
    ) -> None:
        self._registry = registry
        self._object_store = object_store
        self._permissions = permissions
        self._max_file_bytes = max_file_bytes

    async def object_store_available(self) -> bool:
        """Return whether the configured blob store is reachable now.

        The check is intentionally read-only and runs off the event loop
        because S3/local filesystem probes are synchronous. It feeds
        operator-facing capability/status surfaces; upload/download paths
        still fail loudly on their own operations.
        """
        return await asyncio.to_thread(self._object_store.is_available)

    async def upload(
        self,
        *,
        chunks: AsyncIterator[bytes],
        file_name: str,
        content_type: str,
        principal: Principal,
        workspace_id: str | None,
    ) -> FileRecord:
        """Spool, hash, store, and register one uploaded file.

        Raises:
            FileTooLarge: When the stream exceeds the configured limit
                (spool is discarded, nothing is registered).
        """
        if not _SAFE_KEY_SEGMENT.fullmatch(principal.tenant_id):
            raise ValueError(
                f"tenant id unsafe for object keys: {principal.tenant_id!r}"
            )
        spooled = await self._spool(chunks)
        try:
            file_id = f"fl_{uuid.uuid4().hex}"
            object_key = (
                f"tenants/{principal.tenant_id}/files/{file_id}"
            )
            # Blocking store IO (file copy / boto3 upload) leaves the
            # event loop — uploads of large blobs must not stall other
            # requests.
            await asyncio.to_thread(
                self._object_store.put, object_key, spooled.path
            )
            record = FileRecord(
                id=file_id,
                tenant_id=principal.tenant_id,
                owner_sub=principal.sub,
                workspace_id=workspace_id,
                file_name=file_name,
                content_type=content_type or "application/octet-stream",
                size_bytes=spooled.size_bytes,
                sha256=spooled.sha256,
                object_key=object_key,
                created_at=time.time(),
            )
            try:
                await self._registry.create(record)
            except BaseException:
                # A blob without a registry row is unreachable garbage;
                # clean it up best-effort and keep the failure loud.
                try:
                    await asyncio.to_thread(
                        self._object_store.delete, object_key
                    )
                except Exception as cleanup_exc:
                    log.warning(
                        "orphaned blob after failed registry create: "
                        "key=%s cleanup_error=%s",
                        object_key,
                        cleanup_exc,
                    )
                raise
            return record
        finally:
            spooled.path.unlink(missing_ok=True)

    async def _spool(self, chunks: AsyncIterator[bytes]) -> SpooledUpload:
        """Stream the upload to a temp file with running hash/size."""
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(
            prefix="inqtrix-upload-", delete=False
        ) as spool:
            spool_path = Path(spool.name)
            try:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > self._max_file_bytes:
                        raise FileTooLarge(limit_bytes=self._max_file_bytes)
                    digest.update(chunk)
                    # Disk writes leave the event loop — a slow disk
                    # must not stall unrelated requests mid-upload.
                    await asyncio.to_thread(spool.write, chunk)
            except BaseException:
                spool.close()
                spool_path.unlink(missing_ok=True)
                raise
        return SpooledUpload(
            path=spool_path, size_bytes=size, sha256=digest.hexdigest()
        )

    async def get(self, file_id: str, *, principal: Principal) -> FileRecord:
        """Return metadata after the read-access check."""
        record = await self._registry.get(
            file_id, tenant_id=principal.tenant_id
        )
        await self._require(principal, record, SharePermission.VIEW)
        return record

    async def open_stream(
        self, file_id: str, *, principal: Principal
    ) -> tuple[FileRecord, Iterator[bytes]]:
        """Return metadata plus a byte iterator after the access check.

        The store opens the blob eagerly (missing/unreachable raises
        HERE, before any response byte) — that blocking open runs in a
        thread; the returned iterator stays synchronous because
        Starlette drives sync generators in its threadpool.
        """
        record = await self.get(file_id, principal=principal)
        chunks = await asyncio.to_thread(
            self._object_store.stream, record.object_key
        )
        return record, chunks

    async def list(
        self,
        *,
        principal: Principal,
        user_context_is_scoped: bool,
        workspace_id: str | None,
    ) -> list[FileRecord]:
        """List visible files (owner-only for scoped principals).

        Share-granted files appear once list-by-share lands with the
        sharing UI; creator-only listing is the deliberately
        conservative first cut (mirrors run visibility).
        """
        owner_sub = principal.sub if user_context_is_scoped else None
        return await self._registry.list(
            tenant_id=principal.tenant_id,
            owner_sub=owner_sub,
            workspace_id=workspace_id,
        )

    async def delete(self, file_id: str, *, principal: Principal) -> FileRecord:
        """Delete metadata and blob after the manage-access check.

        The registry row is removed first (authorization anchor), the
        blob second; a blob-store failure after the row is gone leaves
        an orphaned blob, which is logged loudly instead of resurrecting
        the record.

        Returns:
            The deleted :class:`FileRecord` — its owner/size let the
            caller free the owner's stored-bytes quota by exactly what
            was held.
        """
        record = await self._registry.get(
            file_id, tenant_id=principal.tenant_id
        )
        await self._require(principal, record, SharePermission.MANAGE)
        await self._registry.delete(file_id, tenant_id=principal.tenant_id)
        try:
            await asyncio.to_thread(
                self._object_store.delete, record.object_key
            )
        except Exception as exc:
            log.warning(
                "orphaned blob after registry delete: file=%s key=%s "
                "error=%s",
                record.id,
                record.object_key,
                exc,
            )
        return record

    async def _require(
        self,
        principal: Principal,
        record: FileRecord,
        permission: SharePermission,
    ) -> None:
        """Owner shortcut, then the permission chokepoint.

        The owner check needs no repository round-trip; everyone else
        goes through ``PermissionService.require`` so denials are
        audited centrally. ``ResourceNotFound`` is translated to
        :class:`FileNotFound` so the router has one absence signal.
        """
        if (
            principal.sub == record.owner_sub
            and principal.tenant_id == record.tenant_id
        ):
            return
        try:
            await self._permissions.require(
                principal,
                permission,
                resource_type=FILE_RESOURCE_TYPE,
                resource_id=record.id,
            )
        except ResourceNotFound as exc:
            raise FileNotFound(record.id) from exc
