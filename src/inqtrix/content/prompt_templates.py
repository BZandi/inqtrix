"""Prompt-template records: the server side of the prompt library.

Templates were browser-only state until sharing v1; this module is
their persistence contract. Ownership follows the knowledge-collection
rule: ``owner_sub`` is the creating OIDC subject, and ``None`` (the
anonymous/static principals) marks open templates visible to every
caller of the deployment.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

TEMPLATE_CATEGORIES = ("instruction", "function", "context")


class PromptTemplateNotFound(KeyError):
    """Raised when a template id is unknown (or hidden — same signal)."""


class PromptTemplateConflict(RuntimeError):
    """Raised when an optimistic-concurrency precondition fails.

    A caller passing the ``updated_at`` it loaded asserts "overwrite
    only if nothing changed since". When the stored record moved on, the
    write is rejected (HTTP 409) instead of silently clobbering the
    intervening edit — the deliberate replacement for the old
    last-write-wins behaviour.
    """


@dataclass(frozen=True)
class PromptTemplateRecord:
    """One stored prompt template.

    Attributes:
        id: Server-assigned stable identifier (``pt_...``).
        tenant_id: Tenant scope (v1 runs one tenant per deployment).
        owner_sub: Creating OIDC subject; ``None`` = open template
            (anonymous/static creators), visible and editable for all.
        title: Display title in the prompt library.
        label: The ``@``-mention label used in the composer.
        category: One of :data:`TEMPLATE_CATEGORIES` or ``None``
            (uncategorized legacy rules).
        content_markdown: The template body.
        visibility: Surface flags, e.g. ``{"chat": true, "editor":
            false}`` — stored verbatim, interpreted by the client.
        include_in_autocomplete: Whether the ``@``-menu offers it.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of the last write — also the
            optimistic-concurrency anchor: an update may pass the value
            it loaded as a precondition, and a mismatch raises
            :class:`PromptTemplateConflict` (HTTP 409) rather than
            overwriting an intervening edit.
    """

    id: str
    tenant_id: str
    owner_sub: str | None
    title: str
    label: str
    category: str | None
    content_markdown: str
    visibility: dict[str, Any] = field(default_factory=dict)
    include_in_autocomplete: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0


@runtime_checkable
class PromptTemplateRepository(Protocol):
    """Persistence port for prompt templates (memory + Postgres)."""

    async def create(self, record: PromptTemplateRecord) -> PromptTemplateRecord: ...

    async def get(
        self, template_id: str, *, tenant_id: str
    ) -> PromptTemplateRecord: ...

    async def list_for_tenant(
        self, *, tenant_id: str
    ) -> list[PromptTemplateRecord]: ...

    async def update(
        self,
        record: PromptTemplateRecord,
        *,
        expected_updated_at: float | None = None,
    ) -> PromptTemplateRecord: ...

    async def delete(self, template_id: str, *, tenant_id: str) -> None: ...


def new_template_id() -> str:
    """Mint one ``pt_``-prefixed identifier."""
    return f"pt_{uuid.uuid4().hex[:20]}"


class MemoryPromptTemplateRepository:
    """Thread-safe in-process implementation (zero-infrastructure default)."""

    def __init__(self) -> None:
        self._records: dict[str, PromptTemplateRecord] = {}
        self._lock = threading.RLock()

    async def create(self, record: PromptTemplateRecord) -> PromptTemplateRecord:
        with self._lock:
            self._records[record.id] = record
            return record

    async def get(
        self, template_id: str, *, tenant_id: str
    ) -> PromptTemplateRecord:
        with self._lock:
            record = self._records.get(template_id)
        if record is None or record.tenant_id != tenant_id:
            raise PromptTemplateNotFound(template_id)
        return record

    async def list_for_tenant(
        self, *, tenant_id: str
    ) -> list[PromptTemplateRecord]:
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.tenant_id == tenant_id
            ]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    async def update(
        self,
        record: PromptTemplateRecord,
        *,
        expected_updated_at: float | None = None,
    ) -> PromptTemplateRecord:
        with self._lock:
            current = self._records.get(record.id)
            if current is None or current.tenant_id != record.tenant_id:
                raise PromptTemplateNotFound(record.id)
            # Optimistic-concurrency guard under the same lock as the
            # write, so the check-then-write is atomic (no read-modify
            # race). None = unconditional overwrite (legacy callers).
            if (
                expected_updated_at is not None
                and current.updated_at != expected_updated_at
            ):
                raise PromptTemplateConflict(record.id)
            stored = replace(record, updated_at=time.time())
            self._records[record.id] = stored
            return stored

    async def delete(self, template_id: str, *, tenant_id: str) -> None:
        with self._lock:
            record = self._records.get(template_id)
            if record is None or record.tenant_id != tenant_id:
                raise PromptTemplateNotFound(template_id)
            del self._records[template_id]
