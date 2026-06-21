"""Prompt-template CRUD with the owned-resource visibility rule.

Same enforcement shape as knowledge collections (the strand's
established pattern): the router resolves the caller's shared-in
grants into ``also_visible``, the service decides per record via
:func:`~inqtrix.auth.permissions.grant_for_owned_resource`. Reads
need view, updates need edit, deletion stays owner-only; ownerless
templates (anonymous/static creators) stay open for everyone.
Conflict policy is OPTIMISTIC concurrency: an update may carry the
``updated_at`` it loaded as a precondition; a mismatch raises
:class:`PromptTemplateConflict` (HTTP 409) instead of silently
overwriting the intervening edit. A caller that omits the precondition
keeps the legacy unconditional (last-write-wins) overwrite.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping

from inqtrix.auth.permissions import (
    SharePermission,
    grant_for_owned_resource,
)
from inqtrix.content.prompt_templates import (
    TEMPLATE_CATEGORIES,
    PromptTemplateConflict,
    PromptTemplateNotFound,
    PromptTemplateRecord,
    PromptTemplateRepository,
    new_template_id,
)

__all__ = [
    "PromptTemplateService",
    "PromptTemplateValidationError",
    "PromptTemplateConflict",
    "template_access",
]

if TYPE_CHECKING:
    from inqtrix.auth.principal import UserContext


class PromptTemplateValidationError(ValueError):
    """Raised for client-payload problems (maps to HTTP 400)."""


def template_access(
    record: PromptTemplateRecord,
    visible_to: "UserContext | None",
    also_visible: "Mapping[str, SharePermission] | None" = None,
) -> SharePermission | None:
    """The caller's grant on *record*; raises the indistinct 404.

    ``None`` means full access (unscoped caller, ownerless template,
    or the owner); a permission means shared-in access at that level.
    """
    visible, shared = grant_for_owned_resource(
        owner_sub=record.owner_sub,
        resource_tenant_id=record.tenant_id,
        resource_id=record.id,
        visible_to=visible_to,
        also_visible=also_visible,
    )
    if not visible:
        raise PromptTemplateNotFound(record.id)
    return shared


def _validated_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the writable template fields."""
    title = str(payload.get("title", "") or "").strip()
    label = str(payload.get("label", "") or "").strip()
    content = str(payload.get("content_markdown", "") or "")
    if not title:
        raise PromptTemplateValidationError("Feld 'title' ist erforderlich")
    if not label:
        raise PromptTemplateValidationError("Feld 'label' ist erforderlich")
    if not content.strip():
        raise PromptTemplateValidationError(
            "Feld 'content_markdown' ist erforderlich"
        )
    category = payload.get("category")
    if category is not None:
        category = str(category)
        if category not in TEMPLATE_CATEGORIES:
            raise PromptTemplateValidationError(
                "Feld 'category' muss eines von "
                f"{', '.join(TEMPLATE_CATEGORIES)} sein"
            )
    visibility = payload.get("visibility") or {}
    if not isinstance(visibility, dict):
        raise PromptTemplateValidationError(
            "Feld 'visibility' muss ein Objekt sein"
        )
    include = payload.get("include_in_autocomplete", True)
    if not isinstance(include, bool):
        raise PromptTemplateValidationError(
            "Feld 'include_in_autocomplete' muss ein Boolean sein"
        )
    return {
        "title": title,
        "label": label,
        "category": category,
        "content_markdown": content,
        "visibility": dict(visibility),
        "include_in_autocomplete": include,
    }


class PromptTemplateService:
    """Application service over the prompt-template repository.

    Args:
        repository: The persistence backend.
        durable: Whether *repository* survives server restarts. Drives
            ``features.prompt_templates`` in the capability manifest:
            a browser must not adopt a VOLATILE store as its sync
            truth — after a restart the empty store would read as
            "everything deleted". The routes stay mounted regardless
            (API consumers may use a memory-backed deployment
            knowingly).
    """

    def __init__(
        self,
        *,
        repository: PromptTemplateRepository,
        durable: bool = True,
    ) -> None:
        self._repository = repository
        self._durable = durable

    @property
    def durable(self) -> bool:
        """Whether templates survive a server restart."""
        return self._durable

    async def create(
        self,
        payload: Mapping[str, Any],
        *,
        tenant_id: str,
        owner_sub: str | None,
    ) -> PromptTemplateRecord:
        """Create one template; scoped principals own what they create."""
        now = time.time()
        record = PromptTemplateRecord(
            id=new_template_id(),
            tenant_id=tenant_id,
            owner_sub=owner_sub,
            created_at=now,
            updated_at=now,
            **_validated_fields(payload),
        )
        return await self._repository.create(record)

    async def list_visible(
        self,
        *,
        tenant_id: str,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> list[tuple[PromptTemplateRecord, SharePermission | None]]:
        """Visible templates with their shared-in grant (annotation)."""
        visible: list[tuple[PromptTemplateRecord, SharePermission | None]] = []
        for record in await self._repository.list_for_tenant(
            tenant_id=tenant_id
        ):
            try:
                shared = template_access(record, visible_to, also_visible)
            except PromptTemplateNotFound:
                continue
            visible.append((record, shared))
        return visible

    async def update(
        self,
        template_id: str,
        payload: Mapping[str, Any],
        *,
        tenant_id: str,
        expected_updated_at: float | None = None,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> PromptTemplateRecord:
        """Replace the writable fields (edit grant required).

        *expected_updated_at* is the optimistic-concurrency
        precondition: the ``updated_at`` the caller loaded. When given
        and the stored record moved on, the repository raises
        :class:`PromptTemplateConflict` (HTTP 409). Omitting it keeps
        the legacy unconditional overwrite.
        """
        current = await self._repository.get(template_id, tenant_id=tenant_id)
        shared = template_access(current, visible_to, also_visible)
        if shared is not None and not shared.at_least(SharePermission.EDIT):
            raise PromptTemplateNotFound(template_id)
        updated = PromptTemplateRecord(
            id=current.id,
            tenant_id=current.tenant_id,
            owner_sub=current.owner_sub,
            created_at=current.created_at,
            updated_at=time.time(),
            **_validated_fields(payload),
        )
        return await self._repository.update(
            updated, expected_updated_at=expected_updated_at
        )

    async def delete(
        self,
        template_id: str,
        *,
        tenant_id: str,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ) -> None:
        """Delete one template (owner-only; shares never delete)."""
        current = await self._repository.get(template_id, tenant_id=tenant_id)
        shared = template_access(current, visible_to, also_visible)
        if shared is not None:
            raise PromptTemplateNotFound(template_id)
        await self._repository.delete(template_id, tenant_id=tenant_id)

    async def owner_sub(self, tenant_id: str, template_id: str) -> str | None:
        """Owner lookup for the share layer (``None`` = unshareable)."""
        try:
            record = await self._repository.get(
                template_id, tenant_id=tenant_id
            )
        except PromptTemplateNotFound:
            return None
        return record.owner_sub
