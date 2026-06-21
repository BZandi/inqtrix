"""Postgres implementation of the prompt-template repository.

Same conventions as the file registry: every operation runs in one
tenant-scoped transaction under the restricted app role, RLS is the
second defense behind the explicit ``tenant_id`` predicates, and
absence equals foreign-tenant invisibility
(:class:`PromptTemplateNotFound` for both).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, insert, select, update

from inqtrix.content.prompt_templates import (
    PromptTemplateConflict,
    PromptTemplateNotFound,
    PromptTemplateRecord,
)
from inqtrix.storage.db import tenant_session
from inqtrix.storage.prompt_template_orm import prompt_templates

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def _record_from_row(row: Any) -> PromptTemplateRecord:
    return PromptTemplateRecord(
        id=row.id,
        tenant_id=row.tenant_id,
        owner_sub=row.owner_sub,
        title=row.title,
        label=row.label,
        category=row.category,
        content_markdown=row.content_markdown,
        visibility=dict(row.visibility or {}),
        include_in_autocomplete=row.include_in_autocomplete,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class PostgresPromptTemplateRepository:
    """Prompt templates over the ``prompt_templates`` table.

    Args:
        session_factory: Factory from
            :func:`inqtrix.storage.db.build_session_factory`.
        app_role: Restricted Postgres role for the tenant sessions.
    """

    def __init__(
        self,
        *,
        session_factory: "async_sessionmaker[AsyncSession]",
        app_role: str,
    ) -> None:
        self._session_factory = session_factory
        self._app_role = app_role

    def _session(
        self, tenant_id: str
    ) -> "AbstractAsyncContextManager[AsyncSession]":
        return tenant_session(
            self._session_factory,
            tenant_id=tenant_id,
            app_role=self._app_role,
        )

    async def create(self, record: PromptTemplateRecord) -> PromptTemplateRecord:
        async with self._session(record.tenant_id) as session:
            await session.execute(
                insert(prompt_templates).values(
                    id=record.id,
                    tenant_id=record.tenant_id,
                    owner_sub=record.owner_sub,
                    title=record.title,
                    label=record.label,
                    category=record.category,
                    content_markdown=record.content_markdown,
                    visibility=record.visibility,
                    include_in_autocomplete=record.include_in_autocomplete,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                )
            )
        return record

    async def get(
        self, template_id: str, *, tenant_id: str
    ) -> PromptTemplateRecord:
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    select(prompt_templates).where(
                        prompt_templates.c.tenant_id == tenant_id,
                        prompt_templates.c.id == template_id,
                    )
                )
            ).one_or_none()
        if row is None:
            raise PromptTemplateNotFound(template_id)
        return _record_from_row(row)

    async def list_for_tenant(
        self, *, tenant_id: str
    ) -> list[PromptTemplateRecord]:
        async with self._session(tenant_id) as session:
            rows = (
                await session.execute(
                    select(prompt_templates)
                    .where(prompt_templates.c.tenant_id == tenant_id)
                    .order_by(prompt_templates.c.created_at.desc())
                )
            ).all()
        return [_record_from_row(row) for row in rows]

    async def update(
        self,
        record: PromptTemplateRecord,
        *,
        expected_updated_at: float | None = None,
    ) -> PromptTemplateRecord:
        now = time.time()
        async with self._session(record.tenant_id) as session:
            where = [
                prompt_templates.c.tenant_id == record.tenant_id,
                prompt_templates.c.id == record.id,
            ]
            # The precondition rides INSIDE the UPDATE's WHERE, so the
            # compare-and-set is one atomic statement (no read-then-write
            # race). None = unconditional overwrite (legacy callers).
            if expected_updated_at is not None:
                where.append(
                    prompt_templates.c.updated_at == expected_updated_at
                )
            row = (
                await session.execute(
                    update(prompt_templates)
                    .where(*where)
                    .values(
                        title=record.title,
                        label=record.label,
                        category=record.category,
                        content_markdown=record.content_markdown,
                        visibility=record.visibility,
                        include_in_autocomplete=record.include_in_autocomplete,
                        updated_at=now,
                    )
                    .returning(prompt_templates)
                )
            ).one_or_none()
            if row is not None:
                return _record_from_row(row)
            # No row matched. With a precondition this is ambiguous —
            # either the record vanished (404) or it moved on (409).
            # One existence probe in the SAME transaction disambiguates.
            if expected_updated_at is not None:
                exists = (
                    await session.execute(
                        select(prompt_templates.c.id).where(
                            prompt_templates.c.tenant_id == record.tenant_id,
                            prompt_templates.c.id == record.id,
                        )
                    )
                ).first()
                if exists is not None:
                    raise PromptTemplateConflict(record.id)
        raise PromptTemplateNotFound(record.id)

    async def delete(self, template_id: str, *, tenant_id: str) -> None:
        async with self._session(tenant_id) as session:
            result = await session.execute(
                delete(prompt_templates).where(
                    prompt_templates.c.tenant_id == tenant_id,
                    prompt_templates.c.id == template_id,
                )
            )
        if not result.rowcount:
            raise PromptTemplateNotFound(template_id)
