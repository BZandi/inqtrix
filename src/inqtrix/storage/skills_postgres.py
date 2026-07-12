"""Postgres implementation of the skill repository (plan M3 `3.1`).

Same conventions as the prompt-template repository: every operation
runs in one tenant-scoped transaction under the restricted app role,
RLS is the second defense behind the explicit ``tenant_id`` predicates,
and absence equals foreign-tenant invisibility (:class:`SkillNotFound`
for both). The optimistic-concurrency precondition rides inside the
UPDATE's WHERE (one atomic compare-and-set).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, insert, select, update

from inqtrix.content.skills import (
    SkillConflict,
    SkillNotFound,
    SkillRecord,
)
from inqtrix.storage.db import tenant_session
from inqtrix.storage.skill_orm import skill_templates

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def _record_from_row(row: Any) -> SkillRecord:
    return SkillRecord(
        id=row.id,
        tenant_id=row.tenant_id,
        owner_sub=row.owner_sub,
        label=row.label,
        title=row.title,
        description=row.description,
        when_to_use=row.when_to_use,
        instructions_markdown=row.instructions_markdown,
        clarification_points=tuple(row.clarification_points or []),
        deliverable=row.deliverable,
        allowed_tools=tuple(row.allowed_tools or []),
        requires_plan=row.requires_plan,
        invocation=row.invocation,
        argument_hint=row.argument_hint,
        model_tier=row.model_tier,
        effort=row.effort,
        include_in_autocomplete=row.include_in_autocomplete,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _writable_values(record: SkillRecord) -> dict[str, Any]:
    return {
        "label": record.label,
        "title": record.title,
        "description": record.description,
        "when_to_use": record.when_to_use,
        "instructions_markdown": record.instructions_markdown,
        "clarification_points": list(record.clarification_points),
        "deliverable": record.deliverable,
        "allowed_tools": list(record.allowed_tools),
        "requires_plan": record.requires_plan,
        "invocation": record.invocation,
        "argument_hint": record.argument_hint,
        "model_tier": record.model_tier,
        "effort": record.effort,
        "include_in_autocomplete": record.include_in_autocomplete,
    }


class PostgresSkillRepository:
    """Skills over the ``skill_templates`` table.

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

    async def create(self, record: SkillRecord) -> SkillRecord:
        async with self._session(record.tenant_id) as session:
            await session.execute(
                insert(skill_templates).values(
                    id=record.id,
                    tenant_id=record.tenant_id,
                    owner_sub=record.owner_sub,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    **_writable_values(record),
                )
            )
        return record

    async def get(self, skill_id: str, *, tenant_id: str) -> SkillRecord:
        async with self._session(tenant_id) as session:
            row = (
                await session.execute(
                    select(skill_templates).where(
                        skill_templates.c.tenant_id == tenant_id,
                        skill_templates.c.id == skill_id,
                    )
                )
            ).one_or_none()
        if row is None:
            raise SkillNotFound(skill_id)
        return _record_from_row(row)

    async def list_for_tenant(self, *, tenant_id: str) -> list[SkillRecord]:
        async with self._session(tenant_id) as session:
            rows = (
                await session.execute(
                    select(skill_templates)
                    .where(skill_templates.c.tenant_id == tenant_id)
                    .order_by(skill_templates.c.created_at.desc())
                )
            ).all()
        return [_record_from_row(row) for row in rows]

    async def update(
        self,
        record: SkillRecord,
        *,
        expected_updated_at: float | None = None,
    ) -> SkillRecord:
        now = time.time()
        async with self._session(record.tenant_id) as session:
            where = [
                skill_templates.c.tenant_id == record.tenant_id,
                skill_templates.c.id == record.id,
            ]
            if expected_updated_at is not None:
                where.append(
                    skill_templates.c.updated_at == expected_updated_at
                )
            row = (
                await session.execute(
                    update(skill_templates)
                    .where(*where)
                    .values(updated_at=now, **_writable_values(record))
                    .returning(skill_templates)
                )
            ).one_or_none()
            if row is not None:
                return _record_from_row(row)
            # No row matched: either vanished (404) or moved on (409) —
            # one existence probe in the SAME transaction disambiguates.
            if expected_updated_at is not None:
                exists = (
                    await session.execute(
                        select(skill_templates.c.id).where(
                            skill_templates.c.tenant_id == record.tenant_id,
                            skill_templates.c.id == record.id,
                        )
                    )
                ).first()
                if exists is not None:
                    raise SkillConflict(record.id)
        raise SkillNotFound(record.id)

    async def delete(self, skill_id: str, *, tenant_id: str) -> None:
        async with self._session(tenant_id) as session:
            result = await session.execute(
                delete(skill_templates).where(
                    skill_templates.c.tenant_id == tenant_id,
                    skill_templates.c.id == skill_id,
                )
            )
        if not result.rowcount:
            raise SkillNotFound(skill_id)
