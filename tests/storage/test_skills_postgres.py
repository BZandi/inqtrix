"""Postgres integration tests for the skill repository (plan M3 `3.1`).

Gated like the sibling suites: a disposable database via
``INQTRIX_TEST_DATABASE_URL``, operations under the restricted app
role. Pins the CRUD roundtrip over the FULL field set (JSONB points and
tool lists included), absence semantics, the compare-and-set update
distinction, and migration 0041 being applied at all.
"""

from __future__ import annotations

import os
import time

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.content.skills import (
    SkillConflict,
    SkillNotFound,
    SkillRecord,
    new_skill_id,
)
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations
from inqtrix.storage.skills_postgres import PostgresSkillRepository

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"


@pytest.fixture(scope="session", autouse=True)
def schema_migrated():
    if TEST_DATABASE_URL:
        run_migrations(TEST_DATABASE_URL)
    yield


@pytest_asyncio.fixture()
async def repository():
    engine = build_engine(TEST_DATABASE_URL)
    factory = build_session_factory(engine)
    async with factory() as session:
        async with session.begin():
            await session.execute(text("DELETE FROM skill_templates"))
    yield PostgresSkillRepository(session_factory=factory, app_role=APP_ROLE)
    await engine.dispose()


def record(**overrides) -> SkillRecord:
    now = time.time()
    base = dict(
        id=new_skill_id(),
        tenant_id="default",
        owner_sub="user-owner",
        label="sprechzettel",
        title="Sprechzettel",
        description="Kompakter Sprechzettel.",
        when_to_use="Fuer Termine.",
        instructions_markdown="Sprechzettel fuer {{anlass}}.",
        clarification_points=(
            {
                "id": "p1",
                "name": "anlass",
                "question": "Welcher Anlass?",
                "options": [
                    {"id": "p1_o1", "label": "Vorstand", "description": ""}
                ],
                "required": True,
                "default_assumption": "Intern",
            },
        ),
        deliverable="talking_points",
        allowed_tools=("search_project_knowledge",),
        requires_plan="never",
        invocation="model_allowed",
        argument_hint="Anlass",
        model_tier="mid",
        effort="low",
        include_in_autocomplete=True,
        created_at=now,
        updated_at=now,
    )
    base.update(overrides)
    return SkillRecord(**base)


@pytest.mark.asyncio
async def test_crud_roundtrip_full_field_set(repository):
    created = await repository.create(record())
    fetched = await repository.get(created.id, tenant_id="default")
    assert fetched == created

    listed = await repository.list_for_tenant(tenant_id="default")
    assert [item.id for item in listed] == [created.id]

    updated = await repository.update(
        SkillRecord(**{**created.__dict__, "title": "Neuer Titel"})
    )
    assert updated.title == "Neuer Titel"
    assert updated.updated_at > created.updated_at
    assert updated.clarification_points == created.clarification_points

    await repository.delete(created.id, tenant_id="default")
    with pytest.raises(SkillNotFound):
        await repository.get(created.id, tenant_id="default")


@pytest.mark.asyncio
async def test_absence_and_foreign_tenant_raise_alike(repository):
    created = await repository.create(record())
    with pytest.raises(SkillNotFound):
        await repository.get("sk_missing", tenant_id="default")
    with pytest.raises(SkillNotFound):
        await repository.get(created.id, tenant_id="tenant-x")
    with pytest.raises(SkillNotFound):
        await repository.delete("sk_missing", tenant_id="default")


@pytest.mark.asyncio
async def test_matching_precondition_updates_stale_one_conflicts(repository):
    created = await repository.create(record(title="v0"))
    advanced = await repository.update(
        SkillRecord(**{**created.__dict__, "title": "v1"}),
        expected_updated_at=created.updated_at,
    )
    assert advanced.title == "v1"

    with pytest.raises(SkillConflict):
        await repository.update(
            SkillRecord(**{**created.__dict__, "title": "stale"}),
            expected_updated_at=created.updated_at,
        )
    with pytest.raises(SkillNotFound):
        await repository.update(
            record(id="sk_missing", title="ghost"),
            expected_updated_at=created.updated_at,
        )
