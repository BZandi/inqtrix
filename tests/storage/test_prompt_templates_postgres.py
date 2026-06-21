"""Postgres integration tests for the prompt-template repository.

Gated like the sibling suites: a disposable database via
``INQTRIX_TEST_DATABASE_URL``, operations under the restricted app
role. Pins the CRUD roundtrip, absence semantics, the LWW
``updated_at`` advance, and migration 0006 being applied.
"""

from __future__ import annotations

import os
import time

import pytest
import pytest_asyncio
from sqlalchemy import text

from dataclasses import replace

from inqtrix.content.prompt_templates import (
    PromptTemplateConflict,
    PromptTemplateNotFound,
    PromptTemplateRecord,
    new_template_id,
)
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations
from inqtrix.storage.prompt_templates_postgres import (
    PostgresPromptTemplateRepository,
)

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
            await session.execute(text("DELETE FROM prompt_templates"))
    yield PostgresPromptTemplateRepository(
        session_factory=factory, app_role=APP_ROLE
    )
    await engine.dispose()


def record(**overrides) -> PromptTemplateRecord:
    now = time.time()
    base = dict(
        id=new_template_id(),
        tenant_id="default",
        owner_sub="user-owner",
        title="Executive Briefing",
        label="briefing",
        category="instruction",
        content_markdown="Fasse die Lage zusammen.",
        visibility={"chat": True, "editor": False},
        include_in_autocomplete=True,
        created_at=now,
        updated_at=now,
    )
    base.update(overrides)
    return PromptTemplateRecord(**base)


@pytest.mark.asyncio
async def test_crud_roundtrip(repository):
    created = await repository.create(record())
    fetched = await repository.get(created.id, tenant_id="default")
    assert fetched == created

    listed = await repository.list_for_tenant(tenant_id="default")
    assert [item.id for item in listed] == [created.id]

    updated = await repository.update(
        PromptTemplateRecord(
            **{
                **created.__dict__,
                "title": "Neuer Titel",
            }
        )
    )
    assert updated.title == "Neuer Titel"
    # The repository stamps the LWW anchor itself.
    assert updated.updated_at > created.updated_at

    await repository.delete(created.id, tenant_id="default")
    with pytest.raises(PromptTemplateNotFound):
        await repository.get(created.id, tenant_id="default")


@pytest.mark.asyncio
async def test_absence_and_foreign_tenant_raise_alike(repository):
    created = await repository.create(record())
    with pytest.raises(PromptTemplateNotFound):
        await repository.get("pt_missing", tenant_id="default")
    with pytest.raises(PromptTemplateNotFound):
        await repository.get(created.id, tenant_id="tenant-x")
    with pytest.raises(PromptTemplateNotFound):
        await repository.delete("pt_missing", tenant_id="default")


@pytest.mark.asyncio
async def test_ownerless_record_roundtrips(repository):
    created = await repository.create(record(owner_sub=None))
    fetched = await repository.get(created.id, tenant_id="default")
    assert fetched.owner_sub is None


@pytest.mark.asyncio
async def test_matching_precondition_updates_stale_one_conflicts(repository):
    """The guarded UPDATE is atomic compare-and-set, with a precise
    404-vs-409 distinction."""
    created = await repository.create(record(title="v0"))

    # Matching precondition: the write lands and advances updated_at.
    fresh = await repository.update(
        replace(created, title="v1"),
        expected_updated_at=created.updated_at,
    )
    assert fresh.title == "v1"
    assert fresh.updated_at > created.updated_at

    # Stale precondition (the original anchor): conflict, no overwrite.
    with pytest.raises(PromptTemplateConflict):
        await repository.update(
            replace(created, title="v2"),
            expected_updated_at=created.updated_at,
        )
    assert (
        await repository.get(created.id, tenant_id="default")
    ).title == "v1"

    # No precondition: unconditional overwrite still works (legacy LWW).
    forced = await repository.update(replace(fresh, title="v3"))
    assert forced.title == "v3"


@pytest.mark.asyncio
async def test_precondition_on_missing_row_is_not_found_not_conflict(repository):
    """A precondition against a vanished row is 404, never a phantom 409."""
    ghost = record(id="pt_ghost")
    with pytest.raises(PromptTemplateNotFound):
        await repository.update(ghost, expected_updated_at=123.0)
