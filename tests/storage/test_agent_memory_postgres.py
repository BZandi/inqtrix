"""P1 safety net: agent-memory tables work under the app's tenant GUC + RLS.

Migrations 0033/0034 created the ``tenant_isolation`` policy on the WRONG GUC
(``app.tenant_id``, which the app never sets), so under FORCE ROW LEVEL
SECURITY and the NOBYPASSRLS ``inqtrix_app`` role, WITH CHECK rejected every
insert and USING hid every row — ``agent_memory_candidates`` / ``agent_feedback``
were silently dead on Postgres. Migration 0036 repaired the policy to
``inqtrix_current_tenant_id()`` (the GUC the app actually sets, DEFAULT_TENANT
per session). These round-trips fail/return empty against the pre-0036 policy
and pass against it — the regression guard the P1 plan promised.

Live Postgres only (RLS never bites the in-memory store); skipped offline.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.agents.memory_ports import AgentFeedbackRecord, AgentMemoryCandidate
from inqtrix.storage.agent_memory_postgres import (
    PostgresAgentFeedbackStore,
    PostgresAgentMemoryCandidateStore,
)
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations

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
async def wiped():
    engine = build_engine(TEST_DATABASE_URL)
    factory = build_session_factory(engine)
    async with factory() as session:
        async with session.begin():
            bypasses = (
                await session.execute(
                    text(
                        "SELECT rolsuper OR rolbypassrls FROM pg_roles "
                        "WHERE rolname = current_user"
                    )
                )
            ).scalar_one()
            if not bypasses:
                pytest.fail(
                    "INQTRIX_TEST_DATABASE_URL must connect as a "
                    "superuser/BYPASSRLS user to clean RLS-forced tables."
                )
            await session.execute(text("DELETE FROM agent_memory_candidates"))
            await session.execute(text("DELETE FROM agent_feedback"))
    await engine.dispose()
    yield


@pytest_asyncio.fixture()
async def candidate_store(wiped):
    store = PostgresAgentMemoryCandidateStore(
        engine=build_engine(TEST_DATABASE_URL), app_role=APP_ROLE
    )
    yield store
    await store.aclose()


@pytest_asyncio.fixture()
async def feedback_store(wiped):
    store = PostgresAgentFeedbackStore(
        engine=build_engine(TEST_DATABASE_URL), app_role=APP_ROLE
    )
    yield store
    await store.aclose()


@pytest.mark.asyncio
async def test_memory_candidate_round_trips_under_the_app_tenant_guc(
    candidate_store,
):
    """A candidate inserted then listed under the restricted role + FORCE RLS
    survives — the exact path the wrong ``app.tenant_id`` GUC broke (WITH CHECK
    reject on insert, USING hide on read) before migration 0036."""
    candidate = AgentMemoryCandidate(
        candidate_id="cand-1",
        tenant_id="default",
        sub="user-a",
        scope="user",
        category="preference",
        content="prefers concise answers",
        reason="observed across runs",
        confidence=0.8,
        source_run_id="run-1",
    )

    created = await candidate_store.create_candidate(candidate)
    assert created.candidate_id == "cand-1"

    listed = await candidate_store.list_candidates(
        tenant_id="default", sub="user-a"
    )
    assert [c.candidate_id for c in listed] == ["cand-1"]
    assert listed[0].content == "prefers concise answers"


@pytest.mark.asyncio
async def test_feedback_round_trips_under_the_app_tenant_guc(feedback_store):
    """The sibling ``agent_feedback`` table (same 0034 wrong-GUC bug) is also
    readable/writable under the correct GUC after 0036."""
    record = AgentFeedbackRecord(
        feedback_id="fb-1",
        tenant_id="default",
        sub="user-a",
        run_id="run-1",
        feedback="positive",
        reason="clear answer",
    )

    created = await feedback_store.create_feedback(record)
    assert created.feedback_id == "fb-1"

    listed = await feedback_store.list_feedback(tenant_id="default", sub="user-a")
    assert [f.feedback_id for f in listed] == ["fb-1"]
