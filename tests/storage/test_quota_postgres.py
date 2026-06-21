"""Postgres integration tests for the quota store (gated suite).

Parity with the memory store: atomic increment, lazy monthly rollover
that leaves history intact, stock-release clamping, reset, and limit
round-trip — all under the restricted app role with RLS as the second
defense.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.quota.models import (
    DEFAULT_SUBJECT,
    STOCK_PERIOD,
    QuotaDimension,
    current_period_start,
)
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations
from inqtrix.storage.quota_postgres import PostgresQuotaStore

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"

JUNE = dt.datetime(2026, 6, 13, 12, 0, tzinfo=dt.timezone.utc).timestamp()
JULY = dt.datetime(2026, 7, 2, 9, 0, tzinfo=dt.timezone.utc).timestamp()


@pytest.fixture(scope="session", autouse=True)
def schema_migrated():
    if TEST_DATABASE_URL:
        run_migrations(TEST_DATABASE_URL)
    yield


@pytest_asyncio.fixture()
async def store():
    engine = build_engine(TEST_DATABASE_URL)
    factory = build_session_factory(engine)
    async with factory() as session:
        async with session.begin():
            await session.execute(text("DELETE FROM quota_usage_counters"))
            await session.execute(text("DELETE FROM quota_limits"))
    await engine.dispose()
    store = PostgresQuotaStore(
        database_url=TEST_DATABASE_URL, app_role=APP_ROLE
    )
    yield store
    # The store owns its NullPool engine; dispose it so the gated suite
    # leaves no engine lingering (parity with the other repo fixtures).
    await store.aclose()


@pytest.mark.asyncio
async def test_increment_accumulates_and_returns_total(store):
    june = current_period_start(JUNE)
    assert await store.add_usage(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.RUNS, period_start=june, amount=1,
    ) == 1
    assert await store.add_usage(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.RUNS, period_start=june, amount=2,
    ) == 3
    usage = await store.read_usage(
        tenant_id="default", subject_subs=["u1"],
        dimensions=[QuotaDimension.RUNS], now=JUNE,
    )
    assert usage["u1"][QuotaDimension.RUNS] == 3


@pytest.mark.asyncio
async def test_monthly_rollover_keeps_history(store):
    june = current_period_start(JUNE)
    july = current_period_start(JULY)
    await store.add_usage(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.RUNS, period_start=june, amount=5,
    )
    # New window reads 0; June row survives as history.
    july_usage = await store.read_usage(
        tenant_id="default", subject_subs=["u1"],
        dimensions=[QuotaDimension.RUNS], now=JULY,
    )
    assert july_usage["u1"][QuotaDimension.RUNS] == 0
    await store.add_usage(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.RUNS, period_start=july, amount=1,
    )
    june_usage = await store.read_usage(
        tenant_id="default", subject_subs=["u1"],
        dimensions=[QuotaDimension.RUNS], now=JUNE,
    )
    assert june_usage["u1"][QuotaDimension.RUNS] == 5


@pytest.mark.asyncio
async def test_stock_release_clamps_at_zero(store):
    await store.add_usage(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.STORED_BYTES, period_start=STOCK_PERIOD,
        amount=1000,
    )
    assert await store.add_usage(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.STORED_BYTES, period_start=STOCK_PERIOD,
        amount=-1500,
    ) == 0


@pytest.mark.asyncio
async def test_reset_zeroes_current_window(store):
    june = current_period_start(JUNE)
    await store.add_usage(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.RUNS, period_start=june, amount=200,
    )
    await store.reset_usage(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.RUNS, now=JUNE,
    )
    usage = await store.read_usage(
        tenant_id="default", subject_subs=["u1"],
        dimensions=[QuotaDimension.RUNS], now=JUNE,
    )
    assert usage["u1"][QuotaDimension.RUNS] == 0


@pytest.mark.asyncio
async def test_limits_roundtrip_and_clear(store):
    await store.set_limit(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.RUNS, value=400, set_by_sub="owner",
    )
    await store.set_limit(
        tenant_id="default", subject_sub=DEFAULT_SUBJECT,
        dimension=QuotaDimension.RUNS, value=200, set_by_sub="owner",
    )
    limits = await store.get_limits(
        tenant_id="default", subject_subs=["u1", DEFAULT_SUBJECT],
        dimensions=[QuotaDimension.RUNS],
    )
    assert limits["u1"][QuotaDimension.RUNS] == 400
    assert limits[DEFAULT_SUBJECT][QuotaDimension.RUNS] == 200
    # Re-set overwrites (upsert), not duplicates.
    await store.set_limit(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.RUNS, value=450, set_by_sub="owner",
    )
    limits = await store.get_limits(
        tenant_id="default", subject_subs=["u1"],
        dimensions=[QuotaDimension.RUNS],
    )
    assert limits["u1"][QuotaDimension.RUNS] == 450
    await store.clear_limit(
        tenant_id="default", subject_sub="u1",
        dimension=QuotaDimension.RUNS,
    )
    after = await store.get_limits(
        tenant_id="default", subject_subs=["u1"],
        dimensions=[QuotaDimension.RUNS],
    )
    assert "u1" not in after


@pytest.mark.asyncio
async def test_reset_rejects_stock_dimension(store):
    with pytest.raises(ValueError):
        await store.reset_usage(
            tenant_id="default", subject_sub="u1",
            dimension=QuotaDimension.STORED_BYTES, now=JUNE,
        )


@pytest.mark.asyncio
async def test_list_subjects_unions_usage_and_limits(store):
    june = current_period_start(JUNE)
    await store.add_usage(
        tenant_id="default", subject_sub="user-usage",
        dimension=QuotaDimension.RUNS, period_start=june, amount=1,
    )
    await store.set_limit(
        tenant_id="default", subject_sub="user-limit",
        dimension=QuotaDimension.RUNS, value=5, set_by_sub="owner",
    )
    # The tenant default sentinel must NOT appear as a subject.
    await store.set_limit(
        tenant_id="default", subject_sub=DEFAULT_SUBJECT,
        dimension=QuotaDimension.RUNS, value=2, set_by_sub="owner",
    )
    subs = await store.list_subjects(tenant_id="default")
    assert set(subs) == {"user-usage", "user-limit"}
    assert DEFAULT_SUBJECT not in subs
