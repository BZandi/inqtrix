"""Postgres integration tests for the account-preferences store (gated, M6c)."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import text

from inqtrix.project.account_preferences_postgres import PostgresAccountPreferencesStore
from inqtrix.storage.account_orm import account_preferences
from inqtrix.storage.db import build_engine, build_session_factory
from inqtrix.storage.migrate import run_migrations

TEST_DATABASE_URL = os.environ.get("INQTRIX_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="INQTRIX_TEST_DATABASE_URL not set (Postgres integration)",
)

APP_ROLE = "inqtrix_app"


@pytest.fixture(scope="session", autouse=True)
def account_schema_migrated():
    if TEST_DATABASE_URL:
        run_migrations(TEST_DATABASE_URL)
    yield


@pytest_asyncio.fixture()
async def store():
    engine = build_engine(TEST_DATABASE_URL)
    factory = build_session_factory(engine)
    async with factory() as session:
        async with session.begin():
            bypasses = (
                await session.execute(
                    text("SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user")
                )
            ).scalar_one()
            if not bypasses:
                pytest.fail("INQTRIX_TEST_DATABASE_URL must connect as superuser/BYPASSRLS.")
            await session.execute(account_preferences.delete())
    prefs_store = PostgresAccountPreferencesStore(engine=engine, app_role=APP_ROLE)
    yield prefs_store
    await prefs_store.aclose()


@pytest.mark.asyncio
async def test_get_none_then_upsert_roundtrip(store) -> None:
    assert await store.get_preferences(sub="u") is None
    await store.upsert_preferences(
        sub="u", contrast_mode="high", locale="de", theme="dark",
        theme_preset="sage", user_bubble_tone="mint", updated_at=1.0,
    )
    prefs = await store.get_preferences(sub="u")
    assert (
        prefs.theme,
        prefs.locale,
        prefs.contrast_mode,
        prefs.theme_preset,
        prefs.user_bubble_tone,
    ) == (
        "dark", "de", "high", "sage", "mint"
    )


@pytest.mark.asyncio
async def test_upsert_replaces_row_for_same_user(store) -> None:
    await store.upsert_preferences(
        sub="u", contrast_mode="standard", locale="en", theme="dark",
        theme_preset="slate", user_bubble_tone="mint", updated_at=1.0,
    )
    await store.upsert_preferences(
        sub="u", contrast_mode="high", locale="de", theme="light",
        theme_preset="graphite", user_bubble_tone="orange", updated_at=2.0,
    )
    prefs = await store.get_preferences(sub="u")
    assert (prefs.theme, prefs.theme_preset, prefs.user_bubble_tone, prefs.updated_at) == (
        "light", "graphite", "orange", 2.0
    )
    # Still a singleton: the upsert updated in place, no second row.
    async with store._session() as session:
        from sqlalchemy import func, select

        count = (await session.execute(
            select(func.count()).select_from(account_preferences)
        )).scalar_one()
    assert count == 1


@pytest.mark.asyncio
async def test_distinct_users_independent(store) -> None:
    await store.upsert_preferences(
        sub="u-a", contrast_mode="high", locale="de", theme="dark",
        theme_preset="slate", user_bubble_tone="sky", updated_at=1.0,
    )
    await store.upsert_preferences(
        sub="u-b", contrast_mode="standard", locale="en", theme="light",
        theme_preset="standard", user_bubble_tone="gray", updated_at=1.0,
    )
    assert (await store.get_preferences(sub="u-a")).theme == "dark"
    assert (await store.get_preferences(sub="u-b")).theme == "light"


@pytest.mark.asyncio
async def test_db_check_rejects_out_of_domain_theme(store) -> None:
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        await store.upsert_preferences(
            sub="u", contrast_mode="standard", locale="en", theme="neon",
            theme_preset="standard", user_bubble_tone="gray", updated_at=1.0,
        )


@pytest.mark.asyncio
async def test_db_check_rejects_out_of_domain_user_bubble_tone(store) -> None:
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        await store.upsert_preferences(
            sub="u", contrast_mode="standard", locale="en", theme="system",
            theme_preset="standard", user_bubble_tone="rainbow", updated_at=1.0,
        )
