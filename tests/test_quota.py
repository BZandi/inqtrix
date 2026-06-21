"""M1: quota domain model + the in-memory store.

Pins the resolution rules (the three layers under the operator ceiling),
the calendar-month window math, lazy rollover, atomic increment with
stock-release clamping, reset, and limit storage.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from inqtrix.quota.memory import MemoryQuotaStore
from inqtrix.quota.models import (
    DEFAULT_SUBJECT,
    STOCK_PERIOD,
    DimensionUsage,
    QuotaDimension,
    QuotaExceeded,
    current_period_start,
    effective_limit,
    period_end,
)

JUNE = dt.datetime(2026, 6, 13, 12, 0, tzinfo=dt.timezone.utc).timestamp()
JULY = dt.datetime(2026, 7, 2, 9, 0, tzinfo=dt.timezone.utc).timestamp()
DEC = dt.datetime(2026, 12, 20, 0, 0, tzinfo=dt.timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# Period math
# ---------------------------------------------------------------------------


def test_period_start_is_month_first_utc():
    start = current_period_start(JUNE)
    assert dt.datetime.fromtimestamp(
        start, tz=dt.timezone.utc
    ) == dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)


def test_period_end_rolls_into_next_month_and_year():
    june_start = current_period_start(JUNE)
    assert dt.datetime.fromtimestamp(
        period_end(june_start), tz=dt.timezone.utc
    ) == dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
    dec_start = current_period_start(DEC)
    assert dt.datetime.fromtimestamp(
        period_end(dec_start), tz=dt.timezone.utc
    ) == dt.datetime(2027, 1, 1, tzinfo=dt.timezone.utc)


def test_stock_period_has_no_end():
    assert period_end(STOCK_PERIOD) == 0.0


# ---------------------------------------------------------------------------
# Effective-limit resolution
# ---------------------------------------------------------------------------


def test_override_wins_over_default_wins_over_env():
    assert effective_limit(
        override=300, tenant_default=200, env_default=100, env_ceiling=0
    ) == 300
    assert effective_limit(
        override=None, tenant_default=200, env_default=100, env_ceiling=0
    ) == 200
    assert effective_limit(
        override=None, tenant_default=None, env_default=100, env_ceiling=0
    ) == 100


def test_zero_is_explicit_unlimited_not_fall_through():
    # An admin who sets a user to 0 means unlimited (capped only by the
    # ceiling), NOT "fall through to the tenant default".
    assert effective_limit(
        override=0, tenant_default=200, env_default=100, env_ceiling=0
    ) is None
    assert effective_limit(
        override=0, tenant_default=200, env_default=100, env_ceiling=500
    ) == 500


def test_ceiling_caps_every_layer():
    assert effective_limit(
        override=900, tenant_default=None, env_default=0, env_ceiling=500
    ) == 500
    assert effective_limit(
        override=None, tenant_default=None, env_default=0, env_ceiling=500
    ) == 500  # env_default unlimited, ceiling still binds
    assert effective_limit(
        override=None, tenant_default=None, env_default=0, env_ceiling=0
    ) is None  # nothing set anywhere -> unlimited


# ---------------------------------------------------------------------------
# DimensionUsage view
# ---------------------------------------------------------------------------


def test_dimension_usage_remaining_and_allows():
    capped = DimensionUsage(
        dimension=QuotaDimension.RUNS, used=198, limit=200, period_start=JUNE
    )
    assert capped.remaining == 2
    assert capped.allows(2) is True
    assert capped.allows(3) is False
    unlimited = DimensionUsage(
        dimension=QuotaDimension.RUNS, used=999, limit=None, period_start=JUNE
    )
    assert unlimited.remaining is None
    assert unlimited.allows(10_000) is True


def test_quota_exceeded_carries_facts():
    exc = QuotaExceeded(
        dimension=QuotaDimension.LLM_TOKENS,
        limit=5_000_000,
        used=5_000_000,
        reset_at=period_end(current_period_start(JUNE)),
    )
    assert exc.dimension is QuotaDimension.LLM_TOKENS
    assert exc.limit == 5_000_000
    assert "llm_tokens" in str(exc)


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------


@pytest.fixture()
def store():
    return MemoryQuotaStore()


def _add(store, sub, dim, period, amount):
    return asyncio.run(
        store.add_usage(
            tenant_id="default",
            subject_sub=sub,
            dimension=dim,
            period_start=period,
            amount=amount,
        )
    )


def _read(store, subs, dims, now):
    return asyncio.run(
        store.read_usage(
            tenant_id="default",
            subject_subs=subs,
            dimensions=dims,
            now=now,
        )
    )


def test_increment_accumulates_within_a_window(store):
    june = current_period_start(JUNE)
    assert _add(store, "u1", QuotaDimension.RUNS, june, 1) == 1
    assert _add(store, "u1", QuotaDimension.RUNS, june, 1) == 2
    usage = _read(store, ["u1"], [QuotaDimension.RUNS], JUNE)
    assert usage["u1"][QuotaDimension.RUNS] == 2


def test_lazy_rollover_starts_next_month_at_zero(store):
    june = current_period_start(JUNE)
    july = current_period_start(JULY)
    _add(store, "u1", QuotaDimension.RUNS, june, 5)
    # July reads as 0 (the June counter belongs to a past window) without
    # being rewritten.
    assert _read(store, ["u1"], [QuotaDimension.RUNS], JULY)["u1"][
        QuotaDimension.RUNS
    ] == 0
    assert _add(store, "u1", QuotaDimension.RUNS, july, 1) == 1
    # June's history is untouched.
    assert _read(store, ["u1"], [QuotaDimension.RUNS], JUNE)["u1"][
        QuotaDimension.RUNS
    ] == 5


def test_stock_release_clamps_at_zero_and_never_rolls(store):
    _add(store, "u1", QuotaDimension.STORED_BYTES, STOCK_PERIOD, 1000)
    assert _add(store, "u1", QuotaDimension.STORED_BYTES, STOCK_PERIOD, -400) == 600
    # Over-release clamps at 0, not negative.
    assert _add(store, "u1", QuotaDimension.STORED_BYTES, STOCK_PERIOD, -5000) == 0
    # Stock ignores the calendar month entirely.
    assert _read(store, ["u1"], [QuotaDimension.STORED_BYTES], JULY)["u1"][
        QuotaDimension.STORED_BYTES
    ] == 0


def test_reset_zeroes_current_window(store):
    june = current_period_start(JUNE)
    _add(store, "u1", QuotaDimension.RUNS, june, 200)
    asyncio.run(
        store.reset_usage(
            tenant_id="default",
            subject_sub="u1",
            dimension=QuotaDimension.RUNS,
            now=JUNE,
        )
    )
    assert _read(store, ["u1"], [QuotaDimension.RUNS], JUNE)["u1"][
        QuotaDimension.RUNS
    ] == 0


def test_reset_rejects_stock_dimension(store):
    """Stock is freed by deletion, never reset — the store enforces it."""
    with pytest.raises(ValueError):
        asyncio.run(
            store.reset_usage(
                tenant_id="default",
                subject_sub="u1",
                dimension=QuotaDimension.STORED_BYTES,
                now=JUNE,
            )
        )


def test_limits_roundtrip_and_clear(store):
    asyncio.run(
        store.set_limit(
            tenant_id="default",
            subject_sub="u1",
            dimension=QuotaDimension.RUNS,
            value=400,
            set_by_sub="owner",
        )
    )
    asyncio.run(
        store.set_limit(
            tenant_id="default",
            subject_sub=DEFAULT_SUBJECT,
            dimension=QuotaDimension.RUNS,
            value=200,
            set_by_sub="owner",
        )
    )
    limits = asyncio.run(
        store.get_limits(
            tenant_id="default",
            subject_subs=["u1", DEFAULT_SUBJECT],
            dimensions=[QuotaDimension.RUNS],
        )
    )
    assert limits["u1"][QuotaDimension.RUNS] == 400
    assert limits[DEFAULT_SUBJECT][QuotaDimension.RUNS] == 200

    asyncio.run(
        store.clear_limit(
            tenant_id="default",
            subject_sub="u1",
            dimension=QuotaDimension.RUNS,
        )
    )
    after = asyncio.run(
        store.get_limits(
            tenant_id="default",
            subject_subs=["u1"],
            dimensions=[QuotaDimension.RUNS],
        )
    )
    assert "u1" not in after


def test_usage_isolated_per_tenant_and_subject(store):
    june = current_period_start(JUNE)
    _add(store, "u1", QuotaDimension.RUNS, june, 3)
    usage = _read(store, ["u1", "u2"], [QuotaDimension.RUNS], JUNE)
    assert usage["u1"][QuotaDimension.RUNS] == 3
    assert usage["u2"][QuotaDimension.RUNS] == 0


def test_list_subjects_unions_usage_and_limits(store):
    """The admin-overview source: usage subjects union limit subjects,
    with the tenant-default sentinel excluded (memory/Postgres parity)."""
    june = current_period_start(JUNE)
    _add(store, "user-usage", QuotaDimension.RUNS, june, 1)
    asyncio.run(
        store.set_limit(
            tenant_id="default",
            subject_sub="user-limit",
            dimension=QuotaDimension.RUNS,
            value=5,
            set_by_sub="owner",
        )
    )
    asyncio.run(
        store.set_limit(
            tenant_id="default",
            subject_sub=DEFAULT_SUBJECT,
            dimension=QuotaDimension.RUNS,
            value=2,
            set_by_sub="owner",
        )
    )
    subs = asyncio.run(store.list_subjects(tenant_id="default"))
    assert set(subs) == {"user-usage", "user-limit"}
    assert DEFAULT_SUBJECT not in subs
