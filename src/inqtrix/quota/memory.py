"""In-process quota store: the zero-infrastructure default.

Thread-safe like the other memory backends. Counters are keyed by
``(tenant_id, subject_user_id, dimension, period_start)`` — exactly the
Postgres primary key — so both backends keep per-window history and
read the active window identically; lazy rollover falls out of the
keying (a new month is a new key at 0). Limits live in a parallel
dict. All check-then-write happens under the lock, so increments stay
atomic.
"""

from __future__ import annotations

import threading
import uuid
from typing import Sequence

from inqtrix.quota.models import (
    DEFAULT_USER_ID,
    STOCK_PERIOD,
    QuotaDimension,
    current_period_start,
)

_CounterKey = tuple[str, uuid.UUID, QuotaDimension, float]
_LimitKey = tuple[str, uuid.UUID | None, QuotaDimension]


def _active_period(dimension: QuotaDimension, now: float) -> float:
    return STOCK_PERIOD if dimension.is_stock else current_period_start(now)


class MemoryQuotaStore:
    """Thread-safe in-process implementation of the quota store port."""

    def __init__(self) -> None:
        self._counters: dict[_CounterKey, int] = {}
        self._limits: dict[_LimitKey, int] = {}
        self._lock = threading.RLock()

    async def add_usage(
        self,
        *,
        tenant_id: str,
        subject_user_id: uuid.UUID,
        dimension: QuotaDimension,
        period_start: float,
        amount: int,
    ) -> int:
        key: _CounterKey = (tenant_id, subject_user_id, dimension, period_start)
        with self._lock:
            # max(0, ...) clamps a stock release; a new window is simply
            # an unseen key that starts at 0.
            new_used = max(0, self._counters.get(key, 0) + amount)
            self._counters[key] = new_used
            return new_used

    async def read_usage(
        self,
        *,
        tenant_id: str,
        subject_user_ids: Sequence[uuid.UUID],
        dimensions: Sequence[QuotaDimension],
        now: float,
    ) -> dict[uuid.UUID, dict[QuotaDimension, int]]:
        result: dict[uuid.UUID, dict[QuotaDimension, int]] = {}
        with self._lock:
            for sub in subject_user_ids:
                result[sub] = {
                    dimension: self._counters.get(
                        (
                            tenant_id,
                            sub,
                            dimension,
                            _active_period(dimension, now),
                        ),
                        0,
                    )
                    for dimension in dimensions
                }
        return result

    async def reset_usage(
        self,
        *,
        tenant_id: str,
        subject_user_id: uuid.UUID,
        dimension: QuotaDimension,
        now: float,
    ) -> None:
        if dimension.is_stock:
            # Resetting a stock level to 0 would decouple it from real
            # object-store occupancy. Stock is freed by deletion, never
            # reset — a stock reset is a caller bug, surfaced loudly.
            raise ValueError(
                f"cannot reset stock dimension {dimension.value}"
            )
        with self._lock:
            self._counters[
                (
                    tenant_id,
                    subject_user_id,
                    dimension,
                    _active_period(dimension, now),
                )
            ] = 0

    async def get_limits(
        self,
        *,
        tenant_id: str,
        subject_user_ids: Sequence[uuid.UUID | None],
        dimensions: Sequence[QuotaDimension],
    ) -> dict[uuid.UUID | None, dict[QuotaDimension, int]]:
        wanted = set(dimensions)
        result: dict[uuid.UUID | None, dict[QuotaDimension, int]] = {}
        with self._lock:
            for (t_id, subject_user_id, dimension), value in self._limits.items():
                if (
                    t_id == tenant_id
                    and subject_user_id in subject_user_ids
                    and dimension in wanted
                ):
                    result.setdefault(subject_user_id, {})[dimension] = value
        return result

    async def set_limit(
        self,
        *,
        tenant_id: str,
        subject_user_id: uuid.UUID | None,
        dimension: QuotaDimension,
        value: int,
        set_by_user_id: uuid.UUID,
    ) -> None:
        with self._lock:
            self._limits[(tenant_id, subject_user_id, dimension)] = value

    async def clear_limit(
        self,
        *,
        tenant_id: str,
        subject_user_id: uuid.UUID | None,
        dimension: QuotaDimension,
    ) -> None:
        with self._lock:
            self._limits.pop((tenant_id, subject_user_id, dimension), None)

    async def list_subjects(self, *, tenant_id: str) -> list[uuid.UUID]:
        with self._lock:
            user_ids = {
                key[1]
                for key in self._counters
                if key[0] == tenant_id
            }
            user_ids.update(
                key[1] for key in self._limits if key[0] == tenant_id
            )
        user_ids.discard(DEFAULT_USER_ID)
        return sorted(user_ids)
