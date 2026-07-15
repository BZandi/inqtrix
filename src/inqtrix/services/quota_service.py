"""Quota enforcement service — admission, recording, and the UI view.

Constructor-first, injected into the cost-incurring surfaces by the
container (only when quotas are enabled and the auth mode is oidc, so
none/apikey/demo deployments never construct it). Authorization stays
in the AuthorizationService; this service answers a different question —
how much, not who.

Two checks, two records:
  * ``check`` runs BEFORE the cost in the async request layer and
    raises :class:`QuotaExceeded` (HTTP 429) when the action would
    cross the user's effective limit.
  * ``record`` books the actual cost AFTER the fact. Most surfaces are
    synchronous-in-an-async-route and record inline; a research run
    completes on a worker thread, so its token consumption is booked
    through :meth:`record_blocking` (a sync bridge — see the NullPool
    store engine that makes that loop-safe).

Unscoped principals (anonymous/static) are never metered: every entry
resolves them to ``None`` and short-circuits, so existing deployments
stay byte-identical even if a misconfiguration enables quotas outside
oidc.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Sequence

from inqtrix.quota.models import (
    DEFAULT_USER_ID,
    STOCK_PERIOD,
    DimensionUsage,
    QuotaDimension,
    QuotaExceeded,
    QuotaSubject,
    current_period_start,
    effective_limit,
    period_end,
)
from inqtrix.urls import sanitize_log_message

if TYPE_CHECKING:
    from inqtrix.auth.principal import Principal
    from inqtrix.quota.ports import QuotaStore
    from inqtrix.settings import QuotaSettings

log = logging.getLogger("inqtrix")

_UNSCOPED_KINDS = frozenset({"anonymous", "static"})

#: dimension -> (QuotaSettings default field, ceiling field)
_ENV_FIELDS: dict[QuotaDimension, tuple[str, str]] = {
    QuotaDimension.RUNS: ("runs_default", "runs_max"),
    QuotaDimension.LLM_TOKENS: ("llm_tokens_default", "llm_tokens_max"),
    QuotaDimension.EMBEDDING_TOKENS: (
        "embedding_tokens_default",
        "embedding_tokens_max",
    ),
    QuotaDimension.STORED_BYTES: (
        "stored_bytes_default",
        "stored_bytes_max",
    ),
}


def _active_period(dimension: QuotaDimension, now: float) -> float:
    return STOCK_PERIOD if dimension.is_stock else current_period_start(now)


class QuotaService:
    """Per-user usage admission + accounting over a :class:`QuotaStore`."""

    def __init__(
        self,
        *,
        store: "QuotaStore",
        settings: "QuotaSettings",
        clock=None,
    ) -> None:
        self._store = store
        self._settings = settings
        # Injectable clock keeps the month-window tests deterministic.
        self._clock = clock

    # -- quota-account resolution ---------------------------------------- #

    def subject_for(self, principal: "Principal | None") -> QuotaSubject | None:
        """The canonical quota account, or ``None`` when exempt.

        Anonymous/static principals (and a missing principal) are never
        metered — the single bypass site for the whole service.
        """
        if principal is None or principal.kind in _UNSCOPED_KINDS:
            return None
        if principal.user_id is None:
            return None
        return QuotaSubject(
            tenant_id=principal.tenant_id, user_id=principal.user_id
        )

    def _now(self) -> float:
        import time

        return self._clock() if self._clock is not None else time.time()

    def _env(self, dimension: QuotaDimension) -> tuple[int, int]:
        default_field, ceiling_field = _ENV_FIELDS[dimension]
        return (
            int(getattr(self._settings, default_field)),
            int(getattr(self._settings, ceiling_field)),
        )

    def _resolve_limit(
        self,
        dimension: QuotaDimension,
        limits: dict[uuid.UUID | None, dict[QuotaDimension, int]],
        user_id: uuid.UUID,
    ) -> int | None:
        env_default, env_ceiling = self._env(dimension)
        return effective_limit(
            override=limits.get(user_id, {}).get(dimension),
            tenant_default=limits.get(DEFAULT_USER_ID, {}).get(dimension),
            env_default=env_default,
            env_ceiling=env_ceiling,
        )

    # -- admission + accounting ------------------------------------------ #

    async def check(
        self,
        principal: "Principal | None",
        dimension: QuotaDimension,
        amount: int = 1,
    ) -> None:
        """Raise :class:`QuotaExceeded` if *amount* more would exceed
        the caller's effective limit for *dimension*. No-op when the
        principal is exempt."""
        subject = self.subject_for(principal)
        if subject is None:
            return
        now = self._now()
        usage = await self._store.read_usage(
            tenant_id=subject.tenant_id,
            subject_user_ids=[subject.user_id],
            dimensions=[dimension],
            now=now,
        )
        limits = await self._store.get_limits(
            tenant_id=subject.tenant_id,
            subject_user_ids=[subject.user_id, DEFAULT_USER_ID],
            dimensions=[dimension],
        )
        used = usage[subject.user_id][dimension]
        limit = self._resolve_limit(dimension, limits, subject.user_id)
        if limit is not None and used + amount > limit:
            # No Silent Fallbacks (Designprinzip 1): every block is
            # visible in the log, at parity with the per-run token-budget
            # abort. The canonical user UUID remains sanitized for log parity.
            log.warning(
                "Quota-Block: dimension=%s used=%d limit=%d user_id=%s",
                dimension.value,
                used,
                limit,
                sanitize_log_message(str(subject.user_id)),
            )
            raise QuotaExceeded(
                dimension=dimension,
                limit=limit,
                used=used,
                reset_at=period_end(_active_period(dimension, now)),
            )

    async def record(
        self,
        principal: "Principal | None",
        dimension: QuotaDimension,
        amount: int,
    ) -> None:
        """Book *amount* against the caller's counter (no-op when
        exempt, or when *amount* is 0)."""
        subject = self.subject_for(principal)
        if subject is None or amount == 0:
            return
        await self._record_subject(subject, dimension, amount)

    async def record_for_subject(
        self,
        subject: QuotaSubject | None,
        dimension: QuotaDimension,
        amount: int,
    ) -> None:
        """Book *amount* against an EXPLICIT subject, not the caller.

        Used for stock attribution: stored bytes belong to a file's
        OWNER, so an upload charges and a delete frees the owner's
        counter even when a different member performs the delete. A
        negative *amount* releases stock (the store clamps at zero).
        No-op for a missing subject or a zero amount.
        """
        if subject is None or amount == 0:
            return
        await self._record_subject(subject, dimension, amount)

    async def _record_subject(
        self, subject: QuotaSubject, dimension: QuotaDimension, amount: int
    ) -> None:
        """Book *amount* into the store as a NON-FATAL side-effect.

        Accounting must never destroy the primary result: a completed,
        already-paid-for run/answer is worth more than a precise counter.
        A store failure is therefore logged loudly (No Silent Fallbacks)
        and swallowed — an unrecorded spend is a recoverable gap; a lost
        result is not. Admission (:meth:`check`) still raises; only the
        post-hoc recording is best-effort.
        """
        try:
            await self._store.add_usage(
                tenant_id=subject.tenant_id,
                subject_user_id=subject.user_id,
                dimension=dimension,
                period_start=_active_period(dimension, self._now()),
                amount=amount,
            )
        except Exception as exc:  # noqa: BLE001 — recording is non-fatal
            log.warning(
                "Quota-Buchung fehlgeschlagen (Verbrauch nicht gezaehlt): "
                "dimension=%s amount=%d user_id=%s error=%s",
                dimension.value,
                amount,
                sanitize_log_message(str(subject.user_id)),
                sanitize_log_message(exc),
            )

    def record_blocking(
        self, subject: QuotaSubject | None, dimension: QuotaDimension, amount: int
    ) -> None:
        """Synchronous bridge for the run-execution thread.

        A research run finishes off the event loop (a worker thread with
        no running loop), so ``asyncio.run`` is safe here. The Postgres
        store's NullPool engine makes the cross-loop write safe; the
        memory store is loop-indifferent. No-op for an exempt subject.

        Recording is a non-fatal side-effect (see :meth:`_record_subject`):
        a store failure is logged, never raised, so a completed run is
        never lost to a bookkeeping error.
        """
        if subject is None or amount == 0:
            return
        asyncio.run(self._record_subject(subject, dimension, amount))

    # -- the UI / admin view --------------------------------------------- #

    async def usage_for(
        self,
        subject: QuotaSubject,
        dimensions: Sequence[QuotaDimension] | None = None,
    ) -> list[DimensionUsage]:
        """Resolved usage per dimension for one subject (the meter +
        admin row source). Two batched reads regardless of dimension
        count."""
        dims = list(dimensions) if dimensions is not None else list(QuotaDimension)
        now = self._now()
        usage = await self._store.read_usage(
            tenant_id=subject.tenant_id,
            subject_user_ids=[subject.user_id],
            dimensions=dims,
            now=now,
        )
        limits = await self._store.get_limits(
            tenant_id=subject.tenant_id,
            subject_user_ids=[subject.user_id, DEFAULT_USER_ID],
            dimensions=dims,
        )
        return [
            DimensionUsage(
                dimension=dimension,
                used=usage[subject.user_id][dimension],
                limit=self._resolve_limit(dimension, limits, subject.user_id),
                period_start=_active_period(dimension, now),
            )
            for dimension in dims
        ]

    # -- the admin view + mutations -------------------------------------- #

    async def admin_snapshot(self, tenant_id: str) -> dict:
        """The full admin overview for one tenant.

        One snapshot the admin panel renders directly: the operator
        ceilings (read-only bounds) and env defaults per dimension, the
        admin-set tenant default ("for all"), and one row per metered
        user (those with usage or an override; users on the plain default
        need no row). Each user row carries, per dimension,
        the current usage, the raw per-user override (``None`` = none),
        and the resolved effective limit. One user enumeration plus two
        batched reads (usage + limits) regardless of user count.

        The caller (the admin router) enriches users with display
        name/email; this service stays identity-agnostic.
        """
        dims = list(QuotaDimension)
        now = self._now()
        subs = await self._store.list_subjects(tenant_id=tenant_id)
        usage = (
            await self._store.read_usage(
                tenant_id=tenant_id,
                subject_user_ids=subs,
                dimensions=dims,
                now=now,
            )
            if subs
            else {}
        )
        limits = await self._store.get_limits(
            tenant_id=tenant_id,
            subject_user_ids=[*subs, DEFAULT_USER_ID],
            dimensions=dims,
        )
        default_raw = limits.get(DEFAULT_USER_ID, {})
        env = {d: self._env(d) for d in dims}
        return {
            "dimensions": [d.value for d in dims],
            "stock_dimensions": [d.value for d in dims if d.is_stock],
            "ceilings": {d.value: env[d][1] for d in dims},
            "env_defaults": {d.value: env[d][0] for d in dims},
            "tenant_default": {
                d.value: default_raw.get(d) for d in dims
            },
            "subjects": [
                {
                    "user_id": sub,
                    "dimensions": {
                        d.value: {
                            "used": usage.get(sub, {}).get(d, 0),
                            "override": limits.get(sub, {}).get(d),
                            "limit": effective_limit(
                                override=limits.get(sub, {}).get(d),
                                tenant_default=default_raw.get(d),
                                env_default=env[d][0],
                                env_ceiling=env[d][1],
                            ),
                            "period_start": _active_period(d, now),
                        }
                        for d in dims
                    },
                }
                for sub in subs
            ],
        }

    async def set_limit_for(
        self,
        *,
        tenant_id: str,
        subject_user_id: uuid.UUID | None,
        dimension: QuotaDimension,
        value: int,
        set_by_user_id: uuid.UUID,
    ) -> None:
        """Upsert one limit (a per-user override, or the tenant default
        when *subject_user_id* is ``DEFAULT_USER_ID``). ``0`` is an explicit
        unlimited; the operator ceiling still clamps at read time."""
        await self._store.set_limit(
            tenant_id=tenant_id,
            subject_user_id=subject_user_id,
            dimension=dimension,
            value=value,
            set_by_user_id=set_by_user_id,
        )

    async def clear_limit_for(
        self,
        *,
        tenant_id: str,
        subject_user_id: uuid.UUID | None,
        dimension: QuotaDimension,
    ) -> None:
        """Drop one limit row so it falls back to the next layer."""
        await self._store.clear_limit(
            tenant_id=tenant_id,
            subject_user_id=subject_user_id,
            dimension=dimension,
        )

    async def reset_for(
        self,
        *,
        tenant_id: str,
        subject_user_id: uuid.UUID,
        dimension: QuotaDimension,
    ) -> None:
        """Zero one user's CURRENT-window flow usage (admin reset).

        Raises ``ValueError`` for a stock dimension (the store enforces
        it): stock is freed by deletion, never reset.
        """
        await self._store.reset_usage(
            tenant_id=tenant_id,
            subject_user_id=subject_user_id,
            dimension=dimension,
            now=self._now(),
        )

    async def aclose(self) -> None:
        """Release any store-owned resources at application shutdown.

        Only the Postgres store owns an engine; the memory store is a
        no-op. Delegates so the lifespan teardown stays uniform with the
        run store's ``close``.
        """
        closer = getattr(self._store, "aclose", None)
        if closer is not None:
            await closer()
