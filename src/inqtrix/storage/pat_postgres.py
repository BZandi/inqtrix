"""Postgres backend for personal access tokens.

Same conventions as the other storage backends: every operation runs
inside :func:`~inqtrix.storage.db.tenant_session` (restricted role +
tenant GUC, forced RLS as the second defense layer), guarded UPDATEs
make revocation and the last-used throttle replica-safe, and the
store shares the identity bundle's HTTP-loop engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import or_, select, update

from inqtrix.auth.pat import PersonalAccessToken
from inqtrix.storage.db import tenant_session
from inqtrix.storage.pat_orm import personal_access_tokens as pats

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

DEFAULT_TENANT = "default"


def _record_from_row(row) -> PersonalAccessToken:
    return PersonalAccessToken(
        token_id=row.token_id,
        tenant_id=row.tenant_id,
        owner_issuer=row.owner_issuer,
        owner_sub=row.owner_sub,
        name=row.name,
        secret_hmac=row.secret_hmac,
        created_at=row.created_at,
        expires_at=row.expires_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
        scopes=tuple(row.scopes or ()),
    )


class PostgresPatStore:
    """Durable token store (tokens survive restarts and replicas).

    Args:
        session_factory: Async session factory on the HTTP loop's
            engine.
        app_role: Restricted role assumed per transaction.
    """

    def __init__(
        self,
        *,
        session_factory: "async_sessionmaker[AsyncSession]",
        app_role: str,
    ) -> None:
        self._session_factory = session_factory
        self._app_role = app_role

    def _scope(self):
        return tenant_session(
            self._session_factory,
            tenant_id=DEFAULT_TENANT,
            app_role=self._app_role,
        )

    async def create(self, token: PersonalAccessToken) -> None:
        async with self._scope() as db:
            await db.execute(
                pats.insert().values(
                    token_id=token.token_id,
                    tenant_id=token.tenant_id,
                    owner_issuer=token.owner_issuer,
                    owner_sub=token.owner_sub,
                    name=token.name,
                    secret_hmac=token.secret_hmac,
                    scopes=list(token.scopes),
                    created_at=token.created_at,
                    expires_at=token.expires_at,
                    last_used_at=token.last_used_at,
                    revoked_at=token.revoked_at,
                )
            )

    async def get(self, token_id: str) -> PersonalAccessToken | None:
        async with self._scope() as db:
            row = (
                await db.execute(
                    select(pats).where(pats.c.token_id == token_id)
                )
            ).first()
        return _record_from_row(row) if row is not None else None

    async def list_for_owner(
        self, *, tenant_id: str, owner_issuer: str, owner_sub: str
    ) -> tuple[PersonalAccessToken, ...]:
        async with self._scope() as db:
            rows = (
                await db.execute(
                    select(pats)
                    .where(
                        pats.c.tenant_id == tenant_id,
                        pats.c.owner_issuer == owner_issuer,
                        pats.c.owner_sub == owner_sub,
                        pats.c.revoked_at.is_(None),
                    )
                    .order_by(pats.c.created_at.desc())
                )
            ).all()
        return tuple(_record_from_row(row) for row in rows)

    async def revoke(
        self,
        *,
        tenant_id: str,
        token_id: str,
        owner_issuer: str,
        owner_sub: str,
        now: float,
    ) -> bool:
        """Guarded soft-revoke: only a LIVE row of THIS owner flips.

        The guard makes concurrent double-revokes flip exactly once
        across replicas and stops cross-user revocation by id.
        """
        async with self._scope() as db:
            result = await db.execute(
                update(pats)
                .where(
                    pats.c.tenant_id == tenant_id,
                    pats.c.token_id == token_id,
                    pats.c.owner_issuer == owner_issuer,
                    pats.c.owner_sub == owner_sub,
                    pats.c.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            )
        return bool(result.rowcount)

    async def touch_last_used(
        self, token_id: str, *, now: float, min_interval: float
    ) -> None:
        """Throttled bookkeeping as ONE guarded statement (no
        read-modify-write, so concurrent verifies write at most once
        per interval)."""
        async with self._scope() as db:
            await db.execute(
                update(pats)
                .where(
                    pats.c.token_id == token_id,
                    or_(
                        pats.c.last_used_at.is_(None),
                        pats.c.last_used_at <= now - min_interval,
                    ),
                )
                .values(last_used_at=now)
            )

    async def revoke_all_for_owner(
        self, *, tenant_id: str, owner_issuer: str, owner_sub: str, now: float
    ) -> int:
        async with self._scope() as db:
            result = await db.execute(
                update(pats)
                .where(
                    pats.c.tenant_id == tenant_id,
                    pats.c.owner_issuer == owner_issuer,
                    pats.c.owner_sub == owner_sub,
                    pats.c.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            )
        return int(result.rowcount or 0)
