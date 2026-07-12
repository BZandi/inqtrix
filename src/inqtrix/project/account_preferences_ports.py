"""Contracts of the account-preferences store (M6c).

The simplest project-persistence port: a single preferences row per user
(``sub``), no workspace dimension, no list, no children. ``get_preferences``
returns ``None`` when the user has never saved — the caller (router) maps that
to 404 so the frontend keeps its own default theme/locale (the defaults are a
frontend SSOT, never fabricated server-side). Two implementations behind one
port: :class:`~inqtrix.project.account_preferences_memory.MemoryAccountPreferencesStore`
and :class:`~inqtrix.project.account_preferences_postgres.PostgresAccountPreferencesStore`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AccountPreferences:
    """One user's account-level UI preferences.

    Attributes:
        sub: The owning principal subject (the row key).
        contrast_mode: ``standard`` or ``high``.
        locale: ``de`` or ``en``.
        theme: ``light`` / ``dark`` / ``system``.
        theme_preset: ``standard`` / ``slate`` / ``graphite`` / ``sage``.
        user_bubble_tone: User-message bubble tone selected in the desk UI.
        enable_agent_memory: Whether the user opted long-term agent memory
            IN. Default ``False`` — memory is off until the user enables it
            (privacy default), so an absent/legacy row means memory-less.
        updated_at: Unix timestamp of the last save.
        tenant_id: The tenant scope (RLS).
    """

    sub: str
    contrast_mode: str
    locale: str
    theme: str
    theme_preset: str
    updated_at: float
    user_bubble_tone: str = "gray"
    enable_agent_memory: bool = False
    tenant_id: str = "default"


@runtime_checkable
class AccountPreferencesStore(Protocol):
    """Persistence port for the per-user preferences singleton."""

    async def get_preferences(self, *, sub: str) -> AccountPreferences | None:
        """The user's stored preferences, or ``None`` when never saved."""
        ...

    async def upsert_preferences(
        self,
        *,
        sub: str,
        contrast_mode: str,
        locale: str,
        theme: str,
        theme_preset: str,
        user_bubble_tone: str,
        updated_at: float,
        enable_agent_memory: bool = False,
    ) -> AccountPreferences:
        """Insert or replace the user's preferences row (whole-row upsert)."""
        ...

    async def aclose(self) -> None: ...
