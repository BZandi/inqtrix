"""In-memory account-preferences store (tier without Postgres + offline test).

Mirrors :class:`~inqtrix.project.account_preferences_postgres.PostgresAccountPreferencesStore`
behavior (whole-row upsert keyed on ``sub``; ``get`` returns ``None`` when
unset). Process-local, not durable.
"""

from __future__ import annotations

from inqtrix.project.account_preferences_ports import AccountPreferences


class MemoryAccountPreferencesStore:
    """Process-local :class:`~inqtrix.project.account_preferences_ports.AccountPreferencesStore`."""

    def __init__(self) -> None:
        self._rows: dict[str, AccountPreferences] = {}

    async def get_preferences(self, *, sub: str) -> AccountPreferences | None:
        return self._rows.get(sub)

    async def upsert_preferences(
        self,
        *,
        sub,
        contrast_mode,
        locale,
        theme,
        theme_preset,
        user_bubble_tone,
        updated_at,
        enable_agent_memory=False,
    ) -> AccountPreferences:
        prefs = AccountPreferences(
            sub=sub, contrast_mode=contrast_mode, locale=locale, theme=theme,
            theme_preset=theme_preset, user_bubble_tone=user_bubble_tone,
            enable_agent_memory=enable_agent_memory,
            updated_at=updated_at,
        )
        self._rows[sub] = prefs
        return prefs

    async def aclose(self) -> None:
        return None
