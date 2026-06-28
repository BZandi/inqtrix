"""Behavior tests for the account-preferences service (M6c, offline tier)."""

from __future__ import annotations

import pytest

from inqtrix.project.account_preferences_memory import MemoryAccountPreferencesStore
from inqtrix.services.account_preferences_service import (
    AccountPreferencesService,
    AccountPreferencesValidationError,
)


@pytest.fixture()
def service() -> AccountPreferencesService:
    return AccountPreferencesService(store=MemoryAccountPreferencesStore(), durable=False)


async def _save(
    service,
    sub,
    *,
    theme="dark",
    locale="de",
    contrast="high",
    preset="slate",
    user_bubble_tone="gray",
    updated_at=1.0,
):
    return await service.save_preferences(
        sub=sub, contrast_mode=contrast, locale=locale, theme=theme,
        theme_preset=preset, user_bubble_tone=user_bubble_tone,
        updated_at=updated_at,
    )


@pytest.mark.asyncio
async def test_get_returns_none_when_unset(service) -> None:
    assert await service.get_preferences(sub="u") is None


@pytest.mark.asyncio
async def test_save_then_get_roundtrip(service) -> None:
    await _save(
        service,
        "u",
        theme="dark",
        locale="de",
        contrast="high",
        preset="sage",
        user_bubble_tone="mint",
    )
    prefs = await service.get_preferences(sub="u")
    assert prefs is not None
    assert (
        prefs.theme,
        prefs.locale,
        prefs.contrast_mode,
        prefs.theme_preset,
        prefs.user_bubble_tone,
    ) == ("dark", "de", "high", "sage", "mint")


@pytest.mark.asyncio
async def test_save_is_whole_row_upsert(service) -> None:
    await _save(
        service,
        "u",
        theme="dark",
        preset="slate",
        user_bubble_tone="mint",
        updated_at=1.0,
    )
    await _save(
        service,
        "u",
        theme="light",
        preset="graphite",
        user_bubble_tone="orange",
        updated_at=2.0,
    )
    prefs = await service.get_preferences(sub="u")
    assert (prefs.theme, prefs.theme_preset, prefs.user_bubble_tone, prefs.updated_at) == (
        "light", "graphite", "orange", 2.0
    )


@pytest.mark.asyncio
async def test_per_user_isolation(service) -> None:
    await _save(service, "u-a", theme="dark")
    await _save(service, "u-b", theme="light")
    assert (await service.get_preferences(sub="u-a")).theme == "dark"
    assert (await service.get_preferences(sub="u-b")).theme == "light"


@pytest.mark.asyncio
async def test_validation_rejects_out_of_domain(service) -> None:
    with pytest.raises(AccountPreferencesValidationError):
        await _save(service, "u", theme="neon")
    with pytest.raises(AccountPreferencesValidationError):
        await _save(service, "u", locale="fr")
    with pytest.raises(AccountPreferencesValidationError):
        await _save(service, "u", contrast="ultra")
    with pytest.raises(AccountPreferencesValidationError):
        await _save(service, "u", preset="neon")
    with pytest.raises(AccountPreferencesValidationError):
        await _save(service, "u", user_bubble_tone="rainbow")
