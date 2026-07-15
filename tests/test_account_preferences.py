"""Behavior tests for the account-preferences service (M6c, offline tier)."""

from __future__ import annotations

import uuid

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
    user_id,
    *,
    theme="dark",
    locale="de",
    contrast="high",
    preset="slate",
    user_bubble_tone="gray",
    enable_agent_memory=False,
    updated_at=1.0,
):
    return await service.save_preferences(
        user_id=user_id,
        contrast_mode=contrast,
        locale=locale,
        theme=theme,
        theme_preset=preset, user_bubble_tone=user_bubble_tone,
        enable_agent_memory=enable_agent_memory,
        updated_at=updated_at,
    )


USER = uuid.UUID("11111111-1111-4111-8111-111111111111")
USER_A = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
USER_B = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


@pytest.mark.asyncio
async def test_get_returns_none_when_unset(service) -> None:
    assert await service.get_preferences(user_id=USER) is None


@pytest.mark.asyncio
async def test_save_then_get_roundtrip(service) -> None:
    await _save(
        service,
        USER,
        theme="dark",
        locale="de",
        contrast="high",
        preset="sage",
        user_bubble_tone="mint",
    )
    prefs = await service.get_preferences(user_id=USER)
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
        USER,
        theme="dark",
        preset="slate",
        user_bubble_tone="mint",
        updated_at=1.0,
    )
    await _save(
        service,
        USER,
        theme="light",
        preset="graphite",
        user_bubble_tone="orange",
        updated_at=2.0,
    )
    prefs = await service.get_preferences(user_id=USER)
    assert (prefs.theme, prefs.theme_preset, prefs.user_bubble_tone, prefs.updated_at) == (
        "light", "graphite", "orange", 2.0
    )


@pytest.mark.asyncio
async def test_agent_memory_opt_in_defaults_off_and_round_trips(service) -> None:
    # Privacy default OFF when omitted; opt-in persists and survives the
    # whole-row upsert of another preference.
    await _save(service, USER)
    assert (
        await service.get_preferences(user_id=USER)
    ).enable_agent_memory is False

    await _save(service, USER, enable_agent_memory=True, updated_at=2.0)
    assert (
        await service.get_preferences(user_id=USER)
    ).enable_agent_memory is True

    await _save(service, USER, theme="light", updated_at=3.0)
    assert (
        await service.get_preferences(user_id=USER)
    ).enable_agent_memory is False


@pytest.mark.asyncio
async def test_per_user_isolation(service) -> None:
    await _save(service, USER_A, theme="dark")
    await _save(service, USER_B, theme="light")
    assert (await service.get_preferences(user_id=USER_A)).theme == "dark"
    assert (await service.get_preferences(user_id=USER_B)).theme == "light"


@pytest.mark.asyncio
async def test_validation_rejects_out_of_domain(service) -> None:
    with pytest.raises(AccountPreferencesValidationError):
        await _save(service, USER, theme="neon")
    with pytest.raises(AccountPreferencesValidationError):
        await _save(service, USER, locale="fr")
    with pytest.raises(AccountPreferencesValidationError):
        await _save(service, USER, contrast="ultra")
    with pytest.raises(AccountPreferencesValidationError):
        await _save(service, USER, preset="neon")
    with pytest.raises(AccountPreferencesValidationError):
        await _save(service, USER, user_bubble_tone="rainbow")
