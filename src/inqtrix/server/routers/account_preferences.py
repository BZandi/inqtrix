"""Account-preferences endpoints (M6c project tier).

The singleton per-user settings surface: ``GET`` / ``PUT
/v1/account/preferences``. Unlike the project tiers, the row is keyed on the
authenticated principal's subject directly — ``principal.sub`` is always
present (``__anonymous__`` / ``__static__`` / the OIDC or PAT subject), so a
caller always addresses exactly their own row. GET returns 404 when the user
has never saved (the frontend then keeps its own default theme/locale).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request

from inqtrix.auth.principal import Principal
from inqtrix.project.account_preferences_ports import AccountPreferences
from inqtrix.services.account_preferences_service import (
    AccountPreferencesValidationError,
)
from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer


def _payload(p: AccountPreferences) -> dict[str, Any]:
    return {
        "contrast_mode": p.contrast_mode, "locale": p.locale, "theme": p.theme,
        "theme_preset": p.theme_preset, "user_bubble_tone": p.user_bubble_tone,
        "updated_at": p.updated_at,
    }


def build_router(container: "AppContainer") -> APIRouter:
    service = container.account_preferences_service
    if service is None:
        raise RuntimeError("build_router(account_preferences) requires a wired service.")
    router = APIRouter()
    principal_dep = container.principal_dependency

    @router.get("/v1/account/preferences")
    async def get_preferences(principal: Principal = Depends(principal_dep)):
        prefs = await service.get_preferences(sub=principal.sub)
        if prefs is None:
            return error_response(404, "Keine Praeferenzen gespeichert", "not_found")
        return _payload(prefs)

    @router.put("/v1/account/preferences")
    async def save_preferences(req: Request, principal: Principal = Depends(principal_dep)):
        body = await _json_object(req)
        if not isinstance(body, dict):
            return error_response(400, "Ungueltiger Body", "invalid_request_error")
        if not isinstance(body.get("updated_at"), (int, float)):
            return error_response(400, "updated_at ist erforderlich", "invalid_request_error")
        try:
            prefs = await service.save_preferences(
                sub=principal.sub,
                contrast_mode=str(body.get("contrast_mode", "standard")),
                locale=str(body.get("locale", "en")),
                theme=str(body.get("theme", "system")),
                theme_preset=str(body.get("theme_preset", "standard")),
                user_bubble_tone=str(body.get("user_bubble_tone", "gray")),
                updated_at=float(body["updated_at"]),
            )
        except AccountPreferencesValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        return _payload(prefs)

    return router


async def _json_object(req: Request) -> Any:
    try:
        return await req.json()
    except Exception:
        return None
