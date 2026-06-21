"""Open discovery endpoints: ``/health`` and ``/v1/models``.

Deliberately unauthenticated (no principal dependency) so Kubernetes
probes and model-discovery clients keep working without credentials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the health/model-discovery routes against the container."""
    router = APIRouter()
    health_service = container.health_service

    @router.get("/health")
    def health():
        status_code, payload = health_service.health_payload()
        if status_code == 200:
            return payload
        return JSONResponse(status_code=status_code, content=payload)

    @router.get("/v1/models")
    def list_models():
        return health_service.models_payload()

    return router
