"""Open discovery endpoints: ``/health`` and ``/v1/models``.

Deliberately unauthenticated (no principal dependency) so Kubernetes
probes and model-discovery clients keep working without credentials.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")


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

    @router.get("/readyz")
    async def readyz(request: Request):
        """Readiness: are the stateful dependencies reachable?

        Liveness (``/health``) stays provider-only; THIS is what a
        load balancer keys traffic on — a pod with a dead database or
        queue answers 503 here and drains instead of serving 500s.
        """
        from inqtrix.services.system_runtime import readiness_payload

        status_code, payload = await readiness_payload(container)
        database_ready = payload["checks"]["database"] in {"ok", "skipped"}
        if (
            database_ready
            and container.settings.sharing.restrict_to_workspace_members
            and not request.app.state.workspace_share_reconciliation_ready
        ):
            from inqtrix.services.workspace_administration import (
                ensure_workspace_share_reconciliation,
            )

            try:
                revoked = await ensure_workspace_share_reconciliation(
                    request.app,
                    container.workspace_admin,
                    tenant_id="default",
                )
                log.warning(
                    "Workspace-Share-Reconciliation completed during database "
                    "contract recovery; revoked=%d.",
                    revoked,
                )
            except Exception as exc:
                database_ready = False
                payload["checks"]["database"] = "unavailable"
                payload["status"] = "not_ready"
                status_code = 503
                log.warning(
                    "Database contract recovered but workspace-share "
                    "reconciliation failed; business routes remain gated: %s",
                    type(exc).__name__,
                )
        request.app.state.database_contract_ready = database_ready
        if status_code == 200:
            return payload
        return JSONResponse(status_code=status_code, content=payload)

    @router.get("/v1/models")
    def list_models():
        return health_service.models_payload()

    return router
