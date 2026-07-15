"""Instance-admin knowledge-maintenance endpoints.

A thin, session-only instance-admin surface for store hygiene. Currently:
the orphan-vector reconcile sweep — deletes vectors whose canonical Postgres
rows are gone (the reverse of the forward ``vector_synced`` repair). Gated by
the shared :func:`require_instance_admin` guard like every other admin surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from inqtrix.server.routers._admin_guard import require_instance_admin
from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the knowledge-maintenance surface against the container."""
    router = APIRouter()
    provider = container.auth_provider
    principal_dep = container.principal_dependency

    @router.post("/v1/admin/knowledge/reconcile")
    async def reconcile(request: Request):
        """Delete orphan vectors whose canonical Postgres rows are gone.

        Returns the count + details of removed document groups. Only the
        Postgres-canonical tier owns this reconcile; on other stores it is a
        visible 409 rather than a silent no-op.
        """
        _resolved, error = await require_instance_admin(
            provider, request, principal_dep
        )
        if error is not None:
            return error
        service = container.knowledge_service
        store = getattr(getattr(service, "knowledge", None), "store", None)
        reconcile_fn = getattr(store, "reconcile_orphans", None)
        if reconcile_fn is None:
            return error_response(
                409,
                "Reconcile wird vom aktuellen Wissensspeicher nicht "
                "unterstuetzt (nur Postgres-Tier).",
                "conflict",
            )
        return await reconcile_fn()

    return router
