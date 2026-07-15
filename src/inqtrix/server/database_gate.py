"""HTTP gate that keeps business routes behind the database contract."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from starlette.responses import Response

from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

_DATABASE_GATE_EXEMPT_PATHS = frozenset({"/health", "/readyz", "/metrics"})


def install_database_contract_gate(
    app: FastAPI,
    *,
    container: "AppContainer",
) -> None:
    """Block product requests until the runtime DB contract is verified.

    Health/readiness and metrics remain reachable so an orchestrator can
    diagnose and recover a failed rollout. ``/readyz`` refreshes the cached
    state, allowing a corrected database to admit traffic without weakening
    the per-request boundary.
    """
    app.state.database_contract_ready = (
        container.settings.storage.backend != "postgres"
    )
    share_reconciliation_required = (
        container.settings.sharing.restrict_to_workspace_members
    )
    app.state.workspace_share_reconciliation_ready = (
        not share_reconciliation_required
    )
    app.state.workspace_share_reconciliation_lock = asyncio.Lock()

    @app.middleware("http")
    async def database_contract_gate(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if (
            request.url.path not in _DATABASE_GATE_EXEMPT_PATHS
            and not request.app.state.database_contract_ready
        ):
            return error_response(
                503,
                "Die Datenbank ist nicht auf dem erwarteten Schema- und "
                "Rollenstand. Der Migrationsjob muss erfolgreich abschliessen.",
                "database_not_ready",
            )
        return await call_next(request)
