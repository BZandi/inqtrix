"""User search for the share typeahead — mounted only in oidc mode.

Privacy posture: any AUTHENTICATED tenant user can find any other by
prefix — acceptable for a closed deployment whose signup is gated by
invitations; enumeration without a query is impossible (minimum two
characters, capped results), disabled users never appear, and the
payload carries only subject, display name, and email.

When ``settings.sharing.restrict_to_workspace_members`` is on, the results
are additionally narrowed to the caller's workspace co-members — the
typeahead half of the workspace-scoped sharing policy (the grant path
enforces the same boundary). Default off keeps the search tenant-wide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

_SCOPED_KINDS = frozenset({"oidc_session", "pat"})
_MIN_QUERY_LENGTH = 2
_MAX_RESULTS = 10


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the user-search route against the container."""
    router = APIRouter()
    principal_dep = container.principal_dependency
    users = container.auth_provider.users
    permissions = container.permission_service
    restrict_to_members = (
        container.settings.sharing.restrict_to_workspace_members
    )

    @router.get("/v1/users/search")
    async def search_users(request: Request, q: str = ""):
        """Prefix search over email and display name."""
        principal = await principal_dep(request)
        if principal.kind not in _SCOPED_KINDS:
            return error_response(404, "Nicht gefunden", "not_found")
        query = q.strip()
        if len(query) < _MIN_QUERY_LENGTH:
            return error_response(
                400,
                "Parameter 'q' braucht mindestens "
                f"{_MIN_QUERY_LENGTH} Zeichen",
                "invalid_request_error",
            )
        matches = await users.search(
            tenant_id=principal.tenant_id,
            query=query,
            limit=_MAX_RESULTS,
            exclude_subject=principal.sub,
        )
        if restrict_to_members:
            # Narrow to the caller's workspace co-members (mirrors the
            # grant-time boundary); a non-co-member is simply not offered.
            # The narrowing runs AFTER the result cap, so a scoped search may
            # return fewer than the cap even when more co-members would match
            # a longer prefix — typing more narrows it. The authoritative
            # boundary is the grant-time check, so this is a convenience cap.
            allowed = await permissions.share_workspace_filter(
                tenant_id=principal.tenant_id,
                grantor_sub=principal.sub,
                candidate_subs=[user.subject for user in matches],
            )
            matches = [user for user in matches if user.subject in allowed]
        return {
            "object": "list",
            "data": [
                {
                    "subject": user.subject,
                    "display_name": user.display_name,
                    "email": user.email,
                }
                for user in matches
            ],
        }

    return router
