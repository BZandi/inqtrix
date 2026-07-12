"""Shared workspace-name validation and admin creation command.

Three routes take a workspace name — self-serve creation
(``POST /v1/workspaces``), admin creation (``POST /v1/admin/workspaces``)
and admin rename (``PATCH /v1/admin/workspaces/{id}``). All of them must
apply the same validation, and both creation surfaces must behave
identically, so the validate/create/audit body lives here exactly once;
the routers keep only their (different) response shapes. Auditing inside
the creation command guarantees no creation path can skip the
``workspace.created`` trail again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from inqtrix.auth.permissions import AuditEntry

if TYPE_CHECKING:
    from inqtrix.auth.principal import Principal

WORKSPACE_NAME_MAX_LEN = 120
"""Upper bound for workspace names, shared by every name-taking surface."""


class WorkspaceNameError(ValueError):
    """Raised when a workspace name fails validation.

    Carries the user-facing German message so every router renders the
    byte-identical 400 the routes produced before the consolidation.
    """


class WorkspaceAdministrationPort(Protocol):
    """The store surface the creation command needs.

    Structural intersection of the membership-admin write port and the
    audit sink — both identity backends (memory and Postgres) implement
    the two protocols on one object, and creating-with-audit needs both.
    """

    async def create_workspace(
        self, *, tenant_id: str, name: str, created_by_sub: str
    ) -> tuple[str, str]:
        """See :class:`inqtrix.auth.permissions.MembershipAdminRepository`."""
        ...

    async def record(self, entry: AuditEntry) -> None:
        """See :class:`inqtrix.auth.permissions.AuditSink`."""
        ...


def normalize_workspace_name(value: object) -> str:
    """Validate one untrusted workspace-name value into its stored form.

    Strict on type: only strings are names. The historical routes ran
    non-string JSON values through ``str(...)`` and would store literal
    ``"None"``/``"123"`` names — a repr leak, not a contract — so the
    consolidated rule rejects them uniformly across create and rename.

    Args:
        value: The raw ``name`` value from the request body.

    Returns:
        The stripped name.

    Raises:
        WorkspaceNameError: When the value is not a string, or the
            stripped name is empty or longer than
            :data:`WORKSPACE_NAME_MAX_LEN`.
    """
    clean = value.strip() if isinstance(value, str) else ""
    if not clean or len(clean) > WORKSPACE_NAME_MAX_LEN:
        raise WorkspaceNameError(
            f"Feld 'name' muss 1 bis {WORKSPACE_NAME_MAX_LEN} Zeichen lang sein"
        )
    return clean


async def create_workspace_for_admin(
    workspace_admin: WorkspaceAdministrationPort,
    *,
    principal: "Principal",
    name: object,
) -> tuple[str, str]:
    """Create one workspace for an already-verified instance admin.

    Args:
        workspace_admin: The membership-admin store (also the audit
            sink); the caller guarantees it is wired.
        principal: The verified instance-admin principal. The caller
            is responsible for the ``require_instance_admin`` gate —
            this command only executes the decision.
        name: The raw, untrusted name value from the request body.

    Returns:
        ``(workspace_id, created_name)`` as stored.

    Raises:
        WorkspaceNameError: When the name fails
            :func:`normalize_workspace_name`.
    """
    clean_name = normalize_workspace_name(name)
    workspace_id, created_name = await workspace_admin.create_workspace(
        tenant_id=principal.tenant_id,
        name=clean_name,
        created_by_sub=principal.sub,
    )
    await workspace_admin.record(
        AuditEntry(
            tenant_id=principal.tenant_id,
            actor_sub=principal.sub,
            action="workspace.created",
            resource_type="workspace",
            resource_id=workspace_id,
            detail={"name": created_name},
        )
    )
    return workspace_id, created_name
