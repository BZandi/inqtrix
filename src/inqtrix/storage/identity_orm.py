"""SQLAlchemy Core definitions of the identity schema.

This metadata is the single source of truth for the identity tables:
the initial migration creates them via ``metadata.create_all`` (no
hand-duplicated DDL to drift) and then layers on what Core cannot
express — roles, grants, the row-level-security policies, and the
fail-closed tenant helper function (see
``migrations/versions/0001_identity_schema.py``).

Schema decisions (researched 2026-06, deviations from the original
plan sketch recorded in the audit history):

* ``tenant_id`` is ``text`` (matches ``Principal.tenant_id``; v1 runs
  one tenant ``"default"`` per deployment) and exists on every table
  so the RLS tenant defense has one uniform policy shape.
* Permission/role values are ``text`` plus CHECK constraints — never
  native Postgres enums. The ordering lives exclusively in the
  application enums (:mod:`inqtrix.auth.permissions`); a second
  ordering authority in the database would be a split-brain risk.
* Users get a surrogate UUID primary key with ``UNIQUE(issuer,
  subject)`` — OIDC ``sub`` is only unique per issuer, and e-mail is
  data, never identity.
* Share tuples reference subjects/resources polymorphically as
  ``text`` (run and knowledge ids are strings), so there are no
  foreign keys on those columns by construction; services must clean
  up shares in the same transaction that deletes a resource.
* ``resource_shares`` revocation is soft (``revoked_at``) with a
  partial unique index on active rows — one active grant per
  (subject, resource), full revocation history retained.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from inqtrix.auth.permissions import SharePermission, WorkspaceRole

identity_metadata = MetaData()

_WORKSPACE_ROLES = ", ".join(f"'{role.value}'" for role in WorkspaceRole)
_SHARE_PERMISSIONS = ", ".join(f"'{p.value}'" for p in SharePermission)

_UUID_PK = dict(
    primary_key=True,
    server_default=text("gen_random_uuid()"),
)
_CREATED_AT = dict(
    nullable=False,
    server_default=text("now()"),
)


users = Table(
    "users",
    identity_metadata,
    Column("id", UUID(as_uuid=True), **_UUID_PK),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("issuer", Text, nullable=False),
    Column("subject", Text, nullable=False),
    Column("email", Text, nullable=False),
    Column("email_verified", Boolean, nullable=False, server_default=text("false")),
    Column("display_name", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), **_CREATED_AT),
    Column("last_login_at", DateTime(timezone=True), nullable=True),
    Column("disabled_at", DateTime(timezone=True), nullable=True),
    # Instance-wide role for the admin surface (NOT the per-workspace
    # WorkspaceRole). 'admin' may manage users; 'user' is the default.
    # Arrives with revision 0009; existing rows default to 'user'.
    Column(
        "instance_role", Text, nullable=False, server_default=text("'user'")
    ),
    # The user's canonical project namespace (a ``ws_...`` string), adopted
    # from the browser's namespace on first authenticated boot and returned in
    # /api/auth/session so every device scopes the user's project to the SAME
    # namespace (data follows the user across devices). Nullable: NULL until
    # first adopted; arrives with revision 0019. This is a project UI-namespace
    # anchor, NOT an authorization input (auth is created_by_sub) and NOT the
    # server-side collaboration ``workspaces`` table.
    Column("default_workspace_id", Text, nullable=True),
    UniqueConstraint("issuer", "subject", name="uq_users_issuer_subject"),
    Index("ix_users_tenant", "tenant_id"),
)
"""Local mirror of IdP-provisioned users (JIT on first login)."""


workspaces = Table(
    "workspaces",
    identity_metadata,
    Column("id", UUID(as_uuid=True), **_UUID_PK),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("name", Text, nullable=False),
    Column("created_by_sub", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), **_CREATED_AT),
    Index("ix_workspaces_tenant", "tenant_id"),
)
"""Collaboration workspaces (the authorization grouping, NOT the
client-supplied UI namespace that the HTTP layer calls workspace_id)."""


workspace_members = Table(
    "workspace_members",
    identity_metadata,
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column(
        "workspace_id",
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("sub", Text, primary_key=True),
    Column("role", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), **_CREATED_AT),
    CheckConstraint(
        f"role IN ({_WORKSPACE_ROLES})", name="ck_workspace_members_role"
    ),
    Index("ix_workspace_members_tenant_sub", "tenant_id", "sub"),
)
"""Workspace membership with the ordered coarse role."""


groups = Table(
    "groups",
    identity_metadata,
    Column("id", UUID(as_uuid=True), **_UUID_PK),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("name", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), **_CREATED_AT),
    Index("ix_groups_tenant", "tenant_id"),
)
"""Flat local groups (share subjects)."""


group_members = Table(
    "group_members",
    identity_metadata,
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column(
        "group_id",
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("sub", Text, primary_key=True),
    Column("created_at", DateTime(timezone=True), **_CREATED_AT),
    Index("ix_group_members_tenant_sub", "tenant_id", "sub"),
)
"""Group membership (one level deep — no nested groups by design)."""


invitations = Table(
    "invitations",
    identity_metadata,
    Column("id", UUID(as_uuid=True), **_UUID_PK),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column(
        "workspace_id",
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("email", Text, nullable=False),
    Column("role", Text, nullable=False),
    Column("invited_by_sub", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), **_CREATED_AT),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("accepted_at", DateTime(timezone=True), nullable=True),
    Column("accepted_by_sub", Text, nullable=True),
    Column("revoked_at", DateTime(timezone=True), nullable=True),
    CheckConstraint(f"role IN ({_WORKSPACE_ROLES})", name="ck_invitations_role"),
    Index(
        "uq_invitations_open",
        "workspace_id",
        text("lower(email)"),
        unique=True,
        postgresql_where=text("accepted_at IS NULL AND revoked_at IS NULL"),
    ),
)
"""Invitation allowlist for closed registration (one open invitation
per workspace and e-mail; acceptance logic lands with the OIDC
provider)."""


resource_shares = Table(
    "resource_shares",
    identity_metadata,
    Column("id", UUID(as_uuid=True), **_UUID_PK),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("subject_type", Text, nullable=False),
    Column("subject_id", Text, nullable=False),
    Column("resource_type", Text, nullable=False),
    Column("resource_id", Text, nullable=False),
    Column("permission", Text, nullable=False),
    Column("granted_by_sub", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), **_CREATED_AT),
    Column("revoked_at", DateTime(timezone=True), nullable=True),
    Column("revoked_by_sub", Text, nullable=True),
    CheckConstraint(
        "subject_type IN ('user', 'group')", name="ck_resource_shares_subject"
    ),
    CheckConstraint(
        f"permission IN ({_SHARE_PERMISSIONS})",
        name="ck_resource_shares_permission",
    ),
    Index(
        "uq_resource_shares_active",
        "tenant_id",
        "subject_type",
        "subject_id",
        "resource_type",
        "resource_id",
        unique=True,
        postgresql_where=text("revoked_at IS NULL"),
    ),
    Index(
        "ix_resource_shares_subject_active",
        "tenant_id",
        "subject_type",
        "subject_id",
        "resource_type",
        postgresql_where=text("revoked_at IS NULL"),
    ),
    Index(
        "ix_resource_shares_resource_active",
        "tenant_id",
        "resource_type",
        "resource_id",
        postgresql_where=text("revoked_at IS NULL"),
    ),
)
"""Generic share tuples (Zanzibar-lite): subject x resource ->
ordered permission, soft-revoked for history."""


audit_log = Table(
    "audit_log",
    identity_metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("tenant_id", Text, nullable=False, server_default=text("'default'")),
    Column("occurred_at", DateTime(timezone=True), **_CREATED_AT),
    Column("actor_sub", Text, nullable=False),
    Column("actor_type", Text, nullable=False, server_default=text("'user'")),
    Column("action", Text, nullable=False),
    Column("resource_type", Text, nullable=False),
    Column("resource_id", Text, nullable=False),
    Column("workspace_id", UUID(as_uuid=True), nullable=True),
    Column("detail", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Index("ix_audit_log_tenant_occurred", "tenant_id", "occurred_at"),
)
"""Append-only audit trail of semantic actions. The application role
holds INSERT/SELECT only — UPDATE/DELETE are not granted (note: grants
bind the runtime role; table owner and superusers bypass them, so this
is WORM-ish, not compliance-grade WORM)."""
