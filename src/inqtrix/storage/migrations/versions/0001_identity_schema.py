"""Identity schema: users, workspaces, groups, shares, audit + RLS.

Revision ID: 0001_identity_schema
Revises: None

Creates the identity tables from the Core metadata (single source of
truth, zero DDL drift) and layers on the security primitives that
SQLAlchemy cannot express:

* the ``inqtrix_app`` role — NOLOGIN/NOSUPERUSER/NOBYPASSRLS, the
  role every application transaction switches to via ``SET LOCAL
  ROLE`` so row-level security applies even when the connection user
  is the table owner or a superuser. NOLOGIN keeps the dev stack free
  of a second password; production may create a LOGIN role that
  inherits it.
* ``inqtrix_current_tenant_id()`` — the fail-closed tenant resolver.
  Raises (ERRCODE 28000) when the ``inqtrix.tenant_id`` GUC is unset
  OR empty; the empty check matters because a transaction-locally set
  GUC survives as an empty string for the connection lifetime after
  the transaction ends, which would otherwise silently match nothing.
* ENABLE + FORCE row-level security on every tenant table with one
  uniform policy. The ``(SELECT ...)`` wrapper around the helper call
  is load-bearing: it makes the planner evaluate the function once
  per query (InitPlan) instead of once per row.
* Grants: full DML for ``inqtrix_app`` on the identity tables, but
  INSERT/SELECT only on ``audit_log`` (append-only enforced by
  grants — WORM-ish, not compliance-grade WORM; owners and superusers
  bypass grants).

Downgrade drops the schema objects but leaves the ``inqtrix_app``
role in place: roles are cluster-wide and may be referenced by other
databases; the upgrade recreates it idempotently.
"""

from __future__ import annotations

from alembic import op

from inqtrix.storage.identity_orm import identity_metadata

revision = "0001_identity_schema"
down_revision = None
branch_labels = None
depends_on = None


TENANT_TABLES = (
    "users",
    "workspaces",
    "workspace_members",
    "groups",
    "group_members",
    "invitations",
    "resource_shares",
    "audit_log",
)

APP_ROLE = "inqtrix_app"

_CREATE_APP_ROLE = f"""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
        CREATE ROLE {APP_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS
            NOCREATEDB NOCREATEROLE;
    END IF;
END
$$;
"""

_CREATE_TENANT_FUNCTION = """
CREATE OR REPLACE FUNCTION inqtrix_current_tenant_id() RETURNS text
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $$
DECLARE
    v text := current_setting('inqtrix.tenant_id', true);
BEGIN
    IF v IS NULL OR v = '' THEN
        RAISE EXCEPTION
            'inqtrix.tenant_id is not set; refusing the tenant-scoped '
            'query (use tenant_session)'
            USING ERRCODE = '28000';
    END IF;
    RETURN v;
END
$$;
"""


def upgrade() -> None:
    bind = op.get_bind()
    identity_metadata.create_all(bind=bind)

    op.execute(_CREATE_APP_ROLE)
    # Membership lets a non-superuser migration/login user SET ROLE to
    # the restricted app role (superusers may anyway).
    op.execute(
        f"""
        DO $$
        BEGIN
            IF current_user <> '{APP_ROLE}' THEN
                EXECUTE format('GRANT {APP_ROLE} TO %I', current_user);
            END IF;
        END
        $$;
        """
    )
    op.execute(_CREATE_TENANT_FUNCTION)

    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    for table in TENANT_TABLES:
        if table == "audit_log":
            op.execute(f"GRANT SELECT, INSERT ON audit_log TO {APP_ROLE}")
        else:
            op.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {APP_ROLE}"
            )
    # The audit_log identity column draws from a sequence; INSERT
    # needs USAGE on it.
    op.execute(f"GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO {APP_ROLE}")

    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
                FOR ALL
                USING (tenant_id = (SELECT inqtrix_current_tenant_id()))
                WITH CHECK (tenant_id = (SELECT inqtrix_current_tenant_id()))
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    # Policies drop with their tables; grants on dropped tables vanish.
    identity_metadata.drop_all(bind=bind)
    op.execute("DROP FUNCTION IF EXISTS inqtrix_current_tenant_id()")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {APP_ROLE}")
    # The role itself stays (cluster-wide object, idempotent upgrade).
