"""Durable run schema: runs and run_events with tenant RLS.

Revision ID: 0003_runs_durability
Revises: 0002_content_files

Creates the ``runs`` and ``run_events`` tables from the runs metadata
and applies the same security layering as revisions 0001/0002: DML
grants for the ``inqtrix_app`` role and ENABLE + FORCE row-level
security with the fail-closed tenant policy (the ``(SELECT ...)``
wrapper keeps the helper call an InitPlan — once per query, not per
row). The status CHECK constraint mirrors
:class:`~inqtrix.server.runs.RunStatus`; the lifecycle ordering itself
lives only in that application enum.
"""

from __future__ import annotations

from alembic import op

from inqtrix.server.runs import RunStatus
from inqtrix.storage.runs_orm import runs_metadata

revision = "0003_runs_durability"
down_revision = "0002_content_files"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"

_STATUS_VALUES = ", ".join(f"'{status.value}'" for status in RunStatus)


def upgrade() -> None:
    bind = op.get_bind()
    runs_metadata.create_all(bind=bind)

    op.execute(
        "ALTER TABLE runs ADD CONSTRAINT ck_runs_status "
        f"CHECK (status IN ({_STATUS_VALUES}))"
    )
    for table in ("runs", "run_events"):
        op.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {APP_ROLE}"
        )
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
    runs_metadata.drop_all(bind=bind)
