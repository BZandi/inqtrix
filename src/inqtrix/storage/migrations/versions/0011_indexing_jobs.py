"""Durable reindex-job schema: indexing_jobs and indexing_job_events.

Revision ID: 0011_indexing_jobs
Revises: 0010_knowledge

Creates the durable reindex-job tables from their metadata snapshot and
applies the established security layering: DML grants for ``inqtrix_app``
and ENABLE + FORCE row-level security with the fail-closed tenant policy
(InitPlan ``(SELECT ...)`` wrapper), identical to ``0003_runs_durability``
and ``0010_knowledge``. The status CHECK constraint mirrors
:class:`~inqtrix.server.indexing.IndexingJobStatus`; the lifecycle
ordering lives only in that application enum.

This is the durable twin of the in-memory reindex-job store: a background
re-embed becomes a row a worker process can claim, so it survives a server
restart, not just closing the browser. Document VECTORS live in Qdrant;
canonical text lives in ``knowledge_documents`` — these tables hold only
job state and the event log.
"""

from __future__ import annotations

from alembic import op

from inqtrix.server.indexing import IndexingJobStatus
from inqtrix.storage.indexing_orm import indexing_metadata

revision = "0011_indexing_jobs"
down_revision = "0010_knowledge"
branch_labels = None
depends_on = None

APP_ROLE = "inqtrix_app"
# Order matters for FK creation/drop: parents before children.
_TABLES = ("indexing_jobs", "indexing_job_events")

_STATUS_VALUES = ", ".join(f"'{status.value}'" for status in IndexingJobStatus)


def upgrade() -> None:
    bind = op.get_bind()
    indexing_metadata.create_all(bind=bind)

    op.execute(
        "ALTER TABLE indexing_jobs ADD CONSTRAINT ck_indexing_jobs_status "
        f"CHECK (status IN ({_STATUS_VALUES}))"
    )
    for table in _TABLES:
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
    indexing_metadata.drop_all(bind=bind)
