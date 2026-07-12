"""Align persisted agent tasks with the execution contract.

Revision ID: 0043_agent_task_contract
Revises: 0042_agent_session_integrity
Create Date: 2026-07-11

Backfills legacy plan-approval subjects, adds the non-error terminal task
outcome ``insufficient_evidence`` and an internal task-result recovery payload,
and canonicalizes historic duration-style web recency values. The existing
control tables remain the only state source.
"""

from __future__ import annotations

from alembic import op

revision = "0043_agent_task_contract"
down_revision = "0042_agent_session_integrity"
branch_labels = None
depends_on = None

_APPROVAL_SUBJECT_BACKFILL_SQL = """
UPDATE run_approvals AS approval
SET subject_id = plan.plan_id
FROM run_plans AS plan
WHERE approval.run_id = plan.run_id
  AND approval.kind IN ('plan', 'replan')
  AND approval.subject_type = 'plan'
  AND btrim(approval.subject_id) = ''
  AND (approval.payload->>'plan_version') ~ '^[1-9][0-9]*$'
  AND plan.version = (approval.payload->>'plan_version')::integer
"""

_LEGACY_CHILD_BUDGET_BACKFILL_SQL = """
UPDATE runs
SET request_payload = jsonb_set(
    request_payload::jsonb,
    '{body}',
    (request_payload::jsonb->'body') - 'token_budget',
    true
)::json
WHERE kind = 'agent_child'
  AND status IN (
      'queued', 'running', 'waiting_for_approval',
      'waiting_for_input', 'waiting_for_children'
  )
  AND jsonb_typeof(request_payload::jsonb->'body') = 'object'
  AND (request_payload::jsonb->'body') ? 'token_budget'
"""

_RUN_ROOT_LINEAGE_BACKFILL_SQL = """
WITH RECURSIVE run_tree AS (
    SELECT run_id, run_id AS canonical_root
    FROM runs
    WHERE parent_run_id IS NULL
    UNION ALL
    SELECT child.run_id, run_tree.canonical_root
    FROM runs AS child
    JOIN run_tree ON child.parent_run_id = run_tree.run_id
)
UPDATE runs AS target
SET root_run_id = run_tree.canonical_root
FROM run_tree
WHERE target.run_id = run_tree.run_id
  AND target.parent_run_id IS NOT NULL
  AND target.root_run_id IS DISTINCT FROM run_tree.canonical_root
"""


def upgrade() -> None:
    op.execute(_APPROVAL_SUBJECT_BACKFILL_SQL)
    op.execute(_RUN_ROOT_LINEAGE_BACKFILL_SQL)
    # Planner-authored child caps were persisted in the replay payload before
    # resource authority moved entirely to operator settings. Scrub active
    # children before workers replay them; terminal history stays untouched.
    op.execute(_LEGACY_CHILD_BUDGET_BACKFILL_SQL)
    # 0030 imports the live ORM metadata, so a fresh database may already
    # contain this additive column before reaching 0043. Existing databases
    # need the same column here; IF NOT EXISTS keeps both paths convergent.
    op.execute(
        "ALTER TABLE run_plan_tasks ADD COLUMN IF NOT EXISTS "
        "result_payload JSON NOT NULL DEFAULT '{}'::json"
    )
    op.execute(
        "UPDATE run_plan_tasks SET params = "
        "jsonb_set(params::jsonb, '{recency}', to_jsonb(CASE "
        "WHEN params->>'recency' = '1d' THEN 'day' "
        "WHEN params->>'recency' = '7d' THEN 'week' "
        "WHEN params->>'recency' = '30d' THEN 'month' "
        "WHEN params->>'recency' = '365d' THEN 'year' END), true)::json "
        "WHERE params->>'recency' IN ('1d', '7d', '30d', '365d')"
    )
    op.execute(
        "ALTER TABLE run_plan_tasks "
        "DROP CONSTRAINT IF EXISTS ck_run_plan_tasks_status"
    )
    op.execute(
        "ALTER TABLE run_plan_tasks ADD CONSTRAINT "
        "ck_run_plan_tasks_status CHECK (status IN "
        "('pending', 'running', 'completed', 'failed', "
        "'insufficient_evidence', 'skipped'))"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE run_plan_tasks SET status = 'failed' "
        "WHERE status = 'insufficient_evidence'"
    )
    op.execute(
        "ALTER TABLE run_plan_tasks "
        "DROP CONSTRAINT IF EXISTS ck_run_plan_tasks_status"
    )
    op.execute(
        "ALTER TABLE run_plan_tasks ADD CONSTRAINT "
        "ck_run_plan_tasks_status CHECK (status IN "
        "('pending', 'running', 'completed', 'failed', 'skipped'))"
    )
    op.execute(
        "ALTER TABLE run_plan_tasks DROP COLUMN IF EXISTS result_payload"
    )
