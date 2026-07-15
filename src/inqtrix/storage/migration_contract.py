"""Canonical schema and tenant-RLS contract for migrations and readiness.

This module is intentionally dependency-free. Runtime readiness checks must be
able to compare the installed revision with the packaged schema contract
without importing Alembic or scanning the migration directory.
"""

from __future__ import annotations

import re
from typing import Final

SCHEMA_HEAD_REVISION: Final = "0049_tenant_integrity"
"""Single Alembic head shipped by this source tree."""

TENANT_RLS_POLICY: Final = "tenant_isolation"
"""Required fail-closed policy name on every tenant-scoped table."""

_TENANT_POLICY_EXPRESSION: Final[re.Pattern[str]] = re.compile(
    r"^\(*\s*tenant_id\s*=\s*(?:inqtrix_current_tenant_id\(\)|"
    r"\(+\s*select\s+inqtrix_current_tenant_id\(\)"
    r"(?:\s+as\s+inqtrix_current_tenant_id)?\s*\)+)\s*\)*$",
    re.IGNORECASE,
)

TENANT_RLS_TABLES: Final[tuple[str, ...]] = (
    "account_preferences",
    "agent_feedback",
    "agent_memory_candidates",
    "agent_session_groups",
    "agent_sessions",
    "asset_groups",
    "asset_records",
    "asset_sections",
    "audit_log",
    "auth_flows",
    "auth_sessions",
    "chat_messages",
    "chat_thread_groups",
    "chat_threads",
    "editor_collaboration_instances",
    "editor_collaboration_leases",
    "editor_collaboration_snapshots",
    "editor_collaboration_updates",
    "editor_comments",
    "editor_documents",
    "editor_folders",
    "editor_patches",
    "files",
    "indexing_job_events",
    "indexing_jobs",
    "invitations",
    "knowledge_chunks",
    "knowledge_collections",
    "knowledge_documents",
    "knowledge_session_groups",
    "knowledge_sessions",
    "local_credentials",
    "personal_access_tokens",
    "prompt_templates",
    "quota_limits",
    "quota_usage_counters",
    "resource_shares",
    "run_approvals",
    "run_artifact_revisions",
    "run_artifacts",
    "run_clarifications",
    "run_events",
    "run_plan_tasks",
    "run_plans",
    "runs",
    "skill_templates",
    "tenant_security_state",
    "user_events",
    "users",
    "vector_index_history",
    "vector_index_members",
    "vector_index_records",
    "workspace_members",
    "workspaces",
)
"""Head-revision tables protected by enabled and forced tenant RLS."""

LEGACY_TENANT_RLS_TABLES: Final[tuple[str, ...]] = (
    "group_members",
    "groups",
)
"""Tenant tables removed by an irreversible migration but still lockable."""

MIGRATION_TENANT_RLS_TABLES: Final[tuple[str, ...]] = tuple(
    sorted((*TENANT_RLS_TABLES, *LEGACY_TENANT_RLS_TABLES))
)
"""All tenant tables a supported source revision may contain."""

RUNTIME_REQUIRED_FUNCTIONS: Final[tuple[str, ...]] = (
    "inqtrix_current_tenant_id()",
)
"""Schema-local functions required while evaluating tenant policies."""

RUNTIME_REQUIRED_SEQUENCES: Final[tuple[str, ...]] = (
    "audit_log_id_seq",
    "user_events_id_seq",
)
"""Schema-local identity sequences used by restricted runtime writes."""

RUNTIME_VERSION_TABLE: Final = "alembic_version"
"""Schema-local revision table readable, but never mutable, by runtime."""


def schema_head_revision() -> str:
    """Return the packaged schema head without loading Alembic."""
    return SCHEMA_HEAD_REVISION


def tenant_policy_expression_matches(expression: object) -> bool:
    """Recognize only the canonical fail-closed tenant equality predicate."""
    if not isinstance(expression, str):
        return False
    return _TENANT_POLICY_EXPRESSION.fullmatch(expression.strip()) is not None


def postgres_role_can_set_sql(
    member_expression: str,
    target_expression: str,
) -> str:
    """Build a PostgreSQL 15+ expression for effective SET ROLE authority.

    PostgreSQL 16 separated the membership and SET options. PostgreSQL 15
    exposes only ``MEMBER``, which still represents the ability to assume the
    target. Callers pass trusted catalog expressions, never user input.

    Args:
        member_expression: SQL expression resolving to the source role OID.
        target_expression: SQL expression resolving to the target role OID.

    Returns:
        Parenthesized SQL boolean expression compatible with PostgreSQL 15+.
    """
    privilege = (
        "CASE WHEN current_setting('server_version_num')::integer >= 160000 "
        "THEN 'SET' ELSE 'MEMBER' END"
    )
    return (
        f"pg_has_role({member_expression}, {target_expression}, {privilege})"
    )


def postgres_direct_relation_acl_sql(
    relation_alias: str,
    grantee_oid_expression: str,
    *,
    expected_privileges_sql: str,
) -> str:
    """Build an exact direct relation-ACL predicate for one application role.

    The owner keeps implicit/explicit owner rights. PUBLIC table/column grants,
    application column grants, an application grant option, or an application
    privilege outside ``expected_privileges_sql`` makes the predicate false.
    Named database roles remain an operator-managed boundary; runtime checks
    separately validate the effective role's complete privileges. Callers pass
    trusted SQL expressions, never user input.

    Args:
        relation_alias: SQL alias for a ``pg_class`` row.
        grantee_oid_expression: SQL expression resolving to the app role OID.
        expected_privileges_sql: PostgreSQL ``text[]`` expression containing
            the sorted direct, non-grantable privilege names.
    Returns:
        Parenthesized SQL boolean expression for catalog checks.
    """
    acl = (
        f"aclexplode(COALESCE({relation_alias}.relacl, "
        f"acldefault('r', {relation_alias}.relowner)))"
    )
    return f"""(
        COALESCE((
            SELECT array_agg(
                DISTINCT acl.privilege_type ORDER BY acl.privilege_type
            )
            FROM {acl} AS acl
            WHERE acl.grantee = {grantee_oid_expression}
              AND NOT acl.is_grantable
        ), ARRAY[]::text[]) = {expected_privileges_sql}
        AND NOT EXISTS (
            SELECT 1 FROM {acl} AS acl
            WHERE acl.grantee = {grantee_oid_expression}
              AND acl.is_grantable
        )
        AND NOT EXISTS (
            SELECT 1 FROM {acl} AS acl
            WHERE acl.grantee = 0
        )
        AND NOT EXISTS (
            SELECT 1
            FROM pg_attribute AS attribute
            CROSS JOIN LATERAL aclexplode(attribute.attacl) AS acl
            WHERE attribute.attrelid = {relation_alias}.oid
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
              AND (
                  acl.grantee = 0
                  OR acl.grantee = {grantee_oid_expression}
              )
        )
    )"""


def postgres_tenant_table_acl_sql(
    relation_alias: str,
    grantee_oid_expression: str,
) -> str:
    """Build the canonical direct ACL predicate for a tenant table."""
    expected = (
        f"CASE WHEN {relation_alias}.relname = 'audit_log' "
        "THEN ARRAY['INSERT', 'SELECT']::text[] "
        "ELSE ARRAY['DELETE', 'INSERT', 'SELECT', 'UPDATE']::text[] END"
    )
    return postgres_direct_relation_acl_sql(
        relation_alias,
        grantee_oid_expression,
        expected_privileges_sql=expected,
    )
