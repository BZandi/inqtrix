"""SQLAlchemy Core definition of the quota schema.

Separate ``MetaData`` on purpose (immutable-snapshot rule): the tables
arrive with revision 0007. Timestamps and counters are unix-seconds
doubles, mirroring the in-memory store exactly.

Two tables: ``quota_usage_counters`` holds one row per
``(tenant, subject, dimension, period_start)`` window (a new window =
a new row, so old months stay as harmless history), and
``quota_limits`` holds the admin-set overrides plus the tenant default
(the sentinel ``__quota_default__`` subject). Neither ``subject_sub``
is a foreign key into the users mirror — quota state outlives mirror
rows, same rule as PATs and templates.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    Index,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    text,
)

quota_metadata = MetaData()

quota_usage_counters = Table(
    "quota_usage_counters",
    quota_metadata,
    Column(
        "tenant_id", Text, nullable=False, server_default=text("'default'")
    ),
    Column("subject_sub", Text, nullable=False),
    Column("dimension", Text, nullable=False),
    Column("period_start", Float, nullable=False),
    Column("used", BigInteger, nullable=False, server_default=text("0")),
    Column("updated_at", Float, nullable=False),
    PrimaryKeyConstraint(
        "tenant_id",
        "subject_sub",
        "dimension",
        "period_start",
        name="pk_quota_usage_counters",
    ),
    Index("ix_quota_usage_subject", "tenant_id", "subject_sub"),
)

quota_limits = Table(
    "quota_limits",
    quota_metadata,
    Column(
        "tenant_id", Text, nullable=False, server_default=text("'default'")
    ),
    Column("subject_sub", Text, nullable=False),
    Column("dimension", Text, nullable=False),
    Column("limit_value", BigInteger, nullable=False),
    Column("set_by_sub", Text, nullable=False),
    Column("set_at", Float, nullable=False),
    PrimaryKeyConstraint(
        "tenant_id",
        "subject_sub",
        "dimension",
        name="pk_quota_limits",
    ),
)
