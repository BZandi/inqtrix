"""SQLAlchemy Core definition of the prompt-template schema.

Separate ``MetaData`` on purpose (immutable-snapshot rule): the table
arrives with revision 0006. Timestamps are unix-seconds doubles,
mirroring the in-memory records exactly.

``owner_sub`` is nullable text, NOT a foreign key into the ``users``
mirror — ``NULL`` marks open templates created by the
anonymous/static principals (visible to everyone, the same legacy
rule knowledge collections use), and template validity never depends
on the mirror row's lifecycle.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Float,
    Index,
    MetaData,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

prompt_template_metadata = MetaData()

prompt_templates = Table(
    "prompt_templates",
    prompt_template_metadata,
    Column("id", Text, primary_key=True),
    Column(
        "tenant_id", Text, nullable=False, server_default=text("'default'")
    ),
    Column("owner_sub", Text, nullable=True),
    Column("title", Text, nullable=False),
    Column("label", Text, nullable=False),
    Column("category", Text, nullable=True),
    Column("content_markdown", Text, nullable=False),
    Column("visibility", JSONB, nullable=False, server_default=text("'{}'")),
    Column(
        "include_in_autocomplete",
        Boolean,
        nullable=False,
        server_default=text("true"),
    ),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    CheckConstraint(
        "category IS NULL OR category IN "
        "('instruction', 'function', 'context')",
        name="ck_prompt_templates_category",
    ),
    Index("ix_prompt_templates_owner", "tenant_id", "owner_sub"),
)
