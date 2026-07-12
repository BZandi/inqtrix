"""Skill records: server-enforced policy objects of the skill library.

Skills are NOT a fourth prompt-template category (plan M3 `3.1`):
templates are client-interpreted text, while a skill's fields are
ENFORCED server-side — ``clarification_points`` drive the intake gate,
``requires_plan`` the plan gate, ``allowed_tools`` the tool dispatch,
``invocation`` the model-disclosure block. The persistence machinery
mirrors prompt templates one-for-one (ownership, optimistic
concurrency, memory + Postgres backends); only the table is new.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

SKILL_DELIVERABLES = ("", "chat", "canvas", "email", "talking_points")
"""Output-form hint the skill pins ('' = the agent decides). ``email``
stays a pure format hint until a sending integration exists."""

SKILL_REQUIRES_PLAN = ("always", "auto", "never")
"""Plan-gate policy (plan `3.5`): ``always`` forces the gate even in
Auto, ``auto`` (default) follows the permission mode, ``never`` skips it
(a mission still shows its plan informatively). The STRICTEST value
across activated skills wins; the patch gate is never affected."""

SKILL_INVOCATIONS = ("user_only", "model_allowed")

SKILL_ALLOWED_TOOLS = (
    "search_project_knowledge",
    "read_project_document",
    "web_instant",
    "run_web_research",
    "run_deep_mission",
    "write_canvas",
    "propose_editor_patch",
)
"""The closed ``allowed_tools`` vocabulary (kernel tool names; the
library editor mirrors this list). A name outside it would be a silent
misconfiguration: the runtime union would restrict the run without any
tool ever matching the typo."""
"""Who may activate the skill: only the user (``/``-mention) or also the
model (disclosure block + ``load_skill``). Shared-in skills are ALWAYS
``user_only`` for the recipient — structural injection defense."""

MAX_CLARIFICATION_POINTS = 5
"""Hard cap on declared clarification points per skill."""

MAX_POINT_OPTIONS = 4
"""Options per clarification point (mirrors the M1 gate-round cap)."""


class SkillNotFound(KeyError):
    """Raised when a skill id is unknown (or hidden — same signal)."""


class SkillConflict(RuntimeError):
    """Raised when an optimistic-concurrency precondition fails.

    Same contract as the prompt-template conflict: a caller passing the
    ``updated_at`` it loaded asserts "overwrite only if nothing changed
    since"; a mismatch answers HTTP 409 instead of silently clobbering
    the intervening edit.
    """


@dataclass(frozen=True)
class SkillRecord:
    """One stored skill.

    Attributes:
        id: Server-assigned stable identifier (``sk_...``).
        tenant_id: Tenant scope (v1 runs one tenant per deployment).
        owner_sub: Creating OIDC subject; ``None`` = open skill
            (anonymous/static creators), visible and editable for all.
        label: The ``/``-mention token (``[a-z0-9-]``, unique per user
            surface by convention, not enforced).
        title: Display title in the skill library.
        description: One-liner for pickers and the disclosure block.
        when_to_use: Model-facing activation guidance (budgeted into
            the disclosure block, plan `3.3`).
        instructions_markdown: The skill body, loaded only on
            activation (progressive disclosure). May contain
            ``{{name}}`` placeholders that the clarification points
            fill (plan `3.4`).
        clarification_points: Declared inputs (max
            :data:`MAX_CLARIFICATION_POINTS`), each
            ``{id, name, question, options, required,
            default_assumption}`` with sanitized deterministic ids —
            the runtime point check asks ONLY what context cannot
            answer, through the M1 clarification machinery.
        deliverable: One of :data:`SKILL_DELIVERABLES`.
        allowed_tools: Tool allowlist enforced at dispatch when
            non-empty (union over activated skills); empty = no
            restriction.
        requires_plan: One of :data:`SKILL_REQUIRES_PLAN`.
        invocation: One of :data:`SKILL_INVOCATIONS`.
        argument_hint: Composer hint after the ``/``-token (free text).
        model_tier: Optional tier pin (R4): '' or high|mid|fast.
        effort: Optional reasoning-effort pin: '' or the effort tokens
            the routing layer accepts.
        include_in_autocomplete: Whether the ``/``-menu offers it.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of the last write; also the
            optimistic-concurrency anchor (:class:`SkillConflict`).
    """

    id: str
    tenant_id: str
    owner_sub: str | None
    label: str
    title: str
    description: str = ""
    when_to_use: str = ""
    instructions_markdown: str = ""
    clarification_points: tuple[dict[str, Any], ...] = ()
    deliverable: str = ""
    allowed_tools: tuple[str, ...] = ()
    requires_plan: str = "auto"
    invocation: str = "user_only"
    argument_hint: str = ""
    model_tier: str = ""
    effort: str = ""
    include_in_autocomplete: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0


@runtime_checkable
class SkillRepository(Protocol):
    """Persistence port for skills (memory + Postgres)."""

    async def create(self, record: SkillRecord) -> SkillRecord: ...

    async def get(self, skill_id: str, *, tenant_id: str) -> SkillRecord: ...

    async def list_for_tenant(
        self, *, tenant_id: str
    ) -> list[SkillRecord]: ...

    async def update(
        self,
        record: SkillRecord,
        *,
        expected_updated_at: float | None = None,
    ) -> SkillRecord: ...

    async def delete(self, skill_id: str, *, tenant_id: str) -> None: ...


def new_skill_id() -> str:
    """Mint one ``sk_``-prefixed identifier."""
    return f"sk_{uuid.uuid4().hex[:20]}"


class MemorySkillRepository:
    """Thread-safe in-process implementation (zero-infrastructure default)."""

    def __init__(self) -> None:
        self._records: dict[str, SkillRecord] = {}
        self._lock = threading.RLock()

    async def create(self, record: SkillRecord) -> SkillRecord:
        with self._lock:
            self._records[record.id] = record
            return record

    async def get(self, skill_id: str, *, tenant_id: str) -> SkillRecord:
        with self._lock:
            record = self._records.get(skill_id)
        if record is None or record.tenant_id != tenant_id:
            raise SkillNotFound(skill_id)
        return record

    async def list_for_tenant(self, *, tenant_id: str) -> list[SkillRecord]:
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.tenant_id == tenant_id
            ]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    async def update(
        self,
        record: SkillRecord,
        *,
        expected_updated_at: float | None = None,
    ) -> SkillRecord:
        with self._lock:
            current = self._records.get(record.id)
            if current is None or current.tenant_id != record.tenant_id:
                raise SkillNotFound(record.id)
            # Optimistic-concurrency guard under the same lock as the
            # write, so the check-then-write is atomic. None =
            # unconditional overwrite (legacy callers).
            if (
                expected_updated_at is not None
                and current.updated_at != expected_updated_at
            ):
                raise SkillConflict(record.id)
            stored = replace(record, updated_at=time.time())
            self._records[record.id] = stored
            return stored

    async def delete(self, skill_id: str, *, tenant_id: str) -> None:
        with self._lock:
            record = self._records.get(skill_id)
            if record is None or record.tenant_id != tenant_id:
                raise SkillNotFound(skill_id)
            del self._records[skill_id]
