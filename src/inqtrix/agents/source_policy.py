"""Shared source and one-shot execution policy for both agent brains.

The Agent Desk exposes source availability independently from the model's
choice to use a tool.  Enforcement therefore belongs at the tool/task
dispatch choke points, not only in prompts.  This module is the single policy
definition consumed by the cognitive kernel, the workspace phase machine,
and child-run submission.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from inqtrix.core.results import SourcePolicy
from inqtrix.exceptions import AgentPolicyDenied

log = logging.getLogger("inqtrix")

WEB_TOOL_NAMES = frozenset(
    {
        "web_instant",
        "run_web_research",
        "delegate_batch",
    }
)
"""Kernel tool names that contact the public web.

``delegate_batch`` sits here for tool-use-counter parity: rehydration
counts one web use per ToolMessage NAME, so the live batch records
exactly ONE web use per batch (never per child). Side effect, accepted
and deliberate: ``source_policy.web=disabled`` removes the whole batch
tool — research assignments would be denied per-assignment anyway, and
a deep_mission-only batch degrades to the still-available single
``run_deep_mission`` tool."""

KNOWLEDGE_TOOL_NAMES = frozenset(
    {"search_project_knowledge", "read_project_document"}
)
"""Kernel tool names that access project knowledge."""

WEB_TASK_KINDS = frozenset({"web_instant", "web_research"})
"""Workspace-agent task kinds that contact the public web."""

KNOWLEDGE_TASK_KINDS = frozenset({"rag_query", "file_analysis"})
"""Workspace-agent task kinds that access project knowledge."""

_ALL_TASK_KINDS = WEB_TASK_KINDS | KNOWLEDGE_TASK_KINDS | {"synthesis"}
_KNOWLEDGE_ONLY_TOOLS = frozenset(
    {
        "ask_user",
        "search_project_knowledge",
        "read_project_document",
        # Read-only, no web exposure — and the recovery pointer of the
        # context-archive offload ("Volltext im Lauf-Archiv:
        # read_canvas(...)") must stay callable in this directive too.
        "read_canvas",
        # Editor reads (P7-E1) are the user's own documents — internal
        # content like the canvas, no web exposure. They are deliberately
        # NOT in KNOWLEDGE_TOOL_NAMES either: source_policy.knowledge
        # governs the knowledge BASE, not the user's editor.
        "read_editor_document",
        "search_editor_document",
    }
)


def coerce_source_policy(value: object = None) -> SourcePolicy:
    """Normalize a policy or a deps-like owner, defaulting legacy omission.

    The both-available default is the documented backward-compatible wire
    behavior.  This helper also keeps focused task-executor test seams that
    predate the new deps attribute representative of an omitted policy.
    """
    candidate = getattr(value, "source_policy", value)
    if isinstance(candidate, SourcePolicy):
        return candidate
    if isinstance(candidate, dict):
        return SourcePolicy.model_validate(candidate)
    return SourcePolicy()


def effective_source_policy(
    policy: SourcePolicy,
    execution_directive: str = "",
) -> SourcePolicy:
    """Return the source policy after a one-shot directive is applied.

    A directive is intentionally stronger than the session preference: it is
    the user's explicit instruction for this message only.  Deployment and
    skill policy remain separate, higher-precedence gates.
    """
    if execution_directive == "quick_web":
        return SourcePolicy(web="available", knowledge="disabled")
    if execution_directive == "knowledge_only":
        return SourcePolicy(web="disabled", knowledge="available")
    return policy.model_copy()


def require_kernel_tool_allowed(
    tool_name: str,
    *,
    policy: SourcePolicy,
    execution_directive: str = "",
) -> None:
    """Fail loudly when a kernel tool violates the enforced run policy."""
    if kernel_tool_allowed(
        tool_name,
        policy=policy,
        execution_directive=execution_directive,
    ):
        return
    if execution_directive == "knowledge_only" and tool_name not in (
        _KNOWLEDGE_ONLY_TOOLS
    ):
        _deny("execution_directive=knowledge_only", tool_name)
    if tool_name in WEB_TOOL_NAMES:
        _deny("source_policy.web=disabled", tool_name)
    if tool_name in KNOWLEDGE_TOOL_NAMES:
        _deny("source_policy.knowledge=disabled", tool_name)
    _deny("effective_tool_policy", tool_name)


def kernel_tool_allowed(
    tool_name: str,
    *,
    policy: SourcePolicy,
    execution_directive: str = "",
) -> bool:
    """Whether a kernel tool belongs to the effective run tool surface."""
    if execution_directive == "knowledge_only" and tool_name not in (
        _KNOWLEDGE_ONLY_TOOLS
    ):
        return False
    if tool_name in WEB_TOOL_NAMES and policy.web != "available":
        return False
    if tool_name in KNOWLEDGE_TOOL_NAMES and policy.knowledge != "available":
        return False
    return True


def require_task_allowed(task_kind: str, *, policy: SourcePolicy) -> None:
    """Fail loudly when a workspace-agent task violates source policy."""
    if task_kind in WEB_TASK_KINDS and policy.web != "available":
        _deny("source_policy.web=disabled", task_kind)
    if task_kind in KNOWLEDGE_TASK_KINDS and policy.knowledge != "available":
        _deny("source_policy.knowledge=disabled", task_kind)


def allowed_task_kinds_for_policy(
    skill_kinds: set[str] | None,
    *,
    policy: SourcePolicy,
) -> set[str] | None:
    """Intersect the skill task allowlist with the effective source policy.

    ``None`` is retained only when neither source nor skills restrict the
    historical surface.  This keeps existing prompts byte-identical for old
    clients while ensuring a disabled source becomes a planner repair error,
    before any task is persisted.
    """
    if skill_kinds is None and (
        policy.web == "available" and policy.knowledge == "available"
    ):
        return None
    allowed = set(skill_kinds) if skill_kinds is not None else set(_ALL_TASK_KINDS)
    if policy.web != "available":
        allowed -= WEB_TASK_KINDS
    if policy.knowledge != "available":
        allowed -= KNOWLEDGE_TASK_KINDS
    allowed.add("synthesis")
    return allowed


def execution_payload(
    *,
    execution_directive: str,
    effective_mode: str,
    response_form: str,
    depth: str,
    model: str | None,
    reasoning_effort: str | None,
    source_policy: SourcePolicy,
    consent_reason: str,
    tool_use_counts: dict[str, int] | None = None,
    limits: dict[str, object] | None = None,
    tool_grants: "Iterable[str] | None" = None,
) -> dict[str, object]:
    """Canonical run/snapshot execution projection for Agent Desk.

    Every agent algorithm emits this exact key set, including zero counts and
    empty optional strings, so the transparency UI never has to infer whether
    a missing field means "off", "unknown", or an older code path.
    """
    counts = tool_use_counts or {}
    return {
        "execution_directive": execution_directive,
        "effective_mode": effective_mode,
        "response_form": response_form,
        "depth": depth,
        "model": model or "",
        "reasoning_effort": reasoning_effort or "",
        "source_policy": {
            "web": source_policy.web,
            "knowledge": source_policy.knowledge,
        },
        "consent_reason": consent_reason,
        "tool_use_counts": {
            "web": max(0, int(counts.get("web", 0) or 0)),
            "knowledge": max(0, int(counts.get("knowledge", 0) or 0)),
        },
        # Server-authored limit facts only. The model never writes this
        # block; absent limits remain an explicit empty object for older or
        # non-agent callers instead of being guessed by the UI.
        "limits": dict(limits or {}),
        # Run-wide tool grants (P6B) — explicit empty list for engines
        # without grants, never inferred by the UI.
        "tool_grants": sorted(tool_grants or ()),
    }


def _deny(reason: str, tool_or_task: str) -> None:
    log.warning(
        "Agent-Werkzeug %s durch effektive Quellenrichtlinie blockiert "
        "(%s).",
        tool_or_task,
        reason,
    )
    raise AgentPolicyDenied(
        f"Werkzeug {tool_or_task} ist durch die Quellenrichtlinie "
        f"nicht erlaubt ({reason})."
    )
