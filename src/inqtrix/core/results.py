"""Algorithm-neutral request and result models.

``RunRequest`` is what an application service hands to an algorithm;
``AgentResult`` is what comes back. Both are deliberately lean in this
phase: the typed evidence/citation fields join ``AgentResult`` when
the unified evidence layer lands, and knowledge-specific request
fields (filters, attachments) join ``RunRequest`` with the knowledge
algorithm. Extensions are additive.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


WebRecency = Literal["day", "week", "month", "year"]


class SourcePolicy(BaseModel):
    """Per-run availability of the Agent Desk's external sources."""

    model_config = ConfigDict(extra="forbid")

    web: Literal["available", "disabled"] = Field(
        "available",
        description=(
            "Whether web tools are available to this run. The default "
            "preserves the historical tool surface."
        ),
    )
    """Whether web tools are available to this run; defaults to available for backward compatibility."""
    knowledge: Literal["available", "disabled"] = Field(
        "available",
        description=(
            "Whether project-knowledge tools are available to this run. "
            "The default preserves the historical tool surface."
        ),
    )
    """Whether project-knowledge tools are available to this run; defaults to available for backward compatibility."""


class RunRequest(BaseModel):
    """One execution request, independent of the HTTP wire format.

    The HTTP layer (chat completions, native runs) and the library
    layer (``ResearchAgent``) both normalize into this shape before
    dispatching through the :class:`~inqtrix.core.algorithms.AlgorithmRegistry`.
    """

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(
        "research",
        description=(
            "Algorithm id to execute (a registered "
            "``AgentAlgorithm.id``). Validation happens at registry "
            "lookup, not here, so the request model stays decoupled "
            "from the registered set."
        ),
    )
    """Algorithm id to execute (a registered ``AgentAlgorithm.id``). Validation happens at registry lookup, not here, so the request model stays decoupled from the registered set."""
    question: str = Field(
        ...,
        description=(
            "The normalized current user question/task. For chat "
            "protocols this is the text of the latest user message."
        ),
    )
    """The normalized current user question/task. For chat protocols this is the text of the latest user message."""
    history: str = Field(
        "",
        description=(
            "Pre-formatted conversation history block handed to the "
            "agent for context. Formatting (role labels, truncation) "
            "is a protocol concern resolved by the calling service, "
            "not by the algorithm."
        ),
    )
    """Pre-formatted conversation history block handed to the agent for context. Formatting (role labels, truncation) is a protocol concern resolved by the calling service, not by the algorithm."""
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "The raw chat messages array as received, for algorithms "
            "or diagnostics that need more than the flattened "
            "``history`` block. Optional; native run submissions may "
            "leave it empty."
        ),
    )
    """The raw chat messages array as received, for algorithms or diagnostics that need more than the flattened ``history`` block. Optional; native run submissions may leave it empty."""
    agent_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Echo of the validated per-request overrides (already "
            "APPLIED to the resolved settings carried by the run "
            "context). Kept on the request for run summaries and "
            "audit, not re-applied by algorithms."
        ),
    )
    """Echo of the validated per-request overrides (already APPLIED to the resolved settings carried by the run context). Kept on the request for run summaries and audit, not re-applied by algorithms."""
    knowledge_filters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Knowledge-retrieval scope for ``mode=knowledge`` requests "
            "(``collection_ids`` list; optional ``profile``, ``top_k``, and "
            "``final_k`` overrides). ``top_k`` is the per-query retrieval width "
            "(1..50); ``final_k`` pins the surfaced-evidence count "
            "(1..EVIDENCE_K_MAX), overriding the profile factor. Inert for "
            "algorithms that do not retrieve from internal documents."
        ),
    )
    """Knowledge-retrieval scope for ``mode=knowledge`` requests (``collection_ids`` list; optional ``profile``, ``top_k``, ``final_k`` overrides). ``top_k`` is the per-query retrieval width (1..50); ``final_k`` pins the surfaced-evidence count (1..EVIDENCE_K_MAX), overriding the profile factor. Inert for algorithms that do not retrieve from internal documents."""
    autonomy: str = Field(
        default="",
        description=(
            "Workspace-agent permission mode (strict/balanced/autonomous, "
            "decision E16); empty for every other mode."
        ),
    )
    """Workspace-agent permission mode (``strict``/``balanced``/``autonomous``, decision E16); empty for every other mode. The route validates the vocabulary, the agent algorithm consumes it — inert everywhere else."""
    session_id: str = Field(
        default="",
        description=(
            "Agent-desk session the run belongs to (decision E15); empty "
            "when sessionless."
        ),
    )
    """Agent-desk session the run belongs to (decision E15); empty when the run is sessionless. Mirrors the run row's ``session_id`` so the agent algorithm can anchor the session memo without a store read."""
    document_id: str = Field(
        default="",
        description=(
            "Target editor document for a workspace-agent patch assignment "
            "(M7); empty when the run has no edit target."
        ),
    )
    """Target editor document for a workspace-agent patch assignment (M7). When set, the agent proposes an ``editor_patch`` against this document after synthesis and parks for the ALWAYS-gated patch approval (E16 write invariant); the agent never applies the patch itself. Empty (the default) skips the patch phase entirely — inert for every other mode."""
    response_form: str = Field(
        default="",
        description=(
            "Workspace-agent output-form override (plan M1): ``chat`` "
            "forces the inline chat answer, ``canvas`` the session memo; "
            "empty lets the intake profile decide. Inert for every "
            "other mode."
        ),
    )
    """Workspace-agent output-form override (plan M1). ``chat`` forces the run-local inline answer artifact, ``canvas`` the session memo canvas; empty (the default, wire value ``auto``) delegates the decision to the intake profile's ``response_form``. A patch assignment (``document_id`` set) always uses canvas — the patch pipeline consumes the memo as source material. Inert for every other mode."""
    skill_ids: tuple[str, ...] = Field(
        default=(),
        description=(
            "Explicitly attached skills (plan M3, composer chips). "
            "Admission (visibility, count cap) happens at the runs "
            "router; the agent runtime loads and enforces them."
        ),
    )
    """Explicitly attached skill ids (plan M3). The runs router admits them (visible-to-caller check — an invisible skill is a loud 404 — and the ``skills_max_attached`` cap); the agent runtime injects their instructions, runs the clarification point check, and applies ``requires_plan``/``allowed_tools``. Inert for every non-agent mode."""
    skill_revisions: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Server-admitted integer revision for every attached skill; "
            "used to prevent policy drift between admission and resume."
        ),
    )
    """Server-pinned skill revisions keyed by skill id; clients do not supply this field directly."""
    tool_directives: tuple[str, ...] = Field(
        default=(),
        description=(
            "Hard planner/kernel tool hints from the composer's "
            "``/``-menu functions group (plan M3, e.g. web_research)."
        ),
    )
    """Hard tool hints (plan M3 `3.2`): the composer's ``/``-functions (``web_research``, ``rag_query``) telling the planner/kernel the user explicitly asked for this tool family. Whitelisted at the runs router (unknown directive = 400). Inert for every non-agent mode."""
    source_policy: SourcePolicy = Field(
        default_factory=SourcePolicy,
        description=(
            "Per-run availability of web and project-knowledge tools. "
            "Missing means both sources remain available."
        ),
    )
    """Web and project-knowledge availability inherited by delegated agent work; both are available when omitted."""
    web_recency: WebRecency | None = Field(
        default=None,
        description=(
            "Provider-neutral web recency filter inherited by a delegated "
            "research run. Null lets the research graph infer recency from "
            "the question."
        ),
    )
    """Provider-neutral recency filter for delegated web research; null delegates to graph classification."""
    execution_directive: Literal["", "quick_web", "knowledge_only"] = Field(
        "",
        description=(
            "Optional one-shot, server-enforced Agent Desk route. Empty "
            "uses normal automatic routing."
        ),
    )
    """One-shot server-enforced route; empty delegates to the normal agent policy."""

    @model_validator(mode="after")
    def _exclusive_directive_contract(self) -> "RunRequest":
        """Reject ambiguous legacy-hint plus enforced-route requests."""
        if self.execution_directive and self.tool_directives:
            raise ValueError(
                "execution_directive und tool_directives duerfen nicht "
                "gleichzeitig gesetzt sein"
            )
        return self


class AgentResult(BaseModel):
    """Outcome of one algorithm execution.

    Carries the answer plus the raw provider-shaped result dict the
    HTTP layer's existing serialization consumes
    (``ResearchResult.from_raw``, usage extraction, cancel detection).
    Keeping ``raw`` verbatim is what makes the registry introduction a
    pure structural change — the wire payloads are built from exactly
    the same dict as before.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        description="Final answer text (empty for cancelled runs).",
    )
    """Final answer text (empty for cancelled runs)."""
    result_type: str = Field(
        "research_result",
        description=(
            "Discriminator for consumers that render different result "
            "kinds (research report vs direct chat vs future "
            "knowledge results)."
        ),
    )
    """Discriminator for consumers that render different result kinds (research report vs direct chat vs future knowledge results)."""
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The unmodified graph/provider result dict (``answer``, "
            "``usage``, ``result_state``). Services keep building "
            "wire payloads from this dict so existing contracts stay "
            "byte-identical; typed projections will be layered on "
            "top, never replace it destructively."
        ),
    )
    """The unmodified graph/provider result dict (``answer``, ``usage``, ``result_state``). Services keep building wire payloads from this dict so existing contracts stay byte-identical; typed projections will be layered on top, never replace it destructively."""

    @property
    def cancelled(self) -> bool:
        """Whether the underlying run finished in the cancelled state."""
        result_state = self.raw.get("result_state") or {}
        return bool(result_state.get("cancelled"))

    @property
    def cancel_reason(self) -> str:
        """Machine-readable cancellation cause emitted by the algorithm."""
        result_state = self.raw.get("result_state") or {}
        return str(result_state.get("cancel_reason") or "")
