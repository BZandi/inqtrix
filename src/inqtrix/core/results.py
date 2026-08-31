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

# Visible request bounds for the canvas context (P4). Violations are
# REJECTED with the offending field named — never silently truncated
# (no-silent-caps doctrine): the full text always either arrives intact
# or the caller learns exactly why it did not.
CANVAS_CONTEXT_MAX_COMMENTS = 20
CANVAS_QUOTE_MAX_CHARS = 2_000
CANVAS_QUOTE_CONTEXT_MAX_CHARS = 500
CANVAS_COMMENT_MAX_CHARS = 4_000


class CanvasComment(BaseModel):
    """One user comment anchored to a text selection in a canvas document.

    The anchor triple (``quote`` plus optional before/after context)
    follows the editor's anchoring convention; the server passes it to
    the model verbatim and never re-resolves it.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Canvas artifact the comment anchors to.",
    )
    """Canvas artifact the comment anchors to."""
    revision: int = Field(
        ...,
        ge=1,
        description="Artifact revision the user commented on.",
    )
    """Artifact revision the user commented on."""
    quote: str = Field(
        ...,
        min_length=1,
        max_length=CANVAS_QUOTE_MAX_CHARS,
        description="The selected text, verbatim.",
    )
    """The selected text, verbatim."""
    quote_before: str = Field(
        "",
        max_length=CANVAS_QUOTE_CONTEXT_MAX_CHARS,
        description="Text immediately before the selection (anchor context).",
    )
    """Text immediately before the selection (anchor context)."""
    quote_after: str = Field(
        "",
        max_length=CANVAS_QUOTE_CONTEXT_MAX_CHARS,
        description="Text immediately after the selection (anchor context).",
    )
    """Text immediately after the selection (anchor context)."""
    comment: str = Field(
        ...,
        min_length=1,
        max_length=CANVAS_COMMENT_MAX_CHARS,
        description="The user's note about the selection, verbatim.",
    )
    """The user's note about the selection, verbatim."""


class CanvasContext(BaseModel):
    """Structured canvas attachment of an Agent Desk submission (P4).

    Travels as a dedicated request field — NEVER serialized into
    ``question`` (the question column is clipped at persistence, reaches
    share-inbox titles before acceptance, and renders raw in the chat
    bubble). Snapshot semantics: the context reflects the moment of
    submission and is frozen with the first run segment's checkpointed
    user message — later canvas edits never rewrite it.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="The canvas document the user had open.",
    )
    """The canvas document the user had open."""
    revision: int = Field(
        ...,
        ge=1,
        description="Revision of that document at submission time.",
    )
    """Revision of that document at submission time."""
    comments: tuple[CanvasComment, ...] = Field(
        default=(),
        max_length=CANVAS_CONTEXT_MAX_COMMENTS,
        description="Queued selection comments, in queue order.",
    )
    """Queued selection comments, in queue order."""


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
            "Workspace-agent permission mode (strict/balanced/autonomous); "
            "empty for every other mode."
        ),
    )
    """Workspace-agent permission mode (``strict``/``balanced``/``autonomous``); empty for every other mode. The route validates the vocabulary, the agent algorithm consumes it — inert everywhere else."""
    session_id: str = Field(
        default="",
        description=(
            "Agent-desk session the run belongs to; empty when sessionless."
        ),
    )
    """Agent-desk session the run belongs to; empty when the run is sessionless. Mirrors the run row's ``session_id`` so the agent algorithm can anchor the session memo without a store read."""
    document_id: str = Field(
        default="",
        description=(
            "Target editor document for a workspace-agent patch assignment "
            "and empty when the run has no edit target."
        ),
    )
    """Target editor document for a workspace-agent patch assignment (M7). When set, the agent proposes an ``editor_patch`` against this document after synthesis and parks for the ALWAYS-gated patch approval (E16 write invariant); the agent never applies the patch itself. Empty (the default) skips the patch phase entirely — inert for every other mode."""
    response_form: str = Field(
        default="",
        description=(
            "Workspace-agent output-form override: ``chat`` "
            "forces the inline chat answer, ``canvas`` the session memo; "
            "empty lets the intake profile decide. Inert for every "
            "other mode."
        ),
    )
    """Workspace-agent output-form override. ``chat`` forces the run-local inline answer artifact, ``canvas`` the session memo canvas; empty (the default, wire value ``auto``) delegates the decision to the intake profile's ``response_form``. A patch assignment (``document_id`` set) always uses canvas — the patch pipeline consumes the memo as source material. Inert for every other mode."""
    skill_ids: tuple[str, ...] = Field(
        default=(),
        description=(
            "Explicitly attached skill ids from the composer chips. "
            "Admission (visibility, count cap) happens at the runs "
            "router; the agent runtime loads and enforces them."
        ),
    )
    """Explicitly attached skill ids. The runs router admits them (visible-to-caller check — an invisible skill is a loud 404 — and the ``skills_max_attached`` cap); the agent runtime injects their instructions, runs the clarification point check, and applies ``requires_plan``/``allowed_tools``. Inert for every non-agent mode."""
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
            "``/``-menu functions group, for example web_research."
        ),
    )
    """Hard tool hints from the composer's ``/``-functions (``web_research``, ``rag_query``) telling the planner/kernel the user explicitly asked for this tool family. Whitelisted at the runs router (unknown directive = 400). Inert for every non-agent mode."""
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
    canvas_context: CanvasContext | None = Field(
        default=None,
        description=(
            "Canvas attachment of an agent-kernel submission: the open "
            "document (id + revision) and queued selection comments. "
            "None for every other mode and for runs without canvas "
            "context."
        ),
    )
    """Canvas attachment of an agent-kernel submission (P4). Injected into the kernel user message as a fenced data section; frozen with the first segment's checkpoint; never inherited by child runs and never part of run summaries. ``None`` everywhere else."""
    report_requirement: str = Field(
        "",
        description=(
            "Composed result requirement set BEFORE the run: how the "
            "result has to look (structure, focus, audience). Already "
            "resolved and composed server-side from free text plus "
            "attached library rules. Empty when the user set none."
        ),
    )
    """Result requirement set at submit time, composed server-side (free text + attached library rules, each with its origin marker). The only way to state one for runs that never reach a plan gate — ``autonomous``, the speed tier, delegated children, and the kernel, which has no plan gate at all. A later plan-gate decision REPLACES it; an approval that says nothing leaves it standing. Never inherited by child runs."""
    attached_reports: tuple[dict[str, Any], ...] = Field(
        default=(),
        description=(
            "Research-Desk reports the user attached, already resolved "
            "server-side to {report_id, title, reference_count}. Names "
            "only — the bodies are fetched by the kernel tool."
        ),
    )
    """Research reports attached to an agent-kernel run, resolved against the caller's own visibility at submit time. Only the NAMES travel: a real report has a median of ~54k characters, and its sources become citable only by passing through the run's evidence ledger, which ``read_research_report`` does on demand. Never inherited by child runs; empty everywhere else."""

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

    @property
    def terminal_failure(self) -> "AgentTerminalFailure | None":
        """Typed projection of the shared returned-failure contract.

        Algorithms sometimes need to return usage and a safe diagnostic after
        a provider/model call while still declaring the run unsuccessful.  The
        native run service already consumes ``result_state._terminal_failure``;
        this projection lets chat and Agent adapters enforce the same contract
        without re-parsing untyped dictionaries or inventing a second status.
        Malformed failure markers fail closed as ``server_error``.
        """

        result_state = self.raw.get("result_state") or {}
        raw = result_state.get("_terminal_failure")
        if raw is None:
            return None
        try:
            return AgentTerminalFailure.model_validate(raw)
        except (TypeError, ValueError):
            return AgentTerminalFailure(
                type="server_error",
                message="The algorithm returned an invalid terminal failure marker.",
            )

    @property
    def successful(self) -> bool:
        """Whether the result is neither cancelled nor terminally failed."""

        return not self.cancelled and self.terminal_failure is None


class AgentTerminalFailure(BaseModel):
    """Stable failure returned with an :class:`AgentResult` and its usage."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        min_length=1,
        description="Stable machine-readable execution failure code.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Safe user-visible explanation of the failed result.",
    )
