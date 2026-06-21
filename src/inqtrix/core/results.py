"""Algorithm-neutral request and result models.

``RunRequest`` is what an application service hands to an algorithm;
``AgentResult`` is what comes back. Both are deliberately lean in this
phase: the typed evidence/citation fields join ``AgentResult`` when
the unified evidence layer lands, and knowledge-specific request
fields (filters, attachments) join ``RunRequest`` with the knowledge
algorithm. Extensions are additive.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
            "(``collection_ids`` list, optional ``top_k``). Inert for "
            "algorithms that do not retrieve from internal documents."
        ),
    )
    """Knowledge-retrieval scope for ``mode=knowledge`` requests (``collection_ids`` list, optional ``top_k``). Inert for algorithms that do not retrieve from internal documents."""


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

