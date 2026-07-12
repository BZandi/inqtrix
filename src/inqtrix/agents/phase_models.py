"""Structured-output models of the workspace-agent phases (§4).

Every LLM-facing phase parses into one of these via
:func:`inqtrix.agents.patterns._structured.structured_call`. The
``complete_structured`` gotcha applies: EVERY field must be required for
the provider's native structured mode, so all fields carry explicit
values in the prompt contract and "nothing" is the empty string/list —
never an omitted key.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

READINESS_ROUTES = ("plan_now", "discover_first", "ask_user_first")
"""Deterministic intake routing outcomes (§4 Phase 0)."""

GAP_KINDS = (
    "missing",
    "outdated",
    "contradictory",
    "insufficient_detail",
    "unknown_scope",
)


class ClarificationOptionModel(BaseModel):
    """One selectable answer option of a clarification question."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        description="Short answer option the user can pick (<= 60 chars)."
    )
    description: str = Field(
        description=(
            "Optional one-line explanation of the option ('' when the "
            "label is self-explanatory)."
        )
    )


class ClarificationQuestionModel(BaseModel):
    """One structured clarification question with pickable options.

    Free text is ALWAYS allowed besides the options (decision #8), so the
    options only cover the LIKELY answers — the model never has to force
    an exhaustive enumeration. Stable ids are assigned deterministically
    by the sanitizer (:func:`inqtrix.agents.clarification.sanitize_questions`),
    never by the LLM.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        description="The full question to the user, German, one sentence."
    )
    options: list[ClarificationOptionModel] = Field(
        description=(
            "2-4 likely answer options (empty when only free text makes "
            "sense — the question is then asked without chips)."
        )
    )
    multi_select: bool = Field(
        description=(
            "True only when several options can apply at once "
            "(the user may then pick more than one)."
        )
    )


class AssignmentProfile(BaseModel):
    """Phase 0 intake output — what the assignment actually asks."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(description="ISO language of the assignment.")
    intent: str = Field(description="One-sentence reading of the goal.")
    scope_clarity: str = Field(
        description="clear | underspecified | ambiguous"
    )
    needs_web: bool
    needs_internal: bool
    needs_files: bool
    recency_sensitive: bool
    contested_topic: bool
    sub_goals: list[str] = Field(
        description="Decomposed sub-goals (empty when atomic)."
    )
    clarification_questions: list[ClarificationQuestionModel] = Field(
        description=(
            "Questions the user must answer BEFORE useful work can start "
            "(empty when none). Each carries 2-4 likely answer options so "
            "the user can click instead of type."
        )
    )
    response_form: str = Field(
        description=(
            "chat | canvas — chat fuer kurze, konversationale Antworten "
            "(Frage, Vergleich, Einschaetzung, Folgefrage); canvas fuer "
            "Berichte, Vermerke und dokumentartige Deliverables, die "
            "eigenstaendig weiterverwendet oder bearbeitet werden."
        )
    )
    success_criteria: list[str] = Field(
        description="3-5 measurable criteria, German."
    )


class DiscoveryGap(BaseModel):
    """One knowledge gap the discovery analyst identified."""

    model_config = ConfigDict(extra="forbid")

    gap_id: str
    kind: str = Field(description="One of the GAP_KINDS values.")
    description: str
    recommended_capability: str = Field(
        description=(
            "web_research | web_instant | rag_query | file_analysis"
        )
    )
    suggested_queries: list[str]
    blocking: bool = Field(
        description="True only when planning cannot proceed without an answer."
    )


class DiscoveryFinding(BaseModel):
    """One already-known fact with its source reference."""

    model_config = ConfigDict(extra="forbid")

    fact: str
    source: str = Field(
        description="Reference label (doc:{id}#{chunk} or URL)."
    )
    fresh: bool


class DiscoveryResult(BaseModel):
    """Phase 1 output — the analyst's compression of all probes."""

    model_config = ConfigDict(extra="forbid")

    known_facts: list[DiscoveryFinding]
    gaps: list[DiscoveryGap]
    questions_for_user: list[ClarificationQuestionModel] = Field(
        description=(
            "Only questions that BLOCK planning (else empty), each with "
            "2-4 likely answer options."
        )
    )
    sufficient_to_plan: bool


class SufficiencyJudgement(BaseModel):
    """Phase 6 fast-tier coverage verdict (gate semantics)."""

    model_config = ConfigDict(extra="forbid")

    coverage: str = Field(description="covered | partial | uncovered")
    missing: list[str] = Field(
        description="Criteria without sufficient evidence (empty when covered)."
    )


class ContradictionFinding(BaseModel):
    """One contradiction between evidence positions."""

    model_config = ConfigDict(extra="forbid")

    internal_position: str
    external_position: str
    severity: str = Field(description="hard | soft")
    likely_cause: str


class ContradictionReport(BaseModel):
    """Phase 6 contradiction-analysis output over overlapping claim pairs."""

    model_config = ConfigDict(extra="forbid")

    contradictions: list[ContradictionFinding]


class ReportSection(BaseModel):
    """One planned memo section."""

    model_config = ConfigDict(extra="forbid")

    title: str
    focus: str = Field(description="What the section must establish.")
    criterion_ids: list[str] = Field(
        description="Indices (as strings) of the success criteria covered."
    )
    evidence_labels: list[str] = Field(
        description="[K#]/[W#] labels the section should cite."
    )


class ReportOutline(BaseModel):
    """Phase 8 outline — sections before any prose is written."""

    model_config = ConfigDict(extra="forbid")

    title: str
    sections: list[ReportSection]


class AgentCriticFinding(BaseModel):
    """One critic finding against the memo."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        description=(
            "uncited_claim | criterion_unmet | contradiction_omitted | "
            "instruction_violation | language_error | memory_conflict"
        )
    )
    detail: str
    suggested_fix: str


class AgentCriticReport(BaseModel):
    """Phase 9 verdict over the finished memo (precomputed facts given)."""

    model_config = ConfigDict(extra="forbid")

    findings: list[AgentCriticFinding]
    criteria_covered: list[str] = Field(
        description="Success criteria the memo demonstrably meets."
    )
    criteria_uncovered: list[str] = Field(
        description="Success criteria the memo misses (empty when none)."
    )
    verdict: str = Field(description="pass | revise | research")


class MemoryCandidateModel(BaseModel):
    """One proposed long-term memory extracted from a completed run."""

    model_config = ConfigDict(extra="forbid")

    scope: str = Field(description="user | workspace | project | agent")
    category: str = Field(
        description="preference | project_fact | strategy | correction"
    )
    content: str = Field(
        description="Short editable memory text, not raw evidence."
    )
    reason: str = Field(description="Why retaining this may help later.")
    confidence: float = Field(description="Confidence from 0.0 to 1.0.")


class MemoryReflection(BaseModel):
    """Candidate-only long-term memory reflection output."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[MemoryCandidateModel]


class FileAnalysisSummary(BaseModel):
    """Quarantined file-analysis output (harness sub-agent or direct)."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Compact findings, <= 300 words.")
    key_quotes: list[str] = Field(
        description="Verbatim quotes worth citing (empty when none)."
    )


class SectionText(BaseModel):
    """One memo section's prose (structured so token usage is metered —
    plain ``complete()`` discards usage metadata on real providers)."""

    model_config = ConfigDict(extra="forbid")

    markdown: str = Field(description="The section prose, Markdown.")


class CitationRepairText(BaseModel):
    """A complete synthesis text after one citation-label repair."""

    model_config = ConfigDict(extra="forbid")

    markdown: str = Field(
        description=(
            "The complete original Markdown with only unsupported citation "
            "labels corrected or removed."
        )
    )
    """Complete Markdown after the sole bounded citation repair call."""


class QuickWebQuery(BaseModel):
    """One precise query for the deterministic quick-web route."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description=(
            "One self-contained search query derived from the current "
            "question and relevant conversation context."
        ),
    )
    """The single query sent to ``web.search.instant`` after any required approval."""
    recency: str = Field(
        description=(
            "Recency filter: '' when none is required, otherwise day, "
            "week, month, or year."
        )
    )
    """Optional provider recency filter; the model must emit an explicit empty string when no filter is needed."""


class DeepReviewFinding(BaseModel):
    """One fixable Deep defect targeted at chat or a concrete canvas."""

    model_config = ConfigDict(extra="forbid")

    target: Literal["chat", "artifact"] = Field(
        description="Output target carrying the defect."
    )
    """Whether the defect belongs to the chat answer or a canvas artifact."""
    artifact_id: str = Field(
        description="Concrete artifact id for target=artifact, otherwise ''."
    )
    """Artifact target id; empty for chat findings."""
    finding: str = Field(description="Concrete, fixable defect.")
    """Actionable issue for the single revision call."""


class DeepReviewVerdict(BaseModel):
    """Rubric verdict of the Deep verification pass (plan M4 `4.1.4`).

    Checkable criteria only — completeness, groundedness, named
    contradictions. ``findings`` empty = the answer passes; non-empty
    findings trigger exactly ONE revision round (process-supervision
    light, never a loop).
    """

    model_config = ConfigDict(extra="forbid")

    complete: bool = Field(
        description="True when the answer covers the full assignment."
    )
    """Whether the answer addresses every part of the assignment."""
    grounded: bool = Field(
        description=(
            "True when factual claims carry evidence labels/sources or "
            "unproven points are named as open."
        )
    )
    """Whether claims are evidenced or honestly marked open."""
    contradictions_named: bool = Field(
        description=(
            "True when uncertainties and counter-positions are named "
            "where relevant."
        )
    )
    """Whether relevant uncertainty/counter-evidence is surfaced."""
    findings: list[DeepReviewFinding] = Field(
        description=(
            "Concrete, fixable defects; EMPTY when the answer passes."
        )
    )
    """Actionable defects driving the single revision round."""


class DeepArtifactRevision(BaseModel):
    """One exact canvas replacement returned by Deep revision."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(description="Known artifact id to revise.")
    """Target artifact from the reviewed output bundle."""
    expected_revision: int = Field(
        ge=1,
        description="Exact reviewed revision required by atomic CAS.",
    )
    """Revision observed by the review call."""
    markdown: str = Field(description="Complete revised artifact body.")
    """Full replacement body; never a patch fragment."""


class DeepRevisionBundle(BaseModel):
    """Validated output of the one-and-only Deep revision call."""

    model_config = ConfigDict(extra="forbid")

    chat_markdown: str = Field(
        description="Complete revised chat answer, or the original unchanged."
    )
    """Complete chat output after applying chat-targeted findings."""
    artifacts: list[DeepArtifactRevision] = Field(
        description="Only canvas artifacts that require a revision."
    )
    """Content-only artifact replacements applied by atomic batch CAS."""
