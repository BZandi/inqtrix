"""Structured result types returned by :class:`~inqtrix.agent.ResearchAgent`.

All models are Pydantic v2 and fully serialisable to JSON. They are the
typed public view of the internal mutable agent state and are the
canonical surface for downstream consumers (HTTP responses, parity
tooling, custom integrations).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


def _empty_tier_counts() -> dict[str, int]:
    return {
        "primary": 0,
        "mainstream": 0,
        "stakeholder": 0,
        "unknown": 0,
        "low": 0,
    }


def _limit_items[T](items: list[T], limit: int | None) -> list[T]:
    if limit is None:
        return list(items)
    return list(items[:limit])


class Source(BaseModel):
    """A single cited source with its quality tier classification.

    Sources are the URLs that contributed to the final answer. Each
    source carries a ``tier`` label assigned by the active
    ``SourceTieringStrategy``; downstream consumers can use the tier to
    weight, filter, or visualise sources without re-running the tiering
    logic.
    """

    url: str = Field(
        ...,
        description=(
            "Absolute URL of the source as captured from the search "
            "provider's citations list. Already normalised (scheme + "
            "host lower-cased, fragment stripped) by the agent."
        ),
    )
    """Absolute URL of the source as captured from the search provider's citations list. Already normalised (scheme + host lower-cased, fragment stripped) by the agent."""
    tier: str = Field(
        "unknown",
        description=(
            "Quality tier assigned by the source-tiering strategy. One "
            "of ``primary`` (peer-reviewed / official), ``mainstream`` "
            "(established media), ``stakeholder`` (industry / NGO / "
            "association), ``unknown`` (no classification), or ``low`` "
            "(known low-quality domains)."
        ),
    )
    """Quality tier assigned by the source-tiering strategy. One of ``primary`` (peer-reviewed / official), ``mainstream`` (established media), ``stakeholder`` (industry / NGO / association), ``unknown`` (no classification), or ``low`` (known low-quality domains)."""


class Claim(BaseModel):
    """A consolidated claim extracted during research.

    This is the public, typed view of one entry in the internal
    consolidated claim ledger. It preserves the verification metadata
    (status, support/contradict counts, source-tier breakdown) needed
    for downstream inspection without leaking the full mutable agent
    state.
    """

    text: str = Field(
        ...,
        description=(
            "Canonical claim text after consolidation. May be a "
            "rewritten merge of several similar claims that the "
            "consolidation strategy considered semantically equivalent."
        ),
    )
    """Canonical claim text after consolidation. May be a rewritten merge of several similar claims that the consolidation strategy considered semantically equivalent."""
    status: str = Field(
        "unverified",
        description=(
            "Verification status assigned by the consolidation "
            "strategy. One of ``verified`` (multiple supporting "
            "sources, no contradiction), ``contested`` (supporting "
            "and contradicting sources both present), or ``unverified`` "
            "(insufficient evidence)."
        ),
    )
    """Verification status assigned by the consolidation strategy. One of ``verified`` (multiple supporting sources, no contradiction), ``contested`` (supporting and contradicting sources both present), or ``unverified`` (insufficient evidence)."""
    claim_type: str = Field(
        "fact",
        description=(
            "Claim taxonomy assigned by the extractor. One of "
            "``fact`` (verifiable factual statement), ``actor_claim`` "
            "(a position attributed to a named actor), or ``forecast`` "
            "(a forward-looking projection). Used by the answer "
            "composer to phrase claims appropriately."
        ),
    )
    """Claim taxonomy assigned by the extractor. One of ``fact`` (verifiable factual statement), ``actor_claim`` (a position attributed to a named actor), or ``forecast`` (a forward-looking projection). Used by the answer composer to phrase claims appropriately."""
    needs_primary: bool = Field(
        False,
        description=(
            "True when the claim's strength would benefit from a "
            "primary-tier source but is currently only backed by "
            "secondary sources. Surfaced in DEEP-profile reports as a "
            "transparency hint."
        ),
    )
    """True when the claim's strength would benefit from a primary-tier source but is currently only backed by secondary sources. Surfaced in DEEP-profile reports as a transparency hint."""
    status_reason: str = Field(
        "",
        description=(
            "Free-text justification produced by the consolidation "
            "strategy explaining why the claim received its ``status``. "
            "Empty string when no reason is available."
        ),
    )
    """Free-text justification produced by the consolidation strategy explaining why the claim received its ``status``. Empty string when no reason is available."""
    support_count: int = Field(
        0,
        description=(
            "Number of distinct sources that explicitly support this "
            "claim, after deduplication of near-identical phrasings."
        ),
    )
    """Number of distinct sources that explicitly support this claim, after deduplication of near-identical phrasings."""
    contradict_count: int = Field(
        0,
        description=(
            "Number of distinct sources that explicitly contradict "
            "this claim. Even one contradiction shifts the status from "
            "``verified`` to ``contested`` under the default policy."
        ),
    )
    """Number of distinct sources that explicitly contradict this claim. Even one contradiction shifts the status from ``verified`` to ``contested`` under the default policy."""
    source_tier_counts: dict[str, int] = Field(
        default_factory=_empty_tier_counts,
        description=(
            "Per-tier breakdown of the supporting sources for this "
            "claim. Keys are the five tier labels; values are integer "
            "counts. Always populated for all five tiers (zero for "
            "absent tiers) for stable downstream consumption."
        ),
    )
    """Per-tier breakdown of the supporting sources for this claim. Keys are the five tier labels; values are integer counts. Always populated for all five tiers (zero for absent tiers) for stable downstream consumption."""
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Source URLs that back this claim, in the order produced "
            "by the consolidation strategy (typically primary tiers "
            "first). Capped per ``ReportProfileTuning.claim_source_url_cap`` "
            "so the list stays bounded for large research runs."
        ),
    )
    """Source URLs that back this claim, in the order produced by the consolidation strategy (typically primary tiers first). Capped per ``ReportProfileTuning.claim_source_url_cap`` so the list stays bounded for large research runs."""

    @classmethod
    def from_consolidated(cls, claim_data: dict) -> Claim:
        """Build a public claim model from one internal consolidated claim.

        Args:
            claim_data: Raw consolidated-claim dict as produced by
                :meth:`~inqtrix.strategies.ClaimConsolidationStrategy.materialize`.
                Missing keys are tolerated and substituted with their
                public defaults so that older state snapshots remain
                consumable.

        Returns:
            A new :class:`Claim` instance with all numeric counts coerced
            to ``int`` and ``source_tier_counts`` normalised to contain
            all five tier keys.
        """
        return cls(
            text=claim_data.get("claim_text", ""),
            status=claim_data.get("status", "unverified"),
            claim_type=claim_data.get("claim_type", "fact"),
            needs_primary=bool(claim_data.get("needs_primary", False)),
            status_reason=str(claim_data.get("status_reason", "") or ""),
            support_count=int(claim_data.get("support_count", 0) or 0),
            contradict_count=int(claim_data.get("contradict_count", 0) or 0),
            source_tier_counts={
                **_empty_tier_counts(),
                **(claim_data.get("source_tier_counts", {}) or {}),
            },
            sources=list(claim_data.get("source_urls", []) or []),
        )


class SourceMetrics(BaseModel):
    """Aggregate quality breakdown of all sources used in the run.

    Computed by the active ``SourceTieringStrategy`` over the union of
    all citations contributed by the search node. Surfaced as the
    ``sources`` slice of :class:`ResearchMetrics`.
    """

    tier_counts: dict[str, int] = Field(
        default_factory=_empty_tier_counts,
        description=(
            "Per-tier count of distinct source URLs used in the run. "
            "Always contains all five tier keys for stable downstream "
            "consumption (zero for absent tiers)."
        ),
    )
    """Per-tier count of distinct source URLs used in the run. Always contains all five tier keys for stable downstream consumption (zero for absent tiers)."""
    quality_score: float = Field(
        0.0,
        description=(
            "Weighted source-quality score in ``[0.0, 1.0]``. Computed "
            "as the tier-weighted mean over all sources, where "
            "``primary=1.0``, ``mainstream=0.8``, ``stakeholder=0.45``, "
            "``unknown=0.35``, ``low=0.1`` (default weights). Higher is "
            "better. ``0.0`` means all sources are low-tier or unknown; "
            "``1.0`` means all primary."
        ),
    )
    """Weighted source-quality score in ``[0.0, 1.0]``. Computed as the tier-weighted mean over all sources, where ``primary=1.0``, ``mainstream=0.8``, ``stakeholder=0.45``, ``unknown=0.35``, ``low=0.1`` (default weights). Higher is better. ``0.0`` means all sources are low-tier or unknown; ``1.0`` means all primary."""


class ClaimMetrics(BaseModel):
    """Aggregate quality breakdown of all consolidated claims.

    Computed by the active ``ClaimConsolidationStrategy`` after the
    final round. Surfaced as the ``claims`` slice of
    :class:`ResearchMetrics`.
    """

    status_counts: dict[str, int] = Field(
        default_factory=lambda: {"verified": 0, "contested": 0, "unverified": 0},
        description=(
            "Per-status count of consolidated claims. Always contains "
            "the three status keys (``verified``, ``contested``, "
            "``unverified``) for stable downstream consumption."
        ),
    )
    """Per-status count of consolidated claims. Always contains the three status keys (``verified``, ``contested``, ``unverified``) for stable downstream consumption."""
    quality_score: float = Field(
        0.0,
        description=(
            "Weighted claim-quality score in ``[0.0, 1.0]`` defined as "
            "``(verified + 0.5 * contested) / total``. Higher is better. "
            "``0.0`` when no claims exist or none are verified or "
            "contested; ``1.0`` when every claim is verified."
        ),
    )
    """Weighted claim-quality score in ``[0.0, 1.0]`` defined as ``(verified + 0.5 * contested) / total``. Higher is better. ``0.0`` when no claims exist or none are verified or contested; ``1.0`` when every claim is verified."""


class ResearchMetrics(BaseModel):
    """Aggregate metrics for a single research run.

    All numeric fields default to ``0`` so that early-failed or
    minimal runs still deserialize cleanly.
    """

    rounds: int = Field(
        0,
        description=(
            "Total number of completed search rounds. ``0`` means the "
            "run never reached the search node (e.g. classify failure). "
            "Bounded above by ``AgentConfig.max_rounds``."
        ),
    )
    """Total number of completed search rounds. ``0`` means the run never reached the search node (e.g. classify failure). Bounded above by ``AgentConfig.max_rounds``."""
    elapsed_seconds: float = Field(
        0.0,
        description=(
            "Wall-clock duration of the run, in seconds, with two-"
            "decimal precision. Filled in by ``ResearchAgent.research`` "
            "after ``ResearchResult.from_raw`` returns; pure ``from_raw`` "
            "calls leave this at ``0.0``."
        ),
    )
    """Wall-clock duration of the run, in seconds, with two-decimal precision. Filled in by ``ResearchAgent.research`` after ``ResearchResult.from_raw`` returns; pure ``from_raw`` calls leave this at ``0.0``."""
    total_queries: int = Field(
        0,
        description=(
            "Total number of search queries dispatched across all "
            "rounds. Includes deduplicated queries; cache hits do not "
            "count as separate dispatches."
        ),
    )
    """Total number of search queries dispatched across all rounds. Includes deduplicated queries; cache hits do not count as separate dispatches."""
    total_citations: int = Field(
        0,
        description=(
            "Total number of distinct cited URLs across all rounds. "
            "Counts unique URLs only; duplicate citations from "
            "different searches collapse to one."
        ),
    )
    """Total number of distinct cited URLs across all rounds. Counts unique URLs only; duplicate citations from different searches collapse to one."""
    confidence: int = Field(
        0,
        description=(
            "Final evaluator confidence in ``[0, 10]``. ``0`` means no "
            "evaluator pass ran (early failure). Compared against "
            "``AgentConfig.confidence_stop`` to decide whether the loop "
            "stopped on confidence."
        ),
    )
    """Final evaluator confidence in ``[0, 10]``. ``0`` means no evaluator pass ran (early failure). Compared against ``AgentConfig.confidence_stop`` to decide whether the loop stopped on confidence."""
    aspect_coverage: float = Field(
        0.0,
        description=(
            "Fraction of required aspects covered by the final answer, "
            "in ``[0.0, 1.0]``. ``1.0`` means all aspects derived by "
            "the risk-scoring strategy are addressed; ``0.0`` means "
            "either no aspects were derived or none were covered."
        ),
    )
    """Fraction of required aspects covered by the final answer, in ``[0.0, 1.0]``. ``1.0`` means all aspects derived by the risk-scoring strategy are addressed; ``0.0`` means either no aspects were derived or none were covered."""
    evidence_consistency: int = Field(
        0,
        description=(
            "Evaluator-assigned consistency score in ``[0, 10]`` for "
            "the evidence pool. Higher means fewer cross-source "
            "contradictions. ``0`` indicates the score was not parsed "
            "from the evaluator response (logged as "
            "``_evidence_consistency_parsed`` fallback)."
        ),
    )
    """Evaluator-assigned consistency score in ``[0, 10]`` for the evidence pool. Higher means fewer cross-source contradictions. ``0`` indicates the score was not parsed from the evaluator response (logged as ``_evidence_consistency_parsed`` fallback)."""
    evidence_sufficiency: int = Field(
        0,
        description=(
            "Evaluator-assigned sufficiency score in ``[0, 10]`` for "
            "answering the question with the available evidence. ``0`` "
            "indicates the score was not parsed from the evaluator "
            "response."
        ),
    )
    """Evaluator-assigned sufficiency score in ``[0, 10]`` for answering the question with the available evidence. ``0`` indicates the score was not parsed from the evaluator response."""
    sources: SourceMetrics = Field(
        default_factory=SourceMetrics,
        description=(
            "Aggregate source-tier breakdown and quality score for the "
            "run. See :class:`SourceMetrics`."
        ),
    )
    """Aggregate source-tier breakdown and quality score for the run. See :class:`SourceMetrics`."""
    claims: ClaimMetrics = Field(
        default_factory=ClaimMetrics,
        description=(
            "Aggregate claim-status breakdown and quality score for the "
            "run. See :class:`ClaimMetrics`."
        ),
    )
    """Aggregate claim-status breakdown and quality score for the run. See :class:`ClaimMetrics`."""
    prompt_tokens: int = Field(
        0,
        description=(
            "Total prompt-token usage across all LLM calls in the run, "
            "summed from per-call ``usage.prompt_tokens`` returned by "
            "the providers. ``0`` when no provider returned token counts."
        ),
    )
    """Total prompt-token usage across all LLM calls in the run, summed from per-call ``usage.prompt_tokens`` returned by the providers. ``0`` when no provider returned token counts."""
    completion_tokens: int = Field(
        0,
        description=(
            "Total completion-token usage across all LLM calls in the "
            "run, summed from per-call ``usage.completion_tokens``. "
            "``0`` when no provider returned token counts."
        ),
    )
    """Total completion-token usage across all LLM calls in the run, summed from per-call ``usage.completion_tokens``. ``0`` when no provider returned token counts."""


class ResearchResultExportOptions(BaseModel):
    """Optional projection settings for serialised result payloads.

    All fields are optional switches or limits so downstream surfaces
    (parity, HTTP responses, custom tooling) can select the public
    result view they need without introducing parallel result models.
    The defaults emit the full payload — set fields to opt out.
    """

    include_answer: bool = Field(
        True,
        description=(
            "When ``True``, include the markdown answer text in the "
            "exported payload. Set ``False`` for metrics-only views "
            "where the answer is shipped through a separate channel."
        ),
    )
    """When ``True``, include the markdown answer text in the exported payload. Set ``False`` for metrics-only views where the answer is shipped through a separate channel."""
    include_metrics: bool = Field(
        True,
        description=(
            "When ``True``, include the full :class:`ResearchMetrics` "
            "object as a nested dict. Set ``False`` for minimal "
            "answer-only payloads."
        ),
    )
    """When ``True``, include the full :class:`ResearchMetrics` object as a nested dict. Set ``False`` for minimal answer-only payloads."""
    include_sources: bool = Field(
        True,
        description=(
            "When ``True``, include the ``top_sources`` list. Combine "
            "with ``max_sources`` to cap the list length."
        ),
    )
    """When ``True``, include the ``top_sources`` list. Combine with ``max_sources`` to cap the list length."""
    include_claims: bool = Field(
        True,
        description=(
            "When ``True``, include the ``top_claims`` list. Combine "
            "with ``max_claims`` to cap the list length."
        ),
    )
    """When ``True``, include the ``top_claims`` list. Combine with ``max_claims`` to cap the list length."""
    max_sources: int | None = Field(
        None,
        description=(
            "Optional hard cap on the exported ``top_sources`` list. "
            "``None`` (default) keeps the full list as produced by the "
            "agent. Use a positive integer to truncate for compact "
            "downstream payloads."
        ),
    )
    """Optional hard cap on the exported ``top_sources`` list. ``None`` (default) keeps the full list as produced by the agent. Use a positive integer to truncate for compact downstream payloads."""
    max_claims: int | None = Field(
        None,
        description=(
            "Optional hard cap on the exported ``top_claims`` list. "
            "``None`` (default) keeps the full list. Use a positive "
            "integer to truncate."
        ),
    )
    """Optional hard cap on the exported ``top_claims`` list. ``None`` (default) keeps the full list. Use a positive integer to truncate."""


class ResearchResult(BaseModel):
    """Complete result of a :meth:`ResearchAgent.research` call.

    The result is fully self-contained: serialise it via
    :meth:`pydantic.BaseModel.model_dump_json` for storage, or via
    :meth:`to_export_payload` for a configurable public view.
    """

    answer: str = Field(
        ...,
        description=(
            "Final markdown-formatted answer text including inline "
            "citation markers. Empty string when the run failed before "
            "the answer node ran."
        ),
    )
    """Final markdown-formatted answer text including inline citation markers. Empty string when the run failed before the answer node ran."""
    metrics: ResearchMetrics = Field(
        default_factory=ResearchMetrics,
        description=(
            "Aggregated quality and performance metrics for the run. "
            "See :class:`ResearchMetrics` for field-level semantics."
        ),
    )
    """Aggregated quality and performance metrics for the run. See :class:`ResearchMetrics` for field-level semantics."""
    top_sources: list[Source] = Field(
        default_factory=list,
        description=(
            "Most relevant sources used in the answer, ordered as "
            "produced by the agent (typically tier-then-recency). "
            "Capped at 60 items by :meth:`from_raw` to bound payload "
            "size; tighten further via "
            ":class:`ResearchResultExportOptions.max_sources`."
        ),
    )
    """Most relevant sources used in the answer, ordered as produced by the agent (typically tier-then-recency). Capped at 60 items by :meth:`from_raw` to bound payload size; tighten further via :class:`ResearchResultExportOptions.max_sources`."""
    top_claims: list[Claim] = Field(
        default_factory=list,
        description=(
            "Key consolidated claims with their verification metadata. "
            "Capped at 30 items by :meth:`from_raw`; tighten further "
            "via :class:`ResearchResultExportOptions.max_claims`."
        ),
    )
    """Key consolidated claims with their verification metadata. Capped at 30 items by :meth:`from_raw`; tighten further via :class:`ResearchResultExportOptions.max_claims`."""

    def to_export_payload(
        self,
        options: ResearchResultExportOptions | None = None,
    ) -> dict[str, Any]:
        """Build a configurable public payload from the typed result model.

        Use this to serialise the result to JSON-friendly dicts while
        opting selected sections in or out (e.g. metrics-only view for
        dashboards, or answer + sources without claim ledger).

        Args:
            options: Projection settings controlling which top-level
                keys are emitted and how the lists are capped. ``None``
                (default) uses :class:`ResearchResultExportOptions` with
                its defaults, which emits the full payload.

        Returns:
            A new ``dict`` with the selected top-level keys
            (``answer``, ``metrics``, ``top_sources``, ``top_claims``),
            each value already converted from the Pydantic model via
            :meth:`pydantic.BaseModel.model_dump`. Caller may mutate
            freely — the dict is independent of the source model.

        Example:
            >>> result.to_export_payload(
            ...     ResearchResultExportOptions(
            ...         include_sources=False,
            ...         max_claims=5,
            ...     )
            ... )
            {'answer': '...', 'metrics': {...}, 'top_claims': [...]}
        """
        export_options = options or ResearchResultExportOptions()
        payload: dict[str, Any] = {}

        if export_options.include_answer:
            payload["answer"] = self.answer

        if export_options.include_metrics:
            payload["metrics"] = self.metrics.model_dump()

        if export_options.include_sources:
            payload["top_sources"] = [
                source.model_dump()
                for source in _limit_items(self.top_sources, export_options.max_sources)
            ]

        if export_options.include_claims:
            payload["top_claims"] = [
                claim.model_dump()
                for claim in _limit_items(self.top_claims, export_options.max_claims)
            ]

        return payload

    @classmethod
    def from_raw(cls, raw: dict) -> ResearchResult:
        """Build a :class:`ResearchResult` from the raw ``graph.run()`` dict.

        This is the bridge between the internal state-dict world and the
        typed public API. Callers other than ``ResearchAgent`` rarely
        need this directly — it is exposed for parity tooling and tests
        that consume the raw graph output.

        Args:
            raw: Result dict produced by :func:`inqtrix.graph.run`. Must
                contain at least ``result_state`` and ``answer`` keys;
                other fields are tolerated as missing and filled with
                neutral defaults. ``elapsed_seconds`` is always set to
                ``0.0`` here and must be filled by the caller after
                measuring the wall-clock duration.

        Returns:
            A populated :class:`ResearchResult`. ``top_sources`` is
            capped at 60 entries; ``top_claims`` at 30. Sources are
            tier-classified via the default
            :class:`~inqtrix.strategies.DefaultSourceTiering` — pass a
            custom strategy via the agent if you need different tiers
            in the result view.
        """
        result_state: dict = raw.get("result_state", {})
        usage: dict = raw.get("usage", {})

        # -- sources --
        from inqtrix.strategies import DefaultSourceTiering
        tiering = DefaultSourceTiering()
        all_urls: list[str] = result_state.get("all_citations", [])
        top_sources = [
            Source(url=u, tier=tiering.tier_for_url(u))
            for u in all_urls[:60]
        ]

        # -- claims --
        consolidated: list[dict] = result_state.get("consolidated_claims", [])
        top_claims = [
            Claim.from_consolidated(c)
            for c in consolidated[:30]
        ]

        # -- metrics --
        tier_counts = result_state.get("source_tier_counts", {})
        claim_counts = result_state.get("claim_status_counts", {})
        metrics = ResearchMetrics(
            rounds=result_state.get("round", 0),
            elapsed_seconds=0.0,  # filled by caller
            total_queries=len(result_state.get("queries", [])),
            total_citations=len(all_urls),
            confidence=result_state.get("final_confidence", 0),
            aspect_coverage=result_state.get("aspect_coverage", 0.0),
            evidence_consistency=result_state.get("evidence_consistency", 0),
            evidence_sufficiency=result_state.get("evidence_sufficiency", 0),
            sources=SourceMetrics(
                tier_counts=tier_counts,
                quality_score=result_state.get("source_quality_score", 0.0),
            ),
            claims=ClaimMetrics(
                status_counts=claim_counts,
                quality_score=result_state.get("claim_quality_score", 0.0),
            ),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

        return cls(
            answer=raw.get("answer", ""),
            metrics=metrics,
            top_sources=top_sources,
            top_claims=top_claims,
        )
