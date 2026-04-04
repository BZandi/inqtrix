"""Structured result types returned by :class:`~inqtrix.agent.ResearchAgent`.

All models are Pydantic v2 and fully serialisable to JSON.
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
    """A single cited source."""

    url: str
    tier: str = "unknown"
    """Quality tier: ``primary``, ``mainstream``, ``stakeholder``,
    ``unknown``, or ``low``."""


class Claim(BaseModel):
    """A consolidated claim extracted during research.

    This is the public, typed view of the internal consolidated claim ledger.
    It preserves the verification metadata needed for downstream inspection
    without leaking the full mutable agent state.
    """

    text: str
    status: str = "unverified"
    """One of ``verified``, ``contested``, or ``unverified``."""
    claim_type: str = "fact"
    """One of ``fact``, ``actor_claim``, or ``forecast``."""
    needs_primary: bool = False
    status_reason: str = ""
    support_count: int = 0
    contradict_count: int = 0
    source_tier_counts: dict[str, int] = Field(default_factory=_empty_tier_counts)
    sources: list[str] = Field(default_factory=list)

    @classmethod
    def from_consolidated(cls, claim_data: dict) -> Claim:
        """Build a public claim model from one consolidated internal claim."""
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
    """Quality breakdown of sources used."""

    tier_counts: dict[str, int] = Field(default_factory=_empty_tier_counts)
    quality_score: float = 0.0
    """Weighted average of source tiers (0.0 = all low, 1.0 = all primary)."""


class ClaimMetrics(BaseModel):
    """Quality breakdown of extracted claims."""

    status_counts: dict[str, int] = Field(
        default_factory=lambda: {"verified": 0, "contested": 0, "unverified": 0}
    )
    quality_score: float = 0.0
    """``(verified + 0.5 * contested) / total``."""


class ResearchMetrics(BaseModel):
    """Aggregate metrics for a research run."""

    rounds: int = 0
    elapsed_seconds: float = 0.0
    total_queries: int = 0
    total_citations: int = 0
    confidence: int = 0
    aspect_coverage: float = 0.0
    evidence_consistency: int = 0
    evidence_sufficiency: int = 0
    sources: SourceMetrics = Field(default_factory=SourceMetrics)
    claims: ClaimMetrics = Field(default_factory=ClaimMetrics)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ResearchResultExportOptions(BaseModel):
    """Optional projection settings for serialized result payloads.

    All fields are optional switches or limits so downstream surfaces like
    parity, HTTP responses, or custom tooling can select the public result
    view they need without introducing parallel result models.
    """

    include_answer: bool = True
    include_metrics: bool = True
    include_sources: bool = True
    include_claims: bool = True
    max_sources: int | None = None
    max_claims: int | None = None


class ResearchResult(BaseModel):
    """Complete result of a :meth:`ResearchAgent.research` call.

    Attributes
    ----------
    answer:
        The final markdown-formatted answer text.
    metrics:
        Aggregated quality and performance metrics.
    top_sources:
        Most relevant sources used in the answer, ordered by quality tier.
    top_claims:
        Key claims extracted during research, ordered by verification status.
    """

    answer: str
    metrics: ResearchMetrics = Field(default_factory=ResearchMetrics)
    top_sources: list[Source] = Field(default_factory=list)
    top_claims: list[Claim] = Field(default_factory=list)

    def to_export_payload(
        self,
        options: ResearchResultExportOptions | None = None,
    ) -> dict[str, Any]:
        """Build a configurable public payload from the typed result model."""
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
        typed public API.
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
