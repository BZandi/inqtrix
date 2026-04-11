"""Pluggable strategy ABCs and their default implementations.

Each strategy encapsulates a single algorithmic concern that nodes can
depend on through the ``StrategyContext`` dataclass.  Default
implementations reproduce the exact behaviour of the original monolithic
``_original_agent.py`` so existing node call-sites remain compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from inqtrix.strategies._source_tiering import DefaultSourceTiering, SourceTieringStrategy
from inqtrix.strategies._claim_extraction import ClaimExtractionStrategy, LLMClaimExtractor
from inqtrix.strategies._claim_consolidation import ClaimConsolidationStrategy, DefaultClaimConsolidator
from inqtrix.strategies._context_pruning import ContextPruningStrategy, RelevanceBasedPruning
from inqtrix.strategies._risk_scoring import KeywordRiskScorer, RiskScoringStrategy
from inqtrix.strategies._stop_criteria import MultiSignalStopCriteria, StopCriteriaStrategy

if TYPE_CHECKING:
    from inqtrix.providers.base import LLMProvider
    from inqtrix.settings import AgentSettings


@dataclass
class StrategyContext:
    """Bundle of all pluggable strategies available to runtime nodes.

    The graph injects a single :class:`StrategyContext` into every node so
    orchestration code can stay agnostic to the concrete implementation of
    source tiering, claim extraction, consolidation, pruning, risk scoring,
    and stopping heuristics.

    Attributes:
        source_tiering: URL-to-quality classification and aggregate source
            scoring.
        claim_extraction: Structured claim extraction from raw search text.
        claim_consolidation: Signature grouping, status derivation, and
            answer-facing claim selection.
        context_pruning: Context-window reduction that preserves the newest
            evidence blocks.
        risk_scoring: Deterministic risk and aspect heuristics used across
            classify, plan, and search.
        stop_criteria: Multi-signal sufficiency and stop heuristics for the
            evaluate node.
    """

    source_tiering: SourceTieringStrategy
    claim_extraction: ClaimExtractionStrategy
    claim_consolidation: ClaimConsolidationStrategy
    context_pruning: ContextPruningStrategy
    risk_scoring: RiskScoringStrategy
    stop_criteria: StopCriteriaStrategy


def create_default_strategies(
    settings: "AgentSettings",
    *,
    llm: "LLMProvider | None" = None,
    summarize_model: str = "",
    summarize_timeout: int = 60,
) -> StrategyContext:
    """Create the default strategy bundle used by :class:`ResearchAgent`.

    Args:
        settings: Agent-level thresholds and loop settings used by the stop
            criteria strategy.
        llm: Optional LLM provider used by the default claim extractor.
            Passing ``None`` keeps claim extraction available but inert.
        summarize_model: Model identifier used for claim extraction calls.
        summarize_timeout: Per-call timeout in seconds for claim extraction.

    Returns:
        A fully wired :class:`StrategyContext` with the default strategy
        implementations.

    Notes:
        A single :class:`DefaultSourceTiering` instance is shared between
        ``source_tiering`` and ``claim_consolidation`` so both layers classify
        source quality with the same domain rules.
    """
    tiering = DefaultSourceTiering()
    return StrategyContext(
        source_tiering=tiering,
        claim_extraction=LLMClaimExtractor(
            llm=llm,
            summarize_model=summarize_model,
            summarize_timeout=summarize_timeout,
        ),
        claim_consolidation=DefaultClaimConsolidator(source_tiering=tiering),
        context_pruning=RelevanceBasedPruning(),
        risk_scoring=KeywordRiskScorer(),
        stop_criteria=MultiSignalStopCriteria(settings=settings),
    )


__all__ = [
    # ABCs
    "SourceTieringStrategy",
    "ClaimExtractionStrategy",
    "ClaimConsolidationStrategy",
    "ContextPruningStrategy",
    "RiskScoringStrategy",
    "StopCriteriaStrategy",
    # Defaults
    "DefaultSourceTiering",
    "LLMClaimExtractor",
    "DefaultClaimConsolidator",
    "RelevanceBasedPruning",
    "KeywordRiskScorer",
    "MultiSignalStopCriteria",
    # Container
    "StrategyContext",
    "create_default_strategies",
]
