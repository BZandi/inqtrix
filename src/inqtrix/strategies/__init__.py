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
    """Bundle of all pluggable strategies available to nodes."""

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
    """Create a :class:`StrategyContext` with all default implementations.

    Parameters
    ----------
    settings:
        Agent-level configuration (thresholds, timeouts, etc.).
    llm:
        LLM provider used for default claim extraction. Custom providers
        can participate without exposing any private client internals.
    summarize_model:
        Model identifier used for claim extraction.
    summarize_timeout:
        Per-call timeout for the summarize model.
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
