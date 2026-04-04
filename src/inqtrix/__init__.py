"""Inqtrix - Iterative AI Research Agent.

Quick start::

    from inqtrix import ResearchAgent

    agent = ResearchAgent()
    result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")
    print(result.answer)

Baukasten (building-block) pattern::

    from inqtrix import ResearchAgent, AgentConfig

    agent = ResearchAgent(AgentConfig(
        max_rounds=3,
        confidence_stop=7,
    ))
"""

__version__ = "0.1.0"

# -- Public API (high-level) --
from inqtrix.agent import AgentConfig, ResearchAgent
from inqtrix.result import (
    Claim,
    ResearchMetrics,
    ResearchResult,
    ResearchResultExportOptions,
    Source,
    SourceMetrics,
    ClaimMetrics,
)

# -- Public API (extension points) --
from inqtrix.providers import LLMProvider, SearchProvider, ProviderContext
from inqtrix.providers_anthropic import AnthropicLLM
from inqtrix.providers_brave import BraveSearch
from inqtrix.strategies import (
    ClaimConsolidationStrategy,
    ClaimExtractionStrategy,
    ContextPruningStrategy,
    RiskScoringStrategy,
    SourceTieringStrategy,
    StopCriteriaStrategy,
    StrategyContext,
)

__all__ = [
    # High-level
    "ResearchAgent",
    "AgentConfig",
    "ResearchResult",
    "ResearchResultExportOptions",
    "ResearchMetrics",
    "Source",
    "Claim",
    "SourceMetrics",
    "ClaimMetrics",
    # Providers (for custom implementations)
    "LLMProvider",
    "SearchProvider",
    "ProviderContext",
    "AnthropicLLM",
    "BraveSearch",
    # Strategies (for custom implementations)
    "SourceTieringStrategy",
    "ClaimExtractionStrategy",
    "ClaimConsolidationStrategy",
    "ContextPruningStrategy",
    "RiskScoringStrategy",
    "StopCriteriaStrategy",
    "StrategyContext",
]
