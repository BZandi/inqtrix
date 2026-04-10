"""Inqtrix - Iterative AI Research Agent.

Quick start::

    from inqtrix import ResearchAgent

    agent = ResearchAgent()
    result = agent.research("Was ist der aktuelle Stand der GKV-Reform?")
    print(result.answer)

Baukasten (building-block) pattern::

    from inqtrix import ResearchAgent, AgentConfig, LiteLLM, PerplexitySearch

    llm = LiteLLM(api_key="...", default_model="gpt-4o")
    search = PerplexitySearch(api_key="...", model="sonar-pro")
    agent = ResearchAgent(AgentConfig(llm=llm, search=search))
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

# -- Public API (providers) --
from inqtrix.providers import (
    LLMProvider,
    SearchProvider,
    ProviderContext,
    LiteLLM,
    PerplexitySearch,
)
from inqtrix.providers_anthropic import AnthropicLLM
from inqtrix.providers_azure import AzureOpenAILLM
from inqtrix.providers_brave import BraveSearch
from inqtrix.providers_azure_bing import AzureFoundryBingSearch
from inqtrix.providers_bedrock import BedrockLLM
from inqtrix.exceptions import AzureOpenAIAPIError, AzureFoundryBingAPIError, BedrockAPIError
from inqtrix.logging_config import configure_logging

# -- Public API (strategies) --
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
    # Providers
    "LLMProvider",
    "SearchProvider",
    "ProviderContext",
    "LiteLLM",
    "PerplexitySearch",
    "AnthropicLLM",
    "AzureOpenAILLM",
    "AzureOpenAIAPIError",
    "AzureFoundryBingSearch",
    "AzureFoundryBingAPIError",
    "BraveSearch",
    "BedrockLLM",
    "BedrockAPIError",
    # Logging
    "configure_logging",
    # Strategies
    "SourceTieringStrategy",
    "ClaimExtractionStrategy",
    "ClaimConsolidationStrategy",
    "ContextPruningStrategy",
    "RiskScoringStrategy",
    "StopCriteriaStrategy",
    "StrategyContext",
]
