"""Provider package — factory for default provider setup.

This package contains the abstract provider contracts (in ``base``),
concrete LLM providers (``litellm``, ``anthropic``, ``azure``,
``bedrock``), and concrete search providers (``perplexity``, ``brave``,
``azure_bing``, ``azure_web_search``).

The :func:`create_providers` factory lives here because it instantiates
concrete providers from sub-modules — placing it in ``base`` would
create a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from inqtrix.providers.base import ProviderContext
    from inqtrix.settings import Settings


def create_providers(settings: "Settings") -> "ProviderContext":
    """Create both providers from *settings* (env-var / auto-create path).

    This is the single entry point for wiring up provider instances
    when no explicit providers are given.  The OpenAI client is shared
    between LLM and search providers for connection pooling.
    """
    from inqtrix.providers.base import ProviderContext, _SDK_MAX_RETRIES
    from inqtrix.providers.litellm import LiteLLM
    from inqtrix.providers.perplexity import PerplexitySearch

    client = OpenAI(
        base_url=settings.server.litellm_base_url,
        api_key=settings.server.litellm_api_key,
        max_retries=_SDK_MAX_RETRIES,
    )

    llm = LiteLLM(
        api_key=settings.server.litellm_api_key,
        base_url=settings.server.litellm_base_url,
        default_model=settings.models.reasoning_model,
        classify_model=settings.models.classify_model,
        summarize_model=settings.models.summarize_model,
        evaluate_model=settings.models.evaluate_model,
    )
    # Share the same client instance for connection pooling.
    llm._client = client

    search = PerplexitySearch(
        api_key=settings.server.litellm_api_key,
        base_url=settings.server.litellm_base_url,
        model=settings.models.search_model,
        cache_maxsize=settings.agent.search_cache_maxsize,
        cache_ttl=settings.agent.search_cache_ttl,
        _client=client,
    )

    return ProviderContext(llm=llm, search=search)
