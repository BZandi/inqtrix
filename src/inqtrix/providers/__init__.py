"""Provider package — factory for default provider setup.

This package contains the abstract provider contracts (in ``base``),
concrete LLM providers (``litellm``, ``anthropic``, ``azure``,
``bedrock``), and concrete search providers (``perplexity``,
``azure_web_search``).

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
    """Create the default provider pair from settings.

    Use this factory when the caller did not inject explicit provider
    instances into ``AgentConfig`` and the runtime should instead build
    the standard LiteLLM-plus-Perplexity stack from environment-backed
    settings. The function keeps the mapping from ``Settings`` to
    provider constructors in one place, which makes the auto-create path
    predictable and avoids duplicated wiring logic in ``agent.py`` and
    the config bridge.

    LLM and search providers own separate clients because their retry
    contracts differ: LLM calls surface retry attempts through provider
    progress, while search calls keep their endpoint-specific SDK policy.

    Args:
        settings: Fully resolved runtime settings. ``settings.server``
            supplies the shared endpoint and API key, ``settings.models``
            supplies the reasoning and search model names, and
            ``settings.agent`` supplies cache configuration for the
            search provider.

    Returns:
        ProviderContext: A provider container with a configured
        ``LiteLLM`` instance in ``llm`` and a configured
        ``PerplexitySearch`` instance in ``search``.

    Example:
        >>> from inqtrix.providers import create_providers
        >>> from inqtrix.settings import Settings
        >>> settings = Settings.model_construct(
        ...     server=Settings.model_fields["server"].annotation.model_construct(
        ...         litellm_base_url="http://localhost:4000/v1",
        ...         litellm_api_key="test-key",
        ...         perplexity_api_key="test-key",
        ...     ),
        ...     models=Settings.model_fields["models"].annotation.model_construct(
        ...         reasoning_model="gpt-4o",
        ...         classify_model="",
        ...         claim_extract_model="gpt-4o-mini",
        ...         evaluate_model="",
        ...         search_model="",
        ...     ),
        ...     agent=Settings.model_fields["agent"].annotation.model_construct(
        ...         search_cache_maxsize=256,
        ...         search_cache_ttl=3600,
        ...     ),
        ... )
        >>> providers = create_providers(settings)
        >>> providers.llm.models.reasoning_model
        'gpt-4o'
    """
    from inqtrix.providers.base import ProviderContext
    from inqtrix.providers.litellm import LiteLLM
    from inqtrix.providers.perplexity import PerplexitySearch

    llm = LiteLLM(
        api_key=settings.server.litellm_api_key,
        base_url=settings.server.litellm_base_url,
        default_model=settings.models.reasoning_model,
        classify_model=settings.models.classify_model,
        claim_extract_model=settings.models.claim_extract_model,
        evaluate_model=settings.models.evaluate_model,
        plan_model=settings.models.plan_model,
        answer_model=settings.models.answer_model,
        direct_chat_model=settings.models.direct_chat_model,
        tier_high_model=settings.models.tier_high_model,
        tier_mid_model=settings.models.tier_mid_model,
        tier_fast_model=settings.models.tier_fast_model,
        tier_high_effort=settings.models.tier_high_effort,
        tier_mid_effort=settings.models.tier_mid_effort,
        tier_fast_effort=settings.models.tier_fast_effort,
    )
    llm._client = OpenAI(
        base_url=settings.server.litellm_base_url,
        api_key=settings.server.litellm_api_key,
        max_retries=0,
    )

    search = PerplexitySearch(
        api_key=settings.server.perplexity_api_key,
        base_url=settings.server.perplexity_base_url or None,
        model=settings.models.search_model or None,
        cache_maxsize=settings.agent.search_cache_maxsize,
        cache_ttl=settings.agent.search_cache_ttl,
    )

    return ProviderContext(llm=llm, search=search)
