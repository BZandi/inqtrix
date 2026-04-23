"""Azure OpenAI adapter for the LLMProvider interface.

This module calls the Azure OpenAI v1 Chat Completions API via the
official ``openai`` SDK's generic ``OpenAI`` client.  The SDK handles
retry with exponential backoff internally, so no custom retry loop is
needed (unlike the direct Anthropic / Bedrock adapters).

Three authentication modes are supported:

* **API key** — simplest path; pass ``api_key`` to the constructor.
* **Entra ID (Service Principal)** — pass ``tenant_id``, ``client_id``
  and ``client_secret`` directly to the constructor; the provider
  builds a ``ClientSecretCredential`` and a token provider internally.
  Alternatively pass a pre-built ``credential`` object.
* **Custom token provider** — pass an ``azure_ad_token_provider``
  callable obtained from ``azure.identity.get_bearer_token_provider``.
  Use this for Managed Identity, AzureCliCredential, or any other
  credential type. Requires the ``azure-identity`` package::

      uv sync

Enterprise environments that route traffic through an HTTP proxy can
set ``proxy_url`` — the provider will create an ``httpx`` client with
that proxy and inject it into the client constructor.

Key differences from the ``LiteLLM`` provider:

* Uses the Azure OpenAI **v1** endpoint format with
    ``OpenAI(base_url="https://.../openai/v1/")``.  This avoids the old
    date-based ``api_version`` churn and follows Microsoft's current
    guidance for new integrations.

* The ``model`` parameter in ``chat.completions.create()`` is the
  **deployment name**, not the model name.  This is an Azure-specific
  convention documented in the example script.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Callable, Mapping, Literal

from openai import DefaultHttpxClient, OpenAI, OpenAIError, RateLimitError, APIStatusError

from inqtrix.constants import REASONING_TIMEOUT, SUMMARIZE_TIMEOUT
from inqtrix.exceptions import AgentRateLimited, AgentTimeout, AzureOpenAIAPIError
from inqtrix.prompts import SUMMARIZE_PROMPT
from inqtrix.providers._azure_common import (
    AZURE_OPENAI_DEFAULT_SCOPE,
    build_azure_openai_token_provider,
    extract_azure_api_error_details,
    normalize_openai_v1_base_url,
)
from inqtrix.providers.base import (
    LLMProvider,
    LLMResponse,
    SummarizeOptions,
    _NonFatalNoticeMixin,
    _bounded_timeout,
    _check_deadline,
    _normalize_completion_response,
    _SDK_MAX_RETRIES,
    prepare_summarize_call,
)
from inqtrix.settings import ModelSettings
from inqtrix.state import track_tokens

log = logging.getLogger("inqtrix")


class AzureOpenAILLM(_NonFatalNoticeMixin, LLMProvider):
    """Call the Azure OpenAI v1 Chat Completions API via the official SDK.

    Use this provider when your reasoning models are deployed on Azure
    OpenAI and you want the current v1 endpoint shape instead of the
    legacy date-based API-version flow. It is a good fit for Azure-native
    deployments that still want the same OpenAI SDK ergonomics as
    LiteLLM-backed providers.

    Attributes:
        _default_model (str): Primary deployment name for reasoning calls.
        _summarize_model (str): Deployment used for summarize-model helper
            calls.
        _default_max_tokens (int): Output-token budget for reasoning
            requests.
        _summarize_max_tokens (int): Output-token budget for helper
            summarization requests.
        _temperature (float | None): Optional sampling temperature.
        _token_budget_parameter (Literal["max_completion_tokens", "max_tokens"]):
            Request field used for output-token budgeting.
        _request_params (dict[str, Any]): Extra request parameters merged
            into reasoning calls.
        _summarize_request_params (dict[str, Any]): Extra request
            parameters merged into summarize-model calls.
        _models (ModelSettings): Effective role-to-model mapping exposed
            to the runtime.
        _client (OpenAI): Shared SDK client for Azure OpenAI requests.
    """

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_ad_token_provider: Callable[[], str] | None = None,
        credential: Any | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_scope: str = AZURE_OPENAI_DEFAULT_SCOPE,
        default_model: str = "gpt-4o",
        classify_model: str = "",
        summarize_model: str = "",
        evaluate_model: str = "",
        default_max_tokens: int = 1024,
        summarize_max_tokens: int = 512,
        temperature: float | None = None,
        token_budget_parameter: Literal["max_completion_tokens",
                                        "max_tokens"] = "max_completion_tokens",
        proxy_url: str | None = None,
        timeout: float = 60.0,
        default_headers: Mapping[str, str] | None = None,
        request_params: Mapping[str, Any] | None = None,
        summarize_request_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the Azure OpenAI provider.

        Use the constructor when the reasoning path should run against an
        Azure OpenAI deployment name rather than a raw model name. Exactly
        one endpoint input and exactly one authentication path must be
        supplied. Role-specific deployments and extra request parameters
        let you optimize cost or compatibility without changing graph code.

        Args:
            azure_endpoint: Azure OpenAI resource endpoint such as
                ``"https://my-resource.openai.azure.com/"``. When this is
                provided, ``base_url`` must be omitted.
            base_url: Full Azure OpenAI v1 base URL such as
                ``"https://my-resource.openai.azure.com/openai/v1/"``.
                When this is provided, ``azure_endpoint`` must be omitted.
            api_key: Azure OpenAI API key. Do not use this together with
                ``azure_ad_token_provider`` or any of the credential
                arguments.
            azure_ad_token_provider: Bearer-token provider from
                ``azure.identity.get_bearer_token_provider(...)``. Use
                this for Managed Identity, AzureCliCredential, or any
                custom credential type. Mutually exclusive with
                ``api_key``, ``credential`` and the Service Principal
                fields.
            credential: Optional pre-built ``azure.identity`` credential
                object (any ``TokenCredential``-like instance). When
                supplied the provider builds an internal bearer-token
                provider from it. Mutually exclusive with ``api_key``,
                ``azure_ad_token_provider`` and the Service Principal
                fields.
            tenant_id: Entra tenant ID for automatic Service Principal
                auth. Must be supplied together with ``client_id`` and
                ``client_secret``. Mutually exclusive with ``api_key``,
                ``azure_ad_token_provider`` and ``credential``.
            client_id: Entra client ID for automatic Service Principal
                auth. See ``tenant_id``.
            client_secret: Entra client secret for automatic Service
                Principal auth. See ``tenant_id``.
            token_scope: OAuth scope used when building an internal
                token provider from ``credential`` or the Service
                Principal fields. Defaults to the Azure OpenAI scope
                (``https://cognitiveservices.azure.com/.default``).
                Override only when targeting a non-OpenAI Azure surface.
            default_model: Primary deployment name for classify, plan,
                evaluate fallback, and answer calls. This must be the Azure
                deployment name, not the base model name.
            classify_model: Optional deployment override for question
                classification. When omitted, classification falls back to
                ``default_model``.
            summarize_model: Optional cheaper deployment for parallel
                summarization and claim extraction. When omitted, helper
                calls use ``default_model``.
            evaluate_model: Optional deployment override for evidence
                evaluation. When omitted, evaluation falls back to
                ``default_model``.
            default_max_tokens: Output-token budget for reasoning calls.
                The default is ``1024``.
            summarize_max_tokens: Output-token budget for summarize-model
                helper calls. The default is ``512``.
            temperature: Optional sampling temperature. The default is
                ``None``.
            token_budget_parameter: Which request field to use for output
                budgets. Keep the default ``"max_completion_tokens"`` for
                newer deployments and use ``"max_tokens"`` only when a
                specific deployment still requires the older field.
            proxy_url: Optional HTTPS proxy URL. When omitted, the default
                HTTP transport is used.
            timeout: Default client-level timeout in seconds. The default
                is ``60.0``.
            default_headers: Optional headers forwarded on every request,
                for example preview feature headers.
            request_params: Optional extra parameters merged into reasoning
                calls after reserved SDK keys are filtered out.
            summarize_request_params: Optional extra parameters merged into
                summarize-model helper calls after reserved SDK keys are
                filtered out.

        Raises:
            ValueError: If neither or both of ``azure_endpoint`` and
                ``base_url`` are provided, if more than one of the auth
                modes (``api_key``, ``azure_ad_token_provider``,
                ``credential``, Service Principal fields) is supplied,
                if none is supplied, if the Service Principal fields
                are partially supplied, or if ``token_budget_parameter``
                is invalid.

        Example:
            >>> from inqtrix import AzureOpenAILLM
            >>> llm = AzureOpenAILLM(
            ...     azure_endpoint="https://example.openai.azure.com/",
            ...     api_key="test-key",
            ...     default_model="my-gpt4o-deployment",
            ... )
            >>> llm.models.reasoning_model
            'my-gpt4o-deployment'
        """
        if bool(azure_endpoint) == bool(base_url):
            raise ValueError(
                "Provide exactly one of azure_endpoint or base_url."
            )

        sp_fields = (tenant_id, client_id, client_secret)
        sp_any = any(sp_fields)
        sp_all = all(sp_fields)
        if sp_any and not sp_all:
            raise ValueError(
                "tenant_id, client_id and client_secret must all be "
                "provided together for Service Principal auth."
            )

        auth_modes_chosen = sum(
            1
            for present in (
                bool(api_key),
                bool(azure_ad_token_provider),
                bool(credential),
                sp_all,
            )
            if present
        )
        if auth_modes_chosen > 1:
            raise ValueError(
                "api_key, azure_ad_token_provider, credential and the "
                "Service Principal fields (tenant_id/client_id/client_secret) "
                "are mutually exclusive — pass exactly one auth mode."
            )
        if auth_modes_chosen == 0:
            raise ValueError(
                "An auth mode must be provided: api_key, azure_ad_token_provider, "
                "credential, or tenant_id+client_id+client_secret."
            )

        if azure_ad_token_provider is None and (credential is not None or sp_all):
            azure_ad_token_provider = build_azure_openai_token_provider(
                credential=credential,
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
                scope=token_scope,
            )

        if token_budget_parameter not in {"max_completion_tokens", "max_tokens"}:
            raise ValueError(
                "token_budget_parameter must be 'max_completion_tokens' or 'max_tokens'."
            )

        self._default_model = default_model
        self._summarize_model = summarize_model or default_model
        self._default_max_tokens = default_max_tokens
        self._summarize_max_tokens = summarize_max_tokens
        self._temperature = temperature
        self._token_budget_parameter = token_budget_parameter
        self._request_params = dict(request_params or {})
        self._summarize_request_params = dict(summarize_request_params or {})
        self._models = ModelSettings(
            reasoning_model=default_model,
            search_model="",
            classify_model=classify_model,
            summarize_model=summarize_model,
            evaluate_model=evaluate_model,
        )

        # Build optional httpx client for proxy support.
        http_client = None
        if proxy_url:
            import httpx

            http_client = DefaultHttpxClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(timeout, connect=10.0),
            )

        auth_value: str | Callable[[], str] = azure_ad_token_provider or api_key or ""

        client_kwargs: dict[str, Any] = {
            "base_url": self._normalize_base_url(base_url or azure_endpoint or ""),
            "api_key": auth_value,
            "timeout": timeout,
            "max_retries": _SDK_MAX_RETRIES,
        }
        if default_headers:
            client_kwargs["default_headers"] = dict(default_headers)
        if http_client is not None:
            client_kwargs["http_client"] = http_client

        self._client = OpenAI(**client_kwargs)

    _normalize_base_url = staticmethod(normalize_openai_v1_base_url)
    _extract_api_error_details = staticmethod(extract_azure_api_error_details)

    def _merge_request_params(
        self,
        base_kwargs: dict[str, Any],
        extra_params: Mapping[str, Any],
    ) -> dict[str, Any]:
        reserved = {
            "model",
            "messages",
            "timeout",
            "stream",
            "max_tokens",
            "max_completion_tokens",
        }
        merged = dict(base_kwargs)
        for key, value in extra_params.items():
            if key in reserved:
                continue
            merged[key] = value
        return merged

    # -- model metadata ----------------------------------------------------

    @property
    def models(self) -> ModelSettings:
        """Return the effective role-to-model mapping for the runtime.

        Returns:
            ModelSettings: Resolved deployment names used by graph nodes.
        """
        return self._models

    # -- thinking compatibility --------------------------------------------
    # Azure OpenAI (GPT models) does not have Anthropic-style extended
    # thinking.  Provide a no-op context manager so graph nodes that call
    # ``provider.without_thinking()`` work without special-casing.

    def without_thinking(self):
        """Return a no-op context manager for graph compatibility.

        Azure OpenAI GPT deployments do not support Anthropic-style
        extended thinking. This method exists so graph code can call
        ``provider.without_thinking()`` uniformly across providers without
        branching on provider type.

        Returns:
            ContextManager[AzureOpenAILLM]: A ``nullcontext`` wrapping this
            provider.
        """
        return nullcontext(self)

    # -- LLMProvider interface ---------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        """Generate text through Azure OpenAI and discard token metadata.

        Args:
            prompt: User-facing input text.
            system: Optional system instruction.
            model: Optional deployment override. When omitted, the
                provider uses ``self._default_model``.
            max_output_tokens: Output-token budget for this call. When
                ``None`` (default), falls back to
                ``self._default_max_tokens`` (constructor argument
                ``default_max_tokens``). The value is forwarded under
                the configured ``token_budget_parameter`` key.
            timeout: Per-call timeout budget in seconds.
            state: Optional mutable agent state for token tracking.
            deadline: Optional absolute monotonic deadline for the full
                run.

        Returns:
            str: Visible assistant text for the completion.

        Raises:
            AgentTimeout: If the full run deadline has elapsed.
            AgentRateLimited: If Azure returns a fatal rate-limit error.
            AzureOpenAIAPIError: If the SDK reports a non-rate-limit
                backend failure.
        """
        return self.complete_with_metadata(
            prompt,
            system=system,
            model=model,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            state=state,
            deadline=deadline,
        ).content

    def complete_with_metadata(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
        """Generate text and metadata through Azure OpenAI.

        Use this method for reasoning calls when the runtime wants token
        accounting in addition to visible content. The method clamps the
        timeout against the remaining deadline, injects the configured
        token-budget field, merges optional request parameters, and lets
        the OpenAI SDK handle retry behavior.

        Args:
            prompt: User-facing input text.
            system: Optional system instruction. The default is ``None``.
            model: Optional deployment override. When omitted, the
                provider uses ``self._default_model``.
            max_output_tokens: Output-token budget for this call. When
                ``None`` (default), falls back to
                ``self._default_max_tokens``. Forwarded under the
                configured ``token_budget_parameter`` key, so newer
                Azure deployments receive ``max_completion_tokens``
                while legacy deployments receive ``max_tokens``.
            timeout: Per-call timeout budget in seconds before deadline
                clamping. The default is ``REASONING_TIMEOUT``.
            state: Optional mutable agent state that receives token counts
                through ``track_tokens()`` when provided.
            deadline: Optional absolute monotonic deadline for the full
                run.

        Returns:
            LLMResponse: Structured response containing visible content,
            token counts, and the effective deployment label.

        Raises:
            AgentTimeout: If the full run deadline has already elapsed.
            AgentRateLimited: If Azure returns HTTP 429 or the SDK raises
                ``RateLimitError``.
            AzureOpenAIAPIError: If Azure responds with a non-rate-limit
                API error or another SDK-level failure occurs.
        """
        if deadline is not None:
            _check_deadline(deadline)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        use_model = model or self._default_model

        create_kwargs: dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "timeout": _bounded_timeout(timeout, deadline),
            "stream": False,
        }
        create_kwargs[self._token_budget_parameter] = max_output_tokens or self._default_max_tokens
        if self._temperature is not None:
            create_kwargs["temperature"] = self._temperature
        create_kwargs = self._merge_request_params(create_kwargs, self._request_params)

        try:
            r = self._client.chat.completions.create(**create_kwargs)
            normalized = _normalize_completion_response(r)
            if state is not None:
                track_tokens(state, normalized)
            return LLMResponse(
                content=normalized.content,
                prompt_tokens=normalized.prompt_tokens,
                completion_tokens=normalized.completion_tokens,
                model=use_model,
                finish_reason=normalized.finish_reason,
                raw=normalized.raw,
                request_max_tokens=int(
                    create_kwargs.get(self._token_budget_parameter) or 0
                ),
            )
        except RateLimitError as e:
            log.error("FATAL Rate-Limit (%s): %s", use_model, e)
            raise AgentRateLimited(use_model, e)
        except APIStatusError as e:
            if e.status_code == 429:
                log.error("FATAL Rate-Limit (%s): %s", use_model, e)
                raise AgentRateLimited(use_model, e)
            details = self._extract_api_error_details(e)
            log.error(
                "Azure-OpenAI-Aufruf fehlgeschlagen (%s, status=%s, code=%s, request-id=%s): %s",
                use_model,
                details.get("status_code"),
                details.get("error_code") or "-",
                details.get("request_id") or "-",
                e,
            )
            raise AzureOpenAIAPIError(
                model=use_model,
                status_code=details.get("status_code") if isinstance(
                    details.get("status_code"), int) else None,
                error_code=str(details.get("error_code") or "").strip(),
                request_id=str(details.get("request_id") or "").strip() or None,
                message=str(details.get("message") or "").strip() or str(e),
                original=e,
            ) from e
        except OpenAIError as e:
            log.error("Azure-OpenAI-Aufruf fehlgeschlagen (%s): %s", use_model, e)
            raise AzureOpenAIAPIError(
                model=use_model,
                message=str(e),
                original=e,
            ) from e

    def summarize_parallel(
        self,
        text: str,
        deadline: float | None = None,
        options: SummarizeOptions | None = None,
    ) -> tuple[str, int, int]:
        """Summarize search text in a thread-safe helper path.

        Helper summarization uses the configured summarize deployment and
        the same token-budget field as reasoning calls. On non-rate-limit
        failure the method degrades locally to truncated raw text and
        stores a nonfatal notice for the search node.

        Args:
            text: Raw search-result text to condense. Blank input returns
                an empty summary payload immediately.
            deadline: Optional absolute monotonic deadline for the full
                run. Used to clamp the per-call timeout via
                ``_bounded_timeout``.
            options: Optional :class:`SummarizeOptions` carrying tuning
                hints (custom prompt template, input character limit,
                fallback character limit, output-token budget). When
                ``None`` (default), the provider uses the built-in
                ``SUMMARIZE_PROMPT`` and the constructor-level token
                budget. This is the standard way for the report-profile
                tuning to flow into per-call summarize requests.

        Returns:
            tuple[str, int, int]: ``(facts_text, prompt_tokens,
            completion_tokens)``. On fallback, the text becomes the first
            ``options.fallback_char_limit`` characters of the raw input
            (default 800) and both token counts are ``0``.

        Raises:
            AgentTimeout: If the global deadline has already elapsed.
            AgentRateLimited: If Azure returns a fatal helper rate limit.
        """
        summarize_model = self._models.effective_summarize_model
        preamble = prepare_summarize_call(
            text,
            options,
            default_prompt=SUMMARIZE_PROMPT,
            default_max_output_tokens=self._summarize_max_tokens,
            deadline=deadline,
            notice_mixin=self,
        )
        if not preamble.ready:
            return ("", 0, 0)

        fallback_char_limit = preamble.fallback_char_limit
        summarize_max_tokens = preamble.max_output_tokens
        prompt = f"{preamble.prompt_template}{preamble.truncated_text}"

        try:
            create_kwargs: dict[str, Any] = {
                "model": summarize_model,
                "messages": [{"role": "user", "content": prompt}],
                "timeout": _bounded_timeout(SUMMARIZE_TIMEOUT, deadline),
                "stream": False,
            }
            create_kwargs[self._token_budget_parameter] = summarize_max_tokens
            create_kwargs = self._merge_request_params(
                create_kwargs,
                self._summarize_request_params,
            )
            r = self._client.chat.completions.create(**create_kwargs)
            normalized = _normalize_completion_response(r)
            return (
                normalized.content,
                normalized.prompt_tokens,
                normalized.completion_tokens,
            )
        except RateLimitError as e:
            raise AgentRateLimited(summarize_model, e)
        except APIStatusError as e:
            if e.status_code == 429:
                raise AgentRateLimited(summarize_model, e)
            self._set_nonfatal_notice(
                f"Azure-OpenAI-Zusammenfassung via {summarize_model} fehlgeschlagen; Fallback auf Rohtext."
            )
            log.warning(
                "Azure-OpenAI-Zusammenfassung fehlgeschlagen (%s): %s",
                summarize_model,
                e,
            )
            return (text[:fallback_char_limit], 0, 0)
        except (OpenAIError, AgentTimeout):
            self._set_nonfatal_notice(
                f"Azure-OpenAI-Zusammenfassung via {summarize_model} fehlgeschlagen; Fallback auf Rohtext."
            )
            log.warning(
                "Azure-OpenAI-Zusammenfassung fehlgeschlagen (%s); Fallback auf Rohtext.",
                summarize_model,
            )
            return (text[:fallback_char_limit], 0, 0)

    def is_available(self) -> bool:
        """Report whether the provider is configured to attempt requests.

        Configuration here means: an OpenAI SDK client was successfully
        constructed (endpoint resolved, exactly one auth mode supplied,
        token budget parameter validated). This does not guarantee that
        the Azure deployment is reachable or that credentials are valid
        — those failures only surface on the first ``complete()`` call.

        Returns:
            ``True`` when the internal SDK client was constructed,
            otherwise ``False``. In practice ``False`` is unreachable
            today because constructor argument validation raises
            ``ValueError`` on bad input rather than returning a half-
            initialised provider; the method is kept for forward
            compatibility and ABC contract compliance.
        """
        return self._client is not None
