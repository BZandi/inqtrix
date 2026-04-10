"""Azure OpenAI adapter for the LLMProvider interface.

This module calls the Azure OpenAI v1 Chat Completions API via the
official ``openai`` SDK's generic ``OpenAI`` client.  The SDK handles
retry with exponential backoff internally, so no custom retry loop is
needed (unlike the direct Anthropic / Bedrock adapters).

Two authentication modes are supported:

* **API key** — simplest path; pass ``api_key`` to the constructor.
* **Entra ID (Service Principal / Managed Identity)** — pass an
  ``azure_ad_token_provider`` callable obtained from
  ``azure.identity.get_bearer_token_provider``.  Requires the
  ``azure-identity`` package::

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
from inqtrix.providers import (
    LLMProvider,
    LLMResponse,
    _NonFatalNoticeMixin,
    _bounded_timeout,
    _check_deadline,
    _normalize_completion_response,
    _SDK_MAX_RETRIES,
)
from inqtrix.settings import ModelSettings
from inqtrix.state import track_tokens

log = logging.getLogger("inqtrix")


class AzureOpenAILLM(_NonFatalNoticeMixin, LLMProvider):
    """Call the Azure OpenAI v1 Chat Completions API via the official SDK.

    Parameters
    ----------
    azure_endpoint:
        Azure OpenAI resource endpoint URL, e.g.
        ``https://mein-openai.openai.azure.com/``.  The provider
        appends ``/openai/v1/`` automatically.
    base_url:
        Full Azure OpenAI v1 base URL, e.g.
        ``https://mein-openai.openai.azure.com/openai/v1/``.  Provide
        exactly one of *azure_endpoint* or *base_url*.
    api_key:
        Azure OpenAI API key.  **Mutually exclusive** with
        *azure_ad_token_provider*.
    azure_ad_token_provider:
        Callable that returns a bearer token string, obtained from
        ``azure.identity.get_bearer_token_provider(credential, scope)``.
        This is passed to the OpenAI client as the ``api_key`` value,
        which is the recommended Azure OpenAI v1 pattern.
    default_model:
        Deployment name for reasoning calls (classify, plan, evaluate,
        answer).  This is the **deployment name** you created in the
        Azure Portal — not the underlying model name.
    classify_model:
        Optional deployment override for question classification.
        Falls back to *default_model*.
    summarize_model:
        Cheaper deployment for search-result summarisation and claim
        extraction (called in parallel threads).
    evaluate_model:
        Optional deployment override for evidence evaluation.
        Falls back to *default_model*.
    default_max_tokens:
        Output-token budget for reasoning calls.
    summarize_max_tokens:
        Output-token budget for summarisation calls.
    temperature:
        Sampling temperature (0.0–2.0).
    token_budget_parameter:
        Which request field to use for output-token budgeting.
        ``"max_completion_tokens"`` is the current default because it
        matches newer Azure/OpenAI semantics and is compatible with
        reasoning-capable models.  Set ``"max_tokens"`` only if a
        deployment explicitly requires the older field.
    proxy_url:
        Optional HTTPS proxy URL.  When set, creates an ``httpx``
        client with that proxy and injects it into the
        client constructor.
    timeout:
        Default client-level timeout in seconds.
    default_headers:
        Optional headers forwarded on every request.  Useful for
        preview feature headers when Azure introduces them.
    request_params:
        Optional extra parameters merged into reasoning calls.
    summarize_request_params:
        Optional extra parameters merged into summarisation calls.
    """

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_ad_token_provider: Callable[[], str] | None = None,
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
        if bool(azure_endpoint) == bool(base_url):
            raise ValueError(
                "Provide exactly one of azure_endpoint or base_url."
            )
        if api_key and azure_ad_token_provider:
            raise ValueError(
                "api_key and azure_ad_token_provider are mutually exclusive — "
                "pass one or the other, not both."
            )
        if not api_key and not azure_ad_token_provider:
            raise ValueError(
                "Either api_key or azure_ad_token_provider must be provided."
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

    @staticmethod
    def _normalize_base_url(endpoint_or_base_url: str) -> str:
        stripped = endpoint_or_base_url.strip().rstrip("/")
        if stripped.endswith("/api"):
            raise ValueError(
                "The provided endpoint looks like an Azure AI Project endpoint (.../api). "
                "For Azure OpenAI use the resource endpoint or the full /openai/v1/ base URL."
            )
        if stripped.endswith("/openai/v1"):
            return f"{stripped}/"
        return f"{stripped}/openai/v1/"

    @staticmethod
    def _extract_api_error_details(exc: APIStatusError) -> dict[str, Any]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        request_id = None
        for header_name in ("apim-request-id", "x-request-id", "request-id"):
            header_value = headers.get(header_name)
            if isinstance(header_value, str) and header_value.strip():
                request_id = header_value.strip()
                break

        error_code = ""
        message = ""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                raw_code = error.get("code")
                raw_message = error.get("message")
                if isinstance(raw_code, str):
                    error_code = raw_code.strip()
                if isinstance(raw_message, str):
                    message = raw_message.strip()
            if not error_code:
                top_code = body.get("code")
                if isinstance(top_code, str):
                    error_code = top_code.strip()
            if not message:
                top_message = body.get("message")
                if isinstance(top_message, str):
                    message = top_message.strip()

        return {
            "status_code": getattr(exc, "status_code", None),
            "request_id": request_id,
            "error_code": error_code,
            "message": message,
        }

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
        return self._models

    # -- thinking compatibility --------------------------------------------
    # Azure OpenAI (GPT models) does not have Anthropic-style extended
    # thinking.  Provide a no-op context manager so graph nodes that call
    # ``provider.without_thinking()`` work without special-casing.

    def without_thinking(self):
        return nullcontext(self)

    # -- LLMProvider interface ---------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> str:
        return self.complete_with_metadata(
            prompt,
            system=system,
            model=model,
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
        timeout: float = REASONING_TIMEOUT,
        state: dict | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
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
        create_kwargs[self._token_budget_parameter] = self._default_max_tokens
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
    ) -> tuple[str, int, int]:
        """Thread-safe fact extraction from a search result.

        Falls back to raw text on failure and sets a nonfatal notice.
        """
        if not text.strip():
            return ("", 0, 0)
        self._clear_nonfatal_notice()
        if deadline is not None:
            _check_deadline(deadline)

        summarize_model = self._models.effective_summarize_model
        prompt = f"{SUMMARIZE_PROMPT}{text[:6000]}"

        try:
            create_kwargs: dict[str, Any] = {
                "model": summarize_model,
                "messages": [{"role": "user", "content": prompt}],
                "timeout": _bounded_timeout(SUMMARIZE_TIMEOUT, deadline),
                "stream": False,
            }
            create_kwargs[self._token_budget_parameter] = self._summarize_max_tokens
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
            return (text[:800], 0, 0)
        except AgentRateLimited:
            raise
        except (OpenAIError, AgentTimeout):
            self._set_nonfatal_notice(
                f"Azure-OpenAI-Zusammenfassung via {summarize_model} fehlgeschlagen; Fallback auf Rohtext."
            )
            log.warning(
                "Azure-OpenAI-Zusammenfassung fehlgeschlagen (%s); Fallback auf Rohtext.",
                summarize_model,
            )
            return (text[:800], 0, 0)

    def is_available(self) -> bool:
        return self._client is not None
