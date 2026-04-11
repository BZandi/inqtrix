""" Custom exceptions for the research agent."""

from __future__ import annotations


class AgentTimeout(Exception):
    """Raised when MAX_TOTAL_SECONDS is exceeded."""


class AgentRateLimited(Exception):
    """Raised on 429 rate-limit or daily token limit."""

    def __init__(self, model: str, original: Exception):
        self.model = model
        self.original = original
        super().__init__(f"Rate-Limit erreicht fuer Modell '{model}': {original}")


class _ProviderAPIError(RuntimeError):
    """Base class for provider-specific API errors.

    Subclasses provide a German header template and may customize the
    set of accepted keyword arguments.  The shared ``__init__`` builds
    ``detail_parts`` from common fields and formats the final message.
    """

    _header_template: str = "API-Aufruf fehlgeschlagen ({identifier})"

    def __init__(
        self,
        *,
        _identifier: str,
        status_code: int | None = None,
        error_code: str = "",
        error_type: str = "",
        message: str = "",
        request_id: str | None = None,
        retry_after: str | None = None,
        original: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type
        self.request_id = request_id
        self.retry_after = retry_after
        self.original = original

        detail_parts: list[str] = []
        if status_code is not None:
            detail_parts.append(f"HTTP {status_code}")
        if error_code:
            detail_parts.append(error_code)
        if error_type:
            detail_parts.append(error_type)
        if request_id:
            detail_parts.append(f"request-id={request_id}")
        if retry_after:
            detail_parts.append(f"retry-after={retry_after}")

        header = self._header_template.format(identifier=_identifier)
        if detail_parts:
            header = f"{header} [{' | '.join(detail_parts)}]"

        default_error = f"Unbekannter {self.__class__.__name__}"
        final_message = message.strip() if message else str(original or default_error)
        super().__init__(f"{header}: {final_message}")


class AnthropicAPIError(_ProviderAPIError):
    """Raised when a direct Anthropic API call fails after retries."""

    _header_template = "Anthropic-Aufruf fehlgeschlagen ({identifier})"

    def __init__(
        self,
        *,
        model: str,
        status_code: int | None = None,
        error_type: str = "",
        message: str = "",
        request_id: str | None = None,
        retry_after: str | None = None,
        original: Exception | None = None,
    ) -> None:
        self.model = model
        super().__init__(
            _identifier=model,
            status_code=status_code,
            error_type=error_type,
            message=message,
            request_id=request_id,
            retry_after=retry_after,
            original=original,
        )


class BedrockAPIError(_ProviderAPIError):
    """Raised when a direct Amazon Bedrock API call fails after retries."""

    _header_template = "Bedrock-Aufruf fehlgeschlagen ({identifier})"

    def __init__(
        self,
        *,
        model: str,
        error_code: str = "",
        status_code: int | None = None,
        message: str = "",
        request_id: str | None = None,
        original: Exception | None = None,
    ) -> None:
        self.model = model
        super().__init__(
            _identifier=model,
            status_code=status_code,
            error_code=error_code,
            message=message,
            request_id=request_id,
            original=original,
        )


class AzureOpenAIAPIError(_ProviderAPIError):
    """Raised when an Azure OpenAI API call fails."""

    _header_template = "Azure-OpenAI-Aufruf fehlgeschlagen ({identifier})"

    def __init__(
        self,
        *,
        model: str,
        status_code: int | None = None,
        error_code: str = "",
        message: str = "",
        request_id: str | None = None,
        original: Exception | None = None,
    ) -> None:
        self.model = model
        super().__init__(
            _identifier=model,
            status_code=status_code,
            error_code=error_code,
            message=message,
            request_id=request_id,
            original=original,
        )


class AzureFoundryBingAPIError(_ProviderAPIError):
    """Raised when an Azure Foundry Bing Search agent call fails."""

    _header_template = "Azure-Foundry-Bing-Aufruf fehlgeschlagen (agent={identifier})"

    def __init__(
        self,
        *,
        agent_id: str,
        status_code: int | None = None,
        error_code: str = "",
        message: str = "",
        request_id: str | None = None,
        original: Exception | None = None,
    ) -> None:
        self.agent_id = agent_id
        super().__init__(
            _identifier=agent_id,
            status_code=status_code,
            error_code=error_code,
            message=message,
            request_id=request_id,
            original=original,
        )


class AzureFoundryWebSearchAPIError(_ProviderAPIError):
    """Raised when an Azure Foundry Web Search (Responses API) call fails."""

    _header_template = "Azure-Foundry-WebSearch-Aufruf fehlgeschlagen (agent={identifier})"

    def __init__(
        self,
        *,
        agent_name: str,
        status_code: int | None = None,
        error_code: str = "",
        message: str = "",
        request_id: str | None = None,
        original: Exception | None = None,
    ) -> None:
        self.agent_name = agent_name
        super().__init__(
            _identifier=agent_name,
            status_code=status_code,
            error_code=error_code,
            message=message,
            request_id=request_id,
            original=original,
        )
