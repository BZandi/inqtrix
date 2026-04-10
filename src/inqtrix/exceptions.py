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


class AnthropicAPIError(RuntimeError):
    """Raised when a direct Anthropic API call fails after retries."""

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
        self.status_code = status_code
        self.error_type = error_type
        self.request_id = request_id
        self.retry_after = retry_after
        self.original = original

        detail_parts: list[str] = []
        if status_code is not None:
            detail_parts.append(f"HTTP {status_code}")
        if error_type:
            detail_parts.append(error_type)
        if request_id:
            detail_parts.append(f"request-id={request_id}")
        if retry_after:
            detail_parts.append(f"retry-after={retry_after}")

        header = f"Anthropic-Aufruf fehlgeschlagen ({model})"
        if detail_parts:
            header = f"{header} [{' | '.join(detail_parts)}]"

        final_message = message.strip() if message else str(original or "Unbekannter Anthropic-Fehler")
        super().__init__(f"{header}: {final_message}")


class BedrockAPIError(RuntimeError):
    """Raised when a direct Amazon Bedrock API call fails after retries."""

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
        self.error_code = error_code
        self.status_code = status_code
        self.request_id = request_id
        self.original = original

        detail_parts: list[str] = []
        if status_code is not None:
            detail_parts.append(f"HTTP {status_code}")
        if error_code:
            detail_parts.append(error_code)
        if request_id:
            detail_parts.append(f"request-id={request_id}")

        header = f"Bedrock-Aufruf fehlgeschlagen ({model})"
        if detail_parts:
            header = f"{header} [{' | '.join(detail_parts)}]"

        final_message = message.strip() if message else str(original or "Unbekannter Bedrock-Fehler")
        super().__init__(f"{header}: {final_message}")


class AzureOpenAIAPIError(RuntimeError):
    """Raised when an Azure OpenAI API call fails."""

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
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id
        self.original = original

        detail_parts: list[str] = []
        if status_code is not None:
            detail_parts.append(f"HTTP {status_code}")
        if error_code:
            detail_parts.append(error_code)
        if request_id:
            detail_parts.append(f"request-id={request_id}")

        header = f"Azure-OpenAI-Aufruf fehlgeschlagen ({model})"
        if detail_parts:
            header = f"{header} [{' | '.join(detail_parts)}]"

        final_message = message.strip() if message else str(
            original or "Unbekannter Azure-OpenAI-Fehler")
        super().__init__(f"{header}: {final_message}")


class AzureFoundryBingAPIError(RuntimeError):
    """Raised when an Azure Foundry Bing Search agent call fails."""

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
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id
        self.original = original

        detail_parts: list[str] = []
        if status_code is not None:
            detail_parts.append(f"HTTP {status_code}")
        if error_code:
            detail_parts.append(error_code)
        if request_id:
            detail_parts.append(f"request-id={request_id}")

        header = f"Azure-Foundry-Bing-Aufruf fehlgeschlagen (agent={agent_id})"
        if detail_parts:
            header = f"{header} [{' | '.join(detail_parts)}]"

        final_message = message.strip() if message else str(
            original or "Unbekannter Azure-Foundry-Bing-Fehler")
        super().__init__(f"{header}: {final_message}")
