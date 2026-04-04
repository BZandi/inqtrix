"""Direct Anthropic adapter for the LLMProvider interface."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inqtrix.constants import REASONING_TIMEOUT, SUMMARIZE_TIMEOUT
from inqtrix.exceptions import AgentRateLimited, AgentTimeout
from inqtrix.prompts import SUMMARIZE_PROMPT
from inqtrix.providers import LLMProvider, LLMResponse, _bounded_timeout, _check_deadline
from inqtrix.state import track_tokens

log = logging.getLogger("inqtrix")


class AnthropicLLM(LLMProvider):
    """Call the Anthropic Messages API directly without LiteLLM."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.anthropic.com/v1/messages",
        anthropic_version: str = "2023-06-01",
        default_model: str = "claude-3-7-sonnet-latest",
        summarize_model: str = "claude-3-5-haiku-latest",
        default_max_tokens: int = 1024,
        summarize_max_tokens: int = 512,
        user_agent: str = "inqtrix/0.1",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._anthropic_version = anthropic_version
        self._default_model = default_model
        self._summarize_model = summarize_model
        self._default_max_tokens = default_max_tokens
        self._summarize_max_tokens = summarize_max_tokens
        self._user_agent = user_agent

    def _request_json(
        self,
        *,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        request = Request(
            self._base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": self._user_agent,
                "x-api-key": self._api_key,
                "anthropic-version": self._anthropic_version,
            },
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        parts: list[str] = []
        for block in payload.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_usage(payload: dict[str, Any]) -> tuple[int, int]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return (0, 0)
        return (
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        )

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

        use_model = model or self._default_model
        payload: dict[str, Any] = {
            "model": use_model,
            "max_tokens": self._default_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        try:
            raw = self._request_json(
                payload=payload,
                timeout=_bounded_timeout(timeout, deadline),
            )
        except HTTPError as exc:
            if exc.code == 429:
                raise AgentRateLimited(use_model, exc)
            raise RuntimeError(
                f"Anthropic-Aufruf fehlgeschlagen ({use_model}): {exc}"
            ) from exc
        except (URLError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"Anthropic-Aufruf fehlgeschlagen ({use_model}): {exc}"
            ) from exc

        prompt_tokens, completion_tokens = self._extract_usage(raw)
        response = LLMResponse(
            content=self._extract_text(raw),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=use_model,
        )
        if state is not None:
            track_tokens(state, response)
        return response

    def summarize_parallel(
        self,
        text: str,
        deadline: float | None = None,
    ) -> tuple[str, int, int]:
        if not text.strip():
            return ("", 0, 0)
        if deadline is not None:
            _check_deadline(deadline)

        payload: dict[str, Any] = {
            "model": self._summarize_model,
            "max_tokens": self._summarize_max_tokens,
            "messages": [{"role": "user", "content": f"{SUMMARIZE_PROMPT}{text[:6000]}"}],
        }

        try:
            raw = self._request_json(
                payload=payload,
                timeout=_bounded_timeout(SUMMARIZE_TIMEOUT, deadline),
            )
        except AgentRateLimited:
            raise
        except HTTPError as exc:
            if exc.code == 429:
                raise AgentRateLimited(self._summarize_model, exc)
            log.error("Anthropic-Summarize fehlgeschlagen (%s): %s", self._summarize_model, exc)
            return (text[:800], 0, 0)
        except (URLError, OSError, ValueError, AgentTimeout) as exc:
            log.error("Anthropic-Summarize fehlgeschlagen (%s): %s", self._summarize_model, exc)
            return (text[:800], 0, 0)

        prompt_tokens, completion_tokens = self._extract_usage(raw)
        return (
            self._extract_text(raw),
            prompt_tokens,
            completion_tokens,
        )

    def is_available(self) -> bool:
        return bool(self._api_key)
