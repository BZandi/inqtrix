"""Behavioral tests for effective-actor provider safepoints."""

from __future__ import annotations

from typing import Any

import pytest

from inqtrix.execution_authority import (
    AuthorizationRevoked,
    guard_provider_context,
    pinned_knowledge_collection_ids,
)
from inqtrix.providers.base import (
    ChatTurn,
    LLMProvider,
    LLMResponse,
    ProviderContext,
    SearchProvider,
    StructuredLLMResponse,
)
from inqtrix.search_result import GroundedSearchResult


class _RecordingLLM(LLMProvider):
    """Provider stub that records externally observable call boundaries."""

    selectable_models = ["reasoning-model"]
    context_window_tokens = 32_000

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(f"llm:{prompt}")
        return "answer"

    def complete_with_metadata(
        self, prompt: str, **kwargs: Any
    ) -> LLMResponse:
        self.calls.append(f"metadata:{prompt}")
        return LLMResponse(content="answer")

    def complete_structured(
        self, prompt: str, **kwargs: Any
    ) -> StructuredLLMResponse:
        self.calls.append(f"structured:{prompt}")
        return StructuredLLMResponse(parsed={"ok": True})

    def chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatTurn:
        self.calls.append("chat")
        return ChatTurn(text="answer")

    def supports_structured_output(self, *, model: str | None = None) -> bool:
        return True

    def supports_tool_calls(self, *, model: str | None = None) -> bool:
        return True

    def is_available(self) -> bool:
        return True


class _RecordingSearch(SearchProvider):
    """Search stub that records the real provider invocation."""

    search_model = "search-model"

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def search(self, query: str, **kwargs: Any) -> GroundedSearchResult:
        self.calls.append(f"search:{query}")
        return GroundedSearchResult(answer="result")

    def is_available(self) -> bool:
        return True


def _providers(calls: list[str]) -> ProviderContext:
    return ProviderContext(
        llm=_RecordingLLM(calls),
        search=_RecordingSearch(calls),
    )


@pytest.mark.parametrize(
    ("invoke", "provider_call"),
    [
        (lambda context: context.llm.complete("plain"), "llm:plain"),
        (
            lambda context: context.llm.complete_with_metadata("metered"),
            "metadata:metered",
        ),
        (
            lambda context: context.llm.complete_structured(
                "json", schema={}, schema_name="test"
            ),
            "structured:json",
        ),
        (lambda context: context.llm.chat([]), "chat"),
        (lambda context: context.search.search("query"), "search:query"),
    ],
)
def test_external_provider_calls_are_guarded_before_and_after(
    invoke: Any,
    provider_call: str,
) -> None:
    calls: list[str] = []
    guarded = guard_provider_context(
        _providers(calls), lambda: calls.append("check")
    )

    invoke(guarded)

    assert calls == ["check", provider_call, "check"]


def test_revocation_during_external_call_discards_its_return_value() -> None:
    calls: list[str] = []
    checks = 0

    def check() -> None:
        nonlocal checks
        checks += 1
        calls.append("check")
        if checks == 2:
            raise AuthorizationRevoked("share was revoked")

    guarded = guard_provider_context(_providers(calls), check)

    with pytest.raises(AuthorizationRevoked) as exc_info:
        guarded.llm.complete("in-flight")

    assert exc_info.value.code == "authorization_revoked"
    assert calls == ["check", "llm:in-flight", "check"]


def test_guard_preserves_provider_capability_metadata() -> None:
    guarded = guard_provider_context(_providers([]), lambda: None)

    assert guarded.llm.selectable_models == ["reasoning-model"]
    assert guarded.llm.context_window_tokens == 32_000
    assert guarded.llm.supports_structured_output()
    assert guarded.llm.supports_tool_calls()
    assert guarded.search.search_model == "search-model"


def test_persisted_knowledge_scope_is_immutable_and_missing_fails_closed() -> None:
    assert pinned_knowledge_collection_ids(
        {"collection_ids": ["kc_a", "kc_b", "kc_a"]},
        scoped_principal=True,
    ) == frozenset({"kc_a", "kc_b"})
    assert pinned_knowledge_collection_ids(
        {"collection_ids": []}, scoped_principal=True
    ) == frozenset()
    assert pinned_knowledge_collection_ids(
        {}, scoped_principal=True
    ) == frozenset()
    assert pinned_knowledge_collection_ids(
        {}, scoped_principal=False
    ) is None


@pytest.mark.parametrize(
    "invalid_scope",
    ["kc_a", [""], [1]],
)
def test_malformed_persisted_knowledge_scope_fails_loudly(
    invalid_scope: object,
) -> None:
    with pytest.raises(RuntimeError, match="list of IDs"):
        pinned_knowledge_collection_ids(
            {"collection_ids": invalid_scope},
            scoped_principal=True,
        )
