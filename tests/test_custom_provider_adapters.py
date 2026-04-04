"""Tests for the built-in direct provider adapters."""

from __future__ import annotations

from inqtrix.providers_anthropic import AnthropicLLM
from inqtrix.providers_brave import BraveSearch


def test_brave_search_maps_response_and_filters(monkeypatch):
    search = BraveSearch(api_key="brave-key")
    captured: dict[str, object] = {}

    def fake_request_json(*, params, timeout):
        captured["params"] = params
        return {
            "web": {
                "results": [
                    {
                        "url": "https://example.com/report",
                        "title": "Titel",
                        "description": "Beschreibung",
                        "extra_snippets": ["Snippet A"],
                    }
                ]
            }
        }

    monkeypatch.setattr(search, "_request_json", fake_request_json)

    result = search.search(
        "gkv reform",
        recency_filter="week",
        language_filter=["de"],
        domain_filter=["bundesgesundheitsministerium.de", "-reddit.com"],
    )

    params = captured["params"]
    assert isinstance(params, dict)
    assert params["freshness"] == "pw"
    assert params["search_lang"] == "de"
    assert "site:bundesgesundheitsministerium.de" in params["q"]
    assert "-site:reddit.com" in params["q"]
    assert result["citations"] == ["https://example.com/report"]
    assert "Titel" in result["answer"]
    assert result["_prompt_tokens"] == 0
    assert result["_completion_tokens"] == 0


def test_anthropic_llm_parses_response_and_tracks_tokens(monkeypatch):
    llm = AnthropicLLM(
        api_key="anthropic-key",
        default_model="claude-3-7-sonnet-latest",
        summarize_model="claude-3-5-haiku-latest",
    )

    def fake_request_json(*, payload, timeout):
        assert payload["model"] == "claude-3-7-sonnet-latest"
        return {
            "content": [
                {"type": "text", "text": "Hallo"},
                {"type": "text", "text": " Welt"},
            ],
            "usage": {"input_tokens": 12, "output_tokens": 7},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)

    state = {"total_prompt_tokens": 0, "total_completion_tokens": 0}
    response = llm.complete_with_metadata("Frage", state=state)

    assert response.content == "Hallo Welt"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 7
    assert state["total_prompt_tokens"] == 12
    assert state["total_completion_tokens"] == 7


def test_anthropic_llm_summarize_uses_summarize_model(monkeypatch):
    llm = AnthropicLLM(
        api_key="anthropic-key",
        default_model="claude-3-7-sonnet-latest",
        summarize_model="claude-3-5-haiku-latest",
    )

    def fake_request_json(*, payload, timeout):
        assert payload["model"] == "claude-3-5-haiku-latest"
        return {
            "content": [{"type": "text", "text": "Kurzfassung"}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }

    monkeypatch.setattr(llm, "_request_json", fake_request_json)

    summary = llm.summarize_parallel("Langer Text")

    assert summary == ("Kurzfassung", 3, 2)
