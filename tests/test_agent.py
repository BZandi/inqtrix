"""Tests for the public API: ResearchAgent, AgentConfig, ResearchResult."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inqtrix.agent import AgentConfig, ResearchAgent
from inqtrix.providers.base import LLMProvider, SearchProvider
from inqtrix.providers.anthropic import AnthropicLLM
from inqtrix.result import (
    Claim,
    ClaimMetrics,
    ResearchMetrics,
    ResearchResult,
    ResearchResultExportOptions,
    Source,
    SourceMetrics,
)
from inqtrix.settings import AgentSettings
from inqtrix.strategies import (
    SourceTieringStrategy,
    StopCriteriaStrategy,
    StrategyContext,
    create_default_strategies,
)


# ------------------------------------------------------------------ #
# AgentConfig tests
# ------------------------------------------------------------------ #


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.max_rounds == 4
        assert cfg.confidence_stop == 8
        assert cfg.answer_prompt_citations_max == 60
        assert cfg.max_total_seconds == 300
        assert cfg.llm is None
        assert cfg.search is None

    def test_override_behaviour(self):
        cfg = AgentConfig(max_rounds=2, confidence_stop=6)
        assert cfg.max_rounds == 2
        assert cfg.confidence_stop == 6

    def test_serialisation_roundtrip(self):
        cfg = AgentConfig(max_rounds=3)
        data = cfg.model_dump(exclude={"llm", "search"})
        restored = AgentConfig(**data)
        assert restored.max_rounds == 3

    def test_accepts_custom_provider(self):
        class DummyLLM(LLMProvider):
            def complete(self, *a, **kw):
                return "test"

            def summarize_parallel(self, *a, **kw):
                return ("", 0, 0)

            def is_available(self):
                return True

        cfg = AgentConfig(llm=DummyLLM())
        assert cfg.llm is not None
        assert cfg.llm.is_available()


# ------------------------------------------------------------------ #
# ResearchResult tests
# ------------------------------------------------------------------ #


class TestResearchResult:
    def test_empty_result(self):
        r = ResearchResult(answer="test")
        assert r.answer == "test"
        assert r.metrics.rounds == 0
        assert r.top_sources == []
        assert r.top_claims == []

    def test_from_raw_minimal(self):
        raw = {
            "answer": "Die Antwort",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "result_state": {
                "round": 2,
                "final_confidence": 7,
                "all_citations": ["https://example.com"],
                "queries": ["q1", "q2"],
                "consolidated_claims": [
                    {
                        "claim_text": "Ein Fakt",
                        "status": "verified",
                        "claim_type": "fact",
                        "needs_primary": True,
                        "status_reason": "mehrfach belegt",
                        "support_count": 2,
                        "contradict_count": 0,
                        "source_tier_counts": {
                            "primary": 1,
                            "mainstream": 0,
                            "stakeholder": 0,
                            "unknown": 0,
                            "low": 0,
                        },
                        "source_urls": ["https://example.com"],
                    }
                ],
                "source_tier_counts": {"primary": 0, "mainstream": 0, "stakeholder": 0, "unknown": 1, "low": 0},
                "source_quality_score": 0.35,
                "claim_status_counts": {"verified": 1, "contested": 0, "unverified": 0},
                "claim_quality_score": 1.0,
                "aspect_coverage": 0.5,
                "evidence_consistency": 8,
                "evidence_sufficiency": 6,
            },
        }
        result = ResearchResult.from_raw(raw)

        assert result.answer == "Die Antwort"
        assert result.metrics.rounds == 2
        assert result.metrics.confidence == 7
        assert result.metrics.total_queries == 2
        assert result.metrics.total_citations == 1
        assert result.metrics.prompt_tokens == 100
        assert result.metrics.completion_tokens == 50
        assert result.metrics.sources.quality_score == 0.35
        assert result.metrics.claims.quality_score == 1.0
        assert len(result.top_sources) == 1
        assert result.top_sources[0].url == "https://example.com"
        assert len(result.top_claims) == 1
        assert result.top_claims[0].text == "Ein Fakt"
        assert result.top_claims[0].status == "verified"
        assert result.top_claims[0].needs_primary is True
        assert result.top_claims[0].status_reason == "mehrfach belegt"
        assert result.top_claims[0].support_count == 2
        assert result.top_claims[0].source_tier_counts["primary"] == 1

    def test_from_raw_empty_state(self):
        raw = {"answer": "Keine Ergebnisse", "usage": {}, "result_state": {}}
        result = ResearchResult.from_raw(raw)
        assert result.answer == "Keine Ergebnisse"
        assert result.metrics.rounds == 0
        assert result.top_sources == []

    def test_json_serialisation(self):
        result = ResearchResult(
            answer="Test",
            metrics=ResearchMetrics(rounds=1, confidence=8),
            top_sources=[Source(url="https://a.com", tier="primary")],
            top_claims=[Claim(text="Fakt", status="verified")],
        )
        data = result.model_dump()
        assert data["answer"] == "Test"
        assert data["metrics"]["rounds"] == 1
        assert data["top_sources"][0]["tier"] == "primary"
        assert data["top_claims"][0]["needs_primary"] is False

        # JSON roundtrip
        json_str = result.model_dump_json()
        restored = ResearchResult.model_validate_json(json_str)
        assert restored.answer == "Test"
        assert restored.metrics.confidence == 8

    def test_claim_defaults_include_full_public_shape(self):
        c = Claim(text="Fakt")
        assert c.needs_primary is False
        assert c.status_reason == ""
        assert c.support_count == 0
        assert c.contradict_count == 0
        assert c.source_tier_counts == {
            "primary": 0,
            "mainstream": 0,
            "stakeholder": 0,
            "unknown": 0,
            "low": 0,
        }

    def test_export_payload_supports_optional_projection(self):
        result = ResearchResult(
            answer="Test",
            metrics=ResearchMetrics(rounds=2, confidence=8),
            top_sources=[
                Source(url="https://www.bundestag.de/dokumente/x", tier="primary"),
                Source(url="https://example.com/x", tier="unknown"),
            ],
            top_claims=[
                Claim(text="Fakt 1", status="verified", support_count=2),
                Claim(text="Fakt 2", status="contested", contradict_count=1),
            ],
        )

        payload = result.to_export_payload(
            ResearchResultExportOptions(
                include_sources=False,
                max_claims=1,
            )
        )

        assert payload["answer"] == "Test"
        assert payload["metrics"]["confidence"] == 8
        assert "top_sources" not in payload
        assert len(payload["top_claims"]) == 1
        assert payload["top_claims"][0]["text"] == "Fakt 1"


# ------------------------------------------------------------------ #
# ResearchAgent construction tests (no LLM calls)
# ------------------------------------------------------------------ #


class TestResearchAgentConstruction:
    def test_default_config(self):
        agent = ResearchAgent()
        assert agent.config.max_rounds == 4

    def test_default_initialisation_reads_dotenv(self, tmp_path, monkeypatch):
        import inqtrix.providers as providers_module

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MAX_ROUNDS", raising=False)
        monkeypatch.delenv("REASONING_MODEL", raising=False)
        monkeypatch.delenv("SEARCH_MODEL", raising=False)
        monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        (tmp_path / ".env").write_text(
            "\n".join([
                "MAX_ROUNDS=7",
                "REASONING_MODEL=dotenv-reasoning",
                "SEARCH_MODEL=dotenv-search",
                "LITELLM_BASE_URL=http://dotenv/v1",
                "LITELLM_API_KEY=dotenv-key",
            ])
            + "\n",
            encoding="utf-8",
        )

        class DummyOpenAI:
            def __init__(self, *, base_url, api_key, **_kw):
                self.base_url = base_url
                self.api_key = api_key
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))

        monkeypatch.setattr(providers_module, "OpenAI", DummyOpenAI)

        agent = ResearchAgent()

        providers, _, settings = agent._ensure_initialised()

        assert settings.max_rounds == 7
        assert providers.llm.models.reasoning_model == "dotenv-reasoning"
        assert providers.search._model == "dotenv-search"
        assert providers.llm._client.base_url == "http://dotenv/v1"
        assert providers.llm._client.api_key == "dotenv-key"

    def test_custom_config(self):
        cfg = AgentConfig(max_rounds=2, confidence_stop=6)
        agent = ResearchAgent(cfg)
        assert agent.config.max_rounds == 2
        assert agent.config.confidence_stop == 6

    def test_config_property(self):
        cfg = AgentConfig(max_rounds=3)
        agent = ResearchAgent(cfg)
        assert agent.config is cfg

    def test_ensure_initialised_respects_explicit_config_over_env(self, monkeypatch):
        import inqtrix.providers as providers_module
        from inqtrix.providers.litellm import LiteLLM
        from inqtrix.providers.perplexity import PerplexitySearch

        monkeypatch.setenv("MAX_ROUNDS", "99")
        monkeypatch.setenv("REASONING_MODEL", "env-model")
        monkeypatch.setenv("LITELLM_BASE_URL", "http://env/v1")
        monkeypatch.setenv("LITELLM_API_KEY", "env-key")

        class DummyOpenAI:
            def __init__(self, *, base_url, api_key, **_kw):
                self.base_url = base_url
                self.api_key = api_key
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))

        monkeypatch.setattr(providers_module, "OpenAI", DummyOpenAI)

        llm = LiteLLM(
            api_key="custom-key",
            base_url="http://custom/v1",
            default_model="custom-model",
        )
        search = PerplexitySearch(
            api_key="custom-key",
            base_url="http://custom/v1",
            model="sonar-pro",
            _client=llm._client,
        )
        cfg = AgentConfig(
            llm=llm,
            search=search,
            max_rounds=2,
            answer_prompt_citations_max=11,
        )
        agent = ResearchAgent(cfg)

        providers, _, settings = agent._ensure_initialised()

        assert settings.max_rounds == 2
        assert settings.answer_prompt_citations_max == 11
        assert providers.llm.models.reasoning_model == "custom-model"
        assert str(providers.llm._client.base_url).rstrip("/") == "http://custom/v1"
        assert providers.llm._client.api_key == "custom-key"

    def test_default_claim_extraction_uses_provider_interface(self):
        class DummyLLM(LLMProvider):
            def complete(self, *a, **kw):
                return (
                    '{"claims":[{"claim_text":"Der Beitrag steigt um 5 Prozent.",'
                    '"claim_type":"fact","polarity":"affirmed","needs_primary":true,'
                    '"source_urls":["https://example.com/report"],'
                    '"published_date":"2026-03-29"}]}'
                )

            def summarize_parallel(self, *a, **kw):
                return ("", 0, 0)

            def is_available(self):
                return True

        strategies = create_default_strategies(
            AgentSettings(),
            llm=DummyLLM(),
            summarize_model="summary-model",
        )
        claims, prompt_tokens, completion_tokens = strategies.claim_extraction.extract(
            "Kurztext",
            ["https://example.com/report"],
            "Was passiert?",
        )

        assert len(claims) == 1
        assert claims[0]["claim_text"] == "Der Beitrag steigt um 5 Prozent."
        assert prompt_tokens == 0
        assert completion_tokens == 0

    def test_custom_llm_without_models_gets_configured_model_metadata(self):
        class DummyLLM(LLMProvider):
            def __init__(self):
                self.models_seen: list[str | None] = []

            def complete(self, *a, **kw):
                self.models_seen.append(kw.get("model"))
                return "ok"

            def summarize_parallel(self, *a, **kw):
                return ("", 0, 0)

            def is_available(self):
                return True

        class DummySearch(SearchProvider):
            def search(self, *a, **kw):
                return {
                    "answer": "",
                    "citations": [],
                    "related_questions": [],
                    "_prompt_tokens": 0,
                    "_completion_tokens": 0,
                }

            def is_available(self):
                return True

        llm = DummyLLM()
        cfg = AgentConfig(
            llm=llm,
            search=DummySearch(),
        )
        agent = ResearchAgent(cfg)

        providers, _, _ = agent._ensure_initialised()

        assert providers.llm.complete("test") == "ok"
        assert llm.models_seen == [None]
        # Custom LLM without .models keeps provider-native model defaults
        assert hasattr(providers.llm, "models")
        assert providers.llm.models.reasoning_model == ""

    def test_anthropic_llm_keeps_own_model_metadata(self):
        class DummySearch(SearchProvider):
            def search(self, *a, **kw):
                return {
                    "answer": "",
                    "citations": [],
                    "related_questions": [],
                    "_prompt_tokens": 0,
                    "_completion_tokens": 0,
                }

            def is_available(self):
                return True

        llm = AnthropicLLM(
            api_key="anthropic-key",
            default_model="claude-opus-4-6",
            classify_model="claude-sonnet-4-6",
            summarize_model="claude-sonnet-4-6",
            evaluate_model="claude-haiku-4-5",
        )
        agent = ResearchAgent(AgentConfig(llm=llm, search=DummySearch()))

        providers, _, _ = agent._ensure_initialised()

        assert providers.llm.models.reasoning_model == "claude-opus-4-6"
        assert providers.llm.models.effective_classify_model == "claude-sonnet-4-6"
        assert providers.llm.models.effective_summarize_model == "claude-sonnet-4-6"
        assert providers.llm.models.effective_evaluate_model == "claude-haiku-4-5"

    def test_stream_yields_answer_in_word_chunks(self, monkeypatch):
        import inqtrix.graph as graph_module

        def fake_run(
            question,
            *,
            history,
            progress_queue,
            prev_session,
            providers,
            strategies,
            settings,
        ):
            progress_queue.put(("progress", "Schritt 1"))
            return {"answer": "Hallo Welt"}

        monkeypatch.setattr(graph_module, "run", fake_run)

        agent = ResearchAgent(AgentConfig())
        monkeypatch.setattr(agent, "_ensure_initialised", lambda: (None, None, None))

        chunks = list(agent.stream("Meine Frage"))

        assert chunks == ["> Schritt 1\n", "---\n", "Hallo ", "Welt"]

    def test_stream_can_skip_progress_updates(self, monkeypatch):
        import inqtrix.graph as graph_module

        def fake_run(
            question,
            *,
            history,
            progress_queue,
            prev_session,
            providers,
            strategies,
            settings,
        ):
            assert progress_queue is None
            return {"answer": "Hallo Welt"}

        monkeypatch.setattr(graph_module, "run", fake_run)

        agent = ResearchAgent(AgentConfig())
        monkeypatch.setattr(agent, "_ensure_initialised", lambda: (None, None, None))

        chunks = list(agent.stream("Meine Frage", include_progress=False))

        assert chunks == ["Hallo ", "Welt"]


# ------------------------------------------------------------------ #
# Pydantic model tests
# ------------------------------------------------------------------ #


class TestPydanticModels:
    def test_source_defaults(self):
        s = Source(url="https://example.com")
        assert s.tier == "unknown"

    def test_claim_defaults(self):
        c = Claim(text="test")
        assert c.status == "unverified"
        assert c.claim_type == "fact"
        assert c.sources == []

    def test_source_metrics_defaults(self):
        m = SourceMetrics()
        assert m.quality_score == 0.0
        assert m.tier_counts["primary"] == 0

    def test_claim_metrics_defaults(self):
        m = ClaimMetrics()
        assert m.quality_score == 0.0
        assert m.status_counts["verified"] == 0

    def test_research_metrics_nesting(self):
        m = ResearchMetrics(
            rounds=3,
            confidence=8,
            sources=SourceMetrics(quality_score=0.7),
            claims=ClaimMetrics(quality_score=0.9),
        )
        assert m.sources.quality_score == 0.7
        assert m.claims.quality_score == 0.9
