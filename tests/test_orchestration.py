"""Regression tests for graph wiring and node orchestration."""

from __future__ import annotations

from queue import Queue
from types import SimpleNamespace

from inqtrix.exceptions import AgentRateLimited, AnthropicAPIError
from inqtrix.graph import default_graph_config, run, run_test
from inqtrix.nodes import answer, evaluate, search
from inqtrix.providers import ProviderContext
from inqtrix.settings import AgentSettings
from inqtrix.state import initial_state
from inqtrix.strategies import StrategyContext, create_default_strategies


class _SearchStub:
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


class _EvalLLMStub:
    def __init__(self, response: str) -> None:
        self._response = response
        self.models = SimpleNamespace(
            reasoning_model="reasoning-model",
            effective_evaluate_model="evaluate-model",
        )

    def complete(self, *a, **kw):
        return self._response

    def summarize_parallel(self, *a, **kw):
        return ("", 0, 0)

    def is_available(self):
        return True


class _DoneStopCriteria:
    def check_contradictions(self, s, eval_text, conf):
        return conf

    def filter_irrelevant_blocks(self, s, eval_text):
        return None

    def extract_competing_events(self, s, eval_text, conf):
        return conf

    def extract_evidence_scores(self, s, eval_text, conf):
        return conf

    def check_falsification(self, s, conf, prev_conf):
        return False

    def check_stagnation(self, s, conf, prev_conf, n_citations, falsification_just_triggered):
        return conf, False

    def should_suppress_utility_stop(self, s):
        return False

    def compute_utility(self, s, conf, prev_conf, n_citations):
        s["done"] = True
        return 0.0, True

    def check_plateau(self, s, conf, prev_conf, stagnation_detected):
        return False

    def should_stop(self, state):
        return False, ""


def test_default_graph_config_uses_answer_node_name():
    settings = AgentSettings()
    providers = ProviderContext(llm=_EvalLLMStub(""), search=_SearchStub())
    strategies = create_default_strategies(settings)

    config = default_graph_config(providers, strategies, settings)

    assert "answer" in config.nodes
    assert "synthesize" not in config.nodes
    routes = {source: router({"done": True}) for source, router in config.conditional_edges}
    assert routes == {
        "classify": "answer",
        "plan": "answer",
        "evaluate": "answer",
    }


def test_run_test_exports_public_sources_and_claims(monkeypatch):
    import inqtrix.graph as graph_module

    class _StubAgent:
        def invoke(self, state):
            return {
                "answer": "Antwort",
                "round": 1,
                "queries": ["q1"],
                "all_citations": ["https://www.bundestag.de/dokumente/x"],
                "consolidated_claims": [
                    {
                        "claim_text": "Ein verifizierter Fakt",
                        "status": "verified",
                        "claim_type": "fact",
                        "needs_primary": True,
                        "status_reason": "primaer belegt",
                        "support_count": 2,
                        "contradict_count": 0,
                        "source_tier_counts": {
                            "primary": 1,
                            "mainstream": 0,
                            "stakeholder": 0,
                            "unknown": 0,
                            "low": 0,
                        },
                        "source_urls": ["https://www.bundestag.de/dokumente/x"],
                    }
                ],
                "iteration_logs": [{"node": "answer"}],
                "final_confidence": 8,
                "source_tier_counts": {
                    "primary": 1,
                    "mainstream": 0,
                    "stakeholder": 0,
                    "unknown": 0,
                    "low": 0,
                },
                "source_quality_score": 1.0,
                "claim_status_counts": {"verified": 1, "contested": 0, "unverified": 0},
                "claim_quality_score": 1.0,
            }

    monkeypatch.setattr(
        graph_module,
        "get_agent",
        lambda providers, strategies, settings: _StubAgent(),
    )

    settings = AgentSettings()
    providers = ProviderContext(llm=_EvalLLMStub(""), search=_SearchStub())
    strategies = create_default_strategies(settings)

    result = run_test(
        "Was ist passiert?",
        providers=providers,
        strategies=strategies,
        settings=settings,
    )

    assert result["top_sources"][0]["tier"] == "primary"
    assert result["top_claims"][0]["needs_primary"] is True
    assert result["top_claims"][0]["support_count"] == 2


def test_evaluate_emits_completion_when_stop_criteria_already_finished():
    settings = AgentSettings(confidence_stop=8, max_rounds=4)
    defaults = create_default_strategies(settings)
    strategies = StrategyContext(
        source_tiering=defaults.source_tiering,
        claim_extraction=defaults.claim_extraction,
        claim_consolidation=defaults.claim_consolidation,
        context_pruning=defaults.context_pruning,
        risk_scoring=defaults.risk_scoring,
        stop_criteria=_DoneStopCriteria(),
    )
    progress_queue = Queue()
    state = initial_state("Was ist passiert?",
                          progress_queue=progress_queue, max_total_seconds=30)
    state["round"] = 2
    state["context"] = ["Kontextblock"]
    state["all_citations"] = ["https://example.com/report"]

    provider_context = ProviderContext(
        llm=_EvalLLMStub(
            "STATUS: INSUFFICIENT\n"
            "CONFIDENCE: 6\n"
            "GAPS: Keine\n"
            "CONTRADICTIONS: Nein\n"
            "IRRELEVANT: Keine\n"
            "COMPETING_EVENTS: Keine\n"
            "EVIDENCE_CONSISTENCY: 8\n"
            "EVIDENCE_SUFFICIENCY: 8"
        ),
        search=_SearchStub(),
    )

    evaluate(
        state,
        providers=provider_context,
        strategies=strategies,
        settings=settings,
    )

    messages = []
    while not progress_queue.empty():
        messages.append(progress_queue.get()[1])

    assert messages[-1].startswith("Recherche abgeschlossen")


def test_search_emits_fallback_progress_messages():
    class _SearchWithNotice:
        def __init__(self) -> None:
            self._notice = None

        def search(self, *a, **kw):
            self._notice = "search fallback"
            return {
                "answer": "Gefundener Text",
                "citations": ["https://example.com/report"],
                "related_questions": [],
                "_prompt_tokens": 0,
                "_completion_tokens": 0,
            }

        def consume_nonfatal_notice(self):
            notice = self._notice
            self._notice = None
            return notice

        def is_available(self):
            return True

    class _SummarizeLLM:
        def __init__(self) -> None:
            self._notice = None
            self.models = SimpleNamespace(
                reasoning_model="reasoning-model",
                effective_summarize_model="summary-model",
            )

        def complete(self, *a, **kw):
            return ""

        def summarize_parallel(self, *a, **kw):
            self._notice = "summarize fallback"
            return ("Fallback-Zusammenfassung", 0, 0)

        def consume_nonfatal_notice(self):
            notice = self._notice
            self._notice = None
            return notice

        def is_available(self):
            return True

    class _ClaimExtractionWithNotice:
        def __init__(self) -> None:
            self._notice = None

        def extract(self, *a, **kw):
            self._notice = "claim fallback"
            return ([], 0, 0)

        def consume_nonfatal_notice(self):
            notice = self._notice
            self._notice = None
            return notice

    settings = AgentSettings(first_round_queries=1, max_rounds=4)
    llm = _SummarizeLLM()
    defaults = create_default_strategies(settings, llm=llm, summarize_model="summary-model")
    strategies = StrategyContext(
        source_tiering=defaults.source_tiering,
        claim_extraction=_ClaimExtractionWithNotice(),
        claim_consolidation=defaults.claim_consolidation,
        context_pruning=defaults.context_pruning,
        risk_scoring=defaults.risk_scoring,
        stop_criteria=defaults.stop_criteria,
    )
    progress_queue = Queue()
    state = initial_state("Was ist passiert?",
                          progress_queue=progress_queue, max_total_seconds=30)
    state["queries"] = ["q1"]

    search(
        state,
        providers=ProviderContext(llm=llm, search=_SearchWithNotice()),
        strategies=strategies,
        settings=settings,
    )

    messages = []
    while not progress_queue.empty():
        messages.append(progress_queue.get()[1])

    assert any("Suchanfragen liefen mit Provider-Fallback weiter" in msg for msg in messages)
    assert any("Quellen-Zusammenfassungen liefen im Fallback-Modus" in msg for msg in messages)
    assert any("Claim-Extraktion" in msg for msg in messages)


def test_run_emits_progress_before_rate_limit_abort(monkeypatch):
    import inqtrix.graph as graph_module

    class _StubAgent:
        def invoke(self, state):
            raise AgentRateLimited("demo-model", RuntimeError("429"))

    monkeypatch.setattr(
        graph_module,
        "get_agent",
        lambda providers, strategies, settings: _StubAgent(),
    )

    progress_queue = Queue()
    settings = AgentSettings()
    providers = ProviderContext(llm=_EvalLLMStub(""), search=_SearchStub())
    strategies = create_default_strategies(settings)

    run(
        "Was ist passiert?",
        progress_queue=progress_queue,
        providers=providers,
        strategies=strategies,
        settings=settings,
    )

    messages = []
    while not progress_queue.empty():
        messages.append(progress_queue.get()[1])

    assert any(msg.startswith("Recherche abgebrochen:") for msg in messages)


def test_answer_catches_anthropic_api_error_and_falls_back():
    """AnthropicAPIError in answer node must trigger context fallback, not crash."""

    class _FailingLLM:
        def __init__(self):
            self.models = SimpleNamespace(
                reasoning_model="claude-sonnet-4-6",
                effective_evaluate_model="claude-sonnet-4-6",
            )

        def complete(self, *a, **kw):
            raise AnthropicAPIError(
                model="claude-sonnet-4-6",
                status_code=529,
                error_type="overloaded_error",
                message="Overloaded",
            )

        def summarize_parallel(self, *a, **kw):
            return ("", 0, 0)

        def is_available(self):
            return True

    settings = AgentSettings()
    defaults = create_default_strategies(settings)
    progress_queue = Queue()
    state = initial_state("Was ist passiert?",
                          progress_queue=progress_queue, max_total_seconds=30)
    state["context"] = ["Recherche-Kontext: Wichtige Informationen"]
    state["all_citations"] = ["https://example.com/report"]
    state["consolidated_claims"] = []

    answer(
        state,
        providers=ProviderContext(llm=_FailingLLM(), search=_SearchStub()),
        strategies=StrategyContext(
            source_tiering=defaults.source_tiering,
            claim_extraction=defaults.claim_extraction,
            claim_consolidation=defaults.claim_consolidation,
            context_pruning=defaults.context_pruning,
            risk_scoring=defaults.risk_scoring,
            stop_criteria=defaults.stop_criteria,
        ),
        settings=settings,
    )

    # Must not crash; should contain context fallback
    assert "Recherche-Kontext" in state["answer"]
