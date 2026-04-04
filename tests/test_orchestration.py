"""Regression tests for graph wiring and node orchestration."""

from __future__ import annotations

from queue import Queue
from types import SimpleNamespace

from inqtrix.graph import default_graph_config, run_test
from inqtrix.nodes import evaluate
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
