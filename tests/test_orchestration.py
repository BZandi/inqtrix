"""Regression tests for graph wiring and node orchestration."""

from __future__ import annotations

from queue import Queue
from types import SimpleNamespace

from inqtrix.exceptions import AgentRateLimited, AnthropicAPIError
from inqtrix.graph import default_graph_config, run, run_test
from inqtrix.nodes import (
    _build_answer_appendix_sections,
    answer,
    apply_confidence_guardrails,
    evaluate,
    search,
)
from inqtrix.providers.base import LLMResponse, ProviderContext
from inqtrix.report_profiles import ReportProfile
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


def test_evaluate_min_rounds_suppresses_early_stop():
    """Phase 13: min_rounds prevents an early confidence-driven stop.

    The stub stop criteria immediately set ``done=True`` (utility_stop) and
    the eval LLM returns CONFIDENCE: 9 which exceeds confidence_stop=8.
    With min_rounds=2 and the agent currently at round 1, both early-stop
    triggers must be suppressed and the loop must continue.
    """
    settings = AgentSettings(confidence_stop=8, max_rounds=4, min_rounds=2)
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
    state["round"] = 1  # below min_rounds (=2)
    state["context"] = ["Kontextblock"]
    state["all_citations"] = ["https://example.com/report"]

    provider_context = ProviderContext(
        llm=_EvalLLMStub(
            "STATUS: SUFFICIENT\n"
            "CONFIDENCE: 9\n"
            "GAPS: Keine\n"
            "CONTRADICTIONS: Nein\n"
            "IRRELEVANT: Keine\n"
            "COMPETING_EVENTS: Keine\n"
            "EVIDENCE_CONSISTENCY: 8\n"
            "EVIDENCE_SUFFICIENCY: 8"
        ),
        search=_SearchStub(),
    )

    evaluate(state, providers=provider_context,
             strategies=strategies, settings=settings)

    assert state["done"] is False, "min_rounds must suppress early stop"
    messages = []
    while not progress_queue.empty():
        messages.append(progress_queue.get()[1])
    assert any("min_rounds=2" in m for m in messages), (
        f"Expected a min_rounds progress hint, got: {messages}"
    )


def test_evaluate_min_rounds_does_not_override_max_rounds():
    """min_rounds must never extend the loop beyond max_rounds.

    Misconfiguration ``min_rounds > max_rounds`` is bounded by the user's
    explicit hard cap.
    """
    settings = AgentSettings(confidence_stop=8, max_rounds=2, min_rounds=5)
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
    state["round"] = 2  # = max_rounds
    state["context"] = ["Kontextblock"]
    state["all_citations"] = ["https://example.com/report"]

    provider_context = ProviderContext(
        llm=_EvalLLMStub(
            "STATUS: SUFFICIENT\n"
            "CONFIDENCE: 9\n"
            "GAPS: Keine\n"
            "CONTRADICTIONS: Nein\n"
            "IRRELEVANT: Keine\n"
            "COMPETING_EVENTS: Keine\n"
            "EVIDENCE_CONSISTENCY: 8\n"
            "EVIDENCE_SUFFICIENCY: 8"
        ),
        search=_SearchStub(),
    )

    evaluate(state, providers=provider_context,
             strategies=strategies, settings=settings)

    assert state["done"] is True, (
        "max_rounds must always win over min_rounds; loop must terminate"
    )


def test_evaluate_min_rounds_default_one_preserves_legacy_behaviour():
    """Default min_rounds=1 keeps the pre-Phase-13 stop semantics."""
    settings = AgentSettings(confidence_stop=8, max_rounds=4)  # min_rounds default = 1
    assert settings.min_rounds == 1
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
    state["round"] = 1  # >= min_rounds (=1) so stop is allowed
    state["context"] = ["Kontextblock"]
    state["all_citations"] = ["https://example.com/report"]

    provider_context = ProviderContext(
        llm=_EvalLLMStub(
            "STATUS: SUFFICIENT\n"
            "CONFIDENCE: 9\n"
            "GAPS: Keine\n"
            "CONTRADICTIONS: Nein\n"
            "IRRELEVANT: Keine\n"
            "COMPETING_EVENTS: Keine\n"
            "EVIDENCE_CONSISTENCY: 8\n"
            "EVIDENCE_SUFFICIENCY: 8"
        ),
        search=_SearchStub(),
    )

    evaluate(state, providers=provider_context,
             strategies=strategies, settings=settings)

    assert state["done"] is True


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

    assert any("Suchanfragen fehlgeschlagen" in msg for msg in messages)
    assert any("Zusammenfassungen" in msg and "fehlgeschlagen" in msg for msg in messages)
    assert any("Claim-Extraktionen fehlgeschlagen" in msg for msg in messages)


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


def test_answer_composes_sections_with_smaller_body_citation_pool():
    class _SectionedLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.models = SimpleNamespace(
                reasoning_model="reasoning-model",
                effective_evaluate_model="evaluate-model",
            )
            self._sections = [
                "Direkte Antwort mit Quelle [1](https://source1.example/report).",
                "- Punkt eins [2](https://source2.example/report)\n- Punkt zwei [3](https://source3.example/report)",
                "### Kosten\n\nDetailanalyse mit Zahl [4](https://source4.example/report).\n\n### Wirkung\n\nWeitere Einordnung [5](https://source5.example/report).",
                "Ausblick mit Einordnung [6](https://source6.example/report).",
            ]

        def complete_with_metadata(self, *a, **kw):
            finish_reason = "length" if not self.calls else "stop"
            self.calls.append(
                {
                    "prompt": a[0],
                    "system": kw.get("system", ""),
                    "max_output_tokens": kw.get("max_output_tokens"),
                }
            )
            return LLMResponse(
                content=self._sections[len(self.calls) - 1],
                prompt_tokens=50,
                completion_tokens=120,
                model="reasoning-model",
                finish_reason=finish_reason,
                raw={"choices": [{"finish_reason": finish_reason}]},
            )

        def complete(self, *a, **kw):
            raise AssertionError("answer() should use section-wise metadata completions")

        def summarize_parallel(self, *a, **kw):
            return ("", 0, 0)

        def is_available(self):
            return True

    settings = AgentSettings(report_profile=ReportProfile.COMPACT, testing_mode=True)
    defaults = create_default_strategies(settings)
    llm = _SectionedLLM()
    state = initial_state("Was ist passiert?", max_total_seconds=30)
    state["round"] = 1
    state["queries"] = ["q1"]
    state["final_confidence"] = 7
    state["all_citations"] = [
        "https://source1.example/report",
        "https://source2.example/report",
        "https://source3.example/report",
        "https://source4.example/report",
        "https://source5.example/report",
        "https://source6.example/report",
        "https://source7.example/report",
        "https://source8.example/report",
        "https://source9.example/report",
        "https://source10.example/report",
        "https://source11.example/report",
        "https://source12.example/report",
    ]

    answer(
        state,
        providers=ProviderContext(llm=llm, search=_SearchStub()),
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

    assert len(llm.calls) == 4
    assert "## Kurzfazit" in state["answer"]
    assert "## Kernaussagen" in state["answer"]
    assert "## Detailanalyse" in state["answer"]
    assert "## Einordnung / Ausblick" in state["answer"]
    assert "## Referenzen" in state["answer"]
    assert state["answer_incomplete"] is False
    assert state["total_prompt_tokens"] == 200
    assert state["total_completion_tokens"] == 480
    assert state["iteration_logs"][-1]["prompt_citation_count"] == 12
    assert state["iteration_logs"][-1]["body_prompt_citation_count"] == 12
    assert state["iteration_logs"][-1]["section_logs"][0]["limit_hit"] is True
    assert state["iteration_logs"][-1]["section_logs"][0]["accepted_with_limit"] is True


def test_answer_omits_truncated_section_and_marks_incomplete_without_recovery():
    """Truncated sections are repaired and included; loop continues through all sections."""

    class _TruncatedLLM:
        def __init__(self) -> None:
            self.calls = 0
            self.models = SimpleNamespace(
                reasoning_model="reasoning-model",
                effective_evaluate_model="evaluate-model",
            )

        def complete_with_metadata(self, *a, **kw):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="Kurzer Befund mit Primaerquelle [1](https://source1.example/report).",
                    prompt_tokens=80,
                    completion_tokens=150,
                    model="reasoning-model",
                    finish_reason="stop",
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            if self.calls == 2:
                # Truncated section with unbalanced bold — will be repaired
                return LLMResponse(
                    content="Der Kontext zeigt, dass die **Kass",
                    prompt_tokens=80,
                    completion_tokens=200,
                    model="reasoning-model",
                    finish_reason="length",
                    raw={"choices": [{"finish_reason": "length"}]},
                )
            # Remaining sections complete normally
            return LLMResponse(
                content=f"Abschnitt {self.calls} mit Inhalt.",
                prompt_tokens=80,
                completion_tokens=100,
                model="reasoning-model",
                finish_reason="stop",
                raw={"choices": [{"finish_reason": "stop"}]},
            )

        def complete(self, *a, **kw):
            raise AssertionError("answer() must not fall back to a plain recovery completion")

        def summarize_parallel(self, *a, **kw):
            return ("", 0, 0)

        def is_available(self):
            return True

    settings = AgentSettings(report_profile=ReportProfile.DEEP, testing_mode=True)
    defaults = create_default_strategies(settings)
    llm = _TruncatedLLM()
    state = initial_state("Was ist passiert?", max_total_seconds=30)
    state["round"] = 1
    state["queries"] = ["q1"]
    state["final_confidence"] = 7
    state["all_citations"] = [
        f"https://source{i}.example/report" for i in range(1, 26)
    ]

    answer(
        state,
        providers=ProviderContext(llm=llm, search=_SearchStub()),
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

    # All 6 DEEP sections were attempted (loop never aborted)
    assert llm.calls == 6
    # First section included normally
    assert "## Executive Summary" in state["answer"]
    # Truncated section was repaired and included (not dropped)
    assert "## Hintergrund / Kontext" in state["answer"]
    assert "die **Kass**" in state["answer"]  # repaired: closing ** added
    # No Synthese-Status stub
    assert "## Synthese-Status" not in state["answer"]
    # Remaining sections were generated
    assert "## Analyse" in state["answer"]
    assert "## Perspektiven / Positionen" in state["answer"]
    assert "## Risiken / Unsicherheiten" in state["answer"]
    assert "## Fazit / Ausblick" in state["answer"]
    # Reference and link sections present
    assert "## Referenzen" in state["answer"]
    assert "## Weiterfuehrende Links" in state["answer"]
    assert state["iteration_logs"][-1]["prompt_citation_count"] == 25
    assert state["iteration_logs"][-1]["body_prompt_citation_count"] == 25
    # Section 2 shows limit_hit in logs
    section_log_2 = state["iteration_logs"][-1]["section_logs"][1]
    assert section_log_2["limit_hit"] is True
    assert section_log_2["finish_reason"] == "length"
    # Log WARNING was emitted for the truncated section (visible in caplog)
    assert state["iteration_logs"][-1]["reference_link_count"] >= 1
    assert state["iteration_logs"][-1]["additional_link_count"] >= 1


def test_reference_extraction_runs_when_answer_complete():
    """Markdown-linked references must be extracted even when the answer
    has no ``incomplete_reasons`` entries."""
    settings = AgentSettings()
    defaults = create_default_strategies(settings)
    strategies = StrategyContext(
        source_tiering=defaults.source_tiering,
        claim_extraction=defaults.claim_extraction,
        claim_consolidation=defaults.claim_consolidation,
        context_pruning=defaults.context_pruning,
        risk_scoring=defaults.risk_scoring,
        stop_criteria=defaults.stop_criteria,
    )
    answer_text = (
        "Ein Verweis auf [1](https://example.com/report/a) und "
        "ein zweiter auf [2](https://example.com/report/b)."
    )
    prompt_citations = [
        "https://example.com/report/a",
        "https://example.com/report/b",
    ]
    all_citations = prompt_citations + ["https://example.com/extra/c"]

    sections, reference_count, _ = _build_answer_appendix_sections(
        answer_text,
        prompt_citations=prompt_citations,
        all_citations=all_citations,
        strategies=strategies,
        incomplete_reasons=[],
        finish_reason="stop",
    )

    # No Hinweis-zur-Vollständigkeit block because incomplete_reasons is empty
    assert all("Hinweis zur Vollständigkeit" not in section for section in sections)
    # But the referenced URLs must still be extracted
    assert reference_count == 2
    joined = "\n".join(sections)
    assert "https://example.com/report/a" in joined
    assert "https://example.com/report/b" in joined


class TestApplyConfidenceGuardrails:

    def _args(self, **overrides):
        base = {
            "has_citations": True,
            "primary_n": 2,
            "mainstream_n": 1,
            "low_n": 0,
            "uncovered_aspects": [],
            "contested_claims": 0,
            "needs_primary": False,
            "existing_gap": "",
        }
        base.update(overrides)
        return base

    def test_no_citations_clamps_and_sets_gap(self):
        result = apply_confidence_guardrails(
            9, **self._args(has_citations=False))
        assert result.confidence == 6
        assert result.gap_suggestion == "Keine belastbaren Quellen gefunden."
        assert any("no_citations" in r for r in result.reasons)

    def test_low_quality_majority_caps_at_seven(self):
        result = apply_confidence_guardrails(
            9, **self._args(primary_n=0, mainstream_n=1, low_n=3))
        assert result.confidence == 7
        assert any("low_quality_majority" in r for r in result.reasons)

    def test_needs_primary_without_primary_caps_at_eight(self):
        result = apply_confidence_guardrails(
            9, **self._args(primary_n=0, mainstream_n=2, needs_primary=True))
        assert result.confidence == 8
        assert result.gap_suggestion == (
            "Zentrale Zahlen/Regelungen nicht mit Primaerquelle belegt."
        )

    def test_uncovered_aspects_caps_at_eight(self):
        result = apply_confidence_guardrails(
            10, **self._args(uncovered_aspects=["Rechtslage", "Kosten"]))
        assert result.confidence == 8
        assert "Rechtslage" in (result.gap_suggestion or "")

    def test_contested_claims_caps_at_seven(self):
        result = apply_confidence_guardrails(
            9, **self._args(contested_claims=2))
        assert result.confidence == 7
        assert "umstritten" in (result.gap_suggestion or "")

    def test_existing_gap_is_preserved(self):
        """When state already has a gap, no suggestion should overwrite it."""
        result = apply_confidence_guardrails(
            10, **self._args(
                has_citations=False,
                uncovered_aspects=["Kosten"],
                existing_gap="Vorher belegter Gap-Text",
            ))
        assert result.gap_suggestion is None

    def test_guardrails_stable_under_multiple_clamps(self):
        """Running the function twice on its own output must be a no-op."""
        first = apply_confidence_guardrails(
            10, **self._args(
                primary_n=0,
                mainstream_n=0,
                low_n=5,
                contested_claims=3,
                uncovered_aspects=["A"],
                needs_primary=True,
            ))
        second = apply_confidence_guardrails(
            first.confidence, **self._args(
                primary_n=0,
                mainstream_n=0,
                low_n=5,
                contested_claims=3,
                uncovered_aspects=["A"],
                needs_primary=True,
                existing_gap=first.gap_suggestion or "",
            ))
        assert second.confidence == first.confidence
        # Second invocation must not emit a new gap suggestion because the
        # first invocation already chose one (or there wasn't one to emit).
        assert second.gap_suggestion is None
