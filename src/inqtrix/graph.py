"""LangGraph state machine construction and entry points.

Provides :func:`run` (production) and :func:`run_test` (testing) as the
two public entry points for executing the research agent pipeline.

The graph topology is:

    classify --> (done? --> answer) | plan
    plan     --> (done? --> answer) | search
    search   --> evaluate
    evaluate --> (done? --> answer) | plan
    answer   --> END
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial
from queue import Queue
from typing import Any, Callable

from langgraph.graph import END, StateGraph

import threading

from inqtrix.exceptions import AgentCancelled, AgentRateLimited
from inqtrix.i18n import t
from inqtrix.providers.base import ProviderContext
from inqtrix.result import ResearchResult, ResearchResultExportOptions
from inqtrix.runtime_logging import log_run_start
from inqtrix.settings import AgentSettings
from inqtrix.state import AgentState, emit_progress, initial_state
from inqtrix.strategies import StrategyContext

log = logging.getLogger("inqtrix")


# ------------------------------------------------------------------ #
# Graph configuration
# ------------------------------------------------------------------ #


@dataclass
class GraphConfig:
    """Declarative description of the agent graph topology.

    *nodes* maps logical node names to callables ``(AgentState) -> AgentState``.
    *edges* is a list of unconditional ``(source, target)`` pairs.
    *conditional_edges* is a list of ``(source, router_fn)`` pairs where the
    router returns the next node name.
    *entry_point* is the first node to execute.
    """

    nodes: dict[str, Callable] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    conditional_edges: list[tuple[str, Callable]] = field(default_factory=list)
    entry_point: str = "classify"


def default_graph_config(
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> GraphConfig:
    """Create the standard pipeline :class:`GraphConfig`.

    Uses :func:`functools.partial` to bind *providers*, *strategies*, and
    *settings* to each node function so LangGraph sees the expected
    ``(AgentState) -> AgentState`` signature.
    """
    from inqtrix.nodes import answer, classify, evaluate, plan, search

    def _bind(fn: Callable, s: AgentState) -> AgentState:
        return fn(s, providers=providers, strategies=strategies, settings=settings)

    return GraphConfig(
        nodes={
            "classify": partial(_bind, classify),
            "plan": partial(_bind, plan),
            "search": partial(_bind, search),
            "evaluate": partial(_bind, evaluate),
            "answer": partial(_bind, answer),
        },
        edges=[
            ("search", "evaluate"),
            ("answer", END),
        ],
        conditional_edges=[
            ("classify", lambda s: "answer" if s["done"] else "plan"),
            ("plan", lambda s: "answer" if s["done"] else "search"),
            ("evaluate", lambda s: "answer" if s["done"] else "plan"),
        ],
        entry_point="classify",
    )


# ------------------------------------------------------------------ #
# Graph compilation
# ------------------------------------------------------------------ #


def build_graph(config: GraphConfig) -> Any:
    """Compile a :class:`~langgraph.graph.StateGraph` from *config*.

    Returns the compiled graph object ready for ``.invoke()``.
    """
    g = StateGraph(AgentState)

    for name, fn in config.nodes.items():
        g.add_node(name, fn)

    g.set_entry_point(config.entry_point)

    for src, dst in config.edges:
        g.add_edge(src, dst)

    for src, router in config.conditional_edges:
        g.add_conditional_edges(src, router)

    return g.compile()


_cached_agent: Any | None = None
_cached_agent_key: tuple | None = None


def get_agent(
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> Any:
    """Build and return the compiled agent graph.

    The compiled graph is cached and reused as long as the same *providers*,
    *strategies*, and *settings* objects are passed (identity check).  This
    avoids the cost of rebuilding all node closures on every request.
    """
    global _cached_agent, _cached_agent_key
    key = (id(providers), id(strategies), id(settings))
    if _cached_agent is not None and _cached_agent_key == key:
        return _cached_agent
    config = default_graph_config(providers, strategies, settings)
    _cached_agent = build_graph(config)
    _cached_agent_key = key
    return _cached_agent


# ------------------------------------------------------------------ #
# Public entry points
# ------------------------------------------------------------------ #


_DIRECT_CHAT_SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Konversations-Assistent ohne Web-Recherche. "
    "Beantworte die Frage klar und präzise auf Basis deines Trainingswissens. "
    "Wenn du eine Information nicht zuverlässig weißt, sage das ehrlich. "
    "Erfinde keine Quellen und gib keine Zitationen an."
)


def _format_direct_chat_prompt(question: str, history: str) -> str:
    """Build the prompt sent to the LLM in skip-search mode.

    History is appended verbatim when present so the provider sees the
    same conversational context the UI already passes to the server.
    """
    if history:
        return (
            f"Bisheriger Gespraechsverlauf:\n{history}\n\n"
            f"Aktuelle Frage:\n{question}"
        )
    return question


def _run_direct_chat(
    question: str,
    history: str,
    *,
    progress_queue: Queue | None,
    providers: ProviderContext,
    settings: AgentSettings,
    cancel_event: threading.Event | None,
) -> dict[str, Any]:
    """Execute the skip-search short-circuit: LLM-only, no graph.

    Emits two progress markers (``chat_only.start`` / ``chat_only.done``)
    so the Streamlit progress expander still has something to show. The
    returned ``result_state`` mimics the shape of the research path so
    the session store and downstream callers behave identically, minus
    the research-specific fields.
    """
    state = initial_state(
        question,
        history,
        progress_queue,
        max_total_seconds=settings.max_total_seconds,
        cancel_event=cancel_event,
    )
    log_run_start(
        question=question,
        history=history,
        prev_session=None,
        providers=providers,
        settings=settings,
        run_mode="run_chat_only",
    )

    emit_progress(state, "chat_only.start")
    t0 = time.monotonic()

    prompt = _format_direct_chat_prompt(question, history)
    try:
        llm_response = providers.llm.complete_with_metadata(
            prompt,
            system=_DIRECT_CHAT_SYSTEM_PROMPT,
            state=state,
            deadline=state.get("deadline"),
        )
    except AgentCancelled as exc:
        log.info("Chat-only run cancelled by client disconnect: %s", exc)
        state["cancelled"] = True
        return {
            "answer": "",
            "usage": {
                "prompt_tokens": state.get("total_prompt_tokens", 0),
                "completion_tokens": state.get("total_completion_tokens", 0),
            },
            "result_state": state,
        }
    except AgentRateLimited as exc:
        emit_progress(state, t(state, "run_aborted", exc=exc))
        log.error("ABBRUCH (chat-only): %s", exc)
        return {
            "answer": (
                f"Die Anfrage wurde abgebrochen: {exc}\n\n"
                "Bitte spaeter erneut versuchen (Tages-Token-Limit erreicht)."
            ),
            "usage": {
                "prompt_tokens": state.get("total_prompt_tokens", 0),
                "completion_tokens": state.get("total_completion_tokens", 0),
            },
            "result_state": state,
        }
    except Exception as exc:
        emit_progress(state, t(state, "chat_mode_failed", exc=exc))
        raise

    answer_text = getattr(llm_response, "text", None) or str(llm_response)
    state["answer"] = answer_text
    state["done"] = True
    emit_progress(state, "chat_only.done")

    elapsed = time.monotonic() - t0
    log.info("Chat-only finished in %.2fs", elapsed)

    return {
        "answer": answer_text,
        "usage": {
            "prompt_tokens": state.get("total_prompt_tokens", 0),
            "completion_tokens": state.get("total_completion_tokens", 0),
        },
        "result_state": state,
    }


def run(
    question: str,
    history: str = "",
    progress_queue: Queue | None = None,
    prev_session: dict[str, Any] | None = None,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Execute the research agent and return answer + usage + state.

    Parameters
    ----------
    question:
        The user question to research.
    history:
        Optional conversation history for context.
    progress_queue:
        Optional :class:`~queue.Queue` receiving ``("progress", msg)`` tuples.
    prev_session:
        Optional session snapshot from a previous round.  If provided the
        state is seeded with prior research results (follow-up support).
    providers:
        The active LLM / search providers.
    strategies:
        The active strategy implementations.
    settings:
        Agent behaviour settings.

    Returns
    -------
    dict with keys ``answer``, ``usage``, ``result_state``.
    """
    if len(question) > settings.max_question_length:
        return {
            "answer": (
                f"Die Frage ist zu lang (max. {settings.max_question_length} Zeichen). "
                "Bitte kuerzer formulieren."
            ),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "result_state": {},
        }

    # "Chat ohne Websuche"-Kurzschluss (ADR-WS-14): wenn der Caller
    # ``skip_search=True`` setzt, umgehen wir den kompletten
    # classify/plan/search/evaluate/answer-Graphen und rufen direkt den
    # LLM-Provider mit Frage + History. Nützlich für UI-Clients, die
    # einen reinen Konversationsmodus anbieten wollen.
    if settings.skip_search:
        return _run_direct_chat(
            question,
            history,
            progress_queue=progress_queue,
            providers=providers,
            settings=settings,
            cancel_event=cancel_event,
        )

    agent = get_agent(providers, strategies, settings)
    state = initial_state(
        question,
        history,
        progress_queue,
        prev_session,
        max_total_seconds=settings.max_total_seconds,
        cancel_event=cancel_event,
    )
    log_run_start(
        question=question,
        history=history,
        prev_session=prev_session,
        providers=providers,
        settings=settings,
        run_mode="run",
    )

    t0 = time.monotonic()
    try:
        result = agent.invoke(state)
    except AgentCancelled as exc:
        # Implicit-cancel-on-disconnect path: the SSE client went away,
        # the streaming layer set the event, and the next node-boundary
        # probe raised. Return a marked-cancelled state instead of
        # propagating; the streaming generator already stopped emitting.
        log.info("Run cancelled by client disconnect: %s", exc)
        state["cancelled"] = True
        return {
            "answer": "",
            "usage": {
                "prompt_tokens": state.get("total_prompt_tokens", 0),
                "completion_tokens": state.get("total_completion_tokens", 0),
            },
            "result_state": state,
        }
    except AgentRateLimited as exc:
        emit_progress(state, t(state, "run_aborted", exc=exc))
        log.error("ABBRUCH: %s", exc)
        return {
            "answer": (
                f"Die Recherche wurde abgebrochen: {exc}\n\n"
                "Bitte spaeter erneut versuchen (Tages-Token-Limit erreicht)."
            ),
            "usage": {
                "prompt_tokens": state.get("total_prompt_tokens", 0),
                "completion_tokens": state.get("total_completion_tokens", 0),
            },
            "result_state": state,
        }
    except Exception as exc:
        emit_progress(state, t(state, "run_failed", exc=exc))
        raise

    elapsed = time.monotonic() - t0
    log.info("Agent finished in %.1fs", elapsed)

    return {
        "answer": result["answer"],
        "usage": {
            "prompt_tokens": result.get("total_prompt_tokens", 0),
            "completion_tokens": result.get("total_completion_tokens", 0),
        },
        "result_state": result,
    }


# ------------------------------------------------------------------ #
# Test entry point
# ------------------------------------------------------------------ #


def _build_test_metrics(
    data: dict[str, Any], elapsed: float, **extra: Any
) -> dict[str, Any]:
    """Build the metrics dict for :func:`run_test` (success and error paths)."""
    iter_logs = data.get("iteration_logs", [])
    eval_logs = [e for e in iter_logs if e.get("node") == "evaluate"]
    return {
        "total_rounds": data.get("round", 0),
        "total_elapsed_s": round(elapsed, 2),
        "final_confidence": data.get("final_confidence", 0),
        "total_queries": len(data.get("queries", [])),
        "total_citations": len(data.get("all_citations", [])),
        "total_context_blocks": len(data.get("context", [])),
        "competing_events_detected": (
            bool(data.get("competing_events", ""))
            or any(bool(e.get("competing_events", "")) for e in eval_logs)
        ),
        "stagnation_detected": any(
            e.get("stagnation_detected", False) for e in eval_logs
        ),
        "total_prompt_tokens": data.get("total_prompt_tokens", 0),
        "total_completion_tokens": data.get("total_completion_tokens", 0),
        "falsification_triggered": bool(data.get("falsification_triggered", False)),
        "evidence_consistency_final": data.get("evidence_consistency", 0),
        "evidence_sufficiency_final": data.get("evidence_sufficiency", 0),
        "source_quality_score_final": data.get("source_quality_score", 0.0),
        "source_tier_counts_final": data.get("source_tier_counts", {}),
        "claim_quality_score_final": data.get("claim_quality_score", 0.0),
        "claim_status_counts_final": data.get("claim_status_counts", {}),
        "consolidated_claims_count_final": len(
            data.get("consolidated_claims", [])
        ),
        "claim_needs_primary_total_final": data.get(
            "claim_needs_primary_total", 0
        ),
        "claim_needs_primary_verified_final": data.get(
            "claim_needs_primary_verified", 0
        ),
        "aspect_coverage_final": data.get("aspect_coverage", 0.0),
        "uncovered_aspects_count_final": len(
            data.get("uncovered_aspects", [])
        ),
        "risk_score": data.get("risk_score", 0),
        "high_risk": bool(data.get("high_risk", False)),
        "utility_scores": data.get("utility_scores", []),
        "utility_stop_triggered": any(
            e.get("utility_stop", False) for e in eval_logs
        ),
        "plateau_stop_triggered": any(
            e.get("plateau_stop", False) for e in eval_logs
        ),
        **extra,
    }


def _build_test_result_export(answer: str, data: dict[str, Any]) -> dict[str, Any]:
    """Project internal state into the public result view for parity surfaces."""
    result = ResearchResult.from_raw(
        {
            "answer": answer,
            "usage": {
                "prompt_tokens": data.get("total_prompt_tokens", 0),
                "completion_tokens": data.get("total_completion_tokens", 0),
            },
            "result_state": data,
        }
    )
    return result.to_export_payload(
        ResearchResultExportOptions(
            include_metrics=False,
        )
    )


def run_test(
    question: str,
    *,
    providers: ProviderContext,
    strategies: StrategyContext,
    settings: AgentSettings,
) -> dict[str, Any]:
    """Execute the research agent in testing mode.

    Returns a structured dict with answer, metrics, and iteration logs.
    Only meaningful when ``settings.testing_mode`` is ``True`` (otherwise
    ``iteration_logs`` will be empty).
    """
    if len(question) > settings.max_question_length:
        return {
            "answer": (
                f"Die Frage ist zu lang (max. {settings.max_question_length} Zeichen)."
            ),
            "metrics": {},
            "iteration_logs": [],
            "top_sources": [],
            "top_claims": [],
        }

    agent = get_agent(providers, strategies, settings)
    state = initial_state(
        question,
        max_total_seconds=settings.max_total_seconds,
    )
    log_run_start(
        question=question,
        history="",
        prev_session=None,
        providers=providers,
        settings=settings,
        run_mode="run_test",
    )

    t0 = time.monotonic()
    try:
        result = agent.invoke(state)
    except AgentRateLimited as exc:
        log.error("ABBRUCH (test): %s", exc)
        elapsed = time.monotonic() - t0
        return {
            **_build_test_result_export(f"ABBRUCH: {exc}", state),
            "metrics": _build_test_metrics(state, elapsed, error=str(exc)),
            "iteration_logs": state.get("iteration_logs", []),
        }

    elapsed = time.monotonic() - t0
    return {
        **_build_test_result_export(result["answer"], result),
        "metrics": _build_test_metrics(result, elapsed),
        "iteration_logs": result.get("iteration_logs", []),
    }
