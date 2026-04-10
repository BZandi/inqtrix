"""Agent state definition and initialization."""

from __future__ import annotations

import copy
import time
from queue import Queue
from typing import Any, TypedDict

from inqtrix.constants import MAX_TOTAL_SECONDS
from inqtrix.runtime_logging import log_iteration_entry


class AgentState(TypedDict):
    """Full state for a single research agent run."""

    question: str
    history: str
    language: str
    search_language: str
    recency: str
    query_type: str
    context: list[str]
    queries: list[str]
    sub_questions: list[str]
    required_aspects: list[str]
    uncovered_aspects: list[str]
    aspect_coverage: float
    related_questions: list[str]
    all_citations: list[str]
    source_tier_counts: dict[str, int]
    source_quality_score: float
    claim_ledger: list[dict[str, Any]]
    consolidated_claims: list[dict[str, Any]]
    claim_status_counts: dict[str, int]
    claim_quality_score: float
    claim_needs_primary_total: int
    claim_needs_primary_verified: int
    search_offset: int
    gaps: str
    risk_score: int
    high_risk: bool
    round: int
    done: bool
    answer: str
    deadline: float
    progress: Queue | None
    start_time: float
    final_confidence: int
    competing_events: str
    prev_competing_events: str
    falsification_triggered: bool
    evidence_consistency: int
    evidence_sufficiency: int
    utility_scores: list[float]
    prev_citation_count: int
    prev_evidence_sufficiency: int
    _conf_stable_rounds: int
    _is_followup: bool
    _prev_question: str
    _prev_answer: str
    iteration_logs: list[dict[str, Any]]
    total_prompt_tokens: int
    total_completion_tokens: int


# Fields carried over from a previous session for follow-up questions
SESSION_CARRY_FIELDS: tuple[str, ...] = (
    "all_citations",
    "context",
    "consolidated_claims",
    "claim_ledger",
    "required_aspects",
    "uncovered_aspects",
    "aspect_coverage",
    "sub_questions",
    "language",
    "search_language",
    "final_confidence",
    "source_tier_counts",
    "source_quality_score",
    "claim_status_counts",
    "claim_quality_score",
    "queries",
    "query_type",
    "recency",
    "risk_score",
    "high_risk",
)


def seed_from_session(state: dict[str, Any], prev: dict[str, Any]) -> None:
    """Seed a new AgentState with data from a previous session.

    Called when a follow-up question is detected. Carries over research
    results so the agent doesn't start from scratch.
    """
    for fld in SESSION_CARRY_FIELDS:
        if fld in prev:
            val = prev[fld]
            if isinstance(val, (list, dict)):
                state[fld] = copy.deepcopy(val)
            else:
                state[fld] = val
    state["_is_followup"] = True
    state["_prev_question"] = prev.get("last_question", "")
    state["_prev_answer"] = prev.get("last_answer", "")


def initial_state(
    question: str,
    history: str = "",
    progress_queue: Queue | None = None,
    prev_session: dict[str, Any] | None = None,
    *,
    max_total_seconds: int = MAX_TOTAL_SECONDS,
) -> dict[str, Any]:
    """Create the initial AgentState for a run.

    If prev_session is provided, research data from the previous session
    is carried over (follow-up support).
    """
    deadline = time.monotonic() + max_total_seconds
    state: dict[str, Any] = {
        "question": question,
        "history": history,
        "language": "",
        "search_language": "",
        "recency": "",
        "query_type": "general",
        "context": [],
        "queries": [],
        "sub_questions": [],
        "required_aspects": [],
        "uncovered_aspects": [],
        "aspect_coverage": 0.0,
        "related_questions": [],
        "all_citations": [],
        "source_tier_counts": {"primary": 0, "mainstream": 0, "stakeholder": 0, "unknown": 0, "low": 0},
        "source_quality_score": 0.0,
        "claim_ledger": [],
        "consolidated_claims": [],
        "claim_status_counts": {"verified": 0, "contested": 0, "unverified": 0},
        "claim_quality_score": 0.0,
        "claim_needs_primary_total": 0,
        "claim_needs_primary_verified": 0,
        "search_offset": 0,
        "gaps": "",
        "risk_score": 0,
        "high_risk": False,
        "round": 0,
        "done": False,
        "answer": "",
        "deadline": deadline,
        "progress": progress_queue,
        "start_time": time.monotonic(),
        "final_confidence": 0,
        "competing_events": "",
        "prev_competing_events": "",
        "falsification_triggered": False,
        "evidence_consistency": 0,
        "evidence_sufficiency": 0,
        "utility_scores": [],
        "prev_citation_count": 0,
        "prev_evidence_sufficiency": 0,
        "_conf_stable_rounds": 0,
        "_is_followup": False,
        "_prev_question": "",
        "_prev_answer": "",
        "iteration_logs": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
    }
    if prev_session is not None:
        seed_from_session(state, prev_session)
    return state


def emit_progress(s: dict, message: str) -> None:
    """Send a progress update to the stream."""
    q = s.get("progress")
    if q is not None:
        q.put(("progress", message))


def append_iteration_log(s: dict, entry: dict[str, Any], *, testing_mode: bool = False) -> None:
    """Add an iteration log entry and mirror it into debug logs."""
    materialized_entry = copy.deepcopy(entry)
    if testing_mode:
        s["iteration_logs"].append(materialized_entry)
    log_iteration_entry(materialized_entry)


def track_tokens(s: dict, response: Any) -> None:
    """Count token usage from an API response."""
    if hasattr(response, "usage") and response.usage:
        s["total_prompt_tokens"] += getattr(response.usage, "prompt_tokens", 0) or 0
        s["total_completion_tokens"] += getattr(response.usage, "completion_tokens", 0) or 0
        return

    prompt_tokens = getattr(response, "prompt_tokens", None)
    completion_tokens = getattr(response, "completion_tokens", None)
    if prompt_tokens is not None or completion_tokens is not None:
        s["total_prompt_tokens"] += int(prompt_tokens or 0)
        s["total_completion_tokens"] += int(completion_tokens or 0)
