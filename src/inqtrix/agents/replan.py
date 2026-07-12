"""Phase 7 — the deterministic replan gate (§4).

Decides WHETHER a replan is needed and whether autonomy auto-approves
it; the replan itself reuses the planner. Deltas are ADDITIVE — results
never get discarded (append-only plan versions, M4).
"""

from __future__ import annotations

from inqtrix.agents.control_ports import PlanTaskRecord
from inqtrix.agents.scheduler import TaskOutcome

READ_ONLY_TOOLS = ("web_research", "web_instant", "rag_query", "file_analysis")

INTERNAL_READ_ONLY_TOOLS = ("rag_query", "file_analysis")
"""Tools that touch ONLY the user's own holdings — no web contact.

The E16 amendment (plan M1 S7): in Standard mode (``balanced``) the plan
gate is the web-search consent (tasks carry their verbatim queries), so
a replan may only auto-approve when it introduces NO new web contact —
otherwise new queries would run that the user never saw."""


def evaluate_replan(
    *,
    outcomes: dict[str, TaskOutcome],
    blocking_gap_uncovered: bool,
    sufficiency_coverage: str,
    replan_rounds_used: int,
    max_replan_rounds: int,
) -> bool:
    """Whether a replan should run (the deterministic §4 failure policy).

    A replan is possible only when execution produced a real unresolved
    task outcome (``failed`` or ``insufficient_evidence``). Blocking-gap and
    sufficiency signals then decide whether more work is justified. A clean
    set of successful tasks proceeds directly to synthesis even when the
    advisory sufficiency model is pessimistic. Returns a bare bool — the gate consumes only
    whether to loop; the ``reason``/``auto_approved`` fields the old
    ``ReplanDecision`` carried were never read (the plan-version reason is
    computed separately in ``_node_plan`` from the route), so they are
    dropped rather than kept as dead scoring (Designprinzip 7).
    """
    if replan_rounds_used >= max_replan_rounds:
        return False
    unresolved = any(
        outcome.status in {"failed", "insufficient_evidence"}
        for outcome in outcomes.values()
    )
    if not unresolved:
        return False
    if blocking_gap_uncovered:
        return True
    if sufficiency_coverage == "uncovered":
        return True
    return sufficiency_coverage == "partial"


def autonomy_auto_approves(
    *,
    autonomy: str,
    new_tasks: list[PlanTaskRecord],
) -> bool:
    """E16 replan policy (amended by plan M1 S7): strict never,
    autonomous always, balanced only for <= 2 NEW INTERNAL read-only
    tasks — a replan that adds web queries always re-gates, because the
    approved plan is the user's web-search consent and new queries must
    be seen before they run."""
    if autonomy == "autonomous":
        return True
    if autonomy == "strict":
        return False
    return len(new_tasks) <= 2 and all(
        task.tool_kind in INTERNAL_READ_ONLY_TOOLS for task in new_tasks
    )
