"""Kernel tool-gating policy per autonomy mode.

THE one mapping from the run's autonomy to the deepagents
``interrupt_on`` configuration — the algorithm compiles one graph
variant per policy (interrupt_on is compile-time, the policy is fixed
per run), tools never check permissions themselves.

Wire vocabulary stays the E16 three-mode set (strict/balanced/
autonomous); the UI presents Standard(balanced)/Auto(autonomous). The
plan-table semantics:

======================  ==================================  ==========
Tool                    Standard (balanced)                 Auto
======================  ==================================  ==========
ask_user                free                                free
web_instant             GATED — query verbatim in args      free
search_project_...      gated only UN-scoped                free
read_project_document   free                                free
======================  ==================================  ==========

``strict`` gates every capability tool (the enterprise mode: nothing
external or internal-broad runs unreviewed). Write-effect tools
(``propose_editor_patch``, M2 step 7) are ALWAYS gated in every mode —
they belong to :data:`ALWAYS_GATED_TOOLS`, not to a mode table.
Canvas reads and writes are gated only in ``strict``; Standard and Auto
keep the canvas itself as their review surface.

The config dicts are LangChain ``InterruptOnConfig`` TypedDicts (plain
dicts at runtime — no langchain import needed here). Allowed decisions
mirror our ``APPROVAL_DECISIONS`` (approve/edit/reject); ``respond``
stays unexposed.
"""

from __future__ import annotations

from typing import Any

GATE_DECISIONS = ["approve", "edit", "reject"]
"""HITL decisions the platform exposes — 1:1 our approval verbs."""

ALWAYS_GATED_TOOLS: tuple[str, ...] = ("propose_editor_patch",)
"""Write-effect tools gated in EVERY mode (E14: patches are never
autonomous — the user's document is the one surface the agent must not
touch unreviewed). Kept separate from the mode table so a new autonomy
mode can never accidentally free a write path. ``write_canvas`` is
deliberately NOT here: it stays free in Standard/Auto and is gated only
by the explicit ``strict`` table."""


def _gate(when: Any = None, *, user_conditional: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {"allowed_decisions": list(GATE_DECISIONS)}
    if when is not None:
        config["when"] = when
    if user_conditional:
        # Published-contract classification only (HITL ignores unknown
        # keys): TRUE means the predicate leaves a user-triggerable path
        # UNGATED (scoped knowledge search). The child-tool predicate is
        # NOT user-conditional — it merely skips a doomed multi-child
        # batch the guard voids, so every real delegation still gates and
        # the tool stays in ``kernel_gated_tools``.
        config["user_conditional"] = True
    return config


def _unscoped_knowledge_search(request: Any) -> bool:
    """Gate predicate: only an UN-scoped project-wide search interrupts.

    A search the user already scoped to explicit collections is the
    plan-approved shape of internal retrieval; the broad sweep over
    everything visible is what Standard asks about first.
    """
    args = getattr(request, "tool_call", {}).get("args", {}) or {}
    return not args.get("collection_ids")


def _single_child_dispatch(request: Any) -> bool:
    """Gate predicate on every child tool: gate only a SINGLE dispatch.

    Two or more child-tool calls in one model turn are voided pre-
    dispatch by ``KernelChildBatchGuardMiddleware`` (corrective
    ToolMessages, nothing submits). Gating them anyway would park an
    approval whose actions can never execute — an approve becomes a
    lie (the guard refuses right after), and a reject double-answers
    the calls (HITL keeps the call on the AIMessage and answers it;
    the guard, running after HITL, would answer again — duplicate
    ToolMessages per call id that providers reject). Skipping the gate
    keeps exactly one authority per call: the guard for the doomed
    batch, HITL for the single dispatch.
    """
    from inqtrix.agents.kernel.middleware import (
        last_turn_child_calls,
        turn_has_interrupting_call,
    )

    state = getattr(request, "state", None) or {}
    last_ai, child_calls = last_turn_child_calls(
        state.get("messages") or []
    )
    if turn_has_interrupting_call(last_ai):
        # ask_user + child turns are voided by the batch guard —
        # gating the doomed child would double-answer it (see the
        # guard's docstring for the single-authority rule).
        return False
    return len(child_calls) <= 1


def interrupt_config_for(autonomy: str) -> dict[str, Any] | None:
    """The ``interrupt_on`` mapping of one policy variant.

    Args:
        autonomy: Wire autonomy mode (``strict``/``balanced``/
            ``autonomous``); unknown values fail loudly — a typo must
            never silently run ungated.

    Returns:
        Tool-name -> InterruptOnConfig mapping, or ``None`` when no
        tool gates (HITL middleware entirely omitted).
    """
    if autonomy == "autonomous":
        gated: dict[str, Any] = {}
    elif autonomy == "balanced":
        gated = {
            "web_instant": _gate(),
            "search_project_knowledge": _gate(
                when=_unscoped_knowledge_search, user_conditional=True
            ),
            "run_web_research": _gate(when=_single_child_dispatch),
            "run_deep_mission": _gate(when=_single_child_dispatch),
            "delegate_batch": _gate(when=_single_child_dispatch),
            "load_skill": _gate(),
        }
    elif autonomy == "strict":
        gated = {
            "web_instant": _gate(),
            "search_project_knowledge": _gate(),
            "read_project_document": _gate(),
            "read_canvas": _gate(),
            "write_canvas": _gate(),
            "run_web_research": _gate(when=_single_child_dispatch),
            "run_deep_mission": _gate(when=_single_child_dispatch),
            "delegate_batch": _gate(when=_single_child_dispatch),
            "load_skill": _gate(),
        }
    else:
        raise ValueError(f"Unbekannter Autonomie-Modus: {autonomy!r}")
    for tool_name in ALWAYS_GATED_TOOLS:
        gated[tool_name] = _gate()
    return gated or None
