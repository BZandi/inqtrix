"""The ONE deepagents boundary (plan decision E10).

Every deepagents import lives in this module: when the library churns
(13 patches in 6 weeks at planning time), this file absorbs it. The rest
of the agent runtime calls the seam functions below and can be tested by
monkeypatching them — the trajectory catalog never enters deepagents.

v1 surface: quarantined file analysis. Large document texts must never
flood the supervisor context (§4 Kontext-Management), so the analysis
runs in a sub-agent whose context holds the document, returning only the
compact :class:`~inqtrix.agents.phase_models.FileAnalysisSummary`.
Further specialized sub-agents (evidence reviewer, data analyst) are
additive definitions here — no architecture change.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from inqtrix.agents.patterns._structured import structured_call
from inqtrix.agents.phase_models import FileAnalysisSummary
from inqtrix.agents.prompts import build_agent_file_analysis_prompt

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from inqtrix.providers.base import LLMProvider

log = logging.getLogger("inqtrix")

_QUARANTINE_CONTENT_LIMIT = 60_000
"""Character cap per quarantined document — beyond it the sub-agent path
splits/summarizes; the direct path truncates VISIBLY (marker in text)."""

KERNEL_EXCLUDED_BUILTIN_TOOLS = frozenset(
    {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "delete",
        "execute",
    }
)
"""deepagents built-ins the kernel never exposes (plan M2 §2.2).

The kernel's world is Inqtrix services (knowledge, web, canvas, child
runs) — a filesystem/sandbox surface would bypass every capability
check. ``write_todos`` deliberately stays: it feeds the task-list
display. ``delete``/``execute`` only exist with a sandbox backend but
excluding them is free future-proofing; the contract test asserts the
REMAINING built-in set, so an upstream addition fails loudly."""

_kernel_profile_lock = threading.Lock()
_kernel_profile_registered = False


def _register_kernel_harness_profile() -> None:
    """Register the kernel's harness profile once per process.

    deepagents resolves profiles via the model's ``provider:identifier``
    key — the bridge model declares ``inqtrix:kernel`` (see
    :mod:`inqtrix.agents.kernel.chat_bridge`). Registration is global and
    additive-merge, so repeating the identical profile would be harmless;
    the guard exists to keep the registry deterministic under concurrent
    worker threads.
    """
    global _kernel_profile_registered
    with _kernel_profile_lock:
        if _kernel_profile_registered:
            return
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            register_harness_profile,
        )

        from inqtrix.agents.kernel.chat_bridge import (
            KERNEL_MODEL_IDENTIFIER,
            KERNEL_MODEL_PROVIDER,
        )

        register_harness_profile(
            f"{KERNEL_MODEL_PROVIDER}:{KERNEL_MODEL_IDENTIFIER}",
            HarnessProfile(
                excluded_tools=KERNEL_EXCLUDED_BUILTIN_TOOLS,
                general_purpose_subagent=GeneralPurposeSubagentProfile(
                    enabled=False
                ),
            ),
        )
        _kernel_profile_registered = True


def build_kernel_agent(
    chat_model: Any,
    *,
    tools: "Sequence[Any]",
    system_prompt: str,
    interrupt_on: "Mapping[str, Any] | None" = None,
    checkpointer: Any = None,
    max_tool_calls: int | None = None,
) -> Any:
    """Assemble the kernel loop — the ONE ``create_deep_agent`` call (E10).

    Args:
        chat_model: Tool-calling bridge from
            :func:`inqtrix.agents.kernel.chat_bridge.build_tool_chat_model`
            (its declared ``inqtrix:kernel`` identity selects the kernel
            harness profile registered here).
        tools: Sync LangChain tools; the Inqtrix tool surface. Anonymous
            sub-agent fan-out stays off — fan-out happens via budgeted
            child runs only (plan M2 §2.2).
        system_prompt: Full German kernel system prompt.
        interrupt_on: Tool-name -> HITL config mapping for policy gates;
            ``None``/empty omits the HITL middleware entirely (auto
            policy). Payload/resume shapes are frozen by the contract
            tests (the deepagents upgrade gate).
        checkpointer: LangGraph checkpointer; REQUIRED for park/resume
            (``thread_id=run_id``). ``None`` is only for throwaway
            in-test agents.
        max_tool_calls: Run-wide kernel tool-call ceiling. Counts are
            recovered from checkpointed AI messages, so park/resume is
            cumulative and idempotent. ``None`` is only for low-level
            harness tests that do not exercise the production kernel.

    Returns:
        The compiled deepagents graph (sync ``stream``/``invoke`` only —
        never ``ainvoke``; async paths are the documented upstream bug
        surface on LiteLLM/Azure).

    Raises:
        RuntimeError: deepagents is not installed (missing ``agent``
            extra).
    """
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "Der Agent-Kernel braucht das 'agent'-Extra "
            "(uv sync --extra agent)."
        ) from exc
    _register_kernel_harness_profile()
    from inqtrix.agents.kernel.middleware import KernelSkillInputMiddleware

    middleware: list[Any] = [KernelSkillInputMiddleware()]
    if max_tool_calls is not None:
        from inqtrix.agents.kernel.middleware import (
            KernelToolBudgetMiddleware,
        )

        middleware.append(KernelToolBudgetMiddleware(max_tool_calls))
    return create_deep_agent(
        model=chat_model,
        tools=list(tools),
        system_prompt=system_prompt,
        interrupt_on=dict(interrupt_on) if interrupt_on else None,
        checkpointer=checkpointer,
        middleware=middleware,
    )


def deepagents_available() -> bool:
    """Whether the optional ``agent`` extra's harness import works."""
    try:
        import deepagents  # noqa: F401
    except ImportError:
        return False
    return True


def run_quarantined_file_analysis(
    llm: "LLMProvider",
    *,
    objective: str,
    content: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
    timeout: float = 240.0,
) -> tuple[FileAnalysisSummary | None, dict[str, int]]:
    """Analyze one document WITHOUT letting its text reach the caller.

    Returns ``(summary, usage)``; ``summary`` is ``None`` when the model
    reply never validated (the loud structured fallback — the caller
    marks the task failed instead of inventing findings).

    The deepagents sub-agent path is used when the library is installed
    AND the content is large enough to benefit from an iterating agent
    with a file workspace; the compact path is ONE structured call over
    the (visibly truncated) content. Both return the same model, so the
    caller cannot tell them apart — quarantine is the contract, not the
    mechanism.
    """
    if len(content) > _QUARANTINE_CONTENT_LIMIT and deepagents_available():
        try:
            return _deepagents_analysis(
                llm,
                objective=objective,
                content=content,
                model=model,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )
        except Exception:  # noqa: BLE001 — degrade VISIBLY to the direct path
            log.warning(
                "deepagents-Quarantaene fehlgeschlagen — Datei-Analyse "
                "laeuft ueber den direkten Pfad (gekuerzt).",
                exc_info=True,
            )
    excerpt = content[:_QUARANTINE_CONTENT_LIMIT]
    if len(content) > _QUARANTINE_CONTENT_LIMIT:
        excerpt += "\n\n[... Inhalt fuer die Analyse gekuerzt ...]"
    outcome = structured_call(
        llm,
        prompt=build_agent_file_analysis_prompt(objective, excerpt),
        model_cls=FileAnalysisSummary,
        node="agent_file_analysis",
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    value = outcome.value
    assert value is None or isinstance(value, FileAnalysisSummary)
    return value, outcome.usage


def _deepagents_analysis(
    llm: "LLMProvider",
    *,
    objective: str,
    content: str,
    model: str | None,
    reasoning_effort: str | None,
    timeout: float,
) -> tuple[FileAnalysisSummary | None, dict[str, int]]:
    """Sub-agent analysis over a file workspace (large documents).

    The document lands in the deepagents in-memory filesystem; the
    sub-agent reads it in slices and answers compactly. The final answer
    is validated through the SAME structured model as the direct path.
    """
    from deepagents import create_deep_agent

    from inqtrix.agents.model_bridge import build_chat_model

    chat_model = build_chat_model(
        llm,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        node="agent_file_analysis",
    )
    agent = create_deep_agent(
        model=chat_model,
        tools=[],
        system_prompt=(
            "Du analysierst das Dokument unter files['/dokument.md'] fuer "
            "das genannte Ziel. Antworte kompakt (max. 300 Woerter) und "
            "liste zitierwuerdige woertliche Passagen."
        ),
    )
    result: dict[str, Any] = agent.invoke(
        {
            "messages": [{"role": "user", "content": objective}],
            "files": {"/dokument.md": content},
        }
    )
    messages = result.get("messages") or []
    final_text = ""
    if messages:
        last = messages[-1]
        final_text = getattr(last, "content", None) or (
            last.get("content", "") if isinstance(last, dict) else ""
        )
    outcome = structured_call(
        llm,
        prompt=(
            "Forme diese Analyse in das geforderte JSON um:\n\n"
            f"{final_text}"
        ),
        model_cls=FileAnalysisSummary,
        node="agent_file_analysis",
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    value = outcome.value
    assert value is None or isinstance(value, FileAnalysisSummary)
    return value, outcome.usage
