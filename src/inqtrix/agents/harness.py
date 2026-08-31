"""The ONE deepagents boundary.

Every deepagents import lives in this module: when the library churns
(and it can do so frequently), this file absorbs it. The rest
of the agent runtime calls the seam functions below and can be tested by
monkeypatching them — the trajectory catalog never enters deepagents.

The initial surface provides quarantined file analysis. Large document texts must never
flood the supervisor context (§4 Kontext-Management), so the analysis
runs in a sub-agent whose context holds the document, returning only the
compact :class:`~inqtrix.agents.phase_models.FileAnalysisSummary`.
Further specialized sub-agents remain additive definitions behind this
same boundary.
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
"""deepagents built-ins the kernel never exposes.

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

        from deepagents.middleware.summarization import (
            _DeepAgentsSummarizationMiddleware,
        )

        register_harness_profile(
            f"{KERNEL_MODEL_PROVIDER}:{KERNEL_MODEL_IDENTIFIER}",
            HarnessProfile(
                excluded_tools=KERNEL_EXCLUDED_BUILTIN_TOOLS,
                # Strip the base-stack summarization (class-form exclusion
                # matches the exact type, so the kernel's SUBCLASS in the
                # user slot survives): its trigger never fires before the
                # kernel's ceilings, it is invisible (no inqtrix event),
                # and its offload points at read_file — excluded above.
                excluded_middleware=frozenset(
                    {_DeepAgentsSummarizationMiddleware}
                ),
                general_purpose_subagent=GeneralPurposeSubagentProfile(
                    enabled=False
                ),
            ),
        )
        _kernel_profile_registered = True


_kernel_summarization_cls: type | None = None


def _kernel_summarization_middleware_cls() -> type:
    """The kernel's summarization subclass (lazy: deepagents boundary E10).

    Defined inside the one deepagents-importing module so
    ``inqtrix.agents.kernel.middleware`` stays importable without the
    ``agent`` extra. Cached so the class identity is stable per process
    (the graph cache may build several policy variants).
    """
    global _kernel_summarization_cls
    if _kernel_summarization_cls is not None:
        return _kernel_summarization_cls

    from deepagents.middleware.summarization import (
        _DeepAgentsSummarizationMiddleware,
    )
    from langchain_core.messages import HumanMessage

    from inqtrix.agents.kernel.middleware import (
        current_compaction_todos,
        emit_compaction_event,
        message_text,
        render_todo_state,
    )

    class KernelSummarizationMiddleware(_DeepAgentsSummarizationMiddleware):
        """The kernel's ONE summarization: ledger-grounded and loud.

        Replaces (never duplicates) the deepagents base-stack instance —
        the harness profile excludes the base CLASS and exact-type
        matching preserves this subclass in the user slot. Three
        deviations, each closing a named gap:

        1. **Ledger-grounded offload** — evicted history is appended to
           the run's ``context_archive`` artifact instead of a backend
           file the kernel cannot read back (``read_file`` is excluded);
           the summary embeds a ``read_canvas`` pointer.
        2. **Per-run trigger through the deps ContextVar** — the compiled
           graph is shared across runs and models, so the threshold comes
           from ``kernel_deps().context_trigger_tokens`` at call time
           (0 disables), never from constructor state.
        3. **Visible compaction** — every compaction emits
           ``inqtrix.agent.context.compacted`` plus a deterministic
           narration, and the summary message recites the current
           ``write_todos`` state (goal recitation exactly where the
           prompt-cache prefix breaks anyway).

        The base implementation is non-mutating (``state["messages"]``
        and their ``usage_metadata`` stay intact), so the token-budget
        basis (``_checkpointed_usage``) and replay survive compaction.

        Known limit (accepted): the base ``ContextOverflowError`` branch
        clips oversized tool results against the dummy ``StateBackend``
        and embeds ``read_file`` pointers the kernel cannot follow
        (``read_file`` is excluded from the tool surface). It only
        triggers when a single turn overflows BEFORE the 0.75-fraction
        trigger fires — rare, and the degradation is visible (clipped
        results say so), never silent. Overriding the overflow path is
        deliberately out of scope; the archive covers the normal path.
        """

        @staticmethod
        def _deps() -> Any | None:
            from inqtrix.agents.kernel.deps import kernel_deps

            try:
                return kernel_deps()
            except RuntimeError:
                return None

        def _should_summarize(
            self, messages: list[Any], total_tokens: int
        ) -> bool:
            deps = self._deps()
            if deps is None or deps.context_trigger_tokens <= 0:
                return False
            return total_tokens >= deps.context_trigger_tokens

        def _offload_to_backend(
            self, backend: Any, messages: list[Any]
        ) -> str | None:
            """Archive evicted history in the run artifact, not a file."""
            del backend
            deps = self._deps()
            if deps is None:
                return None
            try:
                # A chained compaction's eviction window contains the
                # PRIOR summary message — the base filters it via
                # _is_summary_message before offloading, so this
                # replacement must too, or every later compaction
                # re-archives the previous summary as ordinary history.
                evicted = [
                    message
                    for message in messages
                    if (
                        getattr(message, "additional_kwargs", None) or {}
                    ).get("lc_source")
                    != "summarization"
                ]
                if not evicted:
                    return None
                rendered = "\n\n".join(
                    f"[{getattr(message, 'type', '?')}] "
                    f"{message_text(message)}"
                    for message in evicted
                )
                # Lossless multi-section archive: the
                # single-section writer TRUNCATED long evictions. The
                # comma-joined ids stay one string for the upstream
                # file-path seam; the pointer builder splits them.
                ids = deps.append_context_archive_chunked(
                    "Komprimierter Verlauf", rendered
                )
                return ",".join(ids) if ids else None
            except Exception as exc:  # noqa: BLE001 — summary must proceed
                log.warning(
                    "Kontext-Archiv-Offload fehlgeschlagen "
                    "(error_type=%s) — die "
                    "Zusammenfassung ersetzt den Verlauf ohne "
                    "Archiv-Zeiger.",
                    type(exc).__name__,
                )
                return None

        def _build_new_messages_with_path(
            self, summary: str, file_path: str | None
        ) -> list[Any]:
            parts = [
                f"Zusammenfassung des bisherigen Verlaufs:\n{summary}"
            ]
            if file_path:
                sections = [
                    part for part in file_path.split(",") if part
                ]
                listed = "; ".join(
                    f"read_canvas(artifact_id='{section}')"
                    for section in sections
                )
                pointer = (
                    "Der aeltere Verlauf liegt als Sektion im "
                    f"Lauf-Archiv: {listed}."
                    if len(sections) <= 1
                    else (
                        f"Der aeltere Verlauf liegt in {len(sections)} "
                        f"Sektionen im Lauf-Archiv: {listed}."
                    )
                )
                deps = self._deps()
                if deps is not None:
                    pointer += (
                        " Alle Archiv-Sektionen: read_canvas("
                        f"artifact_id='{deps.context_archive_prefix}')."
                    )
                parts.append(pointer)
            todos = render_todo_state(current_compaction_todos.get())
            if todos:
                parts.append(f"Aktueller Aufgabenstand:\n{todos}")
            return [
                HumanMessage(
                    content="\n\n".join(parts),
                    # The base's _is_summary_message keys on this kwarg;
                    # without it a SECOND compaction re-archives the
                    # prior summary as ordinary history (duplicates).
                    additional_kwargs={"lc_source": "summarization"},
                )
            ]

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            before = request.state.get("_summarization_event")
            token = current_compaction_todos.set(
                request.state.get("todos")
            )
            try:
                response = super().wrap_model_call(request, handler)
            finally:
                current_compaction_todos.reset(token)
            command = getattr(response, "command", None)
            update = getattr(command, "update", None)
            after = (
                update.get("_summarization_event")
                if isinstance(update, dict)
                else None
            )
            if isinstance(after, dict) and after is not before:
                emit_compaction_event(before, after)
            return response

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            raise RuntimeError(
                "KernelSummarizationMiddleware ist sync-only — der "
                "Kernel laeuft in einem synchronen Worker-Segment."
            )

    _kernel_summarization_cls = KernelSummarizationMiddleware
    return KernelSummarizationMiddleware


def _kernel_summarization_middleware(
    chat_model: Any, *, keep_messages: int
) -> Any:
    """One configured kernel summarization instance (user middleware slot)."""
    from deepagents.backends import StateBackend

    from inqtrix.agents.kernel.middleware import KERNEL_SUMMARY_PROMPT

    cls = _kernel_summarization_middleware_cls()
    return cls(
        chat_model,
        # The backend is unused (the override archives into the run
        # artifact), but the base constructor requires one.
        backend=StateBackend(),
        trigger=None,
        keep=("messages", keep_messages),
        summary_prompt=KERNEL_SUMMARY_PROMPT,
        # The LangChain default (4000, strategy "last") would let the
        # summary model see only the newest sliver of the evicted
        # history — the citation-token preservation the summary prompt
        # demands needs the WHOLE eviction (the archive caps per
        # SECTION, visibly; cost is bounded by the compaction trigger
        # itself).
        trim_tokens_to_summarize=None,
    )


def build_kernel_agent(
    chat_model: Any,
    *,
    tools: "Sequence[Any]",
    system_prompt: str,
    interrupt_on: "Mapping[str, Any] | None" = None,
    checkpointer: Any = None,
    max_tool_calls: int | None = None,
    context_keep_messages: int | None = None,
) -> Any:
    """Assemble the kernel loop — the ONE ``create_deep_agent`` call (E10).

    Args:
        chat_model: Tool-calling bridge from
            :func:`inqtrix.agents.kernel.chat_bridge.build_tool_chat_model`
            (its declared ``inqtrix:kernel`` identity selects the kernel
            harness profile registered here).
        tools: Sync LangChain tools; the Inqtrix tool surface. Anonymous
            sub-agent fan-out stays off — fan-out happens via budgeted
            child runs only.
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
        context_keep_messages: Newest messages kept verbatim per
            compaction; enables the kernel summarization middleware
            (the run-time trigger comes from the deps ContextVar).
            ``None`` builds an agent without compaction — only for
            low-level harness tests, mirroring ``max_tool_calls``.

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

    from inqtrix.agents.kernel.middleware import (
        KernelChildBatchGuardMiddleware,
        KernelModelTurnMiddleware,
        KernelSufficiencyMiddleware,
    )

    middleware: list[Any] = [
        # Outermost wrap so the model-turn activity brackets EVERYTHING the
        # model boundary does (incl. a compaction pass). wrap_model_call
        # only — a node-creating hook would re-price the pinned supersteps.
        KernelModelTurnMiddleware(),
        KernelSkillInputMiddleware(),
        # The advisory sufficiency nudge is always compiled
        # in — the runtime flag short-circuits inside the hook, so the
        # per-turn super-step price stays constant across deployments
        # (the recursion ceilings are priced against it).
        KernelSufficiencyMiddleware(),
        # Pre-dispatch: >1 child-tool calls in ONE model turn would each
        # submit-then-interrupt and trip the multi-interrupt guard AFTER
        # the submissions (orphaned children) — the guard rewrites the
        # batch into corrective ToolMessages before anything dispatches.
        KernelChildBatchGuardMiddleware(),
    ]
    if max_tool_calls is not None:
        from inqtrix.agents.kernel.middleware import (
            KernelToolBudgetMiddleware,
        )

        middleware.append(KernelToolBudgetMiddleware(max_tool_calls))
    if context_keep_messages is not None:
        middleware.append(
            _kernel_summarization_middleware(
                chat_model, keep_messages=context_keep_messages
            )
        )
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
