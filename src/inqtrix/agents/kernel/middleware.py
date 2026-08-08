"""Inqtrix middleware enforcing kernel-owned execution contracts."""

from __future__ import annotations

import contextvars
import hashlib
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage

log = logging.getLogger("inqtrix")

TOOL_LIMIT_REACHED_EVENT = "inqtrix.agent.tool_limit.reached"
SKILL_INPUTS_RESOLVED_MARKER = "[skill_inputs_resolved]"


def _decision_answer(
    question: dict[str, Any], decision: dict[str, Any]
) -> str:
    entry = (decision.get("answers") or {}).get(question.get("id", "")) or {}
    labels = {
        option["id"]: option["label"]
        for option in question.get("options", [])
    }
    parts = [
        labels[item]
        for item in entry.get("option_ids", [])
        if item in labels
    ]
    text = str(entry.get("text") or "").strip()
    if text:
        parts.append(text)
    if parts:
        return ", ".join(parts)
    return str(decision.get("answer") or "").strip()


def resolve_kernel_skill_inputs(
    skill: Any,
    *,
    clarification_scope: str,
) -> dict[str, str]:
    """Resolve required skill points before activating its instructions.

    The same resolver serves attached skills and dynamic ``load_skill``.
    Missing points become deterministic clarification rows and LangGraph
    interrupts; the skill remains inactive until every batch is answered.
    """
    from langgraph.types import interrupt

    from inqtrix.agents.clarification import build_clarification, sanitize_questions
    from inqtrix.agents.control_ports import ClarificationNotFound
    from inqtrix.agents.kernel.deps import kernel_deps, run_coro
    from inqtrix.agents.phase_models import (
        ClarificationOptionModel,
        ClarificationQuestionModel,
    )
    from inqtrix.agents.skills_runtime import (
        check_skill_points,
        skill_point_key,
        unanswered_required_points,
    )
    from inqtrix.model_routing import (
        describe_resolution,
        describe_unresolved_resolution,
    )

    deps = kernel_deps()
    provider_models = getattr(deps.llm, "models", None)
    resolution = (
        describe_unresolved_resolution("agent_skill_point_check", "")
        if provider_models is None
        else describe_resolution(
            "agent_skill_point_check",
            provider_models,
            "",
            requested_model="",
            requested_effort="",
        )
    )
    deps.emit("inqtrix.node.model_resolution", resolution)
    answers, usage = check_skill_points(
        deps.llm,
        skill=skill,
        question=deps.question,
        history=deps.session_history,
        model=resolution.get("model") or None,
        reasoning_effort=resolution.get("effort") or None,
        timeout=deps.timeout,
    )
    deps.book_usage(
        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    )
    missing = unanswered_required_points(skill, answers)
    for batch_index in range(0, len(missing), 3):
        points = missing[batch_index : batch_index + 3]
        questions = sanitize_questions(
            [
                ClarificationQuestionModel(
                    prompt=str(point.get("question") or ""),
                    options=[
                        ClarificationOptionModel(
                            label=str(option.get("label") or ""),
                            description=str(option.get("description") or ""),
                        )
                        for option in point.get("options", [])
                    ],
                    multi_select=bool(point.get("multi_select")),
                )
                for point in points
            ]
        )
        for question, point in zip(questions, points, strict=True):
            question["skill_key"] = skill_point_key(point)
        scope_hash = hashlib.sha1(
            clarification_scope.encode("utf-8")
        ).hexdigest()[:10]
        clarification_id = (
            f"clr_{deps.run_id[-12:]}_skill_{scope_hash}_{batch_index // 3}"
        )
        try:
            record = run_coro(
                deps.control.get_clarification(deps.run_id, clarification_id)
            )
        except ClarificationNotFound:
            record = run_coro(
                deps.control.create_clarification(
                    build_clarification(
                        deps.run_id,
                        questions=questions,
                        clarification_id=clarification_id,
                    )
                )
            )
            deps.emit(
                "inqtrix.agent.clarification.requested",
                {
                    "clarification_id": record.clarification_id,
                    "question": record.question,
                    "options": [dict(item) for item in record.options],
                    "question_count": len(record.questions),
                },
            )
        if record.status == "answered":
            decision = {
                "answer": record.answer,
                "option_id": record.option_id,
                "answers": dict(record.answers),
            }
        else:
            decision = interrupt(
                {"kind": "clarification", "id": clarification_id}
            )
        for question in record.questions:
            value = _decision_answer(dict(question), dict(decision or {}))
            if value:
                answers[str(question.get("skill_key") or "")] = value
    return answers


SKILL_INPUTS_RESOLVED_FLAG = "inqtrix_skill_inputs_resolved"
"""``additional_kwargs`` key marking the middleware-authored skill block.

The TRUST anchor (F3 hardening): user input reaches the transcript only
as plain content strings, so a user question containing the literal
marker text can neither suppress skill resolution nor smuggle a block
past the deep verifier — both now key on this server-set message
attribute, never on a content substring."""


class KernelSkillInputMiddleware(AgentMiddleware):
    """Gate attached skills and inject only fully resolved skill blocks."""

    def before_model(
        self, state: dict[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        del runtime
        messages = state.get("messages") or []
        # Idempotence anchors are SERVER-SET metadata only (F3): the
        # middleware's own flagged HumanMessage, or a load_skill
        # ToolMessage (dynamic activation resolves inputs inside the
        # tool — re-resolving here would re-ask under the attached
        # scope). User content containing the marker text is inert.
        if any(
            (getattr(message, "additional_kwargs", None) or {}).get(
                SKILL_INPUTS_RESOLVED_FLAG
            )
            or (
                type(message).__name__ == "ToolMessage"
                and getattr(message, "name", "") == "load_skill"
            )
            for message in messages
        ):
            return None
        from inqtrix.agents.kernel.deps import kernel_deps
        from inqtrix.agents.skills_runtime import build_skills_block

        try:
            deps = kernel_deps()
        except RuntimeError:
            # Low-level harness contract tests intentionally assemble a
            # throwaway graph without per-run deps.
            return None
        if not deps.skills:
            return None
        resolved: dict[str, dict[str, str]] = {}
        for skill in deps.skills:
            resolved[skill.id] = resolve_kernel_skill_inputs(
                skill,
                clarification_scope=f"attached:{skill.id}",
            )
        deps.skill_answers.update(resolved)
        block = build_skills_block(deps.skills, resolved)
        return {
            "messages": [
                HumanMessage(
                    # The marker stays as human-readable prose; the FLAG
                    # is the machine-trust anchor (checkpointed with the
                    # message, unforgeable through user content).
                    content=f"{SKILL_INPUTS_RESOLVED_MARKER}\n{block}",
                    additional_kwargs={SKILL_INPUTS_RESOLVED_FLAG: True},
                )
            ]
        }


SUFFICIENCY_NUDGE_FLAG = "inqtrix_sufficiency_nudge"
"""``additional_kwargs`` key of one advisory sufficiency nudge message.

Carries the evidence state (successful source-tool count) the verdict
was formed over. Both idempotency anchors key on it: the same state is
never judged twice (park/resume replays included), and the flagged
messages ARE the run-wide judgement count — checkpointed with the
transcript, unforgeable through user content (the F3 pattern of
``SKILL_INPUTS_RESOLVED_FLAG``)."""


class KernelSufficiencyMiddleware(AgentMiddleware):
    """Advisory evidence-sufficiency nudge before the next model turn.

    After enough successful source-tool calls (adaptive threshold, deep
    raises it), one cheap fast-tier judgement summarizes coverage and a
    flagged ``HumanMessage`` advises the model to either answer now or
    search ONLY for the named gaps. Advisory by design: the hard stops
    stay the tool budget and the recursion ceiling. The nudge can never
    become the final answer (``_final_answer`` accepts only AI
    messages) and runs ``before_model``, so it never breaks a pending
    tool call. Simple runs below the threshold NEVER pay a judge call.
    """

    def before_model(
        self, state: dict[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        del runtime
        from inqtrix.agents.kernel.deps import kernel_deps

        try:
            deps = kernel_deps()
        except RuntimeError:
            # Low-level harness contract tests intentionally assemble a
            # throwaway graph without per-run deps.
            return None
        platform = deps.platform
        if not platform.kernel_sufficiency_gate or deps.tier == "schnell":
            return None
        uses = deps.tool_use_counts.get("web", 0) + deps.tool_use_counts.get(
            "knowledge", 0
        )
        threshold = (
            platform.kernel_sufficiency_min_tool_calls_deep
            if deps.depth == "deep"
            else platform.kernel_sufficiency_min_tool_calls
        )
        if uses < threshold:
            return None
        judged_states = [
            kwargs.get(SUFFICIENCY_NUDGE_FLAG)
            for message in (state.get("messages") or [])
            if (
                kwargs := getattr(message, "additional_kwargs", None) or {}
            ).get(SUFFICIENCY_NUDGE_FLAG)
            is not None
        ]
        if len(judged_states) >= platform.kernel_sufficiency_max_judgements:
            return None
        if uses in judged_states:
            # This evidence state was already judged — a park/resume
            # replay or a turn without new source evidence.
            return None
        from inqtrix.agents.kernel.cognition import (
            SUFFICIENCY_JUDGED_EVENT,
            judge_kernel_sufficiency,
        )
        from inqtrix.exceptions import AgentCancelled

        from inqtrix.observability.otel import operation_span

        try:
            # Real span around the judge LLM call: its generation child
            # nests here, and the verdict marker lands on the span.
            with operation_span(
                "kernel.judge", {"inqtrix.step": "sufficiency_judge"}
            ) as judge_span:
                outcome = judge_kernel_sufficiency(deps)
                if judge_span is not None:
                    judge_span.set_attribute(
                        "inqtrix.judge.marker",
                        str(getattr(outcome, "marker", "") or ""),
                    )
        except AgentCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — advisory, never fatal
            # A hard provider failure must not fail a run the judge only
            # ADVISES: visible in the event stream, then unadvised on.
            log.warning(
                "Suffizienz-Judge fehlgeschlagen (error_type=%s) — der Lauf "
                "laeuft unberaten weiter.",
                type(exc).__name__,
            )
            deps.emit(
                SUFFICIENCY_JUDGED_EVENT,
                {"marker": "error", "nudge": False, "tool_uses": uses},
            )
            return None
        verdict = outcome.value
        if verdict is None:
            # No-silent-fallback: the failed judgement stays visible in
            # the event stream; the loop continues UNADVISED (budget and
            # recursion remain the hard stops).
            deps.emit(
                SUFFICIENCY_JUDGED_EVENT,
                {"marker": outcome.marker, "nudge": False, "tool_uses": uses},
            )
            return None
        coverage = str(getattr(verdict, "coverage", "") or "")
        missing = [
            str(item) for item in (getattr(verdict, "missing", None) or [])
        ]
        deps.emit(
            SUFFICIENCY_JUDGED_EVENT,
            {
                "marker": outcome.marker,
                "nudge": True,
                "tool_uses": uses,
                "coverage": coverage,
                "missing": missing,
            },
        )
        if coverage == "covered":
            text = (
                "Zwischenstand Beleglage: ausreichend fuer den Auftrag. "
                "Beende die Recherche und formuliere jetzt die Antwort "
                "auf Basis der vorliegenden Belege."
            )
        else:
            gaps = "; ".join(missing) if missing else "keine benannt"
            text = (
                "Zwischenstand Beleglage: noch unvollstaendig "
                f"(fehlend: {gaps}). Suche gezielt NUR zu diesen "
                "Luecken oder antworte jetzt und benenne die Luecken "
                "sichtbar in der Antwort."
            )
        return {
            "messages": [
                HumanMessage(
                    content=text,
                    additional_kwargs={SUFFICIENCY_NUDGE_FLAG: uses},
                )
            ]
        }


class KernelToolBudgetExceeded(RuntimeError):
    """The model requested a tool-call batch beyond the run limit."""

    def __init__(self, *, attempted: int, limit: int, batch_size: int) -> None:
        self.attempted = attempted
        self.limit = limit
        self.batch_size = batch_size
        super().__init__(
            "Kernel-Tool-Budget ueberschritten: "
            f"{attempted} angefordert, maximal {limit}."
        )


class KernelToolBudgetMiddleware(AgentMiddleware):
    """Reject an overflowing model batch before any tool is dispatched.

    Counts are derived from checkpointed AI messages rather than mutable
    process state.  Re-entry and park/resume therefore produce the same total
    without double-counting, including multiple calls in one assistant turn.

    Args:
        max_tool_calls: Positive run-wide call ceiling.  The entire model
            batch is rejected when its cumulative total exceeds this value.
    """

    def __init__(self, max_tool_calls: int) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        self.max_tool_calls = max_tool_calls

    def after_model(
        self, state: dict[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        """Validate the latest model batch at the pre-dispatch boundary."""
        del runtime
        messages = state.get("messages") or []
        ai_messages = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        if not ai_messages:
            return None
        latest_calls = list(ai_messages[-1].tool_calls or [])
        if not latest_calls:
            return None
        attempted = sum(
            len(message.tool_calls or []) for message in ai_messages
        )
        if attempted <= self.max_tool_calls:
            return None

        payload = {
            "attempted": attempted,
            "limit": self.max_tool_calls,
            "batch_size": len(latest_calls),
        }
        log.warning(
            "Kernel-Tool-Budget ueberschritten: attempted=%d limit=%d "
            "batch_size=%d; kompletter Batch wird nicht ausgefuehrt.",
            attempted,
            self.max_tool_calls,
            len(latest_calls),
        )
        from inqtrix.agents.kernel.deps import kernel_deps

        try:
            deps = kernel_deps()
        except RuntimeError:
            deps = None
        if deps is not None:
            deps.emit(TOOL_LIMIT_REACHED_EVENT, payload)
        raise KernelToolBudgetExceeded(
            attempted=attempted,
            limit=self.max_tool_calls,
            batch_size=len(latest_calls),
        )



CONTEXT_COMPACTED_EVENT = "inqtrix.agent.context.compacted"

KERNEL_SUMMARY_PROMPT = (
    "Fasse den folgenden aelteren Gespraechsverlauf eines Recherche-"
    "Agenten kompakt zusammen. PFLICHT: Uebernimm alle reference_id-"
    "Werte (ref_...), alle Belege-Labels ([K1], [W2], ...), alle "
    "artifact_id-Werte und alle offenen Aufgaben WOERTLICH — sie sind "
    "Vertragsbestandteil, keine Prosa. Fasse Werkzeugausgaben nach "
    "ihrem Informationsgehalt zusammen, nicht nach Wortlaut.\n\n"
    "{messages}"
)
"""Kernel summary prompt: citation tokens survive compaction verbatim.

The evidence ledger itself is persisted separately (``evidence_bundle``
artifact, rehydrated on resume), so compaction can never destroy the
citation TRUTH — this prompt additionally keeps the citation TOKENS in
the summarized transcript so the model keeps citing correctly."""

current_compaction_todos: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "inqtrix_kernel_compaction_todos", default=None
)
"""The write_todos state of the segment currently inside wrap_model_call.

A ContextVar (not middleware instance state): the compiled graph — and
thus the middleware instance — is shared across concurrent runs, while
the todo recitation must come from THE run being compacted."""


def render_todo_state(todos: Any) -> str:
    """Render the write_todos state for recitation inside a summary."""
    if not isinstance(todos, list) or not todos:
        return ""
    lines = []
    for todo in todos:
        if not isinstance(todo, dict):
            continue
        status = str(todo.get("status") or "pending")
        content = str(todo.get("content") or "").strip()
        if content:
            lines.append(f"- [{status}] {content}")
    return "\n".join(lines)


def message_text(message: Any) -> str:
    """Plain text of a LangChain message (string or content blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "")


def emit_compaction_event(before: Any, after: dict[str, Any]) -> None:
    """Emit the visible compaction event + narration (No Silent Fallbacks).

    Called by the harness summarization middleware after a compaction
    landed. ``before``/``after`` are ``_summarization_event`` dicts; the
    narration id is a content hash of the summary, so replayed segments
    emit the identical id (R1 replay-safe events).
    """
    from inqtrix.agents.kernel.deps import kernel_deps

    try:
        deps = kernel_deps()
    except RuntimeError:
        return
    prior_cutoff = (
        int(before.get("cutoff_index") or 0) if isinstance(before, dict) else 0
    )
    cutoff = int(after.get("cutoff_index") or 0)
    summary_text = message_text(after.get("summary_message"))
    archive_id = str(after.get("file_path") or "")
    # deepagents swallows summary-model failures into a literal error
    # string used AS the summary — a compaction that lost its summary
    # must not report success (No Silent Fallbacks); the archive pointer
    # keeps the evicted history recoverable either way.
    summary_failed = "Error generating summary:" in summary_text
    payload = {
        "archive_artifact_id": archive_id,
        "messages_summarized": max(0, cutoff - prior_cutoff),
        "summary_chars": len(summary_text),
        "trigger_tokens": deps.context_trigger_tokens,
        "summary_failed": summary_failed,
    }
    if summary_failed:
        log.warning(
            "Kernel-Kontext komprimiert OHNE Zusammenfassung "
            "(Provider-Fehler im Summary-Aufruf; Archiv %s vollstaendig).",
            archive_id or "-",
        )
    else:
        log.info(
            "Kernel-Kontext komprimiert: %d Nachrichten zusammengefasst "
            "(Archiv %s, Trigger %d Tokens).",
            payload["messages_summarized"],
            archive_id or "-",
            deps.context_trigger_tokens,
        )
    deps.emit(CONTEXT_COMPACTED_EVENT, payload)
    digest = hashlib.sha1(summary_text.encode("utf-8")).hexdigest()[:8]
    # The narration is the USER surface of No-Silent-Fallbacks: a
    # compaction that lost its summary must say so here too, not only
    # in the structured event flag.
    narration_text = (
        (
            "Kontext-Komprimierung ohne Zusammenfassung (Fehler im "
            "Zusammenfassungs-Aufruf) — aeltere Schritte unverdichtet "
            "im Lauf-Archiv gesichert."
        )
        if summary_failed
        else (
            "Kontext komprimiert: aeltere Schritte zusammengefasst "
            "und im Lauf-Archiv gesichert."
        )
    )
    deps.emit(
        "inqtrix.agent.narration",
        {
            # Content-hash id: a replayed segment emits the identical
            # narration id (R1 replay-safe events).
            "narration_id": f"ctx_{digest}",
            "kind": "status",
            "phase": "execution",
            "text": narration_text,
            "final": True,
        },
    )


CHILD_TOOL_NAMES = frozenset(
    {"run_web_research", "run_deep_mission", "delegate_batch"}
)
"""Kernel tools that submit child runs and interrupt inside their body."""

CHILD_BATCH_GUARDED_EVENT = "inqtrix.agent.child_batch.guarded"


INTERRUPTING_TOOL_NAMES: frozenset[str] = frozenset({"ask_user"})
"""Non-child tools whose BODY interrupts the segment (ask_user parks
``waiting_for_input``). At most one interrupt source may dispatch per
model turn: a child submission next to an ask_user call would trip the
multi-interrupt backstop AFTER the submission — an orphaned child
burning quota behind a crashed parent."""


def turn_has_interrupting_call(last_ai: Any) -> bool:
    """Whether the turn carries an ``ask_user``-class interrupting call."""
    return any(
        call.get("name") in INTERRUPTING_TOOL_NAMES
        for call in (getattr(last_ai, "tool_calls", None) or [])
    )


def last_turn_child_calls(
    messages: list[Any],
) -> tuple[Any | None, list[dict[str, Any]]]:
    """The last AIMessage and its child-tool calls (the current turn).

    THE shared reading of "how many child dispatches does this turn
    carry" — the batch guard voids multi-child turns with it, and the
    gate policy's ``when`` predicate skips HITL for exactly those turns
    (one authority per call: gating a doomed batch would park an
    approval whose actions never execute, and a reject would answer the
    calls twice — once by HITL, once by the guard).
    """
    last_ai = next(
        (
            message
            for message in reversed(messages or [])
            if isinstance(message, AIMessage)
        ),
        None,
    )
    if last_ai is None:
        return None, []
    return last_ai, [
        call
        for call in (last_ai.tool_calls or [])
        if call["name"] in CHILD_TOOL_NAMES
    ]


class KernelChildBatchGuardMiddleware(AgentMiddleware):
    """More than one child-tool call per model turn never reaches dispatch.

    Each child tool interrupts individually inside its body; two of them
    dispatched in one segment would trip the multi-interrupt guard AFTER
    both submissions — orphaned children burning quota behind a crashed
    parent. This guard answers every child call of the offending batch
    with an artificial ToolMessage pointing at ``delegate_batch`` and
    strips them from the AIMessage (the HITL middleware's own rewrite
    mechanics) — PRE-dispatch, so nothing was submitted. The algorithm's
    multi-interrupt RuntimeError stays as the loud backstop for unknown
    future interrupt sources.

    The gate policy's ``when`` predicate keeps HITL away from these
    turns (see :func:`last_turn_child_calls`), so the guard is the ONE
    answering authority; the already-answered skip below stays as the
    structural double-answer backstop should middleware ordering ever
    change (a second ToolMessage per call id corrupts the transcript
    and providers reject the next model call).
    """

    def after_model(
        self, state: dict[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        del runtime
        messages = state.get("messages") or []
        last_ai, child_calls = last_turn_child_calls(messages)
        if last_ai is None or not child_calls:
            return None
        # ask_user + child in ONE turn is the same orphan trap as two
        # children: the child submits, then BOTH bodies interrupt. The
        # clarification wins; the child is voided pre-dispatch.
        ask_turn = turn_has_interrupting_call(last_ai)
        if len(child_calls) <= 1 and not ask_turn:
            return None
        answered: set[str] = set()
        for message in reversed(messages):
            if message is last_ai:
                break
            if getattr(message, "type", "") == "tool":
                answered.add(
                    str(getattr(message, "tool_call_id", "") or "")
                )
        open_calls = [
            call for call in child_calls if call["id"] not in answered
        ]
        if not open_calls:
            return None
        from langchain_core.messages import ToolMessage

        # The calls STAY on the AIMessage — an artificial ToolMessage per
        # call answers them (the tool node skips answered calls), exactly
        # the HITL middleware's reject mechanics. Removing them would
        # leave orphan ToolMessages the next model call rejects.
        corrective_text = (
            (
                "Rueckfrage und Unterauftrag passen nicht in EINEN Zug: "
                "ask_user laeuft jetzt; starte den Unterauftrag erst im "
                "naechsten Zug, wenn die Antwort vorliegt."
            )
            if ask_turn
            else (
                "Mehrere Unterauftrags-Werkzeuge in EINEM Zug sind "
                "nicht erlaubt. Nutze delegate_batch mit bis zu 3 "
                "Auftraegen fuer parallele Unterauftraege — oder rufe "
                "die Einzelwerkzeuge nacheinander auf."
            )
        )
        corrective = [
            ToolMessage(
                content=corrective_text,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )
            for call in open_calls
        ]
        payload = {
            "batch_size": len(child_calls),
            "tools": sorted({call["name"] for call in child_calls}),
        }
        log.warning(
            "Kernel-Child-Batch-Guard: %d parallele Kind-Werkzeuge in "
            "einem Zug abgefangen (%s) — keine Submission erfolgt.",
            payload["batch_size"],
            ", ".join(payload["tools"]),
        )
        from inqtrix.agents.kernel.deps import kernel_deps

        try:
            deps = kernel_deps()
        except RuntimeError:
            deps = None
        if deps is not None:
            deps.emit(CHILD_BATCH_GUARDED_EVENT, payload)
            # Visible interception (No Silent Fallbacks): a guarded batch
            # flows through the same narration channel as compaction, so
            # the user sees WHY nothing was dispatched. Content-hash id
            # over the answered call ids keeps a replayed segment stable.
            digest = hashlib.sha1(
                ",".join(sorted(call["id"] for call in open_calls)).encode(
                    "utf-8"
                )
            ).hexdigest()[:8]
            deps.emit(
                "inqtrix.agent.narration",
                {
                    "narration_id": f"batchguard_{digest}",
                    "kind": "status",
                    "phase": "execution",
                    "text": (
                        "Rueckfrage zuerst — der Unterauftrag startet "
                        "nach der Antwort."
                        if ask_turn
                        else (
                            "Mehrere Unterauftraege in einem Zug "
                            "abgefangen — der Agent buendelt sie nun "
                            "ueber delegate_batch."
                        )
                    ),
                    "final": True,
                },
            )
        return {"messages": [last_ai, *corrective]}
