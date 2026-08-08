"""Kernel tool surface.

Every tool is a sync LangChain tool wrapping Inqtrix services through
the :mod:`inqtrix.agents.kernel.deps` ContextVar — identity is NEVER a
tool argument (E-rules: ``CapabilityContext``/deps carry it, the model
cannot influence it). Denials and failures return as visible tool
results, never as silent empties.

Import guard: LangChain lives behind the optional ``agent`` extra.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from inqtrix.agents.algorithm import (
    ARTIFACT_CREATED_EVENT,
    ARTIFACT_UPDATED_EVENT,
    CLARIFICATION_REQUESTED_EVENT,
)
from inqtrix.agents.clarification import (
    build_clarification,
    round_qa_lines,
    sanitize_questions,
)
from inqtrix.agents.control_ports import (
    ArtifactNotFound,
    ArtifactRevisionConflict,
    ClarificationNotFound,
)
from inqtrix.agents.kernel.deps import kernel_deps, run_coro
from inqtrix.agents.kernel.interrupts import (
    ask_user_clarification_id,
    deliverable_artifact_id as _deliverable_artifact_id,
)
from inqtrix.agents.markdown import normalize_agent_markdown
from inqtrix.agents.report_quality import unknown_citation_labels
from inqtrix.agents.phase_models import (
    ClarificationOptionModel,
    ClarificationQuestionModel,
)
from inqtrix.agents.prompts import (
    UNTRUSTED_FENCE_CLOSE as _UNTRUSTED_CLOSE,
    UNTRUSTED_FENCE_OPEN as _UNTRUSTED_OPEN,
    untrusted_fence as _untrusted_fence,
)
from inqtrix.urls import normalize_url

log = logging.getLogger("inqtrix")

_IMPORT_HINT = (
    "Der Agent-Kernel braucht das 'agent'-Extra (uv sync --extra agent)."
)

_DOCUMENT_TEXT_LIMIT = 20_000
"""Character cap per read document — the model sees the truncation
marker, never a silently shortened text."""

SKILL_LOADED_MARKER = "[skill_geladen:{skill_id}@{revision}]"
"""Machine-readable first line of a successful ``load_skill`` result.
The algorithm reconstructs activated skills from these markers in the
checkpointed transcript at segment start — a restriction acquired
before a park must survive the resume (never a security hole)."""

DELIVERABLE_KINDS = ("memo", "email", "talking_points", "generic")
"""Format hints a kernel canvas deliverable may carry
(``payload.deliverable_kind``). Pure rendering hints —
``email`` implies no sending until an integration exists."""


SOURCE_TOOL_FAILURE_PREFIXES: tuple[str, ...] = (
    "Werkzeug nicht verfuegbar:",
    "Werkzeug-Fehler",
    "Werkzeug blockiert:",
)
"""Stable German prefixes of every VISIBLE source-tool failure text.

The ONE contract between the tool bodies (which return failures as
plain strings, never exceptions) and the checkpoint rehydration
(``_checkpointed_tool_use_counts``): a persisted ToolMessage starting
with one of these carries a FAILED call and must not count as a
successful source-tool use — the live counter only increments after a
successful invoke."""


def _require_allowed(tool_name: str) -> str | None:
    """The allowed_tools chokepoint; a block returns the VISIBLE text."""
    try:
        kernel_deps().require_tool_allowed(tool_name)
    except PermissionError as exc:
        return (
            f"Werkzeug blockiert: {exc} Erwaehne diese Einschraenkung "
            "in der Antwort."
        )
    return None


_OFFLOAD_DIGEST_CHARS = 1_500
"""Verbatim head kept in-context when a bulk tool result is archived."""

_MAX_BATCH_ASSIGNMENTS = 3
"""Ceiling for one delegate_batch call — bounded supervisor fan-out
(reliability-wall consensus), never swarm-width."""

def _offload_bulky_result(
    text: str,
    *,
    tool: str,
    reference_lines: str = "",
    status_lines: str = "",
) -> str:
    """Archive an oversized tool result; keep a digest + ALL citations.

    Ledger-grounded offload (F5-safe by construction): the FULL text is
    appended to the run's ``context_archive`` artifact, while the
    transcript keeps a verbatim head plus every reference line — so the
    model can still cite every source and can re-read the full text via
    ``read_canvas``. Below the threshold (or with the offload disabled,
    threshold 0) the text passes through unchanged.
    """
    deps = kernel_deps()
    limit = int(deps.context_offload_chars or 0)
    if limit <= 0 or len(text) <= limit:
        return text
    try:
        archive_ids = deps.append_context_archive_chunked(
            f"Werkzeugausgabe {tool}", text
        )
    except Exception as exc:  # noqa: BLE001 — never lose the result itself
        log.warning(
            "Archiv-Offload fuer %s fehlgeschlagen (error_type=%s) — die Ausgabe "
            "bleibt ungekuerzt im Kontext.",
            tool,
            type(exc).__name__,
        )
        return text
    log.info(
        "Werkzeugausgabe %s ins Lauf-Archiv ausgelagert (%d Zeichen).",
        tool,
        len(text),
    )
    digest = text[:_OFFLOAD_DIGEST_CHARS]
    if digest.count(_UNTRUSTED_OPEN) > digest.count(_UNTRUSTED_CLOSE):
        # The cut landed inside an untrusted fence — close it so the
        # data-not-instructions boundary stays intact in the digest.
        digest = f"{digest}\n{_UNTRUSTED_CLOSE}"
    archive_pointer = "; ".join(
        f"read_canvas(artifact_id='{archive_id}')"
        for archive_id in archive_ids
    )
    parts = [
        digest,
        (
            f"[... {len(text) - len(digest)} weitere Zeichen — "
            f"Volltext im Lauf-Archiv: {archive_pointer} ...]"
        ),
    ]
    if reference_lines:
        parts.append(f"Belege (vollstaendig):\n{reference_lines}")
    if status_lines:
        # Completeness, dependency and budget notices are part of the result
        # contract, not bulky source prose.  Preserve them verbatim even when
        # the corresponding position in the full result lives beyond the
        # digest head.
        parts.append(f"Statushinweise (vollstaendig):\n{status_lines}")
    return "\n\n".join(parts)


def _ref_note(ref: dict[str, Any]) -> str:
    """Model-facing citation note: citable [K#/W#] label + attach id.

    The label is what the running text may cite (``[W3]``); the
    reference_id is what ``write_canvas`` attaches. Rendering BOTH in
    every evidence-producing tool output closes the F5 gap where the
    model only ever saw opaque ids and could not cite labels that
    ``write_canvas`` later validates against the ledger.
    """
    label = str(ref.get("label") or "")
    rid = str(ref.get("reference_id") or "")
    if label:
        return f"Beleg [{label}] — reference_id: {rid}"
    return f"reference_id: {rid}"


class _CapabilityFailureText(str):
    """Visible tool failure that retains its machine-readable error code."""

    code: str

    def __new__(cls, value: str, *, code: str) -> "_CapabilityFailureText":
        instance = super().__new__(cls, value)
        instance.code = code
        return instance


def _invoke_capability(capability_id: str, payload: dict[str, Any]) -> Any:
    """Run one capability inside a tool span (real execution boundary).

    THIS is where kernel tools actually execute (the stream-update
    translation only sees them afterwards), so the span here carries a
    truthful duration and parents the capability's LLM/search/knowledge
    child spans. Failure texts stay non-exceptional but become a span
    attribute — a denied or broken tool is visible in the waterfall.

    Tool arguments carry raw user content (web queries, memo bodies).
    They are attached ONLY when content capture is enabled (forensic /
    trace_content=on), redacted through the same sanitizer as every
    other content attribute (Design law 3: one redaction path), and the
    preview is built lazily behind the recording gate — never for a
    non-recording span or with tracing off.
    """
    from inqtrix.observability import semconv
    from inqtrix.observability.otel import operation_span, span_is_recording

    with operation_span(
        capability_id,
        {
            semconv.GEN_AI_OPERATION_NAME: semconv.OPERATION_EXECUTE_TOOL,
            "inqtrix.tool.capability": capability_id,
        },
    ) as span:
        if span is not None and span_is_recording():
            _attach_tool_args(span, payload)
        result = _invoke_capability_inner(capability_id, payload)
        if span is not None and isinstance(result, _CapabilityFailureText):
            span.set_attribute(
                "inqtrix.tool.failure_code",
                str(getattr(result, "code", "") or "failure"),
            )
        return result


def _tool_content_policy():
    """The COMPOSED process content policy (published by create_providers).

    Reads the shared holder so tool-arg capture is gated by the exact
    same decision as every other content attribute — never by a fresh
    env-driven ``Settings()`` that could diverge from the app's
    composition.
    """
    from inqtrix.observability.content import active_content_policy

    return active_content_policy()


_tool_policy_warned = False


def _attach_tool_args(span: Any, payload: dict[str, Any]) -> None:
    """Attach tool arguments as span content, redacted and policy-gated."""
    global _tool_policy_warned
    try:
        policy = _tool_content_policy()
    except Exception:  # noqa: BLE001 — telemetry must never break a tool
        # Fail-safe (the tool call proceeds), never fail-SILENT: one
        # WARNING per process, because a broken policy means forensic
        # captures are quietly missing from every tool span.
        if not _tool_policy_warned:
            _tool_policy_warned = True
            log.warning(
                "Tool-Argument-Capture deaktiviert: Content-Policy "
                "nicht aufloesbar.",
                exc_info=True,
            )
        return
    if not policy.capture_content:
        return
    clipped = policy.clip_payload(payload)
    span.set_attribute("inqtrix.tool.args", clipped.text)
    if clipped.truncated:
        # The documented invariant: EVERY capped value raises the
        # truncation event, so an accidentally thin trace is findable
        # instead of silently short (settings-and-env.md on
        # INQTRIX_TRACE_MAX_ATTR_BYTES). provider_tracing._set_content
        # does the same for prompts/responses.
        from inqtrix.observability import semconv

        span.add_event(
            semconv.TRUNCATION_EVENT,
            {
                semconv.TRUNCATION_LIMIT_NAME: "inqtrix.tool.args",
                semconv.TRUNCATION_ORIGINAL_SIZE: clipped.original_size,
                semconv.TRUNCATION_CAPPED_SIZE: len(
                    clipped.text.encode("utf-8")
                ),
            },
        )


def _invoke_capability_inner(
    capability_id: str, payload: dict[str, Any]
) -> Any:
    """Run one capability for the current segment, or a VISIBLE denial.

    Returns the output model on success, or a ``str`` failure text the
    tool hands to the model verbatim (No Silent Fallbacks: every miss
    is a log line AND a transcript-visible tool result the prompt
    obliges the model to acknowledge).
    """
    from inqtrix.capabilities import CapabilityError, UnknownCapability

    deps = kernel_deps()
    if deps.capability_registry is None:
        log.warning(
            "Kernel-Tool ohne Capability-Registry aufgerufen (%s).",
            capability_id,
        )
        return _CapabilityFailureText(
            f"Werkzeug nicht verfuegbar: {capability_id} ist auf diesem "
            "Server nicht eingerichtet. Beantworte den Auftrag ohne "
            "dieses Werkzeug und benenne die Luecke.",
            code="capability_registry_missing",
        )
    try:
        output = run_coro(
            deps.capability_registry.invoke(
                capability_id, payload, deps.capability_context
            )
        )
        if capability_id.startswith("web."):
            deps.record_source_tool_use("web")
        elif capability_id.startswith("knowledge."):
            deps.record_source_tool_use("knowledge")
        return output
    except UnknownCapability:
        log.warning("Capability %s nicht registriert.", capability_id)
        return _CapabilityFailureText(
            f"Werkzeug nicht verfuegbar: {capability_id} ist auf diesem "
            "Server nicht eingerichtet. Benenne die Luecke in der Antwort.",
            code="unknown_capability",
        )
    except CapabilityError as exc:
        log.warning(
            "Capability %s abgelehnt/fehlgeschlagen (code=%s).",
            capability_id,
            exc.code,
        )
        return _CapabilityFailureText(
            f"Werkzeug-Fehler ({exc.code}): {exc.message} "
            "Erwaehne diese Einschraenkung in der Antwort.",
            code=exc.code,
        )

# Module-level guarded import (not builder-local like the bridges):
# LangChain resolves the tool signature's postponed annotations against
# THIS module's globals, so `Annotated[str, InjectedToolCallId]` must be
# importable here. The loud install hint moves to the builder call.
try:
    from langchain_core.tools import InjectedToolCallId, tool
    from langgraph.types import interrupt

    _AGENT_EXTRA_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover - env-dependent
    InjectedToolCallId = None  # type: ignore[assignment,misc]
    tool = None  # type: ignore[assignment]
    interrupt = None  # type: ignore[assignment]
    _AGENT_EXTRA_ERROR = _exc


def build_kernel_tools() -> list[Any]:
    """The kernel's tool list (M2 steps 5-6).

    ``ask_user`` plus the wave-1 read capabilities (project knowledge
    search, document read, instant web search). Which of them gate per
    mode is the policy module's decision (``interrupt_on``), never the
    tools' own. Canvas writes and child-run tools join in steps 7-8.

    Raises:
        RuntimeError: LangChain is not installed (missing ``agent``
            extra).
    """
    if _AGENT_EXTRA_ERROR is not None:  # pragma: no cover - env-dependent
        raise RuntimeError(_IMPORT_HINT) from _AGENT_EXTRA_ERROR

    @tool
    def ask_user(
        question: str,
        options: list[str],
        tool_call_id: Annotated[str, InjectedToolCallId],
        default_assumption: str = "",
    ) -> str:
        """Stelle dem Nutzer GENAU EINE materiell blockierende Rueckfrage.

        Nur verwenden, wenn die Antwort das Ergebnis wesentlich veraendert
        und sich nicht aus Auftrag oder Verlauf ableiten laesst. Gib 2-4
        wahrscheinliche Antwortoptionen an (der Nutzer kann immer frei
        antworten) und nenne in default_assumption die Annahme, mit der du
        ohne Antwort weiterarbeiten wuerdest.

        Args:
            question: Die eine Rueckfrage an den Nutzer.
            options: 2-4 kurze, wahrscheinliche Antwortoptionen.
            default_assumption: Beste Annahme, falls keine Antwort kommt.

        Returns:
            Die Antwort des Nutzers als Frage/Antwort-Text.
        """
        blocked = _require_allowed("ask_user")
        if blocked:
            return (
                f"{blocked} Arbeite mit deiner besten Annahme weiter"
                + (f" ({default_assumption})." if default_assumption else ".")
            )
        deps = kernel_deps()
        run_id = deps.run_id
        clarification_id = ask_user_clarification_id(run_id, tool_call_id)
        # The tool function RE-EXECUTES on resume (LangGraph interrupt
        # semantics) — the deterministic id makes the create idempotent.
        try:
            record = run_coro(
                deps.control.get_clarification(run_id, clarification_id)
            )
        except ClarificationNotFound:
            questions = sanitize_questions(
                [
                    ClarificationQuestionModel(
                        prompt=question,
                        options=[
                            ClarificationOptionModel(
                                label=str(option), description=""
                            )
                            for option in options
                        ],
                        multi_select=False,
                    )
                ]
            )
            if not questions:
                # Same guard as the phase machine's clarify node: an
                # empty prompt cannot be asked — parking on it would
                # strand the run on a question no one can see.
                log.warning(
                    "ask_user ohne brauchbare Frage verworfen "
                    "(run=%s).",
                    run_id,
                )
                return (
                    "Die Rueckfrage war leer und wurde verworfen. "
                    "Arbeite mit deiner besten Annahme weiter"
                    + (
                        f": {default_assumption}"
                        if default_assumption
                        else "."
                    )
                )
            record = run_coro(
                deps.control.create_clarification(
                    build_clarification(
                        run_id,
                        questions=questions,
                        default_assumption=default_assumption,
                        clarification_id=clarification_id,
                    )
                )
            )
            deps.emit(
                CLARIFICATION_REQUESTED_EVENT,
                {
                    "clarification_id": record.clarification_id,
                    "question": record.question,
                    "options": [dict(option) for option in record.options],
                    "question_count": len(record.questions),
                },
            )
        decision = interrupt(
            {"kind": "clarification", "id": clarification_id}
        )
        lines = round_qa_lines(
            questions=list(record.questions),
            question=record.question,
            options=list(record.options),
            answers=decision.get("answers") or {},
            answer=str(decision.get("answer", "")),
            option_id=str(decision.get("option_id", "")),
        )
        if not lines:
            # An answered round always yields at least the legacy pair;
            # an empty transcript means the row resumed unanswered.
            return (
                "Der Nutzer hat nicht geantwortet. Arbeite mit der "
                f"Annahme weiter: {record.default_assumption or 'keine'}."
            )
        return "\n".join(
            f"Frage: {prompt}\nAntwort des Nutzers: {answer}"
            for prompt, answer in lines
        )

    @tool
    def search_project_knowledge(
        query: str,
        collection_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> str:
        """Durchsuche die Wissensdatenbank des Nutzers (interne Dokumente).

        Nutze dieses Werkzeug fuer Fakten aus den hinterlegten Projekt-
        Dokumenten. Mit collection_ids suchst du gezielt in bestimmten
        Sammlungen; ohne Angabe projektweit in allem Sichtbaren.

        Args:
            query: Die Suchanfrage in natuerlicher Sprache.
            collection_ids: Optionale Sammlungs-Ids zur Eingrenzung.
            top_k: Anzahl der Treffer (1-20).

        Returns:
            Die relevantesten Textauszuege mit Dokumenttiteln und Ids.
        """
        blocked = _require_allowed("search_project_knowledge")
        if blocked:
            return blocked
        deps = kernel_deps()
        output = _invoke_capability(
            "knowledge.search",
            {
                "query": query,
                "collection_ids": list(collection_ids or []),
                "top_k": max(1, min(int(top_k), 20)),
            },
        )
        if isinstance(output, str):
            return output
        warning_lines = [
            (
                f"- {warning.message} "
                f"(Code {warning.code}; Stufe {warning.stage or 'retrieval'}; "
                f"Kandidaten {warning.returned_candidate_pool}/"
                f"{warning.requested_candidate_pool}; finale Belege "
                f"{warning.returned_hits}/{warning.final_top_k}; "
                f"final_vollstaendig="
                f"{'ja' if warning.final_evidence_complete else 'nein'})"
            )
            for warning in getattr(output, "warnings", [])
        ]
        warning_block = (
            "\n\nRetrieval-Hinweis – die Trefferliste ist kein "
            "Vollständigkeitsnachweis:\n" + "\n".join(warning_lines)
            if warning_lines
            else ""
        )
        if not output.hits:
            return (
                f"Keine Treffer in der Wissensdatenbank fuer: {query}"
                + warning_block
            )
        raw_refs = [
            {
                "document_id": hit.document_id,
                "chunk_index": hit.chunk_index,
                "title": hit.document_title,
                "excerpt": hit.excerpt,
                "source_span": hit.source_span,
                "revision_id": hit.revision_id,
                "generation_id": hit.generation_id,
                "provenance_status": hit.provenance_status,
            }
            for hit in output.hits
        ]
        # Render from the REGISTERED (canonical) refs — only those carry
        # the citable K#-label the canvas contract validates against.
        registered = {
            str(ref.get("reference_id") or ""): ref
            for ref in deps.register_references(raw_refs)
        }
        blocks = [
            (
                f"{hit.rank}. {hit.document_title} "
                f"(Dokument {hit.document_id}, Abschnitt {hit.chunk_index})"
                f"\n{_ref_note(registered.get(deps.reference_id(ref), ref))}"
                f"\n{_untrusted_fence(hit.excerpt, 'wissensdatenbank')}"
            )
            for hit, ref in zip(output.hits, raw_refs, strict=True)
        ]
        ref_lines = "\n".join(
            f"- {hit.document_title} — "
            f"{_ref_note(registered.get(deps.reference_id(ref), ref))}"
            for hit, ref in zip(output.hits, raw_refs, strict=True)
        )
        return _offload_bulky_result(
            "Treffer aus der Wissensdatenbank:\n\n"
            + warning_block.lstrip()
            + ("\n\n" if warning_block else "")
            + "\n\n".join(blocks),
            tool="search_project_knowledge",
            reference_lines=ref_lines,
            status_lines="\n".join(warning_lines),
        )

    @tool
    def read_project_document(document_id: str) -> str:
        """Lies ein Dokument aus der Wissensdatenbank vollstaendig.

        Nutze dieses Werkzeug, wenn Suchtreffer nicht reichen und du den
        Zusammenhang eines konkreten Dokuments brauchst.

        Args:
            document_id: Die Dokument-Id aus einem Suchtreffer.

        Returns:
            Titel und Text des Dokuments (lange Texte sichtbar gekuerzt).
        """
        blocked = _require_allowed("read_project_document")
        if blocked:
            return blocked
        output = _invoke_capability(
            "knowledge.document.read", {"document_id": document_id}
        )
        if isinstance(output, str):
            return output
        text = output.text
        if len(text) > _DOCUMENT_TEXT_LIMIT:
            text = (
                text[:_DOCUMENT_TEXT_LIMIT]
                + "\n\n[... Dokument fuer die Anzeige gekuerzt ...]"
            )
        return (
            f"# {output.title}\n\n"
            f"{_untrusted_fence(text, 'wissensdatenbank')}"
        )

    @tool
    def web_instant(query: str) -> str:
        """Fuehre EINE schnelle Websuche mit externer Quelle aus.

        Nutze dieses Werkzeug fuer aktuelle oder externe Fakten, die
        nicht in der Wissensdatenbank stehen. Uebergib eine praezise
        SUCHQUERY, keine Gespraechsfrage: die wichtigsten Entitaeten
        und Schluesselwoerter, EIN Suchziel, ohne Fuellwoerter; einen
        Zeitbezug (Jahr, "aktuell") nur bei Aktualitaetsfragen. Die
        Query wird dem Nutzer im Standard-Modus woertlich zur Freigabe
        angezeigt und exakt so gesucht.

        Args:
            query: Die konkrete Suchquery (Keywords, ein Suchziel).

        Returns:
            Antworttext der Suche plus Quellenliste (URL, Titel).
        """
        blocked = _require_allowed("web_instant")
        if blocked:
            return blocked
        deps = kernel_deps()
        output = _invoke_capability("web.search.instant", {"query": query})
        if isinstance(output, str):
            return output
        deps.book_usage(output.prompt_tokens, output.completion_tokens)
        registered_rows = deps.register_instant_web_search(output)
        registered = {
            normalize_url(str(ref.get("url") or "")): ref
            for ref in registered_rows
        }
        sources = "\n".join(
            f"- {source.title or source.url} ({source.url}) — "
            f"{_ref_note(registered.get(normalize_url(source.url), {}))}"
            for source in output.sources
        )
        ledger_note = (
            "\n\nWebsearch-Ledger: "
            f"{len(registered_rows)} von Azure gelieferte Quellen vollständig "
            "registriert."
        )
        answer = normalize_agent_markdown(
            output.answer or "(kein Antworttext)"
        )
        return _offload_bulky_result(
            f"{_untrusted_fence(answer, 'web')}"
            f"\n\nQuellen:\n{sources or '- keine'}{ledger_note}",
            tool="web_instant",
            reference_lines=sources,
        )

    @tool
    def write_canvas(
        title: str,
        content_markdown: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        deliverable_kind: str = "generic",
        artifact_id: str = "",
        expected_revision: int = 0,
        reference_ids: list[str] | None = None,
    ) -> str:
        """Schreibe oder aktualisiere ein Canvas-Dokument fuer den Nutzer.

        Nutze dieses Werkzeug fuer eigenstaendige, wiederverwendbare
        Ergebnisse (Memo, E-Mail-Entwurf, Sprechzettel). Fuer ein NEUES
        Dokument lasse artifact_id leer. Fuer ein UPDATE gib die
        artifact_id UND die dir bekannte expected_revision an — weicht
        die Revision ab, wurde das Dokument zwischenzeitlich geaendert
        und du musst den aktuellen Stand beruecksichtigen.

        Args:
            title: Kurzer Dokumenttitel.
            content_markdown: Vollstaendiger Markdown-Inhalt.
            deliverable_kind: memo | email | talking_points | generic.
            artifact_id: Nur bei Update: die Id des Ziel-Dokuments.
            expected_revision: Nur bei Update: die zuletzt gesehene
                Revision des Dokuments (>= 1).
            reference_ids: Die vom Lese-/Recherchewerkzeug gelieferten
                Beleg-Ids, die dieses Dokument tatsaechlich verwendet.
                Bei Updates bewahrt ein weggelassenes Feld die bisherigen
                Belege; eine explizite leere Liste entfernt sie.

        Returns:
            Bestaetigung mit artifact_id und neuer Revision.
        """
        blocked = _require_allowed("write_canvas")
        if blocked:
            return blocked
        deps = kernel_deps()
        if deliverable_kind not in DELIVERABLE_KINDS:
            return (
                f"Unbekannte Dokumentart {deliverable_kind!r} — erlaubt "
                f"sind: {', '.join(DELIVERABLE_KINDS)}."
            )
        is_update = bool(artifact_id)
        if is_update and expected_revision < 1:
            return (
                "Ein Update braucht die expected_revision (>= 1) des "
                "Dokuments, das du aktualisierst."
            )
        current = None
        if is_update:
            if not deps.session_id:
                return "Canvas-Updates brauchen eine session_id."
            try:
                current = run_coro(
                    deps.control.get_session_artifact_by_id(
                        deps.session_id, artifact_id
                    )
                )
            except ArtifactNotFound:
                return f"Canvas-Dokument nicht gefunden: {artifact_id}."
            if current.kind != "deliverable":
                return f"Canvas-Dokument nicht gefunden: {artifact_id}."
        if reference_ids is None and current is not None:
            refs = [dict(ref) for ref in current.refs]
        else:
            try:
                refs = deps.resolve_reference_ids(list(reference_ids or []))
            except ValueError as exc:
                log.warning(
                    "write_canvas mit unbekanntem Beleg "
                    "(run=%s, error_type=%s).",
                    deps.run_id,
                    type(exc).__name__,
                )
                return f"Belegfehler: {exc}. Lies oder recherchiere erneut."
        target_id = artifact_id or _deliverable_artifact_id(
            deps.run_id, tool_call_id
        )
        normalized_markdown = normalize_agent_markdown(content_markdown)
        # Citation parity with the mission engine (report_quality, one
        # definition): every [K#]/[W#] in the text must resolve against
        # the ATTACHED references — those become the artifact refs the
        # UI links. Unknown labels fail loudly; the kernel loop is its
        # own repair round.
        unknown_labels = unknown_citation_labels(normalized_markdown, refs)
        if unknown_labels:
            attached = ", ".join(
                str(ref.get("label") or "") for ref in refs
            ) or "(keine)"
            log.warning(
                "write_canvas mit unbekannten Belege-Labels "
                "(run=%s, unknown_count=%d, attached_count=%d).",
                deps.run_id,
                len(unknown_labels),
                len(refs),
            )
            return (
                "Belegfehler: Der Text zitiert unbekannte Labels "
                f"{', '.join(unknown_labels)}; angehaengt sind nur: "
                f"{attached}. Haenge die passenden reference_ids an "
                "oder korrigiere die Zitate im Text."
            )
        try:
            record = run_coro(
                deps.control.upsert_artifact(
                    run_id=deps.run_id,
                    kind="deliverable",
                    session_id=deps.session_id or None,
                    title=title,
                    status="ready",
                    content_markdown=normalized_markdown,
                    payload={"deliverable_kind": deliverable_kind},
                    refs=refs,
                    updated_by="agent",
                    artifact_id=target_id,
                    expected_revision=(
                        expected_revision if is_update else 0
                    ),
                )
            )
        except ArtifactRevisionConflict as exc:
            log.warning(
                "write_canvas Revisionskonflikt (run=%s, artifact=%s): "
                "erwartet %s, aktuell %s.",
                deps.run_id,
                target_id,
                expected_revision,
                exc.current_revision,
            )
            if not is_update:
                # Resume re-execution of an already-persisted create
                # (the tool shared its task with an interrupt): the row
                # exists — report it instead of failing the replay.
                return (
                    f"Canvas-Dokument existiert bereits: {target_id} "
                    f"(Revision {exc.current_revision})."
                )
            return (
                f"Revisionskonflikt: Das Dokument steht auf Revision "
                f"{exc.current_revision}, nicht {expected_revision}. "
                "Lies den aktuellen Stand, bevor du es aktualisierst."
            )
        except ArtifactNotFound:
            return f"Canvas-Dokument nicht gefunden: {target_id}."
        deps.emit(
            ARTIFACT_CREATED_EVENT
            if record.revision == 1
            else ARTIFACT_UPDATED_EVENT,
            {
                "artifact_id": record.artifact_id,
                "kind": "deliverable",
                "revision": record.revision,
                "title": record.title,
            },
        )
        verb = "erstellt" if record.revision == 1 else "aktualisiert"
        return (
            f"Canvas-Dokument '{title}' {verb} "
            f"(artifact_id {record.artifact_id}, Revision "
            f"{record.revision})."
        )

    @tool
    def read_canvas(artifact_id: str) -> str:
        """Lies ein bestehendes Canvas-Dokument samt Revision und Belegen.

        Nutze dieses Werkzeug vor jeder vollstaendigen Ueberarbeitung eines
        bestehenden Dokuments. Die zurueckgegebenen ``reference_id``-Werte
        koennen unveraendert an ``write_canvas`` weitergegeben werden.

        Args:
            artifact_id: Id aus der Canvas-Registry des Sitzungskontexts.

        Returns:
            Vollstaendiger Markdown-Inhalt, Revision und vertrauenswuerdige
            Beleg-Ids.
        """
        blocked = _require_allowed("read_canvas")
        if blocked:
            return blocked
        deps = kernel_deps()
        archive_prefix = deps.context_archive_prefix
        if artifact_id.startswith(archive_prefix):
            # The run's compaction archive is session-less and run-local
            # (offload digests and compaction summaries point here) —
            # serve it directly. One artifact per SECTION: the bare
            # prefix lists them, a section id returns its full body
            # (sections are capped at append time, so a pointer is
            # always fully readable — no display truncation here).
            if artifact_id == archive_prefix:
                rows, next_cursor = run_coro(
                    deps.control.list_artifacts(
                        deps.run_id, kind="context_archive", limit=100
                    )
                )
                if not rows:
                    return "Das Lauf-Archiv ist leer."
                lines = "\n".join(
                    f"- {row.artifact_id}: {row.title}"
                    for row in reversed(rows)
                )
                more = (
                    "\n[... weitere aeltere Sektionen vorhanden ...]"
                    if next_cursor
                    else ""
                )
                return (
                    "Lauf-Archiv — Sektionen (aelteste zuerst):\n"
                    + lines
                    + more
                    + "\n\nLies eine Sektion mit "
                    "read_canvas(artifact_id='<Sektions-Id>')."
                )
            try:
                record, _ = run_coro(
                    deps.control.get_artifact(deps.run_id, artifact_id)
                )
            except ArtifactNotFound:
                return f"Canvas-Dokument nicht gefunden: {artifact_id}."
            return (
                f"Lauf-Archiv-Sektion '{record.title}':\n\n"
                f"{record.content_markdown}"
            )
        if not deps.session_id:
            return "Canvas-Lesen braucht eine session_id."
        try:
            record = run_coro(
                deps.control.get_session_artifact_by_id(
                    deps.session_id, artifact_id
                )
            )
        except ArtifactNotFound:
            return f"Canvas-Dokument nicht gefunden: {artifact_id}."
        refs = deps.register_references([dict(ref) for ref in record.refs])
        ref_lines = "\n".join(
            f"- {_ref_note(ref)}: "
            f"{ref.get('title') or ref.get('url') or ref.get('document_id')}"
            for ref in refs
        ) or "- keine"
        return (
            f"Canvas-Dokument: {record.title}\n"
            f"artifact_id: {record.artifact_id}\n"
            f"revision: {record.revision}\n\n"
            f"{record.content_markdown}\n\nBelege:\n{ref_lines}"
        )

    @tool
    def propose_editor_patch(
        document_id: str,
        edits: list[dict[str, Any]],
        summary: str = "",
    ) -> str:
        """Schlage Aenderungen an einem Editor-Dokument des Nutzers vor.

        Der Vorschlag wird NIE direkt angewendet — der Nutzer prueft ihn
        als nachverfolgbare Aenderung im Editor. Jede Aenderung ist ein
        verankertes Edit-Objekt mit den Feldern: position (replace |
        before | after | append), find (zu ersetzender/ankernder Text),
        text (neuer Text), optional quote_before/quote_after (Anker-
        Kontext) und note (Begruendung).

        Args:
            document_id: Das Ziel-Dokument im Editor.
            edits: Liste verankerter Edit-Objekte (mindestens eines).
            summary: Ein Satz, was der Patch als Ganzes tut.

        Returns:
            Patch-Id und Status des Vorschlags.
        """
        blocked = _require_allowed("propose_editor_patch")
        if blocked:
            return blocked
        output = _invoke_capability(
            "editor.patch.propose",
            {
                "document_id": document_id,
                "edits": edits,
                "summary": summary,
            },
        )
        if isinstance(output, str):
            return output
        return (
            f"Patch {output.patch_id} mit {output.edit_count} "
            f"Aenderungen vorgeschlagen (Dokument {output.document_id}, "
            f"Status {output.status}). Der Nutzer prueft ihn im Editor."
        )

    def _submit_or_find_child(
        *,
        origin_key: str,
        mode: str,
        question: str,
        agent_overrides: dict[str, Any],
        autonomy: str = "",
    ) -> dict[str, Any] | str:
        """Idempotently submit one child run (or find the earlier submit).

        The tool re-executes on resume (interrupt semantics), so the
        submission is idempotent via ``origin_key`` — the re-execution
        finds the already-submitted child instead of spawning a second
        one (R5: child run rows are the truth, no control row backs this
        park). Returns the child row, or a VISIBLE failure string.
        """
        deps = kernel_deps()
        if deps.run_service is None or deps.resolver is None:
            log.warning(
                "Kind-Run-Tool ohne run_service/resolver aufgerufen "
                "(mode=%s).",
                mode,
            )
            return (
                "Werkzeug nicht verfuegbar: Kind-Laeufe sind auf diesem "
                "Server nicht eingerichtet. Benenne die Luecke."
            )
        store = deps.run_service.run_store
        child = next(
            (
                row
                for row in store.children(deps.run_id)
                if row.get("origin_key") == origin_key
            ),
            None,
        )
        if child is None:
            from inqtrix.server.runs import RunQueueFull

            resolve_payload: dict[str, Any] = {
                "mode": mode,
                "agent_overrides": agent_overrides,
            }
            if deps.stack_name:
                # F7c: children execute on the PARENT'S provider stack —
                # a run admitted on stack X must never silently fan out
                # on the default stack's models/search.
                resolve_payload["stack"] = deps.stack_name
            resolved_child = deps.resolver.resolve(resolve_payload)
            try:
                child = deps.run_service.submit(
                    question=question,
                    history="",
                    messages=[],
                    resolved=resolved_child,
                    workspace_id=getattr(
                        deps.capability_context, "workspace_id", None
                    ),
                    principal=deps.principal,
                    kind="agent_child",
                    parent_run_id=deps.run_id,
                    root_run_id=deps.run_id,
                    session_id=deps.session_id or None,
                    autonomy=autonomy,
                    parent_task_id=origin_key,
                    parent_task_attempt=1,
                    origin_key=origin_key,
                    source_policy={
                        "web": deps.source_policy.web,
                        "knowledge": deps.source_policy.knowledge,
                    },
                )
            except RunQueueFull:
                # A transient queue condition must not FAIL the parent
                # run (the default tool error handler re-raises): it
                # becomes a visible per-call result. In a batch this is
                # one failed slot among awaited siblings; on resume the
                # re-executed tool simply retries the submit (no origin
                # row exists for a never-submitted child).
                log.warning(
                    "Kind-Run nicht eingeplant — Lauf-Warteschlange "
                    "voll (mode=%s, origin_key=%s).",
                    mode,
                    origin_key,
                )
                return (
                    "Unterauftrag nicht eingeplant: Die "
                    "Lauf-Warteschlange ist derzeit voll. Versuche es "
                    "spaeter erneut oder reduziere die Parallelitaet."
                )
        return child

    def _merge_child_evidence(
        answer: str, references: list[dict[str, Any]]
    ) -> tuple[str, str, str]:
        """Register child refs and translate the child text's labels.

        A child label that collides with a parent-ledger owner is
        deterministically renumbered on merge. The child's text still
        cites the OLD labels — translate them IN THE TOOL (the model must
        never be trusted with the mapping: a verbatim-copied [W3] would
        silently resolve to the parent's DIFFERENT W3, and write_canvas
        only catches unknown labels, not wrong-source ones). Returns
        ``(translated_text, rename_note, source_lines)``.
        """
        deps = kernel_deps()
        raw_child_refs = [dict(ref) for ref in references]
        trusted_refs = deps.register_references(raw_child_refs)
        registered = {
            str(ref.get("reference_id") or ""): ref for ref in trusted_refs
        }
        renames = {
            old: new
            for raw in raw_child_refs
            if (canon := registered.get(deps.reference_id(raw))) is not None
            and (old := str(raw.get("label") or ""))
            and (new := str(canon.get("label") or ""))
            and old != new
        }
        # Two-phase replace: a new label can EQUAL another entry's old
        # label (child W1 -> parent W3 while the child also cites W3), so
        # direct substitution would cascade — placeholders first.
        for index, old in enumerate(renames):
            answer = answer.replace(f"[{old}]", f"\x00{index}\x00")
        for index, new in enumerate(renames.values()):
            answer = answer.replace(f"\x00{index}\x00", f"[{new}]")
        rename_note = (
            "\n\nHinweis: Belege des Unterauftrags wurden auf die "
            "Eltern-Labels uebersetzt: "
            + ", ".join(f"[{old}] -> [{new}]" for old, new in renames.items())
            if renames
            else ""
        )
        source_lines = "\n".join(
            f"- {ref.get('title') or ref.get('url') or ref.get('document_id')} "
            f"— {_ref_note(ref)}"
            for ref in trusted_refs[:10]
        )
        return answer, rename_note, source_lines

    def _child_report(
        child: dict[str, Any], *, mode: str, condensed: bool
    ) -> str:
        """Return one child's report without discarding provider evidence.

        Single-child delegation receives the complete report. Batch delegation
        receives the established bounded summary. In both cases references and
        the producing Websearch Ledger are merged into the parent run.
        """
        deps = kernel_deps()
        store = deps.run_service.run_store
        child_id = str(child["run_id"])
        if child["status"] != "completed":
            reason = str(
                (child.get("error") or {}).get("message", child["status"])
            )
            log.warning(
                "Kind-Run %s endete nicht erfolgreich (status=%s).",
                child_id,
                child["status"],
            )
            return (
                f"Der Unterauftrag ({mode}) ist fehlgeschlagen: {reason}. "
                "Beruecksichtige die Luecke in der Antwort."
            )
        stored_result = store.result(child_id, visible_to=deps.visible_to)
        raw_ledger = stored_result.get("web_search_ledger")
        if isinstance(raw_ledger, dict):
            deps.register_web_search_ledger(raw_ledger)

        if condensed:
            from inqtrix.agents.scheduler import project_child_run_outcome

            outcome = project_child_run_outcome(
                store, child_id, 1, visible_to=deps.visible_to
            )
            if outcome is None or outcome.status != "completed":
                reason = (
                    getattr(outcome, "failure_reason", "") or ""
                ) or "Ergebnis nicht mehr abrufbar"
                log.warning(
                    "Kind-Run %s lieferte kein projizierbares Ergebnis "
                    "(%s).",
                    child_id,
                    reason,
                )
                return (
                    f"Der Unterauftrag ({mode}) lieferte kein "
                    f"abrufbares Ergebnis: {reason}. Beruecksichtige "
                    "die Luecke in der Antwort."
                )
            text = outcome.summary or ""
            references = list(outcome.evidence)
        else:
            text = str(stored_result.get("answer", ""))
            references = [
                dict(reference)
                for reference in stored_result.get("references", []) or []
                if isinstance(reference, dict)
            ]
        if isinstance(raw_ledger, dict):
            from inqtrix.evidence import attach_web_search_lineage

            references = attach_web_search_lineage(
                references, raw_ledger
            )
        text, rename_note, source_lines = _merge_child_evidence(
            text, references
        )
        return _offload_bulky_result(
            f"Ergebnis des Unterauftrags (Run {child_id}):\n\n"
            + _untrusted_fence(text, "unterauftrag")
            + rename_note
            + (f"\n\nQuellen:\n{source_lines}" if source_lines else ""),
            tool=mode,
            reference_lines=source_lines,
        )

    def _await_child_run(
        *,
        tool_call_id: str,
        mode: str,
        question: str,
        agent_overrides: dict[str, Any],
        autonomy: str = "",
    ) -> str:
        """Submit-or-find ONE child run, park until terminal, report."""
        deps = kernel_deps()
        child = _submit_or_find_child(
            origin_key=tool_call_id,
            mode=mode,
            question=question,
            agent_overrides=agent_overrides,
            autonomy=autonomy,
        )
        if isinstance(child, str):
            return child
        if child["status"] not in ("completed", "failed", "cancelled"):
            interrupt({"kind": "children"})
            # Woken by the last child's terminal write: re-read the row
            # (the checkpoint re-executes this tool from the top, so
            # normally the fresh read above already sees the terminal
            # state — this is the same-execution wake path).
            child = deps.run_service.run_store.get(
                str(child["run_id"]), visible_to=deps.visible_to
            )
        return _child_report(child, mode=mode, condensed=False)

    @tool
    def run_web_research(
        question: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """Starte eine mehrstufige Web-Recherche als Unterauftrag.

        Nutze dieses Werkzeug statt web_instant, wenn eine Frage mehrere
        Suchen, Quellenabgleich und belegte Aussagen braucht. Der
        Unterauftrag laeuft eigenstaendig; du erhaeltst den fertigen
        Bericht mit Quellen.

        Args:
            question: Die vollstaendige Rechercheaufgabe.

        Returns:
            Bericht des Recherche-Laufs mit Quellenliste.
        """
        blocked = _require_allowed("run_web_research")
        if blocked:
            return blocked
        deps = kernel_deps()
        if not deps.web_research_allowed:
            log.warning(
                "run_web_research durch die wirksame Agent-Stufe blockiert; "
                "web_instant verwenden."
            )
            deps.emit(
                "inqtrix.agent.activity",
                {
                    "scope": "task",
                    "phase": "execution",
                    "operation": "web.research",
                    "detail": question,
                    "status": "failed",
                    "error": {
                        "code": "research_tier_blocked",
                        "message": (
                            "Die gewählte Agent-Stufe erlaubt keine "
                            "mehrstufige Web-Recherche."
                        ),
                    },
                },
            )
            return (
                "Werkzeug blockiert: Die gewählte Agent-Stufe erlaubt "
                "keine mehrstufige Web-Recherche. Nutze web_instant für "
                "eine einzelne Evidenzfrage."
            )
        # The shared server policy owns the profile; the model cannot select
        # a slower or weaker child after admission.
        report_profile = deps.web_research_profile
        if report_profile is None:
            raise RuntimeError("permitted web research has no profile")
        result = _await_child_run(
            tool_call_id=tool_call_id,
            mode="research",
            question=question,
            agent_overrides={"report_profile": report_profile},
            # Autonomy inheritance is UNIFORM across both child modes
            # (the research graph carries no HITL gates, so the field is
            # inert there — consistency, not behavior).
            autonomy=deps.autonomy,
        )
        deps.record_source_tool_use("web")
        return result

    @tool
    def run_deep_mission(
        assignment: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """Delegiere einen grossen Auftrag an die Missions-Maschine.

        Nutze dieses Werkzeug fuer Auftraege mit mehreren Recherche-
        Straengen, zitierter Multi-Quellen-Evidenz oder explizitem
        Berichtswunsch. Die Mission plant, recherchiert und prueft
        eigenstaendig (mit eigenen Freigaben) und liefert ein Memo.

        Args:
            assignment: Der vollstaendige Auftrag an die Mission.

        Returns:
            Das Ergebnis-Memo der Mission.
        """
        blocked = _require_allowed("run_deep_mission")
        if blocked:
            return blocked
        deps = kernel_deps()
        return _await_child_run(
            tool_call_id=tool_call_id,
            mode="workspace_agent",
            question=assignment,
            # A deep kernel run delegates a deep mission: the child
            # forces DEEP research profiles on ITS children in turn.
            agent_overrides=(
                {"depth": "deep"} if deps.depth == "deep" else {}
            ),
            autonomy=deps.autonomy,
        )

    def _admit_batch_assignment(
        deps: Any, mode: str
    ) -> tuple[str, dict[str, Any]] | str:
        """Per-assignment admission through the EXISTING chokepoints.

        Returns ``(child_mode, agent_overrides)`` or a VISIBLE denial
        string. Reuses the exact checks of the single-child tools —
        zero new policy predicates.
        """
        if mode == "research":
            blocked = _require_allowed("run_web_research")
            if blocked:
                return blocked
            if not deps.web_research_allowed:
                return (
                    "Werkzeug blockiert: Die gewählte Agent-Stufe erlaubt "
                    "keine mehrstufige Web-Recherche."
                )
            profile = deps.web_research_profile
            if profile is None:
                raise RuntimeError("permitted web research has no profile")
            return "research", {"report_profile": profile}
        blocked = _require_allowed("run_deep_mission")
        if blocked:
            return blocked
        return "workspace_agent", (
            {"depth": "deep"} if deps.depth == "deep" else {}
        )

    @tool
    def delegate_batch(
        assignments: list[dict[str, str]],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """Starte bis zu 3 Unterauftraege PARALLEL und warte auf alle.

        Jedes Element: {"objective": <vollstaendiger Auftrag>, "mode":
        "research" | "deep_mission"}. Nutze dieses Werkzeug statt
        mehrerer einzelner Unterauftrags-Aufrufe in einem Zug — die
        Einzelwerkzeuge bleiben fuer EINEN Unterauftrag richtig. Du
        erhaeltst je Auftrag eine kompakte Zusammenfassung (max. 300
        Woerter) mit Quellen und der Kind-Run-Id; die Kinder laufen
        gleichzeitig. Hinweis: Kinder verbrauchen eigenes Token-Budget.

        Args:
            assignments: 1-3 unabhaengige Auftraege mit objective und
                mode.

        Returns:
            Je Auftrag ein Ergebnisblock in Auftragsreihenfolge.
        """
        blocked = _require_allowed("delegate_batch")
        if blocked:
            return blocked
        deps = kernel_deps()
        if not isinstance(assignments, list) or not (
            1 <= len(assignments) <= _MAX_BATCH_ASSIGNMENTS
        ):
            return (
                "Ungueltiger Batch: 1 bis "
                f"{_MAX_BATCH_ASSIGNMENTS} Auftraege erwartet."
            )
        admitted: list[tuple[str, str, dict[str, Any]]] = []
        for index, assignment in enumerate(assignments):
            objective = str(
                (assignment or {}).get("objective") or ""
            ).strip()
            mode = str((assignment or {}).get("mode") or "").strip()
            if not objective or mode not in ("research", "deep_mission"):
                return (
                    f"Ungueltiger Auftrag {index + 1}: objective und "
                    "mode (research | deep_mission) sind Pflicht."
                )
            outcome = _admit_batch_assignment(deps, mode)
            if isinstance(outcome, str):
                return f"Auftrag {index + 1}: {outcome}"
            child_mode, overrides = outcome
            admitted.append((objective, child_mode, overrides))
        # Submit ALL children BEFORE the single park: the whole batch is
        # covered by ONE interrupt by construction — the multi-interrupt
        # crash class (and its orphan race) cannot occur here. A slot
        # whose submit failed visibly (e.g. queue full) stays a string:
        # already-submitted siblings are still awaited and reported —
        # never abandoned mid-batch.
        slots: list[dict[str, Any] | str] = []
        for index, (objective, child_mode, overrides) in enumerate(
            admitted
        ):
            child = _submit_or_find_child(
                origin_key=f"{tool_call_id}:{index}",
                mode=child_mode,
                question=objective,
                agent_overrides=overrides,
                autonomy=deps.autonomy,
            )
            if isinstance(child, str):
                log.warning(
                    "Batch-Auftrag nicht eingeplant (ordinal=%d).",
                    index + 1,
                )
            slots.append(child)
        active = [slot for slot in slots if isinstance(slot, dict)]
        if any(
            child["status"] not in ("completed", "failed", "cancelled")
            for child in active
        ):
            interrupt({"kind": "children"})
            store = deps.run_service.run_store
            slots = [
                store.get(str(slot["run_id"]), visible_to=deps.visible_to)
                if isinstance(slot, dict)
                else slot
                for slot in slots
            ]
        if any(mode == "research" for _, mode, _ in admitted):
            # Counter parity with rehydration, which counts ONE web use
            # per delegate_batch ToolMessage NAME — so live records once
            # per batch, never per child.
            deps.record_source_tool_use("web")
        blocks: list[str] = []
        for index, ((_, child_mode, _overrides), slot) in enumerate(
            zip(admitted, slots, strict=True)
        ):
            if isinstance(slot, str):
                blocks.append(
                    f"## Unterauftrag {index + 1}\n{slot} "
                    "Beruecksichtige die Luecke in der Antwort."
                )
                continue
            blocks.append(
                f"## Unterauftrag {index + 1}\n"
                + _child_report(slot, mode=child_mode, condensed=True)
            )
        return "\n\n".join(blocks)


    @tool
    def load_skill(
        skill_id: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """Aktiviere einen verfuegbaren Skill und lies seine Anleitung.

        Nutze dieses Werkzeug, wenn ein Skill aus der Liste der
        verfuegbaren Skills zum Auftrag passt. Die Anleitung wird Teil
        deines Arbeitskontexts; deklariert der Skill Werkzeug-Grenzen,
        gelten sie ab sofort.

        Args:
            skill_id: Die Id aus der Skill-Liste (sk_...).

        Returns:
            Die Skill-Anleitung samt deklarierter Eingabepunkte.
        """
        # An active skill restriction covers load_skill itself: the
        # allowed_tools vocabulary never contains it, so once a
        # restrictive skill is active the MODEL cannot widen its own
        # tool surface by loading a more permissive one (the user can,
        # by attaching it).
        blocked = _require_allowed("load_skill")
        if blocked:
            return blocked
        deps = kernel_deps()
        if deps.skill_service is None:
            return (
                "Skills sind auf diesem Server nicht eingerichtet. "
                "Arbeite ohne Skill weiter."
            )
        try:
            record, _shared = run_coro(
                deps.skill_service.get_visible(
                    skill_id,
                    tenant_id=(
                        deps.principal.tenant_id
                        if deps.principal is not None
                        else "default"
                    ),
                    visible_to=getattr(
                        deps.capability_context, "visible_to", None
                    ),
                )
            )
        except Exception:  # noqa: BLE001 — absence and denial read alike
            log.warning("load_skill: %s nicht sichtbar/vorhanden.", skill_id)
            return (
                f"Skill {skill_id} ist nicht verfuegbar. Arbeite ohne "
                "ihn weiter und benenne die Luecke, falls relevant."
            )
        if record.invocation != "model_allowed":
            # Structural injection defense (plan 3.1/3.3): shared-in and
            # user_only skills never self-activate — only the USER may
            # attach them.
            log.warning(
                "load_skill: %s ist user_only — Modell-Aktivierung "
                "verweigert.",
                skill_id,
            )
            return (
                f"Der Skill '{record.label}' darf nur vom Nutzer selbst "
                "aktiviert werden."
            )
        if record.requires_plan == "always" and deps.autonomy == "autonomous":
            # Attached always-skills escalate the run onto the gated
            # policy variant at submit time; a MID-RUN load cannot (the
            # interrupt_on set is compile-time) — refusing is the only
            # honest option left.
            log.warning(
                "load_skill: %s verlangt requires_plan=always — im "
                "Auto-Modus nicht nachladbar.",
                skill_id,
            )
            return (
                f"Der Skill '{record.label}' verlangt Freigabe-Gates und "
                "kann im Auto-Modus nicht nachgeladen werden. Der Nutzer "
                "kann ihn beim Start anhaengen."
            )
        from inqtrix.agents.kernel.middleware import (
            SKILL_INPUTS_RESOLVED_MARKER,
            resolve_kernel_skill_inputs,
        )

        answers = resolve_kernel_skill_inputs(
            record,
            clarification_scope=f"load:{tool_call_id}",
        )
        deps.skill_answers[record.id] = answers
        deps.activate_skill(record)
        deps.emit(
            "inqtrix.agent.skill.loaded",
            {"skill_id": record.id, "label": record.label},
        )
        marker = SKILL_LOADED_MARKER.format(
            skill_id=record.id, revision=record.revision
        )
        from inqtrix.agents.skills_runtime import build_skills_block

        return (
            f"{marker}\n{SKILL_INPUTS_RESOLVED_MARKER}\n"
            f"{build_skills_block([record], {record.id: answers})}"
        )

    return [
        ask_user,
        search_project_knowledge,
        read_project_document,
        web_instant,
        read_canvas,
        write_canvas,
        propose_editor_patch,
        run_web_research,
        run_deep_mission,
        delegate_batch,
        load_skill,
    ]
