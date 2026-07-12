"""Kernel tool surface (plan M2 step 4/5; walking skeleton: ``ask_user``).

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
from inqtrix.agents.evidence import enrich_instant_evidence
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

log = logging.getLogger("inqtrix")

_IMPORT_HINT = (
    "Der Agent-Kernel braucht das 'agent'-Extra (uv sync --extra agent)."
)

_DOCUMENT_TEXT_LIMIT = 20_000
"""Character cap per read document — the model sees the truncation
marker, never a silently shortened text."""

SKILL_LOADED_MARKER = "[skill_geladen:{skill_id}@{updated_at}]"
"""Machine-readable first line of a successful ``load_skill`` result.
The algorithm reconstructs activated skills from these markers in the
checkpointed transcript at segment start — a restriction acquired
before a park must survive the resume (never a security hole)."""

DELIVERABLE_KINDS = ("memo", "email", "talking_points", "generic")
"""Format hints a kernel canvas deliverable may carry
(``payload.deliverable_kind``, plan M2 `2.4`). Pure rendering hints —
``email`` implies no sending until an integration exists."""


def _require_allowed(tool_name: str) -> str | None:
    """The allowed_tools chokepoint; a block returns the VISIBLE text."""
    try:
        kernel_deps().require_tool_allowed(tool_name)
    except PermissionError as exc:
        return f"{exc} Erwaehne diese Einschraenkung in der Antwort."
    return None


def _invoke_capability(capability_id: str, payload: dict[str, Any]) -> Any:
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
        return (
            f"Werkzeug nicht verfuegbar: {capability_id} ist auf diesem "
            "Server nicht eingerichtet. Beantworte den Auftrag ohne "
            "dieses Werkzeug und benenne die Luecke."
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
        return (
            f"Werkzeug nicht verfuegbar: {capability_id} ist auf diesem "
            "Server nicht eingerichtet. Benenne die Luecke in der Antwort."
        )
    except CapabilityError as exc:
        log.warning(
            "Capability %s abgelehnt/fehlgeschlagen: %s (%s)",
            capability_id,
            exc.message,
            exc.code,
        )
        return (
            f"Werkzeug-Fehler ({exc.code}): {exc.message} "
            "Erwaehne diese Einschraenkung in der Antwort."
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
        if not output.hits:
            return f"Keine Treffer in der Wissensdatenbank fuer: {query}"
        raw_refs = [
            {
                "document_id": hit.document_id,
                "chunk_index": hit.chunk_index,
                "title": hit.document_title,
                "excerpt": hit.text,
            }
            for hit in output.hits
        ]
        deps.register_references(raw_refs)
        blocks = [
            (
                f"{hit.rank}. {hit.document_title} "
                f"(Dokument {hit.document_id}, Abschnitt {hit.chunk_index})"
                f"\nreference_id: {deps.reference_id(ref)}\n{hit.text}"
            )
            for hit, ref in zip(output.hits, raw_refs, strict=True)
        ]
        return "Treffer aus der Wissensdatenbank:\n\n" + "\n\n".join(blocks)

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
        return f"# {output.title}\n\n{text}"

    @tool
    def web_instant(query: str) -> str:
        """Fuehre EINE schnelle Websuche mit externer Quelle aus.

        Nutze dieses Werkzeug fuer aktuelle oder externe Fakten, die
        nicht in der Wissensdatenbank stehen. Formuliere die Anfrage
        praezise — sie wird dem Nutzer im Standard-Modus woertlich zur
        Freigabe angezeigt.

        Args:
            query: Die konkrete Suchanfrage.

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
        raw_refs = enrich_instant_evidence(
            str(output.answer or ""),
            [
                {
                    "url": source.url,
                    "title": source.title,
                    "excerpt": source.snippet,
                }
                for source in output.sources
            ],
        )
        deps.register_references(raw_refs)
        sources = "\n".join(
            f"- {source.title or source.url} ({source.url}) — "
            f"reference_id: {deps.reference_id(ref)}"
            for source, ref in zip(output.sources, raw_refs, strict=True)
        )
        answer = normalize_agent_markdown(
            output.answer or "(kein Antworttext)"
        )
        return f"{answer}\n\nQuellen:\n{sources or '- keine'}"

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
                log.warning("write_canvas mit unbekanntem Beleg: %s", exc)
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
                "write_canvas mit unbekannten Belege-Labels %s "
                "(run=%s, angehaengt: %s).",
                unknown_labels,
                deps.run_id,
                attached,
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
            f"- {ref['reference_id']}: "
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

    def _await_child_run(
        *,
        tool_call_id: str,
        mode: str,
        question: str,
        agent_overrides: dict[str, Any],
        autonomy: str = "",
    ) -> str:
        """Submit-or-find one child run, park until terminal, report.

        The tool re-executes on resume (interrupt semantics), so the
        submission is idempotent via ``origin_key=tool_call_id`` — the
        re-execution finds the already-submitted child in the parent's
        children instead of spawning a second one (R5: child run rows
        are the truth, no control row backs this park).
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
                if row.get("origin_key") == tool_call_id
            ),
            None,
        )
        if child is None:
            resolved_child = deps.resolver.resolve(
                {"mode": mode, "agent_overrides": agent_overrides}
            )
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
                parent_task_id=tool_call_id,
                parent_task_attempt=1,
                origin_key=tool_call_id,
                source_policy={
                    "web": deps.source_policy.web,
                    "knowledge": deps.source_policy.knowledge,
                },
            )
        if child["status"] not in ("completed", "failed", "cancelled"):
            interrupt({"kind": "children"})
            # Woken by the last child's terminal write: re-read the row
            # (the checkpoint re-executes this tool from the top, so
            # normally the fresh read above already sees the terminal
            # state — this is the same-execution wake path).
            child = store.get(str(child["run_id"]))
        child_id = str(child["run_id"])
        if child["status"] != "completed":
            reason = str(
                (child.get("error") or {}).get("message", child["status"])
            )
            log.warning(
                "Kind-Run %s endete nicht erfolgreich: %s",
                child_id,
                reason,
            )
            return (
                f"Der Unterauftrag ({mode}) ist fehlgeschlagen: {reason}. "
                "Beruecksichtige die Luecke in der Antwort."
            )
        result = store.result(child_id)
        answer = str(result.get("answer", ""))
        references = result.get("references", []) or []
        trusted_refs = deps.register_references(
            [dict(ref) for ref in references]
        )
        source_lines = "\n".join(
            f"- {ref.get('title') or ref.get('url') or ref.get('document_id')} "
            f"— reference_id: {ref['reference_id']}"
            for ref in trusted_refs[:10]
        )
        return (
            f"Ergebnis des Unterauftrags (Run {child_id}):\n\n{answer}"
            + (f"\n\nQuellen:\n{source_lines}" if source_lines else "")
        )

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
        if not deps.explicit_web_research:
            log.warning(
                "run_web_research in normalem Kernel-Lauf ohne explizite "
                "Recherche-Anweisung blockiert; web_instant verwenden."
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
                        "code": "research_not_explicit",
                        "message": (
                            "Mehrstufige Web-Recherche ist nur im Deep-Modus "
                            "oder nach ausdruecklicher Anweisung erlaubt."
                        ),
                    },
                },
            )
            return (
                "Werkzeug blockiert: Mehrstufige Web-Recherche ist in "
                "diesem normalen Lauf nicht ausdruecklich freigegeben. "
                "Nutze web_instant fuer eine einzelne Evidenzfrage."
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
            skill_id=record.id, updated_at=record.updated_at
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
        load_skill,
    ]
