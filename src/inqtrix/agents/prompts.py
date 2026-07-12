"""German prompt templates for workspace-agent phases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from inqtrix.urls import today

if TYPE_CHECKING:
    from collections.abc import Sequence

    from inqtrix.agents.plan_collections import CollectionCatalogEntry

# -- workspace-agent phase prompts (M5, German like all LLM templates) ---- #

_AGENT_INTAKE_SYSTEM = (
    "Du analysierst Arbeitsauftraege fuer einen Recherche-Agenten. "
    "Antworte ausschliesslich mit dem geforderten JSON."
)

_AGENT_ANALYST_SYSTEM = (
    "Du bist ein Discovery-Analyst: Du verdichtest Sondierungsergebnisse "
    "zu bekannten Fakten und Wissensluecken. Erfinde nichts; jede Angabe "
    "braucht eine Referenz aus den Ergebnissen."
)

_AGENT_PLANNER_SYSTEM = (
    "Du planst die Ausfuehrung eines Recherche-Auftrags als kleine, "
    "praezise Task-Liste. Waehle immer das KLEINSTE ausreichende Werkzeug."
)

_AGENT_SYNTHESIS_SYSTEM = (
    "Du schreibst praezise deutsche Memo-Abschnitte. Jede faktische "
    "Aussage traegt mindestens ein Belege-Label ([K1], [W2], ...). "
    "Nutze dafuer die kleinste hinreichende, nicht redundante Auswahl "
    "passender Belege (typischerweise 1-3 Labels pro faktischer Aussage "
    "oder Absatz), niemals pauschal alle verfuegbaren Labels. "
    "Unbelegtes gehoert in den Abschnitt 'Offene Punkte'."
)
# The shared rendering block is appended via agent_synthesis_system_prompt
# (defined after _RENDERING_CAPABILITIES below) — one SSOT for what the
# renderer supports, consumed by memo sections, chat answers and (M2)
# the kernel.

_RENDERING_CAPABILITIES = (
    "Dir stehen folgende Ausgabemittel zur Verfuegung — nutze sie "
    "gezielt: GFM-Tabellen fuer Vergleiche und Rankings — vergleicht "
    "eine Aussage zwei oder mehr Entitaeten in zwei oder mehr "
    "Attributen, ist eine Tabelle Pflicht; Formeln in "
    "KaTeX ($...$ inline, $$...$$ als Block); Code in ```-Bloecken mit "
    "Sprachangabe; Prozess-, Ablauf- und Architekturdiagramme als "
    "```mermaid-Block, NUR wenn ein Diagramm echten Mehrwert bietet. "
    "Nichts Dekoratives: jede Tabellenzelle und jeder Diagramm-Knoten "
    "muss auf Belege oder den Auftragskontext zurueckfuehrbar sein. "
    "Kein HTML, keine Emojis."
)
"""SSOT of what the frontend renderer actually supports (plan M1 S5).

Cross-reference: apps/research-desk/src/components/markdown/
MarkdownRenderer.tsx (remark-gfm, rehype-katex, rehype-pretty-code,
MermaidFigure). The drift test in tests/agents/test_prompts_rendering.py
asserts each feature is named here — removing one from the renderer must
update BOTH places. The M2 kernel imports the SAME accessor."""


def rendering_capabilities_block() -> str:
    """The shared output-capabilities block (SSOT, plan M1 S5)."""
    return _RENDERING_CAPABILITIES


_AGENT_ANSWER_SYSTEM = (
    "Du beantwortest Arbeitsauftraege direkt im Chat: praezises "
    "deutsches Markdown, konversationell und kompakt. Jede faktische "
    "Aussage traegt mindestens ein Belege-Label ([K1], [W2], ...); "
    "zitiere die kleinste hinreichende, nicht redundante Auswahl "
    "(typischerweise 1-3 Labels pro Aussage oder Absatz), nicht alle "
    "verfuegbaren Labels; "
    "Unbelegtes wird als offener Punkt benannt. "
    + _RENDERING_CAPABILITIES
)

_AGENT_CRITIC_SYSTEM = (
    "Du pruefst ein fertiges Memo gegen Erfolgskriterien und vorberechnete "
    "Fakten. Du urteilst, du misst nicht nach."
)

_AGENT_MEMORY_SYSTEM = (
    "Du extrahierst nur langfristig nuetzliche Arbeits-Erinnerungen aus "
    "einem abgeschlossenen Agent-Lauf. Du speicherst keine Evidenz, keine "
    "Quellenbehauptungen als Wahrheit, keine privaten Dokumentinhalte und "
    "keine Chat-Historie. Antworte ausschliesslich mit JSON."
)


def agent_intake_system_prompt() -> str:
    """System prompt of the intake phase."""
    return _AGENT_INTAKE_SYSTEM


def build_agent_intake_prompt(
    question: str,
    history: str = "",
    *,
    skills_block: str = "",
    artifact_registry: tuple[dict, ...] | list[dict] = (),
    last_response_form: str = "",
    prior_evidence_count: int = 0,
) -> str:
    """Intake analysis over the assignment (Phase 0)."""
    session_context = build_agent_session_context_sections(
        history_block=history,
        artifact_registry=artifact_registry,
        last_response_form=last_response_form,
        prior_evidence_count=prior_evidence_count,
    )
    history_block = f"\n\n{session_context}" if session_context else ""
    skills = (
        "\n\nAktivierte Skills (Nutzerinhalt, praegt Absicht und "
        f"Zielform des Auftrags):\n{skills_block}"
        if skills_block.strip()
        else ""
    )
    return (
        f"Heute ist {today()}.\n\n"
        f"Arbeitsauftrag:\n{question}{history_block}{skills}\n\n"
        "Analysiere den Auftrag: Sprache, Absicht, Klarheit des Umfangs "
        "(clear/underspecified/ambiguous), ob Web-Recherche, interner "
        "Bestand oder Dateien gebraucht werden, Aktualitaets-Sensitivitaet, "
        "Umstrittenheit, Teilziele sowie 3-5 messbare Erfolgskriterien "
        "auf Deutsch.\n\n"
        "Rueckfragen (clarification_questions): NUR wenn ohne die Antwort "
        "keine sinnvolle Bearbeitung moeglich ist, formuliere hoechstens "
        "3 konkrete Rueckfragen. Jede Frage bekommt 2-4 WAHRSCHEINLICHE "
        "Antwortoptionen (kurzes label, description nur wenn das Label "
        "allein nicht reicht); multi_select nur, wenn mehrere Optionen "
        "zugleich zutreffen koennen. Freitext ist fuer den Nutzer immer "
        "zusaetzlich moeglich — die Optionen muessen also nicht "
        "vollstaendig sein, nur die naechstliegenden Antworten abdecken. "
        "Ist der Auftrag klar genug, bleibt die Liste leer.\n\n"
        "Antwortform (response_form): 'chat' fuer konversationale "
        "Antworten, die im Gespraech gelesen werden (Frage beantworten, "
        "Vergleich, Ranking, Einschaetzung, Folgefrage zu einem "
        "bestehenden Ergebnis); 'canvas' fuer eigenstaendige, "
        "wiederverwendbare Dokumente (Bericht, Vermerk, E-Mail, "
        "Sprechzettel — typischerweise laenger und zum Weiterbearbeiten "
        "gedacht). Bezieht sich der Auftrag auf ein bestehendes "
        "Canvas-Dokument ('ueberarbeite ...', 'ergaenze ...'), waehle "
        "'canvas'."
    )


def agent_analyst_system_prompt() -> str:
    """System prompt of the discovery analyst."""
    return _AGENT_ANALYST_SYSTEM


def build_agent_analyst_prompt(
    question: str,
    probe_digest: str,
    sub_goals: list[str],
    *,
    history: str = "",
) -> str:
    """Discovery compression over the probe results (Phase 1).

    Args:
        history: Prior conversation and ANSWERED clarification rounds
            (the state ``history`` block). Without it the analyst
            re-asks what the intake round already clarified — the
            never-re-ask rule below only works when the analyst can
            actually see the answers.
    """
    goals = "\n".join(f"- {goal}" for goal in sub_goals) or "- (keine)"
    answered_block = (
        (
            "Bereits geklaert (Gespraechsverlauf und beantwortete "
            f"Rueckfragen):\n{history.strip()}\n\n"
        )
        if history.strip()
        else ""
    )
    return (
        f"Heute ist {today()}.\n\n"
        f"Arbeitsauftrag:\n{question}\n\nTeilziele:\n{goals}\n\n"
        f"{answered_block}"
        f"Sondierungsergebnisse (komprimiert):\n{probe_digest}\n\n"
        "Leite daraus ab: (1) bereits bekannte Fakten mit Referenz und "
        "Frische-Einschaetzung, (2) Wissensluecken mit Art (missing/"
        "outdated/contradictory/insufficient_detail/unknown_scope), "
        "empfohlenem Werkzeug und Suchvorschlaegen, (3) NUR solche "
        "Nutzerfragen, ohne deren Antwort keine Planung moeglich ist — "
        "stelle KEINE Frage, deren Antwort unter 'Bereits geklaert' "
        "steht, auch nicht umformuliert — "
        "jede mit 2-4 wahrscheinlichen Antwortoptionen (kurzes label, "
        "description nur wenn noetig, multi_select nur bei sinnvoller "
        "Mehrfachauswahl; Freitext bleibt immer moeglich), "
        "(4) ob die Planung starten kann."
    )


def agent_planner_system_prompt() -> str:
    """System prompt of the planner."""
    return _AGENT_PLANNER_SYSTEM


def build_agent_planner_prompt(
    question: str,
    discovery_digest: str,
    success_criteria: list[str],
    *,
    max_tasks: int,
    web_allowed: bool,
    knowledge_allowed: bool = True,
    replan_context: str = "",
    memory_briefing: str = "",
    skills_block: str = "",
    repair_errors: list[str] | None = None,
    collection_catalog: "Sequence[CollectionCatalogEntry] | None" = None,
    research_allowed: bool = False,
    research_profile: str | None = None,
    research_profile_ceiling: str | None = None,
    max_web_instant_tasks: int | None = None,
    replan_mode: bool = False,
    history: str = "",
) -> str:
    """One structured planning call (Phase 3), optional repair round.

    ``collection_catalog`` is the caller-visible knowledge catalog
    (name AND canonical id per collection). ``None`` omits the block
    entirely (no knowledge service wired); an empty catalog is stated
    explicitly so the planner never invents collection references.

    ``history`` carries the conversation and ANSWERED clarification
    rounds: the plan must honor what the user already pinned down
    (market, region, timeframe ...) instead of planning past it.
    """
    criteria = "\n".join(f"- {c}" for c in success_criteria) or "- (keine)"
    answered = (
        "\n\nBereits geklaert (Gespraechsverlauf und beantwortete "
        f"Rueckfragen — der Plan MUSS diese Vorgaben umsetzen):\n"
        f"{history.strip()}"
        if history.strip()
        else ""
    )
    memory = (
        f"\n\nNicht zitierfaehiges Arbeitsgedaechtnis "
        f"(nur Kontext, aktuelle Evidenz gewinnt):\n{memory_briefing}"
        if memory_briefing.strip()
        else ""
    )
    replan = (
        f"\n\nReplan-Kontext (bisherige Tasks bleiben bestehen; "
        f"plane additiv nur fehlende Schritte):\n{replan_context}"
        if replan_context.strip()
        else ""
    )
    if web_allowed and knowledge_allowed:
        source_rule = (
            "Web und Projektwissen sind erlaubt. "
            + _planner_web_rule(
                research_allowed,
                research_profile,
                max_instant_tasks=max_web_instant_tasks,
                max_profile=research_profile_ceiling,
            )
        )
    elif knowledge_allowed:
        source_rule = (
            "Web-Recherche ist NICHT erlaubt: plane ausschliesslich "
            "rag_query/file_analysis-Tasks und vermerke die Einschraenkung "
            "in den Annahmen."
        )
    elif web_allowed:
        source_rule = (
            "Projektwissen ist NICHT erlaubt. "
            + _planner_web_rule(
                research_allowed,
                research_profile,
                max_instant_tasks=max_web_instant_tasks,
                max_profile=research_profile_ceiling,
            )
            + " Vermerke die Einschraenkung in den Annahmen."
        )
    else:
        source_rule = (
            "Web-Recherche und Projektwissen sind NICHT erlaubt. Plane "
            "keine Quellen-Tasks; nutze nur Synthese aus dem vorhandenen "
            "Auftragskontext und benenne die Evidenzluecke ausdruecklich."
        )
    repair = ""
    if repair_errors:
        joined = "\n".join(f"- {error}" for error in repair_errors)
        repair = (
            "\n\nDein vorheriger Plan war ungueltig. Behebe ALLE folgenden "
            f"Fehler:\n{joined}"
        )
    collections = ""
    if collection_catalog is not None:
        listing = "\n".join(
            f"- {entry.name} -> {entry.collection_id}"
            + (
                f" ({entry.embedding_model}, "
                f"{entry.document_count} Dokumente)"
                if entry.embedding_model
                else f" ({entry.document_count} Dokumente)"
            )
            for entry in collection_catalog
        ) or "- (keine Sammlungen sichtbar)"
        collections = (
            f"\n\nVerfuegbare Wissens-Sammlungen (Name -> ID):\n{listing}\n"
            "Regel dazu: params.collection_ids WEGLASSEN, dann nutzt der "
            "Task automatisch alle fuer den Auftrag ausgewaehlten "
            "Sammlungen; nur zum bewussten Verengen setzen und dann "
            "AUSSCHLIESSLICH die IDs (kc_...) aus dieser Liste, niemals "
            "Namen."
        )
    skills = (
        f"\n\nAktivierte Skills und Nutzer-Direktiven (Nutzerinhalt — "
        f"beruecksichtige sie bei Werkzeugwahl und Task-Zuschnitt):\n"
        f"{skills_block}"
        if skills_block.strip()
        else ""
    )
    planning_rules = (
        "Erzeuge AUSSCHLIESSLICH ein Replan-Delta. new_tasks enthaelt "
        "nur neue Ersatz- oder Gap-Tasks mit neuen IDs und niemals einen "
        "synthesis-Task. Wiederhole, aendere oder entferne keine erledigten "
        "Tasks; der Server uebernimmt sie unveraendert und baut die Synthese "
        "neu. skip_task_ids darf nur noch nicht gestartete bestehende Tasks "
        "enthalten. Ist keine zusaetzliche Arbeit erforderlich, bleiben "
        "new_tasks und skip_task_ids leer. "
        if replan_mode
        else (
            "Erzeuge einen Ausfuehrungsplan. Regeln: jeder Gap bekommt "
            "mindestens einen Task ODER eine begruendete Annahme; "
            "outdated-Gaps -> Web-Task mit recency; contradictory-Gaps -> zwei "
            "unabhaengige Tasks; bei strittigen Themen ein Falsifikations-Task "
            "(is_falsification=true); GENAU EIN synthesis-Task, der von allen "
            f"anderen abhaengt; hoechstens {max_tasks} Tasks. "
        )
    )
    return (
        f"Heute ist {today()}.\n\n"
        f"Arbeitsauftrag:\n{question}\n\n"
        f"Erkundungslage:\n{discovery_digest}\n\n"
        f"Erfolgskriterien:\n{criteria}{answered}{memory}{replan}{collections}{skills}\n\n"
        f"{planning_rules}Jeder rag_query- "
        "und web_research-Task traegt 1-8 konkrete Leitfragen. Jeder "
        "web_instant-Task traegt GENAU EINE eigenstaendige, natuerlich "
        "formulierte Evidenzfrage mit Gegenstand, Region (falls relevant), "
        "Zeitraum und gesuchter Evidenz — keine Keyword-Kette und keine "
        "Anbieterliste. Diese eine Frage wird woertlich einmal ausgefuehrt. "
        "Bei einem typischen breit angelegten Markt- oder Analyseauftrag "
        "zielen 3-6 klar getrennte Instant-Evidenzfragen auf die echten "
        "unabhaengigen Gaps; fuelle diese Spanne niemals mit redundanten "
        "oder kuenstlich geteilten Fragen auf. "
        "synthesis traegt keine queries. "
        "Jeder title nennt das ZIEL des Tasks als Verb + Gegenstand "
        "(z. B. 'Marktvolumen 2025 beziffern'), niemals das Werkzeug "
        "oder den Tool-Namen; objective beschreibt in einem Satz den "
        "erwarteten Erkenntnisgewinn. "
        f"{source_rule} rag_query-Profile nach Gap-Tiefe: standard als "
        "Default, gruendlich/tief bei insufficient_detail oder "
        "contradictory. Task-IDs kurz und stabil (t1, t2, ..., s)."
        f"{repair}"
    )


def _planner_web_rule(
    research_allowed: bool,
    research_profile: str | None,
    *,
    max_instant_tasks: int | None = None,
    max_profile: str | None = None,
) -> str:
    """Render the one server-selected Agent-Desk web execution policy.

    Tier budgets (``max_instant_tasks``, ``max_profile``) are STATED
    here and enforced by ``validate_plan`` — never validator-only, so a
    compliant plan does not burn its single repair round on a limit the
    prompt could have named.
    """
    if not research_allowed:
        budget = (
            f"Hoechstens {max_instant_tasks} web_instant-Task(s) in "
            "diesem Lauf. "
            if max_instant_tasks is not None
            else ""
        )
        return (
            "Plane Web-Luecken ausschliesslich als web_instant: eine "
            "eigenstaendige Frage pro Task. web_research ist in diesem "
            f"Lauf nicht erlaubt. {budget}"
            "Falsifikation wird als separate "
            "Instant-Frage nach Gegenbelegen geplant."
        )
    profile = research_profile or "compact"
    ceiling = (
        # Live regression guard: "Pro Task ..." read as EVERY task may
        # carry a profile — gpt-5.4 then stamped profiles onto
        # web_instant tasks, which the validator rejects (and kept doing
        # so through both repair rounds). Name the scope explicitly.
        f" NUR web_research-Tasks duerfen params.profile setzen (bis "
        f"maximal {max_profile}); web_instant-Tasks tragen NIEMALS ein "
        "profile."
        if max_profile is not None
        else ""
    )
    return (
        "web_instant bleibt fuer einzelne Evidenzfragen geeignet. Fuer "
        "eine ausdruecklich mehrstufige Recherche darf web_research mit "
        f"profile={profile} verwendet werden.{ceiling} Seine queries sind "
        "Leitfragen EINES Child-Auftrags; der Child plant seine internen "
        "Suchaufrufe selbst."
    )


def agent_synthesis_system_prompt() -> str:
    """System prompt of the memo synthesis (incl. the rendering SSOT)."""
    return _AGENT_SYNTHESIS_SYSTEM + " " + _RENDERING_CAPABILITIES


def _user_guidance_section(user_guidance: str) -> str:
    """Decision-scoped report guidance from the plan gate (P6)."""
    if not user_guidance.strip():
        return ""
    return (
        "\n\nNutzer-Vorgaben zum Bericht (verbindlich fuer Struktur und "
        f"Schwerpunkte, Sicherheitsregeln nicht):\n{user_guidance.strip()}"
    )


def _skills_prompt_section(skills_block: str) -> str:
    """The shared skills section of the synthesis-side prompts."""
    if not skills_block.strip():
        return ""
    return (
        "\n\nAktivierte Skills (Nutzerinhalt — Form und Ton folgen "
        f"ihnen, Sicherheitsregeln nicht):\n{skills_block}"
    )


def build_agent_outline_prompt(
    question: str,
    success_criteria: list[str],
    evidence_digest: str,
    *,
    prior_memo: str = "",
    skills_block: str = "",
    user_guidance: str = "",
) -> str:
    """Report outline before any section prose (Phase 8)."""
    criteria = "\n".join(
        f"- [{index}] {criterion}"
        for index, criterion in enumerate(success_criteria)
    ) or "- (keine)"
    lineage = (
        "Es existiert bereits ein Memo aus einem frueheren Turn dieser "
        "Sitzung. Setze es fort: behalte tragfaehige Abschnitte bei, "
        "aktualisiere Veraltetes und ergaenze das Neue - beginne NICHT bei "
        f"null.\n\nBisheriges Memo:\n{prior_memo}\n\n"
        if prior_memo.strip()
        else ""
    )
    return (
        f"Heute ist {today()}.\n\n"
        f"Arbeitsauftrag:\n{question}"
        f"{_skills_prompt_section(skills_block)}"
        f"{_user_guidance_section(user_guidance)}\n\n"
        f"{lineage}"
        f"Erfolgskriterien:\n{criteria}\n\n"
        f"Belege (Label -> Kurzinhalt):\n{evidence_digest}\n\n"
        "Plane die Gliederung des Memos: Titel und Abschnitte mit Fokus, "
        "abgedeckten Kriterien (Index als String) und den Belege-Labels, "
        "die der Abschnitt tatsaechlich braucht. Waehle pro Abschnitt die "
        "kleinste hinreichende, nicht redundante Belegmenge statt alle "
        "verfuegbaren Labels zu sammeln. Der letzte Abschnitt heisst "
        "immer 'Offene Punkte'."
    )


def build_agent_section_prompt(
    question: str,
    section_title: str,
    section_focus: str,
    evidence_digest: str,
    contradictions_digest: str,
    *,
    skills_block: str = "",
    user_guidance: str = "",
) -> str:
    """One memo section (Phase 8)."""
    contradictions = (
        f"\n\nWidersprueche (im Text ansprechen, wo relevant):\n"
        f"{contradictions_digest}"
        if contradictions_digest
        else ""
    )
    return (
        f"Arbeitsauftrag:\n{question}"
        f"{_skills_prompt_section(skills_block)}"
        f"{_user_guidance_section(user_guidance)}\n\n"
        f"Schreibe den Memo-Abschnitt '{section_title}'.\n"
        f"Fokus: {section_focus}\n\n"
        f"Verfuegbare Belege:\n{evidence_digest}{contradictions}\n\n"
        "Nur Markdown-Prosa des Abschnitts (ohne Ueberschrift), jede "
        "faktische Aussage mit Belege-Label, woertliche Zitate in "
        "Anfuehrungszeichen. Zitiere die kleinste hinreichende, nicht "
        "redundante Auswahl (typischerweise 1-3 passende Labels pro "
        "faktischer Aussage oder Absatz), niemals alle verfuegbaren Labels."
    )


def build_agent_citation_repair_prompt(
    markdown: str,
    *,
    allowed_labels: list[str],
) -> str:
    """Request one bounded repair of unsupported evidence labels."""
    labels = ", ".join(f"[{label}]" for label in allowed_labels) or "(keine)"
    return (
        "Der folgende vollstaendige Markdown-Text enthaelt mindestens ein "
        "Belege-Label, das nicht im kanonischen Belegledger existiert. "
        "Gib den VOLLSTAENDIGEN Text zurueck. Veraendere keine Aussage, "
        "Struktur, Zahl oder Formatierung. Ersetze ein ungueltiges Label nur "
        "durch ein inhaltlich passendes erlaubtes Label, sofern das aus dem "
        "Text eindeutig hervorgeht; andernfalls entferne ausschliesslich das "
        "ungueltige Label. Erfinde keine Labels.\n\n"
        f"Erlaubte Labels: {labels}\n\n"
        f"Markdown:\n{markdown}"
    )


def agent_answer_system_prompt() -> str:
    """System prompt of the chat-form answer (plan M1 S3)."""
    return _AGENT_ANSWER_SYSTEM


def build_agent_answer_prompt(
    question: str,
    evidence_digest: str,
    contradictions_digest: str,
    *,
    history: str = "",
    prior_memo: str = "",
    skills_block: str = "",
    user_guidance: str = "",
) -> str:
    """The ONE chat-form answer call (deliverable ``chat``).

    Same evidence contract as the memo sections (labels mandatory), but
    the output is a conversational answer read inline in the transcript
    — no outline, no section headings unless they genuinely help.
    """
    history_block = (
        f"\n\nBisheriger Verlauf (aeltere Nachrichten zuerst):\n{history}"
        if history.strip()
        else ""
    )
    memo_block = (
        f"\n\nBestehendes Canvas-Dokument der Sitzung (nur Kontext, "
        f"NICHT neu schreiben):\n{prior_memo}"
        if prior_memo.strip()
        else ""
    )
    contradictions = (
        f"\n\nWidersprueche (transparent ansprechen, wo relevant):\n"
        f"{contradictions_digest}"
        if contradictions_digest
        else ""
    )
    skills = _skills_prompt_section(skills_block)
    guidance = _user_guidance_section(user_guidance)
    return (
        f"Heute ist {today()}.\n\n"
        f"Arbeitsauftrag:\n{question}{history_block}{memo_block}{skills}"
        f"{guidance}\n\n"
        f"Verfuegbare Belege:\n{evidence_digest}{contradictions}\n\n"
        "Beantworte den Auftrag direkt und konversationell als Markdown "
        "(die Antwort erscheint als Chat-Nachricht): kompakt, auf den "
        "Punkt, Struktur nur wo sie hilft (Tabelle fuer Vergleiche/"
        "Rankings, Liste fuer Aufzaehlungen). Jede faktische Aussage "
        "traegt ein Belege-Label; nutze die kleinste hinreichende, nicht "
        "redundante Auswahl (typischerweise 1-3 passende Labels pro "
        "Aussage oder Absatz), niemals alle verfuegbaren Labels. "
        "Unbelegtes als offenen Punkt benennen, nicht erfinden."
    )


def agent_critic_system_prompt() -> str:
    """System prompt of the memo critic."""
    return _AGENT_CRITIC_SYSTEM


def build_agent_critic_prompt(
    memo_markdown: str,
    success_criteria: list[str],
    precomputed_facts: str,
    *,
    user_guidance: str = "",
) -> str:
    """Critic verdict over the memo (Phase 9).

    ``user_guidance`` is the decision-scoped report guidance from the
    plan gate: the critic must judge compliance with it, so a memo that
    ignores the user's stated structure/audience is a finding.
    """
    criteria = "\n".join(f"- {c}" for c in success_criteria) or "- (keine)"
    guidance = _user_guidance_section(user_guidance)
    return (
        f"Memo:\n{memo_markdown}{guidance}\n\n"
        f"Erfolgskriterien:\n{criteria}\n\n"
        f"Vorberechnete Fakten (deterministisch gemessen):\n"
        f"{precomputed_facts}\n\n"
        "Beurteile das Memo: Findings (uncited_claim/criterion_unmet/"
        "contradiction_omitted/instruction_violation/language_error/"
        "memory_conflict) mit konkretem Fix-Vorschlag, erfuellte und "
        "unerfuellte Kriterien, Gesamturteil: pass wenn das Memo tragfaehig "
        "ist; revise wenn die Mangel durch genau eine Textrevision behebbar "
        "sind; research wenn neue Recherche noetig ist, weil Erfolgskriterien "
        "ohne weitere Evidenz nicht sauber erfuellt werden koennen. Falls "
        "ein Memory-Briefing den aktuellen Belegen widerspricht, melde "
        "memory_conflict; aktuelle Evidenz gewinnt immer und Memory ist nie "
        "zitierfaehig."
    )


def build_agent_file_analysis_prompt(objective: str, content: str) -> str:
    """Quarantined file analysis (in-process fallback of the harness)."""
    return (
        f"Analysiere den folgenden Dokumentinhalt fuer dieses Ziel:\n"
        f"{objective}\n\n"
        f"Dokument:\n{content}\n\n"
        "Fasse die relevanten Befunde kompakt zusammen (max. 300 Woerter) "
        "und liste zitierwuerdige woertliche Passagen."
    )


def agent_memory_system_prompt() -> str:
    """System prompt for candidate-only memory reflection."""
    return _AGENT_MEMORY_SYSTEM


def build_agent_memory_reflection_prompt(
    *,
    question: str,
    memo_markdown: str,
    critic_digest: str,
    task_digest: str,
) -> str:
    """Ask for safe memory candidates after a completed agent run."""
    return (
        f"Arbeitsauftrag:\n{question}\n\n"
        f"Task-Ergebnisse:\n{task_digest}\n\n"
        f"Kritik/Qualitaet:\n{critic_digest}\n\n"
        f"Memo-Auszug:\n{memo_markdown[:4000]}\n\n"
        "Erzeuge hoechstens 3 Memory-Kandidaten, nur wenn sie kuenftige "
        "Arbeit klar verbessern. Erlaubt sind Nutzerpraeferenzen, stabile "
        "Projektkontexte, verworfene Architekturentscheidungen, erfolgreiche "
        "Arbeitsstrategien und korrigierte Annahmen. Verboten sind "
        "Roh-Evidence, Quellenbehauptungen als Wahrheit, private "
        "Dokumentinhalte, Secrets, temporaere Tool-Ausgaben und komplette "
        "Chat-Historie. Wenn nichts geeignet ist: candidates=[]"
    )

def build_agent_patch_instruction(question: str, *, has_memo: bool) -> str:
    """Instruction for the patch-proposal instruct call (phase M7).

    Feeds the SAME instruct pipeline as ``/v1/editor/instruct``, so this
    only frames the assignment; schema and anchor rules come from the
    shared prompt there.
    """
    memo_note = (
        " Nutze das beigefuegte Memo [Memo] als Quellenmaterial fuer "
        "Fakten und Belege."
        if has_memo
        else ""
    )
    return (
        "Arbeite den folgenden Auftrag in das Dokument ein, mit so wenigen, "
        "praezise verankerten Aenderungen wie moeglich. Erhalte Stil und "
        f"Struktur des Dokuments.{memo_note}\n\nAuftrag: {question}"
    )


_KERNEL_ROLE = (
    "Du bist der Inqtrix-Agent: ein sorgfaeltiger Assistent fuer "
    "Recherche-, Wissens- und Schreibauftraege. Du arbeitest auf Deutsch "
    "(ausser der Nutzer verlangt eine andere Sprache), praezise und ohne "
    "Floskeln."
)

_KERNEL_OUTPUT_ROUTING = (
    "Ausgabeform: Antworte IM CHAT, wenn die Antwort konversationell ist "
    "(Erklaerung, Einschaetzung, kurze Auskunft, grob unter 15 Zeilen). "
    "Schreibe ein CANVAS-Dokument (write_canvas), wenn das Ergebnis "
    "eigenstaendig und wiederverwendbar ist (Memo, E-Mail-Entwurf, "
    "Sprechzettel, Bericht). Ein expliziter Wunsch des Nutzers schlaegt "
    "diese Kriterien immer. Folgeauftraege, die sich auf ein bestehendes "
    "Dokument beziehen, AKTUALISIEREN genau dieses Dokument (write_canvas "
    "mit artifact_id und expected_revision aus dem Sitzungskontext) — im "
    "Chat gibst du dann nur eine kurze Aenderungsnotiz. Bei Unklarheit, "
    "welches Dokument gemeint ist: frage mit ask_user nach."
)

_KERNEL_CLARIFICATION_RULES = (
    "Rueckfragen: Stelle eine Rueckfrage (ask_user) NUR, wenn eine "
    "materiell blockierende Information fehlt, die das Ergebnis "
    "wesentlich veraendert. Gib 2-4 wahrscheinliche Optionen und eine "
    "Default-Annahme an. Hoechstens zwei Rueckfrage-Runden pro Auftrag; "
    "danach arbeitest du mit deiner besten Annahme und benennst sie "
    "sichtbar in der Antwort. Im Auto-Modus bevorzugst du die sichtbar "
    "benannte Annahme statt einer Rueckfrage."
)

_KERNEL_TOOL_DISCIPLINE = (
    "Werkzeugdisziplin: Nutze das kleinste Werkzeug, das den Zweck "
    "erfuellt — search_project_knowledge fuer internes Wissen, "
    "read_project_document fuer den Volltext eines Treffers, web_instant "
    "fuer EINE gezielte externe Suche. Delegiere an run_deep_mission nur "
    "bei Auftraegen mit mehreren Recherche-Straengen, zitierter "
    "Multi-Quellen-Evidenz, strittigen Aussagen oder explizitem "
    "Berichtswunsch; run_web_research fuer eine einzelne mehrstufige "
    "Webrecherche. Aenderungen an Editor-Dokumenten des Nutzers "
    "schlaegst du ausschliesslich ueber propose_editor_patch vor — sie "
    "werden nie direkt angewendet, der Nutzer prueft sie im Editor. "
    "Vor der Ueberarbeitung eines bestehenden Canvas-Dokuments liest du "
    "mit read_canvas immer dessen aktuellen Inhalt, Revision und Belege. "
    "An write_canvas gibst du nur reference_ids weiter, die ein Inqtrix-"
    "Werkzeug geliefert hat. "
    "Direkt (ohne Delegation) erledigst du: Instant-Antworten, bis zu "
    "zwei Suchen, Entwuerfe/Umformulierungen und Canvas-Aenderungen. "
    "Wird ein Werkzeug abgelehnt oder ist nicht verfuegbar, erkennst du "
    "das an und benennst die Luecke in der Antwort — erfinde niemals "
    "Ergebnisse. Bei Auftraegen mit drei oder mehr Schritten pflegst du "
    "write_todos."
)

_KERNEL_THINKING_VS_SPEAKING = (
    "Denken vs. Sprechen: Bevor du Werkzeuge aufrufst, schreibe EINEN "
    "kurzen Absichtssatz fuer den Nutzer (was du jetzt tust und warum). "
    "Keine inneren Monologe, keine rohen Gedankengaenge."
)

_KERNEL_LIMITS = (
    "Grenzen: Du hast ein hartes Schritt- und Token-Budget. Arbeite "
    "zielgerichtet, wiederhole fehlgeschlagene Aufrufe nicht unveraendert "
    "und liefere lieber eine ehrliche Teilantwort mit benannten Luecken "
    "als gar keine."
)


def build_agent_kernel_system_prompt() -> str:
    """The kernel loop's system prompt (plan M2 `2.6`, static parts).

    Per-run context (session history, artifact registry, response-form
    hint) travels in the USER message instead
    (:func:`build_kernel_user_message`) — the compiled graph variants
    are shared across runs, so the system prompt must stay run-free.
    """
    return "\n\n".join(
        (
            _KERNEL_ROLE,
            "Darstellung: " + _RENDERING_CAPABILITIES,
            _KERNEL_OUTPUT_ROUTING,
            _KERNEL_CLARIFICATION_RULES,
            _KERNEL_TOOL_DISCIPLINE,
            _KERNEL_THINKING_VS_SPEAKING,
            _KERNEL_LIMITS,
        )
    )


def build_deep_review_prompt(
    effective_assignment: str, output_bundle: str
) -> str:
    """The Deep rubric over the full assignment and every run output.

    Checkable criteria only; findings must be concrete and fixable —
    an empty list means the answer passes and no revision runs.
    """
    return (
        f"Vollstaendiger effektiver Auftrag:\n{effective_assignment}\n\n"
        f"Zu pruefendes Output-Bundle:\n{output_bundle}\n\n"
        "Pruefe ALLE Outputs gegen diese Rubrik und melde NUR konkrete, "
        "behebbare Maengel:\n"
        "1. Vollstaendigkeit: Beantwortet sie den Auftrag vollstaendig?\n"
        "2. Belegbarkeit: Tragen faktische Aussagen Belege-Labels oder "
        "Quellen, und ist Unbelegtes ehrlich als offen benannt?\n"
        "3. Widersprueche: Sind Unsicherheiten und Gegenpositionen "
        "benannt, wo sie relevant sind?\n\n"
        "Jeder Befund zielt auf chat oder auf eine konkrete, im Bundle "
        "bekannte artifact_id. artifact_id ist fuer chat leer. findings "
        "bleibt LEER, wenn alle Outputs bestehen — erfinde keine Maengel "
        "und fordere nichts, was der Auftrag nicht verlangt."
    )


def build_deep_revision_prompt(
    effective_assignment: str,
    output_bundle: str,
    findings: list[dict],
) -> str:
    """The one Deep revision call over chat and targeted canvases."""
    listed = "\n".join(
        f"- target={item.get('target')} "
        f"artifact_id={item.get('artifact_id') or '-'}: "
        f"{item.get('finding')}"
        for item in findings
    )
    return (
        f"Vollstaendiger effektiver Auftrag:\n{effective_assignment}\n\n"
        f"Bisheriges Output-Bundle:\n{output_bundle}\n\n"
        f"Ein Pruefdurchlauf fand diese Maengel:\n{listed}\n\n"
        "Gib genau eine Revision-Bundle aus. chat_markdown ist IMMER der "
        "vollstaendige Chat-Text. artifacts enthaelt nur bekannte, wirklich "
        "zu aendernde artifact_ids mit ihrer EXAKTEN bisherigen Revision "
        "und dem vollstaendigen neuen Markdown. Behebe nur die genannten "
        "Maengel; Payloads und Beleg-Referenzen duerfen nicht veraendert "
        "werden."
    )


def build_agent_session_context_sections(
    *,
    history_block: str = "",
    artifact_registry: tuple[dict, ...] | list[dict] = (),
    last_response_form: str = "",
    prior_evidence_count: int = 0,
) -> str:
    """Render the shared K1-K4 session context for both agent engines."""
    sections: list[str] = []
    if history_block:
        sections.append(f"Bisheriger Verlauf der Sitzung:\n{history_block}")
    if artifact_registry:
        lines = "\n".join(
            f"- {item.get('title', '(ohne Titel)')} "
            f"(artifact_id {item.get('artifact_id')}, Revision "
            f"{item.get('revision')}, zuletzt durch "
            f"{item.get('updated_by')})"
            for item in artifact_registry
        )
        sections.append(
            "Vorhandene Canvas-Dokumente dieser Sitzung (fuer Updates "
            f"artifact_id + Revision verwenden):\n{lines}"
        )
    if last_response_form:
        sections.append(
            f"Letzte Ausgabeform dieser Sitzung: {last_response_form}."
        )
    if prior_evidence_count:
        sections.append(
            f"Aus frueheren Runden sind {prior_evidence_count} "
            "unterschiedliche Belege in Canvas-Artefakten verfuegbar."
        )
    return "\n\n".join(sections)


def build_kernel_user_message(
    question: str,
    *,
    history_block: str = "",
    artifact_registry: tuple[dict, ...] | list[dict] = (),
    last_response_form: str = "",
    prior_evidence_count: int = 0,
    response_form: str = "",
    autonomy: str = "",
    depth: str = "",
    tier: str = "",
    skills_block: str = "",
    tool_directives_line: str = "",
) -> str:
    """The per-run user message: session context + assignment (K1-K4)."""
    sections: list[str] = []
    session_context = build_agent_session_context_sections(
        history_block=history_block,
        artifact_registry=artifact_registry,
        last_response_form=last_response_form,
        prior_evidence_count=prior_evidence_count,
    )
    if session_context:
        sections.append(session_context)
    if response_form in ("chat", "canvas"):
        label = (
            "Chat-Antwort" if response_form == "chat" else "Canvas-Dokument"
        )
        sections.append(
            f"Der Nutzer verlangt die Ausgabeform {label} — sie schlaegt "
            "alle Routing-Kriterien."
        )
    if autonomy == "autonomous":
        sections.append(
            "Modus: Auto — bevorzuge sichtbar benannte Annahmen statt "
            "Rueckfragen."
        )
    if depth == "deep":
        sections.append(
            "Deep-Modus: Gruendlichkeit vor Tempo. Bevorzuge "
            "run_deep_mission fuer recherchierte Deliverables mit "
            "mehreren Straengen und run_web_research (gruendlich) statt "
            "web_instant; benenne Annahmen ausdruecklich und belege "
            "jede faktische Aussage. Vor der Finalisierung prueft ein "
            "zusaetzlicher Verifikations-Durchlauf deine Antwort."
        )
    if tier == "schnell":
        # The mission machine enforces schnell deterministically; the
        # kernel is prompt-driven by design, and its web budget is
        # already cut server-side (research policy: no children).
        sections.append(
            "Stufe: Schnell — Tempo vor Vollstaendigkeit. Stelle KEINE "
            "Rueckfragen (kein ask_user; triff sichtbar benannte "
            "Annahmen), fuehre hoechstens EINE Websuche aus, nutze "
            "Projektwissen nur wenn der Auftrag ausdruecklich Sammlungen "
            "referenziert, und antworte direkt im Chat statt ein "
            "Canvas-Dokument zu eroeffnen."
        )
    if skills_block:
        sections.append(skills_block)
    if tool_directives_line:
        sections.append(tool_directives_line)
    sections.append(f"Auftrag:\n{question}")
    return "\n\n".join(sections)
