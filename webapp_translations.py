"""UI translation strings for webapp.py. UI-only — never sent to the agent.

The agent backend (prompts.py) stays German regardless of the UI language.
SUGGESTIONS prompt-text and the file-upload fallback prompt also stay German
on purpose so the agent receives consistent input.
"""

TRANSLATIONS: dict[str, dict[str, str]] = {
    # --- Header buttons ---
    "btn_new":            {"de": "Neu",      "en": "New"},
    "btn_new_help":       {"de": "Neuen Chat starten", "en": "Start a new chat"},
    "btn_delete":         {"de": "Löschen",  "en": "Clear"},
    "btn_delete_help":    {"de": "Alle Nachrichten löschen", "en": "Delete all messages"},

    # --- Language toggle tooltip (shown on the button itself) ---
    "lang_toggle_to_en":  {"de": "Switch to English", "en": "Switch to English"},
    "lang_toggle_to_de":  {"de": "Auf Deutsch umschalten", "en": "Auf Deutsch umschalten"},

    # --- Sidebar: Server ---
    "sidebar_server":     {"de": "### Server",      "en": "### Server"},
    "health_ok":          {"de": "bereit",          "en": "ready"},
    "health_degraded":    {"de": "degradiert",      "en": "degraded"},
    "health_unreachable": {"de": "nicht erreichbar", "en": "unreachable"},
    "auth_set":           {"de": "Auth: Bearer gesetzt", "en": "Auth: Bearer set"},
    "auth_none":          {"de": "Auth: keine",     "en": "Auth: none"},
    "btn_refresh":        {"de": "Aktualisieren",   "en": "Refresh"},

    # --- Sidebar: Einstellungen ---
    "sidebar_settings":   {"de": "### Einstellungen", "en": "### Settings"},
    "toggle_streaming":   {"de": "Streaming",        "en": "Streaming"},
    "toggle_progress":    {"de": "Progress-Events",  "en": "Progress events"},

    # --- Welcome (whole HTML block per language) ---
    "welcome_html": {
        "de": (
            '<div class="welcome-wrap">'
            '<div class="prototype-notice">Hinweis: Hierbei handelt sich um einen Prototypen</div>'
            '<div class="welcome-meta">'
            'GitHub: <a href="https://github.com/BZandi/inqtrix" target="_blank" rel="noopener">github.com/BZandi/inqtrix</a>'
            '<span class="sep">·</span>'
            'Lizenz: Apache 2.0'
            '</div>'
            '<div class="welcome-title">Guten Tag.</div>'
            '<div class="welcome-sub">Wie kann inqtrix Ihnen heute helfen?</div>'
            '</div>'
        ),
        "en": (
            '<div class="welcome-wrap">'
            '<div class="prototype-notice">Note: This is a prototype</div>'
            '<div class="welcome-meta">'
            'GitHub: <a href="https://github.com/BZandi/inqtrix" target="_blank" rel="noopener">github.com/BZandi/inqtrix</a>'
            '<span class="sep">·</span>'
            'License: Apache 2.0'
            '</div>'
            '<div class="welcome-title">Hello.</div>'
            '<div class="welcome-sub">How can inqtrix help you today?</div>'
            '</div>'
        ),
    },

    # --- Suggestion pills (display only — prompt-text stays German) ---
    "pills_label":             {"de": "Vorschlaege:",                 "en": "Suggestions:"},
    "suggestion_ai_news":      {"de": "Suche aktuelle KI-Nachrichten", "en": "Search latest AI news"},
    "suggestion_explain_rag":  {"de": "Erkläre RAG",                  "en": "Explain RAG"},
    "suggestion_compare_llms": {"de": "Vergleiche Claude vs GPT-4",   "en": "Compare Claude vs GPT-4"},

    # --- Composer ---
    "composer_placeholder":   {"de": "Wie kann ich helfen?", "en": "How can I help?"},
    "composer_prototype_pill": {"de": "Hinweis: Prototyp",   "en": "Note: Prototype"},

    # --- Profile popover ---
    "popover_profile":        {"de": "Profil",   "en": "Profile"},
    "popover_profile_help":   {"de": "Report-Profil, Antworttiefe und DE-Policy-Heuristik.",
                                "en": "Report profile, answer depth and DE-policy heuristic."},
    "label_profile":          {"de": "Profil",   "en": "Profile"},
    "label_research_mode":    {"de": "Recherchemodus", "en": "Research mode"},
    "label_de_policy":        {"de": "DE-Policy-Heuristik", "en": "DE-policy heuristic"},
    "toggle_de_policy":       {"de": "DE-Policy-Bias aktiv", "en": "DE-policy bias active"},
    "toggle_de_policy_help":  {
        "de": "Serverseitiges Flag `enable_de_policy_bias`. Wirkt auf `KeywordRiskScorer` und die Stop-Kriterien. Default: an.",
        "en": "Server-side flag `enable_de_policy_bias`. Affects `KeywordRiskScorer` and the stop criteria. Default: on.",
    },

    # Profile labels (REPORT_PROFILE_LABELS)
    "profile_compact":        {"de": "Kompakt",       "en": "Compact"},
    "profile_deep":           {"de": "Deep Research", "en": "Deep Research"},

    # Profile explainer (markdown HTML block)
    "profile_explainer_html": {
        "de": (
            '<div class="popover-explainer">'
            'Bestimmt <strong>Tiefe und Länge</strong> der Antwort. '
            '<strong>Kompakt</strong> liefert eine schnelle, knappe '
            'Synthese mit kleinerem Kontextfenster und wenigen Zitaten. '
            '<strong>Deep Research</strong> fächert Suchanfragen '
            'breiter auf, sammelt deutlich mehr Quellen und erzeugt '
            'eine ausführlichere, gegliederte Synthese — dauert aber '
            'entsprechend länger.'
            '</div>'
        ),
        "en": (
            '<div class="popover-explainer">'
            'Determines the <strong>depth and length</strong> of the answer. '
            '<strong>Compact</strong> delivers a fast, concise '
            'synthesis with a smaller context window and few citations. '
            '<strong>Deep Research</strong> fans out search queries '
            'more broadly, collects significantly more sources and produces '
            'a more detailed, structured synthesis — but takes '
            'correspondingly longer.'
            '</div>'
        ),
    },

    # DE-Policy explainer (markdown HTML block)
    "de_policy_explainer_html": {
        "de": (
            '<div class="popover-explainer">'
            'Aktiviert die <strong>deutsche Gesundheits- und '
            'Sozialpolitik-Kalibrierung</strong> des Agenten: '
            'Quality-Site-Injection für DE-Policy-Domänen, '
            'Utility-Stop-Unterdrückung bei politischen Themen '
            'und einen <strong>Risiko-Bonus (+2)</strong> für '
            'einschlägige Keywords. <strong>Ausschalten</strong> '
            'für allgemeine oder nicht-deutsche Einsätze, um den '
            'DE-spezifischen Bias zu entfernen.'
            '</div>'
        ),
        "en": (
            '<div class="popover-explainer">'
            'Activates the agent\'s <strong>German health and '
            'social policy calibration</strong>: '
            'quality-site injection for DE-policy domains, '
            'utility-stop suppression on political topics '
            'and a <strong>risk bonus (+2)</strong> for '
            'relevant keywords. <strong>Disable</strong> '
            'for generic or non-German use cases to remove '
            'the DE-specific bias.'
            '</div>'
        ),
    },

    # --- Model / Stack popover ---
    "popover_model":          {"de": "Modell",  "en": "Model"},
    "popover_model_help":     {"de": "Aktiver Stack und Provider-Verfügbarkeit.",
                                "en": "Active stack and provider availability."},
    "label_stack":            {"de": "Stack",   "en": "Stack"},
    "stack_select_help": {
        "de": (
            "**●** — Provider-Readiness-Check war erfolgreich.\n\n"
            "**○** — Provider antwortet aktuell nicht.\n\n"
            "Die Liste stammt aus `GET /v1/stacks` und wird "
            "bei jedem Seitenaufbau serverseitig (5 s Cache) "
            "über den ADR-WS-12-Readiness-Probe erneuert."
        ),
        "en": (
            "**●** — provider readiness check succeeded.\n\n"
            "**○** — provider is currently not responding.\n\n"
            "The list comes from `GET /v1/stacks` and is "
            "refreshed on every page load server-side (5 s cache) "
            "via the ADR-WS-12 readiness probe."
        ),
    },
    "stack_count_available":  {"de": "verfügbar", "en": "available"},
    "warn_no_stack":          {"de": "Kein Stack verfügbar — Server nicht erreichbar?",
                                "en": "No stack available — server unreachable?"},
    "stack_chip_search_model": {"de": "Search-Modell", "en": "Search model"},

    # --- Effort popover ---
    "popover_effort":         {"de": "Aufwand",  "en": "Effort"},
    "popover_effort_help":    {"de": "Recherche-Tiefe und Feintuning-Schranken.",
                                "en": "Research depth and fine-tuning thresholds."},
    "label_effort":           {"de": "Aufwand",  "en": "Effort"},
    "label_finetuning":       {"de": "Feintuning", "en": "Fine-tuning"},
    "label_current":          {"de": "Aktuell",  "en": "Current"},

    # Effort levels (display only)
    "effort_auto":            {"de": "Auto",    "en": "Auto"},
    "effort_low":             {"de": "Niedrig", "en": "Low"},
    "effort_medium":          {"de": "Mittel",  "en": "Medium"},
    "effort_high":            {"de": "Hoch",    "en": "High"},
    "effort_max":             {"de": "Max",     "en": "Max"},

    "effort_explainer_html": {
        "de": (
            '<div class="popover-explainer">'
            'Steuert, wie viele Recherche-Runden der Agent höchstens '
            'und mindestens ausführt. Jede Runde verfeinert Suchanfragen '
            'und sammelt weitere Quellen. <strong>Mehr Runden</strong> '
            '= tiefere Recherche, aber längere Laufzeit.'
            '</div>'
        ),
        "en": (
            '<div class="popover-explainer">'
            'Controls the maximum and minimum number of research rounds '
            'the agent runs. Each round refines search queries '
            'and collects additional sources. <strong>More rounds</strong> '
            '= deeper research, but longer runtime.'
            '</div>'
        ),
    },

    "effort_help": {
        "de": (
            "Steuert den Recherche-Aufwand des Agenten über `max_rounds` (obere "
            "Schleifen-Schranke) und `min_rounds` (untere Schranke).\n\n"
            "• **Auto** — Server-Default (ergibt sich aus dem Profil).\n"
            "• **Niedrig** — max=1, min=1 (einziger Durchlauf, schnellste Antwort).\n"
            "• **Mittel** — max=2, min=1.\n"
            "• **Hoch** — max=4, min=2 (solide Standardwahl für Deep).\n"
            "• **Max** — max=6, min=3 (maximal tief, langsamste Antwort)."
        ),
        "en": (
            "Controls the agent's research effort via `max_rounds` (upper "
            "loop bound) and `min_rounds` (lower bound).\n\n"
            "• **Auto** — server default (derived from the profile).\n"
            "• **Low** — max=1, min=1 (single pass, fastest answer).\n"
            "• **Medium** — max=2, min=1.\n"
            "• **High** — max=4, min=2 (solid default for Deep).\n"
            "• **Max** — max=6, min=3 (deepest, slowest answer)."
        ),
    },

    "effort_server_default":  {"de": "Server-Default (richtet sich nach Profil)",
                                "en": "Server default (follows the profile)"},

    # Sliders (effort popover)
    "slider_confidence_stop": {"de": "Confidence Stop", "en": "Confidence Stop"},
    "slider_confidence_stop_help": {
        "de": "Evaluator-Confidence (1-10), ab der die Stop-Kaskade 'done' setzen darf.",
        "en": "Evaluator confidence (1-10) at which the stop cascade may set 'done'.",
    },
    "slider_max_seconds":     {"de": "Max Total Seconds", "en": "Max Total Seconds"},
    "slider_max_seconds_help": {
        "de": "Hartes Wall-Clock-Limit pro Durchlauf (Sekunden).",
        "en": "Hard wall-clock limit per run (seconds).",
    },
    "slider_first_round":     {"de": "First Round Queries", "en": "First Round Queries"},
    "slider_first_round_help": {
        "de": "Anzahl der breiten Suchanfragen in Runde 0.",
        "en": "Number of broad search queries in round 0.",
    },

    # --- Web search toggle ---
    "toggle_web_search":      {"de": "Websuche", "en": "Web search"},
    "toggle_web_search_help": {
        "de": (
            "**An** (Standard): Der Agent führt Websuche, "
            "Quellensynthese und Claim-Verifikation durch.\n\n"
            "**Aus**: Reiner Chat — der Server ruft nur den "
            "LLM-Provider direkt mit Frage + Historie, ohne "
            "Recherche. Ideal für Definitions- oder "
            "Konzeptfragen, die keine aktuellen Quellen brauchen."
        ),
        "en": (
            "**On** (default): The agent performs web search, "
            "source synthesis and claim verification.\n\n"
            "**Off**: Pure chat — the server only calls the "
            "LLM provider directly with question + history, without "
            "research. Ideal for definition or "
            "concept questions that don't need current sources."
        ),
    },

    # --- Settings detail popover ---
    "popover_settings_help":  {"de": "Alle aktiven Einstellungen anzeigen",
                                "en": "Show all active settings"},
    "settings_active":        {"de": "Aktive Einstellungen", "en": "Active settings"},
    "settings_value_on":      {"de": "an",  "en": "on"},
    "settings_value_off":     {"de": "aus", "en": "off"},

    "row_label_stack":        {"de": "Stack",     "en": "Stack"},
    "row_label_profile":      {"de": "Profil",    "en": "Profile"},
    "row_label_effort":       {"de": "Aufwand",   "en": "Effort"},
    "row_label_websearch":    {"de": "Websuche",  "en": "Web search"},
    "row_label_de_policy":    {"de": "DE-Policy", "en": "DE-policy"},
    "row_label_streaming":    {"de": "Streaming", "en": "Streaming"},
    "row_label_progress":     {"de": "Progress-Events", "en": "Progress events"},

    "settings_finetuning_caption": {
        "de": "Feintuning: confidence_stop = {confidence} · max_total_seconds = {max_seconds} · first_round_queries = {first_round}",
        "en": "Fine-tuning: confidence_stop = {confidence} · max_total_seconds = {max_seconds} · first_round_queries = {first_round}",
    },

    # --- Stop button + research status ---
    "btn_stop":               {"de": "Recherche stoppen", "en": "Stop research"},
    "btn_stop_help": {
        "de": (
            "Bricht den laufenden Recherche-Request ab. "
            "Der Server erkennt den Disconnect und cancelt "
            "den Agent-Run innerhalb weniger hundert "
            "Millisekunden."
        ),
        "en": (
            "Cancels the running research request. "
            "The server detects the disconnect and cancels "
            "the agent run within a few hundred "
            "milliseconds."
        ),
    },
    "status_researching":     {"de": "Recherche läuft...", "en": "Researching..."},
    "status_progress_label":  {"de": "Rechercheverlauf · {count} Schritt(e)",
                                "en": "Research log · {count} step(s)"},
    "status_error_label":     {"de": "Fehler: {message}", "en": "Error: {message}"},
    "research_cancelled":     {"de": "_Recherche abgebrochen._", "en": "_Research cancelled._"},

    # --- Errors / warnings ---
    "error_streaming":        {"de": "Streaming-Fehler: {error}", "en": "Streaming error: {error}"},
    "error_unknown":          {"de": "Unbekannter Fehler", "en": "Unknown error"},
    "error_no_choices":       {"de": "Antwort enthielt keine 'choices'",
                                "en": "Response contained no 'choices'"},
    "warn_partial":           {"de": "Vorzeitig beendet: {message}",
                                "en": "Terminated early: {message}"},

    # --- Copy button (aria-label / title) ---
    "btn_copy":               {"de": "Kopieren", "en": "Copy"},
}
