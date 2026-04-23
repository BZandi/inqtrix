""" Domain whitelists for source tiering and quality-site injection."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# DE policy-domain heuristic (shared)
#
# Several pipeline stages need to know whether a question targets German
# health/social-policy topics so they can apply specialized behaviour:
# quality-site query injection (plan), utility-stop suppression
# (stop_criteria). The regex lives here so all stages share the same
# definition; without that, the two stages drift and the agent reacts
# inconsistently to the same question.
#
# This is a deliberately narrow, German-specific bias. It is gated by
# AgentSettings.enable_de_policy_bias so deployments outside that domain
# can disable it.
# ---------------------------------------------------------------------------

_DE_POLICY_PATTERN = re.compile(
    r"\b(privatis\w*|gkv|krankenkass\w*|gesetz\w*|recht\w*|verordnung\w*"
    r"|regulier\w*|politik\w*|beitrag\w*|kosten|haushalt\w*)\b",
    re.IGNORECASE,
)


def is_de_policy_question(question: str) -> bool:
    """Return True when the question matches the German policy regex."""
    if not question:
        return False
    return bool(_DE_POLICY_PATTERN.search(question.lower()))

LANG_NAMES: dict[str, str] = {
    "de": "Deutsch", "en": "Englisch", "fr": "Franzoesisch",
    "es": "Spanisch", "it": "Italienisch", "pt": "Portugiesisch",
}

LOW_QUALITY_DOMAINS: list[str] = [
    "-pinterest.com",
    "-pinterest.de",
    "-quora.com",
    "-tiktok.com",
    "-reddit.com",
    "-medium.com",
    "-gutefrage.net",
]

PRIMARY_SOURCE_DOMAINS: set[str] = {
    # Bundesregierung & Ministerien
    "bundesregierung.de",
    "bundesgesundheitsministerium.de",
    "bmg.bund.de",
    "bundestag.de",
    "dip.bundestag.de",            # Dokumentations- und Informationssystem für Parlamentsmaterialien
    "das-parlament.de",            # Bundestag-eigenes Wochenmagazin
    "gesetze-im-internet.de",
    "gkv-spitzenverband.de",
    "bundesrechnungshof.de",       # unabhängige Finanzkontrolle des Bundes
    "abgeordnetenwatch.de",        # parlamentarische Transparenz / amtliche Daten
    # EU & internationale Institutionen
    "ec.europa.eu",
    "europa.eu",
    "who.int",
    "oecd.org",
}

MAINSTREAM_SOURCE_DOMAINS: set[str] = {
    # Deutsche Medien (Tagespresse)
    "aerzteblatt.de",
    "aerztezeitung.de",            # Mainstream-Fachmedium Medizin
    "deutsche-apotheker-zeitung.de",  # Mainstream-Fachmedium Pharma
    "deutschlandfunk.de",
    "tagesspiegel.de",
    "handelsblatt.com",
    "spiegel.de",
    "zdfheute.de",
    "tagesschau.de",
    "stern.de",
    "zeit.de",
    "faz.net",
    "sueddeutsche.de",
    "welt.de",
    "n-tv.de",
    "focus.de",                    # Mainstream-News-Magazin
    "taz.de",                      # Mainstream-Tageszeitung
    # Wirtschafts-/Finanzmedien Deutsch
    "dasinvestment.com",
    "finanztip.de",                # unabhängige Verbraucher-Finanzberatung
    # Datenservices
    "de.statista.com",
    "statista.com",
    # Internationale Nachrichtenagenturen
    "reuters.com",
    "apnews.com",
    "ap.org",
    # US/UK Mainstream
    "nytimes.com",
    "washingtonpost.com",
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "cnn.com",
    "abcnews.go.com",
    "nbcnews.com",
    "cbsnews.com",
    "npr.org",
    "pbs.org",
    # Wirtschaftsmedien international
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "economist.com",
    "businessinsider.com",
    "fortune.com",
    "forbes.com",
    "cnbc.com",
    "marketwatch.com",
    "economictimes.com",
    "morningstar.com",             # Investment-Research / Mainstream-Marktanalyse
    "zacks.com",                   # Investment-Research / Mainstream-Marktanalyse
    # Tech-Medien
    "techcrunch.com",
    "arstechnica.com",
    "theverge.com",
    "wired.com",
    "infoq.com",
    # Wissenschaft
    "nature.com",
    "science.org",
    "scientificamerican.com",
    "arxiv.org",
}

STAKEHOLDER_SOURCE_DOMAINS: set[str] = {
    # Selbstverwaltung & Verbände im Gesundheitswesen
    "kzbv.de",
    "bzaek.de",
    "kbv.de",                      # Kassenärztliche Bundesvereinigung
    "vdek.com",
    "aok.de",
    "pkv.de",                      # PKV-Verband
    "deutscher-pflegerat.de",
    "physio-deutschland.de",
    "marburger-bund.de",
    "dkgev.de",                    # Deutsche Krankenhausgesellschaft
    "arbeitgeber.de",              # BDA
    # Parteien (Positionen offizieller Parteiorganisationen)
    "cdu.de",
    "spd.de",
    "gruene.de",
    "fdp.de",
    "linke.de",
    "csu.de",
}

SOURCE_TIER_WEIGHTS: dict[str, float] = {
    "primary": 1.0,
    "mainstream": 0.8,
    "stakeholder": 0.45,
    "unknown": 0.35,
    "low": 0.1,
}

# Quality-site lists for German policy questions
QUALITY_PRIMARY_SITES_DE: list[str] = [
    "bundesgesundheitsministerium.de",
    "bundesregierung.de",
    "bundestag.de",
    "gkv-spitzenverband.de",
    "gesetze-im-internet.de",
]

QUALITY_MAINSTREAM_SITES_DE: list[str] = [
    "aerzteblatt.de",
    "deutschlandfunk.de",
    "zdfheute.de",
    "spiegel.de",
    "handelsblatt.com",
    "tagesschau.de",
]

GENERIC_QUERY_TERMS_DE: set[str] = {
    "soll", "sollen", "sollte", "sollten", "werden", "wird", "wurde",
    "zukuenftig", "kuenftig", "zukunft", "aktuell", "heute",
    "richtung", "diskussion", "diskussionen", "debatte", "debatten",
    "geht", "gehen", "genau", "eigentlich",
}
