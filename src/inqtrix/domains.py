""" Domain whitelists for source tiering and quality-site injection."""

from __future__ import annotations

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
    "bundesregierung.de",
    "bundesgesundheitsministerium.de",
    "bmg.bund.de",
    "bundestag.de",
    "gesetze-im-internet.de",
    "gkv-spitzenverband.de",
    "ec.europa.eu",
    "europa.eu",
    "who.int",
    "oecd.org",
}

MAINSTREAM_SOURCE_DOMAINS: set[str] = {
    # Deutsche Medien
    "aerzteblatt.de",
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
    # Wirtschaftsmedien
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
    "kzbv.de",
    "bzaek.de",
    "vdek.com",
    "aok.de",
    "cdu.de",
    "spd.de",
    "gruene.de",
    "fdp.de",
    "linke.de",
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
