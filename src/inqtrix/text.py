"""Text processing utilities."""

from __future__ import annotations

import re
from typing import Iterator

STOPWORDS: set[str] = {
    "der", "die", "das", "ein", "eine", "und", "oder", "ist", "sind", "hat",
    "haben", "wird", "wurde", "werden", "von", "mit", "fuer", "auf", "aus",
    "bei", "nach", "ueber", "vor", "zu", "als", "wie", "auch", "nicht",
    "sich", "des", "dem", "den", "es", "im", "in", "an", "um", "noch",
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "has",
    "have", "had", "be", "been", "for", "from", "with", "on", "at", "by",
    "to", "of", "in", "it", "its", "this", "that", "which", "what", "how",
    "not", "but", "also", "about", "as", "can", "do", "does", "did",
}

NEGATION_TOKENS: set[str] = {
    "kein", "keine", "keinen", "keinem", "keiner", "nicht", "ohne",
    "no", "not", "never", "none", "without",
}

_UMLAUT_EXPAND_MAP = str.maketrans({
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
})


def tokenize(text: str) -> list[str]:
    """Simple tokenization for heuristics."""
    return re.findall(r"[a-zA-Z0-9äöüÄÖÜß]+", (text or "").lower())


def norm_match_token(token: str) -> str:
    """Normalize token for matching (lowercase + umlaut expansion)."""
    return (token or "").lower().translate(_UMLAUT_EXPAND_MAP)


def is_none_value(text: str) -> bool:
    """Check if an LLM output field is 'Keine' / 'None' or a variant."""
    t = text.strip().lower()
    if t in ("keine", "none", "keine.", "none.", ""):
        return True
    for prefix in ("keine ", "keine\u2014", "keine\u00a0", "none ", "none\u2014"):
        if t.startswith(prefix):
            return True
    return False


def iter_word_chunks(text: str) -> Iterator[str]:
    """Yield text in word-sized chunks while preserving original spacing."""
    for match in re.finditer(r"\S+\s*|\s+", text or ""):
        yield match.group(0)


# ---------------------------------------------------------------------------
# Aspect synonym expansion (shared)
#
# Both the risk_scoring strategy (substring scan over context blocks) and the
# context_pruning strategy (token-set intersection) need to expand a required
# aspect into a list of related German keywords. Without a single source of
# truth the two stages drift, which causes coverage to be reported as full in
# one place and incomplete in the other for the same evidence.
# ---------------------------------------------------------------------------

_ASPECT_SYNONYM_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("status quo",), ("status", "stand", "aktuell", "derzeit")),
    (("datum",), ("datum", "stand", "heute")),
    (("position akteur",),
     ("position", "regierung", "partei", "verband", "akteur")),
    (("richtung", "diskussion"),
     ("richtung", "trend", "debatte", "diskussion", "entwicklung")),
    (("mehrheitslage", "umsetzbarkeit"),
     ("mehrheit", "mehrheitsfaehig", "durchsetzbar", "umsetzbar")),
    (("hintergrund", "ausgangslage", "kontext"),
     ("hintergrund", "kontext", "ausgangslage", "historisch", "bisher")),
    (("treiber", "mechan", "argument"),
     ("grund", "ursache", "treiber", "argument", "mechanismus")),
    (("stakeholder", "betroffenen"),
     ("stakeholder", "akteur", "verband", "regierung",
      "unternehmen", "patient", "verbraucher", "krankenkasse")),
    (("gegenargument", "risiken", "limitation"),
     ("kritik", "risiko", "problem", "nachteil", "limitation", "unsicherheit")),
    (("alternativen", "vergleich", "gegenmodell"),
     ("alternative", "vergleich", "versus", "option", "gegenmodell")),
    (("zahlenbasis", "primaerbeleg", "primärbeleg"),
     ("prozent", "euro", "mrd", "mio", "zahl", "statistik", "primaerquelle")),
)


def aspect_synonyms(aspect: str) -> list[str]:
    """Return curated synonym tokens for a research-aspect description.

    Pure expansion only — callers are responsible for tokenising the aspect
    text itself and for any normalization (lowercase / umlaut expansion).
    Returned tokens preserve insertion order and are deduplicated.
    """
    aspect_l = (aspect or "").lower()
    out: list[str] = []
    seen: set[str] = set()
    for keys, synonyms in _ASPECT_SYNONYM_RULES:
        if not any(self_check(key, aspect_l) for key in keys):
            continue
        for token in synonyms:
            if token not in seen:
                seen.add(token)
                out.append(token)
    return out


def self_check(key: str, aspect_l: str) -> bool:
    """Helper used by ``aspect_synonyms``: True when all key words are in aspect.

    Multi-word keys (e.g. ``"position akteur"``) match when *all* whitespace
    parts appear in the aspect string. Single-word keys are simple substring
    checks. Kept as a module-level helper so it can be reused by callers that
    want to test for a specific synonym group without touching the rules table.
    """
    parts = key.split()
    if len(parts) <= 1:
        return key in aspect_l
    return all(part in aspect_l for part in parts)
