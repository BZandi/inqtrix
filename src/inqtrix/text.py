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
