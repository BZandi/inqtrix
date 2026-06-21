"""Paragraph-aware text chunking for the knowledge engine.

Deliberately simple for the first cut: split on blank-line paragraph
boundaries, pack paragraphs greedily up to a character budget, and
hard-split single paragraphs that exceed the budget on sentence-ish
boundaries. Layout-aware and contextual chunking arrive with the real
ingestion pipeline; the chunker stays a pure function so swapping it
never touches the stores or the algorithm.
"""

from __future__ import annotations

import re

DEFAULT_MAX_CHUNK_CHARS = 2_000
"""Default chunk budget (~500 tokens at 4 chars/token).

Comfortably below every catalogued embedding model's input limit while
keeping chunks topically coherent. Operators tune per deployment via
``KnowledgeSettings.chunk_max_chars``.
"""

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, *, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Split *text* into retrieval chunks of at most *max_chars*.

    Args:
        text: The full document text. Surrounding whitespace is
            ignored; empty input yields no chunks.
        max_chars: Per-chunk character budget. Must be positive.

    Returns:
        Non-empty chunk strings in document order. Paragraphs are kept
        together where the budget allows; oversize paragraphs are
        split on sentence boundaries, and a single oversize sentence
        is hard-wrapped as a last resort (never dropped).

    Raises:
        ValueError: When *max_chars* is not positive — a silent
            zero-budget chunker would drop content.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    def _flush() -> None:
        nonlocal current, current_length
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_length = 0

    for paragraph in (p.strip() for p in _PARAGRAPH_SPLIT.split(normalized)):
        if not paragraph:
            continue
        pieces = (
            [paragraph]
            if len(paragraph) <= max_chars
            else _split_oversize_paragraph(paragraph, max_chars)
        )
        for piece in pieces:
            extra = len(piece) + (2 if current else 0)
            if current and current_length + extra > max_chars:
                _flush()
            current.append(piece)
            current_length += len(piece) + (2 if current_length else 0)
    _flush()
    return chunks


def _split_oversize_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split one over-budget paragraph on sentence boundaries."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_BOUNDARY.split(paragraph):
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(sentence) <= max_chars:
            current = sentence
        else:
            pieces.extend(
                sentence[offset:offset + max_chars]
                for offset in range(0, len(sentence), max_chars)
            )
            current = ""
    if current:
        pieces.append(current)
    return pieces
