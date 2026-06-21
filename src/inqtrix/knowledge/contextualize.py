"""Chunk contextualization (Anthropic-style contextual retrieval).

At ingestion time an LLM situates every chunk within its document; the
generated context is prepended to the chunk before dense embedding AND
BM25 indexing, so de-contextualized chunks ("die Pflichten nach
Absatz 1" — whose obligations? which article?) become retrievable by
the vocabulary of the questions that target them. Published result
(Anthropic 2024, mainstream production practice by 2026): up to -49%
top-20 retrieval failure combined with BM25.

Cost shape: ONE batched fast-tier call per document (not per chunk),
paid once at ingestion, zero query-time latency.

Failure policy (No Silent Fallbacks): an unparseable or
wrongly-shaped response degrades that document to UNcontextualized
chunks with a loud log + marker — ingestion never fails because a
mini model produced bad JSON, but it never lies about it either.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from inqtrix.model_routing import resolve_model
from inqtrix.prompts import build_chunk_context_prompt

log = logging.getLogger("inqtrix")

CONTEXT_MARKER_APPLIED = "_chunk_context_applied"
CONTEXT_MARKER_FALLBACK = "_chunk_context_fallback"

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)

_MAX_DOCUMENT_PROMPT_CHARS = 60_000
"""Documents above this size skip contextualization for the overflow
chunks' benefit of the full text — the prompt carries a truncated
document body instead (clearly better than failing ingestion; the
truncation is logged)."""


@dataclass(frozen=True)
class ContextualizedChunks:
    """Result of contextualizing one document's chunks.

    Attributes:
        texts: One text per input chunk — context-prefixed when the
            call succeeded, the original chunk otherwise.
        marker: ``_chunk_context_applied`` or
            ``_chunk_context_fallback`` (visible degradation).
    """

    texts: list[str]
    marker: str


class ChunkContextualizer(ABC):
    """Port for ingestion-time chunk contextualization."""

    @abstractmethod
    def contextualize(
        self, *, document_title: str, document_text: str, chunks: list[str]
    ) -> ContextualizedChunks:
        """Return retrieval texts for *chunks* (same order, same count)."""


class LLMChunkContextualizer(ChunkContextualizer):
    """Batched per-document contextualization via the run LLM.

    Args:
        llm: LLM provider (``complete_with_metadata``). Constructor-
            First — the composition root wires the deployment's
            provider in.
        model: Model override for the calls; ``None`` resolves the
            ``knowledge_contextualize`` fast-tier assignment against
            the provider's model bundle (mini-model by default).
        timeout: Per-call timeout in seconds.
    """

    def __init__(
        self,
        llm: Any,
        *,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._llm = llm
        self._model = model
        self._timeout = timeout

    def _resolved_model(self) -> str | None:
        if self._model:
            return self._model
        provider_models = getattr(self._llm, "models", None)
        if provider_models is None:
            return None
        return resolve_model(
            "knowledge_contextualize", provider_models, None
        ) or None

    def contextualize(
        self, *, document_title: str, document_text: str, chunks: list[str]
    ) -> ContextualizedChunks:
        """Prefix each chunk with its generated document context."""
        if not chunks:
            return ContextualizedChunks(texts=[], marker=CONTEXT_MARKER_APPLIED)
        body = document_text
        if len(body) > _MAX_DOCUMENT_PROMPT_CHARS:
            log.warning(
                "Kontextualisierung: Dokument %r auf %d Zeichen gekuerzt.",
                document_title,
                _MAX_DOCUMENT_PROMPT_CHARS,
            )
            body = body[:_MAX_DOCUMENT_PROMPT_CHARS]
        prompt = build_chunk_context_prompt(document_title, body, chunks)
        try:
            response = self._llm.complete_with_metadata(
                prompt, model=self._resolved_model(), timeout=self._timeout
            )
            contexts = self._parse(
                getattr(response, "content", "") or "", expected=len(chunks)
            )
        except Exception as exc:  # noqa: BLE001 — degrade loudly, not fatally
            log.warning(
                "Kontextualisierung fehlgeschlagen fuer %r (%s) — Chunks "
                "bleiben unkontextualisiert (%s).",
                document_title,
                exc,
                CONTEXT_MARKER_FALLBACK,
            )
            return ContextualizedChunks(
                texts=list(chunks), marker=CONTEXT_MARKER_FALLBACK
            )
        if contexts is None:
            log.warning(
                "Kontextualisierung unparsebar fuer %r — Chunks bleiben "
                "unkontextualisiert (%s).",
                document_title,
                CONTEXT_MARKER_FALLBACK,
            )
            return ContextualizedChunks(
                texts=list(chunks), marker=CONTEXT_MARKER_FALLBACK
            )
        texts = [
            f"{context.strip()}\n\n{chunk}" if context.strip() else chunk
            for context, chunk in zip(contexts, chunks)
        ]
        return ContextualizedChunks(
            texts=texts, marker=CONTEXT_MARKER_APPLIED
        )

    @staticmethod
    def _parse(content: str, *, expected: int) -> list[str] | None:
        match = _JSON_ARRAY.search(content)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except ValueError:
            return None
        if not isinstance(payload, list) or len(payload) != expected:
            return None
        if not all(isinstance(item, str) for item in payload):
            return None
        return payload
