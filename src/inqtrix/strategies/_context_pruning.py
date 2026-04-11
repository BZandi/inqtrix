"""Context pruning strategy — prune context blocks to stay within budget."""

from __future__ import annotations

from abc import ABC, abstractmethod

from inqtrix.text import STOPWORDS


class ContextPruningStrategy(ABC):
    """Prune context blocks to stay within budget."""

    @abstractmethod
    def prune(
        self,
        context: list[str],
        question: str,
        sub_questions: list[str],
        max_blocks: int,
        n_new: int,
    ) -> list[str]:
        """Return pruned context list."""


class RelevanceBasedPruning(ContextPruningStrategy):
    """Reproduce ``_prune_context`` -- relevance-based instead of FIFO."""

    def prune(
        self,
        context: list[str],
        question: str,
        sub_questions: list[str],
        max_blocks: int,
        n_new: int,
    ) -> list[str]:
        if len(context) <= max_blocks:
            return context

        # Reference words from question + sub-questions
        ref_text = question.lower()
        for sq in sub_questions:
            ref_text += " " + sq.lower()
        ref_words = {w for w in ref_text.split() if len(w) > 2 and w not in STOPWORDS}

        if not ref_words:
            # Fallback: FIFO
            return context[-max_blocks:]

        # Separate protected (new) and candidate (old) blocks
        n_protected = min(n_new, len(context))
        protected = context[-n_protected:]
        candidates = context[:-n_protected] if n_protected > 0 else context[:]

        # How many old blocks to keep?
        keep_old = max_blocks - len(protected)
        if keep_old <= 0:
            return protected[-max_blocks:]
        if len(candidates) <= keep_old:
            return candidates + protected

        # Score: fraction of reference words present in block
        scored: list[tuple[float, int, str]] = []
        for idx, block in enumerate(candidates):
            block_lower = block.lower()
            block_words = set(block_lower.split())
            overlap = len(ref_words & block_words)
            score = overlap / len(ref_words)
            scored.append((score, idx, block))

        # Sort by score descending, then by index ascending (oldest first out)
        scored.sort(key=lambda x: (-x[0], x[1]))
        kept = [block for _, _, block in scored[:keep_old]]

        return kept + protected
