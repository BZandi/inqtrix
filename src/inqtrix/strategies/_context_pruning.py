"""Context pruning strategy — prune context blocks to stay within budget."""

from __future__ import annotations

from abc import ABC, abstractmethod

from inqtrix.text import STOPWORDS, norm_match_token, tokenize


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
        """Return a bounded context list for downstream prompts.

        Args:
            context: Current ordered context blocks.
            question: Primary user question.
            sub_questions: Derived sub-questions from classify.
            max_blocks: Maximum number of blocks to retain.
            n_new: Number of newest blocks that should be protected from
                pruning whenever possible.
        """


class RelevanceBasedPruning(ContextPruningStrategy):
    """Relevance-based pruning that still protects the newest evidence.

    Older blocks compete on token overlap with the main question and
    sub-questions, while the newest ``n_new`` blocks are kept as a freshness
    guardrail. Ties favor the more recent old block so repeated rounds do not
    drift back toward stale evidence.
    """

    def prune(
        self,
        context: list[str],
        question: str,
        sub_questions: list[str],
        max_blocks: int,
        n_new: int,
    ) -> list[str]:
        """Prune context to the configured block budget.

        Token matching uses the shared tokenizer and normalized match tokens so
        punctuation and umlaut variants do not accidentally zero out overlap
        scores.
        """
        max_blocks = int(max_blocks or 0)
        if max_blocks <= 0:
            return []
        if len(context) <= max_blocks:
            return context

        # Reference words from question + sub-questions
        ref_text = question.lower()
        for sq in sub_questions:
            ref_text += " " + sq.lower()
        ref_words: set[str] = set()
        for token in tokenize(ref_text):
            normalized = norm_match_token(token)
            if len(normalized) <= 2 or normalized in STOPWORDS:
                continue
            ref_words.add(normalized)

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
            block_words = {norm_match_token(token) for token in tokenize(block)}
            overlap = len(ref_words & block_words)
            score = overlap / len(ref_words)
            scored.append((score, idx, block))

        # Prefer higher overlap and, on ties, more recent non-protected blocks.
        scored.sort(key=lambda x: (-x[0], -x[1]))
        kept = [block for _, _, block in sorted(scored[:keep_old], key=lambda x: x[1])]

        return kept + protected
