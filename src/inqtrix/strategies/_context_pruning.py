"""Context pruning strategy — prune context blocks to stay within budget."""

from __future__ import annotations

from abc import ABC, abstractmethod

from inqtrix.text import STOPWORDS, aspect_synonyms, norm_match_token, tokenize


class ContextPruningStrategy(ABC):
    """Contract for trimming the rolling context list between rounds.

    Inqtrix accumulates one context block per processed search hit
    across rounds. Without pruning the prompt would grow unboundedly;
    the pruning strategy enforces a per-round cap (``max_blocks``)
    while protecting the newest evidence so a single bad round
    cannot evict freshly-gathered facts.

    Implementations must be pure with respect to ``context`` (no
    in-place mutation) and must always return at most ``max_blocks``
    items. Returning fewer is allowed when ``max_blocks`` exceeds
    the available context.
    """

    @abstractmethod
    def prune(
        self,
        context: list[str],
        question: str,
        sub_questions: list[str],
        max_blocks: int,
        n_new: int,
        required_aspects: list[str] | None = None,
    ) -> list[str]:
        """Return a bounded context list for downstream prompts.

        Args:
            context: Current ordered context blocks (oldest first).
                Not mutated by the strategy.
            question: Primary user question, used as the main relevance
                anchor for older blocks.
            sub_questions: Derived sub-questions from classify, used
                as additional relevance anchors so that
                decomposition-aligned evidence survives pruning.
            max_blocks: Maximum number of blocks to retain. ``0`` (or
                negative) returns an empty list. Implementations may
                shrink the result below this cap when the input is
                shorter.
            n_new: Number of newest blocks the strategy must retain
                whenever possible (freshness guardrail). Bounded above
                by ``len(context)``; when ``n_new >= max_blocks`` only
                the newest ``max_blocks`` are kept.
            required_aspects: Optional list of aspect labels derived
                by the risk-scoring strategy. When provided, the
                strategy may pre-select one block per aspect to ensure
                aspect coverage in the kept set.

        Returns:
            New list of at most ``max_blocks`` context blocks in
            chronological order (kept-old blocks first, then the
            protected newest blocks). Caller may mutate freely.
        """


class RelevanceBasedPruning(ContextPruningStrategy):
    """Relevance-based pruning that still protects the newest evidence.

    Heuristic in three steps:

    1. Reserve the newest ``n_new`` blocks unconditionally
       (freshness guardrail).
    2. For the remaining old blocks, compute an overlap score
       against the union of question + sub-questions + required-
       aspects tokens (TF-style normalised match-token overlap).
    3. Optionally pre-select up to one old block per
       ``required_aspect`` whose overlap with the aspect's token set
       is highest, so aspect coverage survives even when the
       general overlap score is low.

    Ties favour the more recent old block so repeated rounds do
    not drift back toward stale evidence.
    """

    def prune(
        self,
        context: list[str],
        question: str,
        sub_questions: list[str],
        max_blocks: int,
        n_new: int,
        required_aspects: list[str] | None = None,
    ) -> list[str]:
        """Prune context to the configured block budget.

        Token matching uses the shared tokeniser and normalised match
        tokens so punctuation and umlaut variants do not accidentally
        zero out overlap scores. When the question has no usable
        reference tokens at all (e.g. all stopwords / too short),
        the strategy degrades to FIFO truncation of the newest
        ``max_blocks`` blocks.

        Args:
            context: See :meth:`ContextPruningStrategy.prune`.
            question: See :meth:`ContextPruningStrategy.prune`.
            sub_questions: See :meth:`ContextPruningStrategy.prune`.
            max_blocks: See :meth:`ContextPruningStrategy.prune`.
            n_new: See :meth:`ContextPruningStrategy.prune`.
            required_aspects: See :meth:`ContextPruningStrategy.prune`.

        Returns:
            See :meth:`ContextPruningStrategy.prune`. ``[]`` when
            ``max_blocks <= 0``.
        """
        max_blocks = int(max_blocks or 0)
        if max_blocks <= 0:
            return []
        if len(context) <= max_blocks:
            return context

        # Reference words from question + sub-questions + optional required aspects
        ref_text = question.lower()
        for sq in sub_questions:
            ref_text += " " + sq.lower()
        for aspect in required_aspects or []:
            ref_text += " " + aspect.lower()
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

        def _aspect_tokens(aspect: str) -> set[str]:
            """Return the normalised token set for one required aspect.

            Includes both the aspect's own tokens and its synonyms
            (via :func:`~inqtrix.text.aspect_synonyms`) so that an
            aspect like ``"finanzierung"`` also matches blocks
            mentioning ``"kosten"`` or ``"haushalt"``.
            """
            tokens: set[str] = set()
            for token in tokenize(aspect):
                normalized = norm_match_token(token)
                if len(normalized) <= 2 or normalized in STOPWORDS:
                    continue
                tokens.add(normalized)
            for synonym in aspect_synonyms(aspect):
                normalized = norm_match_token(synonym)
                if len(normalized) > 2 and normalized not in STOPWORDS:
                    tokens.add(normalized)
            return tokens

        # Score: fraction of reference words present in block
        candidate_words: list[set[str]] = []
        scored: list[tuple[float, int, str]] = []
        for idx, block in enumerate(candidates):
            block_words = {norm_match_token(token) for token in tokenize(block)}
            candidate_words.append(block_words)
            overlap = len(ref_words & block_words)
            score = overlap / len(ref_words)
            scored.append((score, idx, block))

        preselected_indices: set[int] = set()
        if required_aspects:
            for aspect in required_aspects:
                if len(preselected_indices) >= keep_old:
                    break
                aspect_words = _aspect_tokens(aspect)
                if not aspect_words:
                    continue
                best_idx = -1
                best_score = 0.0
                for idx, block_words in enumerate(candidate_words):
                    if idx in preselected_indices:
                        continue
                    overlap = len(aspect_words & block_words)
                    if overlap <= 0:
                        continue
                    aspect_score = overlap / len(aspect_words)
                    if aspect_score > best_score or (aspect_score == best_score and idx > best_idx):
                        best_idx = idx
                        best_score = aspect_score
                if best_idx >= 0:
                    preselected_indices.add(best_idx)

        # Prefer higher overlap and, on ties, more recent non-protected blocks.
        scored.sort(key=lambda x: (-x[0], -x[1]))
        kept_indices = list(preselected_indices)
        for _, idx, _ in scored:
            if len(kept_indices) >= keep_old:
                break
            if idx in preselected_indices:
                continue
            kept_indices.append(idx)
        kept = [candidates[idx] for idx in sorted(kept_indices)]

        return kept + protected
