"""Shared retrieval pipeline: embed, (hybrid) search, rerank.

Defined exactly once (Designprinzip 4) and consumed by BOTH the
``/v1/knowledge/search`` debug surface and the ``mode=knowledge``
answer path — before this module existed, the rerank and hybrid
stages applied only to the debug endpoint while answers ran on raw
dense search.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from inqtrix.knowledge.stores.ports import (
    KnowledgeProviderContext,
    RetrievalCandidate,
)
from inqtrix.providers.base import observe_provider_retries


async def retrieve(
    knowledge: KnowledgeProviderContext,
    *,
    query: str,
    collection_ids: list[str] | None,
    top_k: int,
    use_reranker: bool = True,
    rerank_candidate_depth: int | None = None,
    on_provider_retry: Callable[[dict[str, Any]], None] | None = None,
) -> list[RetrievalCandidate]:
    """Run the full retrieval pipeline for one query.

    Stages: query embedding (with the scope's collection model) →
    hybrid search when the store is hybrid-capable, dense otherwise →
    optional cross-encoder rerank reducing a deeper candidate pool to
    *top_k*. A configured-but-broken reranker raises loudly; only the
    unconfigured state skips the stage.

    Args:
        use_reranker: Profile hook — ``False`` skips the rerank stage
            even when a reranker is wired (the fast profile). The
            default keeps every pre-profile caller byte-stable.
        rerank_candidate_depth: Profile hook — candidate pool fetched
            ahead of the rerank stage; ``None`` uses the configured
            default depth.
        on_provider_retry: Optional observer for rerank-provider retry
            notices (the shared ``_RetryNoticeMixin`` dict shape).
            Retries would otherwise only reach the server log; callers
            with an event surface forward them there (no silent
            fallbacks). ``None`` keeps the historical behaviour.
    """
    embedding_model = knowledge.embeddings.default_model
    if collection_ids:
        scoped_collection = await knowledge.store.get_collection(
            collection_ids[0]
        )
        embedding_model = scoped_collection.embedding_model
    query_embedding = await asyncio.to_thread(
        knowledge.embeddings.embed_query, query, model=embedding_model
    )

    reranker = knowledge.reranker if use_reranker else None
    candidate_depth = (
        rerank_candidate_depth
        if rerank_candidate_depth is not None
        else knowledge.rerank_candidate_depth
    )
    retrieve_top_k = (
        max(top_k, candidate_depth) if reranker is not None else top_k
    )

    store = knowledge.store
    if getattr(store, "supports_hybrid", False):
        candidates = await store.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            collection_ids=collection_ids,
            top_k=retrieve_top_k,
            embedding_model=embedding_model,
        )
    else:
        candidates = await store.search(
            query_embedding=query_embedding,
            collection_ids=collection_ids,
            top_k=retrieve_top_k,
            embedding_model=embedding_model,
        )

    if reranker is None or len(candidates) <= 1:
        return candidates[:top_k]
    candidate_texts = [candidate.chunk.text for candidate in candidates]

    def _rerank_observed() -> list[Any]:
        # Observer binding, the call, and the thread-local cleanup must all
        # run on the SAME thread as the rerank (the mixin state is
        # threading.local), hence one closure handed to to_thread.
        with observe_provider_retries(reranker, on_provider_retry):
            try:
                return reranker.rerank(query, candidate_texts, top_n=top_k)
            finally:
                consume = getattr(reranker, "consume_retry_notices", None)
                if callable(consume):
                    # Clear leftover notices so a reused executor thread
                    # cannot bleed them into a later request.
                    consume()

    results = await asyncio.to_thread(_rerank_observed)
    return [
        RetrievalCandidate(
            chunk=candidates[result.index].chunk,
            score=result.relevance_score,
            document_title=candidates[result.index].document_title,
        )
        for result in results
    ]


def interleave_candidates(
    result_lists: list[list[RetrievalCandidate]],
    *,
    limit: int,
) -> list[RetrievalCandidate]:
    """Round-robin merge of per-sub-query result lists, capped at *limit*.

    The decomposition merge: taking one candidate per list in rotation
    guarantees EVERY aspect contributes to the top-k instead of the
    first list crowding the others out (the aggregation failure class
    a plain first-wins union reproduces). The original question's list
    goes first in the rotation; duplicates collapse on chunk id.
    """
    merged: list[RetrievalCandidate] = []
    seen: set[str] = set()
    cursors = [0] * len(result_lists)
    while len(merged) < limit:
        progressed = False
        for list_index, candidates in enumerate(result_lists):
            cursor = cursors[list_index]
            while cursor < len(candidates):
                candidate = candidates[cursor]
                cursor += 1
                if candidate.chunk.id not in seen:
                    seen.add(candidate.chunk.id)
                    merged.append(candidate)
                    progressed = True
                    break
            cursors[list_index] = cursor
            if len(merged) >= limit:
                break
        if not progressed:
            break
    return merged


def merge_candidates(
    primary: list[RetrievalCandidate],
    secondary: list[RetrievalCandidate],
    *,
    limit: int,
) -> list[RetrievalCandidate]:
    """Union two candidate lists, first occurrence wins, capped at *limit*.

    Used by the second retrieval pass: the original ranking stays
    authoritative, rewritten-query hits fill the remaining slots.
    """
    merged: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for candidate in [*primary, *secondary]:
        if candidate.chunk.id in seen:
            continue
        seen.add(candidate.chunk.id)
        merged.append(candidate)
        if len(merged) >= limit:
            break
    return merged
