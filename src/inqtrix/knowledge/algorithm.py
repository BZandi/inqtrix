"""The knowledge algorithm: retrieve from internal documents, answer cited.

First-cut pipeline, fully synchronous:

1. Embed the question with the scoped collections' embedding model.
2. Exact cosine retrieval over the knowledge store (top-k).
3. Render the hits as a ``[K#]``-labelled evidence block (budgeted to
   the LLM's context window).
4. One LLM call through the ``knowledge_answer`` routing node.

Reranking, sufficiency-driven second passes, and hybrid web+knowledge
execution are staged upgrades — each lands as an additional pipeline
stage without changing the algorithm contract or the HTTP surface.

The raw result dict mirrors the web-research shape (``answer``,
``usage``, ``result_state``) so the existing run serialization
(``ResearchResult.from_raw``, snapshots, SSE events) consumes knowledge
runs without a parallel code path.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from inqtrix.core.results import AgentResult, RunRequest
from inqtrix.i18n import detect_ui_language_confident
from inqtrix.knowledge.contextualize import contextualize_followup_question
from inqtrix.knowledge.gate import GateDecision, evaluate_evidence
from inqtrix.knowledge.grounding import (
    GROUNDING_MARKER_FALLBACK,
    check_grounding,
)
from inqtrix.knowledge.decompose import decompose_question
from inqtrix.knowledge.profiles import (
    EVIDENCE_K_MAX,
    KnowledgeStageCeiling,
    parse_knowledge_profile,
    resolve_run_plan,
)
from inqtrix.knowledge.retrieval import (
    interleave_candidates,
    merge_candidates,
    retrieve,
)
from inqtrix.knowledge.stores.ports import (
    KnowledgeProviderContext,
    RetrievalCandidate,
)
from inqtrix.model_routing import resolve_effort, resolve_model
from inqtrix.prompts import build_knowledge_answer_prompt
from inqtrix.sync_bridge import run_coro_sync

if TYPE_CHECKING:
    from inqtrix.core.context import RunContext, RuntimeContext

log = logging.getLogger("inqtrix")

_EVIDENCE_BUDGET_FLOOR_CHARS = 8_000
_PROMPT_RESERVED_TOKENS = 4_000

# Visible-not-silent marker (Designprinzip 1): the BM25 tokenizer language does
# not match the confidently-detected query language, so the lexical branch's
# language-specific normalization (tokenizing/stemming/stopwords) does not apply
# cleanly and keyword retrieval is less reliable. NOT "contributes nothing" —
# query and documents pass through the same encoder, so exact terms can still
# match. This signal is query-vs-TOKENIZER only; query-vs-DOCUMENT visibility
# needs per-collection language (a later phase).
SPARSE_MARKER_TOKENIZER_MISMATCH = "_knowledge_sparse_tokenizer_mismatch"


def _sparse_tokenizer_state(
    knowledge: KnowledgeProviderContext,
    question: str,
    emit: "Any",
) -> dict[str, Any] | None:
    """Flag a confident query-vs-BM25-tokenizer language mismatch, or ``None``.

    Returns a small visibility record (marker + the two language codes) only
    when the lexical branch is active (``sparse_language`` is set) AND the query
    language is *confidently* detected AND it differs from the tokenizer
    language. Otherwise ``None`` — so the same-language default path adds no new
    field, log, or event (byte-/field-identical). The log carries only the two
    language codes plus the marker, never the query text (sensitivity discipline,
    cf. :func:`inqtrix.urls.sanitize_log_message`).
    """
    sparse_language = getattr(knowledge.store, "sparse_language", None)
    if sparse_language is None:
        return None
    query_language = detect_ui_language_confident(question)
    if query_language is None or query_language == sparse_language:
        return None
    log.warning(
        "Knowledge-Sparse: Query-Sprache (%s) passt nicht zur BM25-Tokenizer-"
        "Sprache (%s) — lexikalische Normalisierung greift nicht sauber, "
        "Sparse-Qualitaet weniger verlaesslich (exakte Begriffe koennen weiter "
        "matchen). Optional einen mehrsprachigen Cross-Encoder-Reranker "
        "aktivieren (z. B. Cohere rerank-v3.5). (%s)",
        query_language,
        sparse_language,
        SPARSE_MARKER_TOKENIZER_MISMATCH,
    )
    emit(
        "inqtrix.knowledge.sparse.tokenizer_mismatch",
        {
            "query_language": query_language,
            "sparse_language": sparse_language,
            "marker": SPARSE_MARKER_TOKENIZER_MISMATCH,
        },
    )
    return {
        "marker": SPARSE_MARKER_TOKENIZER_MISMATCH,
        "query_language": query_language,
        "sparse_language": sparse_language,
    }


def _resolve_collection_ids(request: RunRequest) -> list[str] | None:
    """Read the optional collection scope from the request filters."""
    raw = request.knowledge_filters.get("collection_ids")
    if raw is None:
        return None
    return [str(item) for item in raw]


def _scoped_document_count(
    knowledge: KnowledgeProviderContext, collection_ids: list[str] | None
) -> int | None:
    """Total documents in the queried scope — ALL eligible for retrieval (the
    filter is collection-level, so every document is searched). Surfaces the
    "N documents searched" coverage signal. Returns ``None`` if the count can't
    be resolved (logged, never silent) so the UI omits it rather than guess."""

    async def _count() -> int:
        if collection_ids:
            total = 0
            for collection_id in collection_ids:
                total += (
                    await knowledge.store.get_collection(collection_id)
                ).document_count
            return total
        collections = await knowledge.store.list_collections()
        return sum(collection.document_count for collection in collections)

    try:
        return run_coro_sync(_count())
    except Exception as exc:  # noqa: BLE001 - a count must never fail the answer
        log.warning(
            "Knowledge: Dokumentanzahl der Sammlung nicht ermittelbar (%s).", exc
        )
        return None


def _render_evidence_entry(index: int, candidate: RetrievalCandidate) -> str:
    """One ``[K#]``-labelled evidence entry (title line + chunk text).

    The prompt rendering; quote VERIFICATION deliberately runs against
    the chunks' source text instead — a quote of the synthetic header
    or contextualization prefix must not verify as source content.
    """
    return (
        f"[K{index}] {candidate.document_title} "
        f"(Abschnitt {candidate.chunk.chunk_index + 1})\n"
        f"{candidate.chunk.text}"
    )


def _render_evidence_block(
    candidates: list[RetrievalCandidate],
    *,
    max_chars: int,
) -> tuple[str, list[RetrievalCandidate]]:
    """Render ``[K#]`` evidence entries up to the character budget.

    Returns the rendered block and the candidates that actually fit —
    the reference list and the answer prompt must describe the same
    set, so truncation happens once, here, and visibly via the
    returned subset.
    """
    entries: list[str] = []
    used: list[RetrievalCandidate] = []
    total = 0
    for index, candidate in enumerate(candidates, start=1):
        entry = _render_evidence_entry(index, candidate)
        if used and total + len(entry) > max_chars:
            break
        entries.append(entry)
        used.append(candidate)
        total += len(entry)
    return "\n\n".join(entries), used


class KnowledgeAlgorithm:
    """Retrieve-and-answer over the deployment's knowledge collections.

    Args:
        knowledge: The wired knowledge capabilities (embeddings, store,
            retrieval defaults). Constructor-First: the composition
            root builds and injects this bundle; the algorithm never
            reads settings or the environment.
        citation_base_url: Public base URL of the serving deployment.
            When non-empty, references become clickable HTTP links
            into ``/v1/sources/...``; empty keeps the internal
            ``inqtrix://`` URI scheme (deliberate, visible default —
            never a guessed hostname).
        gate_enabled: Operator ceiling for the sufficiency gate;
            ``False`` removes the gate from EVERY retrieval profile
            (the always-answer-from-top-k path).
        grounding_enabled: Operator ceiling for quote-then-answer:
            the answer prompt requires a verbatim-quote block which is
            verified deterministically against the evidence and
            stripped from the user-facing answer; ``False`` keeps the
            plain single-section answer prompt in every profile.
        gate_max_rounds: Hard cap on gate rewrite-and-retrieve rounds
            for every profile; the deep profile requests up to this
            many, standard exactly one. Bounds the worst-case cost of
            an agentic run.

    The constructor flags form the operator CEILING; the per-request
    retrieval profile selects within it (resolved to a frozen
    :class:`~inqtrix.knowledge.profiles.KnowledgeRunPlan` inside
    ``run()``). The instance is a shared singleton across server
    threads and workers — per-request state never lives on ``self``.
    """

    id = "knowledge"
    display_name = "Knowledge Retrieval"

    def __init__(
        self,
        *,
        knowledge: KnowledgeProviderContext,
        citation_base_url: str = "",
        gate_enabled: bool = True,
        grounding_enabled: bool = True,
        gate_max_rounds: int = 3,
    ) -> None:
        self._knowledge = knowledge
        self._citation_base_url = citation_base_url.rstrip("/")
        self._ceiling = KnowledgeStageCeiling(
            gate_available=gate_enabled,
            grounding_available=grounding_enabled,
            reranker_available=knowledge.reranker is not None,
            gate_max_rounds=gate_max_rounds,
            rerank_candidate_depth=knowledge.rerank_candidate_depth,
        )

    def _reference_url(self, candidate: RetrievalCandidate) -> str:
        """Citation target for one evidence chunk.

        HTTP links into the sources view when the deployment knows its
        public base URL; the chunk index travels as a query parameter
        because the export pipeline strips URL fragments.
        """
        document_id = candidate.chunk.document_id
        chunk_index = candidate.chunk.chunk_index
        if self._citation_base_url:
            return (
                f"{self._citation_base_url}/v1/sources/{document_id}"
                f"?chunk={chunk_index}"
            )
        return f"inqtrix://documents/{document_id}#chunk-{chunk_index}"

    def capabilities(self) -> dict:
        """Manifest entry for the capability endpoint and clients.

        Deliberately WITHOUT ``streams_via_research_graph``: the
        streamed chat path executes the research graph directly, which
        would be the wrong engine — the chat router rejects
        ``stream=true`` for this algorithm loudly until streaming
        dispatches through the registry.
        """
        return {
            "requires": ["llm", "embeddings", "knowledge_store"],
            "streams_events": True,
            "supports_chat_completions": True,
            "terminal_node": "knowledge",
            "produces": ["answer", "references"],
            "supports": ["collection_filters"],
        }

    def _run_gate(
        self,
        llm: Any,
        context: "RunContext",
        question: str,
        evidence_block: str,
        emit,
        usage_accumulator: dict[str, int],
        *,
        round_index: int = 0,
        vocabulary_bridge: bool = False,
    ) -> GateDecision:
        """One sufficiency evaluation on the fast tier (mini model)."""
        provider_models = getattr(llm, "models", None)
        requested_tier = (
            context.agent_settings.model_tier or ""
        ).strip() or None
        gate_model = (
            resolve_model("knowledge_gate", provider_models, requested_tier)
            or None
            if provider_models is not None
            else None
        )
        decision, gate_usage = evaluate_evidence(
            llm,
            question=question,
            evidence_block=evidence_block,
            model=gate_model,
            timeout=context.agent_settings.reasoning_timeout,
            vocabulary_bridge=vocabulary_bridge,
        )
        usage_accumulator["prompt_tokens"] += gate_usage["prompt_tokens"]
        usage_accumulator["completion_tokens"] += gate_usage[
            "completion_tokens"
        ]
        emit(
            "inqtrix.knowledge.gate.evaluated",
            {
                "sufficient": decision.sufficient,
                "coverage": decision.coverage,
                "marker": decision.marker,
                "rewritten": decision.rewritten_query is not None,
                "round": round_index,
            },
        )
        return decision

    def _run_decomposition(
        self,
        llm: Any,
        context: "RunContext",
        question: str,
        emit,
        usage_accumulator: dict[str, int],
    ) -> tuple[str, ...]:
        """One decomposition call on the fast tier (deep profile only)."""
        provider_models = getattr(llm, "models", None)
        requested_tier = (
            context.agent_settings.model_tier or ""
        ).strip() or None
        decompose_model = (
            resolve_model(
                "knowledge_decompose", provider_models, requested_tier
            )
            or None
            if provider_models is not None
            else None
        )
        decomposition, usage = decompose_question(
            llm,
            question=question,
            model=decompose_model,
            timeout=context.agent_settings.reasoning_timeout,
        )
        usage_accumulator["prompt_tokens"] += usage["prompt_tokens"]
        usage_accumulator["completion_tokens"] += usage["completion_tokens"]
        emit(
            "inqtrix.knowledge.decomposition.completed",
            {
                "sub_query_count": len(decomposition.sub_queries),
                "marker": decomposition.marker,
            },
        )
        return decomposition.sub_queries

    def _run_query_contextualization(
        self,
        llm: Any,
        context: "RunContext",
        request: RunRequest,
        emit,
        usage_accumulator: dict[str, int],
    ) -> tuple[str, dict[str, Any]]:
        """Resolve the retrieval query for a conversational Knowledge turn.

        Args:
            llm: Provider used for the optional standalone-query rewrite.
            context: Per-run container with model routing and timeout settings.
            request: Incoming Knowledge run request. ``history`` controls
                whether contextualization is attempted.
            emit: Event sink used to expose the contextualization decision.
            usage_accumulator: Mutable prompt/completion-token totals for this
                pre-retrieval step.

        Returns:
            A tuple of the query text used by retrieval and a compact state
            dictionary for the final ``result_state``. No-history turns return
            the original question and are omitted from ``result_state`` to keep
            the historical no-history payload stable.
        """
        if not request.history.strip():
            return request.question, {
                "used_history": False,
                "rewritten": False,
                "marker": None,
            }
        provider_models = getattr(llm, "models", None)
        requested_tier = (
            context.agent_settings.model_tier or ""
        ).strip() or None
        contextualize_model = (
            resolve_model(
                "knowledge_contextualize", provider_models, requested_tier
            )
            or None
            if provider_models is not None
            else None
        )
        result, usage = contextualize_followup_question(
            llm,
            question=request.question,
            history=request.history,
            model=contextualize_model,
            timeout=context.agent_settings.reasoning_timeout,
        )
        usage_accumulator["prompt_tokens"] += usage["prompt_tokens"]
        usage_accumulator["completion_tokens"] += usage["completion_tokens"]
        emit(
            "inqtrix.knowledge.contextualized",
            {
                "used_history": True,
                "rewritten": result.rewritten,
                "marker": result.marker,
            },
        )
        return result.question, {
            "used_history": True,
            "rewritten": result.rewritten,
            "marker": result.marker,
        }

    def run(
        self,
        request: RunRequest,
        *,
        runtime: "RuntimeContext",
        context: "RunContext",
    ) -> AgentResult:
        """Execute retrieval + cited answer synthesis for one question."""
        knowledge = self._knowledge
        emit = context.event_sink or (lambda _event, _payload: None)
        started = time.monotonic()
        llm = context.providers.llm
        contextualize_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        query_question, contextualization_state = self._run_query_contextualization(
            llm, context, request, emit, contextualize_usage
        )

        raw_profile = request.knowledge_filters.get("profile")
        requested_profile = (
            parse_knowledge_profile(raw_profile)
            if raw_profile is not None
            else None
        )
        # The plan is a frozen LOCAL — this instance is shared across
        # threads/workers; per-request flags on `self` would race.
        plan = resolve_run_plan(
            requested_profile,
            question=query_question,
            ceiling=self._ceiling,
        )
        emit(
            "inqtrix.knowledge.profile.resolved",
            {
                "profile": plan.profile.value,
                "requested_profile": (
                    plan.requested_profile.value
                    if plan.requested_profile is not None
                    else None
                ),
                "auto_selected": plan.auto_selected,
                "auto_reason": plan.auto_reason,
                "rerank": plan.rerank,
                "gate_rounds": plan.gate_rewrite_rounds,
                "grounding": plan.grounding_enabled,
                "vocabulary_bridge": plan.vocabulary_bridge,
                "decompose": plan.decompose,
                "report": plan.report,
                "degraded_stages": list(plan.degraded_stages),
            },
        )

        collection_ids = _resolve_collection_ids(request)
        # Make the BM25 tokenizer-language limitation visible (No Silent
        # Fallbacks). Returns None on the same-language default path, so
        # result_state stays field-identical there.
        sparse_state = _sparse_tokenizer_state(knowledge, query_question, emit)
        top_k = int(
            request.knowledge_filters.get("top_k", knowledge.default_top_k)
        )
        # `top_k` is the per-query retrieval width; `final_k` is how many chunks
        # actually reach the answer. The profile widens the latter (deep > 1.0)
        # so its decompose + gate fan-out surfaces a broader, multi-document
        # evidence set instead of being re-collapsed back to `top_k`. An explicit
        # request `top_k` stays the base the factor scales.
        # A request may pin `final_k` directly (validated to 1..EVIDENCE_K_MAX at
        # the resolver); the explicit value then beats the profile factor, and
        # the override is surfaced so the live ledger never hides it.
        explicit_final_k = request.knowledge_filters.get("final_k")
        final_k_overridden = explicit_final_k is not None
        final_k = (
            min(int(explicit_final_k), EVIDENCE_K_MAX)
            if final_k_overridden
            else min(int(top_k * plan.final_k_factor), EVIDENCE_K_MAX)
        )

        decompose_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        sub_queries: tuple[str, ...] = ()
        if plan.decompose:
            sub_queries = self._run_decomposition(
                llm, context, query_question, emit, decompose_usage
            )

        def _retrieve(query: str, k: int = top_k) -> list[RetrievalCandidate]:
            # The research graph is synchronous (a node on a run-worker
            # thread); the knowledge store is async. Bridge per call.
            return run_coro_sync(
                retrieve(
                    knowledge,
                    query=query,
                    collection_ids=collection_ids,
                    top_k=k,
                    use_reranker=plan.rerank,
                    rerank_candidate_depth=plan.rerank_candidate_depth,
                )
            )

        if sub_queries:
            # Round-robin across the original question and every sub-query so
            # each aspect contributes; each list is per-query `top_k`, the union
            # is capped at the profile's wider `final_k`.
            result_lists = [_retrieve(query_question)] + [
                _retrieve(sub_query) for sub_query in sub_queries
            ]
            candidates = interleave_candidates(result_lists, limit=final_k)
        else:
            # No decomposition: a single query must fill the evidence budget
            # itself, so retrieve `final_k` directly.
            candidates = _retrieve(query_question, final_k)
        emit(
            "inqtrix.knowledge.retrieval.completed",
            {
                "candidate_count": len(candidates),
                "top_k": top_k,
                "final_k": final_k,
                "final_k_overridden": final_k_overridden,
                # Coverage: how many documents are in scope — ALL are searched
                # (the filter is collection-level), so this confirms the answer
                # considered every indexed document.
                "collection_document_count": _scoped_document_count(
                    knowledge, collection_ids
                ),
                "embedding_model": knowledge.embeddings.default_model,
            },
        )
        context_window = getattr(llm, "context_window_tokens", None)
        budget_chars = max(
            _EVIDENCE_BUDGET_FLOOR_CHARS,
            ((context_window or 16_000) - _PROMPT_RESERVED_TOKENS) * 3,
        )
        evidence_block, used_candidates = _render_evidence_block(
            candidates, max_chars=budget_chars
        )
        if len(used_candidates) < len(candidates):
            log.warning(
                "Knowledge-Evidenz auf %d von %d Treffern gekuerzt "
                "(Kontextbudget).",
                len(used_candidates),
                len(candidates),
            )
            emit(
                "inqtrix.knowledge.evidence.truncated",
                {
                    "kept": len(used_candidates),
                    "dropped": len(candidates) - len(used_candidates),
                },
            )

        queries_run = [query_question, *sub_queries]
        gate_state: dict[str, Any] = {"enabled": plan.gate_enabled}
        gate_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if plan.gate_enabled and used_candidates:
            decision = self._run_gate(
                llm, context, query_question, evidence_block, emit,
                gate_usage,
                round_index=0,
                vocabulary_bridge=plan.vocabulary_bridge,
            )
            gate_state.update(
                marker=decision.marker,
                sufficient=decision.sufficient,
                coverage=decision.coverage,
                reason=decision.reason,
            )
            rounds_used = 0
            # The agentic loop: rewrite-and-retrieve until the gate is
            # satisfied or the profile's round budget is spent. One
            # round reproduces the pre-profile flow call for call.
            while (
                not decision.sufficient
                and decision.rewritten_query
                and rounds_used < plan.gate_rewrite_rounds
            ):
                rounds_used += 1
                queries_run.append(decision.rewritten_query)
                before = len(candidates)
                second = _retrieve(decision.rewritten_query, final_k)
                candidates = merge_candidates(
                    candidates, second, limit=final_k
                )
                if len(candidates) == before:
                    # The rewrite surfaced NO new evidence (merge dedupes on
                    # chunk id) — the corpus is exhausted for this question, so
                    # re-gating identical evidence would only repeat the verdict.
                    # Stop early: saves this round's gate LLM call AND every
                    # remaining round, and make the reason visible (no silent
                    # spin). Productive rounds (that DO add evidence) are
                    # unaffected.
                    gate_state["exhausted"] = True
                    emit(
                        "inqtrix.knowledge.gate.exhausted",
                        {"round": rounds_used, "reason": "no_new_evidence"},
                    )
                    break
                evidence_block, used_candidates = _render_evidence_block(
                    candidates, max_chars=budget_chars
                )
                decision = self._run_gate(
                    llm, context, query_question, evidence_block, emit,
                    gate_usage,
                    round_index=rounds_used,
                    vocabulary_bridge=plan.vocabulary_bridge,
                )
                gate_state.update(
                    marker=decision.marker,
                    sufficient=decision.sufficient,
                    coverage=decision.coverage,
                    reason=decision.reason,
                )
            gate_state.update(
                rounds_used=rounds_used,
                max_rounds=plan.gate_rewrite_rounds,
            )
            if rounds_used >= 1:
                # Back-compat key consumed by the answer-eval harness.
                gate_state["second_pass"] = True
            if not decision.sufficient and decision.coverage == "none":
                # Honest refusal ONLY for IRRELEVANT evidence: the
                # gate found nothing on-topic after the rewrite
                # budget — answering would fabricate confidence.
                # PARTIAL coverage answers instead: the answer prompt
                # names the gaps explicitly, which serves the user
                # strictly better than a blanket refusal (live-eval
                # finding: the binary verdict refused 10/18 answerable
                # DORA questions that had partial evidence).
                log.warning(
                    "Knowledge-Gate: Evidenz irrelevant (%s) — "
                    "ehrliche Keine-Evidenz-Antwort.",
                    decision.reason,
                )
                used_candidates = []
            elif not decision.sufficient:
                log.info(
                    "Knowledge-Gate: Evidenz nur teilweise ausreichend "
                    "(%s) — Antwort benennt die Luecken.",
                    decision.reason,
                )

        grounding_state: dict[str, Any] = {
            "enabled": plan.grounding_enabled
        }
        if not used_candidates:
            answer = (
                "In den durchsuchten Dokumenten wurden keine relevanten "
                "Inhalte zu dieser Frage gefunden."
            )
            usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0}
        else:
            provider_models = getattr(llm, "models", None)
            requested_tier = (
                context.agent_settings.model_tier or ""
            ).strip() or None
            requested_model = (context.agent_settings.model or "").strip()
            requested_effort = (context.agent_settings.effort or "").strip()
            if requested_model:
                model: str | None = requested_model
                effort: str | None = requested_effort or None
            elif provider_models is not None:
                model = (
                    resolve_model(
                        "knowledge_answer", provider_models, requested_tier
                    )
                    or None
                )
                effort = (
                    resolve_effort(
                        "knowledge_answer", provider_models, requested_tier
                    )
                    or None
                )
            else:
                model = None
                effort = None
            prompt = build_knowledge_answer_prompt(
                request.question,
                evidence_block,
                history=request.history,
                grounding=plan.grounding_enabled,
                report=plan.report,
            )
            response = llm.complete_with_metadata(
                prompt,
                model=model,
                reasoning_effort=effort,
                timeout=context.agent_settings.reasoning_timeout,
            )
            answer = response.content
            usage = {
                "prompt_tokens": getattr(response, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response, "completion_tokens", 0)
                or 0,
            }
            if plan.grounding_enabled:
                # Verification runs against the chunks' SOURCE text:
                # a "verbatim, verified" quote must exist in the cited
                # document, not in the contextualization prefix or the
                # rendering scaffolding the prompt carries.
                report = check_grounding(
                    answer,
                    [
                        candidate.chunk.source_text or candidate.chunk.text
                        for candidate in used_candidates
                    ],
                )
                answer = report.answer
                unverified = [
                    quote for quote in report.quotes if not quote.verified
                ]
                grounding_state.update(
                    marker=report.marker,
                    quotes_total=len(report.quotes),
                    quotes_verified=len(report.quotes) - len(unverified),
                    quotes=[
                        {
                            "label": quote.label,
                            "text": quote.text,
                            "verified": quote.verified,
                        }
                        for quote in report.quotes
                    ],
                )
                if report.marker == GROUNDING_MARKER_FALLBACK:
                    log.warning(
                        "Knowledge-Grounding: Antwort ohne parsebaren "
                        "ZITATE-Block — ungeprueft durchgereicht (%s).",
                        GROUNDING_MARKER_FALLBACK,
                    )
                elif unverified:
                    log.warning(
                        "Knowledge-Grounding: %d von %d Zitaten nicht "
                        "woertlich in der Evidenz belegbar (%s).",
                        len(unverified),
                        len(report.quotes),
                        ", ".join(quote.label for quote in unverified),
                    )
                emit(
                    "inqtrix.knowledge.grounding.checked",
                    {
                        "marker": report.marker,
                        "quotes_total": len(report.quotes),
                        "quotes_verified": len(report.quotes)
                        - len(unverified),
                    },
                )

        usage["prompt_tokens"] += (
            contextualize_usage["prompt_tokens"]
            + gate_usage["prompt_tokens"]
            + decompose_usage["prompt_tokens"]
        )
        usage["completion_tokens"] += (
            contextualize_usage["completion_tokens"]
            + gate_usage["completion_tokens"]
            + decompose_usage["completion_tokens"]
        )

        references = [
            {
                "label": f"K{index}",
                "url": self._reference_url(candidate),
                "tier": "primary",
                "title": candidate.document_title,
                # The exact retrieved passage travels WITH the citation so the
                # client can show "where this came from" (the cited chunk, with
                # the quoted span highlighted) without a second fetch, and open
                # the source reliably via the explicit document id.
                "document_id": candidate.chunk.document_id,
                "chunk_index": candidate.chunk.chunk_index,
                "excerpt": candidate.chunk.text,
                "source_text": candidate.chunk.source_text or candidate.chunk.text,
                # Best-effort source page (PDFs only) for a page-level "open at
                # page N" jump; None when unmapped.
                "page_number": candidate.chunk.page_number,
            }
            for index, candidate in enumerate(used_candidates, start=1)
        ]
        raw: dict[str, Any] = {
            "answer": answer,
            "usage": usage,
            "result_state": {
                "answer": answer,
                "round": 1,
                "queries": queries_run,
                "knowledge_gate": gate_state,
                "knowledge_grounding": grounding_state,
                "all_citations": [reference["url"] for reference in references],
                "report_references": references,
                "knowledge_candidates": len(candidates),
                "knowledge_evidence_used": len(used_candidates),
                "knowledge_collections": collection_ids or [],
                "knowledge_profile": {
                    "id": plan.profile.value,
                    "requested": (
                        plan.requested_profile.value
                        if plan.requested_profile is not None
                        else None
                    ),
                    "auto_selected": plan.auto_selected,
                    "auto_reason": plan.auto_reason,
                    "degraded_stages": list(plan.degraded_stages),
                },
                "elapsed_seconds": round(time.monotonic() - started, 2),
            },
        }
        # Only present on a real tokenizer-language mismatch — keeps the
        # same-language default result_state field-identical (No Silent
        # Fallbacks without new noise).
        if sparse_state is not None:
            raw["result_state"]["knowledge_sparse"] = sparse_state
        if contextualization_state["used_history"]:
            raw["result_state"]["knowledge_contextualization"] = {
                **contextualization_state,
            }
        return AgentResult(
            answer=answer,
            result_type="knowledge_result",
            raw=raw,
        )
