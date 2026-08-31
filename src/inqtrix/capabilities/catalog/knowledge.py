"""Wave-1 knowledge capabilities (read-only).

Thin wrappers over :class:`~inqtrix.services.knowledge_service.KnowledgeService`.
The agent path uses STRICT collection visibility (``knowledge.search``
asserts every requested collection is visible, unlike the legacy debug
route that silently filters) — an agent must never plan against fewer
sources than it asked for (No Silent Fallbacks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from inqtrix.capabilities.contracts import (
    CapabilityContext,
    CapabilityDefinition,
    CapabilityError,
    Effect,
)
from inqtrix.knowledge.stores.ports import CollectionNotFound, DocumentNotFound
from inqtrix.knowledge.evidence import KnowledgeEvidenceProjector
from inqtrix.knowledge.retrieval_warnings import (
    project_retrieval_exclusion_warnings,
)
from inqtrix.services.knowledge_service import KnowledgeValidationError

if TYPE_CHECKING:
    from inqtrix.services.knowledge_service import KnowledgeService

_TOP_K_MAX = 50

# No standalone knowledge.chunk.read capability: a search hit already carries
# its verified excerpt, stable citation coordinates, and page provenance.


class CollectionsListInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CollectionSummary(BaseModel):
    id: str
    name: str
    embedding_model: str
    document_count: int


class CollectionsListOutput(BaseModel):
    collections: list[CollectionSummary]


class KnowledgeSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    collection_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(8, ge=1, le=_TOP_K_MAX)


class KnowledgeHit(BaseModel):
    document_id: str
    collection_id: str
    document_title: str
    chunk_index: int
    chunk_id: str
    rank: int
    excerpt: str
    page_number: int | None
    score: float
    source_span: dict[str, object] | None = None
    revision_id: str | None = None
    generation_id: str | None = None
    provenance_status: str


class KnowledgeSearchWarning(BaseModel):
    code: str
    message: str
    retrieval_mode: str = ""
    stage: str = ""
    requested_candidate_pool: int = Field(0, ge=0)
    returned_candidate_pool: int = Field(0, ge=0)
    final_top_k: int = Field(0, ge=0)
    final_evidence_complete: bool = False
    # Compatibility projection for clients predating the explicit candidate
    # pool fields. These counters now always describe the final evidence set.
    requested_top_k: int = Field(0, ge=0)
    returned_hits: int = Field(0, ge=0)
    candidate_cap: int | None = Field(None, ge=0)
    count: int = Field(0, ge=0)


class KnowledgeSearchOutput(BaseModel):
    query: str
    hits: list[KnowledgeHit]
    warnings: list[KnowledgeSearchWarning] = Field(default_factory=list)


class DocumentReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(..., min_length=1)


class DocumentReadOutput(BaseModel):
    id: str
    collection_id: str
    title: str
    text: str
    chunk_count: int


def build_knowledge_capabilities(
    service: "KnowledgeService",
) -> list[CapabilityDefinition]:
    """Build the wave-1 knowledge capabilities bound to *service*."""

    async def _list(
        _payload: CollectionsListInput, context: CapabilityContext
    ) -> CollectionsListOutput:
        collections = await service.list_collections(
            visible_to=context.visible_to,
        )
        if context.knowledge_collection_ids is not None:
            collections = [
                collection
                for collection in collections
                if collection.id in context.knowledge_collection_ids
            ]
        return CollectionsListOutput(
            collections=[
                CollectionSummary(
                    id=collection.id,
                    name=collection.name,
                    embedding_model=collection.embedding_model,
                    document_count=collection.document_count,
                )
                for collection in collections
            ]
        )

    async def _search(
        payload: KnowledgeSearchInput, context: CapabilityContext
    ) -> KnowledgeSearchOutput:
        collection_ids = list(payload.collection_ids)
        pinned_ids = context.knowledge_collection_ids
        if pinned_ids is not None:
            if collection_ids and not set(collection_ids).issubset(pinned_ids):
                raise CapabilityError(
                    "knowledge.collection_not_found",
                    "Mindestens eine angefragte Sammlung ist nicht sichtbar.",
                    http_status=404,
                )
            if not collection_ids:
                collection_ids = sorted(pinned_ids)
            if not collection_ids:
                # P10-K4: an EMPTY boundary is not "nothing matched" — it
                # is "there is nothing to search". Returning bare empty
                # hits made the two indistinguishable for the model and
                # for the user; the reason travels as a warning so no
                # downstream layer has to guess it.
                return KnowledgeSearchOutput(
                    query=payload.query,
                    hits=[],
                    warnings=[
                        KnowledgeSearchWarning(
                            code="knowledge.no_collections",
                            message=(
                                "Keine Wissenssammlung im Zugriff dieses "
                                "Laufs: es gibt nichts zu durchsuchen."
                            ),
                            stage="scope",
                        )
                    ],
                )

        # Strict: an explicit collection set the agent cannot fully see is
        # a denial, not a quiet narrowing of the search (E5).
        if collection_ids:
            try:
                await service.assert_collections_visible(
                    collection_ids,
                    visible_to=context.visible_to,
                )
            except CollectionNotFound as exc:
                raise CapabilityError(
                    "knowledge.collection_not_found",
                    "Mindestens eine angefragte Sammlung ist nicht sichtbar.",
                    http_status=404,
                ) from exc
        try:
            outcome = await service.search_reported(
                query=payload.query,
                collection_ids=collection_ids or None,
                top_k=payload.top_k,
                visible_to=context.visible_to,
            )
        except KnowledgeValidationError as exc:
            raise CapabilityError("invalid_input", str(exc), http_status=400) from exc
        except CollectionNotFound as exc:
            raise CapabilityError(
                "knowledge.collection_not_found",
                "Sammlung nicht gefunden.",
                http_status=404,
            ) from exc
        if outcome.filtered_collection_ids:
            # Visibility may change between the admission check and retrieval.
            # The strict agent contract must still fail closed instead of
            # silently answering from the surviving subset.
            raise CapabilityError(
                "knowledge.collection_not_found",
                "Mindestens eine angefragte Sammlung ist nicht sichtbar.",
                http_status=404,
            )
        candidates = outcome.candidates
        warnings = [
            KnowledgeSearchWarning(
                code=degradation.reason,
                message=(
                    "Der Vektor-Kandidatenpool blieb unter der für das "
                    "Reranking angeforderten Tiefe; die finale Belegzahl "
                    "wurde dennoch vollständig erreicht."
                    if degradation.final_evidence_complete
                    else "Die Vektorsuche erreichte ihre technische "
                    "Kandidatengrenze, bevor die finale angeforderte "
                    "Belegzahl erreicht war."
                ),
                retrieval_mode=degradation.retrieval_mode,
                stage=degradation.stage,
                requested_candidate_pool=(
                    degradation.requested_candidate_pool or 0
                ),
                returned_candidate_pool=(
                    degradation.returned_candidate_pool or 0
                ),
                final_top_k=degradation.final_top_k or 0,
                final_evidence_complete=(
                    degradation.final_evidence_complete
                ),
                requested_top_k=degradation.requested_top_k,
                returned_hits=degradation.returned_hits,
                candidate_cap=degradation.candidate_cap,
            )
            for degradation in outcome.retrieval_degradations
        ]
        for warning in project_retrieval_exclusion_warnings(
            outcome.retrieval_exclusions
        ):
            warnings.append(
                KnowledgeSearchWarning(
                    code=warning.code,
                    message=warning.message,
                    requested_top_k=payload.top_k,
                    returned_hits=len(candidates),
                    count=warning.count,
                )
            )
        return KnowledgeSearchOutput(
            query=payload.query,
            hits=[
                KnowledgeHit(
                    document_id=evidence.document_id,
                    collection_id=evidence.collection_id,
                    document_title=evidence.title,
                    chunk_index=evidence.chunk_index,
                    chunk_id=evidence.chunk_id,
                    rank=rank,
                    excerpt=evidence.excerpt,
                    page_number=evidence.page_number,
                    score=round(evidence.score, 6),
                    source_span=evidence.as_dict()["source_span"],
                    revision_id=evidence.revision_id,
                    generation_id=evidence.generation_id,
                    provenance_status=evidence.provenance_status,
                )
                for rank, candidate in enumerate(candidates, start=1)
                for evidence in (
                    KnowledgeEvidenceProjector.project(
                        candidate, reference_id=f"K{rank}"
                    ),
                )
            ],
            warnings=warnings,
        )

    async def _read(
        payload: DocumentReadInput, context: CapabilityContext
    ) -> DocumentReadOutput:
        try:
            document = await service.get_document(
                payload.document_id,
                visible_to=context.visible_to,
            )
        except DocumentNotFound as exc:
            raise CapabilityError(
                "knowledge.document_not_found",
                "Dokument nicht gefunden.",
                http_status=404,
            ) from exc
        if (
            context.knowledge_collection_ids is not None
            and document.collection_id not in context.knowledge_collection_ids
        ):
            raise CapabilityError(
                "knowledge.document_not_found",
                "Dokument nicht gefunden.",
                http_status=404,
            )
        return DocumentReadOutput(
            id=document.id,
            collection_id=document.collection_id,
            title=document.title,
            text=document.text,
            chunk_count=document.chunk_count,
        )

    return [
        CapabilityDefinition(
            id="knowledge.collections.list",
            summary="List the knowledge collections the caller can see.",
            input_model=CollectionsListInput,
            output_model=CollectionsListOutput,
            effect=Effect.READ,
            idempotent=True,
            handler=_list,
        ),
        CapabilityDefinition(
            id="knowledge.search",
            summary=(
                "Search project knowledge collections; requested collections "
                "must all be visible (strict)."
            ),
            input_model=KnowledgeSearchInput,
            output_model=KnowledgeSearchOutput,
            effect=Effect.READ,
            idempotent=True,
            handler=_search,
        ),
        CapabilityDefinition(
            id="knowledge.document.read",
            summary="Read one knowledge document's full text and provenance.",
            input_model=DocumentReadInput,
            output_model=DocumentReadOutput,
            effect=Effect.READ,
            idempotent=True,
            handler=_read,
        ),
    ]
