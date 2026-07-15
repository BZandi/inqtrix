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
from inqtrix.services.knowledge_service import KnowledgeValidationError

if TYPE_CHECKING:
    from inqtrix.services.knowledge_service import KnowledgeService

_TOP_K_MAX = 50

# No standalone knowledge.chunk.read capability in wave 1: a search hit
# already carries chunk_id + text + source_text + page_number inline, so
# a per-chunk read would be redundant. A neighbour-context chunk endpoint
# lands with the canvas citation UI (M6), against get_chunks (M1).


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
    text: str
    source_text: str
    page_number: int | None
    score: float


class KnowledgeSearchOutput(BaseModel):
    query: str
    hits: list[KnowledgeHit]


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
                return KnowledgeSearchOutput(query=payload.query, hits=[])

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
            candidates = await service.search(
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
        return KnowledgeSearchOutput(
            query=payload.query,
            hits=[
                KnowledgeHit(
                    document_id=candidate.chunk.document_id,
                    collection_id=candidate.chunk.collection_id,
                    document_title=candidate.document_title,
                    chunk_index=candidate.chunk.chunk_index,
                    chunk_id=candidate.chunk.id,
                    rank=rank,
                    text=candidate.chunk.text,
                    source_text=candidate.chunk.source_text or candidate.chunk.text,
                    page_number=candidate.chunk.page_number,
                    score=round(candidate.score, 6),
                )
                for rank, candidate in enumerate(candidates, start=1)
            ],
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
