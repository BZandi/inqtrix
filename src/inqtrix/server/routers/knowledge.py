"""Knowledge endpoints: collections, documents, and retrieval search.

Registered only when the knowledge engine is enabled — a disabled
deployment has no knowledge surface at all (404s), keeping the
historical route set untouched.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Mapping

from fastapi import APIRouter, Depends, Request

from inqtrix.auth.permissions import SharePermission
from inqtrix.auth.principal import Principal, UserContext
from inqtrix.knowledge.stores.ports import (
    CollectionNotFound,
    DocumentNotFound,
    EmbeddingDimensionMismatch,
    KnowledgeCollection,
    KnowledgeDocument,
    RetrievalCandidate,
)
from inqtrix.content.ports import FileNotFound
from inqtrix.knowledge.page_mapping import extract_pdf_page_texts
from inqtrix.knowledge.parsing import DocumentParseError
from inqtrix.pagination import (
    InvalidCursor,
    clamp_limit,
    decode_cursor,
    list_envelope,
)
from inqtrix.providers.embeddings import EmbeddingProviderError
from inqtrix.quota.models import QuotaDimension, estimate_tokens
from inqtrix.runs.shared import access_annotation
from inqtrix.server.routers import (
    build_shared_grants_dependency,
    quota_admission,
    quota_record,
)
from inqtrix.services.knowledge_service import (
    ChunkNotFound,
    KnowledgeValidationError,
    collection_access,
)
from inqtrix.services.request_parsing import error_response

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")


def _collection_payload(
    collection: KnowledgeCollection,
    *,
    access: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": collection.id,
        "name": collection.name,
        "embedding_model": collection.embedding_model,
        "embedding_dim": collection.embedding_dim,
        "document_count": collection.document_count,
        "created_at": collection.created_at,
        **({"access": access} if access is not None else {}),
    }


def _document_payload(document: KnowledgeDocument) -> dict[str, Any]:
    return {
        "id": document.id,
        "collection_id": document.collection_id,
        "title": document.title,
        "metadata": dict(document.metadata),
        "chunk_count": document.chunk_count,
        "created_at": document.created_at,
    }


_SEARCH_TOP_K_MAX = 50
"""Upper bound for the debug-search ``top_k`` (mirrors the agent_context
retrieval-width validator); requests above it are rejected, not silently
clamped."""

_CHUNK_CONTEXT_MAX = 3
"""Upper bound for the chunk-detail ``context`` query parameter
(neighbour chunks per side). Three chunks each way already exceed what
the evidence view renders; larger windows should read the document text
endpoint instead. Requests above it are rejected, not silently
clamped."""


def _candidate_payload(candidate: RetrievalCandidate, *, rank: int) -> dict[str, Any]:
    """One search hit.

    Additive identity/provenance fields (``chunk_id``, ``rank``,
    ``page_number``, ``source_text``) let an agent cite a hit durably:
    the citation key stays ``(document_id, chunk_index)`` (stable across
    reindex), while ``chunk_id`` and ``source_text`` support exact
    provenance and verbatim-quote checks. Legacy consumers ignore the
    extra keys; the historical fields keep their shape.
    """
    chunk = candidate.chunk
    return {
        "document_id": chunk.document_id,
        "collection_id": chunk.collection_id,
        "document_title": candidate.document_title,
        "chunk_index": chunk.chunk_index,
        "chunk_id": chunk.id,
        "rank": rank,
        "text": chunk.text,
        "source_text": chunk.source_text or chunk.text,
        "page_number": chunk.page_number,
        "score": round(candidate.score, 6),
    }


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the knowledge routes against the container.

    Raises:
        RuntimeError: When called without a wired knowledge service —
            registration is a composition decision, not a runtime
            fallback.
    """
    service = container.knowledge_service
    if service is None:
        raise RuntimeError(
            "build_router(knowledge) requires a wired knowledge service; "
            "register the router only when knowledge is enabled."
        )
    router = APIRouter()
    principal_dep = container.principal_dependency
    user_context_dep = container.user_context_dependency
    file_service = container.file_service
    quota_service = container.quota_service
    share_service = container.share_service
    workspace_admin = container.workspace_admin
    shared_collections_dep = build_shared_grants_dependency(
        share_service, principal_dep, resource_type="knowledge_collection"
    )

    @router.post("/v1/knowledge/collections", status_code=201)
    async def create_collection(
        req: Request,
        principal: Principal = Depends(principal_dep),
    ):
        """Create a collection with an immutable embedding model."""
        try:
            body = await req.json()
        except Exception:
            return error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")
        if not isinstance(body, dict):
            return error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")
        try:
            collection = await service.create_collection(
                name=str(body.get("name", "")),
                embedding_model=(
                    str(body["embedding_model"])
                    if body.get("embedding_model") is not None
                    else None
                ),
                # Scoped principals own what they create; the
                # anonymous/static principals keep minting legacy
                # (visible-to-all) collections.
                created_by_sub=(
                    principal.sub
                    if principal.kind in ("oidc_session", "pat")
                    else None
                ),
            )
        except KnowledgeValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except EmbeddingProviderError as exc:
            log.warning("Collection-Anlage scheiterte am Embedding-Backend: %s", exc)
            return error_response(502, str(exc), "server_error")
        return _collection_payload(collection)

    @router.get("/v1/knowledge/collections")
    async def list_collections(
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_collections_dep),
    ):
        """The caller's visible collections, newest first."""
        payloads = []
        for collection in await service.list_collections(
            visible_to=visible_to, also_visible=also_visible
        ):
            shared = collection_access(collection, visible_to, also_visible)
            payloads.append(
                _collection_payload(
                    collection, access=access_annotation(shared)
                )
            )
        return {"object": "list", "data": payloads}

    @router.delete("/v1/knowledge/collections/{collection_id}", status_code=204)
    async def delete_collection(
        collection_id: str,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_collections_dep),
    ):
        """Delete a collection with all its documents (owner-only)."""
        try:
            await service.delete_collection(
                collection_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except CollectionNotFound:
            return error_response(404, "Collection nicht gefunden", "not_found")
        if workspace_admin is not None and share_service is not None:
            # A deleted collection must not leave grants dangling —
            # the recipients' shared-with-me would otherwise keep
            # naming a resource that no longer exists.
            revoked = await workspace_admin.revoke_shares_for_resource(
                tenant_id=principal.tenant_id,
                resource_type="knowledge_collection",
                resource_id=collection_id,
                revoked_by_sub=principal.sub,
            )
            if revoked:
                log.info(
                    "Collection %s geloescht; %d Freigaben entzogen",
                    collection_id,
                    revoked,
                )

    @router.post(
        "/v1/knowledge/collections/{collection_id}/documents", status_code=201
    )
    async def add_document(
        collection_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_collections_dep),
    ):
        """Ingest one document (chunk + embed) synchronously."""
        try:
            body = await req.json()
        except Exception:
            return error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")
        if not isinstance(body, dict):
            return error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")
        metadata = body.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            return error_response(
                400, "Feld 'metadata' muss ein Objekt sein", "invalid_request_error"
            )
        file_id = str(body.get("file_id", "") or "")
        if file_id and body.get("text"):
            return error_response(
                400,
                "Entweder 'text' ODER 'file_id' angeben, nicht beides",
                "invalid_request_error",
            )
        # Embedding-token admission (block-next): a caller already over
        # their monthly embedding budget is denied before any parse or
        # embed call. A single ingestion cannot run away — per-document
        # size is bounded by the document/file limits — so the exact
        # embedded-text tokens are recorded after success.
        denied = await quota_admission(
            quota_service, principal, QuotaDimension.EMBEDDING_TOKENS
        )
        if denied is not None:
            return denied
        try:
            if file_id:
                document = await _ingest_file(
                    collection_id=collection_id,
                    file_id=file_id,
                    title=str(body.get("title", "")),
                    metadata=metadata,
                    principal=principal,
                    visible_to=visible_to,
                    also_visible=also_visible,
                )
            else:
                # Text path: the caller reuses already-extracted text (e.g. the
                # server's MarkItDown parse) and does NOT re-parse. When the
                # document still references its original server file
                # (metadata.file_id), run a lightweight page pass over that file
                # (pdfminer only — no MarkItDown re-parse, no LLM) so per-chunk
                # page numbers are still captured for the citation page-jump.
                page_texts = await _page_texts_for_metadata(metadata, principal)
                document = await service.add_document(
                    collection_id=collection_id,
                    title=str(body.get("title", "")),
                    text=str(body.get("text", "")),
                    metadata=metadata,
                    visible_to=visible_to,
                    also_visible=also_visible,
                    page_texts=page_texts,
                )
        except FileNotFound:
            return error_response(404, "Datei nicht gefunden", "not_found")
        except DocumentParseError as exc:
            return error_response(422, str(exc), "invalid_request_error")
        except KnowledgeValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except CollectionNotFound:
            return error_response(404, "Collection nicht gefunden", "not_found")
        except EmbeddingDimensionMismatch as exc:
            log.warning("Embedding-Dimension-Konflikt bei Ingestion: %s", exc)
            return error_response(409, str(exc), "embedding_dimension_mismatch")
        except EmbeddingProviderError as exc:
            log.warning("Ingestion scheiterte am Embedding-Backend: %s", exc)
            return error_response(502, str(exc), "server_error")
        # Book the exact embedded-text size (the chunks partition this
        # text; the ~4-char heuristic mirrors the embedding provider's
        # token accounting closely enough for a usage meter).
        await quota_record(
            quota_service,
            principal,
            QuotaDimension.EMBEDDING_TOKENS,
            estimate_tokens(document.text),
        )
        return _document_payload(document)

    async def _ingest_file(
        *,
        collection_id: str,
        file_id: str,
        title: str,
        metadata: dict[str, Any] | None,
        principal: Principal,
        visible_to: "UserContext | None" = None,
        also_visible: "Mapping[str, SharePermission] | None" = None,
    ):
        """Fetch a registered file (access-checked), parse, ingest.

        The file read goes through the FileService so the same
        owner/share rules gate knowledge ingestion as gate downloads;
        parsing and ingestion run off the event loop.
        """
        if file_service is None:
            raise KnowledgeValidationError(
                "Datei-Ingestion ist nicht verfuegbar (files-Feature aus)"
            )
        record, chunks_iter = await file_service.open_stream(
            file_id, principal=principal
        )
        content = await asyncio.to_thread(lambda: b"".join(chunks_iter))
        document_metadata = dict(metadata or {})
        document_metadata.setdefault("file_id", record.id)
        document_metadata.setdefault("file_name", record.file_name)
        return await service.add_document_from_file(
            collection_id=collection_id,
            file_name=record.file_name,
            content=content,
            metadata=document_metadata,
            title=title,
            visible_to=visible_to,
            also_visible=also_visible,
        )

    async def _page_texts_for_metadata(
        metadata: dict[str, Any] | None, principal: Principal
    ) -> list[str] | None:
        """Best-effort per-page text of a text-ingested document's ORIGINAL file
        for chunk page mapping.

        Reads ``metadata.file_id`` through the FileService (same access rules as
        a download), then extracts per-page text with pdfminer only — it does NOT
        re-parse the document (the caller already supplied the text). Returns
        ``None`` when there is no file id, the files feature is off, or the file
        is gone — the document still ingests, just without page numbers.
        """
        source_file_id = str((metadata or {}).get("file_id", "") or "")
        if not source_file_id or file_service is None:
            return None
        try:
            _record, chunks_iter = await file_service.open_stream(
                source_file_id, principal=principal
            )
            content = await asyncio.to_thread(lambda: b"".join(chunks_iter))
        except FileNotFound:
            return None
        return await asyncio.to_thread(extract_pdf_page_texts, content)

    @router.get("/v1/knowledge/collections/{collection_id}/documents")
    async def list_documents(
        collection_id: str,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_collections_dep),
    ):
        """List a collection's documents, newest first (keyset-paginated).

        ``?limit=`` (clamped) and ``?cursor=`` page the list; the response
        carries ``next_cursor`` (null on the last page). Both params are
        optional — without them the first page at the default size returns,
        keeping older callers working."""
        try:
            after = decode_cursor(req.query_params.get("cursor"))
        except InvalidCursor:
            return error_response(400, "Ungueltiger Cursor", "invalid_cursor")
        limit = clamp_limit(req.query_params.get("limit"))
        try:
            documents, next_cursor = await service.list_documents_page(
                collection_id,
                limit=limit,
                after=after,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except CollectionNotFound:
            return error_response(404, "Collection nicht gefunden", "not_found")
        return list_envelope(
            [_document_payload(document) for document in documents],
            next_cursor,
        )

    @router.delete("/v1/knowledge/documents/{document_id}", status_code=204)
    async def delete_document(
        document_id: str,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_collections_dep),
    ):
        """Delete one document and its chunks (edit via parent)."""
        try:
            await service.delete_document(
                document_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except DocumentNotFound:
            return error_response(404, "Dokument nicht gefunden", "not_found")

    @router.get("/v1/knowledge/documents/{document_id}/text")
    async def document_text(
        document_id: str,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_collections_dep),
    ):
        """Full extracted text of one document (the reader's source).

        Serves the document viewer: the extracted view renders exactly
        what was ingested (what retrieval sees), and snippet/quote
        highlighting works by text search within this payload.
        ``metadata.file_id`` (set by file ingestion) links to the
        original binary under ``/v1/files/{file_id}/content``.
        """
        try:
            document = await service.get_document(
                document_id,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except DocumentNotFound:
            return error_response(404, "Dokument nicht gefunden", "not_found")
        payload = _document_payload(document)
        payload["text"] = document.text
        return payload

    @router.get("/v1/knowledge/documents/{document_id}/chunks/{chunk_index}")
    async def document_chunk(
        document_id: str,
        chunk_index: int,
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_collections_dep),
    ):
        """One chunk plus optional neighbour context (the evidence view).

        ``?context=N`` (0..3, default 0) additionally returns up to N
        chunks before and after the target so a cited quote can be read
        in its surroundings. Scoping mirrors the document text route:
        an unknown and an invisible document stay byte-identical 404s.
        """
        raw_context = req.query_params.get("context")
        context = 0
        if raw_context is not None:
            try:
                context = int(raw_context)
            except ValueError:
                context = -1
            if not 0 <= context <= _CHUNK_CONTEXT_MAX:
                return error_response(
                    400,
                    (
                        "Parameter 'context' muss zwischen 0 und "
                        f"{_CHUNK_CONTEXT_MAX} liegen"
                    ),
                    "invalid_request_error",
                )
        try:
            chunk, neighbors = await service.get_chunk(
                document_id,
                chunk_index,
                context=context,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except DocumentNotFound:
            return error_response(404, "Dokument nicht gefunden", "not_found")
        except ChunkNotFound:
            return error_response(404, "Chunk nicht gefunden", "not_found")
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "source_text": chunk.source_text or chunk.text,
            "page_number": chunk.page_number,
            "neighbors": [
                {"chunk_index": neighbor.chunk_index, "text": neighbor.text}
                for neighbor in neighbors
            ],
        }

    @router.post("/v1/knowledge/search")
    async def search(
        req: Request,
        principal: Principal = Depends(principal_dep),
        visible_to: UserContext | None = Depends(user_context_dep),
        also_visible=Depends(shared_collections_dep),
    ):
        """Synchronous retrieval search for debugging and evaluation."""
        try:
            body = await req.json()
        except Exception:
            return error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")
        if not isinstance(body, dict):
            return error_response(400, "Ungueltiger JSON-Body", "invalid_request_error")
        raw_collection_ids = body.get("collection_ids")
        if raw_collection_ids is not None and not isinstance(raw_collection_ids, list):
            return error_response(
                400,
                "Feld 'collection_ids' muss eine Liste sein",
                "invalid_request_error",
            )
        collection_ids = (
            [str(item) for item in raw_collection_ids]
            if raw_collection_ids is not None
            else None
        )
        raw_top_k = body.get("top_k")
        if raw_top_k is not None and (
            not isinstance(raw_top_k, int)
            or not 1 <= raw_top_k <= _SEARCH_TOP_K_MAX
        ):
            return error_response(
                400,
                f"Feld 'top_k' muss zwischen 1 und {_SEARCH_TOP_K_MAX} liegen",
                "invalid_request_error",
            )
        try:
            outcome = await service.search_reported(
                query=str(body.get("query", "")),
                collection_ids=collection_ids,
                top_k=raw_top_k,
                visible_to=visible_to,
                also_visible=also_visible,
            )
        except KnowledgeValidationError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        except CollectionNotFound:
            return error_response(404, "Collection nicht gefunden", "not_found")
        except EmbeddingDimensionMismatch as exc:
            return error_response(409, str(exc), "embedding_dimension_mismatch")
        except EmbeddingProviderError as exc:
            log.warning("Knowledge-Suche scheiterte am Embedding-Backend: %s", exc)
            return error_response(502, str(exc), "server_error")
        warnings: list[dict[str, Any]] = []
        if outcome.filtered_collection_ids:
            warnings.append(
                {
                    "code": "collections_filtered",
                    "message": (
                        "Einzelne angefragte Sammlungen sind nicht sichtbar "
                        "und wurden aus der Suche ausgeschlossen."
                    ),
                    "filtered_ids": outcome.filtered_collection_ids,
                }
            )
        return {
            "object": "list",
            "data": [
                _candidate_payload(candidate, rank=index)
                for index, candidate in enumerate(outcome.candidates, start=1)
            ],
            "warnings": warnings,
        }

    return router
