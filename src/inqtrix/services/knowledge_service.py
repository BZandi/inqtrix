"""Collection/document lifecycle and synchronous ingestion + search.

The service owns what the knowledge routers delegate: validation of
collection/document payloads, the chunk-then-embed ingestion step (one
synchronous pass in this cut — the worker-based pipeline replaces the
embed call site, nothing else), and scoped retrieval for the debug/
evaluation search endpoint.

Collections carry a canonical owner UUID. Ownerless legacy collections are
visible only in anonymous/static modes. Accepted direct shares authorize
scoped reads and edits through a live lookup; documents inherit the parent
collection decision. Deleting a collection stays owner-only. Every denial is
the indistinct :class:`CollectionNotFound`/:class:`DocumentNotFound`
(existence is not disclosed).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from inqtrix.auth.permissions import AccessMode, ResourceAccess, SharePermission
from inqtrix.embedding_cards import resolve_embedding_card
from inqtrix.knowledge.chunking import chunk_text
from inqtrix.knowledge.page_mapping import extract_pdf_page_texts, infer_chunk_pages

log = logging.getLogger("inqtrix")
from inqtrix.knowledge.retrieval import retrieve

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inqtrix.auth.permissions import AuthorizationService
    from inqtrix.auth.principal import UserContext
    from inqtrix.user_events import ResourceInvalidator
    from inqtrix.knowledge.parsing import DocumentParser
from inqtrix.knowledge.stores.ports import (
    CollectionMaintenanceActive,
    CollectionNotFound,
    DocumentChunk,
    DocumentNotFound,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeProviderContext,
    RetrievalCandidate,
)
from inqtrix.sync_bridge import run_coro_sync

_MutationResult = TypeVar("_MutationResult")


@dataclass(frozen=True)
class SearchOutcome:
    """A scoped search result plus the visibility it silently applied.

    The debug search endpoint filters an explicit ``collection_ids``
    list to its visible members instead of rejecting the whole request
    (unlike the strict ask-path gate). That filtering must not be
    invisible to an agent planning against the results, so the outcome
    reports which requested ids were dropped — the router renders them
    as a ``collections_filtered`` warning (No Silent Fallbacks).

    Attributes:
        candidates: The scored hits over the searched collections.
        filtered_collection_ids: Requested ids the caller may not see
            (empty for an unscoped caller or when nothing was dropped).
    """

    candidates: list[RetrievalCandidate]
    filtered_collection_ids: list[str] = field(default_factory=list)


class KnowledgeValidationError(ValueError):
    """Raised for client-payload problems (maps to HTTP 400)."""


class ChunkNotFound(KeyError):
    """Raised when a visible document has no chunk at the given index.

    Service-level sibling of the store's
    :class:`~inqtrix.knowledge.stores.ports.DocumentNotFound`: chunk
    identity is ``(document_id, chunk_index)`` and only the service
    resolves that pair, so the store port stays untouched. Maps to
    HTTP 404 with a chunk-specific message — the document itself was
    found and visible, which is safe to disclose.
    """


class KnowledgeService:
    """Application service over the knowledge provider context.

    Args:
        knowledge: The wired knowledge capabilities.
        chunk_max_chars: Character budget per chunk at ingestion.
        max_document_chars: Upper bound on one document's text size
            (the synchronous ingestion guard).
    """

    def __init__(
        self,
        *,
        knowledge: KnowledgeProviderContext,
        authorization: "AuthorizationService | None" = None,
        chunk_max_chars: int,
        max_document_chars: int,
        parser: "DocumentParser | None" = None,
        invalidator: "ResourceInvalidator | None" = None,
    ) -> None:
        self._knowledge = knowledge
        self._authorization = authorization
        self._chunk_max_chars = chunk_max_chars
        self._max_document_chars = max_document_chars
        self._parser = parser
        self._invalidator = invalidator
        self._maintenance_active_check: Callable[[str], bool] | None = None
        self._mutation_runner: (
            Callable[[str, Callable[[], Any]], Any] | None
        ) = None

    async def collection_access(
        self,
        collection: KnowledgeCollection,
        visible_to: "UserContext | None",
        *,
        minimum: SharePermission = SharePermission.VIEW,
    ) -> ResourceAccess:
        """Resolve current owner/direct-share access for one collection."""
        if visible_to is None:
            if collection.created_by_user_id is None:
                return ResourceAccess(AccessMode.UNSCOPED)
            raise CollectionNotFound(collection.id)
        if self._authorization is None:
            log.warning(
                "Knowledge authorization is unavailable for scoped user %s.",
                visible_to.principal.user_id,
            )
            raise CollectionNotFound(collection.id)
        access = await self._authorization.resolve_resource_access(
            visible_to.principal,
            owner_user_id=collection.created_by_user_id,
            resource_tenant_id=collection.tenant_id,
            resource_type="knowledge_collection",
            resource_id=collection.id,
            minimum=minimum,
        )
        if access is None:
            raise CollectionNotFound(collection.id)
        return access

    @staticmethod
    def _actor_user_id(visible_to: "UserContext | None") -> uuid.UUID | None:
        """Return the canonical actor carried by a scoped request."""
        return visible_to.principal.user_id if visible_to is not None else None

    @property
    def parser(self) -> "DocumentParser | None":
        """The wired document parser (``None`` = text-only ingestion)."""
        return self._parser

    @property
    def knowledge(self) -> KnowledgeProviderContext:
        """The wired knowledge capabilities (algorithm construction)."""
        return self._knowledge

    def bind_collection_maintenance(
        self,
        *,
        active_check: Callable[[str], bool],
        mutation_runner: Callable[[str, Callable[[], Any]], Any] | None = None,
    ) -> None:
        """Bind the reindex maintenance boundary after composition.

        ``KnowledgeService`` is constructed before ``IndexingService`` at the
        composition root, so the boundary is attached explicitly afterwards.
        The check supports every backend. The optional runner is used only by
        the in-memory job store to hold its existing lock across the final
        store mutation; Postgres enforces the same invariant with a collection
        row lock inside the repositories.
        """
        self._maintenance_active_check = active_check
        self._mutation_runner = mutation_runner

    async def _require_collection_mutable(self, collection_id: str) -> None:
        """Reject a mutation while queued/running/cancelling reindex owns it."""
        if self._maintenance_active_check is None:
            return
        active = await asyncio.to_thread(
            self._maintenance_active_check, collection_id
        )
        if active:
            raise CollectionMaintenanceActive(collection_id)

    async def _run_collection_mutation(
        self,
        collection_id: str,
        operation: Callable[[], Coroutine[Any, Any, _MutationResult]],
    ) -> _MutationResult:
        """Run the final store mutation behind the backend's boundary."""
        await self._require_collection_mutable(collection_id)
        if self._mutation_runner is None:
            return await operation()
        return await asyncio.to_thread(
            self._mutation_runner,
            collection_id,
            lambda: run_coro_sync(operation()),
        )

    # -- collections ------------------------------------------------------ #

    async def create_collection(
        self,
        *,
        name: str,
        embedding_model: str | None = None,
        created_by_user_id: uuid.UUID | None = None,
    ) -> KnowledgeCollection:
        """Create a collection with an immutable embedding identity.

        The embedding dimension comes from the embedding catalog when
        the model is catalogued; for uncatalogued models it is probed
        with one real embedding call so the recorded dimension is the
        truth of the actual backend, never a guess.

        Raises:
            KnowledgeValidationError: Empty name, or a model id outside
                the deployment's selectable set (when one is
                configured).
        """
        clean_name = (name or "").strip()
        if not clean_name:
            raise KnowledgeValidationError("Feld 'name' ist erforderlich")
        model = (embedding_model or "").strip() or self._knowledge.embeddings.default_model
        selectable = self._knowledge.embeddings.selectable_embedding_models
        if (
            selectable
            and model != self._knowledge.embeddings.default_model
            and model not in selectable
        ):
            raise KnowledgeValidationError(
                f"embedding_model {model!r} ist nicht verfuegbar "
                f"(verfuegbar: {', '.join(selectable)})"
            )
        card = resolve_embedding_card(model)
        if card is not None:
            dimension = card.dims
        else:
            probe = await asyncio.to_thread(
                self._knowledge.embeddings.embed_query,
                "dimension probe",
                model=model,
            )
            dimension = len(probe)
        collection = await self._knowledge.store.create_collection(
            name=clean_name,
            embedding_model=model,
            embedding_dim=dimension,
            created_by_user_id=created_by_user_id,
        )
        await self._invalidate_collection(collection)
        return collection

    async def list_collections(
        self,
        *,
        visible_to: "UserContext | None" = None,
    ) -> list[KnowledgeCollection]:
        """The caller's visible collections, newest first.

        Owned plus shared-in plus legacy (``created_by_user_id is None``);
        unscoped callers keep the historical see-everything view.
        """
        return [
            collection
            for collection, _access in await self.list_collections_with_access(
                visible_to=visible_to
            )
        ]

    async def list_collections_with_access(
        self,
        *,
        visible_to: "UserContext | None" = None,
    ) -> list[tuple[KnowledgeCollection, ResourceAccess]]:
        """Return one authoritative list snapshot with access annotations."""
        optimized = getattr(
            self._knowledge.store, "list_visible_collections", None
        )
        if callable(optimized):
            return await optimized(actor_user_id=self._actor_user_id(visible_to))
        collections = await self._knowledge.store.list_collections()
        annotated: list[tuple[KnowledgeCollection, ResourceAccess]] = []
        for collection in collections:
            try:
                access = await self.collection_access(collection, visible_to)
            except CollectionNotFound:
                continue
            annotated.append((collection, access))
        return annotated

    async def delete_collection(
        self,
        collection_id: str,
        *,
        visible_to: "UserContext | None" = None,
    ) -> None:
        """Delete a collection with all documents (owner-only).

        Share recipients never delete — v1 grants stop at edit, and a
        deletion destroys the resource for everyone, so even an edit
        grant earns the indistinct 404 here.
        """
        collection = await self._knowledge.store.get_collection(collection_id)
        access = await self.collection_access(collection, visible_to)
        if access.mode is AccessMode.SHARED:
            raise CollectionNotFound(collection_id)
        await self._run_collection_mutation(
            collection_id,
            lambda: self._knowledge.store.delete_collection(
                collection_id,
                actor_user_id=self._actor_user_id(visible_to),
            ),
        )
        if self._invalidator is not None and not getattr(
            self._knowledge.store, "atomic_resource_effects", False
        ):
            await self._invalidator.revoke_deleted(
                tenant_id=collection.tenant_id,
                owner_user_id=collection.created_by_user_id,
                resource_type="knowledge_collection",
                resource_id=collection.id,
                scope="knowledge_collections",
                actor_user_id=self._actor_user_id(visible_to),
            )

    # -- documents --------------------------------------------------------- #

    async def add_document(
        self,
        *,
        collection_id: str,
        title: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        visible_to: "UserContext | None" = None,
        page_texts: list[str] | None = None,
    ) -> KnowledgeDocument:
        """Chunk, embed, and store one document.

        Raises:
            KnowledgeValidationError: Empty title/text or text above
                the synchronous-ingestion size guard.
            inqtrix.knowledge.stores.ports.CollectionNotFound: Unknown
                collection.
            inqtrix.providers.embeddings.EmbeddingProviderError: When
                the embedding backend fails — surfaced, never swallowed
                into a partially indexed document.
        """
        clean_title = (title or "").strip()
        clean_text = (text or "").strip()
        if not clean_title:
            raise KnowledgeValidationError("Feld 'title' ist erforderlich")
        if not clean_text:
            raise KnowledgeValidationError("Feld 'text' ist erforderlich")
        if len(clean_text) > self._max_document_chars:
            raise KnowledgeValidationError(
                f"Dokument zu gross ({len(clean_text)} Zeichen, max. "
                f"{self._max_document_chars})"
            )
        collection = await self._knowledge.store.get_collection(collection_id)
        await self.collection_access(
            collection, visible_to, minimum=SharePermission.EDIT
        )
        await self._require_collection_mutable(collection_id)
        chunks, embeddings, source_chunks, marker = await self._embed_text(
            title=clean_title,
            text=clean_text,
            embedding_model=collection.embedding_model,
        )
        document_metadata = dict(metadata or {})
        if marker is not None:
            # The marker travels with the document so degraded
            # ingestions stay diagnosable per document, not just in
            # the log stream.
            document_metadata["_chunk_context"] = marker
        # Best-effort 1-based source page per chunk (PDFs only). Mapped against
        # the PRE-contextualization source chunks (a synthetic prefix would not
        # match the page text). Persisted on the document so a later re-embed
        # re-aligns by chunk index without re-reading the original file.
        page_numbers = infer_chunk_pages(source_chunks, page_texts)
        if any(page is not None for page in page_numbers):
            document_metadata["_chunk_pages"] = page_numbers
        stored = await self._run_collection_mutation(
            collection_id,
            lambda: self._knowledge.store.add_document(
                collection_id=collection_id,
                title=clean_title,
                text=clean_text,
                metadata=document_metadata,
                chunks=chunks,
                embeddings=embeddings,
                source_chunks=source_chunks,
                page_numbers=page_numbers,
                actor_user_id=self._actor_user_id(visible_to),
            ),
        )
        await self._invalidate_collection(collection)
        return stored

    async def _embed_text(
        self,
        *,
        title: str,
        text: str,
        embedding_model: str,
    ) -> tuple[list[str], list[list[float]], list[str], str | None]:
        """Chunk, optionally contextualize, and embed one document body.

        The single embed pipeline shared by first-time ingestion
        (:meth:`add_document`) and background re-embedding
        (:meth:`reembed_document`) so the two paths cannot drift in how
        they chunk or contextualize.

        Returns:
            ``(chunks, embeddings, source_chunks, marker)`` where
            ``source_chunks`` are the pre-contextualization bodies quote
            verification runs against, and ``marker`` is the
            contextualization diagnostic (``None`` when no
            contextualizer is wired).
        """
        chunks = chunk_text(text, max_chars=self._chunk_max_chars)
        # The pre-contextualization bodies: quote verification must run
        # against what the cited SOURCE actually contains, never against
        # synthetic prefixes.
        source_chunks = list(chunks)
        marker: str | None = None
        contextualizer = self._knowledge.contextualizer
        if contextualizer is not None:
            # Sync LLM call → off the event loop so a long
            # contextualization pass never stalls concurrent requests.
            contextualized = await asyncio.to_thread(
                contextualizer.contextualize,
                document_title=title,
                document_text=text,
                chunks=chunks,
            )
            chunks = contextualized.texts
            marker = contextualized.marker
        embeddings = await asyncio.to_thread(
            self._knowledge.embeddings.embed_documents,
            chunks,
            model=embedding_model,
        )
        return chunks, embeddings, source_chunks, marker

    async def reembed_document(
        self,
        *,
        document: KnowledgeDocument,
        embedding_model: str,
        authority_check: Callable[[], None] | None = None,
        actor_user_id: uuid.UUID | None = None,
    ) -> KnowledgeDocument:
        """Re-chunk and re-embed one document's text, replacing its vectors.

        The per-document unit of work behind a background reindex run:
        the document's stored text is re-run through the same chunk and
        embedding pipeline as first-time ingestion and its vectors are
        replaced in place (id, title, text, metadata, and creation time
        preserved — see
        :meth:`~inqtrix.knowledge.stores.ports.KnowledgeStore.reembed_document`).
        Access is the job's concern (checked once at submission); this
        method is trusted server code.

        Raises:
            inqtrix.knowledge.stores.ports.DocumentNotFound: The
                document vanished between enumeration and re-embed.
            inqtrix.providers.embeddings.EmbeddingProviderError: The
                embedding backend failed — surfaced, never swallowed
                into a partially re-embedded document.
        """
        if authority_check is not None:
            authority_check()
        chunks, embeddings, source_chunks, _marker = await self._embed_text(
            title=document.title,
            text=document.text,
            embedding_model=embedding_model,
        )
        if authority_check is not None:
            authority_check()
        # Re-embed re-chunks the unchanged canonical text with the deterministic
        # chunker, so the chunk set is identical and page numbers captured at
        # first ingest re-align by index — carry them forward (no original file
        # to re-map against here). Truncate/pad defensively to the chunk count.
        stored_pages = document.metadata.get("_chunk_pages")
        page_numbers: list[int | None] | None = None
        if isinstance(stored_pages, list):
            if len(stored_pages) == len(chunks):
                page_numbers = list(stored_pages)
            else:
                # The chunk set changed (e.g. chunk_max_chars was reconfigured
                # between ingest and reindex), so the stored pages no longer
                # align by index — drop them rather than misattribute pages
                # silently (No Silent Fallbacks). A re-ingest from the original
                # PDF is the only path that re-derives them.
                log.warning(
                    "Re-embed page mapping: stored %d pages but re-chunk "
                    "produced %d chunks for document %s (likely a chunk_max_chars "
                    "change); dropping page numbers",
                    len(stored_pages),
                    len(chunks),
                    document.id,
                )
        updated = await self._knowledge.store.reembed_document(
            document_id=document.id,
            chunks=chunks,
            embeddings=embeddings,
            source_chunks=source_chunks,
            page_numbers=page_numbers,
            actor_user_id=actor_user_id,
        )
        if authority_check is not None:
            authority_check()
        return updated

    async def add_document_from_file(
        self,
        *,
        collection_id: str,
        file_name: str,
        content: bytes,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
        visible_to: "UserContext | None" = None,
    ) -> KnowledgeDocument:
        """Parse one uploaded file and ingest the resulting text.

        Raises:
            KnowledgeValidationError: When no parser is wired
                (``INQTRIX_DOCUMENT_PARSER=none``) — the caller maps
                this to a clear client error, never a silent skip.
            inqtrix.knowledge.parsing.DocumentParseError: When the file
                cannot be converted or yields no text.
        """
        if self._parser is None:
            raise KnowledgeValidationError(
                "Datei-Ingestion ist deaktiviert "
                "(INQTRIX_DOCUMENT_PARSER=none)"
            )
        text = await asyncio.to_thread(
            self._parser.parse, file_name=file_name, content=content
        )
        document_metadata = dict(metadata or {})
        document_metadata["parser"] = self._parser.parser_id
        # Best-effort per-page text for chunk→page provenance (PDFs only; a
        # non-PDF or a failure yields None and the document ingests without page
        # numbers). Off the event loop — pdfminer is CPU-bound.
        page_texts = await asyncio.to_thread(extract_pdf_page_texts, content)
        return await self.add_document(
            collection_id=collection_id,
            title=(title or "").strip() or file_name,
            text=text,
            metadata=document_metadata,
            visible_to=visible_to,
            page_texts=page_texts,
        )

    async def list_documents(
        self,
        collection_id: str,
        *,
        visible_to: "UserContext | None" = None,
    ) -> list[KnowledgeDocument]:
        """A collection's documents, newest first (view via parent)."""
        collection = await self._knowledge.store.get_collection(collection_id)
        await self.collection_access(collection, visible_to)
        return await self._knowledge.store.list_documents(collection_id)

    async def list_documents_page(
        self,
        collection_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
        visible_to: "UserContext | None" = None,
    ) -> tuple[list[KnowledgeDocument], str | None]:
        """One keyset page of a collection's documents (view via parent).

        Access is the single parent-collection check, so the store's
        DB-side LIMIT bounds the work without under-filling the page."""
        collection = await self._knowledge.store.get_collection(collection_id)
        await self.collection_access(collection, visible_to)
        return await self._knowledge.store.list_documents_page(
            collection_id, limit=limit, after=after
        )

    async def get_document(
        self,
        document_id: str,
        *,
        visible_to: "UserContext | None" = None,
    ) -> KnowledgeDocument:
        """One document including its full text (the citable source view).

        Visibility is the parent collection's (view suffices); the
        denial is :class:`DocumentNotFound` so a hidden document and a
        missing one stay byte-identical.
        """
        document = await self._knowledge.store.get_document(document_id)
        await self._document_parent_access(document, visible_to)
        return document

    async def get_chunk(
        self,
        document_id: str,
        chunk_index: int,
        *,
        context: int = 0,
        visible_to: "UserContext | None" = None,
    ) -> tuple[DocumentChunk, list[DocumentChunk]]:
        """One chunk plus its neighbour chunks (the citable evidence view).

        Chunk identity is ``(document_id, chunk_index)`` — stable across
        reindex, unlike the physical chunk id. *context* widens the read
        so a cited quote can be shown in its document surroundings.
        Visibility is the parent collection's, exactly like
        :meth:`get_document` (view suffices; denial stays the indistinct
        :class:`DocumentNotFound`).

        Args:
            document_id: The owning document.
            chunk_index: Zero-based chunk position within the document.
            context: Neighbour chunks to include on EACH side of the
                target (``0`` — the default — returns none). Range
                policy (0..3) is the router's concern; the service
                trusts its caller.
            visible_to: Caller scope for the parent-collection check.

        Returns:
            ``(chunk, neighbors)`` — *neighbors* are the up to
            ``2 * context`` surrounding chunks in ``chunk_index`` order,
            target excluded.

        Raises:
            DocumentNotFound: Unknown or invisible document.
            ChunkNotFound: The document is visible but has no chunk at
                *chunk_index*.
        """
        document = await self._knowledge.store.get_document(document_id)
        await self._document_parent_access(document, visible_to)
        chunks = await self._knowledge.store.get_chunks(document_id)
        target = next(
            (chunk for chunk in chunks if chunk.chunk_index == chunk_index),
            None,
        )
        if target is None:
            raise ChunkNotFound(f"{document_id}#{chunk_index}")
        neighbors = [
            chunk
            for chunk in chunks
            if chunk.chunk_index != chunk_index
            and abs(chunk.chunk_index - chunk_index) <= context
        ]
        return target, neighbors

    async def delete_document(
        self,
        document_id: str,
        *,
        visible_to: "UserContext | None" = None,
    ) -> None:
        """Delete one document and its chunks (edit via parent)."""
        document = await self._knowledge.store.get_document(document_id)
        await self._document_parent_access(
            document, visible_to, minimum=SharePermission.EDIT
        )
        await self._run_collection_mutation(
            document.collection_id,
            lambda: self._knowledge.store.delete_document(
                document_id,
                actor_user_id=self._actor_user_id(visible_to),
            ),
        )
        collection = await self._knowledge.store.get_collection(
            document.collection_id
        )
        await self._invalidate_collection(collection)

    async def _invalidate_collection(
        self, collection: KnowledgeCollection
    ) -> None:
        """Publish fallback effects only for volatile knowledge stores."""
        if self._invalidator is None or getattr(
            self._knowledge.store, "atomic_resource_effects", False
        ):
            return
        await self._invalidator.invalidate(
            tenant_id=collection.tenant_id,
            owner_user_id=collection.created_by_user_id,
            resource_type="knowledge_collection",
            resource_id=collection.id,
            scope="knowledge_collections",
        )

    async def _document_parent_access(
        self,
        document: KnowledgeDocument,
        visible_to: "UserContext | None",
        *,
        minimum: SharePermission = SharePermission.VIEW,
    ) -> ResourceAccess:
        """The caller's grant on a document's parent collection.

        Raises:
            DocumentNotFound: No access — re-raised under the
                document's identity, never the collection's.
        """
        try:
            collection = await self._knowledge.store.get_collection(
                document.collection_id
            )
            return await self.collection_access(
                collection, visible_to, minimum=minimum
            )
        except CollectionNotFound:
            raise DocumentNotFound(document.id) from None

    # -- retrieval ----------------------------------------------------------- #

    async def search(
        self,
        *,
        query: str,
        collection_ids: list[str] | None = None,
        top_k: int | None = None,
        visible_to: "UserContext | None" = None,
    ) -> list[RetrievalCandidate]:
        """Embed *query* and return scored chunk candidates.

        Backward-compatible shape (``list[RetrievalCandidate]``);
        callers that need the applied-visibility report use
        :meth:`search_reported`.
        """
        outcome = await self.search_reported(
            query=query,
            collection_ids=collection_ids,
            top_k=top_k,
            visible_to=visible_to,
        )
        return outcome.candidates

    async def search_reported(
        self,
        *,
        query: str,
        collection_ids: list[str] | None = None,
        top_k: int | None = None,
        visible_to: "UserContext | None" = None,
    ) -> SearchOutcome:
        """Embed *query* and return scored candidates plus dropped ids.

        Kept synchronous and side-effect-free on purpose: this is the
        debugging/evaluation surface for retrieval quality, independent
        of answer synthesis.

        Scoped callers search only what they may see: an explicit
        ``collection_ids`` list is reduced to its visible members
        (none visible raises :class:`CollectionNotFound` — the 404),
        and an unscoped search ranges over the visible set instead of
        everything. Dropped-but-requested ids are reported so the
        filtering never stays silent.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            raise KnowledgeValidationError("Feld 'query' ist erforderlich")
        effective_ids = collection_ids
        filtered_ids: list[str] = []
        visible_ids = {
            collection.id
            for collection in await self.list_collections(visible_to=visible_to)
        }
        if collection_ids is not None:
            effective_ids = [item for item in collection_ids if item in visible_ids]
            filtered_ids = [item for item in collection_ids if item not in visible_ids]
            if not effective_ids:
                raise CollectionNotFound(
                    collection_ids[0] if collection_ids else ""
                )
        else:
            if not visible_ids:
                return SearchOutcome(candidates=[])
            effective_ids = sorted(visible_ids)
        # Deliberately no on_provider_retry: this debug/eval surface has no
        # event stream; rerank retries stay visible via the provider log.
        candidates = await retrieve(
            self._knowledge,
            query=clean_query,
            collection_ids=effective_ids,
            top_k=top_k or self._knowledge.default_top_k,
        )
        return SearchOutcome(
            candidates=candidates, filtered_collection_ids=filtered_ids
        )

    async def assert_collections_visible(
        self,
        collection_ids: list[str],
        *,
        visible_to: "UserContext | None" = None,
    ) -> None:
        """Admission gate for the ask paths (chat + native runs).

        Strict on purpose, unlike the search filter: an ask against an
        explicit collection set must not silently answer from fewer
        collections than the caller picked — a single invisible id
        denies the whole request with the indistinct
        :class:`CollectionNotFound`. Admission persists only this bounded
        dependency set; the execution dependency authorizer then repeats the
        same live check at run safepoints so a later revoke cannot inherit the
        admission decision.
        """
        for collection_id in collection_ids:
            collection = await self._knowledge.store.get_collection(collection_id)
            await self.collection_access(collection, visible_to)

    async def resolve_ask_scope(
        self,
        collection_ids: "list[str] | None",
        *,
        visible_to: "UserContext | None" = None,
    ) -> list[str] | None:
        """Pin an ask's retrieval scope to what the caller may see.

        The admission-time counterpart of
        :meth:`assert_collections_visible`: it normalizes the scope once at
        submit and returns the concrete id list to persist into the run's
        ``knowledge_filters``, so the worker re-executes an already bounded
        request. Authorization is not cached: the worker rechecks every pinned
        id at execution safepoints. This closes the unscoped-ask gap — an omitted,
        empty, or ``null`` filter otherwise reaches the ``mode=knowledge``
        algorithm as ``None``, and the shared retrieval pipeline
        (:mod:`inqtrix.knowledge.retrieval`) then searches EVERY
        collection in the tenant (only the agent/capability paths carry
        ``visible_to`` on their own).

        Rules:

        * an explicit, non-empty list is asserted strictly (one invisible
          id raises :class:`CollectionNotFound`) and returned unchanged —
          the caller pinned the scope on purpose;
        * an omitted / ``null`` / ``[]`` / non-list scope with a resolved
          ``visible_to`` expands to the caller-visible set (owned +
          shared-in + legacy), matching :meth:`search_reported`'s
          unscoped expansion and the agent harness's falsy-scope rule —
          an ask must answer from the caller's corpus, never 404 on a
          merely-empty selection. An empty visible set returns ``[]`` —
          fail-closed, because the stores read ``[]`` as "nothing" while
          only ``None`` means "everything";
        * ``visible_to is None`` (``AUTH_MODE`` none / static apikey)
          keeps the historical see-everything view and returns ``None``.

        The expansion honours the stores' one-model-per-query invariant:
        an EXPLICIT multi-model scope is a hard ``KnowledgeError`` there,
        while the pre-pin ``None`` scope silently narrowed to the default
        embedding model's collections. A visible set spanning several
        embedding models therefore pins the default model's subset (the
        exact pre-pin coverage), logged loudly; a single-model visible
        set pins completely regardless of which model that is.

        Args:
            collection_ids: The request's raw ``collection_ids`` filter —
                a list, an empty list, or ``None``. A non-list value is
                treated as unscoped (fail-closed to the visible set).
            visible_to: The caller's resolved visibility, or ``None`` for
                the unauthenticated see-everything modes.

        Returns:
            The concrete scope to persist: the asserted explicit list, the
            caller-visible id set (possibly empty), or ``None`` for the
            see-everything modes.

        Raises:
            CollectionNotFound: An explicit id the caller cannot see.
        """
        explicit = (
            [str(item) for item in collection_ids]
            if isinstance(collection_ids, list)
            else []
        )
        if explicit:
            await self.assert_collections_visible(
                explicit, visible_to=visible_to
            )
            return explicit
        visible = await self.list_collections(visible_to=visible_to)
        models = {collection.embedding_model for collection in visible}
        if len(models) <= 1:
            return [collection.id for collection in visible]
        default_model = self._knowledge.embeddings.default_model
        pinned = [
            collection.id
            for collection in visible
            if collection.embedding_model == default_model
        ]
        log.warning(
            "resolve_ask_scope: visible collections span %d embedding "
            "models; pinning the %d default-model (%s) collection(s) of "
            "%d visible — scoped asks reach the others.",
            len(models),
            len(pinned),
            default_model,
            len(visible),
        )
        return pinned
