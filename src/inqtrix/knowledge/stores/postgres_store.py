"""Postgres-canonical knowledge store (the standard RAG data split).

Collections, documents (full canonical text), and chunk metadata live
relationally in Postgres — the source of truth, queryable and joinable
for ownership/sharing and pagination. The embedding VECTORS live in a
:class:`~inqtrix.knowledge.stores.vector_index.VectorIndex` (Qdrant in
production, in-process for dev/test); original binaries live in the
object store. This store owns the canonical half and delegates vectors
to the index, hydrating retrieval hits back from Postgres.

Sync is flag-based, not an outbox: document mutations hold the collection
row lock while the vector side effect runs, then commit the canonical rows
with ``vector_synced=true``. Reindex submit takes that same lock and therefore
cannot snapshot a half-finished mutation. A vector failure propagates and
rolls back the canonical transaction. A partial first insert is an orphan the
existing reconcile removes; a failed in-place replacement remains a visible
failed reindex and is repaired by rerunning that same operation. There is no
vector duplication in Postgres and no separate sync worker.

Every operation runs inside :func:`~inqtrix.storage.db.tenant_session`
(restricted role + transaction-local tenant GUC), with explicit tenant
predicates as layer 1 and row-level security as layer 2 — identical to
the other Postgres repositories. The engine is its own NullPool engine
(loop-agnostic): this store is awaited from the HTTP loop AND bridged
from the synchronous research graph / reindex worker via
``run_coro_sync`` (a fresh per-call loop), which a pooled asyncpg
connection could not survive.
"""

from __future__ import annotations

import logging
import time
import uuid

from sqlalchemy import delete, func, insert, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from inqtrix.auth.permissions import ResourceAccess, SharePermission
from inqtrix.pagination import encode_cursor
from inqtrix.knowledge.stores.ports import (
    CollectionMaintenanceActive,
    CollectionNotFound,
    DocumentChunk,
    DocumentNotFound,
    EmbeddingDimensionMismatch,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeError,
    RetrievalCandidate,
)

log = logging.getLogger("inqtrix")
from inqtrix.knowledge.stores.vector_index import ChunkVector, VectorIndex
from inqtrix.storage.db import build_session_factory, tenant_session
from inqtrix.storage.knowledge_orm import (
    knowledge_chunks,
    knowledge_collections,
    knowledge_documents,
)
from inqtrix.server.indexing import ACTIVE_INDEXING_STATUS_VALUES
from inqtrix.storage.indexing_orm import indexing_jobs
from inqtrix.storage.resource_access import (
    VISIBLE_SHARE_PERMISSION,
    append_resource_effects,
    listed_resource_access,
    lock_active_users,
    lock_resource_access,
    revoke_resource_shares,
    visible_resource_select,
)

_DEFAULT_TENANT = "default"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


class PostgresKnowledgeStore:
    """Canonical knowledge store over Postgres + a vector index.

    Args:
        engine: A dedicated NullPool async engine (loop-agnostic) for the
            knowledge schema — never the shared HTTP-loop engine.
        app_role: Restricted Postgres role for the tenant sessions.
        vector_index: Where chunk vectors live (Qdrant or in-process).
    """

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        app_role: str,
        vector_index: VectorIndex,
        restrict_to_workspace_members: bool = False,
    ) -> None:
        self._engine = engine
        self._session_factory = build_session_factory(engine)
        self._app_role = app_role
        self._vectors = vector_index
        self._restrict_to_workspace_members = restrict_to_workspace_members

    @property
    def atomic_resource_effects(self) -> bool:
        """Audit, invalidation, and share cleanup join canonical writes."""
        return True

    @property
    def supports_safe_reindex(self) -> bool:
        """Collection-row locking serializes jobs against canonical writes."""
        return True

    @property
    def supports_collection_sharing(self) -> bool:
        """Canonical collection and share rows share Postgres transactions."""
        return True

    @property
    def supports_hybrid(self) -> bool:
        """Delegated to the vector index (hybrid dispatch flag)."""
        return self._vectors.supports_hybrid

    @property
    def sparse_language(self) -> str | None:
        """Delegated BM25 tokenizer language of the vector index.

        ``None`` when the index has no lexical branch (e.g. the in-process
        :class:`MemoryVectorIndex`, which carries no such property). Surfaced
        so the canonical Postgres+Qdrant path also reports the monolingual
        sparse limitation to the capability manifest and the algorithm.
        """
        return getattr(self._vectors, "sparse_language", None)

    async def is_available(self) -> bool:
        """Return whether the delegated vector index is reachable now."""
        return await self._vectors.is_available()

    def _session(self):
        return tenant_session(
            self._session_factory,
            tenant_id=_DEFAULT_TENANT,
            app_role=self._app_role,
        )

    async def aclose(self) -> None:
        """Dispose the dedicated engine at application shutdown."""
        await self._engine.dispose()

    # -- maintenance ------------------------------------------------------ #

    async def reconcile_orphans(self) -> dict[str, object]:
        """Delete vector groups whose canonical Postgres rows are gone.

        The reverse of the forward ``vector_synced`` repair (see the module
        docstring): a cross-store DELETE is non-atomic — Postgres commits, then
        the vector-index delete runs — so an interruption can strand vectors
        with no canonical document. This sweep scrolls each model's vector
        space, left-anti-joins the canonical ``knowledge_documents`` ids, and
        deletes any group whose document no longer exists, logging each removal
        (No Silent Fallbacks). Idempotent; safe to run repeatedly. It only ever
        UNDER-deletes (never removes a live document's vectors): a document id
        present under any collection is excluded from the orphan set.

        Boundary: ``models`` is derived from existing ``knowledge_collections``,
        so a physical chunk collection for a model with NO remaining collection
        is not scrolled — its vectors persist until a collection of that model
        exists again. Acceptable (it never causes over-deletion); revisit if
        models are routinely fully removed.

        Returns:
            ``{"deleted_documents": int, "details": [{document_id, embedding_model}]}``.
        """
        async with self._session() as session:
            model_rows = (
                await session.execute(
                    select(knowledge_collections.c.embedding_model).distinct()
                )
            ).all()
            doc_rows = (
                await session.execute(select(knowledge_documents.c.id))
            ).all()
        models = {row[0] for row in model_rows}
        # SAFETY: this store is hardwired to a single tenant (_DEFAULT_TENANT),
        # so `pg_doc_ids` is the COMPLETE canonical document universe and the
        # Qdrant payload (collection_id/document_id, no tenant) needs no tenant
        # filter. If knowledge ever becomes multi-tenant, this anti-join MUST
        # gain a tenant predicate on the scrolled groups — otherwise another
        # tenant's vectors (read here as absent from `default`'s rows) would be
        # deleted as false orphans.
        pg_doc_ids = {row[0] for row in doc_rows}

        deleted: list[dict[str, str]] = []
        for model in models:
            groups = await self._vectors.scroll_chunk_groups(embedding_model=model)
            orphan_doc_ids = {
                document_id
                for (_collection_id, document_id) in groups
                if document_id not in pg_doc_ids
            }
            for document_id in orphan_doc_ids:
                await self._vectors.delete_document(
                    embedding_model=model, document_id=document_id
                )
                log.warning(
                    "Knowledge-Reconcile: verwaiste Vektoren entfernt "
                    "(document_id=%s, embedding_model=%s) — keine kanonische "
                    "Postgres-Zeile.",
                    document_id,
                    model,
                )
                deleted.append(
                    {"document_id": document_id, "embedding_model": model}
                )
        return {"deleted_documents": len(deleted), "details": deleted}

    # -- collections ------------------------------------------------------ #

    async def create_collection(
        self,
        *,
        name: str,
        embedding_model: str,
        embedding_dim: int,
        created_by_user_id: uuid.UUID | None = None,
    ) -> KnowledgeCollection:
        """Create a collection with its immutable embedding identity."""
        if embedding_dim <= 0:
            raise EmbeddingDimensionMismatch(
                f"embedding_dim must be positive, got {embedding_dim}"
            )
        await self._vectors.ensure_model(
            embedding_model=embedding_model, embedding_dim=embedding_dim
        )
        collection = KnowledgeCollection(
            id=_new_id("kc"),
            name=name,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            created_at=time.time(),
            document_count=0,
            tenant_id=_DEFAULT_TENANT,
            created_by_user_id=created_by_user_id,
        )
        async with self._session() as session:
            if (
                created_by_user_id is not None
                and not await lock_active_users(
                    session,
                    tenant_id=_DEFAULT_TENANT,
                    user_ids=(created_by_user_id,),
                )
            ):
                raise CollectionNotFound(collection.id)
            await session.execute(
                insert(knowledge_collections).values(
                    id=collection.id,
                    tenant_id=collection.tenant_id,
                    name=collection.name,
                    embedding_model=collection.embedding_model,
                    embedding_dim=collection.embedding_dim,
                    created_by_user_id=collection.created_by_user_id,
                    created_at=collection.created_at,
                )
            )
            await append_resource_effects(
                session,
                tenant_id=_DEFAULT_TENANT,
                actor_user_id=created_by_user_id,
                owner_user_id=created_by_user_id,
                action="knowledge_collection.created",
                resource_type="knowledge_collection",
                resource_id=collection.id,
                scope="knowledge_collections",
            )
        return collection

    async def list_collections(self) -> list[KnowledgeCollection]:
        """All collections, newest first (with live document counts)."""
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(knowledge_collections)
                    .where(knowledge_collections.c.tenant_id == _DEFAULT_TENANT)
                    .order_by(knowledge_collections.c.created_at.desc())
                )
            ).all()
            counts = await self._document_counts(session)
        return [self._collection_from_row(row, counts) for row in rows]

    async def list_visible_collections(
        self, *, actor_user_id: uuid.UUID | None
    ) -> list[tuple[KnowledgeCollection, ResourceAccess]]:
        """List owned and accepted-shared collections in one live SQL query."""
        statement = visible_resource_select(
            resource_table=knowledge_collections,
            id_column=knowledge_collections.c.id,
            owner_column=knowledge_collections.c.created_by_user_id,
            resource_type="knowledge_collection",
            tenant_id=_DEFAULT_TENANT,
            actor_user_id=actor_user_id,
            restrict_to_workspace_members=self._restrict_to_workspace_members,
        ).order_by(knowledge_collections.c.created_at.desc())
        async with self._session() as session:
            rows = (await session.execute(statement)).all()
            counts = await self._document_counts(session)
        return [
            (
                self._collection_from_row(row, counts),
                listed_resource_access(
                    owner_user_id=row.created_by_user_id,
                    actor_user_id=actor_user_id,
                    share_permission=getattr(row, VISIBLE_SHARE_PERMISSION),
                ),
            )
            for row in rows
        ]

    async def get_collection(self, collection_id: str) -> KnowledgeCollection:
        """One collection or :class:`CollectionNotFound`."""
        async with self._session() as session:
            row = await self._collection_row(session, collection_id)
            counts = await self._document_counts(session, collection_id)
        return self._collection_from_row(row, counts)

    async def delete_collection(
        self,
        collection_id: str,
        *,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        """Delete a collection with all documents/chunks (DB cascade + vectors)."""
        async with self._session() as session:
            row, access = await self._mutable_collection_row(
                session,
                collection_id,
                actor_user_id=actor_user_id,
                owner_only=True,
            )
            embedding_model = row.embedding_model
            recipients = await revoke_resource_shares(
                session,
                tenant_id=_DEFAULT_TENANT,
                resource_type="knowledge_collection",
                resource_id=collection_id,
                revoked_by_user_id=actor_user_id,
            )
            await session.execute(
                delete(knowledge_collections).where(
                    knowledge_collections.c.tenant_id == _DEFAULT_TENANT,
                    knowledge_collections.c.id == collection_id,
                )
            )
            await append_resource_effects(
                session,
                tenant_id=_DEFAULT_TENANT,
                actor_user_id=actor_user_id,
                owner_user_id=access.owner_user_id,
                action="knowledge_collection.deleted",
                resource_type="knowledge_collection",
                resource_id=collection_id,
                scope="knowledge_collections",
                additional_targets=recipients,
            )
            # Keep the collection row lock until the external side effect
            # finishes. Reindex submit takes the same row lock, so it cannot
            # snapshot a half-finished delete and later resurrect vectors.
            await self._vectors.delete_collection(
                embedding_model=embedding_model,
                collection_id=collection_id,
            )

    # -- documents --------------------------------------------------------- #

    async def add_document(
        self,
        *,
        collection_id: str,
        title: str,
        text: str,
        metadata: dict,
        chunks: list[str],
        embeddings: list[list[float]],
        source_chunks: list[str] | None = None,
        page_numbers: list[int | None] | None = None,
        actor_user_id: uuid.UUID | None = None,
    ) -> KnowledgeDocument:
        """Store a document's canonical rows, then sync its vectors."""
        async with self._session() as session:
            collection, access = await self._mutable_collection_row(
                session,
                collection_id,
                actor_user_id=actor_user_id,
            )
            self._validate_embeddings(chunks, embeddings, collection.embedding_dim)
            document_id = _new_id("kd")
            created_at = time.time()
            chunk_rows = self._build_chunk_rows(
                document_id=document_id,
                collection_id=collection_id,
                chunks=chunks,
                source_chunks=source_chunks,
                page_numbers=page_numbers,
                created_at=created_at,
            )
            await session.execute(
                insert(knowledge_documents).values(
                    id=document_id,
                    collection_id=collection_id,
                    tenant_id=_DEFAULT_TENANT,
                    title=title,
                    text=text,
                    metadata=dict(metadata),
                    chunk_count=len(chunks),
                    vector_synced=False,
                    created_at=created_at,
                )
            )
            if chunk_rows:
                await session.execute(insert(knowledge_chunks), chunk_rows)
            await append_resource_effects(
                session,
                tenant_id=_DEFAULT_TENANT,
                actor_user_id=actor_user_id,
                owner_user_id=access.owner_user_id,
                action="knowledge_document.created",
                resource_type="knowledge_collection",
                resource_id=collection_id,
                scope="knowledge_collections",
            )
            embedding_model = collection.embedding_model
            await self._sync_vectors(
                session=session,
                embedding_model=embedding_model,
                collection_id=collection_id,
                document_id=document_id,
                chunk_rows=chunk_rows,
                embeddings=embeddings,
            )
        return KnowledgeDocument(
            id=document_id,
            collection_id=collection_id,
            title=title,
            text=text,
            metadata=dict(metadata),
            chunk_count=len(chunks),
            created_at=created_at,
        )

    async def list_documents(self, collection_id: str) -> list[KnowledgeDocument]:
        """A collection's documents, newest first."""
        async with self._session() as session:
            await self._collection_row(session, collection_id)
            rows = (
                await session.execute(
                    select(knowledge_documents)
                    .where(
                        knowledge_documents.c.tenant_id == _DEFAULT_TENANT,
                        knowledge_documents.c.collection_id == collection_id,
                    )
                    .order_by(knowledge_documents.c.created_at.desc())
                )
            ).all()
        return [self._document_from_row(row) for row in rows]

    async def list_documents_page(
        self,
        collection_id: str,
        *,
        limit: int,
        after: tuple[float, str] | None,
    ) -> tuple[list[KnowledgeDocument], str | None]:
        """One keyset page (newest first); the DB does the LIMIT.

        Visibility is the single parent-collection check (``_collection_row``)
        — all of a readable collection's documents are visible — so the
        DB-side ``LIMIT`` never under-fills a page.
        """
        async with self._session() as session:
            await self._collection_row(session, collection_id)
            query = select(knowledge_documents).where(
                knowledge_documents.c.tenant_id == _DEFAULT_TENANT,
                knowledge_documents.c.collection_id == collection_id,
            )
            if after is not None:
                query = query.where(
                    tuple_(
                        knowledge_documents.c.created_at,
                        knowledge_documents.c.id,
                    )
                    < tuple_(after[0], after[1])
                )
            query = query.order_by(
                knowledge_documents.c.created_at.desc(),
                knowledge_documents.c.id.desc(),
            ).limit(limit + 1)
            rows = (await session.execute(query)).all()
        documents = [self._document_from_row(row) for row in rows[:limit]]
        next_cursor = (
            encode_cursor(documents[-1].created_at, documents[-1].id)
            if len(rows) > limit and documents
            else None
        )
        return documents, next_cursor

    async def get_document(self, document_id: str) -> KnowledgeDocument:
        """One document (full text) or :class:`DocumentNotFound`."""
        async with self._session() as session:
            row = await self._document_row(session, document_id)
        return self._document_from_row(row)

    async def get_chunks(self, document_id: str) -> list[DocumentChunk]:
        """One document's chunks ordered by ``chunk_index`` (no vectors)."""
        async with self._session() as session:
            await self._document_row(session, document_id)  # 404 if unknown
            rows = (
                await session.execute(
                    select(
                        knowledge_chunks.c.id,
                        knowledge_chunks.c.document_id,
                        knowledge_chunks.c.collection_id,
                        knowledge_chunks.c.chunk_index,
                        knowledge_chunks.c.text,
                        knowledge_chunks.c.source_text,
                        knowledge_chunks.c.page_number,
                    )
                    .where(
                        knowledge_chunks.c.tenant_id == _DEFAULT_TENANT,
                        knowledge_chunks.c.document_id == document_id,
                    )
                    .order_by(knowledge_chunks.c.chunk_index)
                )
            ).all()
        return [
            DocumentChunk(
                id=row[0],
                document_id=row[1],
                collection_id=row[2],
                chunk_index=row[3],
                text=row[4],
                source_text=row[5],
                page_number=row[6],
            )
            for row in rows
        ]

    async def delete_document(
        self,
        document_id: str,
        *,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        """Delete one document and its chunks (DB cascade + vectors)."""
        async with self._session() as session:
            preliminary = await self._document_row(session, document_id)
            collection, access = await self._mutable_collection_row(
                session,
                preliminary.collection_id,
                actor_user_id=actor_user_id,
            )
            row = await self._document_row(
                session, document_id, for_update=True
            )
            if row.collection_id != preliminary.collection_id:
                raise DocumentNotFound(document_id)
            embedding_model = collection.embedding_model
            result = await session.execute(
                delete(knowledge_documents).where(
                    knowledge_documents.c.tenant_id == _DEFAULT_TENANT,
                    knowledge_documents.c.id == document_id,
                )
            )
            if not result.rowcount:
                raise DocumentNotFound(document_id)
            await append_resource_effects(
                session,
                tenant_id=_DEFAULT_TENANT,
                actor_user_id=actor_user_id,
                owner_user_id=access.owner_user_id,
                action="knowledge_document.deleted",
                resource_type="knowledge_collection",
                resource_id=row.collection_id,
                scope="knowledge_collections",
            )
            await self._vectors.delete_document(
                embedding_model=embedding_model,
                document_id=document_id,
            )

    async def reembed_document(
        self,
        *,
        document_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_chunks: list[str] | None = None,
        page_numbers: list[int | None] | None = None,
        actor_user_id: uuid.UUID | None = None,
    ) -> KnowledgeDocument:
        """Rebuild one document's chunks/vectors in place (keep its id)."""
        async with self._session() as session:
            preliminary = await self._document_row(session, document_id)
            collection, access = await self._mutable_collection_row(
                session,
                preliminary.collection_id,
                actor_user_id=actor_user_id,
                allow_active_maintenance=True,
            )
            document = await self._document_row(
                session, document_id, for_update=True
            )
            if document.collection_id != preliminary.collection_id:
                raise DocumentNotFound(document_id)
            self._validate_embeddings(chunks, embeddings, collection.embedding_dim)
            created_at = time.time()
            # Reuse chunk ids by position so a re-embed keeps exact
            # provenance links alive across reindex ((document_id,
            # chunk_index) is the citation key, but a stable chunk_id is
            # the durable provenance anchor). Delete-then-insert within
            # one transaction reinserts the reused ids; positions beyond
            # the prior count get fresh ids, a shrunk doc drops the tail.
            prior_ids = [
                row.id
                for row in (
                    await session.execute(
                        select(knowledge_chunks.c.id)
                        .where(
                            knowledge_chunks.c.tenant_id == _DEFAULT_TENANT,
                            knowledge_chunks.c.document_id == document_id,
                        )
                        .order_by(knowledge_chunks.c.chunk_index)
                    )
                ).all()
            ]
            chunk_rows = self._build_chunk_rows(
                document_id=document_id,
                collection_id=document.collection_id,
                chunks=chunks,
                source_chunks=source_chunks,
                page_numbers=page_numbers,
                created_at=created_at,
                reuse_ids=prior_ids,
            )
            await session.execute(
                delete(knowledge_chunks).where(
                    knowledge_chunks.c.tenant_id == _DEFAULT_TENANT,
                    knowledge_chunks.c.document_id == document_id,
                )
            )
            if chunk_rows:
                await session.execute(insert(knowledge_chunks), chunk_rows)
            await session.execute(
                update(knowledge_documents)
                .where(
                    knowledge_documents.c.tenant_id == _DEFAULT_TENANT,
                    knowledge_documents.c.id == document_id,
                )
                .values(chunk_count=len(chunks), vector_synced=False)
            )
            collection_id = document.collection_id
            embedding_model = collection.embedding_model
            updated = KnowledgeDocument(
                id=document.id,
                collection_id=document.collection_id,
                title=document.title,
                text=document.text,
                metadata=dict(document.metadata),
                chunk_count=len(chunks),
                created_at=document.created_at,
            )
            await append_resource_effects(
                session,
                tenant_id=_DEFAULT_TENANT,
                actor_user_id=actor_user_id,
                owner_user_id=access.owner_user_id,
                action="knowledge_document.reembedded",
                resource_type="knowledge_collection",
                resource_id=document.collection_id,
                scope="knowledge_collections",
            )
            # Delete+replace is one collection maintenance write. The row
            # lock remains held across both vector calls, so cancel, revoke,
            # reindex submit and user mutations observe one linear order.
            await self._vectors.delete_document(
                embedding_model=embedding_model,
                document_id=document_id,
            )
            await self._sync_vectors(
                session=session,
                embedding_model=embedding_model,
                collection_id=collection_id,
                document_id=document_id,
                chunk_rows=chunk_rows,
                embeddings=embeddings,
            )
        return updated

    # -- retrieval --------------------------------------------------------- #

    async def search(
        self,
        *,
        query_embedding: list[float],
        collection_ids: list[str] | None,
        top_k: int,
        embedding_model: str | None = None,
    ) -> list[RetrievalCandidate]:
        """Dense retrieval: vector index ranks, Postgres hydrates."""
        model, scope = await self._resolve_scope(embedding_model, collection_ids)
        if not scope:
            return []
        hits = await self._vectors.search(
            embedding_model=model,
            query_embedding=query_embedding,
            collection_ids=scope,
            top_k=top_k,
        )
        return await self._hydrate(hits)

    async def hybrid_search(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        collection_ids: list[str] | None,
        top_k: int,
        embedding_model: str | None = None,
    ) -> list[RetrievalCandidate]:
        """Fused dense + sparse retrieval, Postgres-hydrated."""
        model, scope = await self._resolve_scope(embedding_model, collection_ids)
        if not scope:
            return []
        hits = await self._vectors.hybrid_search(
            embedding_model=model,
            query_text=query_text,
            query_embedding=query_embedding,
            collection_ids=scope,
            top_k=top_k,
        )
        return await self._hydrate(hits)

    # -- internals --------------------------------------------------------- #

    async def _collection_row(
        self, session, collection_id: str, *, for_update: bool = False
    ):
        statement = select(knowledge_collections).where(
            knowledge_collections.c.tenant_id == _DEFAULT_TENANT,
            knowledge_collections.c.id == collection_id,
        )
        if for_update:
            statement = statement.with_for_update()
        row = (await session.execute(statement)).one_or_none()
        if row is None:
            raise CollectionNotFound(collection_id)
        return row

    async def _mutable_collection_row(
        self,
        session,
        collection_id: str,
        *,
        actor_user_id: uuid.UUID | None,
        owner_only: bool = False,
        allow_active_maintenance: bool = False,
    ):
        """Lock a collection and reject active reindex maintenance.

        Reindex submission takes the same short row lock before inserting its
        active job. Therefore either the mutation commits first and belongs to
        the reindex snapshot, or the job row lands first and this mutation gets
        the visible ``collection_maintenance`` conflict.
        """
        access = await lock_resource_access(
            session,
            tenant_id=_DEFAULT_TENANT,
            actor_user_id=actor_user_id,
            resource_type="knowledge_collection",
            resource_table=knowledge_collections,
            id_column=knowledge_collections.c.id,
            resource_id=collection_id,
            owner_column=knowledge_collections.c.created_by_user_id,
            minimum=(
                SharePermission.VIEW if owner_only else SharePermission.EDIT
            ),
            restrict_to_workspace_members=(
                self._restrict_to_workspace_members
            ),
            owner_only=owner_only,
        )
        if access is None:
            raise CollectionNotFound(collection_id)
        row = await self._collection_row(session, collection_id)
        if allow_active_maintenance:
            return row, access
        active_job_id = await session.scalar(
            select(indexing_jobs.c.job_id)
            .where(
                indexing_jobs.c.collection_id == collection_id,
                indexing_jobs.c.status.in_(ACTIVE_INDEXING_STATUS_VALUES),
            )
            .limit(1)
        )
        if active_job_id is not None:
            raise CollectionMaintenanceActive(collection_id)
        return row, access

    async def _document_row(
        self,
        session,
        document_id: str,
        *,
        for_update: bool = False,
    ):
        statement = select(knowledge_documents).where(
            knowledge_documents.c.tenant_id == _DEFAULT_TENANT,
            knowledge_documents.c.id == document_id,
        )
        if for_update:
            statement = statement.with_for_update()
        row = (await session.execute(statement)).one_or_none()
        if row is None:
            raise DocumentNotFound(document_id)
        return row

    async def _document_counts(
        self, session, collection_id: str | None = None
    ) -> dict[str, int]:
        statement = select(
            knowledge_documents.c.collection_id,
            func.count(knowledge_documents.c.id),
        ).where(knowledge_documents.c.tenant_id == _DEFAULT_TENANT)
        if collection_id is not None:
            statement = statement.where(
                knowledge_documents.c.collection_id == collection_id
            )
        statement = statement.group_by(knowledge_documents.c.collection_id)
        rows = (await session.execute(statement)).all()
        return {row[0]: int(row[1]) for row in rows}

    def _collection_from_row(self, row, counts: dict[str, int]) -> KnowledgeCollection:
        return KnowledgeCollection(
            id=row.id,
            name=row.name,
            embedding_model=row.embedding_model,
            embedding_dim=row.embedding_dim,
            created_at=row.created_at,
            document_count=counts.get(row.id, 0),
            tenant_id=row.tenant_id,
            created_by_user_id=row.created_by_user_id,
        )

    def _document_from_row(self, row) -> KnowledgeDocument:
        return KnowledgeDocument(
            id=row.id,
            collection_id=row.collection_id,
            title=row.title,
            text=row.text,
            metadata=dict(row.metadata or {}),
            chunk_count=row.chunk_count,
            created_at=row.created_at,
        )

    @staticmethod
    def _validate_embeddings(
        chunks: list[str], embeddings: list[list[float]], embedding_dim: int
    ) -> None:
        if len(chunks) != len(embeddings):
            raise EmbeddingDimensionMismatch(
                f"chunk/embedding count mismatch: {len(chunks)} chunks vs "
                f"{len(embeddings)} embeddings"
            )
        for index, embedding in enumerate(embeddings):
            if len(embedding) != embedding_dim:
                raise EmbeddingDimensionMismatch(
                    f"chunk {index} has dimension {len(embedding)}, "
                    f"collection requires {embedding_dim}"
                )

    @staticmethod
    def _build_chunk_rows(
        *,
        document_id: str,
        collection_id: str,
        chunks: list[str],
        source_chunks: list[str] | None,
        created_at: float,
        page_numbers: list[int | None] | None = None,
        reuse_ids: list[str] | None = None,
    ) -> list[dict]:
        sources = source_chunks or []
        pages = page_numbers or []
        prior_ids = reuse_ids or []
        return [
            {
                "id": (
                    prior_ids[index]
                    if index < len(prior_ids)
                    else _new_id("kch")
                ),
                "document_id": document_id,
                "collection_id": collection_id,
                "tenant_id": _DEFAULT_TENANT,
                "chunk_index": index,
                "text": chunk_text,
                "source_text": (
                    sources[index] if index < len(sources) else ""
                ),
                "page_number": pages[index] if index < len(pages) else None,
                "created_at": created_at,
            }
            for index, chunk_text in enumerate(chunks)
        ]

    async def _sync_vectors(
        self,
        *,
        session: AsyncSession,
        embedding_model: str,
        collection_id: str,
        document_id: str,
        chunk_rows: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert vectors and mark the locked canonical row synchronized.

        The caller owns the collection-row transaction. A failure propagates
        and rolls back the canonical mutation; reindex submission cannot pass
        that row lock while the vector side effect is still in flight.
        """
        vectors = [
            ChunkVector(
                chunk_id=row["id"],
                dense=tuple(embedding),
                text=row["text"],
            )
            for row, embedding in zip(chunk_rows, embeddings)
        ]
        await self._vectors.upsert(
            embedding_model=embedding_model,
            collection_id=collection_id,
            document_id=document_id,
            vectors=vectors,
        )
        await session.execute(
            update(knowledge_documents)
            .where(
                knowledge_documents.c.tenant_id == _DEFAULT_TENANT,
                knowledge_documents.c.id == document_id,
            )
            .values(vector_synced=True)
        )

    async def _resolve_scope(
        self, embedding_model: str | None, collection_ids: list[str] | None
    ) -> tuple[str, list[str]]:
        """Resolve the (model, concrete-collection-id-list) search scope.

        The vector index is per-model and needs concrete ids; an unscoped
        search (``collection_ids is None``) expands to every collection
        of the resolved model.
        """
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(
                        knowledge_collections.c.id,
                        knowledge_collections.c.embedding_model,
                    ).where(knowledge_collections.c.tenant_id == _DEFAULT_TENANT)
                )
            ).all()
        by_id = {row[0]: row[1] for row in rows}
        if collection_ids is not None:
            # Parity with the memory + sole-store contract: an unknown
            # explicit id is CollectionNotFound, a mixed-model scope is a
            # hard error — never a silently narrowed result set.
            for cid in collection_ids:
                if cid not in by_id:
                    raise CollectionNotFound(cid)
            scoped_models = {by_id[cid] for cid in collection_ids}
            if len(scoped_models) > 1:
                raise KnowledgeError(
                    "scoped collections use different embedding models "
                    f"({sorted(scoped_models)}); query one model scope at a time"
                )
            model = embedding_model or (
                by_id[collection_ids[0]] if collection_ids else ""
            )
            return model, list(collection_ids)
        if embedding_model is None:
            return "", []
        scope = [cid for cid, model in by_id.items() if model == embedding_model]
        return embedding_model, scope

    async def _hydrate(self, hits) -> list[RetrievalCandidate]:
        """Join vector hits back to chunk text + document title (in rank order)."""
        if not hits:
            return []
        chunk_ids = [hit.chunk_id for hit in hits]
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(
                        knowledge_chunks.c.id,
                        knowledge_chunks.c.document_id,
                        knowledge_chunks.c.collection_id,
                        knowledge_chunks.c.chunk_index,
                        knowledge_chunks.c.text,
                        knowledge_chunks.c.source_text,
                        knowledge_chunks.c.page_number,
                        knowledge_documents.c.title,
                    )
                    .select_from(
                        knowledge_chunks.join(
                            knowledge_documents,
                            knowledge_chunks.c.document_id
                            == knowledge_documents.c.id,
                        )
                    )
                    .where(
                        knowledge_chunks.c.tenant_id == _DEFAULT_TENANT,
                        knowledge_chunks.c.id.in_(chunk_ids),
                    )
                )
            ).all()
        by_chunk = {row[0]: row for row in rows}
        candidates: list[RetrievalCandidate] = []
        for hit in hits:
            row = by_chunk.get(hit.chunk_id)
            if row is None:
                # Vector index ahead of canonical (mid-reconcile or a chunk
                # deleted between search and hydrate); skip rather than emit
                # a hit we cannot cite — but visibly (No Silent Fallbacks).
                log.warning(
                    "knowledge hydrate: chunk %s ranked by the vector index "
                    "but absent in Postgres (mid-reconcile); skipping",
                    hit.chunk_id,
                )
                continue
            candidates.append(
                RetrievalCandidate(
                    chunk=DocumentChunk(
                        id=row[0],
                        document_id=row[1],
                        collection_id=row[2],
                        chunk_index=row[3],
                        text=row[4],
                        source_text=row[5],
                        page_number=row[6],
                    ),
                    score=hit.score,
                    document_title=row[7],
                )
            )
        return candidates
