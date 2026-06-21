"""Composition root: build the service/algorithm wiring for the HTTP app.

``build_container`` is the single place where providers, strategies,
settings, the algorithm registry, the auth provider, and the
application services are assembled into one :class:`AppContainer`
(Constructor-First: services never read the environment; this builder
hands them everything). Both ``create_app`` (via ``register_routes``)
and tests that wire routers manually go through here, so there is
exactly one wiring truth.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from fastapi import Depends

from inqtrix.auth.api_key import CallableGateAuthProvider
from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import PermissionService
from inqtrix.content.memory import MemoryFileRegistry
from inqtrix.services.file_service import FileService
from inqtrix.auth.principal import (
    AuthProvider,
    NoneAuthProvider,
    Principal,
    UserContext,
)
from inqtrix.core.algorithms import AlgorithmRegistry
from inqtrix.core.context import RuntimeContext
from inqtrix.knowledge.algorithm import KnowledgeAlgorithm
from inqtrix.knowledge.profiles import KnowledgeStageCeiling
from inqtrix.knowledge.stores.memory import MemoryKnowledgeStore
from inqtrix.knowledge.stores.ports import KnowledgeProviderContext
from inqtrix.providers.embeddings import LiteLLMEmbeddings
from inqtrix.research.web_research import DirectLlmAlgorithm, WebResearchAlgorithm
from inqtrix.server.runs import RunStore
from inqtrix.services.agent_context import AgentContextResolver
from inqtrix.services.chat_service import ChatService
from inqtrix.services.health_service import HealthService
from inqtrix.services.knowledge_service import KnowledgeService
from inqtrix.services.run_service import RunService
from inqtrix.settings import Settings

log = logging.getLogger("inqtrix")

if TYPE_CHECKING:
    from fastapi import Request

    from inqtrix.content.ports import FileRegistry
    from inqtrix.providers.base import ProviderContext
    from inqtrix.runs.ports import RunStorePort
    from inqtrix.storage.object_store import ObjectStore
    from inqtrix.strategies import StrategyContext


def build_default_registry() -> AlgorithmRegistry:
    """Register the built-in algorithms in their presentation order.

    Registration order is contract-relevant: the mode-validation error
    message lists ids in this order
    (``"mode muss 'research' oder 'direct_llm' sein"``).
    """
    registry = AlgorithmRegistry()
    registry.register(WebResearchAlgorithm())
    registry.register(DirectLlmAlgorithm())
    return registry


@dataclass(frozen=True)
class PlatformPersistence:
    """Bundle of the persistence collaborators one backend wave shares.

    Attributes:
        permissions: Authorization chokepoint over the identity ports.
        file_registry: File-metadata repository.
        audit: Append-only audit sink (the identity backend in both
            modes — the memory store implements the port too).
        session_factory: Async session factory in postgres mode;
            ``None`` in memory mode. For HTTP-loop consumers ONLY —
            the durable run store deliberately builds its OWN engine
            because asyncpg pools are event-loop-affine.
        workspace_admin: The identity backend instance for the
            workspace bootstrap surface (create/list) — the same
            object that backs the permission ports, never a second
            store.
    """

    permissions: PermissionService
    file_registry: "FileRegistry"
    audit: Any
    session_factory: Any | None
    workspace_admin: Any = None
    prompt_templates: Any = None


def build_platform_persistence_bundle(
    settings: Settings,
) -> PlatformPersistence:
    """Settings bridge for the platform persistence layer.

    ``memory`` (the default) wires the no-infrastructure backends:
    scoped principals start with zero memberships, file metadata is
    process-local. ``postgres`` builds the identity repositories and
    the file registry on ONE shared engine/session factory; a missing
    URL is a contradiction and fails at startup instead of degrading
    silently.
    """
    if settings.storage.backend == "postgres":
        if not settings.storage.database_url.strip():
            raise RuntimeError(
                "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
                "INQTRIX_DATABASE_URL."
            )
        from inqtrix.storage.content_postgres import PostgresFileRegistry
        from inqtrix.storage.db import build_engine, build_session_factory
        from inqtrix.storage.identity_postgres import PostgresIdentityBackend

        session_factory = build_session_factory(
            build_engine(settings.storage.database_url)
        )
        backend = PostgresIdentityBackend(
            session_factory=session_factory,
            app_role=settings.storage.app_role,
        )
        from inqtrix.storage.prompt_templates_postgres import (
            PostgresPromptTemplateRepository,
        )

        return PlatformPersistence(
            permissions=PermissionService(
                members=backend, groups=backend, shares=backend, audit=backend
            ),
            file_registry=PostgresFileRegistry(
                session_factory=session_factory,
                app_role=settings.storage.app_role,
            ),
            audit=backend,
            session_factory=session_factory,
            workspace_admin=backend,
            prompt_templates=PostgresPromptTemplateRepository(
                session_factory=session_factory,
                app_role=settings.storage.app_role,
            ),
        )
    from inqtrix.content.prompt_templates import (
        MemoryPromptTemplateRepository,
    )

    store = MemoryIdentityStore()
    return PlatformPersistence(
        permissions=PermissionService(
            members=store, groups=store, shares=store, audit=store
        ),
        file_registry=MemoryFileRegistry(),
        audit=store,
        session_factory=None,
        workspace_admin=store,
        prompt_templates=MemoryPromptTemplateRepository(),
    )


def build_platform_persistence(
    settings: Settings,
) -> tuple[PermissionService, "FileRegistry"]:
    """Legacy 2-tuple view of :func:`build_platform_persistence_bundle`."""
    bundle = build_platform_persistence_bundle(settings)
    return bundle.permissions, bundle.file_registry


def _require_valid_queue_storage(settings: Settings) -> None:
    """Reject a Valkey queue without the Postgres source of truth.

    Shared by :func:`build_run_store` and :func:`build_indexing_store`:
    both stream backends carry only dispatch messages, so a queue
    without a durable Postgres row is a contradiction that must fail at
    startup instead of degrading silently.
    """
    if settings.queue.backend != "valkey":
        return
    if settings.storage.backend != "postgres":
        raise RuntimeError(
            "INQTRIX_QUEUE_BACKEND=valkey verlangt "
            "INQTRIX_STORAGE_BACKEND=postgres — die Job-Zeile in "
            "Postgres ist die Quelle der Wahrheit, der Stream nur "
            "der Dispatch-Kanal."
        )
    if not settings.queue.valkey_url.strip():
        raise RuntimeError(
            "INQTRIX_QUEUE_BACKEND=valkey verlangt eine gesetzte "
            "INQTRIX_VALKEY_URL."
        )


def build_run_store(settings: Settings) -> "RunStorePort":
    """Settings bridge for the run-store backend (env-coupled surface).

    Memory storage keeps the historical in-process store bit-for-bit.
    ``INQTRIX_STORAGE_BACKEND=postgres`` makes run records, events,
    and results durable (execution stays in-process);
    ``INQTRIX_QUEUE_BACKEND=valkey`` additionally dispatches execution
    to ``inqtrix-worker`` processes. Contradictory combinations fail
    at startup instead of degrading silently.

    The durable store gets its OWN engine and a store-loop audit sink:
    asyncpg pools are event-loop-affine, so the run store (background
    loop) must never share a pool with the identity/file backends
    (HTTP loop) — two pools here are correctness, not waste.
    """
    _require_valid_queue_storage(settings)
    if settings.storage.backend != "postgres":
        return RunStore.from_settings(settings.server)

    import os
    import socket

    from inqtrix.runs.postgres_store import PostgresRunStore

    if not settings.storage.database_url.strip():
        raise RuntimeError(
            "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
            "INQTRIX_DATABASE_URL."
        )
    from inqtrix.storage.db import build_engine, build_session_factory
    from inqtrix.storage.identity_postgres import PostgresIdentityBackend

    engine = build_engine(settings.storage.database_url)
    audit = PostgresIdentityBackend(
        session_factory=build_session_factory(engine),
        app_role=settings.storage.app_role,
    )
    queue = None
    if settings.queue.backend == "valkey":
        from inqtrix.runs.valkey_queue import ValkeyRunQueue

        queue = ValkeyRunQueue(url=settings.queue.valkey_url)
    return PostgresRunStore(
        engine=engine,
        app_role=settings.storage.app_role,
        queue=queue,
        max_concurrent=(
            settings.server.run_max_concurrent
            or settings.server.max_concurrent
        ),
        max_queue_size=settings.server.run_queue_max_size,
        # The durable store keeps terminal reports for a generous retention
        # window (default 90 days), NOT the in-memory store's short replay TTL
        # (run_completed_ttl_seconds) -- so research reports survive reloads,
        # re-logins, and other devices instead of being pruned after minutes.
        completed_ttl_seconds=settings.server.run_durable_retention_seconds,
        worker_id=f"api-{socket.gethostname()}-{os.getpid()}",
        audit=audit,
    )


def build_indexing_store(settings: Settings) -> Any:
    """Settings bridge for the reindex-job-store backend.

    Mirrors :func:`build_run_store` exactly: memory storage keeps the
    in-process job store byte-for-byte; ``INQTRIX_STORAGE_BACKEND=postgres``
    makes reindex records and events durable (in-process execution);
    ``INQTRIX_QUEUE_BACKEND=valkey`` additionally dispatches re-embeds to
    ``inqtrix-worker`` processes via a SEPARATE reindex stream. The
    durable store gets its OWN engine (asyncpg loop-affinity), distinct
    from both the run store and the knowledge document store.
    """
    from inqtrix.server.indexing import IndexingJobStore

    _require_valid_queue_storage(settings)
    if settings.storage.backend != "postgres":
        return IndexingJobStore.from_settings(settings.knowledge)

    import os
    import socket

    from inqtrix.runs.indexing_postgres import PostgresIndexingJobStore

    if not settings.storage.database_url.strip():
        raise RuntimeError(
            "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
            "INQTRIX_DATABASE_URL."
        )
    from inqtrix.storage.db import build_engine

    engine = build_engine(settings.storage.database_url)
    queue = None
    if settings.queue.backend == "valkey":
        from inqtrix.runs.indexing_queue import ValkeyIndexingQueue

        queue = ValkeyIndexingQueue(url=settings.queue.valkey_url)
    return PostgresIndexingJobStore(
        engine=engine,
        app_role=settings.storage.app_role,
        queue=queue,
        max_concurrent=settings.knowledge.reindex_max_concurrent,
        max_queue_size=settings.knowledge.reindex_queue_max_size,
        completed_ttl_seconds=settings.knowledge.reindex_completed_ttl_seconds,
        history_limit=settings.knowledge.reindex_history_limit,
        worker_id=f"api-{socket.gethostname()}-{os.getpid()}",
    )


def build_chat_store(settings: Settings) -> Any:
    """Settings bridge for the chat-history store backend (M6a).

    Mirrors the other tiered store builders: the in-memory store is the
    default (and the offline test backend); ``INQTRIX_STORAGE_BACKEND=
    postgres`` makes chat threads/groups/messages durable. The Postgres
    store gets its OWN NullPool engine (loop-agnostic), distinct from the
    run/knowledge/indexing engines — it is awaited only from the HTTP
    loop, but NullPool keeps it immune to loop-affinity at wiring time.
    """
    if settings.storage.backend != "postgres":
        from inqtrix.project.chat_memory import MemoryChatStore

        return MemoryChatStore()
    if not settings.storage.database_url.strip():
        raise RuntimeError(
            "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
            "INQTRIX_DATABASE_URL."
        )
    from inqtrix.project.chat_postgres import PostgresChatStore
    from inqtrix.storage.db import build_engine

    return PostgresChatStore(
        engine=build_engine(settings.storage.database_url, null_pool=True),
        app_role=settings.storage.app_role,
    )


def build_editor_store(settings: Settings) -> Any:
    """Settings bridge for the editor-persistence store backend (M6b).

    Mirrors :func:`build_chat_store`: in-memory default (and offline test
    backend); ``INQTRIX_STORAGE_BACKEND=postgres`` makes editor documents/
    folders/comments durable. The Postgres store gets its OWN NullPool
    engine, distinct from the chat/run/knowledge/indexing engines.
    """
    if settings.storage.backend != "postgres":
        from inqtrix.project.editor_memory import MemoryEditorStore

        return MemoryEditorStore()
    if not settings.storage.database_url.strip():
        raise RuntimeError(
            "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
            "INQTRIX_DATABASE_URL."
        )
    from inqtrix.project.editor_postgres import PostgresEditorStore
    from inqtrix.storage.db import build_engine

    return PostgresEditorStore(
        engine=build_engine(settings.storage.database_url, null_pool=True),
        app_role=settings.storage.app_role,
    )


def build_asset_store(settings: Settings) -> Any:
    """Settings bridge for the file-asset-record store backend (M6c).

    Mirrors :func:`build_chat_store`/:func:`build_editor_store`: in-memory
    default (offline/test); ``INQTRIX_STORAGE_BACKEND=postgres`` makes the
    file-library records (sections/groups/assets + extracted text) durable,
    on its OWN NullPool engine.
    """
    if settings.storage.backend != "postgres":
        from inqtrix.project.asset_records_memory import MemoryAssetStore

        return MemoryAssetStore()
    if not settings.storage.database_url.strip():
        raise RuntimeError(
            "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
            "INQTRIX_DATABASE_URL."
        )
    from inqtrix.project.asset_records_postgres import PostgresAssetStore
    from inqtrix.storage.db import build_engine

    return PostgresAssetStore(
        engine=build_engine(settings.storage.database_url, null_pool=True),
        app_role=settings.storage.app_role,
    )


def build_knowledge_session_store(settings: Settings) -> Any:
    """Settings bridge for the knowledge-session store backend (Wissensmodus).

    Mirrors :func:`build_asset_store`: in-memory default (offline/test);
    ``INQTRIX_STORAGE_BACKEND=postgres`` makes saved Ask sessions durable on
    their OWN NullPool engine.
    """
    if settings.storage.backend != "postgres":
        from inqtrix.project.knowledge_sessions_memory import (
            MemoryKnowledgeSessionStore,
        )

        return MemoryKnowledgeSessionStore()
    if not settings.storage.database_url.strip():
        raise RuntimeError(
            "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
            "INQTRIX_DATABASE_URL."
        )
    from inqtrix.project.knowledge_sessions_postgres import (
        PostgresKnowledgeSessionStore,
    )
    from inqtrix.storage.db import build_engine

    return PostgresKnowledgeSessionStore(
        engine=build_engine(settings.storage.database_url, null_pool=True),
        app_role=settings.storage.app_role,
    )


def build_vector_index_store(settings: Settings) -> Any:
    """Settings bridge for the vector-index-record store backend (M6c).

    Mirrors :func:`build_asset_store`: in-memory default (offline/test);
    ``INQTRIX_STORAGE_BACKEND=postgres`` makes the client vector-index
    records (file<->collection mapping + members + run history) durable, on
    its OWN NullPool engine.
    """
    if settings.storage.backend != "postgres":
        from inqtrix.project.vector_index_memory import MemoryVectorIndexStore

        return MemoryVectorIndexStore()
    if not settings.storage.database_url.strip():
        raise RuntimeError(
            "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
            "INQTRIX_DATABASE_URL."
        )
    from inqtrix.project.vector_index_postgres import PostgresVectorIndexStore
    from inqtrix.storage.db import build_engine

    return PostgresVectorIndexStore(
        engine=build_engine(settings.storage.database_url, null_pool=True),
        app_role=settings.storage.app_role,
    )


def build_account_preferences_store(settings: Settings) -> Any:
    """Settings bridge for the account-preferences store backend (M6c).

    Mirrors :func:`build_vector_index_store`: in-memory default (offline/test);
    ``INQTRIX_STORAGE_BACKEND=postgres`` makes the per-user UI preferences
    (theme/locale/contrast) durable, on its OWN NullPool engine.
    """
    if settings.storage.backend != "postgres":
        from inqtrix.project.account_preferences_memory import (
            MemoryAccountPreferencesStore,
        )

        return MemoryAccountPreferencesStore()
    if not settings.storage.database_url.strip():
        raise RuntimeError(
            "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
            "INQTRIX_DATABASE_URL."
        )
    from inqtrix.project.account_preferences_postgres import (
        PostgresAccountPreferencesStore,
    )
    from inqtrix.storage.db import build_engine

    return PostgresAccountPreferencesStore(
        engine=build_engine(settings.storage.database_url, null_pool=True),
        app_role=settings.storage.app_role,
    )


def build_permission_service(settings: Settings) -> PermissionService:
    """Permission-service half of :func:`build_platform_persistence`.

    Kept as the established seam for callers and tests that need only
    the authorization layer.
    """
    return build_platform_persistence(settings)[0]


def build_object_store(settings: Settings) -> "ObjectStore":
    """Settings bridge for the blob store (env-coupled surface).

    ``local`` (default) needs nothing; ``s3`` without endpoint or
    credentials is a contradiction and fails at startup.
    """
    from inqtrix.storage.object_store import LocalFSObjectStore, S3ObjectStore

    storage = settings.storage
    if storage.object_store_backend == "s3":
        if (
            not storage.s3_endpoint_url.strip()
            or not storage.s3_access_key
            or not storage.s3_secret_key
        ):
            raise RuntimeError(
                "INQTRIX_OBJECT_STORE_BACKEND=s3 verlangt gesetzte "
                "INQTRIX_S3_ENDPOINT_URL, INQTRIX_S3_ACCESS_KEY und "
                "INQTRIX_S3_SECRET_KEY."
            )
        return S3ObjectStore(
            endpoint_url=storage.s3_endpoint_url,
            bucket=storage.s3_bucket,
            access_key=storage.s3_access_key,
            secret_key=storage.s3_secret_key,
            region=storage.s3_region,
        )
    return LocalFSObjectStore(root=Path(storage.object_store_path))


def build_user_context_dependency(
    permissions: PermissionService,
    principal_dependency: Callable[..., Principal],
) -> Callable[..., Any]:
    """Build the FastAPI dependency resolving the request's user context.

    Composes with the principal dependency (FastAPI caches it
    per-request, so the auth gate still runs exactly once) and returns
    ``None`` for the legacy unscoped principals — routers pass the
    value straight into visibility-filtered store reads.
    """

    async def get_user_context(
        principal: Principal = Depends(principal_dependency),
    ) -> UserContext | None:
        return await permissions.resolve_user_context(principal)

    return get_user_context


def build_knowledge_context(
    settings: Settings,
    *,
    llm: Any | None = None,
) -> KnowledgeProviderContext | None:
    """Settings bridge for the knowledge engine (env-coupled surface).

    Returns ``None`` when the engine is disabled — no embedding
    provider is constructed and no knowledge surface exists. The
    embedding endpoint falls back to the LiteLLM gateway configuration
    so a standard proxy deployment enables knowledge with a single
    flag.
    """
    if not settings.knowledge.enabled:
        return None
    if settings.knowledge.embedding_provider == "azure":
        from inqtrix.providers.embeddings import AzureOpenAIEmbeddings

        if not settings.knowledge.embedding_azure_endpoint.strip():
            raise RuntimeError(
                "INQTRIX_EMBEDDING_PROVIDER=azure verlangt einen "
                "gesetzten Azure-Endpoint "
                "(INQTRIX_EMBEDDING_AZURE_ENDPOINT oder "
                "AZURE_AI_PROJECT_ENDPOINT)."
            )
        embeddings = AzureOpenAIEmbeddings(
            api_key=settings.knowledge.embedding_azure_api_key,
            azure_endpoint=settings.knowledge.embedding_azure_endpoint,
            api_version=settings.knowledge.embedding_azure_api_version,
            default_model=settings.knowledge.embedding_model,
            selectable_models=(
                settings.knowledge.selectable_embedding_model_list()
            ),
        )
    else:
        embeddings = LiteLLMEmbeddings(
            api_key=(
                settings.knowledge.embedding_api_key.strip()
                or settings.server.litellm_api_key
            ),
            base_url=(
                settings.knowledge.embedding_base_url.strip()
                or settings.server.litellm_base_url
            ),
            default_model=settings.knowledge.embedding_model,
            selectable_models=(
                settings.knowledge.selectable_embedding_model_list()
            ),
        )
    if settings.storage.backend == "postgres":
        # Postgres-canonical tier: collections/documents/chunks live
        # relationally (source of truth), the vectors in the vector index
        # (Qdrant when configured, in-process otherwise). The store gets
        # its OWN NullPool engine — it is awaited from the HTTP loop AND
        # bridged from the sync research graph / reindex worker via
        # run_coro_sync (a fresh per-call loop), which a pooled asyncpg
        # connection could not survive (same rule as the durable run store).
        from inqtrix.knowledge.stores.postgres_store import PostgresKnowledgeStore
        from inqtrix.storage.db import build_engine

        if not settings.storage.database_url.strip():
            raise RuntimeError(
                "INQTRIX_STORAGE_BACKEND=postgres verlangt eine gesetzte "
                "INQTRIX_DATABASE_URL."
            )
        if settings.knowledge.vector_backend == "qdrant":
            from inqtrix.knowledge.stores.qdrant_store import QdrantVectorIndex

            vector_index: Any = QdrantVectorIndex(
                url=settings.knowledge.qdrant_url,
                api_key=settings.knowledge.qdrant_api_key,
                sparse=settings.knowledge.sparse,
            )
        else:
            from inqtrix.knowledge.stores.vector_index import MemoryVectorIndex

            vector_index = MemoryVectorIndex()
        store = PostgresKnowledgeStore(
            engine=build_engine(
                settings.storage.database_url, null_pool=True
            ),
            app_role=settings.storage.app_role,
            vector_index=vector_index,
        )
    elif settings.knowledge.vector_backend == "qdrant":
        from inqtrix.knowledge.stores.qdrant_store import QdrantKnowledgeStore

        store = QdrantKnowledgeStore(
            url=settings.knowledge.qdrant_url,
            api_key=settings.knowledge.qdrant_api_key,
            sparse=settings.knowledge.sparse,
        )
    else:
        store = MemoryKnowledgeStore()

    reranker = None
    if settings.knowledge.reranker_provider == "cohere":
        from inqtrix.providers.rerankers import CohereRerank

        if (
            not settings.knowledge.reranker_base_url.strip()
            or not settings.knowledge.reranker_api_key
            or not settings.knowledge.reranker_model.strip()
        ):
            raise RuntimeError(
                "INQTRIX_RERANKER_PROVIDER=cohere verlangt gesetzte "
                "INQTRIX_RERANKER_BASE_URL, INQTRIX_RERANKER_API_KEY "
                "und INQTRIX_RERANKER_MODEL."
            )
        reranker = CohereRerank(
            api_key=settings.knowledge.reranker_api_key,
            base_url=settings.knowledge.reranker_base_url,
            default_model=settings.knowledge.reranker_model,
        )
    elif settings.knowledge.reranker_provider == "llm":
        from inqtrix.model_routing import resolve_model
        from inqtrix.providers.rerankers import LLMReranker

        if llm is None:
            raise RuntimeError(
                "INQTRIX_RERANKER_PROVIDER=llm verlangt einen "
                "konfigurierten LLM-Provider."
            )
        provider_models = getattr(llm, "models", None)
        reranker = LLMReranker(
            llm,
            default_model=(
                (resolve_model("knowledge_rerank", provider_models, None) or "")
                if provider_models is not None
                else ""
            ),
        )

    if (
        settings.knowledge.vector_backend == "qdrant"
        and settings.knowledge.sparse != "off"
        and reranker is None
    ):
        # Eval-backed finding (golden set, text-embedding-3-large):
        # plain RRF fusion without a rerank stage degrades rank-1 on
        # paraphrase queries versus dense-only. Hybrid pays off for
        # exact/out-of-vocabulary terms and with a reranker on top.
        log.warning(
            "Hybrid-Retrieval ohne Reranker-Stufe: RRF kann die "
            "Top-1-Praezision bei Paraphrase-Fragen verschlechtern. "
            "Reranker konfigurieren (INQTRIX_RERANKER_PROVIDER) oder "
            "INQTRIX_KNOWLEDGE_SPARSE=off setzen."
        )
    contextualizer = None
    if settings.knowledge.contextualize == "on":
        if llm is None:
            raise RuntimeError(
                "INQTRIX_KNOWLEDGE_CONTEXTUALIZE=on verlangt einen "
                "verfuegbaren LLM-Provider fuer die Ingestion."
            )
        from inqtrix.knowledge.contextualize import LLMChunkContextualizer

        contextualizer = LLMChunkContextualizer(llm)

    return KnowledgeProviderContext(
        embeddings=embeddings,
        store=store,
        default_top_k=settings.knowledge.default_top_k,
        reranker=reranker,
        rerank_candidate_depth=settings.knowledge.rerank_candidate_depth,
        contextualizer=contextualizer,
    )


@dataclass(frozen=True)
class AppContainer:
    """Fully wired collaborators for the HTTP routers.

    Attributes:
        settings: Root settings of the deployment.
        providers: Default provider bundle.
        strategies: Default strategy bundle.
        registry: Algorithm registry (built-ins plus any extensions
            registered by the caller before route registration).
        runtime: App-level runtime context handed to algorithms.
        auth_provider: Active principal resolver.
        principal_dependency: FastAPI dependency yielding the request
            principal; attached to every gated route.
        permission_service: Authorization chokepoint (membership,
            shares, visibility). Memory-backed by default.
        user_context_dependency: FastAPI dependency yielding the
            server-resolved :class:`UserContext` for scoped principals
            and ``None`` for the legacy unscoped ones.
        resolver: Per-request stack/override/mode resolution.
        chat_service: Non-streaming chat execution.
        run_service: Native run submission.
        health_service: Health/model payload assembly.
        run_store: Run registry/queue (read-side surface) — the
            in-memory default or the durable Postgres backend, both
            behind :class:`~inqtrix.runs.ports.RunStorePort`.
        semaphore_factory: Lazy provider of the shared concurrency
            limiter (the event loop may not exist at wiring time).
        knowledge_service: Knowledge collection/document/search service;
            ``None`` when the knowledge engine is disabled (no
            knowledge routes, no ``knowledge`` algorithm).
        stacks: Optional multi-stack registry for request resolution.
        default_stack: Default stack name when *stacks* is set.
    """

    settings: Settings
    providers: "ProviderContext"
    strategies: "StrategyContext"
    registry: AlgorithmRegistry
    runtime: RuntimeContext
    auth_provider: AuthProvider
    principal_dependency: Callable[..., Principal]
    permission_service: PermissionService
    user_context_dependency: Callable[..., Any]
    resolver: AgentContextResolver
    chat_service: ChatService
    run_service: RunService
    health_service: HealthService
    run_store: "RunStorePort"
    semaphore_factory: Callable[[], Any]
    knowledge_service: KnowledgeService | None = None
    file_service: FileService | None = None
    knowledge_ceiling: "KnowledgeStageCeiling | None" = None
    workspace_admin: Any = None
    share_service: Any = None
    prompt_template_service: Any = None
    quota_service: Any = None
    indexing_service: Any = None
    chat_history_service: Any = None
    editor_persistence_service: Any = None
    asset_records_service: Any = None
    knowledge_sessions_service: Any = None
    document_parser: Any = None
    vector_index_service: Any = None
    account_preferences_service: Any = None
    object_store_backend: str = "none"
    stacks: dict[str, Any] | None = None
    default_stack: str = ""


def build_container(
    *,
    providers: "ProviderContext",
    strategies: "StrategyContext",
    settings: Settings,
    semaphore_factory: Callable[[], Any],
    auth_provider: AuthProvider | None = None,
    api_key_dependency: Callable[["Request"], None] | None = None,
    stacks: dict[str, Any] | None = None,
    default_stack: str = "",
    run_store: "RunStorePort | None" = None,
    indexing_store: Any = None,
    registry: AlgorithmRegistry | None = None,
    knowledge: KnowledgeProviderContext | None = None,
    permissions: PermissionService | None = None,
    file_service: FileService | None = None,
    workspace_admin: Any = None,
    object_store_impl: "ObjectStore | None" = None,
) -> AppContainer:
    """Assemble the container from resolved collaborators.

    Auth resolution precedence (additive backwards compatibility):

    1. An injected *auth_provider* wins.
    2. A legacy injected *api_key_dependency* callable is adapted via
       :class:`~inqtrix.auth.api_key.CallableGateAuthProvider` —
       the historical ``register_routes(api_key_dependency=...)``
       seam keeps working unchanged.
    3. Otherwise the open-server :class:`NoneAuthProvider` applies
       (callers that want env-driven mode resolution build the
       provider via :func:`inqtrix.auth.api_key.build_auth_provider`
       and pass it in — ``create_app`` does exactly that).
    """
    if auth_provider is None:
        if api_key_dependency is not None:
            auth_provider = CallableGateAuthProvider(gate=api_key_dependency)
        else:
            auth_provider = NoneAuthProvider()

    active_knowledge = knowledge or build_knowledge_context(
        settings, llm=providers.llm
    )
    document_parser = None
    if (
        active_knowledge is not None
        and settings.knowledge.document_parser == "markitdown"
    ):
        from inqtrix.knowledge.parsing import MarkItDownParser

        document_parser = MarkItDownParser()
    knowledge_service = (
        KnowledgeService(
            knowledge=active_knowledge,
            chunk_max_chars=settings.knowledge.chunk_max_chars,
            max_document_chars=settings.knowledge.max_document_chars,
            parser=document_parser,
        )
        if active_knowledge is not None
        else None
    )

    active_registry = registry or build_default_registry()
    knowledge_ceiling: KnowledgeStageCeiling | None = None
    if active_knowledge is not None:
        # ONE ceiling instance feeds both the algorithm and the
        # capabilities manifest — a second derivation would let the
        # two drift apart.
        knowledge_ceiling = KnowledgeStageCeiling(
            gate_available=settings.knowledge.gate == "on",
            grounding_available=settings.knowledge.grounding == "on",
            reranker_available=active_knowledge.reranker is not None,
            gate_max_rounds=settings.knowledge.gate_max_rounds,
            rerank_candidate_depth=settings.knowledge.rerank_candidate_depth,
        )
    if active_knowledge is not None and "knowledge" not in active_registry.ids():
        active_registry.register(
            KnowledgeAlgorithm(
                knowledge=active_knowledge,
                citation_base_url=settings.server.public_base_url,
                gate_enabled=settings.knowledge.gate == "on",
                grounding_enabled=settings.knowledge.grounding == "on",
                gate_max_rounds=settings.knowledge.gate_max_rounds,
            )
        )
    runtime = RuntimeContext(
        settings=settings,
        registry=active_registry,
        providers=providers,
        strategies=strategies,
    )
    resolver = AgentContextResolver(
        providers=providers,
        strategies=strategies,
        settings=settings,
        registry=active_registry,
        stacks=stacks,
        default_stack=default_stack,
    )
    active_run_store = run_store or build_run_store(settings)
    injected_file_service = file_service is not None
    needs_persistence = permissions is None or file_service is None
    active_object_store_backend = (
        "custom"
        if object_store_impl is not None or injected_file_service
        else settings.storage.object_store_backend
    )
    bundle = (
        build_platform_persistence_bundle(settings)
        if needs_persistence
        else None
    )
    if needs_persistence:
        assert bundle is not None
        active_permissions = permissions or bundle.permissions
        active_file_service = file_service or FileService(
            registry=bundle.file_registry,
            # An injected object store wins over the env enum dispatch — the
            # Enterprise-Austausch seam for a custom blob backend (the other
            # stores ride run_store=/knowledge=/permissions=).
            object_store=object_store_impl or build_object_store(settings),
            permissions=active_permissions,
            max_file_bytes=settings.storage.max_file_bytes,
        )
    else:
        active_permissions = permissions
        active_file_service = file_service
    active_workspace_admin = workspace_admin or (
        bundle.workspace_admin if bundle is not None else None
    )
    from inqtrix.services.prompt_template_service import (
        PromptTemplateService,
    )

    if bundle is not None and bundle.prompt_templates is not None:
        template_repository = bundle.prompt_templates
    else:
        # Injected-permissions composition (tests, embedders) gets the
        # in-process repository — visible here, never a silent skip.
        from inqtrix.content.prompt_templates import (
            MemoryPromptTemplateRepository,
        )

        template_repository = MemoryPromptTemplateRepository()
    prompt_template_service = PromptTemplateService(
        repository=template_repository,
        # Only the Postgres backend survives restarts; the capability
        # manifest must not invite browsers to sync against a store
        # that reads as "everything deleted" after a bounce.
        durable=settings.storage.backend == "postgres",
    )
    share_service = None
    provider_users = getattr(auth_provider, "users", None)
    # Sharing needs scoped principals + a user mirror — every cookie-session
    # mode qualifies (oidc/local/ldap), not just oidc. The single-operator
    # none/apikey modes are deliberately excluded: their unscoped principal
    # has no identity to share with or as (use local for single-user-with-
    # sharing). ADR-AUTH-4 withdrawn — no static-principal rescoping.
    if (
        auth_provider.mode in {"oidc", "local", "ldap"}
        and active_workspace_admin is not None
        and provider_users is not None
    ):
        from inqtrix.auth.shares import ShareService

        async def _run_owner(tenant_id: str, resource_id: str):
            return active_run_store.owner_sub(resource_id)

        async def _user_exists(tenant_id: str, sub: str) -> bool:
            return await provider_users.has_subject(
                tenant_id=tenant_id, sub=sub
            )

        owner_resolvers: dict = {"run": _run_owner}
        if knowledge_service is not None:

            async def _collection_owner(tenant_id: str, resource_id: str):
                from inqtrix.knowledge.stores.ports import CollectionNotFound

                try:
                    collection = await knowledge_service.knowledge.store.get_collection(
                        resource_id
                    )
                except CollectionNotFound:
                    return None
                # Legacy collections (created_by_sub None) have no
                # owner and need none — they are visible to everyone
                # already; None makes them unshareable (404).
                return collection.created_by_sub

            owner_resolvers["knowledge_collection"] = _collection_owner

        async def _template_owner(tenant_id: str, resource_id: str):
            return await prompt_template_service.owner_sub(
                tenant_id, resource_id
            )

        owner_resolvers["prompt_template"] = _template_owner

        share_service = ShareService(
            shares=active_workspace_admin,
            permissions=active_permissions,
            owner_resolvers=owner_resolvers,
            user_lookup=_user_exists,
            audit=active_workspace_admin,
            restrict_to_members=settings.sharing.restrict_to_workspace_members,
        )
    # Quota service: only when enabled AND a multi-user mode
    # (oidc/local/ldap). The single-operator none/apikey/demo modes are
    # never metered — unscoped principals would bypass anyway, and not
    # constructing it keeps them byte-identical. Store mirrors the backend.
    quota_service = None
    if settings.quota.enabled and auth_provider.mode in {"oidc", "local", "ldap"}:
        from inqtrix.services.quota_service import QuotaService

        if settings.storage.backend == "postgres":
            from inqtrix.storage.quota_postgres import PostgresQuotaStore

            quota_store: Any = PostgresQuotaStore(
                database_url=settings.storage.database_url,
                app_role=settings.storage.app_role,
            )
        else:
            from inqtrix.quota.memory import MemoryQuotaStore

            quota_store = MemoryQuotaStore()
        quota_service = QuotaService(
            store=quota_store, settings=settings.quota
        )
    # Background reindex (re-embed) jobs: only when the knowledge engine
    # is wired. The in-memory job store mirrors the run store; quota (when
    # on) meters the per-document embedding spend. None keeps deployments
    # without knowledge byte-identical (no reindex surface).
    indexing_service = None
    if knowledge_service is not None:
        from inqtrix.services.indexing_service import IndexingService

        indexing_service = IndexingService(
            knowledge_service=knowledge_service,
            job_store=indexing_store or build_indexing_store(settings),
            quota_service=quota_service,
        )
    # Project-persistence tier (M6a): chat history becomes server-persistent
    # when a Postgres backend is present. Always wired (the memory store is
    # the offline/test tier); the `durable` flag — true only for Postgres —
    # gates the capability the frontend reads, so the volatile tier is never
    # advertised as a durable replacement for the local markdown project.
    from inqtrix.services.chat_history_service import ChatHistoryService

    chat_history_service = ChatHistoryService(
        store=build_chat_store(settings),
        durable=settings.storage.backend == "postgres",
    )
    # Editor persistence (M6b): same project-persistence tier as chat, same
    # durability gate (the capability flag reflects the whole tier).
    from inqtrix.services.editor_persistence_service import (
        EditorPersistenceService,
    )

    editor_persistence_service = EditorPersistenceService(
        store=build_editor_store(settings),
        durable=settings.storage.backend == "postgres",
    )
    from inqtrix.services.asset_records_service import AssetRecordsService

    asset_records_service = AssetRecordsService(
        store=build_asset_store(settings),
        durable=settings.storage.backend == "postgres",
    )
    from inqtrix.services.knowledge_sessions_service import (
        KnowledgeSessionsService,
    )

    knowledge_sessions_service = KnowledgeSessionsService(
        store=build_knowledge_session_store(settings),
        durable=settings.storage.backend == "postgres",
    )
    from inqtrix.services.vector_index_service import VectorIndexService

    vector_index_service = VectorIndexService(
        store=build_vector_index_store(settings),
        durable=settings.storage.backend == "postgres",
    )
    from inqtrix.services.account_preferences_service import (
        AccountPreferencesService,
    )

    account_preferences_service = AccountPreferencesService(
        store=build_account_preferences_store(settings),
        durable=settings.storage.backend == "postgres",
    )
    principal_dependency = auth_provider.build_principal_dependency()

    return AppContainer(
        settings=settings,
        providers=providers,
        strategies=strategies,
        registry=active_registry,
        runtime=runtime,
        auth_provider=auth_provider,
        principal_dependency=principal_dependency,
        permission_service=active_permissions,
        user_context_dependency=build_user_context_dependency(
            active_permissions, principal_dependency
        ),
        resolver=resolver,
        chat_service=ChatService(registry=active_registry, runtime=runtime),
        run_service=RunService(
            registry=active_registry,
            runtime=runtime,
            run_store=active_run_store,
            quota_service=quota_service,
        ),
        health_service=HealthService(
            providers=providers,
            settings=settings,
            auth_provider=auth_provider,
            stacks=stacks,
            default_stack=default_stack,
        ),
        run_store=active_run_store,
        semaphore_factory=semaphore_factory,
        knowledge_service=knowledge_service,
        file_service=active_file_service,
        knowledge_ceiling=knowledge_ceiling,
        workspace_admin=active_workspace_admin,
        share_service=share_service,
        prompt_template_service=prompt_template_service,
        quota_service=quota_service,
        indexing_service=indexing_service,
        chat_history_service=chat_history_service,
        editor_persistence_service=editor_persistence_service,
        asset_records_service=asset_records_service,
        knowledge_sessions_service=knowledge_sessions_service,
        document_parser=document_parser,
        vector_index_service=vector_index_service,
        account_preferences_service=account_preferences_service,
        object_store_backend=active_object_store_backend,
        stacks=stacks,
        default_stack=default_stack,
    )
