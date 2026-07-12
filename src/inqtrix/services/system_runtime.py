"""Sanitized runtime manifest for the instance-admin system view."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from inqtrix.urls import sanitize_log_message

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer

log = logging.getLogger("inqtrix")

_RUNTIME_PROBE_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class RuntimeProbeResults:
    """Read-only backend availability checks for runtime/capability views."""

    object_store_available: bool | None = None
    queue_available: bool | None = None
    vector_store_available: bool | None = None


async def system_runtime_payload_checked(
    container: "AppContainer",
) -> dict[str, Any]:
    """Return the runtime payload with current backend availability probes."""
    probes = await runtime_probe_results(container)
    return system_runtime_payload(container, probes=probes)


def system_runtime_payload(
    container: "AppContainer",
    *,
    probes: RuntimeProbeResults | None = None,
) -> dict[str, Any]:
    """Return the deployment runtime shape without secrets or endpoints.

    Args:
        container: The already-wired application container. The payload is
            derived from the same settings and collaborators used by the
            routers and algorithms, so the admin UI describes what is
            configured to execute rather than re-inferring it on the client.
        probes: Optional availability booleans from read-only runtime
            checks. When omitted, the payload keeps its historical
            configuration-only semantics for pure unit call sites.

    Returns:
        A JSON-serializable manifest of backend categories only. It
        deliberately omits database URLs, object-store paths, bucket names,
        service endpoints, and credentials. Queue values describe the
        configured dispatch mode; they do not claim live worker heartbeats.
    """
    settings = container.settings
    knowledge = _knowledge_payload(container, probes=probes)
    queue_backend = settings.queue.backend
    object_store = (
        container.object_store_backend
        if container.file_service is not None
        else "none"
    )
    object_store_available = (
        probes.object_store_available
        if probes is not None and probes.object_store_available is not None
        else container.file_service is not None
    )
    queue_available = (
        probes.queue_available
        if probes is not None and probes.queue_available is not None
        else True
    )

    return {
        "api": {
            "openapi": settings.server.enable_openapi,
        },
        "files": {
            "blob_storage": _blob_storage_label(object_store),
            "enabled": container.file_service is not None,
            "max_file_bytes": (
                settings.storage.max_file_bytes
                if container.file_service is not None
                else None
            ),
            "object_store": object_store,
            "object_store_available": object_store_available,
        },
        "knowledge": knowledge,
        "runs": {
            "execution": (
                "worker_dispatch" if queue_backend == "valkey" else "in_process"
            ),
            "queue": queue_backend,
            "queue_available": queue_available,
            "store": (
                "postgres" if settings.storage.backend == "postgres" else "memory"
            ),
            "worker_dispatch": queue_backend == "valkey",
        },
        "storage": {
            "backend": settings.storage.backend,
            "durable": settings.storage.backend == "postgres",
        },
    }


async def runtime_probe_results(container: "AppContainer") -> RuntimeProbeResults:
    """Probe optional backing services without exposing connection details."""
    object_store, queue, vector_store = await asyncio.gather(
        _probe_object_store(container),
        _probe_queue(container),
        _probe_vector_store(container),
    )
    return RuntimeProbeResults(
        object_store_available=object_store,
        queue_available=queue,
        vector_store_available=vector_store,
    )


def runtime_feature_overrides(
    runtime: dict[str, Any],
) -> dict[str, bool]:
    """Derive effective feature booleans from a runtime payload."""
    files = runtime.get("files") or {}
    knowledge = runtime.get("knowledge") or {}
    files_on = bool(files.get("enabled")) and bool(
        files.get("object_store_available", True)
    )
    knowledge_enabled = bool(knowledge.get("enabled"))
    knowledge_on = knowledge_enabled and bool(
        knowledge.get("vector_store_available", True)
    )
    return {
        # Document parsing is pure CPU work (MarkItDown) with no vector-store
        # dependency: the parser exists whenever knowledge is enabled, and the
        # GET /v1/files/{id}/text endpoint needs only the parser. Gating this on
        # knowledge_on would, on a transient vector-store outage, silently
        # downgrade uploads to the weaker client parser (Baukasten +
        # No-Silent-Fallbacks). So gate on knowledge_enabled, not knowledge_on.
        "document_parser": knowledge_enabled
        and knowledge.get("document_parser") not in (None, "none"),
        "embedding_provider": knowledge_on
        and knowledge.get("embedding_provider") not in (None, "none"),
        "files": files_on,
        "hybrid_retrieval": knowledge_on
        and bool(knowledge.get("hybrid_retrieval")),
        "knowledge": knowledge_on,
        "reranker": knowledge_on and knowledge.get("reranker") not in (None, "none"),
    }


def _knowledge_payload(
    container: "AppContainer",
    *,
    probes: RuntimeProbeResults | None = None,
) -> dict[str, Any]:
    settings = container.settings
    service = container.knowledge_service
    if service is None:
        return {
            "contextual_retrieval": False,
            "cross_lingual_recommendation": "reranker",
            "default_top_k": None,
            "document_parser": "none",
            "embedding_model": None,
            "embedding_provider": None,
            "enabled": False,
            "hybrid_retrieval": False,
            "reranker": "none",
            "sparse": None,
            "sparse_mode": "off",
            "sparse_language": None,
            "sparse_multilingual": False,
            "vector_store": "none",
            "vector_store_available": False,
        }

    context = service.knowledge
    vector_store_available = (
        probes.vector_store_available
        if probes is not None and probes.vector_store_available is not None
        else True
    )
    # The store's lexical-branch language is the single truth for both the
    # normalized mode (bm25/off) and the language code — derived once, never
    # re-asserted independently.
    sparse_language = getattr(context.store, "sparse_language", None)
    return {
        "contextual_retrieval": context.contextualizer is not None,
        # Keyword (BM25) retrieval is monolingual and never cross-lingual; the
        # cross-lingual lever is a multilingual cross-encoder reranker. Static
        # facts so clients can surface the limitation honestly (the per-run
        # query-vs-document mismatch needs collection language — a later phase).
        "cross_lingual_recommendation": "reranker",
        "default_top_k": context.default_top_k,
        "document_parser": (
            settings.knowledge.document_parser
            if service.parser is not None
            else "none"
        ),
        "embedding_model": context.embeddings.default_model,
        "embedding_provider": settings.knowledge.embedding_provider,
        "enabled": True,
        "hybrid_retrieval": bool(getattr(context.store, "supports_hybrid", False))
        and vector_store_available,
        "reranker": (
            settings.knowledge.reranker_provider
            if context.reranker is not None
            else "none"
        ),
        "sparse": (
            settings.knowledge.sparse
            if settings.knowledge.vector_backend == "qdrant"
            else None
        ),
        "sparse_mode": "bm25" if sparse_language is not None else "off",
        "sparse_language": sparse_language,
        "sparse_multilingual": False,
        "vector_store": settings.knowledge.vector_backend,
        "vector_store_available": vector_store_available,
    }


def _blob_storage_label(object_store: str) -> str:
    if object_store == "local":
        return "volume"
    if object_store == "s3":
        return "s3"
    return object_store


async def _probe_object_store(container: "AppContainer") -> bool:
    service = container.file_service
    if service is None:
        return False

    async def probe() -> bool:
        checker = getattr(service, "object_store_available", None)
        if checker is None:
            return True
        return bool(await checker())

    return await _bounded_probe("object_store", probe)


async def _probe_queue(container: "AppContainer") -> bool:
    settings = container.settings
    if settings.queue.backend != "valkey":
        return True
    return await _bounded_probe(
        "valkey_queue",
        lambda: asyncio.to_thread(_ping_valkey, settings.queue.valkey_url),
    )


async def _probe_vector_store(container: "AppContainer") -> bool:
    service = container.knowledge_service
    if service is None:
        return False
    store = service.knowledge.store
    checker = getattr(store, "is_available", None)
    if checker is None:
        return True

    async def probe() -> bool:
        result = checker()
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    return await _bounded_probe("vector_store", probe)


async def readiness_payload(
    container: "AppContainer",
) -> tuple[int, dict[str, Any]]:
    """Build the ``/readyz`` payload and its HTTP status code.

    Readiness differs from ``/health`` (liveness, provider-only): a pod
    whose DATABASE or QUEUE is unreachable cannot serve requests and
    must leave the load-balancer rotation (503). A down VECTOR store
    only degrades the knowledge feature — research/chat/files still
    work and knowledge routes fail loudly per-request — so it reports
    ``degraded`` but stays ready (200). Every probe is read-only and
    bounded (:data:`_RUNTIME_PROBE_TIMEOUT_SECONDS`), well below usual
    kubelet probe timeouts; the memory backends are trivially ready so
    the zero-infrastructure default stays green.
    """
    database_ok, queue_ok, vector_ok = await asyncio.gather(
        _probe_database(container),
        _probe_queue(container),
        _probe_vector_store_ready(container),
    )
    ready = database_ok and queue_ok
    status = "ready" if ready and vector_ok else (
        "degraded" if ready else "not_ready"
    )
    checks = {
        "database": _check_label(
            database_ok, skipped=container.settings.storage.backend != "postgres"
        ),
        "queue": _check_label(
            queue_ok, skipped=container.settings.queue.backend != "valkey"
        ),
        "vector_store": _check_label(
            vector_ok, skipped=container.knowledge_service is None
        ),
    }
    return (200 if ready else 503), {"status": status, "checks": checks}


def _check_label(ok: bool, *, skipped: bool) -> str:
    if skipped:
        return "skipped"
    return "ok" if ok else "unavailable"


async def _probe_database(container: "AppContainer") -> bool:
    settings = container.settings
    if settings.storage.backend != "postgres":
        return True
    session_factory = container.session_factory
    if session_factory is None:
        # postgres declared but no factory wired — a composition bug
        # that must read as not-ready, never as silently green.
        log.warning(
            "Readiness: Storage-Backend postgres ohne Session-Factory — "
            "Datenbank-Probe meldet unavailable."
        )
        return False

    async def probe() -> bool:
        from sqlalchemy import text

        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
        return True

    return await _bounded_probe("database", probe)


async def _probe_vector_store_ready(container: "AppContainer") -> bool:
    # Readiness variant: NO knowledge service means the feature is off —
    # trivially ready (the admin runtime view reports False there
    # because it describes feature availability, not pod readiness).
    if container.knowledge_service is None:
        return True
    return await _probe_vector_store(container)


async def _bounded_probe(
    name: str,
    probe: Callable[[], Awaitable[bool]],
) -> bool:
    try:
        return bool(
            await asyncio.wait_for(
                probe(),
                timeout=_RUNTIME_PROBE_TIMEOUT_SECONDS,
            )
        )
    except Exception as exc:  # noqa: BLE001 - status payload degrades visibly.
        log.warning(
            "Runtime availability probe failed for %s: %s",
            name,
            sanitize_log_message(exc),
        )
        return False


def _ping_valkey(url: str) -> bool:
    import valkey

    client = valkey.Valkey.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
    )
    try:
        return bool(client.ping())
    finally:
        client.close()
