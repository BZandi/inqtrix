"""Capability manifest endpoint: clients discover features, never hardcode.

Unauthenticated by design, like ``/health``, ``/v1/models``, and
``/v1/stacks`` (ADR-MS-3 reasoning): the UI needs the manifest before
any credential prompt, and the payload exposes only feature/algorithm
identity — no secrets, no per-deployment internals beyond what
``/health`` already reveals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter

from inqtrix.embedding_cards import build_embedding_catalog
from inqtrix.knowledge.profiles import (
    EVIDENCE_K_MAX,
    KnowledgeProfile,
    build_profile_manifest,
)
from inqtrix.services.request_parsing import (
    editor_wait_seconds,
    request_timeout_seconds,
    text_wait_seconds,
)
from inqtrix.services.system_runtime import (
    runtime_feature_overrides,
    system_runtime_payload_checked,
)

if TYPE_CHECKING:
    from inqtrix.server.container import AppContainer


def build_router(container: "AppContainer") -> APIRouter:
    """Bind the capability manifest route against the container."""
    router = APIRouter()
    registry = container.registry
    settings = container.settings
    knowledge_service = container.knowledge_service

    @router.get("/v1/capabilities")
    async def capabilities():
        runtime = await system_runtime_payload_checked(container)
        feature_overrides = runtime_feature_overrides(runtime)
        knowledge_enabled = feature_overrides.get("knowledge", False)
        files_enabled = feature_overrides.get("files", False)
        payload = {
            "algorithms": registry.manifest(),
            "features": {
                "knowledge": knowledge_enabled,
                "embedding_provider": feature_overrides.get(
                    "embedding_provider", knowledge_enabled
                ),
                "openapi": settings.server.enable_openapi,
                "files": files_enabled,
                # Server-side document parsing is pure MarkItDown CPU work with
                # no vector-store dependency, so it is advertised at the top
                # level (not inside the knowledge block) and stays available on
                # a transient vector-store outage. The browser gates its upload
                # parse on this; the GET /v1/files/{id}/text endpoint mirrors it.
                "document_parser": feature_overrides.get("document_parser", False),
                "sharing": container.share_service is not None,
                # Browsers sync rules against this surface — only a
                # DURABLE store may advertise it (a volatile store
                # reads as "everything deleted" after a restart).
                "prompt_templates": (
                    container.prompt_template_service is not None
                    and container.prompt_template_service.durable
                ),
                # Per-user quotas are enforced — the UI shows the meter
                # and (for owners) the admin panel only when this is on.
                "quota": container.quota_service is not None,
                # The project-persistence tier (chat/editor/assets/vector-index)
                # is server-durable — the frontend may move its local project
                # to the server only when this is on (a volatile store
                # would read as "everything deleted" after a restart, the
                # same rule as prompt_templates).
                "project_persistence": (
                    container.chat_history_service is not None
                    and container.chat_history_service.durable
                ),
            },
            # Effective HTTP wait deadlines (seconds) the browser must NOT
            # abort before. Derived from the SAME helpers the editor/text/chat
            # routes enforce, so the published value provably equals what
            # actually runs — the client derives its own AbortController
            # timeout from these instead of hardcoding one (which would
            # silently cap a raised server-side timeout). Plain integers from
            # non-secret settings, consistent with this endpoint's contract.
            "timeouts": {
                "editor_wait_seconds": editor_wait_seconds(settings.agent),
                "chat_wait_seconds": request_timeout_seconds(settings.agent),
                "text_wait_seconds": text_wait_seconds(settings.agent),
            },
        }
        if files_enabled:
            payload["files"] = {
                "max_file_bytes": settings.storage.max_file_bytes,
            }
        if knowledge_enabled:
            context = knowledge_service.knowledge
            payload["features"]["hybrid_retrieval"] = feature_overrides.get(
                "hybrid_retrieval", False
            )
            payload["features"]["reranker"] = feature_overrides.get(
                "reranker", False
            )
            payload["features"]["contextual_retrieval"] = (
                context.contextualizer is not None
            )
            embeddings = context.embeddings
            # The store's lexical-branch language is the single truth for both
            # the normalized mode (bm25/off) and the language code.
            sparse_language = getattr(context.store, "sparse_language", None)
            payload["knowledge"] = {
                "default_embedding_model": embeddings.default_model,
                "embedding_catalog": build_embedding_catalog(
                    embeddings.selectable_embedding_models
                    or [embeddings.default_model]
                ),
                "default_top_k": knowledge_service.knowledge.default_top_k,
                # Hard ceiling on the FINAL evidence count (``final_k``), so a
                # client can bound its ``final_k`` override field to the same
                # cap the algorithm clamps to.
                "evidence_k_max": EVIDENCE_K_MAX,
                # Keyword (BM25) retrieval is language-bound and never
                # cross-lingual; the cross-lingual lever is a multilingual
                # cross-encoder reranker. Static facts so the UI can show the
                # limitation honestly instead of silently expecting more.
                "sparse_mode": "bm25" if sparse_language is not None else "off",
                "sparse_language": sparse_language,
                "sparse_multilingual": False,
                "cross_lingual_recommendation": "reranker",
            }
            if container.knowledge_ceiling is not None:
                # The SAME ceiling instance the algorithm runs against
                # — the manifest must describe what would actually
                # execute, including operator degradation per profile.
                payload["knowledge"]["profiles"] = build_profile_manifest(
                    container.knowledge_ceiling
                )
                payload["knowledge"]["default_profile"] = (
                    KnowledgeProfile.STANDARD.value
                )
                payload["knowledge"]["reranker_provider"] = (
                    settings.knowledge.reranker_provider
                )
        return payload

    return router
