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

from inqtrix.agents.tier_policy import (
    DEFAULT_AGENT_TIER,
    tier_capabilities_payload,
)
from inqtrix.embedding_cards import build_embedding_catalog
from inqtrix.knowledge.profiles import (
    EVIDENCE_K_MAX,
    KnowledgeProfile,
    build_profile_manifest,
)
from inqtrix.providers.base import MAX_PROVIDER_ATTEMPTS
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


def _permission_mode_entry(mode: str) -> dict[str, object]:
    """What one autonomy mode gates, derived from the ENFORCING code.

    Kernel tool gates come from the compiled policy config
    (``interrupt_config_for``); the web-replan rule is probed against
    the ACTUAL E16 decision function with a synthetic web task — no
    literal here can drift from runtime behavior. ``plan_gate`` is the
    one graph-routing fact stated as a constant (autonomous skips the
    plan interrupt, E16); ``patch_gate`` is invariant (E14).
    """
    from inqtrix.agents.control_ports import PlanTaskRecord
    from inqtrix.agents.kernel.policy import (
        ALWAYS_GATED_TOOLS,
        interrupt_config_for,
    )
    from inqtrix.agents.replan import autonomy_auto_approves

    config = interrupt_config_for(mode) or {}
    web_probe = PlanTaskRecord(
        task_id="probe",
        plan_id="probe",
        run_id="probe",
        ordinal=0,
        title="probe",
        tool_kind="web_research",
    )
    return {
        "plan_gate": mode != "autonomous",
        "web_replan_regate": not autonomy_auto_approves(
            autonomy=mode, new_tasks=[web_probe]
        ),
        "patch_gate": True,
        "kernel_gated_tools": sorted(
            name
            for name, entry in config.items()
            if name not in ALWAYS_GATED_TOOLS and "when" not in entry
        ),
        "kernel_conditional_tools": sorted(
            name for name, entry in config.items() if "when" in entry
        ),
        "kernel_always_gated": list(ALWAYS_GATED_TOOLS),
    }


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
        capability_ids = (
            set(container.capability_registry.ids())
            if container.capability_registry is not None
            else set()
        )
        kernel_available = "agent_kernel" in container.registry.ids()
        web_source_available = "web.search.instant" in capability_ids
        knowledge_source_available = "knowledge.search" in capability_ids
        collaboration_service = container.editor_collaboration_service
        collaboration_available = bool(
            collaboration_service is not None
            and await collaboration_service.service_available()
        )
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
                # Skills follow the same durable-store rule (plan M3).
                "skills": (
                    container.skill_service is not None
                    and container.skill_service.durable
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
                "collaboration": collaboration_available,
                # Wave-1 agent capability tools are registered — the same
                # curated registry the (later) deepagents and MCP adapters
                # consume. On when at least one capability is wired.
                "agent_tools": bool(
                    container.capability_registry is not None
                    and container.capability_registry.ids()
                ),
                "workspace_agent": "workspace_agent"
                in container.registry.ids(),
                "workspace_agent_durable": bool(
                    "workspace_agent" in container.registry.ids()
                    and settings.storage.backend == "postgres"
                ),
                # Cognitive kernel (plan M2): True only when the
                # registration gate passed (rollout switch AND
                # checkpointer AND native tool calling) — frontends
                # feature-detect mode=agent_kernel through this flag.
                "agent_kernel": "agent_kernel" in container.registry.ids(),
            },
            "collaboration": {
                "configured": collaboration_service is not None,
                "service_available": collaboration_available,
                "transport_path": "/collaboration",
                "protocol_version": settings.collaboration.protocol_version,
                "schema_version": settings.collaboration.schema_version,
                "mode": "single_replica",
            },
            # Workspace-agent limits + vocabulary (M5): the desk reads
            # these instead of hardcoding, decision E16/E8.
            "agent": {
                "autonomy_modes": ["strict", "balanced", "autonomous"],
                "default_autonomy": settings.agent_platform.default_autonomy,
                # The EFFECTIVE desk default (plan M2 rollout): the
                # configured kernel default only publishes once the
                # kernel actually registered — the frontend never
                # submits an unregistered mode.
                "default_mode": (
                    "agent_kernel"
                    if settings.agent_platform.default_agent_mode
                    == "agent_kernel"
                    and "agent_kernel" in container.registry.ids()
                    else "workspace_agent"
                ),
                # Two-mode UI presets (plan M1 S7, the Cowork pattern):
                # the composer renders Standard/Auto and maps onto the
                # UNCHANGED wire vocabulary above. advanced_autonomy=True
                # republishes the legacy three-way control instead.
                "mode_presets": [
                    {"id": "standard", "autonomy": "balanced"},
                    {"id": "auto", "autonomy": "autonomous"},
                ],
                # Thoroughness (plan M4): orthogonal to the permission
                # mode; the composer renders the toggle only when the
                # server publishes it (feature detection, no hardcoded
                # claim).
                "depth_modes": [
                    {"id": "normal"},
                    {"id": "deep"},
                ],
                # Mirrors the apply_overrides env-tier bridge: with an
                # env AGENT_TIER set, every no-override run gets the
                # bridged depth — publish THAT, not the raw env value.
                "default_depth": (
                    ("deep" if settings.agent.agent_tier == "tief" else "normal")
                    if settings.agent.agent_tier
                    else settings.agent.depth
                ),
                # Stufen (published == enforced: generated from THE
                # policy table every consumer reads). `depth_modes`
                # stays published for older clients; a tiers-aware
                # composer renders the Stufe control instead.
                "tiers": tier_capabilities_payload(
                    max_clarification_rounds=(
                        settings.agent_platform.max_clarification_rounds
                    ),
                ),
                # The composer's default SELECTION (env AGENT_TIER wins
                # over the vocabulary default). A request that OMITS
                # agent_tier keeps legacy depth semantics — tiers-aware
                # clients therefore always send their selection.
                "default_tier": (
                    settings.agent.agent_tier or DEFAULT_AGENT_TIER
                ),
                "advanced_autonomy": (
                    settings.agent_platform.advanced_autonomy
                ),
                "max_parallel_children": (
                    settings.agent_platform.max_parallel_children
                ),
                "discovery_max_tool_calls": (
                    settings.agent_platform.discovery_max_tool_calls
                ),
                "max_plan_tasks": settings.agent_platform.max_plan_tasks,
                "durable": settings.storage.backend == "postgres",
                "tools": (
                    container.capability_registry.manifest()
                    if container.capability_registry is not None
                    else []
                ),
                # What each permission mode actually gates — generated
                # from THE enforcing sources (kernel policy config +
                # the E16 replan rule), so published == enforced and
                # the composer's run overview cannot drift from the
                # runtime (Designprinzip 5).
                "permission_modes": {
                    mode: _permission_mode_entry(mode)
                    for mode in ("strict", "balanced", "autonomous")
                },
                # Skill LIMITS only (plan M3): the skill list itself
                # comes from the authenticated GET /v1/skills — this
                # endpoint is unauthenticated and must never leak
                # titles or labels.
                "skills": {
                    "max_attached": (
                        settings.agent_platform.skills_max_attached
                    ),
                    "disclosure_budget_chars": (
                        settings.agent_platform.skills_disclosure_budget_chars
                    ),
                },
                # Composer source controls and enforced one-shot routes.
                # Availability is derived from the SAME registries runtime
                # dispatch uses, so the desk never advertises a dead control.
                "source_controls": [
                    {
                        "id": "web",
                        "default": "available",
                        "available": web_source_available,
                    },
                    {
                        "id": "knowledge",
                        "default": "available",
                        "available": knowledge_source_available,
                    },
                ],
                "execution_directives": [
                    {
                        "id": "quick_web",
                        "available": (
                            kernel_available and web_source_available
                        ),
                    },
                    {
                        "id": "knowledge_only",
                        "available": (
                            kernel_available and knowledge_source_available
                        ),
                    },
                ],
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
                "reasoning_operation_seconds": settings.agent.reasoning_timeout,
                "editor_operation_seconds": (
                    settings.agent.editor_assistant_timeout
                ),
                "search_operation_seconds": settings.agent.search_timeout,
                "claim_extract_operation_seconds": (
                    settings.agent.claim_extract_timeout
                ),
                "research_run_seconds": settings.agent.max_total_seconds,
                "max_attempts": MAX_PROVIDER_ATTEMPTS,
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
                # Configured vector backend NAME (e.g. "qdrant"), a
                # descriptive label for retrieval-source displays — not a
                # reachability claim (that stays with the system-runtime
                # probes).
                "vector_backend": settings.knowledge.vector_backend,
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
