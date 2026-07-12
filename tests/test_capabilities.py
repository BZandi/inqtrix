"""Capability layer: contract, registry, and wave-1 catalog.

The capabilities wrap existing services with a typed envelope and an
injected :class:`CapabilityContext`. These tests pin (1) the registry
behaviour (ordered, loud on duplicates, input validation), (2) that
each capability returns the SAME data as calling its service directly
(the "one implementation, three adapters" guarantee), and (3) the
strict-collections semantics for the agent search path.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import threading

import pytest

from inqtrix.capabilities import (
    CapabilityContext,
    CapabilityError,
    build_capability_registry,
)
from inqtrix.capabilities.contracts import (
    CapabilityDefinition,
    Effect,
)
from inqtrix.capabilities.registry import CapabilityRegistry, UnknownCapability
from inqtrix.auth.principal import Principal, UserContext
from inqtrix.knowledge.stores.memory import MemoryKnowledgeStore
from inqtrix.knowledge.stores.ports import KnowledgeProviderContext
from inqtrix.search_result import GroundedSearchResult, GroundedSource
from inqtrix.services.knowledge_service import KnowledgeService
from pydantic import BaseModel

from tests.test_knowledge_engine import StubEmbeddings


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #


def make_knowledge_service() -> KnowledgeService:
    return KnowledgeService(
        knowledge=KnowledgeProviderContext(
            embeddings=StubEmbeddings(),
            store=MemoryKnowledgeStore(),
            default_top_k=4,
        ),
        chunk_max_chars=2_000,
        max_document_chars=100_000,
    )


class _StubSearch:
    """Search provider stub returning two grounded sources."""

    def search(self, query, **kwargs):
        self.last_kwargs = kwargs
        return GroundedSearchResult(
            answer=f"Antwort zu {query}",
            sources=[
                GroundedSource(url="https://a.example", title="A", snippet="s1", rank=1),
                GroundedSource(url="https://b.example", title="B", snippet="s2", rank=2),
            ],
            prompt_tokens=11,
            completion_tokens=7,
        )

    def is_available(self) -> bool:
        return True


class _FallbackSearch(_StubSearch):
    """Provider-style fallback exposed from the same search thread."""

    def search(self, query: str, **kwargs: object) -> GroundedSearchResult:
        del query, kwargs
        self._search_thread = threading.get_ident()
        return GroundedSearchResult()

    def consume_nonfatal_notice_detail(self) -> dict[str, object]:
        assert threading.get_ident() == self._search_thread
        return {
            "code": "upstream_5xx",
            "message": "Upstream search unavailable.",
            "http_status": 503,
        }


class _ProviderTimeoutFallbackSearch(_FallbackSearch):
    def consume_nonfatal_notice_detail(self) -> dict[str, object]:
        assert threading.get_ident() == self._search_thread
        return {
            "code": "provider_timeout",
            "message": "Upstream search timed out.",
            "http_status": 504,
        }


_ANON_PRINCIPAL = Principal(sub="__anonymous__", kind="anonymous")
# Unscoped context: visible_to=None means "no membership filtering",
# exactly what an anonymous/static principal resolves to.
ANON = CapabilityContext(principal=_ANON_PRINCIPAL)


def scoped_context(sub: str) -> CapabilityContext:
    """A scoped (oidc-session) caller with an empty membership set —
    sees only what it owns or was granted (nothing here)."""
    principal = Principal(sub=sub, kind="oidc_session")
    return CapabilityContext(
        principal=principal, visible_to=UserContext(principal=principal)
    )


# ------------------------------------------------------------------ #
# registry
# ------------------------------------------------------------------ #


def _dummy(effect=Effect.READ) -> CapabilityDefinition:
    class _In(BaseModel):
        x: int

    class _Out(BaseModel):
        y: int

    async def _h(payload, _ctx):
        return _Out(y=payload.x + 1)

    return CapabilityDefinition(
        id="dummy.op",
        summary="s",
        input_model=_In,
        output_model=_Out,
        effect=effect,
        idempotent=True,
        handler=_h,
    )


def test_registry_is_ordered_and_loud_on_duplicates():
    registry = CapabilityRegistry()
    registry.register(_dummy())
    assert registry.ids() == ("dummy.op",)
    with pytest.raises(ValueError):
        registry.register(_dummy())
    with pytest.raises(UnknownCapability):
        registry.get("nope")


def test_registry_validates_input_and_maps_to_capability_error():
    registry = CapabilityRegistry()
    registry.register(_dummy())
    # Missing required field -> CapabilityError(invalid_input, 400).
    with pytest.raises(CapabilityError) as excinfo:
        asyncio.run(registry.invoke("dummy.op", {}, ANON))
    assert excinfo.value.code == "invalid_input"
    assert excinfo.value.http_status == 400
    # Valid input runs the handler.
    out = asyncio.run(registry.invoke("dummy.op", {"x": 41}, ANON))
    assert out.y == 42


def test_manifest_entry_mirrors_mcp_annotation_vocabulary():
    entry = _dummy(effect=Effect.READ).manifest_entry()
    assert entry["read_only"] is True
    assert entry["destructive"] is False
    assert entry["idempotent"] is True
    write = _dummy(effect=Effect.WRITE).manifest_entry()
    assert write["read_only"] is False


# ------------------------------------------------------------------ #
# catalog: knowledge (service-identity guarantee)
# ------------------------------------------------------------------ #


def test_knowledge_search_capability_matches_direct_service_call():
    service = make_knowledge_service()
    registry = build_capability_registry(knowledge_service=service)

    collection = asyncio.run(service.create_collection(name="Recht"))
    asyncio.run(
        service.add_document(
            collection_id=collection.id,
            title="Haftung",
            text="Die Haftung ist auf den Auftragswert begrenzt.",
        )
    )

    # Direct service call and capability invocation return the same hits.
    direct = asyncio.run(service.search(query="Haftung Auftragswert"))
    out = asyncio.run(
        registry.invoke(
            "knowledge.search", {"query": "Haftung Auftragswert"}, ANON
        )
    )
    assert [hit.document_id for hit in out.hits] == [
        candidate.chunk.document_id for candidate in direct
    ]
    assert out.hits[0].rank == 1
    assert out.hits[0].chunk_id.startswith("kch_")
    assert out.hits[0].source_text  # provenance present


def test_knowledge_document_read_capability_returns_full_text():
    service = make_knowledge_service()
    registry = build_capability_registry(knowledge_service=service)
    collection = asyncio.run(service.create_collection(name="Recht"))
    document = asyncio.run(
        service.add_document(
            collection_id=collection.id, title="D", text="Voller Text."
        )
    )
    out = asyncio.run(
        registry.invoke("knowledge.document.read", {"document_id": document.id}, ANON)
    )
    assert out.text == "Voller Text."
    assert out.id == document.id


def test_knowledge_document_read_unknown_is_capability_error_404():
    service = make_knowledge_service()
    registry = build_capability_registry(knowledge_service=service)
    with pytest.raises(CapabilityError) as excinfo:
        asyncio.run(
            registry.invoke(
                "knowledge.document.read", {"document_id": "kd_nope"}, ANON
            )
        )
    assert excinfo.value.http_status == 404


# ------------------------------------------------------------------ #
# catalog: web instant
# ------------------------------------------------------------------ #


def test_web_instant_capability_wraps_one_search_call():
    search = _StubSearch()
    registry = build_capability_registry(search_provider=search)
    out = asyncio.run(
        registry.invoke(
            "web.search.instant",
            {"query": "KI Regulierung", "recency": "week", "max_sources": 1},
            ANON,
        )
    )
    assert out.answer == "Antwort zu KI Regulierung"
    # max_sources caps the returned list; recency reached the provider.
    assert len(out.sources) == 1
    assert search.last_kwargs["recency_filter"] == "week"
    assert out.prompt_tokens == 11
    assert out.completion_tokens == 7


def test_web_instant_forwards_provider_retry_notice_with_query_context():
    notices: list[dict[str, object]] = []

    class RetryingSearch(_StubSearch):
        retry_callback = None

        @contextmanager
        def observe_retries(self, callback):
            self.retry_callback = callback
            try:
                yield self
            finally:
                self.retry_callback = None

        def search(self, query, **kwargs):
            assert self.retry_callback is not None
            self.retry_callback({
                "attempt": 1,
                "max_attempts": 3,
                "delay_seconds": 1.0,
                "error_code": "upstream_5xx",
            })
            return super().search(query, **kwargs)

    registry = build_capability_registry(search_provider=RetryingSearch())
    context = CapabilityContext(
        principal=None,
        on_provider_retry=lambda notice: notices.append(notice),
    )

    asyncio.run(
        registry.invoke(
            "web.search.instant",
            {"query": "Current market evidence"},
            context,
        )
    )

    assert notices == [
        {
            "attempt": 1,
            "max_attempts": 3,
            "delay_seconds": 1.0,
            "error_code": "upstream_5xx",
            "operation": "web.search.instant",
            "query": "Current market evidence",
        }
    ]


def test_web_instant_rejects_bad_recency():
    registry = build_capability_registry(search_provider=_StubSearch())
    with pytest.raises(CapabilityError) as excinfo:
        asyncio.run(
            registry.invoke(
                "web.search.instant",
                {"query": "x", "recency": "fortnight"},
                ANON,
            )
        )
    assert excinfo.value.code == "invalid_input"


def test_web_instant_preserves_provider_failure_instead_of_empty_evidence(
) -> None:
    registry = build_capability_registry(search_provider=_FallbackSearch())
    with pytest.raises(CapabilityError) as excinfo:
        asyncio.run(
            registry.invoke(
                "web.search.instant",
                {"query": "current evidence"},
                ANON,
            )
        )
    assert excinfo.value.code == "upstream_5xx"
    assert excinfo.value.http_status == 503


def test_web_instant_preserves_provider_timeout_code() -> None:
    registry = build_capability_registry(
        search_provider=_ProviderTimeoutFallbackSearch()
    )
    with pytest.raises(CapabilityError) as excinfo:
        asyncio.run(
            registry.invoke(
                "web.search.instant",
                {"query": "current evidence"},
                ANON,
            )
        )

    assert excinfo.value.code == "provider_timeout"
    assert excinfo.value.http_status == 504


# ------------------------------------------------------------------ #
# conditional registration
# ------------------------------------------------------------------ #


def test_registry_registers_only_wired_catalogs():
    # No services wired -> empty manifest (degrades visibly).
    assert build_capability_registry().ids() == ()
    only_web = build_capability_registry(search_provider=_StubSearch())
    assert only_web.ids() == ("web.search.instant",)


def test_knowledge_search_strict_denies_hidden_collection_for_scoped_caller():
    """Agent search is STRICT: a real collection owned by someone else,
    named explicitly by a scoped caller who cannot see it, DENIES the
    whole call (404) — where the legacy debug route would silently drop
    it and search the rest (E5). This is the guarantee wave-2 write
    capabilities lean on, so it is tested against a hidden-but-existing
    collection, not merely an absent id."""
    service = make_knowledge_service()
    registry = build_capability_registry(knowledge_service=service)
    owned = asyncio.run(
        service.create_collection(name="Privat", created_by_sub="owner-1")
    )
    asyncio.run(
        service.add_document(
            collection_id=owned.id, title="D", text="Geheim."
        )
    )
    stranger = scoped_context("stranger-2")
    with pytest.raises(CapabilityError) as excinfo:
        asyncio.run(
            registry.invoke(
                "knowledge.search",
                {"query": "Geheim", "collection_ids": [owned.id]},
                stranger,
            )
        )
    assert excinfo.value.http_status == 404
    assert excinfo.value.code == "knowledge.collection_not_found"
    # Sanity: the owner CAN search the same collection (the id is real,
    # the denial above is visibility, not absence).
    owner_out = asyncio.run(
        registry.invoke(
            "knowledge.search",
            {"query": "Geheim", "collection_ids": [owned.id]},
            scoped_context("owner-1"),
        )
    )
    assert owner_out.hits


def test_editor_document_read_capability_bundles_doc_and_comments():
    from inqtrix.project.editor_memory import MemoryEditorStore
    from inqtrix.services.editor_persistence_service import (
        EditorPersistenceService,
    )

    service = EditorPersistenceService(store=MemoryEditorStore(), durable=False)
    registry = build_capability_registry(editor_service=service)
    asyncio.run(
        service.save_document(
            id="ed_1",
            title="Vermerk",
            content_markdown="Erster Absatz.",
            folder_id=None,
            source="blank",
            source_run_id=None,
            revision=3,
            diff_anchor_markdown=None,
            diff_anchor_updated_at=None,
            created_at=1.0,
            updated_at=1.0,
            caller_sub=None,
            workspace_id=None,
            visible_to=None,
        )
    )
    out = asyncio.run(
        registry.invoke("editor.document.read", {"document_id": "ed_1"}, ANON)
    )
    assert out.title == "Vermerk"
    assert out.content_markdown == "Erster Absatz."
    assert out.revision == 3
    assert out.comments == []
