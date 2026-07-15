"""Tests for the knowledge engine: chunking, memory store, cards, service."""

from __future__ import annotations

import pytest

from inqtrix.auth.identity_memory import MemoryIdentityStore
from inqtrix.auth.permissions import AuthorizationService
from inqtrix.embedding_cards import (
    EMBEDDING_CARDS,
    build_embedding_catalog,
    resolve_embedding_card,
)
from inqtrix.knowledge.chunking import chunk_text
from inqtrix.knowledge.stores.memory import MemoryKnowledgeStore
from inqtrix.knowledge.stores.ports import (
    CollectionNotFound,
    DocumentNotFound,
    EmbeddingDimensionMismatch,
    KnowledgeProviderContext,
)
from inqtrix.services.knowledge_service import (
    KnowledgeService,
    KnowledgeValidationError,
)


class StubEmbeddings:
    """Deterministic word-bucket embeddings for network-free tests."""

    DIM = 8

    def __init__(self, *, selectable: list[str] | None = None) -> None:
        self._selectable = selectable or []
        self.document_calls = 0
        self.query_calls = 0

    @property
    def selectable_embedding_models(self) -> list[str]:
        return list(self._selectable)

    @property
    def default_model(self) -> str:
        return "stub-embed-8"

    def _vector(self, text: str) -> list[float]:
        # zlib.crc32 instead of hash(): the builtin is salted per
        # process (PYTHONHASHSEED), which would make test vectors
        # non-deterministic across runs.
        import zlib

        vector = [0.0] * self.DIM
        for word in text.lower().split():
            vector[zlib.crc32(word.encode("utf-8")) % self.DIM] += 1.0
        return vector

    def embed_documents(self, texts, *, model=None):
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text, *, model=None):
        self.query_calls += 1
        return self._vector(text)


def make_knowledge_context(**kwargs) -> KnowledgeProviderContext:
    return KnowledgeProviderContext(
        embeddings=kwargs.pop("embeddings", StubEmbeddings()),
        store=kwargs.pop("store", MemoryKnowledgeStore()),
        **kwargs,
    )


def make_service(context: KnowledgeProviderContext | None = None) -> KnowledgeService:
    identity = MemoryIdentityStore()
    return KnowledgeService(
        knowledge=context or make_knowledge_context(),
        authorization=AuthorizationService(
            members=identity,
            shares=identity,
            audit=identity,
        ),
        chunk_max_chars=2_000,
        max_document_chars=100_000,
    )


# ------------------------------------------------------------------ #
# Chunking
# ------------------------------------------------------------------ #


def test_chunker_keeps_short_text_whole():
    assert chunk_text("Ein kurzer Absatz.") == ["Ein kurzer Absatz."]


def test_chunker_packs_paragraphs_up_to_budget():
    text = "Absatz eins.\n\nAbsatz zwei.\n\nAbsatz drei."
    assert chunk_text(text, max_chars=30) == [
        "Absatz eins.\n\nAbsatz zwei.",
        "Absatz drei.",
    ]


def test_chunker_splits_oversize_paragraph_on_sentences():
    text = "Erster Satz hier. Zweiter Satz hier. Dritter Satz hier."
    chunks = chunk_text(text, max_chars=25)
    assert chunks == ["Erster Satz hier.", "Zweiter Satz hier.", "Dritter Satz hier."]


def test_chunker_hard_wraps_single_oversize_sentence():
    chunks = chunk_text("x" * 50, max_chars=20)
    assert chunks == ["x" * 20, "x" * 20, "x" * 10]


def test_chunker_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="positive"):
        chunk_text("text", max_chars=0)


def test_chunker_empty_input_yields_no_chunks():
    assert chunk_text("   \n\n  ") == []


# ------------------------------------------------------------------ #
# Memory store
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_collection_lifecycle_and_document_count():
    store = MemoryKnowledgeStore()
    collection = await store.create_collection(
        name="Vertraege", embedding_model="stub-embed-8", embedding_dim=8
    )
    assert collection.embedding_model == "stub-embed-8"

    await store.add_document(
        collection_id=collection.id,
        title="Rahmenvertrag",
        text="Inhalt",
        metadata={},
        chunks=["Inhalt"],
        embeddings=[[1.0] * 8],
        actor_user_id=None,
    )
    assert (await store.get_collection(collection.id)).document_count == 1

    documents = await store.list_documents(collection.id)
    assert [doc.title for doc in documents] == ["Rahmenvertrag"]

    await store.delete_document(documents[0].id, actor_user_id=None)
    assert (await store.get_collection(collection.id)).document_count == 0

    await store.delete_collection(collection.id, actor_user_id=None)
    with pytest.raises(CollectionNotFound):
        await store.get_collection(collection.id)


@pytest.mark.asyncio
async def test_dimension_mismatch_is_rejected_loudly():
    store = MemoryKnowledgeStore()
    collection = await store.create_collection(
        name="K", embedding_model="stub-embed-8", embedding_dim=8
    )
    with pytest.raises(EmbeddingDimensionMismatch, match="dimension 4"):
        await store.add_document(
            collection_id=collection.id,
            title="Falsch",
            text="x",
            metadata={},
            chunks=["x"],
            embeddings=[[1.0] * 4],
            actor_user_id=None,
        )


@pytest.mark.asyncio
async def test_chunk_embedding_count_mismatch_is_rejected():
    store = MemoryKnowledgeStore()
    collection = await store.create_collection(
        name="K", embedding_model="stub-embed-8", embedding_dim=8
    )
    with pytest.raises(EmbeddingDimensionMismatch, match="count mismatch"):
        await store.add_document(
            collection_id=collection.id,
            title="Falsch",
            text="x",
            metadata={},
            chunks=["a", "b"],
            embeddings=[[1.0] * 8],
            actor_user_id=None,
        )


@pytest.mark.asyncio
async def test_search_ranks_matching_document_first_and_scopes_collections():
    embeddings = StubEmbeddings()
    store = MemoryKnowledgeStore()
    service = make_service(make_knowledge_context(embeddings=embeddings, store=store))

    legal = await service.create_collection(name="Recht")
    tech = await service.create_collection(name="Technik")
    await service.add_document(
        collection_id=legal.id,
        title="Haftungsvertrag",
        text="Die Haftung ist auf den Auftragswert begrenzt.",
    )
    await service.add_document(
        collection_id=tech.id,
        title="Serverhandbuch",
        text="Der Server startet mit systemctl start inqtrix.",
    )

    hits = await service.search(query="Haftung Auftragswert begrenzt")
    assert hits[0].document_title == "Haftungsvertrag"

    scoped = await service.search(
        query="Haftung Auftragswert begrenzt", collection_ids=[tech.id]
    )
    assert all(hit.document_title == "Serverhandbuch" for hit in scoped)


@pytest.mark.asyncio
async def test_search_with_unknown_collection_raises():
    service = make_service()
    with pytest.raises(CollectionNotFound):
        await service.search(query="x", collection_ids=["kc_unknown"])


@pytest.mark.asyncio
async def test_reembed_preserves_chunk_ids_and_citation_key():
    """Reindex must not orphan citations: chunk ids AND the
    ``(document_id, chunk_index)`` citation key survive a re-embed."""
    embeddings = StubEmbeddings()
    store = MemoryKnowledgeStore()
    service = make_service(make_knowledge_context(embeddings=embeddings, store=store))

    collection = await service.create_collection(name="Recht")
    document = await service.add_document(
        collection_id=collection.id,
        title="Haftungsvertrag",
        text="Die Haftung ist begrenzt.\n\nDie Frist betraegt 24 Stunden.",
    )

    before = await store.get_chunks(document.id)
    assert before, "fixture must produce at least one chunk"
    ids_before = [(chunk.chunk_index, chunk.id) for chunk in before]

    await service.reembed_document(
        document=document,
        embedding_model=collection.embedding_model,
        actor_user_id=None,
    )

    after = await store.get_chunks(document.id)
    ids_after = [(chunk.chunk_index, chunk.id) for chunk in after]
    assert ids_after == ids_before
    # The read surface carries no vectors (lockstep with Postgres).
    assert all(chunk.embedding == () for chunk in after)


@pytest.mark.asyncio
async def test_get_chunks_orders_by_index_and_404s():
    """``get_chunks`` returns chunk_index order regardless of physical
    order, and raises for an unknown document."""
    store = MemoryKnowledgeStore()
    service = make_service(make_knowledge_context(store=store))
    collection = await service.create_collection(name="Recht")
    document = await service.add_document(
        collection_id=collection.id,
        title="D",
        text="Erster Absatz.\n\nZweiter Absatz.\n\nDritter Absatz.",
    )
    # Physically reverse the stored chunk list; get_chunks must still
    # return chunk_index order (0,1,2), not physical order.
    store._chunks[document.id] = list(reversed(store._chunks[document.id]))
    chunks = await store.get_chunks(document.id)
    assert [chunk.chunk_index for chunk in chunks] == sorted(
        chunk.chunk_index for chunk in chunks
    )
    with pytest.raises(DocumentNotFound):
        await store.get_chunks("kd_unknown")


@pytest.mark.asyncio
async def test_delete_unknown_document_raises():
    store = MemoryKnowledgeStore()
    with pytest.raises(DocumentNotFound):
        await store.delete_document("kd_unknown", actor_user_id=None)


@pytest.mark.asyncio
async def test_memory_vector_index_scrolls_distinct_groups_scoped_to_model():
    """scroll_chunk_groups returns the distinct (collection, document) groups
    for one model — the reverse-reconcile input. Chunks of one doc collapse to
    a single group; other models are excluded."""
    from inqtrix.knowledge.stores.vector_index import ChunkVector, MemoryVectorIndex

    index = MemoryVectorIndex()
    await index.upsert(
        embedding_model="m", collection_id="kc1", document_id="kd1",
        vectors=[ChunkVector("ch1", (1.0,)), ChunkVector("ch2", (0.0,))],
    )
    await index.upsert(
        embedding_model="m", collection_id="kc1", document_id="kd2",
        vectors=[ChunkVector("ch3", (1.0,))],
    )
    await index.upsert(
        embedding_model="other", collection_id="kc9", document_id="kd9",
        vectors=[ChunkVector("ch9", (1.0,))],
    )
    groups = await index.scroll_chunk_groups(embedding_model="m")
    assert groups == {("kc1", "kd1"), ("kc1", "kd2")}


# ------------------------------------------------------------------ #
# Service validation
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_service_rejects_empty_fields():
    service = make_service()
    with pytest.raises(KnowledgeValidationError, match="'name'"):
        await service.create_collection(name="  ")
    collection = await service.create_collection(name="K")
    with pytest.raises(KnowledgeValidationError, match="'title'"):
        await service.add_document(collection_id=collection.id, title=" ", text="x")
    with pytest.raises(KnowledgeValidationError, match="'text'"):
        await service.add_document(collection_id=collection.id, title="T", text=" ")
    with pytest.raises(KnowledgeValidationError, match="'query'"):
        await service.search(query="  ")


@pytest.mark.asyncio
async def test_service_enforces_document_size_guard():
    identity = MemoryIdentityStore()
    service = KnowledgeService(
        knowledge=make_knowledge_context(),
        authorization=AuthorizationService(
            members=identity,
            shares=identity,
            audit=identity,
        ),
        chunk_max_chars=2_000,
        max_document_chars=100,
    )
    collection = await service.create_collection(name="K")
    with pytest.raises(KnowledgeValidationError, match="zu gross"):
        await service.add_document(
            collection_id=collection.id, title="T", text="x" * 200
        )


@pytest.mark.asyncio
async def test_service_rejects_unselectable_embedding_model():
    context = make_knowledge_context(
        embeddings=StubEmbeddings(selectable=["allowed-model"]),
    )
    service = make_service(context)
    with pytest.raises(KnowledgeValidationError, match="nicht verfuegbar"):
        await service.create_collection(name="K", embedding_model="forbidden-model")


@pytest.mark.asyncio
async def test_service_probes_dimension_for_uncatalogued_model():
    service = make_service()
    collection = await service.create_collection(name="K")
    assert collection.embedding_dim == StubEmbeddings.DIM


# ------------------------------------------------------------------ #
# Embedding cards
# ------------------------------------------------------------------ #


def test_catalogued_models_resolve_with_dims():
    card = resolve_embedding_card("BAAI/bge-m3")
    assert card is not None
    assert card.dims == 1024
    assert card.multilingual is True


def test_unknown_model_degrades_to_none_card():
    catalog = build_embedding_catalog(["text-embedding-3-large", "custom/model"])
    assert catalog[0]["card"]["dims"] == 3072
    assert catalog[1] == {"model_id": "custom/model", "card": None}


def test_every_card_has_positive_dims_and_source():
    for card in EMBEDDING_CARDS:
        assert card.dims > 0
        assert card.max_input_tokens > 0
        assert card.source_url.startswith("https://")
