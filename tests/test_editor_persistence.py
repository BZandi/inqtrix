"""Behavior tests for the editor-persistence service (M6b).

Runs against the in-memory tier but asserts the contract both tiers honour:
owner isolation, the document-body lazy-load (list excludes the body,
get includes it), the idempotent autosave upsert, comment composite-PK
isolation + delete, the folder orphan rule, and payload validation.
"""

from __future__ import annotations

import pytest

from inqtrix.auth.principal import Principal, UserContext
from inqtrix.pagination import decode_cursor
from inqtrix.project.editor_memory import MemoryEditorStore
from inqtrix.project.editor_ports import DocumentNotFound, FolderNotFound
from inqtrix.services.editor_persistence_service import (
    EditorPersistenceService,
    EditorValidationError,
)


def _scoped(sub: str) -> UserContext:
    return UserContext(
        principal=Principal(
            sub=sub, kind="oidc_session", tenant_id="default", role="member"
        )
    )


@pytest.fixture()
def service() -> EditorPersistenceService:
    return EditorPersistenceService(store=MemoryEditorStore(), durable=False)


async def _save_doc(
    service: EditorPersistenceService,
    *,
    document_id: str,
    caller_sub: str | None,
    created_at: float,
    title: str = "Doc",
    body: str = "body text",
    folder_id: str | None = None,
) -> None:
    await service.save_document(
        id=document_id,
        title=title,
        content_markdown=body,
        folder_id=folder_id,
        source="blank",
        source_run_id=None,
        revision=1,
        diff_anchor_markdown=None,
        diff_anchor_updated_at=None,
        created_at=created_at,
        updated_at=created_at,
        caller_sub=caller_sub,
        workspace_id=None,
        visible_to=_scoped(caller_sub) if caller_sub else None,
    )


def _comment(comment_id: str, *, created_at: float, status: str = "open") -> dict:
    return {
        "id": comment_id,
        "comment_markdown": f"c-{comment_id}",
        "anchor": {"from": 0, "to": 5, "selectedText": "hello", "quoteBefore": "", "quoteAfter": ""},
        "kind": "collect",
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
    }


@pytest.mark.asyncio
async def test_document_body_is_lazy_listed_but_loaded_on_get(service) -> None:
    """list_documents returns metadata only (no body); get_document loads it."""
    await _save_doc(
        service, document_id="ed_1", caller_sub="u", created_at=1.0,
        body="the heavy body",
    )
    page, _ = await service.list_documents(
        caller_sub="u", workspace_id=None, limit=50, after=None
    )
    assert len(page) == 1
    assert page[0].content_markdown == ""  # body excluded from the list
    full = await service.get_document("ed_1", visible_to=_scoped("u"))
    assert full.content_markdown == "the heavy body"


@pytest.mark.asyncio
async def test_document_keyset_walks_with_tiebreaker(service) -> None:
    stamps = [10.0, 10.0, 20.0, 30.0, 30.0]
    for n, stamp in enumerate(stamps):
        await _save_doc(service, document_id=f"ed_{n}", caller_sub="u", created_at=stamp)
    seen: list[str] = []
    cursor = None
    for _ in range(10):
        page, next_cursor = await service.list_documents(
            caller_sub="u", workspace_id=None, limit=2, after=cursor
        )
        seen.extend(d.id for d in page)
        if next_cursor is None:
            break
        cursor = decode_cursor(next_cursor)
    assert len(seen) == len(set(seen)) == 5


@pytest.mark.asyncio
async def test_document_upsert_preserves_owner_and_created_at(service) -> None:
    await _save_doc(service, document_id="ed_1", caller_sub="u-a", created_at=100.0, title="first")
    # base+1 (stored is 1) — a normal next save; created_at/owner must survive
    # even though this save passes a different created_at.
    await service.save_document(
        id="ed_1", title="second", content_markdown="new body", folder_id=None,
        source="pasted", source_run_id=None, revision=2,
        diff_anchor_markdown=None, diff_anchor_updated_at=None,
        created_at=999.0, updated_at=200.0,
        caller_sub="u-a", workspace_id=None, visible_to=_scoped("u-a"),
    )
    doc = await service.get_document("ed_1", visible_to=_scoped("u-a"))
    assert doc.title == "second"
    assert doc.content_markdown == "new body"
    assert doc.revision == 2
    assert doc.created_at == 100.0  # creation time stable
    assert doc.created_by_sub == "u-a"


@pytest.mark.asyncio
async def test_scoped_owner_isolation(service) -> None:
    await _save_doc(service, document_id="ed_a", caller_sub="u-a", created_at=1.0)
    with pytest.raises(DocumentNotFound):
        await service.get_document("ed_a", visible_to=_scoped("u-b"))
    page, _ = await service.list_documents(
        caller_sub="u-b", workspace_id=None, limit=50, after=None
    )
    assert page == []
    with pytest.raises(DocumentNotFound):
        await service.delete_document("ed_a", visible_to=_scoped("u-b"))


@pytest.mark.asyncio
async def test_comments_upsert_list_and_cross_document_isolation(service) -> None:
    await _save_doc(service, document_id="ed_a", caller_sub="u", created_at=1.0)
    await _save_doc(service, document_id="ed_b", caller_sub="u", created_at=1.0)
    await service.save_comments(
        "ed_a",
        comments=[_comment("edc_dup", created_at=1.0)],
        visible_to=_scoped("u"),
    )
    # Same comment id in a DIFFERENT document must be a distinct row.
    await service.save_comments(
        "ed_b",
        comments=[{**_comment("edc_dup", created_at=1.0), "comment_markdown": "B-text"}],
        visible_to=_scoped("u"),
    )
    a_page, _ = await service.list_comments(
        "ed_a", limit=50, after=None, visible_to=_scoped("u")
    )
    assert a_page[0].comment_markdown == "c-edc_dup"  # NOT clobbered by B
    b_page, _ = await service.list_comments(
        "ed_b", limit=50, after=None, visible_to=_scoped("u")
    )
    assert b_page[0].comment_markdown == "B-text"


@pytest.mark.asyncio
async def test_comment_delete_and_document_cascade(service) -> None:
    await _save_doc(service, document_id="ed_a", caller_sub="u", created_at=1.0)
    await service.save_comments(
        "ed_a", comments=[_comment("edc_1", created_at=1.0)], visible_to=_scoped("u")
    )
    await service.delete_comment("ed_a", "edc_1", visible_to=_scoped("u"))
    gone, _ = await service.list_comments(
        "ed_a", limit=50, after=None, visible_to=_scoped("u")
    )
    assert gone == []
    # Re-add then delete the document -> comments cascade.
    await service.save_comments(
        "ed_a", comments=[_comment("edc_2", created_at=2.0)], visible_to=_scoped("u")
    )
    await service.delete_document("ed_a", visible_to=_scoped("u"))
    with pytest.raises(DocumentNotFound):
        await service.get_document("ed_a", visible_to=_scoped("u"))


@pytest.mark.asyncio
async def test_folder_delete_orphans_documents(service) -> None:
    await service.save_folder(
        id="edf_1", title="F", created_at=1.0, updated_at=1.0,
        caller_sub="u", workspace_id=None, visible_to=_scoped("u"),
    )
    await _save_doc(
        service, document_id="ed_1", caller_sub="u", created_at=2.0, folder_id="edf_1"
    )
    await service.delete_folder("edf_1", visible_to=_scoped("u"))
    doc = await service.get_document("ed_1", visible_to=_scoped("u"))
    assert doc.folder_id is None
    with pytest.raises(FolderNotFound):
        await service.delete_folder("edf_1", visible_to=_scoped("u"))


@pytest.mark.asyncio
async def test_invalid_source_kind_status_rejected(service) -> None:
    with pytest.raises(EditorValidationError):
        await service.save_document(
            id="ed_1", title="D", content_markdown="", folder_id=None,
            source="bogus", source_run_id=None, revision=1,
            diff_anchor_markdown=None, diff_anchor_updated_at=None,
            created_at=1.0, updated_at=1.0,
            caller_sub="u", workspace_id=None, visible_to=_scoped("u"),
        )
    await _save_doc(service, document_id="ed_2", caller_sub="u", created_at=1.0)
    with pytest.raises(EditorValidationError):
        await service.save_comments(
            "ed_2",
            comments=[{**_comment("edc_1", created_at=1.0), "kind": "nope"}],
            visible_to=_scoped("u"),
        )
    with pytest.raises(EditorValidationError):
        await service.save_comments(
            "ed_2",
            comments=[{**_comment("edc_2", created_at=1.0), "status": "weird"}],
            visible_to=_scoped("u"),
        )


# -- A2: revision CAS (stored == base) ---------------------------------------- #


@pytest.mark.asyncio
async def test_revision_cas_accepts_base_plus_one_rejects_stale_base(service) -> None:
    """A save writes only when the stored revision is EXACTLY its base.

    `revision` is base+1 (the client tracks its last-synced server revision
    as the base). A stale writer — one whose base is behind the server
    because it never saw a concurrent agent patch or peer edit — fails the
    CAS and gets a 409 to rebase, instead of silently clobbering. The
    forward-jump that a monotonic guard used to wave through (the P1
    data-loss shape) is exactly what must conflict now.
    """
    from inqtrix.project.editor_ports import DocumentRevisionConflict

    await _save_doc(service, document_id="ed_cas", caller_sub="u", created_at=1.0)

    async def save(revision: int, body: str) -> None:
        await service.save_document(
            id="ed_cas", title="Doc", content_markdown=body, folder_id=None,
            source="blank", source_run_id=None, revision=revision,
            diff_anchor_markdown=None, diff_anchor_updated_at=None,
            created_at=1.0, updated_at=2.0,
            caller_sub="u", workspace_id=None, visible_to=_scoped("u"),
        )

    # Forward jump: stored is 1, this writer's base is 4 (revision 5) — it
    # never synced the current state. A monotonic guard accepted this; the CAS
    # rejects it (the P1 fix). Content untouched.
    with pytest.raises(DocumentRevisionConflict) as excinfo:
        await save(5, "stale writer with a higher counter")
    assert excinfo.value.current_revision == 1
    doc = await service.get_document("ed_cas", visible_to=_scoped("u"))
    assert doc.content_markdown == "body text"
    assert doc.revision == 1

    # base+1 (base == stored == 1): accepted.
    await save(2, "proper next revision")
    doc = await service.get_document("ed_cas", visible_to=_scoped("u"))
    assert doc.revision == 2
    assert doc.content_markdown == "proper next revision"

    # Same-base double write (both based on 1, one already won): rejected.
    with pytest.raises(DocumentRevisionConflict) as excinfo:
        await save(2, "same-base clobber")
    assert excinfo.value.current_revision == 2

    # Rewind (stale writer with a lower base): rejected.
    with pytest.raises(DocumentRevisionConflict):
        await save(1, "stale rewind")

    doc = await service.get_document("ed_cas", visible_to=_scoped("u"))
    assert doc.content_markdown == "proper next revision"
    assert doc.revision == 2
