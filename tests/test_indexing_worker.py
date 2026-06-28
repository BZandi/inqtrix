"""Offline tests for the reindex worker's fenced job handle.

The generic claim/ack/fence/reclaim machinery is exercised by
``test_worker_loop.py`` (the run loop inherits the same
:class:`~inqtrix.worker.loop.BaseWorkerLoop`); this suite pins the
reindex-specific piece — that the fenced handle threads its claim
attempt into every store write and records whether the terminal write
actually landed (the ack gate).
"""

from __future__ import annotations

import threading

from inqtrix.worker.indexing_loop import FencedIndexingJobHandle


class StubStore:
    """Recording stand-in for the durable store's handle surface."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.terminal_lands = True

    def set_total(self, job_id, total_documents, *, fence_attempt=None):
        self.calls.append(("set_total", total_documents, fence_attempt))

    def progress(
        self,
        job_id,
        *,
        completed_documents,
        current_document_title="",
        fence_attempt=None,
    ):
        self.calls.append(("progress", completed_documents, fence_attempt))

    def complete(self, job_id, *, fence_attempt=None):
        self.calls.append(("complete", fence_attempt))
        return self.terminal_lands

    def fail(self, job_id, message, *, error_type="server_error", fence_attempt=None):
        self.calls.append(("fail", message, error_type, fence_attempt))
        return self.terminal_lands

    def mark_cancelled(self, job_id, *, reason, fence_attempt=None):
        self.calls.append(("cancelled", reason, fence_attempt))
        return self.terminal_lands

    def document_completed(self, job_id, document_id, *, fence_attempt=None):
        self.calls.append(("document_completed", document_id, fence_attempt))


def test_fenced_handle_threads_the_attempt_into_every_write():
    store = StubStore()
    handle = FencedIndexingJobHandle(store, "ix_1", threading.Event(), attempt=7)

    handle.begin(5)
    handle.progress(completed_documents=2, current_document_title="doc")
    handle.document_completed("kd_1")
    handle.complete()
    handle.fail("kaputt")
    handle.cancel("client_requested_cancel")

    assert ("set_total", 5, 7) in store.calls
    assert ("progress", 2, 7) in store.calls
    assert ("document_completed", "kd_1", 7) in store.calls
    assert ("complete", 7) in store.calls
    assert any(c[0] == "fail" and c[3] == 7 for c in store.calls)
    assert ("cancelled", "client_requested_cancel", 7) in store.calls
    assert handle.terminal_landed is True


def test_fenced_out_terminal_write_records_no_landing():
    """A superseded attempt's terminal write returns False — the worker
    must then NOT ack, so terminal_landed must reflect that."""
    store = StubStore()
    store.terminal_lands = False
    handle = FencedIndexingJobHandle(store, "ix_1", threading.Event(), attempt=2)

    handle.complete()

    assert handle.terminal_landed is False
