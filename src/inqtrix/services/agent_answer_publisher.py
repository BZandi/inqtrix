"""Atomic persistence and event publication for native agent answers.

The agent algorithms produce answer Markdown and evidence references, but they
do not own the transport lifecycle.  This publisher is the single boundary
that keeps the durable ``answer`` artifact and the run event stream in one
observable order:

1. persist an empty ``writing`` artifact;
2. signal the committed artifact revision;
3. emit the answer start and text deltas;
4. CAS-finalize the same artifact with the exported references;
5. signal the committed revision and emit ``answer.ready``.

The durable worker's claim attempt is carried into both artifact writes.  A
reclaimed worker can therefore neither restart nor finalize an answer after a
new attempt owns the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inqtrix.agents.control_ports import (
    ArtifactNotFound,
    ArtifactPublicationFenced,
    ArtifactRevisionConflict,
    artifact_event_payload,
)
from inqtrix.exceptions import AgentCancelled
from inqtrix.execution_failures import RunExecutionFailure
from inqtrix.runs.shared import answer_artifact_id, answer_publication_id
from inqtrix.sync_bridge import run_coro_sync

if TYPE_CHECKING:
    from inqtrix.agents.control_ports import AgentControlStore, ArtifactRecord
    from inqtrix.server.runs import RunHandle


ARTIFACT_CREATED_EVENT = "inqtrix.agent.artifact.created"
ARTIFACT_UPDATED_EVENT = "inqtrix.agent.artifact.updated"


class AgentAnswerPublisher:
    """Publish one agent answer through its durable artifact and event log."""

    def __init__(self, store: "AgentControlStore") -> None:
        self._store = store

    def publish(
        self,
        handle: "RunHandle",
        answer: str,
        *,
        references: list[dict[str, Any]],
        question: str = "",
    ) -> None:
        """Persist and stream one final answer without pre-stream visibility.

        ``references`` is the exact list already exported in the native run
        result.  It is copied only to detach caller-owned containers; no
        filtering, re-ranking, or second citation interpretation happens here.
        """
        artifact_id = answer_artifact_id(handle.run_id)
        publication_id = answer_publication_id(handle.run_id)
        refs = self._copy_references(references)
        # Citation validation belongs to the producing algorithm because it
        # has the complete task context.  This shared transport boundary must
        # never reinterpret or discard an already completed answer.
        _ = question
        try:
            staged = self._stage(
                handle,
                artifact_id=artifact_id,
                publication_id=publication_id,
            )
        except (ArtifactPublicationFenced, ArtifactRevisionConflict) as exc:
            # No start event has been emitted yet, so make the failed
            # publication attempt visible explicitly instead of silently
            # completing with a result but no answer artifact.
            handle.emit(
                "inqtrix.answer.interrupted",
                {
                    "artifact_id": artifact_id,
                    "publication_id": publication_id,
                    "offset": 0,
                    "status": "interrupted",
                    "stage": "staging",
                },
            )
            raise self._typed_conflict(handle, exc) from exc

        self._emit_artifact_signal(handle, staged)

        def _finalize() -> None:
            finalized = run_coro_sync(
                self._store.upsert_artifact(
                    run_id=handle.run_id,
                    kind="answer",
                    session_id=None,
                    title="Antwort",
                    status="ready",
                    content_markdown=answer,
                    payload={"publication_id": publication_id},
                    refs=refs,
                    updated_by="agent",
                    artifact_id=artifact_id,
                    expected_revision=staged.revision,
                    expected_run_attempt=handle.publication_fence_attempt,
                )
            )
            self._emit_artifact_signal(handle, finalized)

        try:
            handle.emit_answer(
                answer,
                reference_labels=[
                    label
                    for ref in refs
                    if (label := str(ref.get("label", "") or "").strip())
                ],
                before_ready=_finalize,
            )
        except (ArtifactPublicationFenced, ArtifactRevisionConflict) as exc:
            # RunHandle has already emitted ``answer.interrupted`` with the
            # exact byte offset and the ``finalizing`` stage.
            raise self._typed_conflict(handle, exc) from exc

    def _stage(
        self,
        handle: "RunHandle",
        *,
        artifact_id: str,
        publication_id: str,
    ) -> "ArtifactRecord":
        try:
            current, _revisions = run_coro_sync(
                self._store.get_artifact(handle.run_id, artifact_id)
            )
            expected_revision = current.revision
        except ArtifactNotFound:
            expected_revision = 0
        return run_coro_sync(
            self._store.upsert_artifact(
                run_id=handle.run_id,
                kind="answer",
                session_id=None,
                title="Antwort",
                status="writing",
                content_markdown="",
                payload={"publication_id": publication_id},
                refs=[],
                updated_by="agent",
                artifact_id=artifact_id,
                expected_revision=expected_revision,
                expected_run_attempt=handle.publication_fence_attempt,
            )
        )

    @staticmethod
    def _copy_references(
        references: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not all(isinstance(reference, dict) for reference in references):
            raise RunExecutionFailure(
                "answer_publication_invalid_references",
                "Die exportierten Antwortreferenzen sind nicht serialisierbar.",
            )
        return [dict(reference) for reference in references]

    @staticmethod
    def _emit_artifact_signal(
        handle: "RunHandle", artifact: "ArtifactRecord"
    ) -> None:
        handle.emit(
            (
                ARTIFACT_CREATED_EVENT
                if artifact.revision == 1
                else ARTIFACT_UPDATED_EVENT
            ),
            artifact_event_payload(artifact),
        )

    @staticmethod
    def _typed_conflict(
        handle: "RunHandle",
        exc: ArtifactPublicationFenced | ArtifactRevisionConflict,
    ) -> Exception:
        if isinstance(exc, ArtifactPublicationFenced):
            if (
                handle.cancel_event.is_set()
                or exc.status == "cancel_requested"
            ):
                # The fence fired because a requested cancel superseded
                # this attempt mid-publication. ``exc.status`` is the
                # store's answer from the same transaction that refused
                # the write, so this classification does not depend on
                # the cancel poller having already set the local event.
                # Reporting the race as a failure would contradict the
                # user's own action — the honest terminal is cancelled.
                return AgentCancelled(
                    "Die Antwort-Publikation wurde durch den "
                    "angeforderten Abbruch beendet."
                )
            return RunExecutionFailure(
                "answer_publication_fenced",
                "Die Publikation war nicht mehr moeglich, weil der Lauf "
                "diesem Ausfuehrungsversuch nicht mehr gehoert.",
            )
        return RunExecutionFailure(
            "answer_publication_conflict",
            "Die Antwort konnte wegen einer konkurrierenden "
            "Artefaktrevision nicht publiziert werden.",
        )
