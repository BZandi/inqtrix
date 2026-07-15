"""Narrow authenticated HTTP client for the private collaboration service."""

from __future__ import annotations

import base64
import binascii
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import httpx

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
log = logging.getLogger("inqtrix")


class CollaborationServiceUnavailable(RuntimeError):
    """Raised when the optional Node service is unreachable or unready."""


class CollaborationNodeConflict(RuntimeError):
    """Raised when Node rejects a schema, sequence, or decision contract."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CollaborationConversion:
    """Side-effect-free conversion result used by the activation transaction."""

    schema_hash: str
    state_update: bytes
    state_vector: bytes
    state_hash: str
    projection_markdown: str
    projection_hash: str


@dataclass(frozen=True)
class CollaborationProjection:
    """Canonical Markdown projection after Node drains a document queue."""

    generation: int
    sequence: int
    markdown: str
    projection_hash: str
    schema_hash: str
    authoritative_sequence: int | None = None
    state_update: bytes | None = None
    state_vector: bytes | None = None
    state_hash: str | None = None


@dataclass(frozen=True)
class CollaborationDecisionResult:
    """Durable result of an accept/reject command executed by Node."""

    command_id: uuid.UUID
    sequence: int
    suggestion_ids: tuple[str, ...]


@dataclass(frozen=True)
class CollaborationSuggestionResult:
    """Durable result of publishing one private AI patch into Yjs."""

    command_id: uuid.UUID
    patch_id: str
    sequence: int
    suggestion_ids: tuple[str, ...]


class CollaborationNodeClient:
    """Constructor-configured client; secrets never enter URLs or logs."""

    def __init__(
        self,
        *,
        base_url: str,
        secret: str,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {secret}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    async def available(self) -> bool:
        """Whether Node currently reports its fenced instance as ready."""
        try:
            response = await self._client.get("/health/ready")
        except httpx.HTTPError:
            log.warning("Collaboration service readiness request failed.")
            return False
        if response.status_code != 200:
            log.warning(
                "Collaboration service is not ready (status=%d).",
                response.status_code,
            )
        return response.status_code == 200

    async def convert(
        self,
        *,
        document_id: str,
        markdown: str,
        schema_version: int,
        max_document_bytes: int,
    ) -> CollaborationConversion:
        payload = await self._post(
            "/internal/convert",
            {
                "document_id": document_id,
                "markdown": markdown,
                "schema_version": schema_version,
                "max_document_bytes": max_document_bytes,
            },
        )
        snapshot = _object(payload.get("snapshot"), "snapshot")
        return CollaborationConversion(
            schema_hash=_hash(payload.get("schema_hash"), "schema_hash"),
            state_update=_bytes(
                snapshot.get("state_update_base64"), "state_update_base64"
            ),
            state_vector=_bytes(
                snapshot.get("state_vector_base64"), "state_vector_base64"
            ),
            state_hash=_hash(snapshot.get("state_hash"), "state_hash"),
            projection_markdown=_string(
                payload.get("projection_markdown"), "projection_markdown"
            ),
            projection_hash=_hash(
                payload.get("projection_hash"), "projection_hash"
            ),
        )

    async def project(
        self,
        *,
        document_id: str,
        generation: int,
        minimum_sequence: int,
    ) -> CollaborationProjection:
        payload = await self._post(
            f"/internal/documents/{document_id}/project",
            {
                "generation": generation,
                "minimum_sequence": minimum_sequence,
            },
        )
        snapshot_value = payload.get("snapshot")
        snapshot = (
            _object(snapshot_value, "snapshot")
            if snapshot_value is not None
            else None
        )
        return CollaborationProjection(
            generation=_positive_int(payload.get("generation"), "generation"),
            sequence=_non_negative_int(payload.get("sequence"), "sequence"),
            markdown=_string(
                payload.get("projection_markdown"), "projection_markdown"
            ),
            projection_hash=_hash(
                payload.get("projection_hash"), "projection_hash"
            ),
            schema_hash=_hash(payload.get("schema_hash"), "schema_hash"),
            state_update=(
                _bytes(snapshot.get("state_update_base64"), "state_update_base64")
                if snapshot is not None
                else None
            ),
            state_vector=(
                _bytes(snapshot.get("state_vector_base64"), "state_vector_base64")
                if snapshot is not None
                else None
            ),
            state_hash=(
                _hash(snapshot.get("state_hash"), "state_hash")
                if snapshot is not None
                else None
            ),
        )

    async def decide(
        self,
        *,
        document_id: str,
        generation: int,
        expected_sequence: int,
        command_id: uuid.UUID,
        patch_ids: tuple[str, ...],
        decision: Literal["accept", "reject"],
        actor_user_id: uuid.UUID,
    ) -> CollaborationDecisionResult:
        payload = await self._post(
            f"/internal/documents/{document_id}/decisions",
            {
                "generation": generation,
                "expected_sequence": expected_sequence,
                "command_id": str(command_id),
                "patch_ids": list(patch_ids),
                "decision": decision,
                "actor_user_id": str(actor_user_id),
            },
        )
        returned_command = uuid.UUID(
            _string(payload.get("command_id"), "command_id")
        )
        if returned_command != command_id:
            raise CollaborationServiceUnavailable(
                "collaboration service returned the wrong command id"
            )
        return CollaborationDecisionResult(
            command_id=returned_command,
            sequence=_positive_int(payload.get("sequence"), "sequence"),
            suggestion_ids=tuple(
                _string(value, "suggestion_id")
                for value in _array(payload.get("suggestion_ids"), "suggestion_ids")
            ),
        )

    async def publish_suggestion(
        self,
        *,
        document_id: str,
        generation: int,
        expected_sequence: int,
        command_id: uuid.UUID,
        patch_id: str,
        actor_kind: Literal["assistant", "agent"],
        actor_user_id: uuid.UUID,
        target_markdown: str,
    ) -> CollaborationSuggestionResult:
        """Publish a canonical target as a tracked shared suggestion."""
        payload = await self._post(
            f"/internal/documents/{document_id}/suggestions",
            {
                "generation": generation,
                "expected_sequence": expected_sequence,
                "command_id": str(command_id),
                "patch_id": patch_id,
                "actor_kind": actor_kind,
                "actor_user_id": str(actor_user_id),
                "target_markdown": target_markdown,
            },
        )
        returned_command = uuid.UUID(
            _string(payload.get("command_id"), "command_id")
        )
        returned_patch = _string(payload.get("patch_id"), "patch_id")
        if returned_command != command_id or returned_patch != patch_id:
            raise CollaborationServiceUnavailable(
                "collaboration service returned the wrong suggestion identity"
            )
        return CollaborationSuggestionResult(
            command_id=returned_command,
            patch_id=returned_patch,
            sequence=_positive_int(payload.get("sequence"), "sequence"),
            suggestion_ids=tuple(
                _string(value, "suggestion_id")
                for value in _array(payload.get("suggestion_ids"), "suggestion_ids")
            ),
        )

    async def aclose(self) -> None:
        """Close the pooled internal HTTP client."""
        await self._client.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise CollaborationServiceUnavailable(
                "collaboration service is unreachable"
            ) from exc
        if response.status_code == 409:
            raise CollaborationNodeConflict(_response_reason(response))
        if response.status_code >= 500:
            raise CollaborationServiceUnavailable(
                "collaboration service is not ready"
            )
        if response.status_code >= 400:
            raise CollaborationNodeConflict(_response_reason(response))
        try:
            return _object(response.json(), "response")
        except ValueError as exc:
            raise CollaborationServiceUnavailable(
                "collaboration service returned invalid JSON"
            ) from exc


def _response_reason(response: httpx.Response) -> str:
    try:
        payload = _object(response.json(), "response")
    except ValueError:
        return "node_rejected"
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("reason"), str):
        return error["reason"]
    if isinstance(payload.get("reason"), str):
        return payload["reason"]
    return "node_rejected"


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _hash(value: Any, field: str) -> str:
    text = _string(value, field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return text


def _bytes(value: Any, field: str) -> bytes:
    text = _string(value, field)
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field} must be canonical base64") from exc


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value
