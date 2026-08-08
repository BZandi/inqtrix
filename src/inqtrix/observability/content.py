"""Content-capture policy for trace attributes (the ONLY content path).

Whether prompts, raw responses, and tool payloads reach span attributes
is decided here — instrumentation code never writes content without
this policy. Langfuse OSS applies no server-side truncation or masking,
so redaction and byte caps on THIS side are the only guard.

Capture resolution (``INQTRIX_TRACE_CONTENT``):

* ``auto`` (default) — capture follows ``OBSERVABILITY_PROFILE``:
  content only in the ``forensic`` profile.
* ``on`` / ``off`` — explicit override, independent of the profile.

Redaction reuses the EXISTING sanitizers (no second redaction system):
:func:`inqtrix.runtime_logging.sanitize_event_payload` for structured
payloads (drop-list for credential keys, URL secret scrubbing) and
``sanitize_log_message`` for plain strings; a hard per-attribute byte
cap with a visible truncation marker bounds every value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from inqtrix.runtime_logging import sanitize_event_payload
from inqtrix.urls import sanitize_log_message

if TYPE_CHECKING:
    from inqtrix.settings import Settings

_TRUNCATION_MARKER = "…[truncated]"


@dataclass(frozen=True)
class ClippedText:
    """One redacted, byte-capped attribute value plus its original size."""

    text: str
    original_size: int
    truncated: bool


@dataclass(frozen=True)
class ContentCapturePolicy:
    """Decides IF content is captured and bounds HOW MUCH of it."""

    capture_content: bool
    max_attr_bytes: int

    def clip_text(self, value: object) -> ClippedText:
        """Redact one string and enforce the per-attribute byte cap."""
        redacted = sanitize_log_message(str(value or ""))
        encoded = redacted.encode("utf-8")
        if len(encoded) <= self.max_attr_bytes:
            return ClippedText(redacted, len(encoded), False)
        clipped = encoded[: self.max_attr_bytes].decode("utf-8", "ignore")
        return ClippedText(
            clipped + _TRUNCATION_MARKER, len(encoded), True
        )

    def clip_payload(self, payload: Any) -> ClippedText:
        """Sanitize a structured payload and render it as capped JSON.

        Runs the payload through the runtime-logging sanitizer (schema-
        less path: credential-key drop-list + URL redaction) before
        serializing, so ``raw_response``-style fields can never leak
        into a span attribute even under content capture.
        """
        sanitized = sanitize_event_payload(
            "trace_content", {"value": payload}
        ).get("value")
        try:
            rendered = json.dumps(
                sanitized, ensure_ascii=False, default=str
            )
        except (TypeError, ValueError):
            rendered = str(sanitized)
        return self.clip_text(rendered)


def build_content_policy(settings: "Settings") -> ContentCapturePolicy:
    """Resolve the policy from settings (see module docstring)."""
    mode = settings.observability.trace_content
    profile = str(
        getattr(settings.agent, "observability_profile", "summary")
    ).lower()
    capture = mode == "on" or (mode == "auto" and profile == "forensic")
    return ContentCapturePolicy(
        capture_content=capture,
        max_attr_bytes=settings.observability.trace_max_attr_bytes,
    )


# ONE process-wide policy, published by ``create_providers`` from the
# COMPOSED settings. Deep call sites that cannot be threaded a policy
# (the kernel tool boundary) read it here, so every content attribute in
# the process is gated by the same decision — a fresh env-driven
# ``Settings()`` could diverge from what the app was actually composed
# with (embedded callers, tests).
_active_policy: ContentCapturePolicy | None = None


def set_active_content_policy(policy: ContentCapturePolicy) -> None:
    """Publish the composed policy (called by ``create_providers``)."""
    global _active_policy
    _active_policy = policy


def active_content_policy() -> ContentCapturePolicy:
    """The published policy; falls back to env settings ONCE if none.

    The fallback covers processes that trace without building providers
    (tooling, partial test apps); it is cached as the active policy so
    repeated reads stay consistent.
    """
    global _active_policy
    if _active_policy is None:
        from inqtrix.settings import Settings

        _active_policy = build_content_policy(Settings())
    return _active_policy
