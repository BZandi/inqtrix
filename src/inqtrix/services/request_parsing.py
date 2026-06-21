"""Protocol-level request parsing shared by the HTTP routers.

Pure functions moved verbatim from the monolithic route factory:
history formatting, OpenAI-content flattening, question/messages
normalization with payload caps, the client workspace namespace
parser, and the OpenAI-style error envelope. Behaviour (status codes,
German message strings, cap math) is contract-locked by
``tests/contract`` and must not drift.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from inqtrix.settings import AgentSettings, ServerSettings

WORKSPACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
WORKSPACE_ID_HEADER = "x-inqtrix-workspace-id"


def error_response(
    status_code: int,
    message: str,
    error_type: str,
    **extra: Any,
) -> JSONResponse:
    """Return the OpenAI-style error envelope."""
    error = {"message": message, "type": error_type}
    error.update(extra)
    return JSONResponse(status_code=status_code, content={"error": error})


def format_history(messages: list[dict], max_messages: int = 20) -> str:
    """Format message history for agent context."""
    if len(messages) <= 1:
        return ""
    history_msgs = messages[:-1][-max_messages:]
    parts: list[str] = []
    for msg in history_msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if content:
            label = {"user": "Nutzer", "assistant": "Assistent",
                     "system": "System"}.get(role, role)
            parts.append(f"{label}: {content[:500]}")
    return "\n".join(parts)


def text_from_content(content: Any) -> str:
    """Extract user-visible text from OpenAI chat content."""
    if isinstance(content, list):
        return " ".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content or "")


def workspace_id_from_request(
    req: Request,
    body: dict[str, Any] | None = None,
) -> str | None:
    """Resolve and validate the optional browser/project workspace namespace.

    This value is a CLIENT-SUPPLIED UI namespace used to filter run
    listings per browser profile — it is not, and must never become,
    an authorization input. The authorization workspace (a server-side
    membership fact) is a separate concept resolved from the verified
    principal.
    """
    body_workspace_id = body.get("workspace_id") if isinstance(body, dict) else None
    raw = (
        req.headers.get(WORKSPACE_ID_HEADER)
        or req.query_params.get("workspace_id")
        or body_workspace_id
    )
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or not WORKSPACE_ID_PATTERN.fullmatch(raw):
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": "Invalid workspace_id.",
                "type": "invalid_request_error",
            }},
        )
    return raw


def enforce_payload_caps(
    question: str,
    messages: list[dict[str, Any]],
    server: ServerSettings,
) -> None:
    """Reject oversize ``messages[]`` arrays with HTTP 413.

    Two caps from :class:`ServerSettings`:

    * ``max_message_count`` rejects array-bomb payloads that would
      stretch the agent's token bookkeeping for no real-user reason.
    * ``max_total_input_tokens`` rejects bodies whose combined
      *question + messages content* approximate-token count
      (4 chars per token) exceeds the operator-configured limit.
      Defends against pathological 1-message-with-megabytes payloads.

    The defaults are deliberately generous (200 messages, 500k
    tokens) so realistic multi-turn flows never trip them. The
    check runs after content normalization so it sees what the
    agent will actually consume.
    """
    max_count = server.max_message_count
    if len(messages) > max_count:
        raise HTTPException(
            status_code=413,
            detail={"error": {
                "message": (
                    f"messages array exceeds limit ({len(messages)} > "
                    f"{max_count})"
                ),
                "type": "payload_too_large",
            }},
        )
    total_chars = sum(
        len(text_from_content(msg.get("content", ""))) for msg in messages
    ) + len(question)
    approx_tokens = total_chars // 4
    max_tokens = server.max_total_input_tokens
    if approx_tokens > max_tokens:
        raise HTTPException(
            status_code=413,
            detail={"error": {
                "message": (
                    f"input size ~{approx_tokens} tokens exceeds limit "
                    f"({max_tokens})"
                ),
                "type": "payload_too_large",
            }},
        )


def question_and_messages(
    body: dict[str, Any],
    server: ServerSettings,
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve either native ``question`` or OpenAI ``messages`` input."""
    raw_question = body.get("question")
    messages = body.get("messages", [])
    if isinstance(raw_question, str) and raw_question.strip():
        question = raw_question.strip()
        normalized_messages = (
            messages
            if (
                isinstance(messages, list)
                and messages
                and all(isinstance(msg, dict) for msg in messages)
            )
            else [{"role": "user", "content": question}]
        )
        enforce_payload_caps(question, normalized_messages, server)
        return question, normalized_messages
    if not isinstance(messages, list) or not messages:
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": "Feld 'question' oder nicht-leere 'messages' ist erforderlich",
                "type": "invalid_request_error",
            }},
        )
    if not all(isinstance(msg, dict) for msg in messages):
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": "messages muss eine Liste von Objekten sein",
                "type": "invalid_request_error",
            }},
        )
    last = messages[-1]
    question = text_from_content(last.get("content", "")).strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": "Letzte Nachricht hat keinen Inhalt",
                "type": "invalid_request_error",
            }},
        )
    enforce_payload_caps(question, messages, server)
    return question, messages


REQUEST_WAIT_MARGIN_SECONDS = 30
"""Grace the outer HTTP wait adds over the inner per-call budget (seconds).

Every HTTP-level ``asyncio.wait_for`` deliberately outlives the inner per-call
timeout by this margin so the inner call raises its specific, localized error
(a provider timeout with a clear message) before the outer wait fires a generic
504. Defined once and reused by every derived wait below so the relationship
lives in a single place rather than four inlined ``+ 30`` literals.
"""


def request_timeout_seconds(agent_settings: AgentSettings) -> int:
    """HTTP-level deadline for a whole-run agent execution (graph cap + margin).

    Bounds the synchronous HTTP wait for a chat run and the SSE streaming
    deadline. Derives from ``max_total_seconds`` (the wall-clock run budget),
    so raising the run budget widens this wait in lockstep.
    """
    return agent_settings.max_total_seconds + REQUEST_WAIT_MARGIN_SECONDS


def editor_wait_seconds(agent_settings: AgentSettings) -> int:
    """HTTP wait for a single editor suggest/instruct call (seconds).

    Bounds the outer ``asyncio.wait_for`` around the editor LLM call, which
    itself runs under ``editor_assistant_timeout`` (the dedicated editor budget,
    decoupled from research ``reasoning_timeout``). Capped by the whole-run HTTP
    deadline so the editor wait can never outlive ``max_total_seconds``.
    """
    return min(
        request_timeout_seconds(agent_settings),
        agent_settings.editor_assistant_timeout + REQUEST_WAIT_MARGIN_SECONDS,
    )


def text_wait_seconds(agent_settings: AgentSettings) -> int:
    """HTTP wait for a single ``/v1/text`` improvement call (seconds).

    Bounds the outer ``asyncio.wait_for`` around the text-improvement LLM call,
    which runs under ``claim_extract_timeout``. Capped by the whole-run HTTP
    deadline. This is the path whose wait hangs off ``claim_extract_timeout``
    rather than the reasoning budget.
    """
    return min(
        request_timeout_seconds(agent_settings),
        agent_settings.claim_extract_timeout + REQUEST_WAIT_MARGIN_SECONDS,
    )
