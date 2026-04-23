""" In-memory session store for follow-up question support.

Holds :class:`SessionSnapshot` instances keyed by a stable hash derived
from the user/assistant conversation history. Sessions expire after a
configurable TTL and are capped at a maximum count (LRU eviction).
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------ #
# SessionSnapshot
# ------------------------------------------------------------------ #


@dataclass
class SessionSnapshot:
    """Snapshot of research results from a completed agent turn.

    Kept in the in-memory store so that follow-up questions can build on
    prior research instead of starting from scratch.
    Graceful degradation: on pod restart or TTL expiry sessions are lost
    and the agent falls back to fresh research.
    """

    session_id: str
    last_access: float                                      # time.monotonic()
    created_at: float                                       # time.monotonic()
    # Sources & research
    all_citations: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)        # capped
    consolidated_claims: list[dict[str, Any]] = field(default_factory=list)
    claim_ledger: list[dict[str, Any]] = field(default_factory=list)  # capped
    # Quality metrics
    source_tier_counts: dict[str, int] = field(default_factory=dict)
    source_quality_score: float = 0.0
    claim_status_counts: dict[str, int] = field(default_factory=dict)
    claim_quality_score: float = 0.0
    aspect_coverage: float = 0.0
    # Structure
    required_aspects: list[str] = field(default_factory=list)
    uncovered_aspects: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    # Metadata
    language: str = ""
    search_language: str = ""
    query_type: str = "general"
    recency: str = ""
    risk_score: int = 0
    high_risk: bool = False
    final_confidence: int = 0
    # Follow-up context
    last_question: str = ""
    last_answer: str = ""


# ------------------------------------------------------------------ #
# SessionStore
# ------------------------------------------------------------------ #


class SessionStore:
    """Thread-safe, TTL-based in-memory session store.

    Parameters
    ----------
    ttl_seconds:
        Maximum inactivity time before a session is evicted.
    max_count:
        Hard cap on the number of concurrent sessions (LRU eviction).
    max_context_blocks:
        Maximum context blocks stored per snapshot.
    max_claim_ledger:
        Maximum claim ledger entries per snapshot.
    """

    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_count: int = 20,
        max_context_blocks: int = 8,
        max_claim_ledger: int = 50,
        max_answer_chars: int = 2000,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_count = max_count
        self._max_context_blocks = max_context_blocks
        self._max_claim_ledger = max_claim_ledger
        self._max_answer_chars = max_answer_chars
        self._store: dict[str, SessionSnapshot] = {}
        self._lock = threading.Lock()

    # -- internal helpers --------------------------------------------------

    def _cleanup(self) -> None:
        """Remove expired sessions. Must be called under lock."""
        now = time.monotonic()
        expired = [
            sid for sid, snap in self._store.items()
            if (now - snap.last_access) > self._ttl
        ]
        for sid in expired:
            del self._store[sid]
        # LRU eviction on overflow
        if len(self._store) > self._max_count:
            sorted_sessions = sorted(
                self._store.items(), key=lambda x: x[1].last_access
            )
            to_remove = len(self._store) - self._max_count
            for sid, _ in sorted_sessions[:to_remove]:
                del self._store[sid]

    # -- public interface --------------------------------------------------

    def get(self, session_id: str) -> SessionSnapshot | None:
        """Retrieve a deep copy of a session snapshot.

        Returns a copy so callers cannot mutate the internal store.
        """
        with self._lock:
            self._cleanup()
            snap = self._store.get(session_id)
            if snap is None:
                return None
            snap.last_access = time.monotonic()
            import copy
            return copy.deepcopy(snap)

    def save(
        self,
        session_id: str,
        result_state: dict[str, Any],
        question: str,
        answer_text: str,
    ) -> None:
        """Persist research results as a session snapshot."""
        context = list(result_state.get("context", []))
        if len(context) > self._max_context_blocks:
            context = context[-self._max_context_blocks:]
        claim_ledger = list(result_state.get("claim_ledger", []))
        if len(claim_ledger) > self._max_claim_ledger:
            claim_ledger = claim_ledger[-self._max_claim_ledger:]

        snap = SessionSnapshot(
            session_id=session_id,
            last_access=time.monotonic(),
            created_at=time.monotonic(),
            all_citations=list(result_state.get("all_citations", [])),
            context=context,
            consolidated_claims=list(result_state.get("consolidated_claims", [])),
            claim_ledger=claim_ledger,
            source_tier_counts=dict(result_state.get("source_tier_counts", {})),
            source_quality_score=float(result_state.get("source_quality_score", 0.0)),
            claim_status_counts=dict(result_state.get("claim_status_counts", {})),
            claim_quality_score=float(result_state.get("claim_quality_score", 0.0)),
            aspect_coverage=float(result_state.get("aspect_coverage", 0.0)),
            required_aspects=list(result_state.get("required_aspects", [])),
            uncovered_aspects=list(result_state.get("uncovered_aspects", [])),
            sub_questions=list(result_state.get("sub_questions", [])),
            queries=list(result_state.get("queries", [])),
            language=result_state.get("language", ""),
            search_language=result_state.get("search_language", ""),
            query_type=result_state.get("query_type", "general"),
            recency=result_state.get("recency", ""),
            risk_score=int(result_state.get("risk_score", 0)),
            high_risk=bool(result_state.get("high_risk", False)),
            final_confidence=int(result_state.get("final_confidence", 0)),
            last_question=question[:500],
            last_answer=(answer_text or "")[:self._max_answer_chars],
        )
        with self._lock:
            self._cleanup()
            self._store[session_id] = snap


# ------------------------------------------------------------------ #
# Session ID helpers
# ------------------------------------------------------------------ #


def _message_text(msg: dict[str, Any]) -> str:
    """Normalize a chat message into a single text string."""
    content = msg.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content or "").strip()


def _extract_session_parts(messages: list[dict]) -> list[str]:
    """Extract truncated role-tagged message texts for session hashing."""
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _message_text(msg)
        if content:
            parts.append(f"{role}:{content[:200]}")
    return parts


def _hash_parts(parts: list[str]) -> str:
    """Compute a session ID from role-tagged conversation parts."""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def derive_session_id(
    messages: list[dict],
    stack_name: str = "",
) -> str | None:
    """Derive a stable session ID from the conversation history.

    Hashes the role-tagged conversation history (excluding the current
    question) so follow-up detection distinguishes conversations that
    happen to share the same assistant wording.

    Args:
        messages: OpenAI-compatible chat-message list.
        stack_name: Optional Multi-Stack identifier. When non-empty
            the stack name is mixed into the hash so the same
            conversation against two different stacks resolves to
            two distinct sessions, preventing cross-stack follow-up
            confusion (ADR-MS-4). Default ``""`` keeps the historic
            single-stack behaviour.

    Returns:
        A stable session id, or ``None`` when no assistant turn has
        happened yet (first turn).
    """
    if len(messages) <= 1:
        return None
    history_msgs = messages[:-1]  # Exclude current question
    parts = _extract_session_parts(history_msgs)
    if not any(part.startswith("assistant:") for part in parts):
        return None
    if stack_name:
        parts = parts + [f"stack:{stack_name}"]
    return _hash_parts(parts)


def prospective_session_id(
    messages: list[dict],
    answer_text: str,
    stack_name: str = "",
) -> str:
    """Compute the session ID that the NEXT turn will derive.

    Called after turn N to store the session under the ID that turn N+1
    will compute via :func:`derive_session_id` (includes current answer
    in hash). ``stack_name`` mirrors :func:`derive_session_id` and must
    be passed identically on save and on load to stay consistent.
    """
    parts = _extract_session_parts(messages)
    if answer_text:
        parts.append(f"assistant:{answer_text[:200]}")
    if stack_name:
        parts.append(f"stack:{stack_name}")
    return _hash_parts(parts)
