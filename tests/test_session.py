""" Tests for session management — SessionStore and ID derivation."""

import time

from inqtrix.server.session import (
    SessionStore,
    derive_session_id,
    prospective_session_id,
)
from inqtrix.server.routes import _format_history


class TestFormatHistory:

    def test_single_message_returns_empty(self):
        assert _format_history([{"role": "user", "content": "Hello"}]) == ""

    def test_empty_returns_empty(self):
        assert _format_history([]) == ""

    def test_two_messages_returns_first(self):
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        result = _format_history(msgs)
        assert "First" in result
        assert "Second" not in result

    def test_role_labels_german(self):
        msgs = [
            {"role": "user", "content": "Frage"},
            {"role": "assistant", "content": "Antwort"},
            {"role": "user", "content": "Neue Frage"},
        ]
        result = _format_history(msgs)
        assert "Nutzer:" in result
        assert "Assistent:" in result

    def test_multimodal_messages(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "url": "data:..."},
            ]},
            {"role": "user", "content": "Follow up"},
        ]
        result = _format_history(msgs)
        assert "Describe this" in result

    def test_truncates_long_content(self):
        msgs = [
            {"role": "user", "content": "x" * 1000},
            {"role": "user", "content": "current"},
        ]
        result = _format_history(msgs)
        assert len(result) <= 510


class TestSessionStore:

    def test_save_and_get(self):
        store = SessionStore(ttl_seconds=60, max_count=10)
        state = {
            "all_citations": ["https://example.com"],
            "context": ["block1"],
            "consolidated_claims": [],
            "claim_ledger": [],
        }
        store.save("sid1", state, "question?", "answer text")
        snap = store.get("sid1")
        assert snap is not None
        assert snap.session_id == "sid1"
        assert snap.all_citations == ["https://example.com"]
        assert snap.last_question == "question?"

    def test_get_nonexistent_returns_none(self):
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_ttl_expiration(self):
        store = SessionStore(ttl_seconds=0, max_count=10)
        store.save("sid1", {}, "q", "a")
        # TTL=0 means immediately expired on next access
        assert store.get("sid1") is None

    def test_max_count_eviction(self):
        store = SessionStore(ttl_seconds=3600, max_count=2)
        store.save("sid1", {}, "q1", "a1")
        store.save("sid2", {}, "q2", "a2")
        store.save("sid3", {}, "q3", "a3")
        # sid1 should have been evicted (LRU)
        assert store.get("sid1") is None
        assert store.get("sid2") is not None
        assert store.get("sid3") is not None


class TestSessionIdDerivation:

    def test_first_turn_returns_none(self):
        assert derive_session_id([{"role": "user", "content": "Hello"}]) is None

    def test_empty_messages_returns_none(self):
        assert derive_session_id([]) is None

    def test_second_turn_returns_id(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        sid = derive_session_id(msgs)
        assert sid is not None
        assert len(sid) == 24

    def test_prospective_matches_derive(self):
        msgs_turn1 = [
            {"role": "user", "content": "q1"},
        ]
        answer = "a1"
        # Prospective ID after turn 1
        pid = prospective_session_id(msgs_turn1, answer)
        # Derived ID at turn 2
        msgs_turn2 = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "q2"},
        ]
        did = derive_session_id(msgs_turn2)
        assert pid == did

    def test_same_assistant_answer_different_user_history_yields_different_ids(self):
        conv_a = [
            {"role": "user", "content": "Frage A"},
            {"role": "assistant", "content": "Gleiche Antwort"},
            {"role": "user", "content": "Follow-up A"},
        ]
        conv_b = [
            {"role": "user", "content": "Frage B"},
            {"role": "assistant", "content": "Gleiche Antwort"},
            {"role": "user", "content": "Follow-up B"},
        ]

        assert derive_session_id(conv_a) != derive_session_id(conv_b)
