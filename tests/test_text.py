"""Tests for inqtrix.text — text utility functions."""

from inqtrix.text import is_none_value, iter_word_chunks


class TestIsNoneValue:

    def test_keine(self):
        assert is_none_value("Keine") is True

    def test_none_word(self):
        assert is_none_value("None") is True

    def test_keine_with_dot(self):
        assert is_none_value("Keine.") is True

    def test_none_with_dot(self):
        assert is_none_value("None.") is True

    def test_empty_string(self):
        assert is_none_value("") is True

    def test_keine_prefix_with_text(self):
        assert is_none_value("Keine relevanten Luecken") is True

    def test_case_insensitive(self):
        assert is_none_value("KEINE") is True
        assert is_none_value("NONE") is True
        assert is_none_value("keine") is True

    def test_whitespace(self):
        assert is_none_value("  Keine  ") is True

    def test_actual_value(self):
        assert is_none_value("Es fehlen Daten zu X") is False

    def test_partial_match_not_triggered(self):
        assert is_none_value("Dieses Ergebnis ist keines") is False


class TestIterWordChunks:

    def test_preserves_spaces_between_words(self):
        assert list(iter_word_chunks("Hallo Welt")) == ["Hallo ", "Welt"]

    def test_preserves_newlines(self):
        assert list(iter_word_chunks("A\n\nB")) == ["A\n\n", "B"]
