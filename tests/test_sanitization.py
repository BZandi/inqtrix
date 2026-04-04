""" Tests for error sanitization — sensitive data redaction."""

from inqtrix.urls import sanitize_error


class TestSanitizeError:

    def test_redacts_sk_key(self):
        err = Exception("Error with key sk-abc123def456")
        result = sanitize_error(err)
        assert "sk-abc123" not in result
        assert "[KEY]" in result

    def test_redacts_pplx_key(self):
        err = Exception("Error with key pplx-6MdfNZ6uKh9q")
        result = sanitize_error(err)
        assert "pplx-6MdfNZ6u" not in result
        assert "[KEY]" in result

    def test_redacts_url(self):
        err = Exception("Failed to connect to https://api.example.com/v1")
        result = sanitize_error(err)
        assert "https://api.example.com" not in result
        assert "[URL]" in result

    def test_redacts_bearer_token(self):
        err = Exception("Auth failed: Bearer eyJhbGciOi.token.here")
        result = sanitize_error(err)
        assert "eyJhbGciOi" not in result
        assert "Bearer [REDACTED]" in result

    def test_preserves_safe_text(self):
        err = Exception("Connection timeout after 30 seconds")
        result = sanitize_error(err)
        assert result == "Connection timeout after 30 seconds"

    def test_multiple_redactions(self):
        err = Exception("Key sk-abc at https://example.com with pplx-xyz")
        result = sanitize_error(err)
        assert "sk-abc" not in result
        assert "https://example.com" not in result
        assert "pplx-xyz" not in result
