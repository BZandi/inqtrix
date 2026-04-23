"""Direct regression tests for claim extraction strategy behavior."""

from types import SimpleNamespace

from inqtrix.exceptions import BedrockAPIError
from inqtrix.strategies import LLMClaimExtractor


class _BedrockFailingLLM:
    def complete_with_metadata(self, *args, **kwargs):
        raise BedrockAPIError(model="bedrock-sonnet", message="temporary failure")


class _RecordingLLM:
    def __init__(self) -> None:
        self.last_prompt = ""

    def complete_with_metadata(self, prompt, *args, **kwargs):
        self.last_prompt = prompt
        return SimpleNamespace(
            content=(
                '{"claims": ['
                '{"claim_text": "Erster Claim mit Zahl 10 Prozent.", "claim_type": "fact", '
                '"polarity": "affirmed", "needs_primary": true, '
                '"source_urls": ["https://example.com/a", "https://example.com/b"], '
                '"published_date": "2026-04-13"}, '
                '{"claim_text": "Zweiter Claim mit Verbandssicht.", "claim_type": "actor_claim", '
                '"polarity": "affirmed", "needs_primary": false, '
                '"source_urls": ["https://example.com/a"], '
                '"published_date": "2026-04-13"}, '
                '{"claim_text": "Dritter Claim fuer den Overflow.", "claim_type": "fact", '
                '"polarity": "affirmed", "needs_primary": false, '
                '"source_urls": ["https://example.com/a"], '
                '"published_date": "2026-04-13"}'
                ']} '
            ),
            prompt_tokens=11,
            completion_tokens=7,
        )


def test_claim_extraction_bedrock_error_falls_back_nonfatally():
    extractor = LLMClaimExtractor(
        _BedrockFailingLLM(),
        summarize_model="bedrock-sonnet",
    )

    claims, prompt_tokens, completion_tokens = extractor.extract(
        "Kurzer Testtext",
        ["https://example.com/report"],
        "Was ist passiert?",
    )

    assert claims == []
    assert prompt_tokens == 0
    assert completion_tokens == 0
    notice = extractor.consume_nonfatal_notice()
    # Phase-2 visibility: notice must include the model AND the underlying
    # exception class + message, not just a generic "failed".
    assert notice is not None
    assert "Claim-Extraktion via bedrock-sonnet fehlgeschlagen" in notice
    assert "BedrockAPIError" in notice
    assert "Quelle wird ohne Claims weiterverwendet" in notice


def test_claim_extraction_respects_custom_limits():
    llm = _RecordingLLM()
    extractor = LLMClaimExtractor(
        llm,
        summarize_model="stub-model",
    )

    claims, prompt_tokens, completion_tokens = extractor.extract(
        "ABCDE-truncated-text",
        ["https://example.com/a", "https://example.com/b"],
        "Was ist passiert?",
        text_char_limit=5,
        citation_cap=1,
        max_claims=2,
        source_url_limit=1,
    )

    assert len(claims) == 2
    assert all(len(claim["source_urls"]) <= 1 for claim in claims)
    assert "Quellenliste:\n[\"https://example.com/a\"]" in llm.last_prompt
    assert "Text:\nABCDE" in llm.last_prompt
    assert prompt_tokens == 11
    assert completion_tokens == 7
