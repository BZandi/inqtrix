"""Direct regression tests for claim extraction strategy behavior."""

from inqtrix.exceptions import BedrockAPIError
from inqtrix.strategies import LLMClaimExtractor


class _BedrockFailingLLM:
    def complete_with_metadata(self, *args, **kwargs):
        raise BedrockAPIError(model="bedrock-sonnet", message="temporary failure")


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
    assert extractor.consume_nonfatal_notice() == (
        "Claim-Extraktion via bedrock-sonnet fehlgeschlagen; Quelle wird ohne Claims weiterverwendet."
    )
