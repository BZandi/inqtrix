"""Provider failure classification uses typed protocol evidence only."""

from __future__ import annotations

from inqtrix.exceptions import AgentProviderTimeout, AgentRateLimited
from inqtrix.provider_failure_contract import (
    ProviderFailureKind,
    classify_provider_failure,
)


class _StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__("message text is not classification input")


def test_inqtrix_semantic_wrappers_are_classified_directly() -> None:
    timeout = classify_provider_failure(AgentProviderTimeout("deadline"))
    rate_limit = classify_provider_failure(
        AgentRateLimited("deployment", RuntimeError("opaque SDK error"))
    )

    assert timeout.kind == ProviderFailureKind.TIMEOUT
    assert rate_limit.kind == ProviderFailureKind.RATE_LIMITED


def test_explicit_http_status_across_chain_outranks_transport_type() -> None:
    wrapper = RuntimeError("wrapper")
    wrapper.source = TimeoutError("incidental timeout")  # type: ignore[attr-defined]
    wrapper.__cause__ = _StatusError(400)

    result = classify_provider_failure(wrapper)

    assert result.kind == ProviderFailureKind.TERMINAL
    assert result.status_code == 400


def test_bedrock_response_metadata_status_is_structured_evidence() -> None:
    error = RuntimeError("bedrock wrapper")
    error.response = {  # type: ignore[attr-defined]
        "ResponseMetadata": {"HTTPStatusCode": 503}
    }

    result = classify_provider_failure(error)

    assert result.kind == ProviderFailureKind.UNAVAILABLE
    assert result.status_code == 503


def test_unknown_programming_error_is_not_assumed_to_be_a_provider_outage() -> None:
    result = classify_provider_failure(RuntimeError("implementation defect"))

    assert result.kind == ProviderFailureKind.UNKNOWN
    assert result.transient is False
