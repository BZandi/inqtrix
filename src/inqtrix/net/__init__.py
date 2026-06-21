"""Network egress hardening for outbound fetches.

Home of the SSRF guard every fetcher of non-operator-controlled URLs
must call before connecting. See :mod:`inqtrix.net.egress_guard`.
"""

from inqtrix.net.egress_guard import (
    EgressBlockedError,
    assert_safe_url,
    resolve_and_check_host,
)

__all__ = ["EgressBlockedError", "assert_safe_url", "resolve_and_check_host"]
