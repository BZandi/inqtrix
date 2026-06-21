"""SSRF egress guard: validate outbound URLs before fetching them.

Inqtrix runs inside networks where internal services (cloud metadata
endpoints, databases, queues) are one HTTP request away. Any code path
that fetches a URL which is not operator-configured — user-supplied
document URLs in the ingestion pipeline, future generic web fetchers,
redirect targets — MUST call :func:`assert_safe_url` first and treat
:class:`EgressBlockedError` as a hard, visible failure (log a WARNING,
fail the job; never silently skip).

Blocked by policy (the union, not just RFC 1918):

* Non-HTTP(S) schemes (``file://``, ``ftp://``, ``gopher://``, ...).
* Loopback (``127.0.0.0/8``, ``::1``) and unspecified (``0.0.0.0``).
* Private ranges (``10/8``, ``172.16/12``, ``192.168/16``, ``fc00::/7``).
* Link-local (``169.254/16`` — includes the cloud metadata service
  ``169.254.169.254`` — and ``fe80::/10``).
* Carrier-grade NAT (``100.64/10``), multicast, and reserved ranges.
* IPv4-mapped IPv6 addresses are unwrapped and checked as IPv4 so
  ``::ffff:169.254.169.254`` cannot smuggle past the v4 rules.

DNS handling: hostnames are resolved and EVERY resolved address must
pass the policy. This blocks names that point at internal addresses at
validation time. Known limitation (documented, not silent): a hostile
DNS server can still rebind between this check and the actual connect
(TOCTOU). Callers that need rebind-proof fetching must pin the
connection to the addresses returned by
:func:`resolve_and_check_host` instead of re-resolving in their HTTP
client, and must re-validate every redirect hop.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

_ALLOWED_SCHEMES = frozenset({"http", "https"})

_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


class EgressBlockedError(RuntimeError):
    """Raised when an outbound URL violates the egress policy.

    Attributes:
        url: The offending URL as received.
        reason: Operator-facing explanation of which rule matched.
    """

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Egress blocked for {url!r}: {reason}")
        self.url = url
        self.reason = reason


def _classify_blocked_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str | None:
    """Return the violated rule name for *ip*, or ``None`` when allowed."""
    candidate = ip
    if isinstance(candidate, ipaddress.IPv6Address) and candidate.ipv4_mapped:
        candidate = candidate.ipv4_mapped
    if candidate.is_loopback:
        return "loopback address"
    if candidate.is_link_local:
        return "link-local address (includes cloud metadata endpoints)"
    if candidate.is_unspecified:
        return "unspecified address"
    if candidate.is_private:
        return "private network address"
    if candidate.is_multicast:
        return "multicast address"
    if candidate.is_reserved:
        return "reserved address"
    if (
        isinstance(candidate, ipaddress.IPv4Address)
        and candidate in _CGNAT_NETWORK
    ):
        return "carrier-grade NAT address"
    return None


def resolve_and_check_host(host: str, *, url: str) -> tuple[str, ...]:
    """Resolve *host* and enforce the egress policy on every address.

    Args:
        host: Hostname or IP literal from the URL.
        url: The full URL, used only for error messages.

    Returns:
        Every resolved address as a string tuple — callers that want
        DNS-rebind-proof fetching pin their connection to these.

    Raises:
        EgressBlockedError: When the host is a blocked IP literal,
            resolves to any blocked address, or cannot be resolved at
            all (an unresolvable host is treated as a policy failure,
            not silently passed through to the HTTP client).
    """
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        reason = _classify_blocked_ip(literal)
        if reason is not None:
            raise EgressBlockedError(url, reason)
        return (str(literal),)

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise EgressBlockedError(url, f"hostname did not resolve ({exc})") from exc

    addresses: list[str] = []
    for info in infos:
        address = str(info[4][0])
        resolved = ipaddress.ip_address(address.split("%", 1)[0])
        reason = _classify_blocked_ip(resolved)
        if reason is not None:
            raise EgressBlockedError(
                url, f"hostname resolves to {resolved} ({reason})"
            )
        addresses.append(address)
    if not addresses:
        raise EgressBlockedError(url, "hostname resolved to no addresses")
    return tuple(dict.fromkeys(addresses))


def assert_safe_url(url: str) -> tuple[str, ...]:
    """Validate one outbound URL against the egress policy.

    Args:
        url: The URL to validate, exactly as it would be fetched.
            Redirect targets count as new URLs and must be validated
            per hop by the caller.

    Returns:
        The resolved, policy-clean addresses of the host (see
        :func:`resolve_and_check_host`).

    Raises:
        EgressBlockedError: On a disallowed scheme, a missing or
            credential-bearing host part, or any blocked address.
    """
    split = urlsplit(url)
    scheme = (split.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise EgressBlockedError(url, f"scheme {scheme!r} is not allowed")
    if split.username is not None or split.password is not None:
        raise EgressBlockedError(url, "userinfo in URL is not allowed")
    host = split.hostname
    if not host:
        raise EgressBlockedError(url, "URL has no host")
    return resolve_and_check_host(host, url=url)
