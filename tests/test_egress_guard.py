"""Tests for the SSRF egress guard (network-free via getaddrinfo stub)."""

from __future__ import annotations

import socket

import pytest

from inqtrix.net.egress_guard import (
    EgressBlockedError,
    assert_safe_url,
    resolve_and_check_host,
)


def _stub_resolver(mapping: dict[str, list[str]]):
    """Build a getaddrinfo replacement serving canned addresses."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        try:
            addresses = mapping[host]
        except KeyError as exc:
            raise socket.gaierror(8, "nodename nor servname provided") from exc
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))
            for address in addresses
        ]

    return fake_getaddrinfo


# ------------------------------------------------------------------ #
# Scheme and URL-shape rules (no DNS involved)
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "gopher://example.com/",
        "javascript:alert(1)",
        "//example.com/no-scheme",
    ],
)
def test_disallowed_schemes_are_blocked(url):
    with pytest.raises(EgressBlockedError, match="scheme|host"):
        assert_safe_url(url)


def test_userinfo_in_url_is_blocked():
    with pytest.raises(EgressBlockedError, match="userinfo"):
        assert_safe_url("https://admin:secret@example.com/")


def test_missing_host_is_blocked():
    with pytest.raises(EgressBlockedError, match="no host"):
        assert_safe_url("https:///path-only")


# ------------------------------------------------------------------ #
# IP-literal policy
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("url", "reason_fragment"),
    [
        ("http://127.0.0.1/", "loopback"),
        ("http://127.0.0.1:8080/admin", "loopback"),
        ("http://0.0.0.0/", "unspecified"),
        ("http://10.1.2.3/", "private"),
        ("http://172.16.0.9/", "private"),
        ("http://192.168.1.1/router", "private"),
        ("http://169.254.169.254/latest/meta-data/", "link-local"),
        ("http://100.64.0.1/", "carrier-grade"),
        ("http://[::1]/", "loopback"),
        ("http://[fc00::1]/", "private"),
        ("http://[fe80::1]/", "link-local"),
        ("http://[::ffff:169.254.169.254]/", "link-local"),
        ("http://[::ffff:10.0.0.1]/", "private"),
        ("http://224.0.0.1/", "multicast"),
    ],
)
def test_blocked_ip_literals(url, reason_fragment):
    with pytest.raises(EgressBlockedError, match=reason_fragment):
        assert_safe_url(url)


def test_public_ip_literal_is_allowed():
    assert assert_safe_url("https://93.184.216.34/") == ("93.184.216.34",)


# ------------------------------------------------------------------ #
# Hostname resolution policy
# ------------------------------------------------------------------ #


def test_hostname_resolving_to_metadata_ip_is_blocked(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _stub_resolver({"rebind.example": ["169.254.169.254"]}),
    )
    with pytest.raises(EgressBlockedError, match="resolves to 169.254.169.254"):
        assert_safe_url("https://rebind.example/doc.pdf")


def test_hostname_with_one_blocked_address_among_many_is_blocked(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _stub_resolver({"mixed.example": ["93.184.216.34", "10.0.0.5"]}),
    )
    with pytest.raises(EgressBlockedError, match="10.0.0.5"):
        assert_safe_url("https://mixed.example/")


def test_unresolvable_hostname_fails_closed(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _stub_resolver({}))
    with pytest.raises(EgressBlockedError, match="did not resolve"):
        assert_safe_url("https://does-not-exist.invalid/")


def test_clean_hostname_returns_pinnable_addresses(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _stub_resolver({"docs.example": ["93.184.216.34", "93.184.216.34"]}),
    )
    addresses = assert_safe_url("https://docs.example/report.pdf")
    assert addresses == ("93.184.216.34",)


def test_resolve_and_check_host_accepts_clean_literal_directly():
    assert resolve_and_check_host("93.184.216.34", url="https://x/") == (
        "93.184.216.34",
    )
