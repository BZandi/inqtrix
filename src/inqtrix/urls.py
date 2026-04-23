""" URL processing utilities."""

from __future__ import annotations

import datetime
import re
from urllib.parse import urlparse

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def today() -> str:
    """Current date as string."""
    return datetime.date.today().strftime("%d. %B %Y")


def normalize_url(url: str) -> str:
    """Normalize URL for better deduplication.

    Removes trailing slashes, fragments, and tracking parameters.
    """
    url = url.rstrip(".,;:!?)")
    url = re.sub(r'#[^?]*$', '', url)
    if url.count('/') > 3:
        url = url.rstrip('/')
    url = re.sub(r'[?&](utm_[a-z]+|ref|source|fbclid|gclid)=[^&]*', '', url)
    url = url.rstrip('?')
    return url


def extract_urls(text: str) -> list[str]:
    """Extract and deduplicate URLs from text."""
    urls = re.findall(r'https?://[^\s\)\]\},\"\']+', text)
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        normalized = normalize_url(url)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def sanitize_answer_links(answer: str, allowed_urls: set[str]) -> tuple[str, int]:
    """Remove non-allowed markdown links from the final answer."""
    if not answer or not allowed_urls:
        return answer, 0

    removed = 0

    def _repl(m: re.Match[str]) -> str:
        nonlocal removed
        label = m.group(1)
        url = m.group(2)
        normalized = normalize_url(url)
        if normalized in allowed_urls:
            return f"[{label}]({normalized})"
        removed += 1
        return f"[{label}]"

    sanitized = _MARKDOWN_LINK_RE.sub(_repl, answer)
    return sanitized, removed


def count_allowed_links(answer: str, allowed_urls: set[str]) -> int:
    """Count unique allowed markdown links in the answer."""
    if not answer or not allowed_urls:
        return 0
    seen: set[str] = set()
    for m in _MARKDOWN_LINK_RE.finditer(answer):
        normalized = normalize_url(m.group(2))
        if normalized in allowed_urls:
            seen.add(normalized)
    return len(seen)


def domain_from_url(url: str) -> str:
    """Extract domain from URL (without www prefix)."""
    try:
        host = (urlparse(url).hostname or "").lower().strip()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def domain_matches(domain: str, candidates: set[str]) -> bool:
    """Check if domain matches exactly or as subdomain."""
    if not domain:
        return False
    for cand in candidates:
        c = cand.lower().strip()
        if domain == c or domain.endswith(f".{c}"):
            return True
    return False


_CREDENTIAL_QUERY_PARAM_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|bearer[_-]?token|"
    r"client[_-]?secret|password|secret|token|signature|sig|x-[a-z-]*key)"
    r"=([^&\s\"'<>]+)"
)


def _redact_credential_url(url: str) -> str:
    """Replace secret-bearing query-parameter values inside a single URL.

    Only the *value* of the credential parameter is replaced, not the URL itself.
    A URL like ``https://example.com/api?api_key=sk-abc&page=2`` becomes
    ``https://example.com/api?api_key=[REDACTED]&page=2``.
    """
    return _CREDENTIAL_QUERY_PARAM_RE.sub(r"\1=[REDACTED]", url)


def _scrub_credentials(msg: str) -> str:
    """Common credential-scrubbing rules used by the log filter and error helper.

    Removes API keys, bearer tokens, AWS access keys, and credential values
    inside URL query parameters. URLs themselves are intentionally NOT replaced
    by ``[URL]`` — final answers, citation lists, and trace logs need to keep
    their links intact for debugging.
    """
    # Per-URL credential redaction first so that the surrounding URL stays visible.
    msg = re.sub(
        r"https?://[^\s\"'<>]+",
        lambda m: _redact_credential_url(m.group(0)),
        msg,
    )
    msg = re.sub(r"(sk-|pplx-)[a-zA-Z0-9_\-]{16,}", "[KEY]", msg)
    msg = re.sub(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer [REDACTED]", msg)
    msg = re.sub(r"AKIA[A-Z0-9]{16}", "[AWS_KEY]", msg)
    msg = re.sub(
        r"(?i)(aws_secret_access_key|aws_session_token)[=:]\s*\S+",
        r"\1=[REDACTED]",
        msg,
    )
    return msg


def sanitize_log_message(message: str | Exception) -> str:
    """Scrub credentials from a log message while keeping benign URLs intact.

    Designed for the centralized logging filter. Differs from
    :func:`sanitize_error` only in intent and naming — both currently share
    the same credential-scrubbing rules. Call this from log handlers; call
    ``sanitize_error`` from explicit error-formatting code paths.
    """
    return _scrub_credentials(str(message))


def sanitize_error(error: str | Exception) -> str:
    """Strip sensitive data from error messages.

    Use for explicit error stringification (HTTP responses, stderr prints,
    user-visible failure dialogs). Credentials are removed; URLs themselves
    are kept so support tickets carry the failing endpoint context.
    """
    return _scrub_credentials(str(error))
