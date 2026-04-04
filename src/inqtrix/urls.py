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
    """Count allowed markdown links in the answer."""
    if not answer or not allowed_urls:
        return 0
    count = 0
    for m in _MARKDOWN_LINK_RE.finditer(answer):
        if normalize_url(m.group(2)) in allowed_urls:
            count += 1
    return count


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


def sanitize_error(error: str | Exception) -> str:
    """Strip sensitive data from error messages."""
    msg = str(error)
    msg = re.sub(r"https?://[^\s]+", "[URL]", msg)
    msg = re.sub(r"(sk-|pplx-)[a-zA-Z0-9_\-]+", "[KEY]", msg)
    msg = re.sub(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer [REDACTED]", msg)
    return msg
