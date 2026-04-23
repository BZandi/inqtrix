""" Tests for inqtrix.urls — URL normalization and extraction."""

from inqtrix.urls import normalize_url, extract_urls, count_allowed_links


class TestNormalizeUrl:

    def test_removes_trailing_slash(self):
        assert normalize_url("https://example.com/path/") == "https://example.com/path"

    def test_keeps_root_domain_slash(self):
        assert normalize_url("https://example.com/") == "https://example.com/"

    def test_removes_fragment(self):
        assert normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_removes_tracking_params(self):
        assert normalize_url("https://example.com/page?utm_source=google&ref=abc") == \
            "https://example.com/page"

    def test_removes_fbclid(self):
        assert normalize_url("https://example.com/page?fbclid=abc123") == \
            "https://example.com/page"

    def test_preserves_non_tracking_params(self):
        assert normalize_url("https://example.com/page?id=42") == \
            "https://example.com/page?id=42"

    def test_removes_trailing_punctuation(self):
        assert normalize_url("https://example.com/page.") == "https://example.com/page"
        assert normalize_url("https://example.com/page,") == "https://example.com/page"
        assert normalize_url("https://example.com/page)") == "https://example.com/page"

    def test_empty_string(self):
        assert normalize_url("") == ""

    def test_complex_url(self):
        url = "https://arxiv.org/abs/2401.12345?utm_medium=email&ref=twitter#abstract"
        result = normalize_url(url)
        assert "utm_medium" not in result
        assert "ref=twitter" not in result
        assert "#abstract" not in result
        assert "2401.12345" in result


class TestExtractUrls:

    def test_single_url(self):
        assert extract_urls("See https://example.com for details") == ["https://example.com"]

    def test_multiple_urls(self):
        text = "Check https://a.com and https://b.com"
        result = extract_urls(text)
        assert len(result) == 2
        assert "https://a.com" in result
        assert "https://b.com" in result

    def test_deduplication(self):
        text = "https://example.com/page and https://example.com/page again"
        assert len(extract_urls(text)) == 1

    def test_dedup_with_normalization(self):
        text = "https://example.com/page/ and https://example.com/page"
        assert len(extract_urls(text)) == 1

    def test_no_urls(self):
        assert extract_urls("No URLs here") == []

    def test_empty_string(self):
        assert extract_urls("") == []

    def test_http_url(self):
        assert extract_urls("Visit http://example.com") == ["http://example.com"]

    def test_url_in_markdown_link(self):
        text = "[Link](https://example.com/page)"
        result = extract_urls(text)
        assert len(result) == 1
        assert "example.com/page" in result[0]


class TestCountAllowedLinks:

    def test_counts_unique_urls(self):
        allowed = {"https://example.com/a", "https://example.com/b"}
        answer = "[1](https://example.com/a) foo [2](https://example.com/a) bar [3](https://example.com/b)"
        assert count_allowed_links(answer, allowed) == 2

    def test_ignores_non_allowed(self):
        allowed = {"https://example.com/a"}
        answer = "[1](https://example.com/a) [2](https://other.com/b)"
        assert count_allowed_links(answer, allowed) == 1

    def test_empty_answer(self):
        assert count_allowed_links("", {"https://example.com"}) == 0

    def test_empty_allowed(self):
        assert count_allowed_links("[1](https://example.com)", set()) == 0

    def test_no_links(self):
        assert count_allowed_links("plain text", {"https://example.com"}) == 0
