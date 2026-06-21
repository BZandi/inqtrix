"""Tests for editor paragraph-rewrite helpers and the HTTP route."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inqtrix.exceptions import AgentStructuredOutputError
from inqtrix.providers.base import ProviderContext, StructuredLLMResponse
from inqtrix.search_result import GroundedSearchResult
from inqtrix.server.app import create_app
from inqtrix.server.editor_suggestions import (
    EditorSuggestError,
    EditorSuggestRequestData,
    build_editor_suggest_prompt,
    clamp_background,
    parse_editor_suggest_payload,
    parse_editor_suggest_response,
    result_from_parsed,
    validate_editor_suggest_result,
    warnings_for_validation_issues,
)
from inqtrix.server.reference_documents import (
    ReferenceDocument,
    render_reference_documents,
)
from inqtrix.settings import AgentSettings, ModelSettings, ServerSettings, Settings

_VALID_JSON = '{"rewritten_text":"Die Arbeitsgruppe erweitert das Sortiment.","changes":["aktiv"]}'


class _CapturingLLM:
    def __init__(
        self,
        *,
        response: str | list[str] = _VALID_JSON,
        structured: bool = False,
        models: ModelSettings | None = None,
    ) -> None:
        self.prompt: str | None = None
        self.kwargs: dict[str, object] = {}
        self.structured_called = False
        self.responses = [response] if isinstance(response, str) else list(response)
        self._structured = structured
        if models is not None:
            self.models = models

    def supports_structured_output(self, *, model: str | None = None) -> bool:
        return self._structured

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.prompt = prompt
        self.kwargs = kwargs
        return self._next_response()

    def complete_structured(self, prompt: str, **kwargs: object) -> StructuredLLMResponse:
        self.prompt = prompt
        self.kwargs = kwargs
        self.structured_called = True
        import json

        return StructuredLLMResponse(parsed=json.loads(self._next_response()))

    def is_available(self) -> bool:
        return True

    def _next_response(self) -> str:
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class _DummySearch:
    def search(self, *args: object, **kwargs: object) -> GroundedSearchResult:
        return GroundedSearchResult()

    def is_available(self) -> bool:
        return True


def _make_client(
    *,
    llm: _CapturingLLM | None = None,
    server_settings: ServerSettings | None = None,
) -> tuple[TestClient, _CapturingLLM]:
    active_llm = llm or _CapturingLLM()
    settings = Settings(
        models=ModelSettings(),
        agent=AgentSettings(),
        server=server_settings or ServerSettings(),
    )
    app = create_app(
        settings=settings,
        providers=ProviderContext(llm=active_llm, search=_DummySearch()),
    )
    return TestClient(app), active_llm


def _direkt_request(**overrides: object) -> EditorSuggestRequestData:
    base = {
        "block_text": "Es wurde beschlossen, das Sortiment zu erweitern.",
        "block_markdown": "",
        "background": "# Bericht\nHintergrundtext.",
        "instruction": "Mach den Satz aktiver.",
        "global_instruction": "",
        "current_suggestion_markdown": "",
        "refinement_instruction": "",
        "snippet": "",
        "locale": "de",
    }
    base.update(overrides)
    return EditorSuggestRequestData(**base)  # type: ignore[arg-type]


# -- prompt building ------------------------------------------------------- #


def test_direkt_prompt_has_english_rules_and_language_preservation() -> None:
    prompt = build_editor_suggest_prompt(_direkt_request())

    assert "rewrite EXACTLY ONE highlighted passage" in prompt
    assert "write \"rewritten_text\" in that same language" in prompt
    assert "interface language: German" in prompt
    assert "<paragraph_instruction>" in prompt
    assert "<rewrite>" in prompt and "Es wurde beschlossen" in prompt
    # No global/template SECTIONS when not provided (the rules text still
    # references the tag names, hence the trailing newline check).
    assert "<global_instruction>\n" not in prompt
    assert "<template>\n" not in prompt


def test_prompt_prioritizes_comment_before_background_for_long_passage() -> None:
    block = (
        "Der Ursprung der Verwechslung liegt vermutlich in der schnellen "
        "Ausdifferenzierung der Google-KI-Architektur, bei der mehrere Marken "
        "und Modellnamen parallel verwendet werden."
    )
    prompt = build_editor_suggest_prompt(
        _direkt_request(
            background="# Report\n" + ("Kontext. " * 400),
            block_text=block,
            instruction="Der Text ist viel zu lang, kuerze den Absatz zu einem Satz runter!",
        )
    )

    assert prompt.index("\n\n<paragraph_instruction>") < prompt.index("\n\n<rewrite>")
    assert prompt.index("\n\n<rewrite>") < prompt.index("\n\n<background>")
    assert "mandatory, not optional commentary" in prompt


_REFERENCE_NOTE = "You are also given the following reference documents"


def test_prompt_places_reference_block_after_rewrite_before_background() -> None:
    block = render_reference_documents(
        [ReferenceDocument(label="alpha", content="Paris.", page_count=None, size_bytes=None)]
    )
    prompt = build_editor_suggest_prompt(_direkt_request(), reference_block=block)

    assert prompt.index("<rewrite>\n") < prompt.index(_REFERENCE_NOTE)
    assert prompt.index(_REFERENCE_NOTE) < prompt.index("<background>\n")
    assert "[1] alpha" in prompt


def test_prompt_without_attachments_omits_reference_block() -> None:
    prompt = build_editor_suggest_prompt(_direkt_request())

    assert _REFERENCE_NOTE not in prompt


def test_system_rules_describe_reference_documents_as_non_authoritative() -> None:
    prompt = build_editor_suggest_prompt(_direkt_request())

    assert "<reference_documents> block, when present, is citable source material only" in prompt


def test_route_includes_attachments_in_prompt() -> None:
    client, llm = _make_client()

    response = client.post(
        "/v1/editor/suggest",
        json={
            "block_text": "Es wurde beschlossen, das Sortiment zu erweitern.",
            "instruction": "Nutze Fakten aus [1].",
            "locale": "de",
            "attachments": [
                {"label": "alpha", "content": "Paris ist die Hauptstadt.", "page_count": 4}
            ],
        },
    )

    assert response.status_code == 200
    assert llm.prompt is not None
    assert "<reference_documents>" in llm.prompt
    assert "[1] alpha (pages: 4)" in llm.prompt


def test_prompt_uses_markdown_rewrite_when_available() -> None:
    prompt = build_editor_suggest_prompt(
        _direkt_request(
            block_text="Google Gemini bleibt aktiv E10.",
            block_markdown="**Google Gemini** bleibt aktiv [E10](https://example.test/e10).",
            instruction="Kuerze den Satz, aber erhalte Link und Hervorhebung.",
        )
    )

    assert "<rewrite>\n**Google Gemini** bleibt aktiv [E10](https://example.test/e10)." in prompt
    assert "<rewrite_plaintext>\nGoogle Gemini bleibt aktiv E10." in prompt
    assert "Preserve meaning, technical terms, proper nouns, numbers, quotes, Markdown formatting, Markdown links" in prompt


def test_sammeln_prompt_includes_global_instruction_and_template() -> None:
    prompt = build_editor_suggest_prompt(
        _direkt_request(
            global_instruction="Formuliere durchgaengig auf Sie um.",
            snippet="Sachlicher Behoerdenstil.",
            locale="en",
        )
    )

    assert "<global_instruction>" in prompt and "auf Sie um" in prompt
    assert "<template>" in prompt and "Behoerdenstil" in prompt
    assert "interface language: English" in prompt


def test_refinement_prompt_uses_current_suggestion_as_revision_target() -> None:
    prompt = build_editor_suggest_prompt(
        _direkt_request(
            block_text="Der urspruengliche Absatz enthaelt drei Details.",
            current_suggestion_markdown="Der Absatz enthaelt drei Details und bleibt zu lang.",
            refinement_instruction="Noch knapper, aber die drei Details behalten.",
        )
    )

    assert "<rewrite>\nDer urspruengliche Absatz enthaelt drei Details." in prompt
    assert "<current_suggestion>\nDer Absatz enthaelt drei Details und bleibt zu lang." in prompt
    assert "<refinement_instruction>\nNoch knapper" in prompt
    assert "revise the current suggestion instead of starting over" in prompt


# -- request parsing ------------------------------------------------------- #


def test_parse_payload_allows_refinement_without_original_instruction() -> None:
    request = parse_editor_suggest_payload(
        {
            "block_text": "Der markierte Absatz enthaelt die Faktenbasis.",
            "current_suggestion_markdown": "Der aktuelle Vorschlag ist noch zu lang.",
            "refinement_instruction": "Kuerze ihn auf einen Satz.",
            "locale": "de",
        },
        max_background_chars=10_000,
        max_text_chars=2_000,
    )

    assert request.instruction == ""
    assert request.current_suggestion_markdown == "Der aktuelle Vorschlag ist noch zu lang."
    assert request.refinement_instruction == "Kuerze ihn auf einen Satz."


def test_parse_payload_requires_refinement_fields_as_pair() -> None:
    base = {"block_text": "Ein Absatz.", "locale": "de"}

    with pytest.raises(ValueError, match="refinement_instruction must be provided"):
        parse_editor_suggest_payload(
            {**base, "current_suggestion_markdown": "Vorschlag."},
            max_background_chars=10_000,
            max_text_chars=2_000,
        )

    with pytest.raises(ValueError, match="current_suggestion_markdown must be provided"):
        parse_editor_suggest_payload(
            {**base, "refinement_instruction": "Noch knapper."},
            max_background_chars=10_000,
            max_text_chars=2_000,
        )


# -- response parsing ------------------------------------------------------ #


def test_parse_response_accepts_clean_json() -> None:
    result = parse_editor_suggest_response(_VALID_JSON)
    assert result.rewritten_text == "Die Arbeitsgruppe erweitert das Sortiment."
    assert result.changes == ["aktiv"]
    assert result.warnings == []


def test_parse_response_extracts_json_with_prose_and_warns() -> None:
    result = parse_editor_suggest_response(
        'Sure, here is the result:\n' + _VALID_JSON + '\nHope it helps!'
    )
    assert result.rewritten_text == "Die Arbeitsgruppe erweitert das Sortiment."
    assert any("extra text" in w for w in result.warnings)


def test_parse_response_rejects_non_json() -> None:
    with pytest.raises(EditorSuggestError):
        parse_editor_suggest_response("not json at all")


def test_parse_response_rejects_empty_rewritten_text() -> None:
    with pytest.raises(EditorSuggestError):
        parse_editor_suggest_response('{"rewritten_text":"  ","changes":[]}')


def test_result_from_parsed_requires_rewritten_text() -> None:
    with pytest.raises(EditorSuggestError):
        result_from_parsed({"changes": ["x"]})


def test_validate_result_rejects_ignored_one_sentence_shortening() -> None:
    request = _direkt_request(
        block_text=(
            "Dieser Absatz ist lang und enthaelt mehrere Details. Er soll im "
            "Editor drastisch gekuerzt werden, weil der Nutzer nur eine knappe "
            "Aussage benoetigt."
        ),
        instruction="Kuerze den Absatz zu einem Satz.",
    )
    result = result_from_parsed({
        "rewritten_text": (
            "Dieser Absatz ist lang und enthaelt mehrere Details, die weiterhin "
            "ausfuehrlich wiederholt werden. Er soll im Editor drastisch "
            "gekuerzt werden, bleibt aber dennoch zu umfangreich."
        ),
        "changes": [],
    })

    assert [issue.code for issue in validate_editor_suggest_result(request, result)] == [
        "sentence_limit",
        "not_shortened",
    ]


def test_validate_one_sentence_allows_german_dates_and_decimals() -> None:
    request = _direkt_request(
        block_text=(
            "Der Absatz enthaelt mehrere Details ueber den Stand vom 17. Mai 2026, "
            "die Entwicklung von Gemini 3.1 und einen Qualitaetsscore von 0.52."
        ),
        instruction="Kuerze den Absatz zu einem Satz.",
    )
    result = result_from_parsed({
        "rewritten_text": "Stand 17. Mai 2026 ist Gemini 3.1 aktiv und der Qualitaetsscore liegt bei 0.52.",
        "changes": ["auf einen Satz gekuerzt"],
    })

    assert validate_editor_suggest_result(request, result) == []


def test_validation_warnings_are_human_readable_without_raw_codes() -> None:
    request = _direkt_request(
        block_text="Der Originalabsatz war sehr lang und enthaelt mehrere Details.",
        current_suggestion_markdown="Der Vorschlag ist noch immer laenger als gewuenscht.",
        refinement_instruction="Noch kuerzer.",
    )
    result = result_from_parsed({
        "rewritten_text": "Der Vorschlag ist noch immer laenger als gewuenscht und wiederholt sich.",
        "changes": [],
    })

    issues = validate_editor_suggest_result(request, result)
    # The validator still flags it (it drives the retry and is logged) ...
    assert [issue.code for issue in issues] == ["not_shortened"]
    # ... but the user never sees the raw code; the warning is plain language.
    for locale in ("de", "en"):
        warnings = warnings_for_validation_issues(issues, locale=locale)
        assert warnings
        assert all("not_shortened" not in warning for warning in warnings)
    assert any(
        "kuerzer" in warning.lower()
        for warning in warnings_for_validation_issues(issues, locale="de")
    )


def test_validate_refinement_shortening_compares_against_current_suggestion() -> None:
    request = _direkt_request(
        block_text="Der Originalabsatz war sehr lang und enthaelt mehrere Details.",
        current_suggestion_markdown="Der Vorschlag ist noch immer laenger als gewuenscht.",
        refinement_instruction="Noch kuerzer.",
    )
    result = result_from_parsed({
        "rewritten_text": "Der Vorschlag ist noch immer laenger als gewuenscht und wiederholt sich.",
        "changes": [],
    })

    assert [issue.code for issue in validate_editor_suggest_result(request, result)] == [
        "not_shortened",
    ]


# -- background windowing -------------------------------------------------- #


def test_clamp_background_keeps_short_report() -> None:
    report = "# Title\nshort body"
    out, truncated = clamp_background(report, "short body", max_chars=1000)
    assert out == report
    assert truncated is False


def test_clamp_background_windows_around_block_and_keeps_headings() -> None:
    block = "TARGET PARAGRAPH"
    report = "# Heading A\n" + ("filler. " * 2000) + block + (" trailing. " * 2000)
    out, truncated = clamp_background(report, block, max_chars=400)
    assert truncated is True
    assert block in out
    assert "# Heading A" in out
    assert len(out) <= 400 + 64


# -- HTTP route ------------------------------------------------------------ #


def test_route_prompt_json_path_returns_rewrite() -> None:
    client, llm = _make_client()
    response = client.post(
        "/v1/editor/suggest",
        json={
            "block_text": "Es wurde beschlossen, das Sortiment zu erweitern.",
            "background": "# Bericht\nKontext.",
            "instruction": "Mach den Satz aktiver.",
            "locale": "de",
        },
    )
    assert response.status_code == 200
    assert response.json()["improved_text"] == "Die Arbeitsgruppe erweitert das Sortiment."
    assert llm.prompt is not None and "<rewrite>" in llm.prompt
    assert llm.kwargs["max_output_tokens"] == 4000


def test_route_retries_when_model_ignores_sentence_limit() -> None:
    too_long = (
        '{"rewritten_text":"Die Arbeitsgruppe erweitert das Sortiment. '
        'Sie plant weitere Schritte.","changes":["gekürzt"]}'
    )
    valid = '{"rewritten_text":"Die Arbeitsgruppe erweitert das Sortiment.","changes":["auf einen Satz gekuerzt"]}'
    client, llm = _make_client(llm=_CapturingLLM(response=[too_long, valid]))

    response = client.post(
        "/v1/editor/suggest",
        json={
            "block_text": (
                "Es wurde beschlossen, das Sortiment zu erweitern, weitere "
                "Sortimentsgruppen zu pruefen und die Roadmap zu aktualisieren."
            ),
            "instruction": "Kuerze den Absatz zu einem Satz.",
            "locale": "de",
        },
    )

    assert response.status_code == 200
    assert response.json()["improved_text"] == "Die Arbeitsgruppe erweitert das Sortiment."
    assert llm.prompt is not None and "<validation_feedback>" in llm.prompt


def test_route_structured_path_uses_complete_structured() -> None:
    client, llm = _make_client(llm=_CapturingLLM(structured=True))
    response = client.post(
        "/v1/editor/suggest",
        json={
            "block_text": "Ein langer Absatz.",
            "instruction": "Mach den Satz aktiver.",
            "locale": "de",
        },
    )
    assert response.status_code == 200
    assert llm.structured_called is True
    assert response.json()["improved_text"] == "Die Arbeitsgruppe erweitert das Sortiment."


def test_route_requires_an_instruction() -> None:
    client, _llm = _make_client()
    response = client.post(
        "/v1/editor/suggest",
        json={"block_text": "Ein Absatz ohne Anweisung."},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_route_rejects_sensitive_text_before_model_call() -> None:
    client, llm = _make_client()
    response = client.post(
        "/v1/editor/suggest",
        json={
            "block_text": "token=sk-proj-abcdefghijklmnopqrstuvwxyz123456",
                "instruction": "Mach den Satz aktiver.",
        },
    )
    assert response.status_code == 400
    assert "secret material" in response.json()["error"]["message"]
    assert llm.prompt is None


def test_route_surfaces_model_json_failure() -> None:
    client, _llm = _make_client(llm=_CapturingLLM(response="not json"))
    response = client.post(
        "/v1/editor/suggest",
            json={"block_text": "Ein Absatz.", "instruction": "Mach den Satz aktiver."},
    )
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "server_error"


def test_route_routes_by_selected_tier() -> None:
    models = ModelSettings(
        reasoning_model="default-model",
        tier_high_model="big-model",
        tier_mid_model="mid-model",
    )
    client, llm = _make_client(llm=_CapturingLLM(models=models))

    high = client.post(
        "/v1/editor/suggest",
        json={
            "block_text": "Ein Absatz.",
            "instruction": "Mach den Satz aktiver.",
            "agent_overrides": {"model_tier": "high"},
        },
    )
    assert high.status_code == 200
    assert llm.kwargs["model"] == "big-model"

    default = client.post(
        "/v1/editor/suggest",
        json={"block_text": "Ein Absatz.", "instruction": "Mach den Satz aktiver."},
    )
    assert default.status_code == 200
    # direct_chat defaults to the mid tier.
    assert llm.kwargs["model"] == "mid-model"


def test_route_uses_bearer_auth_when_configured() -> None:
    client, _llm = _make_client(server_settings=ServerSettings(api_key="secret-token-123"))
    missing = client.post(
        "/v1/editor/suggest",
        json={"block_text": "Ein Absatz.", "instruction": "Mach den Satz aktiver."},
    )
    allowed = client.post(
        "/v1/editor/suggest",
        json={"block_text": "Ein Absatz.", "instruction": "Mach den Satz aktiver."},
        headers={"Authorization": "Bearer secret-token-123"},
    )
    assert missing.status_code == 401
    assert allowed.status_code == 200


def test_structured_output_error_surfaces_as_502() -> None:
    class _FailingStructuredLLM(_CapturingLLM):
        def complete_structured(self, prompt: str, **kwargs: object) -> StructuredLLMResponse:
            raise AgentStructuredOutputError("model", "schema", "boom")

    client, _llm = _make_client(llm=_FailingStructuredLLM(structured=True))
    response = client.post(
        "/v1/editor/suggest",
        json={"block_text": "Ein Absatz.", "instruction": "Kuerzen."},
    )
    assert response.status_code == 502
