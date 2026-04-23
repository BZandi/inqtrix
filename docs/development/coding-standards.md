# Coding standards

## Scope

The binding coding conventions for the Inqtrix repository. Deviations are called out in code review; new modules are expected to match the conventions out of the gate.

## Language and version

- **Python ≥ 3.12** for the library itself (matches `pyproject.toml`).
- **Type hints** on every new function, method, and class. Prefer `TypedDict`, `Literal`, and `Protocol` where they add meaningful constraints; avoid `Any`.
- **English** for all committed artefacts: code, comments, docstrings, variable names, README, `docs/**`, commit messages, test IDs. Exceptions (explicit, not accidental): LLM prompt templates in `src/inqtrix/prompts.py`, user-facing German HTTP strings, demo questions in `examples/`.

## Docstrings

- **Google style** (`Args`, `Returns`, `Raises`, optional `Example`). The reference implementation is `AzureOpenAILLM.__init__` in `src/inqtrix/providers/azure.py`.
- **Semantic, not tautological.** A parameter description explains what the parameter steers, the value range, the default rationale, interactions with other parameters, and consequences of misuse. It does not repeat the type (type hints are the source of truth).
- **Class docstrings** additionally cover use-case (when to pick this class), lifecycle obligations (singleton vs per-run, thread/async safety), and known limitations.
- **Examples** are mandatory for non-trivial constructors and for methods with complex argument combinations. They never replace the `Args` section.

### Forbidden anti-patterns

- Tautological echoes (`foo: The foo.`).
- "See ``Settings`` for details." as the sole description.
- Code-only examples without a textual `Args` block.
- Listing the type in prose (`"str — a string"`); the type hint carries it.
- `See also` links used as a stand-in for an actual description.

## Pydantic models

Every field in a `BaseModel` / `BaseSettings` class has both:

- `Field(description="...")` so `model_json_schema()` exports carry the description (FastAPI OpenAPI, parity tooling).
- An attribute docstring immediately after the field definition so Pylance / VS Code hover shows the same string.

```python
field: Type = Field(
    default=...,
    alias="ENV_NAME",
    description=(
        "Meaning. Value range. Default rationale. "
        "Interactions with other fields."
    ),
)
"""Meaning. Value range. Default rationale. Interactions with other fields."""
```

The two strings are **word-for-word identical**. Do not summarise, translate, or paraphrase one of them. `model_config` must not set `use_attribute_docstrings=True`.

## Inline comments

- Comments explain *why*, not *what*. The code itself tells the reader *what*.
- No narrative comments (`# increment counter`, `# loop over items`, `# return result`).
- Block comments above module-level constants name their source (provider docs link, spec section, audit log) when the value is not self-explanatory.
- No emojis — anywhere.

## Logging and secrets

- Use `logging.getLogger("inqtrix")`.
- Route every potentially sensitive message through `sanitize_log_message(...)` from `runtime_logging.py` before logging.
- Every fallback path emits a `log.warning(...)` **and** an iteration-log marker. "No Silent Fallbacks" is Design Principle 1.
- Never log raw API keys, tokens, endpoints with embedded credentials, or full prompt bodies.

## Test discipline

- Every new branch in a node, strategy, or provider has a unit test.
- Replay tests use VCR cassettes or `botocore.Stubber`; see [Testing strategy](testing-strategy.md).
- Cassettes are scrubbed before commit (`tests/fixtures/sanitize.py` + protective scan in `tests/replay/test_sanitization.py`).
- When a test changes a logger or any other global state, add an `autouse=True` reset fixture to avoid test-order pollution.
- Do not add `time.sleep` in tests; the replay conftest stubs backoff sleeps to keep the suite fast.

## Linter / formatter

- `ruff` (rules in `pyproject.toml`) is the primary lint gate.
- Ruff's `D` (pydocstyle-equivalent) selectors are **recommended** but not globally enforced today (see ADR-E in the internal notes). Enabling them globally would flag many out-of-scope modules (`state.py`, `server/**`, `parity/**`); a dedicated clean-up task owns that transition.
- Do not commit code that fails the currently enabled ruff rules.

## Backwards compatibility

- Never remove or rename a public constructor signature, a Settings field, or a `ResearchAgent` method without a deprecation cycle.
- Additive changes are always preferred. New fields on `AgentState` must be `NotRequired[...]` and underscore-prefixed when internal (see ADR-MS-6).
- Providers must honour the Constructor-First convention: no direct environment-variable reads inside provider modules.

## Git hygiene

- No destructive Git operations without explicit user request (force pushes, history rewrites, hard resets on `main`).
- Never skip pre-commit / hook validation (`--no-verify`, `--no-gpg-sign`) without request.
- One logical change per commit; rebase to squash noise before opening a PR.

## Related docs

- [Contributing](contributing.md)
- [Testing strategy](testing-strategy.md)
- [Docs maintenance](docs-maintenance.md)
