# Coding standards

## Scope

The binding coding conventions for the Inqtrix repository. Deviations are called out in code review; new modules are expected to match the conventions out of the gate.

## Language and version

- **Python ≥ 3.11** for the library itself (matches `pyproject.toml`).
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
- Describe the current technical cause and invariant without referring to an
  internal plan, priority, review, incident, private decision id, branch,
  commit, or the date on which a change was made.
- Dates and versions remain appropriate when executable behaviour or external
  interoperability depends on them, such as protocol versions, schema
  revisions, retention calculations, source provenance, and deterministic
  temporal fixtures.
- No emojis — anywhere.

## Reuse and abstraction boundaries

Use the smallest abstraction that has a demonstrated shared responsibility:

- Search for an existing semantic token, motion contract, type, service, or UI
  primitive before introducing another owner for the same contract.
- Similar appearance alone does not justify sharing. Components with different
  behaviour, lifecycle, or ownership may remain separate.
- A genuinely unique feature composition may remain local while reusing the
  applicable colour, typography, spacing, control, icon, and motion language.
- When the same semantic role or interaction contract appears independently in
  more than one feature, extend an existing primitive or extract one shared
  implementation instead of copying it.
- Do not create global abstractions for speculative reuse. New shared tokens
  and variants name their purpose and document their intended consumers.
- Prefer deliberate variants over feature-owned forks of a shared primitive.
  If consumers need different contracts, keep them separate rather than
  accumulating unrelated conditionals.

For the Research Desk, the ownership flow is:

```text
Design tokens and motion contracts
        ↓
Shared UI primitives
        ↓
Feature-owned compositions
```

The binding visual roles and extension rules are in
[`apps/research-desk/DESIGN.md`](../../apps/research-desk/DESIGN.md).

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

- `ruff` (rules in `[tool.ruff.lint]` of `pyproject.toml`) is the lint gate. It runs green on `src/` today: `uv run ruff check src/`.
- The enabled set is deliberately small and enforceable: `F` (pyflakes) and `E9` (syntax and IO errors). Both catch defects rather than style. `E9` in particular catches syntax that only parses on a Python newer than `requires-python` promises, which no test on the development interpreter can see.
- `target-version` is **not** set. Ruff derives it from `requires-python`, so the minimum supported Python is stated once.
- Rules beyond this set (`I`, `UP`, `B`, `SIM`, and the pydocstyle `D` family) are not enabled. They would flag a large volume of out-of-scope modules at once, which makes a risk-appropriate review impossible; enabling any of them is its own change with its own review.
- Do not commit code that fails the enabled ruff rules.

## Backwards compatibility

- Removing a public constructor signature, a Settings field, or a `ResearchAgent` method is a deliberate decision, not an automatic one. Redundancy is removed at the root rather than softened with a compatibility layer: no shim module, no alias, no `deprecated=True` flag. Call sites move to the surviving path in the same change, and the tests that pinned the removed behaviour move with them. `AGENTS.md` holds the full procedure, including that the removal decision belongs to the operator.
- Additive changes are always preferred. New fields on `AgentState` must be `NotRequired[...]` and underscore-prefixed when internal.
- Providers must honour the Constructor-First convention: no direct environment-variable reads inside provider modules.

## Git hygiene

- No destructive Git operations without explicit user request (force pushes, history rewrites, hard resets on `main`).
- Never skip pre-commit / hook validation (`--no-verify`, `--no-gpg-sign`) without request.
- One logical change per commit; rebase to squash noise before opening a PR.

## Related docs

- [Contributing](contributing.md)
- [Testing strategy](testing-strategy.md)
- [Docs maintenance](docs-maintenance.md)
