# Contributing

## Scope

How to set up a development environment, propose a change, and keep it aligned with the project's coding standards. Detailed testing guidance lives in [Testing strategy](testing-strategy.md) and [Running tests](running-tests.md); release mechanics live in [Release process](release-process.md).

## Environment setup

Inqtrix targets Python ≥ 3.12 and uses a `src/` layout, so an editable install is the recommended workflow.

```bash
git clone https://github.com/BZandi/inqtrix.git
cd inqtrix

# Option A: uv
uv sync --extra dev
source .venv/bin/activate

# Option B: conda
conda create -n inqtrix python=3.12
conda activate inqtrix
pip install -e ".[dev]"
```

Offline regression check:

```bash
uv run pytest tests/ -v
```

See [Installation](../getting-started/installation.md) for the full bootstrap.

## Change workflow

1. **Discuss before coding** when the change touches a provider interface, strategy contract, or HTTP surface. Small behavioural bugs can go straight to a PR.
2. **Start from a clean branch.** Never commit to `main` directly.
3. **Follow the coding standards.** See [Coding standards](coding-standards.md) for Python version, docstring format, logging rules, and type hints.
4. **Write tests.** Every code-path branch gets a test. Mocks are preferred over real network calls; see [Testing strategy](testing-strategy.md) for the four-layer pyramid.
5. **Keep the docs honest.** Change docstrings and `docs/**` pages together with the code. Link-check locally before pushing.
6. **Do not commit secrets.** `.env`, credentials, cassettes with real keys — never. The sanitization helpers in `tests/fixtures/sanitize.py` and the protective scan in `tests/replay/test_sanitization.py` exist to catch mistakes.

## Docstring and inline-comment style

- Google-style docstrings with `Args`, `Returns`, `Raises`, optional `Example` — the reference is `AzureOpenAILLM.__init__` in `src/inqtrix/providers/azure.py`.
- `Args` entries are semantic — they explain what the parameter steers, the value range, the default rationale, interactions, and consequences of misuse. They do **not** repeat the type (that comes from the signature).
- Pydantic fields use `Field(description=...)` **and** an attribute docstring immediately after the field definition. The two strings are kept identical word-for-word. See [Coding standards](coding-standards.md) for the full rule set.
- Inline comments explain *why*, never just *what*. Narrative comments ("`# increment counter`") are not accepted.
- No emojis in code, logs, docs, or commit messages unless explicitly requested.

## Commit and PR conventions

- Commit subject line in English, imperative mood, ≤ 72 characters.
- Body paragraph when the diff is non-trivial — explain motivation, not mechanics.
- One focused change per PR; do not bundle refactors with feature additions unless the refactor is required by the feature.
- If the change touches an architectural contract, document the public behaviour in `docs/**` and keep the maintainer memory in sync locally.

## Security and legal

- Never commit real API keys, cassette files with live data, or customer-identifying information.
- The project is explicitly experimental; do not add marketing claims to documentation.
- The Apache License 2.0 applies to all contributions; by opening a PR you confirm you have the right to license the content under it.

## Related docs

- [Coding standards](coding-standards.md)
- [Testing strategy](testing-strategy.md)
- [Running tests](running-tests.md)
- [Release process](release-process.md)
- [Docs maintenance](docs-maintenance.md)
