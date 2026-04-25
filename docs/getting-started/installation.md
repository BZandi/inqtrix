# Installation

## Scope

How to install Inqtrix for local development and for consumer-style use, including the `src/` layout gotcha and the dev-extras.

## Requirements

- Python 3.12 or newer.
- A package manager: [`uv`](https://github.com/astral-sh/uv) is recommended; `conda` and plain `pip` also work.
- Credentials for at least one LLM and one search provider if you intend to run real research.

## From a fresh clone

Create a dedicated Python 3.12 environment first:

```bash
git clone https://github.com/BZandi/inqtrix.git
cd inqtrix

# Option A: uv (recommended)
uv sync --extra dev
source .venv/bin/activate

# Option B: conda
conda create -n inqtrix python=3.12
conda activate inqtrix
pip install -e ".[dev]"
```

Editable install is the recommended workflow for local development and testing:

- `pip install -e .` — editable install for normal local use.
- `pip install -e ".[dev]"` — editable install plus test dependencies.
- `uv sync --extra dev` — equivalent for `uv` users.

Code changes under `src/inqtrix/` are picked up immediately without re-installing after every edit.

## The `src/` layout caveat

The repository uses a `src/` layout, so a plain clone is not importable by default. In other words: being inside the project directory is not enough on its own for `import inqtrix`, `python -m inqtrix`, or the `inqtrix-parity` entry point to work reliably. Installing the project in editable mode links the environment to your working tree, which is exactly what you want while developing.

If you only want a quick experiment without installing, you can set `PYTHONPATH=src` manually, but that is a temporary workaround rather than the recommended setup.

## First local check

This is the fastest offline regression check after cloning. It runs the local `pytest` suite only, does **not** call real model or search providers, and does not require API keys.

```bash
uv run pytest tests/ -v
```

Use `uv run pytest tests/ --collect-only -q` when you need the exact current count; the suite grows as provider and server coverage expands.

## Consumer-style install (non-editable)

For a consumer-style install outside active development, use a normal non-editable install instead:

```bash
pip install .

# Editable install without test extras
pip install -e .

# Editable install with test extras
pip install -e ".[dev]"
```

## Next steps

- [First research run](first-research-run.md) — run a live question against your own providers.
- [Library mode](../deployment/library-mode.md) — embed in your own script.
- [Web server mode](../deployment/webserver-mode.md) — run as an HTTP service.

## Related docs

- [Overview](overview.md)
- [Running tests](../development/running-tests.md)
- [Providers overview](../providers/overview.md)
