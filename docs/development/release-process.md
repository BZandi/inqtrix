# Release process

## Scope

The release process is maintainer-owned. This page exists so contributors know what to expect and what they can prepare, not to prescribe a specific cadence.

## Today's state

The repository is marked experimental and is at `0.2.0`, a placeholder. The version is defined once as `__version__` in `src/inqtrix/__init__.py`; `pyproject.toml` derives the package version from it (Hatchling dynamic version). No version has been tagged or published yet. A formal release process — signing, PyPI publication, GitHub Releases, change categorisation — is a dedicated follow-up task.

## Automation status

Source-controlled GitHub Actions automation is intentionally disabled. There
are no active workflow YAML files under `.github/workflows`; the previous CI
and image-release definitions are retained byte-for-byte as non-executable
audit snapshots in
[`docs/archive/github-actions/`](../archive/github-actions/README.md).

The archived definitions are not an alternative execution path and must not be
restored unchanged. Reactivation requires a separate review of triggers,
least-privilege permissions, immutable source and image digests, per-platform
scanning, protected release approval, and required-check configuration. This
source-tree status does not assert that repository-level GitHub Actions
settings are disabled or that queued, running, or historical runs have been
administratively blocked.

## Bumping the version

The version lives in **one place**: `__version__` in [`src/inqtrix/__init__.py`](../../src/inqtrix/__init__.py). `pyproject.toml` declares `dynamic = ["version"]` with a `[tool.hatch.version]` source pointing at that file, so Hatchling derives the package version (and the wheel metadata) from that single line. Verify with:

```bash
# uv
uv run python -c "import inqtrix; print(inqtrix.__version__)"
uv run python -c "from importlib.metadata import version; print(version('inqtrix'))"

# or, after `python -m pip install -e .`
python -c "import inqtrix; print(inqtrix.__version__)"
python -c "from importlib.metadata import version; print(version('inqtrix'))"
```

A few surfaces cannot read the Python version and are bumped by hand:

1. **`src/inqtrix/__init__.py`**: `__version__` (the source; `pyproject.toml` derives from it). Run `uv sync` afterwards so the installed metadata and `uv.lock` update.
2. **`README.md`**: the static `Version X.Y.Z` shields.io badge (markdown, with no CI to derive it).
3. **`apps/research-desk/package.json`**: the React app's `version` (npm; internal and not user-facing, but keep it in sync for tidiness).
4. **Docs prose** that names the number: this page's *Today's state* and [`docs/reference/changelog.md`](../reference/changelog.md).

There is currently no active source-controlled CI or release pipeline, so the
version is bumped manually at milestones.

## What contributors can do

- Keep `docs/reference/changelog.md` in the `Unreleased` section up to date when a PR lands a user-visible change.
- Write PR descriptions so that a future changelog entry can be assembled quickly (motivation, user-facing change, migration notes if any).
- When a contract changes (provider method signature, Settings field name, HTTP body schema), call it out explicitly in the PR description so the future release note can flag it.

## What a maintainer-driven release step would look like (indicative)

1. Pick the new version number following SemVer (`0.x.y` until the API stabilises; `1.y.z` once semver applies strictly).
2. Move `Unreleased` entries in `docs/reference/changelog.md` into a dated section, grouped by `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.
3. Bump the version (see [Bumping the version](#bumping-the-version)): edit `__version__` in `src/inqtrix/__init__.py`, run `uv sync`, then update the README badge and `apps/research-desk/package.json` to match.
4. Tag the commit with `vX.Y.Z`, push the tag.
5. Create a GitHub Release pointing at the tag, paste the changelog section as the body.
6. Publish to PyPI if applicable (`uv build`, inspection, `uv publish`).
7. Refresh and verify third-party notices before packaging distributable assets:
   ```bash
   # Locked uv release environment
   uv sync --all-extras
   npm ci
   uv run python scripts/generate_third_party_notices.py
   uv run python scripts/generate_third_party_notices.py --check

   # The generator itself also runs in a standard pip environment:
   python -m pip install -e ".[dev]"
   python scripts/generate_third_party_notices.py
   python scripts/generate_third_party_notices.py --check
   ```

   The uv path remains authoritative for refreshing `uv.lock`; the plain
   Python commands document that the project scripts do not require uv as
   their runtime.

Automated release publication remains disabled and is explicitly out of scope
until the reactivation review described above has been completed.

## Related docs

- [Changelog](../reference/changelog.md)
- [Contributing](contributing.md)
