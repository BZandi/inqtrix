# Release process

## Scope

The release process is maintainer-owned. This page exists so contributors know what to expect and what they can prepare, not to prescribe a specific cadence.

## Today's state

The repository is marked experimental and carries `version = "0.1.0"` in `pyproject.toml` as a placeholder. No version has been tagged or published yet. A formal release process — signing, PyPI publication, GitHub Releases, change categorisation — is a dedicated follow-up task.

## What contributors can do

- Keep `docs/reference/changelog.md` in the `Unreleased` section up to date when a PR lands a user-visible change.
- Write PR descriptions so that a future changelog entry can be assembled quickly (motivation, user-facing change, migration notes if any).
- When a contract changes (provider method signature, Settings field name, HTTP body schema), call it out explicitly in the PR description so the future release note can flag it.

## What a maintainer-driven release step would look like (indicative)

1. Pick the new version number following SemVer (`0.x.y` until the API stabilises; `1.y.z` once semver applies strictly).
2. Move `Unreleased` entries in `docs/reference/changelog.md` into a dated section, grouped by `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.
3. Update `pyproject.toml` with the new version.
4. Tag the commit with `vX.Y.Z`, push the tag.
5. Create a GitHub Release pointing at the tag, paste the changelog section as the body.
6. Publish to PyPI if applicable (`uv build`, inspection, `uv publish`).

Automating this workflow is explicitly out of scope today.

## Related docs

- [Changelog](../reference/changelog.md)
- [Contributing](contributing.md)
