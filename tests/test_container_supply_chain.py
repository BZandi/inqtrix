"""Static release contracts for first-party container images."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOWS = _ROOT / ".github" / "workflows"
_WORKFLOW_ARCHIVE = _ROOT / "docs" / "archive" / "github-actions"


def test_source_tree_contains_no_active_github_workflow_yaml() -> None:
    """A branch push cannot execute a workflow defined by this checkout."""
    active = [
        path
        for pattern in ("*.yml", "*.yaml")
        for path in _WORKFLOWS.glob(pattern)
    ]
    assert active == []


def test_disabled_workflow_snapshots_match_the_documented_checksums() -> None:
    """The quarantine remains immutable audit evidence, not executable YAML."""
    expected = {
        "ci.yml.disabled": (
            "1c2789f26b65316c8fac2d132efc0f27"
            "aa0cd5e27afababd52e9958cd3973dc2"
        ),
        "release-images.yml.disabled": (
            "6d7ef47243e77861fa63534c548973a9c"
            "a1dc0556cb290231d2842733ec59c85"
        ),
    }
    readme = (_WORKFLOW_ARCHIVE / "README.md").read_text(encoding="utf-8")

    assert {
        path.name
        for path in _WORKFLOW_ARCHIVE.iterdir()
        if path.is_file() and path.name != "README.md"
    } == set(expected)
    for filename, digest in expected.items():
        path = _WORKFLOW_ARCHIVE / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert digest in readme

    assert "not supported execution templates" in readme
    assert "arm64 image was not scanned" in readme


def test_production_container_references_are_qualified_and_immutable() -> None:
    """Compose and bundled Helm defaults reject mutable image references."""
    compose = yaml.safe_load(
        (_ROOT / "deploy" / "compose" / "compose.stack.yaml").read_text(
            encoding="utf-8"
        )
    )
    for name, service in compose["services"].items():
        image = service.get("image")
        if image is None:
            continue
        assert "/" in image, f"{name} uses an unqualified registry path"
        assert "@sha256:" in image, f"{name} uses a mutable image reference"
        assert not re.search(r":(?:latest|stable)(?:@|$)", image)

    values = yaml.safe_load(
        (_ROOT / "deploy" / "helm" / "inqtrix" / "values.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert values["image"]["allowUnpinned"] is False
    for component in ("postgres", "pgbouncer", "qdrant", "valkey", "s3"):
        image = values[component]["image"]
        match = re.fullmatch(
            r"(?P<repository>[^@\s]+):(?P<tag>[^:@\s]+)"
            r"@(?P<digest>sha256:[0-9a-f]{64})",
            image,
        )
        assert match, f"unexpected image contract for {component}"
        repository = match.group("repository")
        tag = match.group("tag")
        digest = match.group("digest")
        assert "/" in repository
        assert tag not in {"latest", "stable"}
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
