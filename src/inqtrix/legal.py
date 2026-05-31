"""Project licensing metadata for public discovery surfaces."""

PROJECT_NAME = "Inqtrix"
PROJECT_SOURCE_URL = "https://github.com/BZandi/inqtrix"
LICENSE_ID = "AGPL-3.0-only"
COPYRIGHT_NOTICE = "Copyright (c) 2026 Babak Zandi."
ATTRIBUTION_NOTICE = (
    "Inqtrix - Copyright (c) 2026 Babak Zandi - "
    "https://github.com/BZandi/inqtrix"
)
WARRANTY_NOTICE = (
    "This software is provided without warranty under AGPL-3.0-only; "
    "see LICENSE for details."
)


def legal_metadata() -> dict[str, str]:
    """Return stable project legal metadata for public discovery endpoints.

    The HTTP server exposes this payload on ``/health`` so network users
    can find the project source, active license identifier, copyright
    notice, and attribution notice without first locating repository docs.
    These values describe the project itself, so they are intentionally
    independent from the active provider stack.
    """
    return {
        "project": PROJECT_NAME,
        "license": LICENSE_ID,
        "source_url": PROJECT_SOURCE_URL,
        "copyright": COPYRIGHT_NOTICE,
        "notice": ATTRIBUTION_NOTICE,
        "warranty_notice": WARRANTY_NOTICE,
    }
