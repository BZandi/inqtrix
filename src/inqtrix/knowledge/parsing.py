"""Document parsing behind a Baukasten port (file-to-text for ingestion).

The parser ladder mirrors the auth and object-store ports: standards first,
nothing hardwired.

* :class:`MarkItDownParser` — the DEFAULT. Pure-Python conversion to
  Markdown (Microsoft MarkItDown, MIT): no ML models, no cloud
  dependency, always available. Handles PDF (text layer), DOCX, PPTX,
  XLSX, HTML, Markdown, plain text.
* Azure Document Intelligence — the optional cloud tier for scans and
  complex table layouts; a drop-in behind this port when a deployment
  configures it (not built until needed).
* Docling — the documented air-gapped option (local layout models);
  deliberately NOT the default because it hosts model weights.

Parsing failures are loud: a file that cannot be converted raises
:class:`DocumentParseError`, mapped to a clear client error — never a
silently empty document.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod

log = logging.getLogger("inqtrix")


class DocumentParseError(RuntimeError):
    """Raised when a file cannot be converted to text."""


class DocumentParser(ABC):
    """Port for converting an uploaded file into ingestion text."""

    @property
    @abstractmethod
    def parser_id(self) -> str:
        """Stable identifier recorded in document metadata."""

    @abstractmethod
    def parse(self, *, file_name: str, content: bytes) -> str:
        """Convert *content* to text (Markdown preferred).

        Args:
            file_name: Original filename — the extension steers format
                detection.
            content: Raw file bytes.

        Raises:
            DocumentParseError: When conversion fails or produces no
                text (scanned PDFs without a text layer being the
                classic case — the error says so).
        """


class MarkItDownParser(DocumentParser):
    """Default parser: MarkItDown file-to-Markdown conversion.

    Construction imports the converter stack once; per-call work is
    pure CPU parsing with no network and no model inference.
    """

    def __init__(self) -> None:
        from markitdown import MarkItDown

        # Plugins stay off: deterministic, dependency-free conversion.
        self._converter = MarkItDown(enable_plugins=False)

    @property
    def parser_id(self) -> str:
        """``"markitdown"``."""
        return "markitdown"

    def parse(self, *, file_name: str, content: bytes) -> str:
        """Convert the file to Markdown text."""
        stream = io.BytesIO(content)
        try:
            result = self._converter.convert_stream(
                stream, file_extension=_extension_of(file_name)
            )
        except Exception as exc:  # noqa: BLE001 — normalized below, visibly
            log.warning(
                "Datei-Parsing fehlgeschlagen (error_type=%s)",
                type(exc).__name__,
            )
            raise DocumentParseError(
                f"Datei {file_name!r} konnte nicht konvertiert werden: {exc}"
            ) from exc
        text = (result.text_content or "").strip()
        if not text:
            raise DocumentParseError(
                f"Datei {file_name!r} ergab keinen Text (gescanntes PDF "
                "ohne Textebene?)"
            )
        return text


def _extension_of(file_name: str) -> str:
    """Dot-prefixed lowercase extension (MarkItDown's detection hint)."""
    dot = file_name.rfind(".")
    return file_name[dot:].lower() if dot >= 0 else ""
