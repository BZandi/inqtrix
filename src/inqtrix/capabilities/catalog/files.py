"""Wave-1 file capability (read-only): server-side text extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from inqtrix.capabilities.contracts import (
    CapabilityContext,
    CapabilityDefinition,
    CapabilityError,
    Effect,
)
from inqtrix.content.ports import FileNotFound
from inqtrix.services.file_service import (
    FileParserUnavailable,
    FileTextExtractionError,
)
from inqtrix.storage.object_store import ObjectStoreError

if TYPE_CHECKING:
    from inqtrix.services.file_service import FileService


class FileTextReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(..., min_length=1)


class FileTextReadOutput(BaseModel):
    file_id: str
    parser_id: str
    text: str


def build_file_capabilities(
    service: "FileService",
) -> list[CapabilityDefinition]:
    """Build the wave-1 file capabilities bound to *service*."""

    async def _read_text(
        payload: FileTextReadInput, context: CapabilityContext
    ) -> FileTextReadOutput:
        try:
            extracted = await service.extract_text(
                payload.file_id, principal=context.principal
            )
        except FileParserUnavailable as exc:
            raise CapabilityError(
                "files.parser_unavailable", str(exc), http_status=501
            ) from exc
        except FileNotFound as exc:
            raise CapabilityError(
                "files.not_found", "Datei nicht gefunden.", http_status=404
            ) from exc
        except ObjectStoreError as exc:
            raise CapabilityError(
                "files.object_store_unavailable",
                "Dateiinhalt nicht abrufbar (Object Store).",
                http_status=502,
            ) from exc
        except FileTextExtractionError as exc:
            raise CapabilityError(
                "files.parse_failed", str(exc), http_status=422
            ) from exc
        return FileTextReadOutput(
            file_id=extracted.file_id,
            parser_id=extracted.parser_id,
            text=extracted.text,
        )

    return [
        CapabilityDefinition(
            id="file.text.read",
            summary="Extract the server-side parsed text of one uploaded file.",
            input_model=FileTextReadInput,
            output_model=FileTextReadOutput,
            effect=Effect.READ,
            idempotent=True,
            handler=_read_text,
        ),
    ]
