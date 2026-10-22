import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.core.exceptions import (
    DuplicateRFPError,
    ExtractionNotReadyError,
    ExtractionRunningError,
    InvalidFileError,
    RFPNotFoundError,
    RFPNotReadyError,
)
from app.repositories.base import RFPRepository
from app.schemas.extraction import TenderExtraction
from app.schemas.rfp import ExtractionStatus, RFPRecord, RFPStatus
from app.services.ingestion_service import IngestionService
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

# extension -> Check the file's starting bytes to detect fake file extensions.
FILE_SIGNATURES = {
    ".pdf": b"%PDF-",
    ".docx": b"PK\x03\x04",
}


class RFPService:
    def __init__(
        self,
        repository: RFPRepository,
        ingestion_service: IngestionService,
        qdrant_service: QdrantService,
        extraction_service_factory: Callable[[], object] | None = None,
    ):
        self.settings = get_settings()
        self.repository = repository
        self.ingestion_service = ingestion_service
        self.qdrant_service = qdrant_service

        # A factory, not an instance: building the extractor needs the Gemini key,
        # and the app must still start (and accept uploads) without it.
        self._extraction_service_factory = extraction_service_factory

        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # ---------- read ----------

    def get(self, rfp_id: str) -> RFPRecord:
        record = self.repository.get(rfp_id)

        if record is None:
            raise RFPNotFoundError(f"RFP {rfp_id} not found.")

        return record

    def list_all(self) -> list[RFPRecord]:
        return self.repository.list_all()

    def ensure_ready(self, rfp_ids: list[str]) -> None:
        for rfp_id in rfp_ids:
            record = self.get(rfp_id)

            if record.status != RFPStatus.READY:
                raise RFPNotReadyError(
                    f"RFP {rfp_id} is not ready (status: {record.status})."
                )

    # ---------- upload ----------

    def create_from_upload(self, filename: str, content: bytes) -> RFPRecord:
        extension = Path(filename).suffix.lower()

        if extension not in FILE_SIGNATURES:
            raise InvalidFileError("Only PDF and DOCX files are supported.")

        if not content:
            raise InvalidFileError("Uploaded file is empty.")

        if len(content) > self.settings.max_upload_bytes:
            raise InvalidFileError(
                f"File size must be less than {self.settings.max_upload_mb} MB."
            )

        if not content.startswith(FILE_SIGNATURES[extension]):
            raise InvalidFileError("File content does not match its extension.")

        content_hash = hashlib.sha256(content).hexdigest()

        for existing in self.repository.list_all():
            if (
                existing.content_hash == content_hash
                and existing.status != RFPStatus.FAILED
            ):
                raise DuplicateRFPError(
                    f"This file was already uploaded (rfp_id: {existing.rfp_id})."
                )

        rfp_id = str(uuid4())
        stored_path = self.settings.upload_dir / f"{rfp_id}{extension}"
        stored_path.write_bytes(content)

        record = RFPRecord(
            rfp_id=rfp_id,
            filename=filename,
            stored_path=str(stored_path),
            content_hash=content_hash,
            size_bytes=len(content),
            status=RFPStatus.PROCESSING,
        )

        return self.repository.create(record)

    # ---------- ingestion (run in background) ----------

    def process(self, rfp_id: str) -> None:
        record = self.get(rfp_id)

        try:
            result = self.ingestion_service.ingest(
                file_path=Path(record.stored_path),
                rfp_id=rfp_id,
                filename=record.filename,
            )
            self.repository.update(
                rfp_id,
                status=RFPStatus.READY,
                pages=result["pages"],
                chunks=result["chunks"],
                error=None,
            )
            logger.info("RFP %s ingested: %s", rfp_id, result)

        except Exception as e:
            logger.exception("Ingestion failed for RFP %s", rfp_id)
            self.repository.update(rfp_id, status=RFPStatus.FAILED, error=str(e))

    def reingest(self, rfp_id: str) -> RFPRecord:
        record = self.get(rfp_id)

        if not Path(record.stored_path).exists():
            raise InvalidFileError("Original file is missing, cannot re-ingest.")

        return self.repository.update(rfp_id, status=RFPStatus.PROCESSING, error=None)

    # ---------- structured extraction ----------

    def request_extraction(self, rfp_id: str, force: bool = False) -> RFPRecord:
        """Decide quickly (inside the HTTP request) whether an extraction run is needed.

        Returns the record. If its extraction_status is RUNNING the caller must schedule
        run_extraction() in the background; DONE means the cached result is still valid.
        """
        record = self.get(rfp_id)

        if record.extraction_status == ExtractionStatus.RUNNING:
            raise ExtractionRunningError(
                f"Extraction for RFP {rfp_id} is already running."
            )

        path = Path(record.stored_path)
        if not path.exists():
            raise InvalidFileError("Original file is missing, cannot extract.")

        if not force and self._cache_is_valid(record, path):
            return record

        return self.repository.update(
            rfp_id, extraction_status=ExtractionStatus.RUNNING, extraction_error=None
        )

    def run_extraction(self, rfp_id: str) -> None:
        """One Gemini call for the whole RFP (run in background)."""
        record = self.get(rfp_id)
        path = Path(record.stored_path)

        try:
            extractor = self._extractor()
            result = extractor.extract(path)
            key = self._extraction_key(record, extractor.fingerprint(path))

            self._write_extraction(rfp_id, result.extraction)
            self.repository.update(
                rfp_id,
                extraction_status=ExtractionStatus.DONE,
                extraction_error=None,
                extraction_key=key,
                extraction_needs_review=result.extraction.meta.needs_review,
                extraction_prompt_tokens=result.usage.prompt_tokens,
                extraction_output_tokens=result.usage.output_tokens
                + result.usage.thinking_tokens,
                extraction_seconds=round(result.seconds, 1),
                extracted_at=datetime.now(timezone.utc),
            )
            logger.info(
                "RFP %s extracted: %d in / %d out tokens, %.0fs",
                rfp_id,
                result.usage.prompt_tokens,
                result.usage.output_tokens,
                result.seconds,
            )

        except Exception as e:
            logger.exception("Extraction failed for RFP %s", rfp_id)
            self.repository.update(
                rfp_id,
                extraction_status=ExtractionStatus.FAILED,
                extraction_error=str(e),
            )

    def get_extraction(self, rfp_id: str) -> TenderExtraction:
        record = self.get(rfp_id)

        if record.extraction_status != ExtractionStatus.DONE:
            detail = f"Extraction for RFP {rfp_id} is {record.extraction_status.value}."
            if record.extraction_error:
                detail += f" {record.extraction_error}"
            raise ExtractionNotReadyError(detail)

        path = self._extraction_file(rfp_id)
        if not path.exists():
            raise ExtractionNotReadyError(
                "Extraction file is missing. Run extract again."
            )

        return TenderExtraction.model_validate_json(path.read_text(encoding="utf-8"))

    def _extractor(self):
        if self._extraction_service_factory is None:
            raise InvalidFileError("Extraction is not configured.")
        return self._extraction_service_factory()

    def _extraction_file(self, rfp_id: str) -> Path:
        return self.settings.data_dir / "rfp_extractions" / f"{rfp_id}.json"

    @staticmethod
    def _extraction_key(record: RFPRecord, fingerprint: str) -> str:
        return hashlib.sha256(
            f"{record.content_hash}|{fingerprint}".encode()
        ).hexdigest()[:16]

    def _cache_is_valid(self, record: RFPRecord, path: Path) -> bool:
        """Same file, same schema, model, prompt and mode as the stored result -> no new LLM call."""
        if record.extraction_status != ExtractionStatus.DONE:
            return False
        if not self._extraction_file(record.rfp_id).exists():
            return False

        key = self._extraction_key(record, self._extractor().fingerprint(path))
        return record.extraction_key == key

    def _write_extraction(self, rfp_id: str, extraction: TenderExtraction) -> None:
        target = self._extraction_file(rfp_id)
        target.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = target.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(
                extraction.model_dump(mode="json"), indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        tmp_path.replace(target)

    # ---------- startup recovery ----------

    def recover_interrupted(self) -> None:
        """Mark the tasks that were stuck in "processing"/"running" as failed when the server restarts."""
        for record in self.repository.list_all():
            if record.status == RFPStatus.PROCESSING:
                self.repository.update(
                    record.rfp_id,
                    status=RFPStatus.FAILED,
                    error="Server restarted during processing. Use re-ingest.",
                )

            if record.extraction_status == ExtractionStatus.RUNNING:
                self.repository.update(
                    record.rfp_id,
                    extraction_status=ExtractionStatus.FAILED,
                    extraction_error="Server restarted during extraction. Run extract again.",
                )

    # ---------- delete ----------

    def delete(self, rfp_id: str) -> None:
        record = self.get(rfp_id)

        self.qdrant_service.delete_by_rfp(rfp_id)
        Path(record.stored_path).unlink(missing_ok=True)
        self._extraction_file(rfp_id).unlink(missing_ok=True)
        self.repository.delete(rfp_id)

    def resolve_rfp_ids(self, rfp_ids: list[str] | None) -> list[str]:
        """If IDs are provided, validate them. Otherwise, get all READY RFPs."""

        if rfp_ids:
            self.ensure_ready(rfp_ids=rfp_ids)
            return rfp_ids

        ready = [
            record.rfp_id
            for record in self.repository.list_all()
            if record.status == RFPStatus.READY
        ]

        if not ready:
            raise RFPNotReadyError("No ready RFPs found. Upload an RFP first.")

        return ready
