import logging
from pathlib import Path
from uuid import uuid4

import hashlib

from app.core.config import get_settings
from app.core.exceptions import (
    InvalidFileError,
    RFPNotFoundError,
    RFPNotReadyError,
    DuplicateRFPError
)
from app.repositories.base import RFPRepository
from app.schemas.rfp import RFPRecord, RFPStatus
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
    ):
        self.settings = get_settings()
        self.repository = repository
        self.ingestion_service = ingestion_service
        self.qdrant_service = qdrant_service

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

    def recover_interrupted(self) -> None:
        """Mark the tasks that were stuck in "processing" as failed when the server restarts."""
        for record in self.repository.list_all():
            if record.status == RFPStatus.PROCESSING:
                self.repository.update(
                    record.rfp_id,
                    status=RFPStatus.FAILED,
                    error="Server restarted during processing. Use re-ingest.",
                )

    # ---------- delete ----------

    def delete(self, rfp_id: str) -> None:
        record = self.get(rfp_id)

        self.qdrant_service.delete_by_rfp(rfp_id)
        Path(record.stored_path).unlink(missing_ok=True)
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
