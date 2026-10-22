from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RFPStatus(StrEnum):
    """Status of INGESTION (parse + embed for Q&A)."""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ExtractionStatus(StrEnum):
    """Status of the structured EXTRACTION (independent of ingestion)."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class RFPRecord(BaseModel):
    """Internal record ( just like DB row )"""

    rfp_id: str
    filename: str
    stored_path: str
    size_bytes: int
    status: RFPStatus
    pages: int | None = None
    chunks: int | None = None
    error: str | None = None
    content_hash: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    # ---- structured extraction (old records without these fields still load) ----
    extraction_status: ExtractionStatus = ExtractionStatus.NOT_STARTED
    extraction_error: str | None = None
    extraction_key: str | None = None  # file hash + schema + model + prompt + mode
    extraction_needs_review: bool | None = None
    extraction_prompt_tokens: int | None = None
    extraction_output_tokens: int | None = None
    extraction_seconds: float | None = None
    extracted_at: datetime | None = None


class RFPResponse(BaseModel):
    """For Frontend"""

    rfp_id: str
    filename: str
    size_bytes: int
    status: RFPStatus
    pages: int | None = None
    chunks: int | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    extraction_status: ExtractionStatus = ExtractionStatus.NOT_STARTED
    extraction_error: str | None = None
    extraction_needs_review: bool | None = None
    extracted_at: datetime | None = None
