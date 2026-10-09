from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RFPStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
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
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


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
