import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.core.exceptions import RFPNotFoundError
from app.repositories.base import RFPRepository
from app.schemas.rfp import RFPRecord


class JsonRFPRepository(RFPRepository):
    """Simple file based storage, exceptable for single process"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self._write({})

    def _read(self) -> dict[str, dict]:
        text = self.path.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else {}

    def _write(self, data: dict[str, dict]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp_path.replace(self.path)

    def create(self, record: RFPRecord) -> RFPRecord:
        with self._lock:
            data = self._read()
            data[record.rfp_id] = record.model_dump(mode="json")
            self._write(data)
        return record

    def get(self, rfp_id: str) -> RFPRecord | None:
        with self._lock:
            raw = self._read().get(rfp_id)
        return RFPRecord.model_validate(raw) if raw else None

    def list_all(self) -> list[RFPRecord]:
        with self._lock:
            data = self._read()
        records = [RFPRecord.model_validate(raw) for raw in data.values()]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def update(self, rfp_id: str, **fields) -> RFPRecord:
        with self._lock:
            data = self._read()

            if rfp_id not in data:
                raise RFPNotFoundError(f"RFP {rfp_id} not found.")

            record = RFPRecord.model_validate(data[rfp_id])
            updated = record.model_copy(
                update={**fields, "updated_at": datetime.now(timezone.utc)}
            )
            data[rfp_id] = updated.model_dump(mode="json")
            self._write(data)

        return updated

    def delete(self, rfp_id: str) -> None:
        with self._lock:
            data = self._read()
            data.pop(rfp_id, None)
            self._write(data)
