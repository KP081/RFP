from abc import ABC, abstractmethod

from app.schemas.rfp import RFPRecord


class RFPRepository(ABC):
    @abstractmethod
    def create(self, record: RFPRecord) -> RFPRecord: ...

    @abstractmethod
    def get(self, rfp_id: str) -> RFPRecord | None: ...

    @abstractmethod
    def list_all(self) -> list[RFPRecord]: ...

    @abstractmethod
    def update(self, rfp_id: str, **fields) -> RFPRecord: ...

    @abstractmethod
    def delete(self, rfp_id: str) -> None: ...
