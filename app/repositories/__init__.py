from app.core.config import Settings
from app.repositories.base import RFPRepository
from app.repositories.json_repository import JsonRFPRepository


def build_repository(settings: Settings) -> RFPRepository:
    if settings.repository_backend == "json":
        return JsonRFPRepository(settings.data_dir / "rfps.json")

    # next when database is available:
    # if settings.repository_backend == "postgres":
    #     return SqlRFPRepository(settings.database_url)

    raise ValueError(f"Unknown REPOSITORY_BACKEND: {settings.repository_backend}")
