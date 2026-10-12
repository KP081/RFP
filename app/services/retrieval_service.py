from app.core.config import get_settings
from app.services.embeddings import EmbeddingService
from app.services.qdrant_service import QdrantService


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_service: QdrantService | None = None,
    ):
        self.settings = get_settings()
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService()

    def search(
        self,
        query: str,
        rfp_ids: list[str] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query can not be empty.")

        if top_k is None:
            top_k = self.settings.top_k

        if score_threshold is None:
            score_threshold = self.settings.score_threshold

        query_vector = self.embedding_service.embed_query(query)

        return self.qdrant_service.search(
            query_vector=query_vector,
            limit=top_k,
            rfp_ids=rfp_ids,
            score_threshold=score_threshold,
        )
