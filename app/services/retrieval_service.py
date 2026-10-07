from app.services.embeddings import EmbeddingService
from app.services.qdrant_service import QdrantService


class RetrievalService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()

    def search(
        self, query: str, rfp_id: str | None = None, top_k: int = 5,
        score_threshold: float | None = None
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query can not be empty.")

        query_vector = self.embedding_service.embed_text(query)

        return self.qdrant_service.search(
            query_vector=query_vector, limit=top_k, rfp_id=rfp_id,
            score_threshold=score_threshold
        )
