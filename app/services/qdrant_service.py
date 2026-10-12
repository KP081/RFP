from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.services.embeddings import EmbeddingService


class QdrantService:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
    ):
        settings = get_settings()

        self.client = QdrantClient(
            host=host or settings.qdrant_host,
            port=port or settings.qdrant_port,
        )
        self.collection_name = collection_name or settings.qdrant_collection
        self.batch_size = settings.qdrant_upsert_batch_size
        self._vector_size = vector_size

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            self._vector_size = EmbeddingService().dimension
        return self._vector_size

    # ---------- collection ----------

    def is_healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def create_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self._validate_vector_size()
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="rfp_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def _validate_vector_size(self) -> None:
        info = self.client.get_collection(self.collection_name)
        existing_size = getattr(info.config.params.vectors, "size", None)

        if existing_size is not None and existing_size != self.vector_size:
            raise ValueError(
                f"Collection '{self.collection_name}' has vector size {existing_size}, "
                f"but the embedding model produces {self.vector_size}. "
                f"Change QDRANT_COLLECTION in .env to use a new collection."
            )

    def delete_collection(self, collection_name: str | None = None) -> None:
        self.client.delete_collection(
            collection_name=collection_name or self.collection_name
        )

    # ---------- write ----------

    def upsert_chunks(
        self,
        chunks: list,
        vectors: list[list[float]],
        rfp_id: str,
        filename: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors.")

        points = [
            models.PointStruct(
                # Deterministic ID: if same RFP + chunk reingest when it's overwrite
                id=str(uuid5(NAMESPACE_URL, f"{rfp_id}:{chunk.chunk_id}")),
                vector=vector,
                payload={
                    "rfp_id": rfp_id,
                    "filename": filename,
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.metadata["page_number"],
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        for i in range(0, len(points), self.batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + self.batch_size],
                wait=True,
            )

    def delete_by_rfp(self, rfp_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=self._rfp_filter([rfp_id])),
            wait=True,
        )

    # ---------- read ----------

    @staticmethod
    def _rfp_filter(rfp_ids: list[str] | None) -> models.Filter | None:
        if not rfp_ids:
            return None

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="rfp_id",
                    match=models.MatchAny(any=list(rfp_ids)),
                )
            ]
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        rfp_ids: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._rfp_filter(rfp_ids),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            {
                "score": result.score,
                "rfp_id": result.payload["rfp_id"],
                "filename": result.payload["filename"],
                "chunk_id": result.payload["chunk_id"],
                "page_number": result.payload["page_number"],
                "text": result.payload["text"],
            }
            for result in results.points
        ]

    def count_points(self, rfp_id: str | None = None) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=self._rfp_filter([rfp_id]) if rfp_id else None,
            exact=True,
        )
        return result.count
