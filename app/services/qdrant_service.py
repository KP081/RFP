from uuid import uuid4

from qdrant_client import QdrantClient, models


class QdrantService:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "rfp_documents",
        vector_size: int = 384,
    ):
        self.client = QdrantClient(
            host=host,
            port=port,
        )
        self.collection_name = collection_name
        self.vector_size = vector_size

    def create_collection(self) -> None:
        collections = self.client.get_collections().collections

        collection_exists = any(
            collection.name == self.collection_name for collection in collections
        )

        if collection_exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: list,
        vectors: list[list[float]],
        rfp_id: str,
        filename: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors.")

        points = []

        for chunk, vector in zip(chunks, vectors):
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "rfp_id": rfp_id,
                        "filename": filename,
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.metadata["page_number"],
                        "text": chunk.text,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        rfp_id: str | None = None,
    ) -> list[dict]:
        query_filter = None

        if rfp_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="rfp_id",
                        match=models.MatchValue(value=rfp_id),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
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

    def count_points(self) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count
