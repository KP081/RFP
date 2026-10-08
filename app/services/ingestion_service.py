from pathlib import Path

from app.services.chunker import chunk_pages
from app.services.document_parser import extract_pages
from app.services.embeddings import EmbeddingService
from app.services.qdrant_service import QdrantService

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


class IngestionService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()

    def ingest(
        self,
        file_path: Path,
        rfp_id: str,
        filename: str,
    ) -> dict:
        # 1. Extract document pages
        pages = extract_pages(file_path)

        if not pages:
            raise ValueError("No text could be extracted from the document.")

        # 2. Convert pages into chunks
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        if not chunks:
            raise ValueError("No chunks were created from the document.")

        # 3. Generate embeddings for every chunk
        texts = [chunk.text for chunk in chunks]

        vectors = self.embedding_service.embed_documents(texts)

        # 4. Create Qdrant collection if it doesn't exist
        self.qdrant_service.create_collection()

        # 5. Store vectors + metadata in Qdrant
        self.qdrant_service.upsert_chunks(
            chunks=chunks,
            vectors=vectors,
            rfp_id=rfp_id,
            filename=filename,
        )

        return {
            "rfp_id": rfp_id,
            "filename": filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "vectors": len(vectors),
            "vector_dimension": len(vectors[0]),
            "qdrant_points": self.qdrant_service.count_points(),
        }
