from pathlib import Path

from app.core.config import get_settings
from app.services.chunker import chunk_pages
from app.services.document_parser import extract_pages
from app.services.embeddings import EmbeddingService
from app.services.qdrant_service import QdrantService


class IngestionService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_service: QdrantService | None = None,
    ):
        self.settings = get_settings()
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService()

    def ingest(self, file_path: Path, rfp_id: str, filename: str) -> dict:
        # 1. Pages
        pages = extract_pages(file_path)

        if not pages:
            raise ValueError("No text could be extracted from the document.")

        # 2. Chunks (size/overlap config se)
        chunks = chunk_pages(
            pages,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        if not chunks:
            raise ValueError(
                "No extractable text found. The PDF may be scanned (image-only) "
                "and would need OCR."
            )

        # 3. Embeddings
        vectors = self.embedding_service.embed_documents([c.text for c in chunks])

        # 4. Collection ready karo
        self.qdrant_service.create_collection()

        # 5. Purane vectors hatao (re-ingest safe). Embedding success hone ke
        #    BAAD delete karte hain, taaki fail hone pe purana data na jaye.
        self.qdrant_service.delete_by_rfp(rfp_id)

        # 6. Store
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
            "qdrant_points": self.qdrant_service.count_points(rfp_id),
        }
