from functools import lru_cache

from app.core.config import get_settings
from app.repositories import build_repository
from app.repositories.base import RFPRepository
from app.services.embeddings import EmbeddingService
from app.services.extraction_service import ExtractionService
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.qdrant_service import QdrantService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.rfp_service import RFPService


@lru_cache
def get_rfp_repository() -> RFPRepository:
    return build_repository(get_settings())


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_qdrant_service() -> QdrantService:
    return QdrantService()


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(
        embedding_service=get_embedding_service(),
        qdrant_service=get_qdrant_service(),
    )


@lru_cache
def get_extraction_service() -> ExtractionService:
    return ExtractionService(llm_service=LLMService())


@lru_cache
def get_rfp_service() -> RFPService:
    ingestion = IngestionService(
        embedding_service=get_embedding_service(),
        qdrant_service=get_qdrant_service(),
    )
    return RFPService(
        repository=get_rfp_repository(),
        ingestion_service=ingestion,
        qdrant_service=get_qdrant_service(),
        extraction_service_factory=get_extraction_service,  # built on first use
    )


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService(
        retrieval_service=get_retrieval_service(),
        llm_service=LLMService(),
    )
