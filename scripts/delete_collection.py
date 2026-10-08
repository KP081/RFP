from app.services.qdrant_service import QdrantService

qdrant_service = QdrantService()

qdrant_service.delete_collection("rfp_documents")
