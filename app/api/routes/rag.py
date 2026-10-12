from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service, get_retrieval_service, get_rfp_service
from app.schemas.rag import AskRequest, AskResponse, SearchRequest, SearchResponse
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.rfp_service import RFPService

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    rfp_service: RFPService = Depends(get_rfp_service),
    rag_service: RAGService = Depends(get_rag_service),
):
    rfp_ids = rfp_service.resolve_rfp_ids(body.rfp_ids)

    return rag_service.answer(
        question=body.question,
        rfp_ids=rfp_ids,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
    )


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    rfp_service: RFPService = Depends(get_rfp_service),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    rfp_ids = rfp_service.resolve_rfp_ids(body.rfp_ids)

    results = retrieval_service.search(
        query=body.query,
        rfp_ids=rfp_ids,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
    )
    return {"results": results}