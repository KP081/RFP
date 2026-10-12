from app.core.config import get_settings
from app.core.prompts import load_prompt
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService


class RAGService:
    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        llm_service: LLMService | None = None,
    ):
        self.settings = get_settings()
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_service = llm_service or LLMService()

    def answer(
        self,
        question: str,
        rfp_ids: list[str] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve
        results = self.retrieval_service.search(
            query=question,
            rfp_ids=rfp_ids,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        if not results:
            return {"answer": self.settings.no_answer_message, "sources": []}

        # 2. Context
        context = "\n\n---\n\n".join(
            f"Source: {r['filename']}\nPage: {r['page_number']}\n\n{r['text']}".strip()
            for r in results
        )

        # 3. Prompt
        template = load_prompt(self.settings.rag_prompt_file)
        prompt = template.format(context=context, question=question)

        # 4. Generate
        answer = self.llm_service.generate(prompt)

        # 5. Answer + sources
        return {
            "answer": answer,
            "sources": [
                {
                    "rfp_id": r["rfp_id"],
                    "filename": r["filename"],
                    "page_number": r["page_number"],
                    "chunk_id": r["chunk_id"],
                    "score": r["score"],
                    "text": r["text"],
                }
                for r in results
            ],
        }
