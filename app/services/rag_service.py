from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService


class RAGService:
    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.llm_service = LLMService()

    def answer(
        self,
        question: str,
        rfp_id: str | None = None,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve relevant chunks
        results = self.retrieval_service.search(
            query=question, rfp_id=rfp_id, top_k=top_k, score_threshold=score_threshold
        )

        if not results:
            return {
                "answer": "I could not find relevant information in the document.",
                "sources": [],
            }

        # 2. Build context from retrieved chunks
        context_parts = []

        for result in results:
            context_parts.append(
                f"""
Source: {result["filename"]}
Page: {result["page_number"]}

{result["text"]}
""".strip()
            )

        context = "\n\n---\n\n".join(context_parts)

        # 3. Build grounded prompt
        prompt = f"""
You are an assistant answering questions about an RFP document.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the context does not contain enough information, say:
  "I could not find enough information in the provided document."
- Keep the answer clear and concise.
- Mention the relevant page number(s) in the answer.

Context:
{context}

User question:
{question}

Answer:
""".strip()

        # 4. Generate answer
        answer = self.llm_service.generate(prompt)

        # 5. Return answer + sources
        sources = [
            {
                "filename": result["filename"],
                "page_number": result["page_number"],
                "chunk_id": result["chunk_id"],
                "score": result["score"],
            }
            for result in results
        ]

        return {
            "answer": answer,
            "sources": sources,
        }
