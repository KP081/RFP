from app.services.rag_service import RAGService


RFP_ID = "cef53ab4-84a8-407d-aef7-176f7926b306"


def test_rag_answer():
    service = RAGService()

    question = "What are the eligibility requirements for the consultant?"

    result = service.answer(
        question=question,
        rfp_id=RFP_ID,
        top_k=5,
        score_threshold=0.50
    )

    print("\n\n===== FINAL ANSWER =====")
    print(result["answer"])

    print("\n===== SOURCES =====")

    for source in result["sources"]:
        print(
            f"Page: {source['page_number']} | "
            f"Chunk: {source['chunk_id']} | "
            f"Score: {source['score']:.4f}"
        )

    assert result["answer"]
    assert result["sources"]

def test_rag_rejects_unrelated_question():
    service = RAGService()

    question = "What are LLMs and how do they work internally?"

    result = service.answer(
        question=question,
        rfp_id=RFP_ID,
        top_k=5,
        score_threshold=0.50,
    )

    print("\n\n===== UNANSWERABLE QUESTION =====")
    print(result["answer"])

    print("\n===== SOURCES =====")
    print(result["sources"])

    assert result["answer"] == (
        "I could not find relevant information in the document."
    )
    assert result["sources"] == []