from app.services.llm_service import LLMService


def test_gemini_generation():
    service = LLMService()

    response = service.generate("Explain RAG in 2 simple sentences.")

    print("\nGemini response:")
    print(response)

    assert response
