from app.services.retrieval_service import RetrievalService


RFP_ID = "cef53ab4-84a8-407d-aef7-176f7926b306"


def test_retrieve_rfp_chunks():
    service = RetrievalService()

    query = "What are the eligibility requirements for the consultant?"

    results = service.search(
        query=query,
        rfp_id=RFP_ID,
        top_k=5,
    )

    print("\nSearch results:")

    for index, result in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Score: {result['score']:.4f}")
        print(f"Page: {result['page_number']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Text: {result['text'][:500]}")

    assert len(results) > 0

    for result in results:
        assert result["rfp_id"] == RFP_ID
        assert result["text"]
        assert result["page_number"] > 0
