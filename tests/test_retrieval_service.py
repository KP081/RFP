from app.services.retrieval_service import RetrievalService


RFP_ID = "c41b1645-96b3-46ae-b0d2-4830f272d1fe"


def test_retrieve_rfp_chunks():
    service = RetrievalService()

    query = "What are the eligibility requirements for the consultant?"

    results = service.search(
        query=query,
        rfp_ids=[RFP_ID],
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
