from app.services.retrieval_service import RetrievalService
from tests.test_retrieval_evaluation import EVALUATION_DATA, RFP_ID

service = RetrievalService()

for item in EVALUATION_DATA:
    results = service.search(item["question"], rfp_id=RFP_ID, top_k=10)
    pages = [r["page_number"] for r in results]

    rank = next(
        (i for i, p in enumerate(pages, start=1) if p in item["expected_pages"]),
        None,
    )

    print("=" * 80)
    print("Q:", item["question"])
    print("expected:", item["expected_pages"])
    print("retrieved:", pages)
    print("first correct rank:", rank)

    if rank is None or rank > 1:
        top = results[0]
        print(f"top-1 page {top['page_number']} score {top['score']:.3f}")
        print(top["text"][:300].replace("\n", " "))