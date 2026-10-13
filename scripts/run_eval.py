import argparse
import json

from app.api.deps import get_rfp_repository
from app.schemas.rfp import RFPStatus
from app.services.retrieval_service import RetrievalService
from tests.test_retrieval_evaluation import EVALUATION_DATA

EVAL_FILENAME = "RFP_DPR.pdf"
DEPTH = 10
KS = (1, 3, 5)


def resolve_rfp_id(filename: str) -> str:
    matches = [
        r
        for r in get_rfp_repository().list_all()
        if r.filename == filename and r.status == RFPStatus.READY
    ]

    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly 1 READY RFP named '{filename}', "
            f"found {len(matches)}: {[r.rfp_id for r in matches]}"
        )

    return matches[0].rfp_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfp-id")
    args = parser.parse_args()

    rfp_id = args.rfp_id or resolve_rfp_id(EVAL_FILENAME)
    service = RetrievalService()
    ranks: list[int | None] = []

    for item in EVALUATION_DATA:
        results = service.search(item["question"], rfp_ids=[rfp_id], top_k=DEPTH)
        pages = [r["page_number"] for r in results]
        rank = next(
            (i for i, p in enumerate(pages, start=1) if p in item["expected_pages"]),
            None,
        )
        ranks.append(rank)

        print(
            f"rank={rank or '-':>2} expected={item['expected_pages']} "
            f"got={pages[:5]} | {item['question'][:55]}"
        )

    total = len(ranks)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / total for k in KS}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / total

    print("RESULT " + json.dumps(metrics))


if __name__ == "__main__":
    main()
