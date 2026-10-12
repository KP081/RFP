from pathlib import Path

from app.services.ingestion_service import IngestionService


def test_ingest_rfp():
    file_path = Path("uploads/a9378dcd-8470-4843-a963-de558c32f7bc.pdf")

    service = IngestionService()

    result = service.ingest(
        file_path=file_path,
        rfp_id="a9378dcd-8470-4843-a963-de558c32f7bc",
        filename="RFP_DPR.pdf",
    )

    print("\nIngestion result:")
    print(result)

    assert result["pages"] == 382
    assert result["chunks"] > 0
    assert result["chunks"] == result["vectors"]
    assert result["vector_dimension"] == 384
