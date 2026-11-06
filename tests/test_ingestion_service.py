from pathlib import Path

from app.services.ingestion_service import IngestionService


def test_ingest_rfp():
    file_path = Path("uploads/c41b1645-96b3-46ae-b0d2-4830f272d1fe.pdf")

    service = IngestionService()

    result = service.ingest(
        file_path=file_path,
        rfp_id="c41b1645-96b3-46ae-b0d2-4830f272d1fe",
        filename="RFP_DPR 6.pdf",
    )

    print("\nIngestion result:")
    print(result)

    assert result["pages"] == 382
    assert result["chunks"] > 0
    assert result["chunks"] == result["vectors"]
    assert result["vector_dimension"] == 384
