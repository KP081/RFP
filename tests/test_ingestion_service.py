from pathlib import Path

from app.services.ingestion_service import IngestionService


def test_ingest_rfp():
    file_path = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")

    service = IngestionService()

    result = service.ingest(
        file_path=file_path,
        rfp_id="cef53ab4-84a8-407d-aef7-176f7926b306",
        filename="RFP_DPR.pdf",
    )

    print("\nIngestion result:")
    print(result)

    assert result["pages"] == 379
    assert result["chunks"] > 0
    assert result["chunks"] == result["vectors"]
    assert result["vector_dimension"] == 384
