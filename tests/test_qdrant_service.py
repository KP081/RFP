from app.services.qdrant_service import QdrantService


def test_qdrant_connection():
    service = QdrantService()

    service.create_collection()

    count = service.count_points()

    print(f"\nPoint currently strored: {count}")

    assert count >= 0
