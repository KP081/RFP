from app.services.embeddings import EmbeddingService


def test_embedding():
    service = EmbeddingService()

    vector = service.embed_text("The consultant shall submit the technical proposal.")

    assert len(vector) == 384

    print(f"\nvector dimensions: {len(vector)}")
    print(f"first 10 values: {vector[:10]}")
