from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@lru_cache(maxsize=4)
def _load_model(model_name: str, max_seq_length: int | None) -> SentenceTransformer:
    """Model process only one time (cache)."""
    model = SentenceTransformer(model_name)

    if max_seq_length:
        model.max_seq_length = max_seq_length

    return model


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        settings = get_settings()

        self.model_name = model_name or settings.embedding_model
        self.query_prefix = settings.embedding_query_prefix
        self.document_prefix = settings.embedding_document_prefix
        self.batch_size = settings.embedding_batch_size
        self.model = _load_model(self.model_name, settings.embedding_max_seq_length)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_text("dimension probe"))
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(self.query_prefix + text)

    def embed_documents(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        vectors = self.model.encode(
            [self.document_prefix + text for text in texts],
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
        )
        return vectors.tolist()
