"""Small embedding interface and the default CPU implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingBackend(Protocol):
    """Turn text into vectors without exposing a specific ML library."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbeddings:
    """Lazy CPU embeddings using Sentence Transformers.

    The model is not loaded when the API starts. It is loaded on the first
    registration or search, which keeps simple health checks quick. CPU is
    intentional: vLLM owns the limited GPU memory in this project.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._model = None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one normalized 384-number vector for each input string.

        Example: ``['add an item to a cart']`` becomes ``[[0.02, -0.11, ...]]``.
        Similar sentences point in similar directions, which lets PostgreSQL
        compare meanings instead of requiring exact words.
        """
        if not texts:
            return []
        if self._model is None:
            # Import lazily too, so importing RepoWren does not initialize any
            # machine-learning libraries before embeddings are actually needed.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device="cpu")
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()
