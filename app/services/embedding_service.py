"""Local embedding service using sentence-transformers."""

import threading
from typing import ClassVar

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Thread-safe singleton embedding service.

    Loads the sentence-transformers model once at first instantiation
    and reuses it across all subsequent calls.
    """

    _MODEL_NAME: ClassVar[str] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    _instance: ClassVar["EmbeddingService | None"] = None
    _init_lock: ClassVar[threading.Lock] = threading.Lock()

    _model: SentenceTransformer
    _encode_lock: threading.Lock

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._model = SentenceTransformer(cls._MODEL_NAME)
                    instance._encode_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text.

        Args:
            text: The input string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts.

        Args:
            texts: A list of input strings to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """
        with self._encode_lock:
            embeddings = self._model.encode(texts)
        return embeddings.tolist()
