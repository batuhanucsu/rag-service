"""Cross-encoder reranker service for improving retrieval quality.

How it works
------------
After an initial vector-store retrieval (which uses approximate cosine
similarity), the reranker scores every (query, chunk) pair with a
cross-encoder model.  Cross-encoders jointly encode the query and the
document, producing a much more accurate relevance score than the
bi-encoder approach used for initial retrieval.

Recommended model: ``BAAI/bge-reranker-base``
- Multilingual (covers Turkish)
- ~280 MB on disk
- Good balance of speed and accuracy

The model is downloaded automatically on first use and cached by
Hugging Face Hub (``~/.cache/huggingface``).
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankService:
    """Rerank a list of (document, score) pairs using a cross-encoder model.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``BAAI/bge-reranker-base`` which supports multilingual text
            (including Turkish) and runs well on CPU.
        top_n: Number of top-ranked documents to return after reranking.
            ``None`` returns all documents in reranked order.
        batch_size: Number of pairs to score in a single forward pass.
        max_length: Maximum token length for the cross-encoder.  Pairs that
            exceed this are truncated.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_n: int | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self._top_n = top_n
        self._batch_size = batch_size
        self._max_length = max_length
        self._model_name = model_name

        logger.info("Loading reranker model: %s", model_name)
        self._model = CrossEncoder(
            model_name,
            max_length=max_length,
        )
        logger.info("Reranker model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[str, float]]:
        """Score and rerank *documents* against *query*.

        Args:
            query: The user's natural-language question or search query.
            documents: Candidate document chunks to rerank.
            top_n: Override the instance-level ``top_n`` for this call.
                Pass ``None`` to return all documents in reranked order.

        Returns:
            List of ``(document_text, score)`` tuples sorted by descending
            relevance score.  Higher is more relevant.
        """
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores: list[float] = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        ).tolist()

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        limit = top_n if top_n is not None else self._top_n
        return ranked[:limit] if limit is not None else ranked

    @property
    def model_name(self) -> str:
        """Return the loaded model identifier."""
        return self._model_name
