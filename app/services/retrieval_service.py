"""Semantic search retrieval service with LLM-powered answer generation."""

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.semantic_cache import SemanticCache
from app.services.vector_store import VectorStore


class RetrievalService:
    """Retrieve relevant documents and generate natural-language answers.

    Args:
        embedding_service: Service that produces embedding vectors.
        vector_store: Persistent vector store to search against.
        llm_service: LLM service for answer generation.
        semantic_cache: Optional semantic cache for skipping repeated LLM calls.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
        semantic_cache: SemanticCache | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._llm_service = llm_service
        self._cache = semantic_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Chunks with cosine distance above this are considered irrelevant.
    _RELEVANCE_THRESHOLD: float = 0.75

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Find the most relevant document chunks for a query.

        Args:
            query: The search query in plain text.
            top_k: Number of results to return.

        Returns:
            A list of document texts ranked by semantic similarity.
        """
        query_embedding = self._embedding_service.embed_text(query)
        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        return [doc for doc, _dist in results]

    def ask(self, query: str, top_k: int = 5) -> tuple[str, list[str], bool]:
        """Retrieve relevant chunks and generate a natural-language answer.

        Checks the semantic cache first; on a miss runs the full RAG pipeline
        and stores the result in the cache for future requests.

        Args:
            query: The user's natural-language question.
            top_k: Number of context chunks to retrieve.

        Returns:
            A tuple of (generated_answer, source_chunks, cache_hit).
        """
        query_embedding = self._embedding_service.embed_text(query)

        # --- Cache lookup ---
        if self._cache is not None:
            cached = self._cache.get(query_embedding)
            if cached is not None:
                answer, sources = cached
                return answer, sources, True

        # --- Full RAG pipeline ---
        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # Only keep chunks that are truly relevant
        relevant = [(doc, dist) for doc, dist in results if dist <= self._RELEVANCE_THRESHOLD]

        if not relevant:
            return "Bu soruyla ilgili yüklenen dokümanlarda bilgi bulunamadı.", [], False

        chunks = [doc for doc, _dist in relevant]
        answer = self._llm_service.generate_answer(query=query, context_chunks=chunks)

        # --- Store in cache ---
        if self._cache is not None:
            self._cache.set(
                query=query,
                query_embedding=query_embedding,
                answer=answer,
                sources=chunks,
            )

        return answer, chunks, False
