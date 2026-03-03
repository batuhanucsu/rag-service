"""RAG document ingestion pipeline."""

from app.services.chunker import TextChunker
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class IngestionService:
    """Orchestrates the text → chunk → embed → store pipeline.

    Args:
        embedding_service: Service that produces embedding vectors.
        vector_store: Persistent vector store for document retrieval.
        chunker: Text chunker for splitting long documents.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        chunker: TextChunker,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._chunker = chunker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_document(self, document_id: str, content: str) -> None:
        """Ingest a single document into the vector store.

        The document is split into chunks, each chunk is embedded, and
        the resulting vectors are persisted with deterministic IDs
        derived from *document_id*.

        Args:
            document_id: Unique identifier for the source document.
            content: Raw document text.
        """
        chunks = self._chunker.chunk(content)
        if not chunks:
            return

        embeddings = self._embedding_service.embed_texts(chunks)

        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]

        self._vector_store.add_documents(
            ids=ids,
            texts=chunks,
            embeddings=embeddings,
        )
