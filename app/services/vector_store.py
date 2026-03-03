"""ChromaDB-backed vector store service."""

import chromadb


class VectorStore:
    """Persistent vector store using ChromaDB.

    Stores document embeddings on disk under ``./chroma_db`` and exposes
    simple add / search operations against a single collection.
    """

    _PERSIST_DIR: str = "./chroma_db"
    _COLLECTION_NAME: str = "documents"

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=self._PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Add documents with pre-computed embeddings to the store.

        Args:
            ids: Unique identifiers for each document.
            texts: Raw document texts (stored as metadata for retrieval).
            embeddings: Embedding vectors corresponding to each document.
        """
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        max_distance: float = 1.0,
    ) -> list[tuple[str, float]]:
        """Find the most similar documents to a query embedding.

        Args:
            query_embedding: The embedding vector to search against.
            top_k: Number of results to return.
            max_distance: Maximum cosine distance threshold. Results farther
                than this value are considered irrelevant and filtered out.

        Returns:
            A list of (document_text, distance) tuples ranked by similarity.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"],
        )
        documents: list[list[str]] | None = results.get("documents")
        distances: list[list[float]] | None = results.get("distances")
        if not documents or not distances:
            return []

        # Filter out results that exceed the distance threshold
        return [
            (doc, dist)
            for doc, dist in zip(documents[0], distances[0])
            if dist <= max_distance
        ]
