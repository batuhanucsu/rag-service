"""In-memory semantic cache for RAG query results.

How it works
------------
Each cached entry stores the embedding of the original query together with
the generated answer and source chunks.  On a new request the cosine
similarity between the incoming query embedding and every cached embedding is
computed.  If the best match exceeds *similarity_threshold* the cached answer
is returned immediately, skipping the vector-store lookup and LLM call.

Entries expire automatically after *ttl_seconds* (default 1 hour).
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _CacheEntry:
    query: str
    embedding: list[float]
    answer: str
    sources: list[str]
    created_at: float = field(default_factory=time.time)


class SemanticCache:
    """Thread-safe in-memory semantic cache.

    Args:
        similarity_threshold: Minimum cosine similarity [0, 1] required for a
            cache hit.  Higher → stricter matching.  Default is 0.92.
        ttl_seconds: Time-to-live in seconds for each entry.  Expired entries
            are purged lazily on every access.  Default is 3600 (1 hour).
        max_size: Maximum number of entries to keep.  When the limit is
            reached the oldest entry is evicted (FIFO).  Default is 256.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        ttl_seconds: float = 3600,
        max_size: int = 256,
    ) -> None:
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._entries: list[_CacheEntry] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, query_embedding: list[float]
    ) -> tuple[str, list[str]] | None:
        """Look up a cached answer by query embedding.

        Args:
            query_embedding: Embedding vector of the incoming query.

        Returns:
            ``(answer, sources)`` if a similar-enough unexpired entry exists,
            otherwise ``None``.
        """
        with self._lock:
            self._evict_expired()
            if not self._entries:
                return None

            q_vec = self._normalise(np.array(query_embedding, dtype=np.float32))
            cache_matrix = np.array(
                [e.embedding for e in self._entries], dtype=np.float32
            )
            # Each row is already normalised; dot product == cosine similarity.
            similarities = cache_matrix @ q_vec
            best_idx = int(np.argmax(similarities))

            if similarities[best_idx] >= self._threshold:
                entry = self._entries[best_idx]
                return entry.answer, entry.sources

        return None

    def set(
        self,
        query: str,
        query_embedding: list[float],
        answer: str,
        sources: list[str],
    ) -> None:
        """Store a query/answer pair in the cache.

        Args:
            query: Original query text (for debugging / introspection).
            query_embedding: Embedding vector of the query.
            answer: LLM-generated answer to cache.
            sources: Source chunks used to generate the answer.
        """
        normalised = self._normalise(
            np.array(query_embedding, dtype=np.float32)
        ).tolist()

        with self._lock:
            # Evict oldest entry if at capacity
            if len(self._entries) >= self._max_size:
                self._entries.pop(0)

            self._entries.append(
                _CacheEntry(
                    query=query,
                    embedding=normalised,
                    answer=answer,
                    sources=sources,
                )
            )

    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns:
            Number of entries that were removed.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def stats(self) -> dict:
        """Return basic cache statistics."""
        with self._lock:
            return {
                "size": len(self._entries),
                "max_size": self._max_size,
                "similarity_threshold": self._threshold,
                "ttl_seconds": self._ttl,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove entries whose TTL has elapsed.  Must be called under lock."""
        now = time.time()
        self._entries = [e for e in self._entries if now - e.created_at < self._ttl]

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
