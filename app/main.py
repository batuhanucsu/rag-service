"""FastAPI RAG service — ingest documents and search semantically."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.services.chunker import TextChunker
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService
from app.services.semantic_cache import SemanticCache
from app.services.vector_store import VectorStore

# ------------------------------------------------------------------
# Service singletons (initialised once at startup)
# ------------------------------------------------------------------
ingestion_service: IngestionService
retrieval_service: RetrievalService
semantic_cache: SemanticCache


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialise heavyweight services once when the app starts."""
    global ingestion_service, retrieval_service, semantic_cache  # noqa: PLW0603

    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    chunker = TextChunker()

    ingestion_service = IngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        chunker=chunker,
    )
    llm_service = LLMService()
    semantic_cache = SemanticCache()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        semantic_cache=semantic_cache,
    )
    yield


app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation API — ingest documents and search semantically.",
    version="1.0.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------


class IngestRequest(BaseModel):
    document_id: str = Field(..., examples=["doc-002"], description="Unique document identifier.")
    content: str = Field(..., examples=["Python is a programming language..."], description="Raw document text.")


class IngestResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    document_id: str = Field(..., examples=["doc-002"])


class SearchRequest(BaseModel):
    query: str = Field(..., examples=["What is Python?"], description="Natural-language search query.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return.")


class SearchResponse(BaseModel):
    results: list[str] = Field(..., description="Retrieved document chunks ranked by relevance.")


class AskRequest(BaseModel):
    query: str = Field(..., examples=["Yapay zeka bankacılıkta ne işe yarar?"], description="Natural-language question.")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of context chunks to use.")


class AskResponse(BaseModel):
    answer: str = Field(..., description="LLM-generated natural-language answer.")
    sources: list[str] = Field(..., description="Source chunks used to generate the answer.")
    cached: bool = Field(False, description="True if the answer was served from semantic cache.")


class CacheStatsResponse(BaseModel):
    size: int = Field(..., description="Current number of cached entries.")
    max_size: int = Field(..., description="Maximum cache capacity.")
    similarity_threshold: float = Field(..., description="Cosine similarity threshold for cache hits.")
    ttl_seconds: float = Field(..., description="Time-to-live per entry in seconds.")


class CacheClearResponse(BaseModel):
    cleared: int = Field(..., description="Number of entries removed.")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a document",
    description="Split the document into chunks, embed them, and store in the vector database.",
)
async def ingest(request: IngestRequest) -> IngestResponse:
    ingestion_service.ingest_document(
        document_id=request.document_id,
        content=request.content,
    )
    return IngestResponse(status="ok", document_id=request.document_id)


@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search",
    description="Search for the most relevant document chunks using semantic similarity.",
)
async def search(request: SearchRequest) -> SearchResponse:
    results = retrieval_service.search(
        query=request.query,
        top_k=request.top_k,
    )
    return SearchResponse(results=results)


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question (RAG)",
    description="Retrieve relevant chunks and generate a natural-language answer using LLM.",
)
async def ask(request: AskRequest) -> AskResponse:
    answer, sources, cached = retrieval_service.ask(
        query=request.query,
        top_k=request.top_k,
    )
    return AskResponse(answer=answer, sources=sources, cached=cached)


@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache statistics",
    description="Return current semantic cache size and configuration.",
)
async def cache_stats() -> CacheStatsResponse:
    return CacheStatsResponse(**semantic_cache.stats)


@app.delete(
    "/cache",
    response_model=CacheClearResponse,
    summary="Clear cache",
    description="Remove all entries from the semantic cache.",
)
async def cache_clear() -> CacheClearResponse:
    cleared = semantic_cache.clear()
    return CacheClearResponse(cleared=cleared)
