from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str


class IngestDirectoryRequest(BaseModel):
    path: str


class IngestDirectoryResponse(BaseModel):
    owner_id: str
    root: str
    files_ingested: int
    chunks_written: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=30)


class SearchHit(BaseModel):
    id: str | None = None
    source: str
    text: str = ""
    score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[dict[str, Any]]


class ChatRequest(BaseModel):
    user_text: str
    top_k: int = Field(default=4, ge=1, le=20)


class ChatResponse(BaseModel):
    owner_id: str
    state: str
    answer: str
    references: list[dict[str, str]]


class KnowledgeChunk(BaseModel):
    chunk_id: str
    source_path: str
    text: str


class StatsResponse(BaseModel):
    owner_id: str
    chunks: int
    sources: int


# Legacy names used in older docs / tests
class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Directory or file path to ingest.")
    recursive: bool = Field(default=True)


class IngestResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int
    storage_path: str


class LegacySearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class LegacySearchHit(BaseModel):
    source: str
    score: float
    snippet: str


class LegacySearchResponse(BaseModel):
    query: str
    total_hits: int
    hits: list[LegacySearchHit]


class LegacyChatRequest(BaseModel):
    message: str
    top_k: int = Field(default=4, ge=1, le=20)
    mode: Literal["chat", "search_only"] = "chat"


class LegacyChatResponse(BaseModel):
    answer: str
    references: list[LegacySearchHit]
