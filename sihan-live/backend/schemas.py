from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Directory or file path to ingest.")
    recursive: bool = Field(default=True)
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".txt", ".md", ".json", ".csv", ".log"]
    )


class IngestResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int
    storage_path: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    source: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total_hits: int
    hits: list[SearchHit]


class ChatRequest(BaseModel):
    message: str
    top_k: int = Field(default=4, ge=1, le=20)
    mode: Literal["chat", "search_only"] = "chat"


class ChatResponse(BaseModel):
    answer: str
    references: list[SearchHit]
