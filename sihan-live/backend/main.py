from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .auth import UserContext, require_user_context
from .config import Settings, load_settings
from .llm_engine import EmotionalEngine
from .memory import KnowledgeBase
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestDirectoryRequest,
    IngestDirectoryResponse,
    KnowledgeChunk,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)


def _build_services() -> tuple[Settings, KnowledgeBase, EmotionalEngine]:
    settings = load_settings()
    kb = KnowledgeBase(
        storage_path=settings.knowledge_base.storage_path,
        chunk_size=settings.knowledge_base.chunk_size,
        chunk_overlap=settings.knowledge_base.chunk_overlap,
        allowed_roots=settings.knowledge_base.allowed_roots,
        default_owner=settings.app.owner_id,
    )
    engine = EmotionalEngine(persona_name="赵思涵")
    return settings, kb, engine


settings, kb, emotion_engine = _build_services()
app = FastAPI(title=settings.app.name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app.name)


@app.get("/kb/stats", response_model=StatsResponse)
def kb_stats(ctx: UserContext = Depends(require_user_context)) -> StatsResponse:
    return kb.stats(owner_id=ctx.owner_id)


@app.post("/kb/ingest-directory", response_model=IngestDirectoryResponse)
def ingest_directory(
    payload: IngestDirectoryRequest,
    ctx: UserContext = Depends(require_user_context),
) -> IngestDirectoryResponse:
    base_path = Path(payload.path).expanduser().resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise HTTPException(status_code=400, detail="The target path must be an existing directory.")
    file_count, chunk_count = kb.ingest_directory(base_path, owner_id=ctx.owner_id)
    return IngestDirectoryResponse(
        owner_id=ctx.owner_id,
        root=str(base_path),
        files_ingested=file_count,
        chunks_written=chunk_count,
    )


@app.post("/kb/search", response_model=SearchResponse)
def kb_search(
    payload: SearchRequest,
    ctx: UserContext = Depends(require_user_context),
) -> SearchResponse:
    results = kb.search(payload.query, owner_id=ctx.owner_id, top_k=payload.top_k)
    return SearchResponse(query=payload.query, top_k=payload.top_k, results=results)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, ctx: UserContext = Depends(require_user_context)) -> ChatResponse:
    hits = kb.search(payload.user_text, owner_id=ctx.owner_id, top_k=payload.top_k)
    answer = emotion_engine.generate_reply(payload.user_text, memory_hits=hits)
    return ChatResponse(
        owner_id=ctx.owner_id,
        state=answer.state,
        answer=answer.answer,
        references=hits,
    )


@app.get("/kb/debug/chunks", response_model=list[KnowledgeChunk])
def debug_chunks(
    limit: int = Query(20, ge=1, le=100),
    ctx: UserContext = Depends(require_user_context),
) -> list[KnowledgeChunk]:
    return kb.list_chunks(owner_id=ctx.owner_id, limit=limit)
