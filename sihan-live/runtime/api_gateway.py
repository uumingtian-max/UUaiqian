"""Sihan API gateway: OpenAI-style chat + companion context (KB + web + memory), proxies to llama.cpp and TTS."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from companion_brain import (
    MiniKnowledgeBase,
    build_context_block,
    remember_turn,
)

LLAMA_BASE = os.environ.get("SIHAN_LLAMA_BASE", "http://127.0.0.1:8002")
TTS_BASE = os.environ.get("SIHAN_TTS_BASE", "http://127.0.0.1:9880")

_COMPANION_DIR = Path(os.environ.get("SIHAN_COMPANION_DIR", Path(__file__).resolve().parent / "companion_data"))
_OWNER_ID = os.environ.get("SIHAN_OWNER_ID", "nac")
_API_KEY = os.environ.get("SIHAN_API_KEY", "local-dev-key")
_KB_PATH = Path(os.environ.get("SIHAN_KB_PATH", _COMPANION_DIR / "knowledge_store.json"))
_KB_IMPORT = Path(os.environ.get("SIHAN_KB_IMPORT", _COMPANION_DIR / "kb_import"))

PERSONA_SYSTEM = """你叫赵思涵，小名小涵。始终用中文。和阿权像微信/电话那样自然聊天。
说话像真人：句子长短不一，可以有嗯…、好啦、别闹；少用分点列表、套话。
动作和表情只用很短的括号（笑）（愣一下）写在屏幕上看，不要说「括号里」这种话。
若系统消息里带了【知识库】【联网摘要】【最近说过】，要自然用进去，不要整段照念，不要暴露「搜索结果」字眼。"""

_kb: MiniKnowledgeBase | None = None


def get_kb() -> MiniKnowledgeBase:
    global _kb
    if _kb is None:
        _KB_IMPORT.mkdir(parents=True, exist_ok=True)
        _kb = MiniKnowledgeBase(_KB_PATH, allowed_roots=[_KB_IMPORT, _COMPANION_DIR], chunk_size=420, overlap=86)
    return _kb


def extract_last_user_text(messages: list) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text") or "")
                return "".join(parts).strip()
    return ""


def _auth_ok(request: Request) -> bool:
    if request.headers.get("x-api-key") != _API_KEY:
        return False
    if request.headers.get("x-owner-id") != _OWNER_ID:
        return False
    return True


app = FastAPI(title="sihan-api-gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(f"{LLAMA_BASE}/health")
            llama = r.json() if r.status_code == 200 else {"error": r.status_code}
        except Exception as e:  # pragma: no cover
            llama = {"error": str(e)}
    try:
        kb = get_kb()
        nchunks = len(kb._by_owner.get(_OWNER_ID, []))  # noqa: SLF001
    except Exception:
        nchunks = -1
    return {
        "status": "ok",
        "llama": llama,
        "tts": f"{TTS_BASE}/health",
        "companion": {"kb_chunks": nchunks, "owner": _OWNER_ID},
    }


@app.post("/kb/ingest-directory")
async def kb_ingest(request: Request):
    if not _auth_ok(request):
        return Response(content=json.dumps({"detail": "unauthorized"}), status_code=401, media_type="application/json")
    body = await request.json()
    path = (body or {}).get("path") or str(_KB_IMPORT)
    kb = get_kb()
    files, chunks = kb.ingest_path(_OWNER_ID, path)
    return {"owner_id": _OWNER_ID, "files_ingested": files, "chunks_written": chunks}


@app.api_route("/v1/audio/speech", methods=["POST"])
async def audio_speech(req: Request):
    body = await req.body()
    headers = {"content-type": req.headers.get("content-type", "application/json")}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{TTS_BASE}/v1/audio/speech", content=body, headers=headers)
    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "audio/mpeg"))


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(req: Request):
    body = await req.body()
    payload = json.loads(body.decode("utf-8")) if body else {}
    messages = list(payload.get("messages") or [])

    user_line = extract_last_user_text(messages)
    kb = get_kb()
    ctx = build_context_block(
        _COMPANION_DIR,
        kb,
        _OWNER_ID,
        user_line,
        web_enabled=True,
    )
    if ctx:
        comp_msg = {"role": "system", "content": "【伴读上下文】\n" + ctx}
        if messages and messages[0].get("role") == "system":
            orig = messages[0].get("content") or ""
            messages[0] = {"role": "system", "content": PERSONA_SYSTEM + "\n\n" + orig + "\n\n" + comp_msg["content"]}
        else:
            messages.insert(0, {"role": "system", "content": PERSONA_SYSTEM + "\n\n" + comp_msg["content"]})
    else:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": PERSONA_SYSTEM})
        else:
            c = messages[0].get("content") or ""
            messages[0] = {"role": "system", "content": PERSONA_SYSTEM + "\n\n" + c}

    payload["messages"] = messages
    payload.setdefault("chat_template_kwargs", {"enable_thinking": False})

    headers = {"content-type": "application/json"}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            f"{LLAMA_BASE}/v1/chat/completions",
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
        )

    # Fire-and-forget memory update on success
    if r.status_code == 200:
        try:
            data = json.loads(r.content.decode("utf-8"))
            reply = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            if user_line and reply:
                remember_turn(_COMPANION_DIR, _OWNER_ID, user_line, reply)
        except Exception:
            pass

    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_v1(path: str, req: Request):
    url = f"{LLAMA_BASE}/v1/{path}"
    body = await req.body()
    headers = dict(req.headers)
    headers.pop("host", None)
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.request(req.method, url, content=body, headers=headers, params=req.query_params)
    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
