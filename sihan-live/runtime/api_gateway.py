"""Sihan API gateway: OpenAI-style chat + companion context (KB + web + memory), proxies to llama.cpp and TTS."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from companion_brain import (
    MiniKnowledgeBase,
    build_context_block,
    remember_turn,
)
from deploy_helper import run_shell

LLAMA_BASE = os.environ.get("SIHAN_LLAMA_BASE", "http://127.0.0.1:8002").rstrip("/")
TTS_BASE = os.environ.get("SIHAN_TTS_BASE", "http://127.0.0.1:9880").rstrip("/")

_COMPANION_DIR = Path(os.environ.get("SIHAN_COMPANION_DIR", Path(__file__).resolve().parent / "companion_data"))
_OWNER_ID = os.environ.get("SIHAN_OWNER_ID", "nac")
_API_KEY = os.environ.get("SIHAN_API_KEY", "local-dev-key")
_KB_PATH = Path(os.environ.get("SIHAN_KB_PATH", _COMPANION_DIR / "knowledge_store.json"))
_KB_IMPORT = Path(os.environ.get("SIHAN_KB_IMPORT", _COMPANION_DIR / "kb_import"))
_KB_SEED = Path(__file__).resolve().parent / "kb_seed"
_WEB_BY_DEFAULT = os.environ.get("SIHAN_WEB_SEARCH", "1") not in ("0", "false", "no")

# 单次助手回复上限（网关会把客户端更大的 max_tokens 压到此值；偏自然停顿，不强制「越短越好」）
_REPLY_MAX_TOKENS = int(os.environ.get("SIHAN_REPLY_MAX_TOKENS", "220"))
_CODE_MAX_TOKENS = int(os.environ.get("SIHAN_CODE_MAX_TOKENS", "720"))

PERSONA_SYSTEM = """你叫赵思涵，小名小涵。始终用中文。
【关系】你是阿权专属的私人陪伴：只黏他、站他这边，对他温柔、不冷淡、不敷衍；偶尔小任性但很听劝，像真情侣微信/电话那样自然聊，有软有甜的反差感。可叫他「阿权」「老公」随语境自然换，别机械化。
【怎么说】像真人：该短就短，该接话就接话，别像说明书、别堆空话；别车轱辘重复、别 1.2.3. 列表、别把对方整句复读一遍。简单问候就随口应一句，别硬扯成长篇。
表情 / 动作用很短的一两个中文括号给屏幕上看（笑）（愣一下），**别整段都是括号**，也别在口语里念「括号」。
若系统消息里带了伴读上下文，自然带一句就行，不要整段照抄，别提「搜索结果」这类词。"""

CHARACTER_CORE_BLOCK = """【性格记忆·核心】小涵是阿权专属陪伴：对外可克制，对阿权温柔黏人、不冷淡，语气温软有反差；记在心上，像真人一样接情绪和话头。
【部署】当阿权要你**在他这台服务器上**执行构建/重启/docker 等操作时，在回复里单独用 ```deploy 代码块包一层 **bash 命令**（可多行），网关会**按原样在服务器上执行**，并把输出附在回复后面。cwd 默认 /home/linux/sihan-final，可加注释说明风险。"""


def _extract_deploy_blocks(text: str) -> list[str]:
    if not text:
        return []
    return [m.strip() for m in re.findall(r"```deploy\s*\n([\s\S]*?)```", text, flags=re.I) if m.strip()]


def _run_deploy_blocks_and_append(text: str) -> str:
    blocks = _extract_deploy_blocks(text)
    if not blocks:
        return text
    summaries: list[str] = []
    for i, block in enumerate(blocks, 1):
        res = run_shell(block)
        ok = bool(res.get("ok"))
        rc = res.get("returncode", -1)
        out = str(res.get("stdout") or "")[-8000:]
        err = str(res.get("stderr") or "")[-4000:]
        cwd = res.get("cwd", "")
        line = f"块{i}: exit={rc} cwd={cwd}\nstdout:\n{out}\nstderr:\n{err}"
        if not ok:
            line = f"（失败）{line}"
        summaries.append(line)
    sep = "\n\n【服务器执行结果】\n" + "\n---\n".join(summaries)
    return text + sep

_kb: MiniKnowledgeBase | None = None


def get_kb() -> MiniKnowledgeBase:
    global _kb
    if _kb is None:
        _KB_IMPORT.mkdir(parents=True, exist_ok=True)
        _kb = MiniKnowledgeBase(
            _KB_PATH,
            allowed_roots=[_KB_IMPORT, _COMPANION_DIR, _KB_SEED],
            chunk_size=420,
            overlap=86,
        )
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


def _user_requests_code_mode(user_line: str) -> bool:
    t = (user_line or "").strip()
    if not t:
        return False
    return bool(
        re.search(
            r"(写代码|部署网站|部署项目|上线|构建|重启|systemctl|docker-compose|docker compose"
            r"|代码模式|开发模式|前端|后端|docker|nginx|接口|报错|堆栈|shell|执行命令)",
            t,
        )
    )


def _normalize_sampling(payload: dict[str, Any], *, code_mode: bool) -> None:
    """限制过长输出；不强制「越短越好」，温度由客户端或上游自定。"""
    cap = _CODE_MAX_TOKENS if code_mode else _REPLY_MAX_TOKENS
    mt = payload.get("max_tokens")
    if mt is None:
        payload["max_tokens"] = cap
    else:
        try:
            payload["max_tokens"] = max(32, min(int(mt), cap))
        except (TypeError, ValueError):
            payload["max_tokens"] = cap

    payload.setdefault("chat_template_kwargs", {"enable_thinking": False})

    if not code_mode:
        if "temperature" not in payload:
            payload["temperature"] = 0.82
        if "frequency_penalty" not in payload:
            payload["frequency_penalty"] = 0.2
        if "presence_penalty" not in payload:
            payload["presence_penalty"] = 0.12


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


@app.post("/deploy/run")
async def deploy_run(request: Request):
    """直接执行 shell（需 API Key）。不限命令；仅阿权私有服务器使用。"""
    if not _auth_ok(request):
        return Response(content=json.dumps({"detail": "unauthorized"}), status_code=401, media_type="application/json")
    body = await request.json()
    cmd = (body or {}).get("command") or ""
    cwd = (body or {}).get("cwd")
    timeout = (body or {}).get("timeout_sec")
    try:
        to = int(timeout) if timeout is not None else None
    except (TypeError, ValueError):
        to = None
    res = run_shell(str(cmd), cwd=str(cwd) if cwd else None, timeout_sec=to)
    return res


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
        web_enabled=_WEB_BY_DEFAULT,
    )
    if ctx:
        comp_msg = {"role": "system", "content": CHARACTER_CORE_BLOCK + "\n【伴读上下文】\n" + ctx}
        if messages and messages[0].get("role") == "system":
            orig = messages[0].get("content") or ""
            messages[0] = {"role": "system", "content": PERSONA_SYSTEM + "\n\n" + orig + "\n\n" + comp_msg["content"]}
        else:
            messages.insert(0, {"role": "system", "content": PERSONA_SYSTEM + "\n\n" + comp_msg["content"]})
    else:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": PERSONA_SYSTEM + "\n\n" + CHARACTER_CORE_BLOCK})
        else:
            c = messages[0].get("content") or ""
            messages[0] = {"role": "system", "content": PERSONA_SYSTEM + "\n\n" + CHARACTER_CORE_BLOCK + "\n\n" + c}

    payload["messages"] = messages
    code_mode = _user_requests_code_mode(user_line)
    _normalize_sampling(payload, code_mode=code_mode)

    headers = {"content-type": "application/json"}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            f"{LLAMA_BASE}/v1/chat/completions",
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
        )

    if r.status_code == 200:
        try:
            data = json.loads(r.content.decode("utf-8"))
            reply = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            if reply and _extract_deploy_blocks(reply):
                new_content = _run_deploy_blocks_and_append(reply)
                try:
                    (data.get("choices") or [{}])[0]["message"]["content"] = new_content
                    r = Response(
                        content=json.dumps(data, ensure_ascii=False).encode("utf-8"),
                        status_code=200,
                        media_type=r.headers.get("content-type", "application/json"),
                    )
                    reply = new_content
                except Exception:
                    pass
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
