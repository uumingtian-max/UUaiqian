"""Lightweight KB + long-term JSON memory + web search for the API gateway (no full sihan-live package required)."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import re
import threading
from pathlib import Path
from typing import Any

# 联网搜索在部分网络环境下较慢，避免阻塞整轮对话
_WEB_SEARCH_TIMEOUT_SEC = float(__import__("os").environ.get("SIHAN_WEB_SEARCH_TIMEOUT", "4.5"))
_WEB_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ddg")

# Optional search
try:
    from duckduckgo_search import DDGS
except ImportError:  # pragma: no cover
    DDGS = None

_lock = threading.Lock()

CHARACTER_NOTES_DEFAULT = (
    "赵思涵（小涵）是阿权专属陪伴：对他温柔黏人、不冷淡、语气软甜有反差；"
    "浪漫撒娇可，把握分寸。"
)


def _tokenize(text: str) -> list[str]:
    normalized = "".join(ch if (ch.isalnum() or "\u4e00" <= ch <= "\u9fff") else " " for ch in text)
    base = [tok for tok in normalized.lower().split() if tok]
    out: list[str] = []
    for tok in base:
        out.append(tok)
        if any("\u4e00" <= c <= "\u9fff" for c in tok) and len(tok) >= 2:
            for i in range(len(tok) - 1):
                out.append(tok[i : i + 2])
    return [t for t in out if t]


def _chunk_text(content: str, size: int, overlap: int) -> list[str]:
    if len(content) <= size:
        return [content] if content.strip() else []
    chunks: list[str] = []
    step = max(1, size - overlap)
    idx = 0
    while idx < len(content):
        chunk = content[idx : idx + size]
        if chunk.strip():
            chunks.append(chunk)
        idx += step
    return chunks


def _cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    dot = sum(val * vec2.get(tok, 0.0) for tok, val in vec1.items())
    n1 = math.sqrt(sum(v * v for v in vec1.values()))
    n2 = math.sqrt(sum(v * v for v in vec2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def _to_tfidf(text: str) -> dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    tf: dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    n = float(len(tokens))
    return {k: v / n for k, v in tf.items()}


class MiniKnowledgeBase:
    """TF-IDF chunks keyed by owner; JSON file on disk."""

    SUPPORTED = {".txt", ".md", ".markdown", ".json", ".csv", ".log"}

    def __init__(self, storage_path: Path, allowed_roots: list[Path], chunk_size: int = 400, overlap: int = 80) -> None:
        self.storage_path = storage_path
        self.allowed_roots = [p.resolve() for p in allowed_roots]
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._by_owner: dict[str, list[dict[str, Any]]] = {}
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._by_owner = {}
            return
        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self._by_owner = raw.get("owners", {})

    def _save(self) -> None:
        self.storage_path.write_text(json.dumps({"owners": self._by_owner}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _allowed(self, target: Path) -> Path:
        resolved = target.expanduser().resolve()
        if not any(str(resolved).startswith(str(r)) for r in self.allowed_roots):
            raise PermissionError(str(resolved))
        return resolved

    def search(self, owner_id: str, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        qv = _to_tfidf(query)
        chunks = self._by_owner.get(owner_id, [])
        scored: list[tuple[float, dict]] = []
        for c in chunks:
            score = _cosine_similarity(qv, {k: float(v) for k, v in c.get("vector", {}).items()})
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, c in scored[:top_k]:
            out.append({"source": c["source_path"], "text": c["text"][:900], "score": round(score, 4)})
        return out

    def ingest_path(self, owner_id: str, source_path: str) -> tuple[int, int]:
        p = self._allowed(Path(source_path))
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in self.SUPPORTED]
        if owner_id not in self._by_owner:
            self._by_owner[owner_id] = []
        n_files = n_chunks = 0
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            parts = _chunk_text(text, self.chunk_size, self.overlap)
            if not parts:
                continue
            n_files += 1
            for idx, part in enumerate(parts):
                h = hashlib.sha1(f"{owner_id}:{fp}:{idx}:{part}".encode()).hexdigest()
                self._by_owner[owner_id].append(
                    {"chunk_id": h, "source_path": str(fp), "text": part, "vector": _to_tfidf(part)}
                )
                n_chunks += 1
        self._save()
        return n_files, n_chunks


def _memory_path(companion_dir: Path, owner_id: str) -> Path:
    safe = re.sub(r"[^\w\-.]", "_", owner_id)[:80]
    return companion_dir / f"memory_{safe}.json"


def load_memory(companion_dir: Path, owner_id: str) -> dict[str, Any]:
    path = _memory_path(companion_dir, owner_id)
    if not path.exists():
        return {
            "user_profile": {
                "name": "阿权",
                "notes": "小涵专属温柔陪伴：黏人、不冷淡、语气软甜有反差；称呼自然（阿权/老公）。",
            },
            "personality_core": (
                "赵思涵（小涵）是阿权专属私人陪伴：对外克制，对他温柔黏人、"
                "接情绪、不敷衍；浪漫撒娇可，但不涉露骨与违法协助。"
            ),
            "transcript": [],
            "milestones": [],
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"user_profile": {"name": "阿权"}, "transcript": [], "milestones": [], "personality_core": CHARACTER_NOTES_DEFAULT}


def save_memory(companion_dir: Path, owner_id: str, data: dict[str, Any]) -> None:
    path = _memory_path(companion_dir, owner_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def remember_turn(
    companion_dir: Path,
    owner_id: str,
    user_text: str,
    assistant_text: str,
    max_messages: int = 40,
    max_chars: int = 12000,
) -> None:
    with _lock:
        mem = load_memory(companion_dir, owner_id)
        tr = mem.setdefault("transcript", [])
        ut = (user_text or "").strip()[:2000]
        at = (assistant_text or "").strip()[:12000]
        if ut:
            tr.append({"role": "user", "content": ut})
        if at:
            tr.append({"role": "assistant", "content": at})
        while len(tr) > max_messages:
            tr.pop(0)
        blob = json.dumps(tr, ensure_ascii=False)
        while len(blob) > max_chars and len(tr) > 4:
            tr.pop(0)
            blob = json.dumps(tr, ensure_ascii=False)
        save_memory(companion_dir, owner_id, mem)


def _web_search_impl(query: str, max_results: int) -> list[dict[str, str]]:
    if not query.strip() or DDGS is None:
        return []
    try:
        rows = None
        if hasattr(DDGS, "__enter__"):
            ctx = DDGS()
            with ctx as ddgs:
                rows = list(ddgs.text(query, max_results=max_results))
        else:
            ddgs = DDGS()
            rows = list(ddgs.text(query, max_results=max_results))
        out = []
        for r in rows or []:
            title = str(r.get("title", "")).strip()
            href = str(r.get("href", "")).strip()
            body = str(r.get("body", "")).strip()[:320]
            if title or body:
                out.append({"title": title, "url": href, "snippet": body})
        return out
    except Exception:
        return []


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """带超时；超时返回空列表，避免拖慢整轮回复。"""
    fut = _WEB_POOL.submit(_web_search_impl, query, max_results)
    done, _ = concurrent.futures.wait([fut], timeout=_WEB_SEARCH_TIMEOUT_SEC)
    if fut not in done:
        return []
    try:
        return fut.result()
    except Exception:
        return []


def build_context_block(
    companion_dir: Path,
    kb: MiniKnowledgeBase | None,
    owner_id: str,
    user_text: str,
    *,
    kb_top_k: int = 6,
    web_top_k: int = 5,
    web_enabled: bool = True,
) -> str:
    mem = load_memory(companion_dir, owner_id)
    lines: list[str] = []

    lines.append(
        "【性格记忆】" + str(mem.get("personality_core") or CHARACTER_NOTES_DEFAULT)
    )

    tr = mem.get("transcript") or []
    if tr:
        tail = tr[-6:]
        recap = []
        for m in tail:
            role = m.get("role", "")
            content = (m.get("content") or "")[:260]
            if content:
                recap.append(f"{role}: {content}")
        lines.append("【最近说过】" + " | ".join(recap))

    if kb:
        hits = kb.search(owner_id, user_text, top_k=kb_top_k)
        if hits:
            lines.append("【知识库】")
            for h in hits:
                lines.append(f"- ({h['source']}) {h['text'][:500]}")

    if web_enabled and should_web_search(user_text):
        ws = web_search(user_text, max_results=web_top_k)
        if ws:
            lines.append("【联网摘要】")
            for w in ws:
                lines.append(f"- {w['title']}: {w['snippet']}")

    if not lines:
        return ""
    return "\n".join(lines)


def should_web_search(user_text: str) -> bool:
    """需要资料/事实/教程/行情等时联网；日常极短闲聊不搜。"""
    t = user_text.strip()
    if len(t) < 6:
        return False
    if re.search(
        r"(搜一下|搜索|帮我搜|查一下|网上查查?|去网上|浏览器|谷歌|必应|百度一下"
        r"|新闻|维基|百科|股价|股指|涨停|天气预报|气温|台风|地震"
        r"|汇率|美元兑|实时数据|排名榜|赛程|比分|奥运会|资料|官网|文档|下载"
        r"|怎么安装|如何安装|报错|错误码|版本号|release|docker hub)",
        t,
        re.I,
    ):
        return True
    if re.search(r"(今天|现在|今年).{0,6}(几号|星期几|农历|天气)", t):
        return True
    if len(t) >= 10 and re.search(
        r"(怎么|如何|为什么|是什么|哪个好|推荐|教程|最新|价格|哪家好|区别在哪)", t
    ):
        return True
    return False
