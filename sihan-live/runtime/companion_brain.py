"""Lightweight KB + long-term JSON memory + web search for the API gateway (no full sihan-live package required)."""

from __future__ import annotations

import concurrent.futures
import hashlib
import html as html_std
import ipaddress
import json
import math
import re
import secrets
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

# 联网搜索在部分网络环境下较慢，避免阻塞整轮对话
_WEB_SEARCH_TIMEOUT_SEC = float(__import__("os").environ.get("SIHAN_WEB_SEARCH_TIMEOUT", "3.0"))
_WEB_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ddg")

_FETCH_ENABLED = __import__("os").environ.get("SIHAN_FETCH", "1") not in ("0", "false", "no")
_FETCH_TIMEOUT = float(__import__("os").environ.get("SIHAN_FETCH_TIMEOUT", "14"))
_FETCH_MAX_BYTES = int(__import__("os").environ.get("SIHAN_FETCH_MAX_BYTES", str(2_000_000)))
_FETCH_MAX_URLS = int(__import__("os").environ.get("SIHAN_FETCH_MAX_URLS", "5"))
_FETCH_ALLOW_PRIVATE = __import__("os").environ.get("SIHAN_FETCH_ALLOW_PRIVATE", "0") in ("1", "true", "yes")
_FETCH_COOKIE = (__import__("os").environ.get("SIHAN_FETCH_COOKIE") or "").strip() or None
_UA = __import__("os").environ.get(
    "SIHAN_FETCH_UA",
    "Mozilla/5.0 (compatible; SihanCompanion/1.0; +private) AppleWebKit/537.36 (KHTML, like Gecko) Chrome-like",
)

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


def _read_file_as_index_text(fp: Path, max_bytes: int = 12_000_000) -> str:
    """任意扩展名：尽量当文本解码；否则登记为元数据块（仍可被关键词搜到）。"""
    try:
        raw = fp.read_bytes()
    except OSError:
        return ""
    if len(raw) > max_bytes:
        return f"[文件过大已截断索引: {fp.name}，约 {len(raw)} 字节。可改用较小 txt/md 或分包上传。]"
    if b"\x00" in raw[:4096] and not fp.suffix.lower() in {".txt", ".md", ".log", ".csv", ".json", ".xml", ".html", ".htm"}:
        return f"[二进制或编码未知，已登记文件名: {fp.name}，{len(raw)} 字节。需要全文时请提供 UTF-8 文本或常见文档格式。]"
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            t = raw.decode(enc)
            if t and not t.isspace():
                return t
        except UnicodeDecodeError:
            continue
    return f"[无法解码为文本: {fp.name}，{len(raw)} 字节]"


class MiniKnowledgeBase:
    """TF-IDF chunks keyed by owner; JSON file on disk."""

    # 仍保留集合供展示；ingest 时实际接受任意文件，见 _read_file_as_index_text
    SUPPORTED = {".txt", ".md", ".markdown", ".json", ".csv", ".log", ".py", ".yml", ".yaml", ".html", ".htm", ".xml", ".pdf", ".doc"}

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
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}
        if p.is_file():
            files = [p]
        else:
            files = []
            for f in p.rglob("*"):
                if not f.is_file():
                    continue
                if any(part in skip_dirs for part in f.parts):
                    continue
                files.append(f)
        if owner_id not in self._by_owner:
            self._by_owner[owner_id] = []
        n_files = n_chunks = 0
        for fp in files:
            text = _read_file_as_index_text(fp)
            if not text or not str(text).strip():
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

    def save_upload(self, owner_id: str, filename: str, data: bytes) -> Path:
        """把上传落到 kb_import/{owner}/uploads/ 下并返回绝对路径。"""
        root = self.allowed_roots[0].expanduser().resolve()
        self._allowed(root)
        sub = root / owner_id / "uploads"
        sub.mkdir(parents=True, exist_ok=True)
        self._allowed(sub)
        safe = re.sub(r"[^\w\-. \u4e00-\u9fff]", "_", Path(filename or "file").name)[:180]
        dest = sub / f"{secrets.token_hex(4)}_{safe}"
        dest.write_bytes(data)
        return dest.resolve()

    def ingest_uploaded_file(self, owner_id: str, dest_path: Path) -> tuple[int, int]:
        return self.ingest_path(owner_id, str(dest_path))

    def ingest_text_memory(self, owner_id: str, text: str, source_tag: str = "memory") -> int:
        """把一段纯文本直接切块写入 KB（不落盘为独立文件）。"""
        raw = (text or "").strip()
        if not raw:
            return 0
        if owner_id not in self._by_owner:
            self._by_owner[owner_id] = []
        parts = _chunk_text(raw, self.chunk_size, self.overlap)
        n = 0
        tag = re.sub(r"[^\w\-.]", "_", source_tag)[:80] or "note"
        for idx, part in enumerate(parts):
            h = hashlib.sha1(f"{owner_id}:{tag}:{idx}:{part}".encode()).hexdigest()
            self._by_owner[owner_id].append(
                {"chunk_id": h, "source_path": f"<{tag}>", "text": part, "vector": _to_tfidf(part)}
            )
            n += 1
        self._save()
        return n


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
            "long_term": {"facts": [], "preferences": "", "relationship_notes": ""},
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("long_term", {"facts": [], "preferences": "", "relationship_notes": ""})
        lt0 = data["long_term"]
        if isinstance(lt0, dict):
            lt0.setdefault("facts", [])
            lt0.setdefault("preferences", "")
            lt0.setdefault("relationship_notes", "")
        return data
    except Exception:
        return {"user_profile": {"name": "阿权"}, "transcript": [], "milestones": [], "personality_core": CHARACTER_NOTES_DEFAULT, "long_term": {"facts": [], "preferences": "", "relationship_notes": ""}}


def save_memory(companion_dir: Path, owner_id: str, data: dict[str, Any]) -> None:
    path = _memory_path(companion_dir, owner_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_MAX_FACTS = 64
_MAX_FACT_LEN = 220


def _strip_overlap(new_items: list[str], old: list[str]) -> list[str]:
    out: list[str] = []
    for s in new_items:
        t = (s or "").strip()[:_MAX_FACT_LEN]
        if len(t) < 8:
            continue
        if any(t in o or o in t for o in old):
            continue
        out.append(t)
    return out


def merge_long_term_from_turn(mem: dict[str, Any], user_text: str, assistant_text: str) -> None:
    """从对话里粗抽「像人记住的事」：偏好/事实/约定，写入 long_term；不靠整段聊天堆砌。"""
    ut = (user_text or "").strip()
    at = (assistant_text or "").strip()
    if len(ut) < 6 and len(at) < 10:
        return
    blob = f"{ut}\n{at}"
    lt = mem.setdefault("long_term", {})
    facts = list(lt.get("facts") or [])
    prefs = str(lt.get("preferences") or "").strip()
    rel = str(lt.get("relationship_notes") or "").strip()

    # 简单规则：用户自述偏好
    for m in re.finditer(
        r"(我(?:不)?喜欢|我爱|我讨厌|我习惯|别喊我|叫我)([^。！？\n]{2,80})",
        ut,
    ):
        frag = (m.group(1) + m.group(2)).strip()
        for f in _strip_overlap([frag], facts):
            facts.append(f)
    for m in re.finditer(r"以后(?:都)?要记(?:住)?[:：]\s*([^。\n]{4,120})", ut):
        for f in _strip_overlap([m.group(1).strip()], facts):
            facts.append(f)

    # 助手侧明确复述「记住啦」类
    if re.search(r"记住|记下了|记心上|帮你记", at) and len(ut) > 8:
        line = f"曾约定/提到过：{ut[:100]}"
        for f in _strip_overlap([line], facts):
            facts.append(f)

    if re.search(r"(口味|爱吃|不吃|忌口|怕辣|素食|海鲜)", blob):
        hit = re.search(r"([^。！？\n]{0,40}(?:口味|爱吃|不吃|忌口|怕辣|素食|海鲜)[^。！？\n]{0,60})", ut)
        if hit:
            p = hit.group(1).strip()[:180]
            if p and p not in prefs:
                prefs = (prefs + "；" + p).strip("；") if prefs else p
    if re.search(r"(纪念日|在一起|第一次|咱们|老公|阿权)", blob) and len(ut) > 10:
        hit = re.search(r"([^。！？\n]{8,100}(?:纪念日|在一起|第一次)[^。！？\n]{0,40})", ut + at)
        if hit:
            rnote = hit.group(1).strip()[:200]
            if rnote and rnote not in rel:
                rel = (rel + "；" + rnote).strip("；") if rel else rnote

    while len(facts) > _MAX_FACTS:
        facts.pop(0)
    lt["facts"] = facts
    lt["preferences"] = prefs[-1200:] if len(prefs) > 1200 else prefs
    lt["relationship_notes"] = rel[-1200:] if len(rel) > 1200 else rel


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
        merge_long_term_from_turn(mem, ut, at)
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


def _extract_urls_from_text(text: str) -> list[str]:
    raw = re.findall(r"https?://[^\s\]>\\)\"'）\]〉】]+", text or "", flags=re.I)
    out: list[str] = []
    seen: set[str] = set()
    for u in raw:
        u = u.rstrip(".,;:!?，。；：（）【】)")
        if len(u) < 12:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= _FETCH_MAX_URLS:
            break
    return out


def _host_is_literal_private(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local)
    except ValueError:
        return False


def _url_fetch_allowed(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").strip().lower()
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "::1"):
        return _FETCH_ALLOW_PRIVATE
    if _host_is_literal_private(host) and not _FETCH_ALLOW_PRIVATE:
        return False
    return True


def _html_to_text(html: str, limit: int = 12000) -> str:
    s = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    s = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = html_std.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def _fetch_one_url(url: str) -> dict[str, str]:
    if not _FETCH_ENABLED or not _url_fetch_allowed(url):
        return {"url": url, "title": "", "text": "", "error": "blocked_or_disabled"}
    headers = {"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8"}
    if _FETCH_COOKIE:
        headers["Cookie"] = _FETCH_COOKIE
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(_FETCH_TIMEOUT),
            headers=headers,
        ) as client:
            r = client.get(url)
        if r.status_code >= 400:
            return {"url": url, "title": "", "text": "", "error": f"HTTP {r.status_code}"}
        ct = (r.headers.get("content-type") or "").lower()
        body = r.content[:_FETCH_MAX_BYTES]
        text = ""
        title = ""
        if "html" in ct or body.lstrip().startswith(b"<"):
            dec = body.decode("utf-8", errors="replace")
            tm = re.search(r"(?is)<title[^>]*>([^<]{0,200})</title>", dec)
            if tm:
                title = html_std.unescape(tm.group(1).strip())
            text = _html_to_text(dec)
        else:
            text = body.decode("utf-8", errors="replace")
            text = re.sub(r"\s+", " ", text).strip()[:12000]
        if not text and not title:
            return {"url": url, "title": title, "text": "", "error": "empty_body"}
        return {"url": url, "title": title, "text": text, "error": ""}
    except Exception as e:
        return {"url": url, "title": "", "text": "", "error": str(e)[:200]}


def fetch_urls_text(urls: list[str]) -> list[dict[str, str]]:
    if not urls:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(urls))) as pool:
        futs = [pool.submit(_fetch_one_url, u) for u in urls]
        out = []
        for fut in concurrent.futures.as_completed(futs, timeout=_FETCH_TIMEOUT + 25):
            try:
                out.append(fut.result())
            except Exception as e:
                out.append({"url": "", "title": "", "text": "", "error": str(e)[:120]})
    return out


def build_context_block(
    companion_dir: Path,
    kb: MiniKnowledgeBase | None,
    owner_id: str,
    user_text: str,
    *,
    retrieval_query: str | None = None,
    multimodal: bool = False,
    kb_top_k: int = 6,
    web_top_k: int = 5,
    web_enabled: bool = True,
) -> str:
    mem = load_memory(companion_dir, owner_id)
    lines: list[str] = []
    rq = (retrieval_query if retrieval_query is not None else user_text) or ""
    rq = rq.strip() or (user_text or "").strip()

    lines.append(
        "【性格记忆】" + str(mem.get("personality_core") or CHARACTER_NOTES_DEFAULT)
    )

    lt = mem.get("long_term") or {}
    lt_parts: list[str] = []
    facts = lt.get("facts") or []
    if isinstance(facts, list) and facts:
        for f in facts[-16:]:
            if isinstance(f, str) and f.strip():
                lt_parts.append(f.strip()[:_MAX_FACT_LEN])
    pref = str(lt.get("preferences") or "").strip()
    if pref:
        lt_parts.append("偏好：" + pref[:400])
    reln = str(lt.get("relationship_notes") or "").strip()
    if reln:
        lt_parts.append("关系：" + reln[:400])
    if lt_parts:
        lines.append("【长期记得（摘）】" + " | ".join(lt_parts))

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
        hits = kb.search(owner_id, rq, top_k=kb_top_k)
        if hits:
            lines.append("【知识库】")
            for h in hits:
                lines.append(f"- ({h['source']}) {h['text'][:500]}")

    if _FETCH_ENABLED:
        blob_urls = f"{user_text}\n{rq}"
        direct = _extract_urls_from_text(blob_urls)
        if direct:
            rows = fetch_urls_text(direct)
            lines.append("【直链页面摘录】(服务器代抓取，未收录站也可用)")
            for row in rows:
                err = (row.get("error") or "").strip()
                if err:
                    lines.append(f"- {row.get('url', '')} → {err}")
                    continue
                tit = row.get("title") or ""
                tx = (row.get("text") or "")[:3500]
                head = f"{row.get('url', '')}"
                if tit:
                    head += f" | {tit}"
                lines.append(f"- {head}\n  {tx}")

    if web_enabled and should_web_search(user_text, multimodal=multimodal):
        q_web = rq if len(rq) >= 6 else user_text
        ws = web_search(q_web, max_results=web_top_k)
        if ws:
            lines.append("【联网摘要】")
            for w in ws:
                lines.append(f"- {w['title']}: {w['snippet']}")

    if not lines:
        return ""
    return "\n".join(lines)


def should_web_search(user_text: str, *, multimodal: bool = False) -> bool:
    """需要事实/资料时联网；带图或视频帧时略放宽，方便「自己接着搜」。"""
    t = user_text.strip()
    if multimodal and len(t) < 6:
        return True
    if len(t) < 4:
        return False
    if re.search(
        r"(https?://|打开这个链接|看这个网址|页面\s*http|原文链接|未收录|内网文档|局域网)",
        t,
        re.I,
    ):
        return True
    if re.search(
        r"(搜一下|搜索|帮我搜|查一下|网上查查?|去网上|浏览器|谷歌|必应|百度一下"
        r"|全世界|全球|国外|海外|国际新闻|外媒"
        r"|新闻|维基|百科|股价|股指|涨停|天气预报|气温|台风|地震"
        r"|汇率|美元兑|实时数据|排名榜|赛程|比分|奥运会|官网|文档\s*下载"
        r"|怎么安装|如何安装|安装教程|报错|错误码|版本号|release|docker hub)",
        t,
        re.I,
    ):
        return True
    if re.search(r"(今天|现在|今年).{0,6}(几号|星期几|农历|天气)", t):
        return True
    if len(t) >= 10 and re.search(
        r"(怎么|如何|为什么|是什么|啥是|哪个|哪些|多少钱|价格|推荐吗|靠谱吗"
        r"|最新|教程|对比|区别|原理|标准|法规|政策|数据|统计)",
        t,
    ):
        return True
    if multimodal and len(t) >= 8 and not re.search(
        r"^(在吗|在么|嗯|哦|好|想你|抱抱|老婆|老公|小涵|干嘛|早|晚安)[…!！?？。.~\s]*$",
        t,
    ):
        return True
    return False


def build_retrieval_query_from_messages(messages: list) -> tuple[str, bool]:
    """从最后一轮 user 消息拼检索用 query；multimodal 表示含图/图链。"""
    payload = None
    for m in reversed(messages or []):
        if m.get("role") == "user":
            payload = m.get("content")
            break
    if payload is None:
        return "", False
    if isinstance(payload, str):
        return payload.strip(), False
    if not isinstance(payload, list):
        return "", False
    texts: list[str] = []
    has_image = False
    for p in payload:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            texts.append(p.get("text") or "")
        elif p.get("type") == "image_url":
            has_image = True
    s = " ".join(texts).strip()
    extra: list[str] = []
    if has_image:
        extra.append("附图画面 场景 人物 物体 外观 可能相关实体")
    if re.search(r"视频|截帧|一帧|画面里", s):
        extra.append("视频镜头 动作 环境")
    if extra:
        s = (s + " " + " ".join(extra)).strip()
    return s, has_image or bool(re.search(r"视频|截帧|一帧", s))


def append_kb_note(kb: MiniKnowledgeBase, owner_id: str, text: str, source_tag: str = "assistant_kb") -> int:
    """把助手产出的 ```kb 文本追加为虚拟文档并入库。"""
    raw = (text or "").strip()
    if not raw:
        return 0
    return kb.ingest_text_memory(owner_id, raw, source_tag)
