"""Persistent emotional companion with long-term memory and knowledge snippets."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .memory import KnowledgeBase


class EmotionalGirlfriendEngine:
    def __init__(self, knowledge_store: KnowledgeBase, memory_dir: Path | None = None) -> None:
        self.knowledge_store = knowledge_store
        self.memory_dir = memory_dir or Path(__file__).resolve().parents[1] / "data"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _memory_path(self, user_id: str) -> Path:
        safe = re.sub(r"[^\w\-.]", "_", user_id)[:64]
        return self.memory_dir / f"girlfriend_memory_{safe}.json"

    def _load(self, user_id: str) -> dict:
        path = self._memory_path(user_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "user_profile": {
                "name": "阿权",
                "status": "",
                "preferences": "",
            },
            "relationship_notes": "",
            "emotional_history": [],
            "transcript": [],
        }

    def _save(self, user_id: str, data: dict) -> None:
        try:
            self._memory_path(user_id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _append_transcript(self, mem: dict, role: str, content: str, max_items: int = 40) -> None:
        tr = mem.setdefault("transcript", [])
        content = (content or "").strip()[:4000]
        if not content:
            return
        tr.append({"role": role, "content": content, "t": datetime.now().isoformat(timespec="seconds")})
        while len(tr) > max_items:
            tr.pop(0)

    def respond(self, user_id: str, message: str, top_k: int = 4) -> tuple[str, list[dict[str, str]]]:
        mem = self._load(user_id)
        snippets = self.knowledge_store.search(query=message, owner_id=user_id, top_k=top_k)

        profile = mem.get("user_profile") or {}
        name = profile.get("name") or "你"
        recall_parts: list[str] = []
        if profile.get("status"):
            recall_parts.append(f"我记得{name}：{profile['status']}")
        if profile.get("preferences"):
            recall_parts.append(f"你提过喜欢{profile['preferences']}")

        hist = mem.get("emotional_history", [])[-4:]
        if hist:
            recall_parts.append("最近心情线索：" + "；".join(hist))

        tr = mem.get("transcript", [])[-4:]
        if tr:
            last_bits = []
            for row in tr:
                pref = "他" if row.get("role") == "user" else "我"
                last_bits.append(f"{pref}：{(row.get('content') or '')[:80]}")
            recall_parts.append("接上口气：" + " | ".join(last_bits))

        kb_lines = []
        for s in snippets[:3]:
            t = (s.get("text") or "")[:200].replace("\n", " ")
            src = s.get("source") or ""
            if t:
                kb_lines.append(f"· {t}（出自 {src}）")

        recall = "\n".join(recall_parts)
        kb_block = "\n".join(kb_lines)

        opinion = (
            f"{name}，我刚在听你说话呢。"
            f"{' 我这边想起：' + recall if recall else ''}"
            f"{' 知识库里也有能对上的：' + kb_block if kb_block else ''}"
        )

        answer = (
            f"{opinion}\n\n"
            f"嗯……你怎么突然找我啦？（声音放软一点）我就在这儿呢，不急，你慢慢说……"
        )

        mem["last_interaction"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        mem.setdefault("emotional_history", []).append(
            f"{mem['last_interaction']}：他说「{message[:40]}…」"
        )
        if len(mem["emotional_history"]) > 60:
            mem["emotional_history"] = mem["emotional_history"][-60:]

        self._append_transcript(mem, "user", message)
        self._append_transcript(mem, "assistant", answer)
        self._save(user_id, mem)

        references = [{"id": str(s.get("id", "")), "source": str(s.get("source", ""))} for s in snippets]
        return answer, references
