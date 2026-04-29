"""Simple retrieval-augmented response engine."""

from __future__ import annotations

from datetime import datetime

from .memory import KnowledgeStore


class LLMEngine:
    """Lightweight stand-in for a persona LLM with RAG."""

    def __init__(self, knowledge_store: KnowledgeStore) -> None:
        self.knowledge_store = knowledge_store

    def respond(self, user_id: str, message: str) -> tuple[str, list[dict[str, str]]]:
        """Generate a response grounded in retrieved memory chunks."""
        snippets = self.knowledge_store.search(query=message, top_k=3)
        today = datetime.now().strftime("%Y-%m-%d")
        if snippets:
            recall = "\n".join(
                f"- {item['text'][:100]} (source: {item['source']})" for item in snippets
            )
            answer = (
                "我记得你之前提过这些重点，我按你的偏好继续：\n"
                f"{recall}\n"
                f"基于这些记忆，今天({today})我建议先从你最常提到的主题继续：{message}"
            )
        else:
            answer = (
                "我还没有检索到相关记忆，先把你的知识库喂满会更聪明。"
                f"你刚说的是：{message}"
            )
        self.knowledge_store.add_note(user_id=user_id, text=message, source="live-chat")
        references = [{"id": item["id"], "source": item["source"]} for item in snippets]
        return answer, references
