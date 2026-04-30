from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from backend.config import KnowledgeBaseConfig


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".log"}


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
        return [content]
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
    norm1 = math.sqrt(sum(val * val for val in vec1.values()))
    norm2 = math.sqrt(sum(val * val for val in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    source_path: str
    text: str
    vector: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "text": self.text,
            "vector": self.vector,
        }
    @classmethod
    def from_dict(cls, raw: dict) -> DocumentChunk:
        return cls(
            chunk_id=raw["chunk_id"],
            source_path=raw["source_path"],
            text=raw["text"],
            vector={k: float(v) for k, v in raw.get("vector", {}).items()},
        )


@dataclass(slots=True)
class IngestReport:
    imported_files: int = 0
    imported_chunks: int = 0
    skipped_files: int = 0
    errors: list[str] = field(default_factory=list)


class KnowledgeBase:
    """Per-owner TF-IDF chunks persisted in one JSON file."""

    def __init__(self, config: KnowledgeBaseConfig) -> None:
        self.config = config
        self.storage_path = config.storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._by_owner: dict[str, list[DocumentChunk]] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._by_owner = {}
            return
        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        if "owners" in raw:
            self._by_owner = {
                oid: [DocumentChunk.from_dict(x) for x in chunks]
                for oid, chunks in raw.get("owners", {}).items()
            }
        elif "chunks" in raw:
            self._by_owner = {"default": [DocumentChunk.from_dict(x) for x in raw.get("chunks", [])]}
        else:
            self._by_owner = {}

    def _persist(self) -> None:
        payload = {
            "owners": {
                oid: [c.to_dict() for c in chunks]
                for oid, chunks in self._by_owner.items()
            }
        }
        self.storage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _chunks_for(self, owner_id: str) -> list[DocumentChunk]:
        if owner_id in self._by_owner:
            return self._by_owner[owner_id]
        return self._by_owner.get("default", [])

    def _ensure_allowed(self, target: Path) -> Path:
        resolved = target.expanduser().resolve()
        allowed = any(str(resolved).startswith(str(root.resolve())) for root in self.config.allowed_roots)
        if not allowed:
            raise PermissionError(f"path {resolved} is outside allowed roots")
        return resolved

    def _iter_files(self, root: Path) -> Iterable[Path]:
        if root.is_file():
            if root.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield root
            return
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield file_path

    def ingest_directory(self, source_path: Path, owner_id: str) -> tuple[int, int]:
        report = self.ingest_path(str(source_path), owner_id=owner_id)
        return report.imported_files, report.imported_chunks

    def ingest_path(self, source_path: str, owner_id: str = "default") -> IngestReport:
        report = IngestReport()
        target = self._ensure_allowed(Path(source_path))
        all_files = list(self._iter_files(target))
        if owner_id not in self._by_owner:
            self._by_owner[owner_id] = []

        for file_path in all_files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                report.skipped_files += 1
                report.errors.append(f"skip {file_path}: non-utf8 file")
                continue
            chunks = _chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
            if not chunks:
                report.skipped_files += 1
                continue
            report.imported_files += 1
            for idx, chunk_text in enumerate(chunks):
                chunk_hash = hashlib.sha1(f"{owner_id}:{file_path}:{idx}:{chunk_text}".encode("utf-8")).hexdigest()
                vector = self._to_tfidf(chunk_text)
                self._by_owner[owner_id].append(
                    DocumentChunk(
                        chunk_id=chunk_hash,
                        source_path=str(file_path),
                        text=chunk_text,
                        vector=vector,
                    )
                )
                report.imported_chunks += 1

        if report.imported_chunks:
            self._persist()
        return report

    def _to_tfidf(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        tf: dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0.0) + 1.0
        token_count = float(len(tokens))
        return {token: freq / token_count for token, freq in tf.items()}

    def query(self, prompt: str, owner_id: str, top_k: int = 4) -> list[tuple[DocumentChunk, float]]:
        query_vec = self._to_tfidf(prompt)
        chunks = self._chunks_for(owner_id)
        scored = [(chunk, _cosine_similarity(query_vec, chunk.vector)) for chunk in chunks]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [item for item in scored if item[1] > 0][:top_k]

    def search(self, query: str, owner_id: str, top_k: int = 4) -> list[dict[str, str | float]]:
        out: list[dict[str, str | float]] = []
        for chunk, score in self.query(query, owner_id, top_k):
            out.append(
                {
                    "id": chunk.chunk_id,
                    "source": chunk.source_path,
                    "text": chunk.text[:1200],
                    "score": round(float(score), 4),
                }
            )
        return out

    def stats(self, owner_id: str) -> dict[str, int]:
        chunks = self._chunks_for(owner_id)
        return {
            "chunks": len(chunks),
            "sources": len({c.source_path for c in chunks}),
        }

    def list_chunks(self, owner_id: str, limit: int = 20) -> list[dict]:
        chunks = self._chunks_for(owner_id)[:limit]
        return [c.to_dict() for c in chunks]
