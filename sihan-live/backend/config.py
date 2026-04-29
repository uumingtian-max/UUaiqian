from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppConfig:
    name: str
    owner_id: str
    api_key: str
    host: str
    port: int


@dataclass(slots=True)
class KnowledgeBaseConfig:
    storage_path: Path
    chunk_size: int
    chunk_overlap: int
    allowed_roots: list[Path]


@dataclass(slots=True)
class Settings:
    app: AppConfig
    knowledge_base: KnowledgeBaseConfig


def _to_abs_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_settings(config_path: Path | None = None) -> Settings:
    """Load runtime settings from YAML config."""
    resolved_path = config_path or (Path(__file__).resolve().parents[1] / "config.yaml")
    base_dir = resolved_path.parent

    raw: dict[str, Any]
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    app_raw = raw.get("app", {})
    kb_raw = raw.get("knowledge_base", {})
    app = AppConfig(
        name=app_raw.get("name", "sihan-live-backend"),
        owner_id=app_raw.get("owner_id", "nac"),
        api_key=app_raw.get("api_key", "local-dev-key"),
        host=app_raw.get("host", "0.0.0.0"),
        port=int(app_raw.get("port", 8000)),
    )
    allowed_roots = [
        _to_abs_path(base_dir, root) for root in kb_raw.get("allowed_roots", [])
    ]
    knowledge_base = KnowledgeBaseConfig(
        storage_path=_to_abs_path(base_dir, kb_raw.get("storage_path", "./data/knowledge_store.json")),
        chunk_size=int(kb_raw.get("chunk_size", 400)),
        chunk_overlap=int(kb_raw.get("chunk_overlap", 80)),
        allowed_roots=allowed_roots,
    )
    return Settings(app=app, knowledge_base=knowledge_base)
