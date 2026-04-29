from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import create_app


def build_client(tmp_path: Path) -> TestClient:
    config = tmp_path / "config.yaml"
    kb_root = tmp_path / "kb_data"
    kb_root.mkdir(parents=True, exist_ok=True)
    (kb_root / "notes.txt").write_text(
        "你最喜欢的电影是《海上钢琴师》。\n我们每月纪念日是 18 号。\n",
        encoding="utf-8",
    )
    config.write_text(
        "\n".join(
            [
                "app:",
                "  name: sihan-live-backend",
                "  owner_id: nac",
                "  api_key: test-key",
                "  host: 127.0.0.1",
                "  port: 8000",
                "knowledge_base:",
                f"  storage_path: {tmp_path / 'store.json'}",
                "  chunk_size: 80",
                "  chunk_overlap: 10",
                "  allowed_roots:",
                f"    - {kb_root}",
            ]
        ),
        encoding="utf-8",
    )
    app = create_app(config_path=config)
    return TestClient(app)


def auth_headers() -> dict[str, str]:
    return {"x-owner-id": "nac", "x-api-key": "test-key"}


def test_ingest_search_and_chat(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    ingest_resp = client.post(
        "/memory/ingest",
        json={"path": str(tmp_path / "kb_data"), "recursive": True},
        headers=auth_headers(),
    )
    assert ingest_resp.status_code == 200
    ingest_data = ingest_resp.json()
    assert ingest_data["documents_added"] == 1
    assert ingest_data["chunks_added"] >= 1

    search_resp = client.post(
        "/memory/search",
        json={"query": "纪念日", "top_k": 3},
        headers=auth_headers(),
    )
    assert search_resp.status_code == 200
    hits = search_resp.json()["results"]
    assert hits
    assert "18 号" in hits[0]["content"]

    chat_resp = client.post(
        "/chat",
        json={"message": "我们的纪念日是什么时候？", "top_k": 2},
        headers=auth_headers(),
    )
    assert chat_resp.status_code == 200
    payload = chat_resp.json()
    assert "18 号" in payload["reply"]
    assert payload["context_used"] >= 1
