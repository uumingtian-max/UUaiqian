# sihan-live

最小可运行版本：提供 “仅本人可用” 的知识库导入、检索和检索增强对话（RAG）后端。

## 功能
- `POST /memory/ingest`：从本地目录（可映射 NAS）导入文本知识。
- `POST /memory/search`：按关键词检索历史知识。
- `POST /chat`：先检索记忆，再返回带人设风格的响应。
- Header 鉴权：`X-Owner-Id` + `X-API-Key`。

## 快速启动
1) 安装依赖

`pip install -r requirements.txt`

2) 启动服务

`uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`

3) 导入默认示例知识

`curl -X POST 'http://127.0.0.1:8000/memory/ingest' -H 'X-Owner-Id: nac' -H 'X-API-Key: local-dev-key' -H 'Content-Type: application/json' -d '{"source_path":"./data/kb_seed"}'`

4) 检索

`curl -X POST 'http://127.0.0.1:8000/memory/search' -H 'X-Owner-Id: nac' -H 'X-API-Key: local-dev-key' -H 'Content-Type: application/json' -d '{"query":"咖啡"}'`

## NAS 使用方式
- 将 NAS 路径挂载到本机后，把路径加入 `config.yaml` 的 `knowledge_base.allowed_roots`。
- 再通过 `/memory/ingest` 传入该挂载路径即可增量导入。
