#!/usr/bin/env bash
set -e
echo "[start] running migrations/init tasks (noop)"
uvicorn src.app.fastapi_app:app --host 0.0.0.0 --port ${PORT:-8000}
