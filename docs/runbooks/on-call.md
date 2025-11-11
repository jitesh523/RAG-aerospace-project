# On-Call Runbook

This runbook helps diagnose and mitigate incidents quickly.

## Quick Checks
- **Readiness**: `GET /ready` should be `true`. If false:
  - FAISS mode: verify `./faiss_store` exists in the pod.
  - Milvus mode: check `MILVUS_HOST/PORT` and `collection` loaded.
  - Non-local env: ensure `API_KEY` or valid JWT config is set.
- **Health**: `GET /health` should be `ok`.
- **Metrics**: `GET /metrics` (requires auth in non-local). Check rate limits, retries, circuit state.

## Common Symptoms → Actions
- **429 Rate limit exceeded**
  - Short term: raise `RATE_LIMIT_PER_MIN` per env if appropriate.
  - Long term: provision Redis and enable distributed limits via `REDIS_URL`.
- **503 LLM unavailable (circuit open)**
  - Check upstream LLM provider status.
  - Lower `LLM_TIMEOUT_SECONDS` or increase `CB_RESET_SECONDS` cautiously.
  - Consider enabling response cache.
- **503 Service not ready**
  - FAISS: run ingestion to create `faiss_store` then restart pod.
  - Milvus: verify DB health and collection load.
  - Non-local: configure `API_KEY` or JWT.
- **Slow /ask p95**
  - Reduce `RETRIEVER_K` or enable `HYBRID_ENABLED` with tuned `HYBRID_ALPHA`.
  - Enable GZip and streaming (`STREAMING_ENABLED=true`).
  - Check vector search histogram and LLM timeout metrics.

## Logs to Inspect
- API pod logs (structured JSON with `request_id`).
- Milvus logs/health endpoint.
- Ingestion worker logs (if async pipeline enabled).

## Rollback/Recovery
- Use canary/blue-green (if enabled) to roll back last release.
- For FAISS corruption: restore from backup or re-ingest.
- For Milvus: load previous snapshot/backup.

## Escalation
- Pager for platform/SRE if outage >15 minutes or repeated circuit opens.
