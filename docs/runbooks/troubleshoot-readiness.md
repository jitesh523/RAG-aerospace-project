# Troubleshoot Readiness

- **Symptoms**: `/ready` returns 503, pods restarting, liveness flaps.

## Checklist
- **FAISS**: Ensure `./faiss_store` exists if using `RETRIEVER_BACKEND=faiss`.
- **Milvus**: Verify connectivity and collection present/loaded.
  - Port, host, and `MILVUS_COLLECTION` correct.
- **OPENAI_API_KEY**: Required to build embeddings/LLM in non-MOCK mode.
- **Logs**: Check structured JSON logs for `event=request_end` and errors.

## Commands
```bash
kubectl logs deploy/rag-aerospace -n rag-aerospace
kubectl get pods -n rag-aerospace
kubectl describe pod <pod> -n rag-aerospace
```
