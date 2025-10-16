from fastapi import FastAPI
from pydantic import BaseModel
from starlette_exporter import PrometheusMiddleware, handle_metrics
from src.app.deps import build_chain
import os
from fastapi import HTTPException, Request
from src.config import Config
from src.index.milvus_index import check_milvus_readiness
import time
import uuid
import logging
import json

app = FastAPI(title="Aerospace RAG API", version="1.0.0")

# Metrics
app.add_middleware(PrometheusMiddleware)
if Config.METRICS_PUBLIC:
    app.add_route("/metrics", handle_metrics)
else:
    @app.get("/metrics")
    def metrics(request: Request):
        api_key = request.headers.get("x-api-key")
        if not (Config.API_KEY and api_key == Config.API_KEY):
            raise HTTPException(status_code=403, detail="Forbidden")
        return handle_metrics(request)

class AskReq(BaseModel):
    query: str

qa_chain = None
READY = False

# Structured logging setup
logger = logging.getLogger("api")
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

# Simple in-memory rate limiter (per client IP or API key)
_rate_state = {}

@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    req_id = str(uuid.uuid4())
    start = time.time()
    client_ip = request.client.host if request.client else ""
    request.state.request_id = req_id
    # Before
    logger.info(json.dumps({
        "request_id": req_id,
        "event": "request_start",
        "method": request.method,
        "path": request.url.path,
        "client_ip": client_ip,
    }))
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = int((time.time() - start) * 1000)
        logger.info(json.dumps({
            "request_id": req_id,
            "event": "request_end",
            "status_code": getattr(locals().get('response', None), 'status_code', None),
            "duration_ms": dur_ms,
        }))

@app.on_event("startup")
def _startup():
    global qa_chain, READY
    try:
        qa_chain = build_chain()
        # Basic readiness check: FAISS store presence + chain built
        faiss_path_exists = os.path.isdir("./faiss_store")
        if Config.RETRIEVER_BACKEND == "milvus":
            milvus = check_milvus_readiness()
            READY = qa_chain is not None and milvus.get("connected") and milvus.get("has_collection") and milvus.get("loaded")
        else:
            READY = qa_chain is not None and faiss_path_exists
    except Exception as e:
        # Do not crash on startup; mark as not ready
        READY = False
        qa_chain = None

@app.post("/ask")
def ask(req: AskReq, request: Request):
    # Optional API key
    if Config.API_KEY:
        api_key = request.headers.get("x-api-key")
        if api_key != Config.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    # Rate limiting
    key = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
    now = int(time.time())
    window = now // 60
    st = _rate_state.get(key)
    if not st or st["window"] != window:
        st = {"window": window, "count": 0}
    st["count"] += 1
    _rate_state[key] = st
    if st["count"] > Config.RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    if len(q) > 4000:
        raise HTTPException(status_code=400, detail="Query too long (max 4000 chars)")
    if not READY or qa_chain is None:
        raise HTTPException(status_code=503, detail="Service not ready. Ingest documents to create ./faiss_store and restart.")
    result = qa_chain.invoke(q)
    sources = [
        {
            "source": d.metadata.get("source", ""),
            "page": d.metadata.get("page", None),
        }
        for d in result["source_documents"]
    ]
    return {"answer": result["result"], "sources": sources}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ready")
def ready():
    if READY:
        return {"ready": True}
    # Not ready yet
    raise HTTPException(status_code=503, detail="Not ready")
