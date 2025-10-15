from fastapi import FastAPI
from pydantic import BaseModel
from starlette_exporter import PrometheusMiddleware, handle_metrics
from src.app.deps import build_chain
import os
from fastapi import HTTPException

app = FastAPI(title="Aerospace RAG API", version="1.0.0")

# Metrics
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

class AskReq(BaseModel):
    query: str

qa_chain = None
READY = False

@app.on_event("startup")
def _startup():
    global qa_chain, READY
    try:
        qa_chain = build_chain()
        # Basic readiness check: FAISS store presence + chain built
        faiss_path_exists = os.path.isdir("./faiss_store")
        READY = qa_chain is not None and faiss_path_exists
    except Exception as e:
        # Do not crash on startup; mark as not ready
        READY = False
        qa_chain = None

@app.post("/ask")
def ask(req: AskReq):
    if not READY or qa_chain is None:
        raise HTTPException(status_code=503, detail="Service not ready. Ingest documents to create ./faiss_store and restart.")
    result = qa_chain.invoke(req.query)
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
