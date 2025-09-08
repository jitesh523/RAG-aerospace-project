from fastapi import FastAPI
from pydantic import BaseModel
from starlette_exporter import PrometheusMiddleware, handle_metrics
from src.app.deps import build_chain

app = FastAPI(title="Aerospace RAG API", version="1.0.0")

# Metrics
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

class AskReq(BaseModel):
    query: str

qa_chain = None

@app.on_event("startup")
def _startup():
    global qa_chain
    qa_chain = build_chain()

@app.post("/ask")
def ask(req: AskReq):
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
