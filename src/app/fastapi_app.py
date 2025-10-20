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
import jwt
import redis
import requests
from typing import Optional
from opentelemetry import trace as ot_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

app = FastAPI(
    title="Aerospace RAG API",
    version="1.0.0",
    description="API for question answering over aerospace documents using retrieval augmented generation.",
    contact={"name": "Aerospace RAG Team"},
    license_info={"name": "MIT"},
    swagger_ui_parameters={"displayOperationId": True},
)

# Metrics
app.add_middleware(PrometheusMiddleware)
"""Optional Sentry initialization"""
if Config.SENTRY_DSN:
    try:
        sentry_sdk.init(dsn=Config.SENTRY_DSN, traces_sample_rate=0.0)
        app.add_middleware(SentryAsgiMiddleware)
    except Exception:
        pass

"""Optional OpenTelemetry tracing initialization"""
_tracer = None
if Config.OTEL_ENABLED and Config.OTEL_EXPORTER_OTLP_ENDPOINT:
    try:
        resource = Resource.create({"service.name": Config.OTEL_SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=Config.OTEL_EXPORTER_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        ot_trace.set_tracer_provider(provider)
        _tracer = ot_trace.get_tracer(__name__)
    except Exception:
        _tracer = None
if Config.METRICS_PUBLIC:
    app.add_route("/metrics", handle_metrics)
else:
    @app.get("/metrics")
    def metrics(request: Request):
        # Allow if API key matches or JWT is valid
        api_key = request.headers.get("x-api-key")
        authz = request.headers.get("authorization", "")
        jwt_ok = False
        if authz.lower().startswith("bearer "):
            token = authz.split(" ", 1)[1]
            jwt_ok = _verify_jwt(token)
        if not ((Config.API_KEY and api_key == Config.API_KEY) or jwt_ok):
            raise HTTPException(status_code=403, detail="Forbidden")
        return handle_metrics(request)

class AskReq(BaseModel):
    query: str

class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None

class AskResp(BaseModel):
    answer: str
    sources: list[SourceItem]

class HealthResp(BaseModel):
    status: str

class ReadyResp(BaseModel):
    ready: bool

qa_chain = None
READY = False

# Structured logging setup
logger = logging.getLogger("api")
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

# Rate limiter: Redis (if configured) with fallback to in-memory
_rate_state = {}
_redis = None
if Config.REDIS_URL:
    try:
        _redis = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
        # ping to verify connectivity
        _redis.ping()
    except Exception:
        _redis = None
    
# Simple in-memory cache structure: key -> {v: response_json, t: epoch}
_cache = {}

# JWKS cache and verification helpers
_jwks_cache = {"keys": None, "fetched_at": 0}

def _fetch_jwks() -> Optional[dict]:
    if not Config.JWT_JWKS_URL:
        return None
    now = int(time.time())
    if _jwks_cache["keys"] and now - _jwks_cache["fetched_at"] < Config.JWT_JWKS_CACHE_SECONDS:
        return _jwks_cache["keys"]
    try:
        resp = requests.get(Config.JWT_JWKS_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        _jwks_cache["keys"] = data
        _jwks_cache["fetched_at"] = now
        return data
    except Exception:
        return _jwks_cache["keys"]

def _verify_jwt(token: str) -> bool:
    try:
        headers = jwt.get_unverified_header(token)
    except Exception:
        headers = {}
    alg = (Config.JWT_ALG or "HS256").upper()
    try:
        if alg == "RS256" and Config.JWT_JWKS_URL:
            jwks = _fetch_jwks()
            if not jwks:
                return False
            kid = headers.get("kid")
            key = None
            for k in jwks.get("keys", []):
                if k.get("kid") == kid:
                    key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(k))
                    break
            if not key:
                return False
            jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                issuer=Config.JWT_ISSUER,
                audience=Config.JWT_AUDIENCE,
            )
            return True
        elif Config.JWT_SECRET:
            jwt.decode(
                token,
                key=Config.JWT_SECRET,
                algorithms=["HS256"],
                issuer=Config.JWT_ISSUER,
                audience=Config.JWT_AUDIENCE,
            )
            return True
        else:
            return False
    except Exception:
        return False

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

@app.post(
    "/ask",
    response_model=AskResp,
    tags=["Query"],
    summary="Ask a question",
    description="Returns an answer and source citations using the configured retriever and LLM.",
)
def ask(req: AskReq, request: Request):
    span_ctx = None
    if _tracer is not None:
        span_ctx = _tracer.start_as_current_span("ask")
        span_ctx.__enter__()
    # Optional API key
    if Config.API_KEY:
        api_key = request.headers.get("x-api-key")
        if api_key != Config.API_KEY:
            # Allow JWT alternative if configured
            authz = request.headers.get("authorization", "")
            if authz.lower().startswith("bearer ") and _verify_jwt(authz.split(" ", 1)[1]):
                pass
            else:
                raise HTTPException(status_code=401, detail="Invalid or missing API key/JWT")
    # Rate limiting
    key = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
    now = int(time.time())
    window = now // 60
    if _redis is not None:
        rl_key = f"rl:{key}:{window}"
        try:
            newv = _redis.incr(rl_key)
            if newv == 1:
                _redis.expire(rl_key, 65)
            if newv > Config.RATE_LIMIT_PER_MIN:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except Exception:
            # Fallback to in-memory if redis error
            st = _rate_state.get(key)
            if not st or st["window"] != window:
                st = {"window": window, "count": 0}
            st["count"] += 1
            _rate_state[key] = st
            if st["count"] > Config.RATE_LIMIT_PER_MIN:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
    else:
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
    # Cache get (if enabled)
    if Config.CACHE_ENABLED:
        ckey = f"ask:{q}"
        if _redis is not None:
            try:
                cached = _redis.get(ckey)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass
        else:
            ent = _cache.get(ckey)
            if ent and (time.time() - ent["t"]) < Config.CACHE_TTL_SECONDS:
                return ent["v"]

    try:
        if _tracer is not None:
            cur = ot_trace.get_current_span()
            cur.set_attribute("query.length", len(q))
            cur.set_attribute("retriever.backend", os.getenv("RETRIEVER_BACKEND", Config.RETRIEVER_BACKEND))
        result = qa_chain.invoke(q)
    finally:
        if span_ctx is not None:
            span_ctx.__exit__(None, None, None)
    sources = [
        {
            "source": d.metadata.get("source", ""),
            "page": d.metadata.get("page", None),
        }
        for d in result["source_documents"]
    ]
    resp = {"answer": result["result"], "sources": sources}
    # Cache set
    if Config.CACHE_ENABLED:
        ckey = f"ask:{q}"
        if _redis is not None:
            try:
                _redis.setex(ckey, Config.CACHE_TTL_SECONDS, json.dumps(resp))
            except Exception:
                pass
        else:
            _cache[ckey] = {"v": resp, "t": time.time()}
    return resp

@app.get("/health", response_model=HealthResp, tags=["System"], summary="Liveness probe")
def health():
    return {"status": "healthy"}

@app.get("/ready", response_model=ReadyResp, tags=["System"], summary="Readiness probe")
def ready():
    if READY:
        return {"ready": True}
    # Not ready yet
    return {"ready": False}
