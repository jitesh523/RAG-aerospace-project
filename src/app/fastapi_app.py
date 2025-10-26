def require_auth(request: Request) -> None:
    if not Config.API_KEY:
        return
    api_key = request.headers.get("x-api-key")
    if api_key == Config.API_KEY:
        return
    authz = request.headers.get("authorization", "")
    if authz.lower().startswith("bearer ") and _verify_jwt(authz.split(" ", 1)[1]):
        return
    raise HTTPException(status_code=401, detail="Invalid or missing API key/JWT")

def _tenant_from_key(api_key: str | None) -> str:
    if not api_key:
        return "anon"
    try:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, api_key))[:8]
    except Exception:
        return "anon"

def _quota_inc_and_check(key_label: str) -> int:
    day = int(time.time()) // 86400
    if _redis is not None:
        qkey = f"quota:{key_label}:{day}"
        try:
            newv = _redis.incr(qkey)
            if newv == 1:
                _redis.expire(qkey, 90000)
            return int(newv)
        except Exception:
            pass
    st = _rate_state.get(f"q:{key_label}")
    if not st or st.get("day") != day:
        st = {"day": day, "count": 0}
    st["count"] += 1
    _rate_state[f"q:{key_label}"] = st
    return st["count"]
# Prometheus: retries for LLM/ask
ASK_RETRIES = Counter(
    "ask_retries_total",
    "Total retries performed for LLM invocations in /ask",
)
ASK_USAGE_TOTAL = Counter(
    "ask_usage_total",
    "Total /ask requests counted towards quota",
    labelnames=["tenant"],
)
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.responses import StreamingResponse, PlainTextResponse
from prometheus_client import Counter

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

# Basic security headers middleware
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-XSS-Protection", "0")
    if Config.CONTENT_SECURITY_POLICY:
        response.headers.setdefault("Content-Security-Policy", Config.CONTENT_SECURITY_POLICY)
    if Config.SECURITY_HSTS_ENABLED and request.url.scheme == "https":
        response.headers.setdefault(
            "Strict-Transport-Security",
            f"max-age={Config.SECURITY_HSTS_MAX_AGE}; includeSubDomains; preload",
        )
    return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ALLOWED_ORIGINS,
    allow_credentials=Config.CORS_ALLOW_CREDENTIALS,
    allow_methods=Config.CORS_ALLOWED_METHODS,
    allow_headers=Config.CORS_ALLOWED_HEADERS,
)
# GZip compression (optional)
if Config.GZIP_ENABLED:
    app.add_middleware(GZipMiddleware, minimum_size=500)

# Request size limit middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > Config.MAX_REQUEST_BYTES:
                return PlainTextResponse("Request entity too large", status_code=413)
        except Exception:
            pass
    return await call_next(request)
if Config.METRICS_PUBLIC:
    app.add_route("/metrics", handle_metrics)
else:
    @app.get("/metrics")
    def metrics(request: Request):
        require_auth(request)
        return handle_metrics(request)

class AskFilters(BaseModel):
    sources: Optional[list[str]] = None

class AskReq(BaseModel):
    query: str
    filters: Optional[AskFilters] = None

class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None

class AskResp(BaseModel):
    answer: str
    sources: list[SourceItem]

class UsageResp(BaseModel):
    limit: int
    used_today: int

class HealthResp(BaseModel):
    status: str

class ReadyResp(BaseModel):
    ready: bool

qa_chain = None
READY = False
_rerank_model = None

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
            # Retry Milvus readiness with backoff
            attempt = 0
            delay = max(0.001, Config.RETRY_BASE_DELAY_MS / 1000.0)
            while True:
                try:
                    milvus = check_milvus_readiness()
                    break
                except Exception:
                    attempt += 1
                    if attempt >= max(1, Config.RETRY_MAX_ATTEMPTS):
                        raise
                    time.sleep(delay)
                    delay *= 2
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
    # Authorization (optional, enabled when API_KEY is set)
    require_auth(request)
    # Rate limiting
    api_key_hdr = request.headers.get("x-api-key")
    key = api_key_hdr or (request.client.host if request.client else "unknown")
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
    # Quota check
    if Config.QUOTA_ENABLED:
        tenant = _tenant_from_key(api_key_hdr)
        used = _quota_inc_and_check(tenant)
        ASK_USAGE_TOTAL.labels(tenant=tenant).inc()
        if used > Config.QUOTA_DAILY_LIMIT:
            raise HTTPException(status_code=429, detail="Daily quota exceeded")

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

    # LLM invocation with retry/backoff
    attempt = 0
    delay = max(0.001, Config.RETRY_BASE_DELAY_MS / 1000.0)
    result = None
    while True:
        try:
            if _tracer is not None:
                cur = ot_trace.get_current_span()
                cur.set_attribute("query.length", len(q))
                cur.set_attribute("retriever.backend", os.getenv("RETRIEVER_BACKEND", Config.RETRIEVER_BACKEND))
            result = qa_chain.invoke(q)
            break
        except Exception:
            attempt += 1
            if attempt >= max(1, Config.RETRY_MAX_ATTEMPTS):
                if span_ctx is not None:
                    span_ctx.__exit__(None, None, None)
                raise
            ASK_RETRIES.inc()
            time.sleep(delay)
            delay *= 2
    if span_ctx is not None:
        span_ctx.__exit__(None, None, None)
    docs = result["source_documents"]
    # Apply simple metadata filters (include-only by source)
    if req.filters and req.filters.sources:
        try:
            allowed = set(req.filters.sources)
            docs = [d for d in docs if d.metadata.get("source") in allowed]
        except Exception:
            pass
    # Reranking: ML model if configured, else TF-based if enabled
    if Config.RERANK_MODEL:
        global _rerank_model
        try:
            if _rerank_model is None:
                from sentence_transformers import SentenceTransformer
                _rerank_model = SentenceTransformer(Config.RERANK_MODEL)
            # encode query and docs, rank by cosine similarity
            texts = [getattr(d, "page_content", "") for d in docs]
            if texts:
                from numpy import dot
                from numpy.linalg import norm
                import numpy as np
                qv = _rerank_model.encode([q], normalize_embeddings=True)[0]
                dvs = _rerank_model.encode(texts, normalize_embeddings=True)
                scores = [float(dot(qv, dv)) for dv in dvs]
                pairs = list(zip(scores, docs))
                pairs.sort(key=lambda x: x[0], reverse=True)
                docs = [d for _, d in pairs]
        except Exception:
            # fall back silently
            pass
    elif Config.RERANK_ENABLED:
        q_terms = [t for t in q.lower().split() if t]
        def _score(doc):
            text = getattr(doc, "page_content", "").lower()
            return sum(text.count(t) for t in q_terms)
        try:
            docs = sorted(docs, key=_score, reverse=True)
        except Exception:
            pass
    sources = [
        {
            "source": d.metadata.get("source", ""),
            "page": d.metadata.get("page", None),
        }
        for d in docs
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

# SSE streaming endpoint (flag-gated)
@app.get("/ask/stream")
def ask_stream(query: str, request: Request):
    if not Config.STREAMING_ENABLED:
        raise HTTPException(status_code=404, detail="Streaming disabled")
    # Authorization (optional)
    require_auth(request)
    # Basic rate limiting (reuse same as /ask)
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

    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    if len(q) > 4000:
        raise HTTPException(status_code=400, detail="Query too long (max 4000 chars)")

    def _gen():
        # compute once, then stream in chunks
        try:
            result = qa_chain.invoke(q)
            full = result.get("result", "")
            docs = result.get("source_documents", [])
            # first send sources metadata
            srcs = []
            for d in docs:
                srcs.append({"source": d.metadata.get("source", ""), "page": d.metadata.get("page", None)})
            yield f"event: sources\ndata: {json.dumps(srcs)}\n\n"
            # stream answer in small chunks
            chunk = []
            count = 0
            for ch in full.split():
                chunk.append(ch)
                count += len(ch) + 1
                if count >= 128:
                    yield f"data: {' '.join(chunk)}\n\n"
                    chunk = []
                    count = 0
            if chunk:
                yield f"data: {' '.join(chunk)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")

@app.get("/health", response_model=HealthResp, tags=["System"], summary="Liveness probe")
def health():
    return {"status": "healthy"}

@app.get("/ready", response_model=ReadyResp, tags=["System"], summary="Readiness probe")
def ready():
    if READY:
        return {"ready": True}
    # Not ready yet
    return {"ready": False}

@app.get("/usage", response_model=UsageResp, tags=["System"], summary="Usage and quota for caller")
def usage(request: Request):
    require_auth(request)
    api_key_hdr = request.headers.get("x-api-key")
    tenant = _tenant_from_key(api_key_hdr)
    if not Config.QUOTA_ENABLED:
        return {"limit": 0, "used_today": 0}
    # read current without increment
    day = int(time.time()) // 86400
    used = 0
    if _redis is not None:
        try:
            val = _redis.get(f"quota:{tenant}:{day}")
            used = int(val) if val else 0
        except Exception:
            used = 0
    else:
        st = _rate_state.get(f"q:{tenant}") or {}
        used = int(st.get("count", 0)) if st.get("day") == day else 0
    return {"limit": Config.QUOTA_DAILY_LIMIT, "used_today": used}
