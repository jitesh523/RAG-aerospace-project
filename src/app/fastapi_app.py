def _get_cache_ns() -> int:
    if _redis is not None:
        try:
            v = _redis.get("cache:ns")
            if v:
                return int(v)
            _redis.set("cache:ns", 1)
            return 1
        except Exception:
            pass
    return _cache_ns_mem

def _bump_cache_ns() -> int:
    global _cache_ns_mem
    if _redis is not None:
        try:
            return int(_redis.incr("cache:ns"))
        except Exception:
            pass
    _cache_ns_mem += 1
    return _cache_ns_mem
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

CACHE_HITS_TOTAL = Counter(
    "cache_hits_total",
    "/ask cache hits",
    labelnames=["tenant"],
)
CACHE_MISSES_TOTAL = Counter(
    "cache_misses_total",
    "/ask cache misses",
    labelnames=["tenant"],
)
DOCS_SOFT_DELETES_TOTAL = Counter(
    "docs_soft_deletes_total",
    "Total soft-deleted sources",
)
DOCS_SOFT_UNDELETES_TOTAL = Counter(
    "docs_soft_undeletes_total",
    "Total undeleted sources",
)

def require_admin(request: Request) -> None:
    """Require caller to be admin.

    - API key auth: treated as admin
    - JWT auth: must include 'admin' in scope/scopes/scp or in roles claim
    """
    require_auth(request)
    # If API key is used, allow
    api_key = request.headers.get("x-api-key")
    if Config.API_KEY and api_key == Config.API_KEY:
        return
    # If JWT used, verify admin scope
    authz = request.headers.get("authorization", "")
    if authz.lower().startswith("bearer "):
        token = authz.split(" ", 1)[1]
        try:
            # We only read claims; signature verification already enforced by require_auth
            claims = jwt.decode(token, options={"verify_signature": False}, algorithms=["RS256", "HS256"])
        except Exception:
            raise HTTPException(status_code=403, detail="Invalid token claims")
        scopes = claims.get("scope") or claims.get("scopes") or claims.get("scp") or []
        if isinstance(scopes, str):
            scopes = scopes.split()
        roles = claims.get("roles") or []
        if isinstance(roles, str):
            roles = [roles]
        if ("admin" in scopes) or ("admin" in roles):
            return
        raise HTTPException(status_code=403, detail="Admin scope required")
    # Fallback deny
    raise HTTPException(status_code=403, detail="Admin scope required")

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
ASK_TIMEOUTS = Counter(
    "ask_timeouts_total",
    "Total LLM timeouts in /ask",
)
CIRCUIT_OPEN = Counter(
    "circuit_open_total",
    "Number of times a circuit was opened",
    labelnames=["component"],
)
CIRCUIT_STATE = Gauge(
    "circuit_state",
    "Current circuit state (0=closed,1=open)",
    labelnames=["component"],
)
ASK_USAGE_TOTAL = Counter(
    "ask_usage_total",
    "Total /ask requests counted towards quota",
    labelnames=["tenant"],
)
TOKENS_PROMPT_TOTAL = Counter(
    "tokens_prompt_total",
    "Estimated prompt tokens",
    labelnames=["tenant"],
)
TOKENS_COMPLETION_TOTAL = Counter(
    "tokens_completion_total",
    "Estimated completion tokens",
    labelnames=["tenant"],
)
COST_USD_TOTAL = Counter(
    "cost_usd_total",
    "Estimated USD cost",
    labelnames=["tenant"],
)
DENYLIST_SIZE = Gauge(
    "denylist_size",
    "Number of sources currently in denylist",
)
NEG_CACHE_HITS_TOTAL = Counter(
    "cache_negative_hits_total",
    "/ask negative cache hits",
    labelnames=["tenant"],
)
AB_DECISIONS_TOTAL = Counter(
    "ab_decisions_total",
    "AB routing decisions for /ask",
    labelnames=["tenant", "llm", "rerank"],
)
SEM_CACHE_HITS_TOTAL = Counter(
    "semantic_cache_hits_total",
    "/ask semantic cache hits",
    labelnames=["tenant"],
)
SEM_CACHE_MISSES_TOTAL = Counter(
    "semantic_cache_misses_total",
    "/ask semantic cache misses",
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
from datetime import datetime, timedelta
import logging
import json
import jwt
import redis
import requests
import hashlib
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
from prometheus_client import Counter, Gauge
import concurrent.futures

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
    doc_type: Optional[str] = None
    date_from: Optional[str] = None  # ISO date YYYY-MM-DD
    date_to: Optional[str] = None    # ISO date YYYY-MM-DD

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
# cache of QA chains by (model, rerank_enabled)
_qa_cache = {}
# AB routing maps (in-memory fallback)
_ab_llm_mem = {}   # tenant -> 'A'|'B'
_ab_rerank_mem = {}  # tenant -> 'true'|'false'
_rerank_model = None
_cb_failures = 0
_cb_open_until = 0

# Structured logging setup
logger = logging.getLogger("api")
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

# Rate limiter: Redis (if configured) with fallback to in-memory
_rate_state = {}
_deny_sources_mem = set()
_cache_ns_mem = 1
_cache_neg = {}
_source_index_mem = {}
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
# In-memory semantic cache: key -> {v: resp_json, t: epoch}
_sem_cache = {}
# In-memory feedback buffer fallback (tenant -> list of dict)
_feedback_mem = {}

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
        # Build a default chain for readiness checks
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
    # Determine AB routing for this tenant
    tenant_label = _tenant_from_key(api_key_hdr)
    def _get_llm_variant(t: str) -> str:
        # redis hash ab:llm maps tenant->'A'|'B'
        if _redis is not None:
            try:
                v = _redis.hget("ab:llm", t)
                if v in ("A","B"):
                    return v
            except Exception:
                pass
        return _ab_llm_mem.get(t) or "A"
    def _get_rerank_enabled(t: str) -> bool:
        if _redis is not None:
            try:
                v = _redis.hget("ab:rerank", t)
                if v is not None:
                    return str(v).lower() == "true"
            except Exception:
                pass
        if t in _ab_rerank_mem:
            return str(_ab_rerank_mem.get(t)).lower() == "true"
        return bool(Config.HYBRID_ENABLED)
    llm_variant = _get_llm_variant(tenant_label)
    model_name = Config.LLM_MODEL_A if llm_variant == "A" else Config.LLM_MODEL_B
    rerank_enabled = _get_rerank_enabled(tenant_label)
    # Build or reuse chain for this routing
    key_rt = (model_name, bool(rerank_enabled))
    chain = _qa_cache.get(key_rt)
    if chain is None:
        try:
            chain = build_chain(filters=req.filters, llm_model=model_name, rerank_enabled=rerank_enabled)
            _qa_cache[key_rt] = chain
        except Exception:
            chain = None
    if chain is None:
        # Fallback to default
        chain = qa_chain
    try:
        AB_DECISIONS_TOTAL.labels(tenant=tenant_label, llm=llm_variant, rerank=str(bool(rerank_enabled)).lower()).inc()
    except Exception:
        pass
    # Cache get (if enabled) keyed by query+filters+tenant
    ns = _get_cache_ns()
    if Config.CACHE_ENABLED:
        try:
            filt = req.filters.dict() if req.filters else {}
        except Exception:
            filt = {}
        ckey = f"ask:{ns}:{tenant_label}:{q}:{json.dumps(filt, sort_keys=True)}"
        nkey = f"{ckey}:neg"
        if _redis is not None:
            try:
                # negative first
                if Config.NEGATIVE_CACHE_ENABLED:
                    if _redis.get(nkey):
                        NEG_CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                        return {"answer": "", "sources": []}
                cached = _redis.get(ckey)
                if cached:
                    CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                    return json.loads(cached)
                else:
                    CACHE_MISSES_TOTAL.labels(tenant=tenant_label).inc()
            except Exception:
                pass
        else:
            ent = _cache.get(ckey)
            if ent and (time.time() - ent["t"]) < Config.CACHE_TTL_SECONDS:
                CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                return ent["v"]
            else:
                CACHE_MISSES_TOTAL.labels(tenant=tenant_label).inc()
                # check in-memory negative cache
                if Config.NEGATIVE_CACHE_ENABLED:
                    nent = _cache_neg.get(nkey)
                    if nent and (time.time() - nent) < Config.NEGATIVE_CACHE_TTL_SECONDS:
                        NEG_CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                        return {"answer": "", "sources": []}

    # Semantic cache (feature-flagged)
    if Config.SEMANTIC_CACHE_ENABLED:
        simhash_key = _simhash(q, req.filters)
        skey = f"sem:{ns}:{tenant_label}:{simhash_key}"
        if _redis is not None:
            try:
                cached = _redis.get(skey)
                if cached:
                    SEM_CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                    return json.loads(cached)
                else:
                    SEM_CACHE_MISSES_TOTAL.labels(tenant=tenant_label).inc()
            except Exception:
                pass
        else:
            ent = _sem_cache.get(skey)
            if ent and (time.time() - ent["t"]) < Config.SEMANTIC_CACHE_TTL_SECONDS:
                SEM_CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                return ent["v"]
            else:
                SEM_CACHE_MISSES_TOTAL.labels(tenant=tenant_label).inc()

    # Circuit breaker: short-circuit if open
    now_ts = int(time.time())
    if _cb_open_until and now_ts < _cb_open_until:
        CIRCUIT_STATE.labels(component="llm").set(1)
        raise HTTPException(status_code=503, detail="LLM unavailable (circuit open)")

    # LLM invocation with retry/backoff + timeout
    attempt = 0
    delay = max(0.001, Config.RETRY_BASE_DELAY_MS / 1000.0)
    result = None
    while True:
        try:
            if _tracer is not None:
                cur = ot_trace.get_current_span()
                cur.set_attribute("query.length", len(q))
                cur.set_attribute("retriever.backend", os.getenv("RETRIEVER_BACKEND", Config.RETRIEVER_BACKEND))
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(chain.invoke, q)
                try:
                    result = fut.result(timeout=max(1, Config.LLM_TIMEOUT_SECONDS))
                except concurrent.futures.TimeoutError:
                    ASK_TIMEOUTS.inc()
                    # record failure for CB
                    global _cb_failures, _cb_open_until
                    _cb_failures += 1
                    if _cb_failures >= Config.CB_FAIL_THRESHOLD:
                        _cb_open_until = int(time.time()) + max(1, Config.CB_RESET_SECONDS)
                        CIRCUIT_OPEN.labels(component="llm").inc()
                        CIRCUIT_STATE.labels(component="llm").set(1)
                    raise HTTPException(status_code=504, detail="LLM timeout")
            break
        except HTTPException:
            # propagate HTTP errors like timeout
            raise
        except Exception:
            attempt += 1
            if attempt >= max(1, Config.RETRY_MAX_ATTEMPTS):
                if span_ctx is not None:
                    span_ctx.__exit__(None, None, None)
                # record failure for CB
                _cb_failures += 1
                if _cb_failures >= Config.CB_FAIL_THRESHOLD:
                    _cb_open_until = int(time.time()) + max(1, Config.CB_RESET_SECONDS)
                    CIRCUIT_OPEN.labels(component="llm").inc()
                    CIRCUIT_STATE.labels(component="llm").set(1)
                raise
            ASK_RETRIES.inc()
            time.sleep(delay)
            delay *= 2
    # CB success path: reset failures when success
    _cb_failures = 0
    _cb_open_until = 0
    CIRCUIT_STATE.labels(component="llm").set(0)
    if span_ctx is not None:
        span_ctx.__exit__(None, None, None)
    docs = result["source_documents"]
    # Exclude soft-deleted sources (denylist)
    try:
        deny = set()
        if _redis is not None:
            try:
                members = _redis.smembers("deny:sources")
                deny = set([m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m) for m in members])
            except Exception:
                deny = set(_deny_sources_mem)
        else:
            deny = set(_deny_sources_mem)
        if deny:
            docs = [d for d in docs if d.metadata.get("source") not in deny]
        try:
            DENYLIST_SIZE.set(len(deny))
        except Exception:
            pass
    except Exception:
        pass
    # Apply simple metadata filters
    if req.filters:
        try:
            if req.filters.sources:
                allowed = set(req.filters.sources)
                docs = [d for d in docs if d.metadata.get("source") in allowed]
            if req.filters.doc_type:
                dt = req.filters.doc_type
                docs = [d for d in docs if str(d.metadata.get("doc_type", "")) == dt]
            if req.filters.date_from or req.filters.date_to:
                from datetime import datetime
                df = datetime.fromisoformat(req.filters.date_from) if req.filters.date_from else None
                dt_ = datetime.fromisoformat(req.filters.date_to) if req.filters.date_to else None
                def _in_range(meta_date: str) -> bool:
                    try:
                        if not meta_date:
                            return False
                        d = datetime.fromisoformat(str(meta_date)[:10])
                        if df and d < df:
                            return False
                        if dt_ and d > dt_:
                            return False
                        return True
                    except Exception:
                        return False
                docs = [d for d in docs if _in_range(str(d.metadata.get("date", "")))]
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
    # Cost and token estimation metrics
    if Config.COST_ENABLED:
        try:
            tenant = _tenant_from_key(api_key_hdr)
            # very rough token estimate: ~4 chars per token
            p_tokens = max(1, int(len(q) / 4))
            c_tokens = max(1, int(len(resp.get("answer", "")) / 4))
            TOKENS_PROMPT_TOTAL.labels(tenant=tenant).inc(p_tokens)
            TOKENS_COMPLETION_TOTAL.labels(tenant=tenant).inc(c_tokens)
            cost = (p_tokens / 1000.0) * Config.COST_PER_1K_PROMPT_TOKENS + (c_tokens / 1000.0) * Config.COST_PER_1K_COMPLETION_TOKENS
            COST_USD_TOTAL.labels(tenant=tenant).inc(cost)
        except Exception:
            pass
    # Cache set
    if Config.CACHE_ENABLED:
        try:
            filt = req.filters.dict() if req.filters else {}
        except Exception:
            filt = {}
        ns = _get_cache_ns()
        ckey = f"ask:{ns}:{tenant_label}:{q}:{json.dumps(filt, sort_keys=True)}"
        nkey = f"{ckey}:neg"
        if _redis is not None:
            try:
                # negative caching for empty answers
                if Config.NEGATIVE_CACHE_ENABLED and ((not resp.get("answer")) or (not resp.get("sources"))):
                    _redis.setex(nkey, Config.NEGATIVE_CACHE_TTL_SECONDS, "1")
                else:
                    _redis.setex(ckey, Config.CACHE_TTL_SECONDS, json.dumps(resp))
            except Exception:
                pass
        else:
            # negative caching in memory
            if Config.NEGATIVE_CACHE_ENABLED and ((not resp.get("answer")) or (not resp.get("sources"))):
                _cache_neg[nkey] = time.time()
            else:
                _cache[ckey] = {"v": resp, "t": time.time()}
    # Semantic cache set
    if Config.SEMANTIC_CACHE_ENABLED:
        simhash_key = _simhash(q, req.filters)
        skey = f"sem:{ns}:{tenant_label}:{simhash_key}"
        if _redis is not None:
            try:
                _redis.setex(skey, Config.SEMANTIC_CACHE_TTL_SECONDS, json.dumps(resp))
            except Exception:
                pass
        else:
            _sem_cache[skey] = {"v": resp, "t": time.time()}
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
            # respect circuit breaker and timeout logic by reusing same flow as /ask
            # (no retries for stream; we apply single timeout)
            if _cb_open_until and int(time.time()) < _cb_open_until:
                yield f"event: error\ndata: {json.dumps({'error': 'LLM unavailable (circuit open)'})}\n\n"
                return
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(qa_chain.invoke, q)
                try:
                    result = fut.result(timeout=max(1, Config.LLM_TIMEOUT_SECONDS))
                except concurrent.futures.TimeoutError:
                    ASK_TIMEOUTS.inc()
                    yield f"event: error\ndata: {json.dumps({'error': 'LLM timeout'})}\n\n"
                    return
            full = result.get("result", "")
            docs = result.get("source_documents", [])
            # Exclude soft-deleted sources (denylist)
            try:
                deny = set()
                if _redis is not None:
                    try:
                        members = _redis.smembers("deny:sources")
                        deny = set([m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m) for m in members])
                    except Exception:
                        deny = set(_deny_sources_mem)
                else:
                    deny = set(_deny_sources_mem)
                if deny:
                    docs = [d for d in docs if d.metadata.get("source") not in deny]
                try:
                    DENYLIST_SIZE.set(len(deny))
                except Exception:
                    pass
            except Exception:
                pass
            # first send sources metadata
            srcs = []
            for d in docs:
                srcs.append({"source": d.metadata.get("source", ""), "page": d.metadata.get("page", None)})
            yield f"event: sources\ndata: {json.dumps(srcs)}\n\n"
            # Update source index (for retention) with any observed date
            try:
                for d in docs:
                    src = str(d.metadata.get("source", "")).strip()
                    ds = str(d.metadata.get("date", ""))[:10]
                    if not src or not ds:
                        continue
                    try:
                        dt = datetime.fromisoformat(ds)
                        score = int(dt.timestamp())
                        if _redis is not None:
                            try:
                                _redis.zadd("sources:index", {src: score})
                            except Exception:
                                _source_index_mem[src] = score
                        else:
                            _source_index_mem[src] = score
                    except Exception:
                        continue
            except Exception:
                pass
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
            # record estimated tokens/cost at end
            if Config.COST_ENABLED:
                try:
                    tenant = _tenant_from_key(request.headers.get("x-api-key"))
                    p_tokens = max(1, int(len(q) / 4))
                    c_tokens = max(1, int(len(full) / 4))
                    TOKENS_PROMPT_TOTAL.labels(tenant=tenant).inc(p_tokens)
                    TOKENS_COMPLETION_TOTAL.labels(tenant=tenant).inc(c_tokens)
                    cost = (p_tokens / 1000.0) * Config.COST_PER_1K_PROMPT_TOKENS + (c_tokens / 1000.0) * Config.COST_PER_1K_COMPLETION_TOKENS
                    COST_USD_TOTAL.labels(tenant=tenant).inc(cost)
                except Exception:
                    pass
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")

class AdminSoftDeleteReq(BaseModel):
    source: str

class AdminABLLMReq(BaseModel):
    tenant: str
    variant: str  # 'A' or 'B'

class AdminABRerankReq(BaseModel):
    tenant: str
    enabled: bool

@app.post("/admin/docs/delete", tags=["Admin"], summary="Soft-delete a source (denylist)")
def admin_delete(req: AdminSoftDeleteReq, request: Request):
    require_admin(request)
    src = (req.source or "").strip()
    if not src:
        raise HTTPException(status_code=400, detail="source is required")
    ok = True
    if _redis is not None:
        try:
            _redis.sadd("deny:sources", src)
        except Exception:
            ok = False
    if not ok or _redis is None:
        _deny_sources_mem.add(src)
    DOCS_SOFT_DELETES_TOTAL.inc()
    _bump_cache_ns()
    try:
        # update gauge after mutation
        cur = set()
        if _redis is not None:
            try:
                cur = set(_redis.smembers("deny:sources"))
            except Exception:
                cur = set(_deny_sources_mem)
        else:
            cur = set(_deny_sources_mem)
        DENYLIST_SIZE.set(len(cur))
    except Exception:
        pass
    return {"status": "ok", "source": src}

@app.post("/admin/docs/undelete", tags=["Admin"], summary="Remove a source from denylist")
def admin_undelete(req: AdminSoftDeleteReq, request: Request):
    require_admin(request)
    src = (req.source or "").strip()
    if not src:
        raise HTTPException(status_code=400, detail="source is required")
    ok = True
    if _redis is not None:
        try:
            _redis.srem("deny:sources", src)
        except Exception:
            ok = False
    if not ok or _redis is None:
        try:
            _deny_sources_mem.discard(src)
        except Exception:
            pass
    DOCS_SOFT_UNDELETES_TOTAL.inc()
    _bump_cache_ns()
    try:
        cur = set()
        if _redis is not None:
            try:
                cur = set(_redis.smembers("deny:sources"))
            except Exception:
                cur = set(_deny_sources_mem)
        else:
            cur = set(_deny_sources_mem)
        DENYLIST_SIZE.set(len(cur))
    except Exception:
        pass
    return {"status": "ok", "source": src}

@app.post("/admin/ab/llm", tags=["Admin"], summary="Set LLM variant (A/B) for a tenant")
def admin_set_llm(req: AdminABLLMReq, request: Request):
    require_admin(request)
    t = (req.tenant or "").strip()
    v = (req.variant or "").strip().upper()
    if v not in ("A","B") or not t:
        raise HTTPException(status_code=400, detail="variant must be 'A' or 'B' and tenant required")
    try:
        if _redis is not None:
            try:
                _redis.hset("ab:llm", t, v)
            except Exception:
                _ab_llm_mem[t] = v
        else:
            _ab_llm_mem[t] = v
        # Clear cached chains for this tenant's variants is not tenant-specific; clear all caches to be safe
        _qa_cache.clear()
        return {"status": "ok", "tenant": t, "variant": v}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/ab/rerank", tags=["Admin"], summary="Enable/disable reranker for a tenant")
def admin_set_rerank(req: AdminABRerankReq, request: Request):
    require_admin(request)
    t = (req.tenant or "").strip()
    val = bool(req.enabled)
    if not t:
        raise HTTPException(status_code=400, detail="tenant required")
    try:
        if _redis is not None:
            try:
                _redis.hset("ab:rerank", t, "true" if val else "false")
            except Exception:
                _ab_rerank_mem[t] = "true" if val else "false"
        else:
            _ab_rerank_mem[t] = "true" if val else "false"
        _qa_cache.clear()
        return {"status": "ok", "tenant": t, "enabled": val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retention sweep metrics
DOCS_RETENTION_SWEEPS_TOTAL = Counter(
    "docs_retention_sweeps_total",
    "Total retention sweeps executed",
)
DOCS_RETENTION_SOFT_DELETES_TOTAL = Counter(
    "docs_retention_soft_deletes_total",
    "Total sources soft-deleted by retention sweeps",
)

@app.post("/admin/retention/sweep", tags=["Admin"], summary="Apply retention window to soft-delete old sources")
def retention_sweep(request: Request):
    require_admin(request)
    days = max(0, int(Config.DOC_RETENTION_DAYS))
    if days <= 0:
        return {"status": "skipped", "reason": "DOC_RETENTION_DAYS=0"}
    cutoff_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    deleted = 0
    try:
        if _redis is not None:
            try:
                olds = _redis.zrangebyscore("sources:index", "-inf", cutoff_ts)
            except Exception:
                olds = []
        else:
            olds = [s for s, sc in _source_index_mem.items() if sc <= cutoff_ts]
    except Exception:
        olds = []
    for src in olds:
        try:
            if _redis is not None:
                try:
                    _redis.sadd("deny:sources", src)
                except Exception:
                    _deny_sources_mem.add(src)
            else:
                _deny_sources_mem.add(src)
            deleted += 1
        except Exception:
            continue
    DOCS_RETENTION_SWEEPS_TOTAL.inc()
    if deleted:
        DOCS_RETENTION_SOFT_DELETES_TOTAL.inc(deleted)
        _bump_cache_ns()
    try:
        cur = set()
        if _redis is not None:
            try:
                cur = set(_redis.smembers("deny:sources"))
            except Exception:
                cur = set(_deny_sources_mem)
        else:
            cur = set(_deny_sources_mem)
        DENYLIST_SIZE.set(len(cur))
    except Exception:
        pass
    return {"status": "ok", "deleted": deleted, "cutoff_ts": cutoff_ts}

@app.get("/system", tags=["System"], summary="System state")
def system_state():
    now_ts = int(time.time())
    open_state = 1 if (_cb_open_until and now_ts < _cb_open_until) else 0
    try:
        CIRCUIT_STATE.labels(component="llm").set(open_state)
    except Exception:
        pass
    # also set denylist gauge on system read
    try:
        cur = set()
        if _redis is not None:
            try:
                cur = set(_redis.smembers("deny:sources"))
            except Exception:
                cur = set(_deny_sources_mem)
        else:
            cur = set(_deny_sources_mem)
        DENYLIST_SIZE.set(len(cur))
    except Exception:
        pass
    return {
        "circuit": {
            "component": "llm",
            "open": bool(open_state),
            "open_until": _cb_open_until,
            "failures": _cb_failures,
            "threshold": Config.CB_FAIL_THRESHOLD,
            "reset_seconds": Config.CB_RESET_SECONDS,
        },
        "ready": READY,
    }

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
