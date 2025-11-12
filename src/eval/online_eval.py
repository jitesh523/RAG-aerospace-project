import time
import json
import threading
from typing import Optional, Dict, Any

from prometheus_client import Counter, Histogram
from src.config import Config
from src.app.deps import build_chain

ONLINE_EVAL_EVENTS = Counter(
    "online_eval_events_total",
    "Online eval events (shadow requests)",
    labelnames=["tenant", "bucket"],
)
ONLINE_EVAL_LATENCY = Histogram(
    "online_eval_latency_seconds",
    "Latency of shadow path",
    labelnames=["bucket"],
)

# Redis is optional; imported lazily to avoid import cycles in app startup
_redis = None

def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis as _r
        if Config.REDIS_URL:
            _redis = _r.Redis.from_url(Config.REDIS_URL, decode_responses=True)
            _redis.ping()
            return _redis
    except Exception:
        _redis = None
    return None


def _record_event(obj: Dict[str, Any]):
    try:
        r = _get_redis()
        if r is not None:
            r.lpush("online:eval", json.dumps(obj))
            r.ltrim("online:eval", 0, 9999)
    except Exception:
        pass


def _shadow_worker(tenant: str, q: str, filters, control_model: str, treatment_model: str, treatment_rerank: bool):
    try:
        # CONTROL: model A, no rerank
        t0 = time.time()
        try:
            c_chain = build_chain(filters=filters, llm_model=control_model, rerank_enabled=False)
            c_res = c_chain.invoke(q)
            ONLINE_EVAL_LATENCY.labels("control").observe(time.time() - t0)
            ONLINE_EVAL_EVENTS.labels(tenant, "control").inc()
        except Exception as e:
            c_res = {"error": str(e)}
        # TREATMENT: configured model, rerank flag
        t1 = time.time()
        try:
            t_chain = build_chain(filters=filters, llm_model=treatment_model, rerank_enabled=treatment_rerank)
            t_res = t_chain.invoke(q)
            ONLINE_EVAL_LATENCY.labels("treatment").observe(time.time() - t1)
            ONLINE_EVAL_EVENTS.labels(tenant, "treatment").inc()
        except Exception as e:
            t_res = {"error": str(e)}
        _record_event({
            "ts": int(time.time()),
            "tenant": tenant,
            "query": q,
            "filters": getattr(filters, "__dict__", {}) or {},
            "control": {
                "model": control_model,
                "rerank": False,
                "result": c_res.get("result") if isinstance(c_res, dict) else None,
                "sources": [getattr(d, "metadata", {}).get("source", "") for d in (c_res.get("source_documents") or [])] if isinstance(c_res, dict) else [],
                "error": c_res.get("error") if isinstance(c_res, dict) else None,
            },
            "treatment": {
                "model": treatment_model,
                "rerank": bool(treatment_rerank),
                "result": t_res.get("result") if isinstance(t_res, dict) else None,
                "sources": [getattr(d, "metadata", {}).get("source", "") for d in (t_res.get("source_documents") or [])] if isinstance(t_res, dict) else [],
                "error": t_res.get("error") if isinstance(t_res, dict) else None,
            },
        })
    except Exception:
        pass


def run_shadow_eval(tenant: str, q: str, filters, treatment_model: str, treatment_rerank: bool, control_model: Optional[str] = None):
    """
    Fire-and-forget shadow evaluation comparing control vs treatment.
    control_model defaults to Config.LLM_MODEL_A; treatment is the live routing.
    """
    try:
        if not Config.ONLINE_EVAL_ENABLED:
            return
        c_model = control_model or Config.LLM_MODEL_A
        t = threading.Thread(
            target=_shadow_worker,
            args=(tenant, q, filters, c_model, treatment_model, bool(treatment_rerank)),
            daemon=True,
        )
        t.start()
    except Exception:
        pass
