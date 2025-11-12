from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Milvus as LC_Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from src.config import Config
from prometheus_client import Histogram, Counter
import redis as _redis_mod
import time


# Prometheus histogram for vector search latency
VECTOR_SEARCH_DURATION = Histogram(
    "vector_search_duration_seconds",
    "Latency of retriever.get_relevant_documents",
    labelnames=["backend"],
)

# Retry counter
VECTOR_SEARCH_RETRIES = Counter(
    "vector_search_retries_total",
    "Total retries performed for vector search calls",
    labelnames=["backend"],
)


class TimedRetriever:
    """Wrapper that times get_relevant_documents and records to Prometheus."""

    def __init__(self, inner, backend_label: str):
        self._inner = inner
        self._backend_label = backend_label

    def get_relevant_documents(self, query):
        start = time.time()
        attempt = 0
        delay = max(0.001, Config.RETRY_BASE_DELAY_MS / 1000.0)
        last_exc = None
        while attempt < max(1, Config.RETRY_MAX_ATTEMPTS):
            try:
                res = self._inner.get_relevant_documents(query)
                elapsed = time.time() - start
                VECTOR_SEARCH_DURATION.labels(self._backend_label).observe(elapsed)
                return res
            except Exception:
                attempt += 1
                if attempt < Config.RETRY_MAX_ATTEMPTS:
                    VECTOR_SEARCH_RETRIES.labels(self._backend_label).inc()
                    time.sleep(delay)
                    delay *= 2
                else:
                    elapsed = time.time() - start
                    VECTOR_SEARCH_DURATION.labels(self._backend_label).observe(elapsed)
                    raise


class HybridRetriever:
    """Blend vector similarity with simple term-frequency scoring.

    Uses vs.similarity_search_with_score(query, k=fetch_k) then re-ranks top-k by
    blended score: alpha*vector_sim + (1-alpha)*tf_norm.
    """

    def __init__(self, vectorstore, backend_label: str, search_kwargs: dict | None = None):
        self._vs = vectorstore
        self._backend_label = backend_label
        self._search_kwargs = search_kwargs or {}
        self.last_scores = None  # list of dicts: {doc_id, blended, v_sim, tf}

    def _tf_score(self, query: str, doc_text: str) -> float:
        q_terms = [t for t in (query or "").lower().split() if t]
        if not q_terms:
            return 0.0
        text = (doc_text or "").lower()
        return float(sum(text.count(t) for t in q_terms))

    def get_relevant_documents(self, query: str):
        start = time.time()
        attempt = 0
        delay = max(0.001, Config.RETRY_BASE_DELAY_MS / 1000.0)
        while attempt < max(1, Config.RETRY_MAX_ATTEMPTS):
            try:
                # fetch_k for broader candidate set
                fetch_k = max(Config.RETRIEVER_FETCH_K, Config.RETRIEVER_K)
                pairs = self._vs.similarity_search_with_score(query, k=fetch_k, **self._search_kwargs)
                # Extract distances/scores (lower is better) and convert to similarity
                if not pairs:
                    elapsed = time.time() - start
                    VECTOR_SEARCH_DURATION.labels(self._backend_label).observe(elapsed)
                    return []
                dists = [p[1] for p in pairs]
                dmin, dmax = min(dists), max(dists)
                sims = []
                for (doc, dist) in pairs:
                    if dmax == dmin:
                        v_sim = 1.0
                    else:
                        # Invert and normalize distance to similarity in [0,1]
                        v_sim = (dmax - dist) / (dmax - dmin)
                    tf = self._tf_score(query, getattr(doc, "page_content", ""))
                    sims.append((doc, v_sim, tf))
                # Normalize tf
                tf_max = max([t for _, _, t in sims]) or 1.0
                alpha = max(0.0, min(1.0, Config.HYBRID_ALPHA))
                ranked = []
                for doc, v_sim, tf in sims:
                    tf_n = (tf / tf_max) if tf_max else 0.0
                    blended = alpha * v_sim + (1 - alpha) * tf_n
                    ranked.append((blended, doc, v_sim, tf))
                ranked.sort(key=lambda x: x[0], reverse=True)
                top = ranked[: Config.RETRIEVER_K]
                top_docs = [d for (blended, d, v_sim, tf) in top]
                # capture explainability
                self.last_scores = [
                    {"doc_id": id(d), "blended": float(blended), "v_sim": float(v_sim), "tf": float(tf)}
                    for (blended, d, v_sim, tf) in top
                ]
                elapsed = time.time() - start
                VECTOR_SEARCH_DURATION.labels(self._backend_label).observe(elapsed)
                return top_docs
            except Exception:
                attempt += 1
                if attempt < Config.RETRY_MAX_ATTEMPTS:
                    VECTOR_SEARCH_RETRIES.labels(self._backend_label).inc()
                    time.sleep(delay)
                    delay *= 2
                else:
                    elapsed = time.time() - start
                    VECTOR_SEARCH_DURATION.labels(self._backend_label).observe(elapsed)
                    raise

def _milvus_expr_from_filters(filters) -> str | None:
    if not filters:
        return None
    parts = []
    try:
        if getattr(filters, "sources", None):
            arr = ",".join([f'"{s}"' for s in filters.sources])
            parts.append(f"source in [{arr}]")
        if getattr(filters, "doc_type", None):
            parts.append(f'doc_type == "{filters.doc_type}"')
        if getattr(filters, "date_from", None):
            parts.append(f'date >= "{filters.date_from}"')
        if getattr(filters, "date_to", None):
            parts.append(f'date <= "{filters.date_to}"')
        if Config.MULTITENANT_ENABLED and getattr(filters, "tenant", None):
            field = Config.TENANT_METADATA_FIELD
            parts.append(f'{field} == "{filters.tenant}"')
    except Exception:
        return None
    return " and ".join(parts) if parts else None

def _faiss_filter_callable_from_filters(filters):
    if not filters:
        return None
    def _pred(md: dict) -> bool:
        try:
            if getattr(filters, "sources", None):
                if md.get("source") not in set(filters.sources):
                    return False
            if getattr(filters, "doc_type", None):
                if str(md.get("doc_type", "")) != str(filters.doc_type):
                    return False
            if getattr(filters, "date_from", None) or getattr(filters, "date_to", None):
                from datetime import datetime
                ds = str(md.get("date", ""))[:10]
                if not ds:
                    return False
                d = datetime.fromisoformat(ds)
                if getattr(filters, "date_from", None):
                    if d < datetime.fromisoformat(filters.date_from):
                        return False
                if getattr(filters, "date_to", None):
                    if d > datetime.fromisoformat(filters.date_to):
                        return False
            if Config.MULTITENANT_ENABLED and getattr(filters, "tenant", None):
                field = Config.TENANT_METADATA_FIELD
                if str(md.get(field, "")) != str(filters.tenant):
                    return False
            return True
        except Exception:
            return False
    return _pred

def build_chain(filters=None, llm_model: str | None = None, rerank_enabled: bool | None = None):
    if Config.MOCK_MODE:
        class _FakeChain:
            def invoke(self, q):
                from types import SimpleNamespace
                doc = SimpleNamespace(metadata={"source": "mock.pdf", "page": 1})
                return {"result": f"mock answer: {q}", "source_documents": [doc]}
        return _FakeChain()
    model = llm_model or "gpt-4o-mini"
    llm = ChatOpenAI(model=model, temperature=0, api_key=Config.OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model=Config.EMBED_MODEL, api_key=Config.OPENAI_API_KEY)
    backend = Config.RETRIEVER_BACKEND
    if backend == "milvus":
        # Build Milvus-backed vector store retriever with fallback to FAISS
        try:
            # Determine read-preferred region
            host = Config.MILVUS_HOST
            port = str(Config.MILVUS_PORT)
            cname = Config.MILVUS_COLLECTION
            if Config.DR_ENABLED and Config.MILVUS_HOST_SECONDARY:
                pref = (Config.DR_READ_PREFERRED or "primary").lower()
                try:
                    if Config.REDIS_URL:
                        r = _redis_mod.Redis.from_url(Config.REDIS_URL, decode_responses=True)
                        v = (r.get("dr:read_preferred") or pref).lower()
                        if v in ("primary","secondary"):
                            pref = v
                except Exception:
                    pass
                if pref == "secondary":
                    host = Config.MILVUS_HOST_SECONDARY
                    port = str(Config.MILVUS_PORT_SECONDARY)
                    cname = Config.MILVUS_COLLECTION_SECONDARY
            vs = LC_Milvus(
                embedding_function=embeddings,
                collection_name=cname,
                connection_args={"host": host, "port": port},
                auto_id=False,
            )
        except Exception:
            # Fallback: FAISS local store
            vs = FAISS.load_local("./faiss_store", embeddings=embeddings)
            backend = "faiss"
    else:
        # Default to FAISS
        vs = FAISS.load_local("./faiss_store", embeddings=embeddings)

    # search kwargs based on filters and backend
    search_kwargs = {"k": Config.RETRIEVER_K, "fetch_k": Config.RETRIEVER_FETCH_K}
    if backend == "milvus":
        expr = _milvus_expr_from_filters(filters)
        if expr:
            search_kwargs["expr"] = expr
        # If partitioning enabled and tenant provided, restrict to that partition
        try:
            if Config.MILVUS_PARTITIONED and getattr(filters, "tenant", None):
                search_kwargs["partition_names"] = [str(filters.tenant)]
        except Exception:
            pass
    else:
        pred = _faiss_filter_callable_from_filters(filters)
        if pred:
            search_kwargs["filter"] = pred

    if Config.HYBRID_ENABLED:
        retriever = HybridRetriever(vs, backend_label=backend, search_kwargs=search_kwargs)
    else:
        base_retriever = vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
        retriever = TimedRetriever(base_retriever, backend_label=backend)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    try:
        setattr(qa, "_retriever_ref", retriever)
    except Exception:
        pass
    return qa
