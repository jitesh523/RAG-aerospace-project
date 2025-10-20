from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Milvus as LC_Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from src.config import Config
from prometheus_client import Histogram, Counter
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
            except Exception as e:
                last_exc = e
                attempt += 1
                if attempt < Config.RETRY_MAX_ATTEMPTS:
                    VECTOR_SEARCH_RETRIES.labels(self._backend_label).inc()
                    time.sleep(delay)
                    delay *= 2
                else:
                    elapsed = time.time() - start
                    VECTOR_SEARCH_DURATION.labels(self._backend_label).observe(elapsed)
                    raise

def build_chain():
    if Config.MOCK_MODE:
        class _FakeChain:
            def invoke(self, q):
                from types import SimpleNamespace
                doc = SimpleNamespace(metadata={"source": "mock.pdf", "page": 1})
                return {"result": f"mock answer: {q}", "source_documents": [doc]}
        return _FakeChain()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=Config.OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model=Config.EMBED_MODEL, api_key=Config.OPENAI_API_KEY)
    backend = Config.RETRIEVER_BACKEND
    if backend == "milvus":
        # Build Milvus-backed vector store retriever
        vs = LC_Milvus(
            embedding_function=embeddings,
            collection_name=Config.MILVUS_COLLECTION,
            connection_args={"host": Config.MILVUS_HOST, "port": str(Config.MILVUS_PORT)},
            auto_id=False,
        )
    else:
        # Default to FAISS
        vs = FAISS.load_local("./faiss_store", embeddings=embeddings)

    base_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": Config.RETRIEVER_K, "fetch_k": Config.RETRIEVER_FETCH_K},
    )
    timed_retriever = TimedRetriever(base_retriever, backend_label=backend)
    return RetrievalQA.from_chain_type(llm=llm, retriever=timed_retriever, return_source_documents=True)
