from fastapi.testclient import TestClient
from types import SimpleNamespace

from src.app import fastapi_app as appmod
from src.app.deps import HybridRetriever
from src.config import Config


def test_rerank_orders_sources(monkeypatch):
    # Enable rerank
    monkeypatch.setenv("RERANK_ENABLED", "true")
    # Reload module-level flags by re-importing config users if needed
    # Patch app config at runtime
    appmod.Config.RERANK_ENABLED = True

    # Prepare app state
    client = TestClient(appmod.app)
    appmod.READY = True

    class FakeChain:
        def invoke(self, q):
            # Two docs: d1 has more occurrences of query terms than d2
            d1 = SimpleNamespace(
                page_content="engine engine throttle control",
                metadata={"source": "doc1.pdf", "page": 1},
            )
            d2 = SimpleNamespace(
                page_content="engine control",
                metadata={"source": "doc2.pdf", "page": 2},
            )
            return {"result": "ok", "source_documents": [d2, d1]}

    appmod.qa_chain = FakeChain()

    r = client.post("/ask", json={"query": "engine throttle"})
    assert r.status_code == 200
    body = r.json()
    # Expect doc1 to come before doc2 due to higher term frequency
    assert body["sources"][0]["source"] == "doc1.pdf"
    assert body["sources"][1]["source"] == "doc2.pdf"


def test_hybrid_blends_vector_and_tf(monkeypatch):
    # Configure alpha so TF has noticeable effect
    monkeypatch.setenv("HYBRID_ENABLED", "true")
    monkeypatch.setenv("HYBRID_ALPHA", "0.5")
    # Update Config used by deps
    Config.HYBRID_ENABLED = True
    Config.HYBRID_ALPHA = 0.5

    class DummyVS:
        def similarity_search_with_score(self, query, k=4):
            # distances: lower is better; we set similar distances so TF can change order
            d_high_tf = SimpleNamespace(page_content="engine engine engine")
            d_low_tf = SimpleNamespace(page_content="engine")
            # return pairs (doc, distance)
            return [
                (d_high_tf, 0.6),
                (d_low_tf, 0.55),
            ]

    retriever = HybridRetriever(DummyVS(), backend_label="faiss")
    docs = retriever.get_relevant_documents("engine engine")
    # With alpha=0.5 and higher TF in first doc, expect it to rank first
    assert docs[0].page_content.count("engine") >= docs[1].page_content.count("engine")
