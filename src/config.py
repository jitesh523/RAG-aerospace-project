import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "aero_docs_v1")
    PORT = int(os.getenv("PORT", "8000"))
    ENV = os.getenv("ENV", "local")
    # Retriever backend: "faiss" or "milvus"
    RETRIEVER_BACKEND = os.getenv("RETRIEVER_BACKEND", "faiss").lower()
    # Chunking params
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    # Retrieval params
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
    RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "25"))
