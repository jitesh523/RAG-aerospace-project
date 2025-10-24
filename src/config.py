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
    # Auth & rate limiting
    API_KEY = os.getenv("API_KEY")  # if set, required for /ask
    METRICS_PUBLIC = os.getenv("METRICS_PUBLIC", "true").lower() == "true"
    RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
    REDIS_URL = os.getenv("REDIS_URL")
    # JWT (HMAC) support
    JWT_SECRET = os.getenv("JWT_SECRET")
    JWT_ISSUER = os.getenv("JWT_ISSUER")
    JWT_AUDIENCE = os.getenv("JWT_AUDIENCE")
    # JWT hardening (RS256 via JWKS)
    JWT_ALG = os.getenv("JWT_ALG", "HS256").upper()
    JWT_JWKS_URL = os.getenv("JWT_JWKS_URL")
    JWT_JWKS_CACHE_SECONDS = int(os.getenv("JWT_JWKS_CACHE_SECONDS", "3600"))
    # Pushgateway for ingestion metrics (optional)
    PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL")
    # Mock mode for e2e and CI (bypass LLM/retriever)
    MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
    # Response cache
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() == "true"
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    # OpenTelemetry tracing (optional)
    OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "rag-aerospace")
    # Sentry error reporting (optional)
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    # Security headers
    SECURITY_HSTS_ENABLED = os.getenv("SECURITY_HSTS_ENABLED", "true").lower() == "true"
    SECURITY_HSTS_MAX_AGE = int(os.getenv("SECURITY_HSTS_MAX_AGE", "31536000"))
    # Content Security Policy (optional)
    CONTENT_SECURITY_POLICY = os.getenv("CONTENT_SECURITY_POLICY", "")
    # Retry/backoff
    RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    RETRY_BASE_DELAY_MS = int(os.getenv("RETRY_BASE_DELAY_MS", "100"))
    # CORS
    CORS_ALLOWED_ORIGINS = [s.strip() for s in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")]
    CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
    CORS_ALLOWED_METHODS = [s.strip() for s in os.getenv("CORS_ALLOWED_METHODS", "GET,POST,OPTIONS").split(",")]
    CORS_ALLOWED_HEADERS = [s.strip() for s in os.getenv("CORS_ALLOWED_HEADERS", "*").split(",")]
    # Reranking
    RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
    # Hybrid search (vector + term scoring)
    HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "false").lower() == "true"
    HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))
    # Embedding batching controls
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
