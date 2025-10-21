# RAG Aerospace

A production-ready Retrieval-Augmented Generation (RAG) system for aerospace documentation, built with LangChain, FAISS, Milvus, and FastAPI. Designed for deployment on Azure Kubernetes Service (AKS) with comprehensive monitoring and CI/CD pipeline.

## 🚀 Features

- **Dual Vector Storage**: FAISS for fast in-memory retrieval + Milvus for persistent storage
- **Production Architecture**: FastAPI with Prometheus metrics, health checks, and observability
- **Kubernetes Ready**: Helm charts and manifests for AKS deployment
- **CI/CD Pipeline**: GitHub Actions with automated testing and Docker image builds
- **Batch Processing**: Scalable document ingestion with configurable batch sizes
- **Evaluation Framework**: Built-in tools for RAG performance assessment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Docs      │────│  Ingestion      │────│  Vector Store   │
│   (Aerospace)   │    │  Pipeline       │    │  (FAISS+Milvus) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│   User Query    │────│  FastAPI        │───────────┘
│                 │    │  Application    │
└─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  LLM Response   │
                       │  + Sources      │
                       └─────────────────┘
```

## 🚦 Quick Start (Local Development)

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- OpenAI API Key

### 1. Setup Environment

```bash
# Clone and setup
git clone <repo-url>
cd rag-aerospace

# Create environment file
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and other settings

# Install dependencies
make install
```

### 2. Start Services

```bash
# Start Milvus database
docker compose up -d milvus-standalone

# Or start everything including the API
docker compose up -d
```

### 3. Ingest Documents

```bash
# Create data directory and add your PDF files
mkdir -p data/aerospace_pdfs
# Place your aerospace PDF documents in this directory

# Run ingestion (processes 50k+ docs efficiently)
make ingest INPUT_DIR=./data/aerospace_pdfs
```

### 4. Query the System

```bash
# Start the API (if not already running via docker-compose)
make run

# Test a query
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain thermal stress analysis in aerospace engines"}'
```

## 🛠️ Development

### Project Structure

```
rag-aerospace/
├─ src/
│  ├─ app/           # FastAPI application
│  ├─ index/         # Vector storage (FAISS + Milvus)
│  ├─ ingest/        # Document processing pipeline
│  ├─ eval/          # Evaluation and testing
│  └─ config.py      # Configuration management
├─ k8s/              # Kubernetes deployment files
├─ docker/           # Docker configuration
└─ tests/            # Test suite
```

### Available Commands

```bash
make install         # Install dependencies
make run            # Start development server
make test           # Run tests
make ingest         # Process documents (requires INPUT_DIR)
make eval           # Run evaluation suite
make docker-build   # Build Docker image
make helm-dryrun    # Preview Kubernetes deployment
```

## ☸️ Kubernetes Deployment (AKS)

### Prerequisites
- Azure CLI configured
- kubectl configured for your AKS cluster
- Helm 3.x installed
- Container registry access (GitHub Container Registry)

### 1. Build and Push Image

```bash
# Build and push to GHCR (handled by GitHub Actions)
docker build -t ghcr.io/YOUR_USERNAME/rag-aerospace:latest .
docker push ghcr.io/YOUR_USERNAME/rag-aerospace:latest
```

### 2. Deploy with Helm

```bash
# Create namespace
kubectl create namespace rag-aerospace

# Create secrets
kubectl create secret generic openai-secret \
  --from-literal=api_key=YOUR_OPENAI_API_KEY \
  -n rag-aerospace

# Deploy Milvus (if not using external instance)
helm repo add milvus https://milvus-io.github.io/milvus-helm/
helm install milvus milvus/milvus -n milvus --create-namespace

# Deploy the application
helm install rag-aerospace ./k8s/helm -n rag-aerospace
```

### 3. Probes, Access the Application

# Port forward for testing
kubectl port-forward svc/rag-aerospace 8080:80 -n rag-aerospace

# Or configure ingress (see k8s/helm/templates/ingress.yaml)
```

## 📊 Monitoring & Observability

### Metrics
The application exposes Prometheus metrics at `/metrics`:
- Vector search latency histogram: `vector_search_duration_seconds`
- Ingestion counter: `ingest_documents_total`
- Starlette request metrics: `starlette_requests_total`, etc.
- Retry counters:
  - `vector_search_retries_total{backend}`
  - `ingest_retries_total{stage="embed|insert"}`

Grafana resources:
- Dashboard: `docs/grafana/dashboard-api-overview.json`
- Alerts: `docs/grafana/alerts.json`

Runbooks:
- `docs/runbooks/scaling.md`
- `docs/runbooks/secrets.md`
- `docs/runbooks/troubleshoot-readiness.md`
- `docs/runbooks/rate-limiting.md`

### Health Checks
- Readiness probe: `/ready` (verifies FAISS presence or Milvus connectivity and collection load)
- Liveness probe: `/health`

### OpenTelemetry Configuration
- `OTEL_ENABLED`: enable OpenTelemetry tracing when `true`.
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP endpoint (e.g., `http://otel-collector:4318/v1/traces`).
- `OTEL_SERVICE_NAME`: service name for tracing; defaults to `rag-aerospace`.

### Sentry Configuration
- `SENTRY_DSN`: if set, enables Sentry ASGI middleware for error reporting.
Create dashboards for:
- API response times
- Vector search latency
- Document processing throughput
- Error rates by endpoint

## 🔒 Security Considerations

### Current Implementation
- Environment-based secrets management
- Optional API key for `/ask` (set `API_KEY`) and optional metrics protection (`METRICS_PUBLIC=false`)
- Simple in-memory rate limiting via `RATE_LIMIT_PER_MIN`

### Production Recommendations
- **API Gateway**: Use Azure API Management with OAuth2/Azure AD
- **Network Security**: Deploy in private subnet with proper NSG rules
- **Secrets**: Use Azure Key Vault integration
- **RBAC**: Configure Kubernetes RBAC for pod security

```yaml
# Example Azure AD integration (not implemented)
apiVersion: v1
kind: Secret
metadata:
  name: azure-ad-secret
data:
  client_id: <base64-encoded-client-id>
  client_secret: <base64-encoded-client-secret>
```

## 🎯 Evaluation & Accuracy

### Built-in Evaluation
```bash
# Run evaluation suite
make eval

# Custom evaluation questions
echo '[{"question": "What is wing loading?"}]' > custom_eval.json
python src/eval/evaluate.py --questions custom_eval.json
```

### Accuracy Improvements
1. **MMR Retriever**: Already configured for diverse results
2. **Source Grounding**: Returns source documents with responses
3. **Confidence Scoring**: Consider implementing confidence thresholds
4. **Re-ranking**: Add bge-reranker-base for improved precision

### Hallucination Reduction
- Source document attribution in all responses
- Configurable confidence thresholds
- "Need more context" responses for low-confidence queries

## 📈 Scaling Considerations

### Horizontal Scaling
```yaml
# Update values.yaml for auto-scaling
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

### Performance Optimization
- **FAISS to Persistent**: Move FAISS to persistent volume for faster restarts
- **Connection Pooling**: Configure Milvus connection pooling
- **Caching**: Add Redis for frequently accessed embeddings
- **Load Balancing**: Use NGINX ingress with session affinity

### Storage Scaling
- **Milvus Cluster**: Deploy multi-node Milvus for > 100M vectors
- **Azure Blob Storage**: Store raw documents in blob storage
- **Database Partitioning**: Partition by document type/date

## 🔧 Configuration

### Environment Variables
See `.env.example` for all configuration options.

Key runtime settings:
- `RETRIEVER_BACKEND`: `faiss` (default) or `milvus`.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: controls PDF chunking in ingestion.
- `RETRIEVER_K`, `RETRIEVER_FETCH_K`: retrieval parameters.
- `API_KEY`: if set, `/ask` requires header `x-api-key`. Also used to protect `/metrics` when `METRICS_PUBLIC=false`.
- `METRICS_PUBLIC`: `true` to expose `/metrics` openly; `false` to require `API_KEY`.
- `RATE_LIMIT_PER_MIN`: requests per minute per API key/IP for `/ask`.
- `MOCK_MODE`: when `true`, API uses a fake chain for E2E/CI without external dependencies.
- `CACHE_ENABLED`, `CACHE_TTL_SECONDS`: enable Redis/in-memory response cache for `/ask`.
- `JWT_ALG`: `HS256` (default) or `RS256` for JWKS.
- `JWT_JWKS_URL`, `JWT_JWKS_CACHE_SECONDS`: configure JWKS fetch/caching for RS256.
- `REDIS_URL`: if set, enables Redis-backed rate limiting and cache.
- `OTEL_ENABLED`: enable OpenTelemetry tracing when `true`.
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP endpoint (e.g., `http://otel-collector:4318/v1/traces`).
- `OTEL_SERVICE_NAME`: service name for tracing; defaults to `rag-aerospace`.
- `SENTRY_DSN`: if set, enables Sentry ASGI middleware for error reporting.
- `SECURITY_HSTS_ENABLED`, `SECURITY_HSTS_MAX_AGE`: control HSTS header behavior.
- `RETRY_MAX_ATTEMPTS`, `RETRY_BASE_DELAY_MS`: exponential backoff settings for vector search and ingestion.
- `CORS_ALLOWED_ORIGINS`, `CORS_ALLOW_CREDENTIALS`, `CORS_ALLOWED_METHODS`, `CORS_ALLOWED_HEADERS`: CORS configuration.
- `RERANK_ENABLED`: when `true`, applies a lightweight query-term based rerank to sources.

### Key Settings
- `EMBED_MODEL`: Choose embedding model (OpenAI vs HuggingFace)
- `MILVUS_COLLECTION`: Collection name for document vectors
- `PORT`: API server port

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run the test suite: `make test`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the [troubleshooting guide](docs/troubleshooting.md)
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

**Built for production aerospace RAG workloads** 🚀
