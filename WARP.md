# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies
make install

# Setup environment (copy and edit with your OPENAI_API_KEY)
cp .env.example .env
```

### Running Services
```bash
# Start development server only (requires external Milvus)
make run

# Start full stack with Milvus database
docker compose up -d

# Start only Milvus for local development
docker compose up -d milvus-standalone
```

### Document Processing
```bash
# Ingest documents (required before querying)
make ingest INPUT_DIR=./path/to/pdfs

# Example: Process aerospace PDFs
mkdir -p data/aerospace_pdfs
# Place PDF files in the directory, then:
make ingest INPUT_DIR=./data/aerospace_pdfs
```

### Testing and Evaluation
```bash
# Run test suite
make test

# Run evaluation framework (requires ingested documents)
make eval

# Custom evaluation with specific questions
python src/eval/evaluate.py --questions ./path/to/questions.json
```

### Development Testing
```bash
# Test API endpoint (after starting services)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain thermal stress analysis in aerospace engines"}'

# Health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Docker Operations
```bash
# Build Docker image
make docker-build

# Run containerized application
make docker-run

# Preview Kubernetes deployment
make helm-dryrun
```

## Architecture Overview

### High-Level System Architecture
This is a production-ready RAG (Retrieval-Augmented Generation) system with dual vector storage architecture:

- **Ingestion Pipeline**: Processes PDF documents → chunks → embeddings → dual storage
- **Dual Vector Storage**: FAISS (in-memory, fast) + Milvus (persistent, scalable)
- **Query Processing**: FastAPI app → FAISS retrieval → LLM generation → response with sources
- **Observability**: Prometheus metrics, health checks, structured logging

### Code Structure and Responsibilities

#### `src/config.py`
Centralized configuration management using environment variables. All service connections and model settings are configured here.

#### `src/app/` - FastAPI Application Layer
- **`fastapi_app.py`**: Main API server with `/ask` endpoint, health checks, and Prometheus metrics
- **`deps.py`**: Dependency injection for LangChain components (LLM, embeddings, retriever chain)

#### `src/ingest/` - Document Processing Pipeline
- **`ingest.py`**: Batch document processor that:
  - Loads PDFs recursively from directories
  - Chunks documents with RecursiveCharacterTextSplitter
  - Creates dual storage: FAISS index (saved to `./faiss_store`) + Milvus collection
  - Processes in configurable batches for memory management

#### `src/index/` - Vector Storage Abstractions
- **`faiss_index.py`**: FAISS vector store builder for fast in-memory retrieval
- **`milvus_index.py`**: Persistent vector storage with schema management, auto-collection creation, and indexing

#### `src/eval/` - Evaluation Framework
- **`evaluate.py`**: Retrieval evaluation using test question sets

### Key Architectural Patterns

#### Dual Vector Storage Strategy
- **FAISS**: Hot storage loaded at API startup for sub-second retrieval
- **Milvus**: Persistent storage for data durability and horizontal scaling
- **Trade-off**: Memory usage vs. query latency vs. persistence

#### LangChain Integration
- Uses **MMR (Maximal Marginal Relevance)** retriever for diverse result selection
- **RetrievalQA** chain with source document tracking
- Configurable retrieval parameters: `k=5` results, `fetch_k=25` for MMR diversity

#### Configuration Management
All runtime configuration via environment variables with sensible defaults. Critical settings:
- `EMBED_MODEL`: Controls embedding dimensions and compatibility
- `MILVUS_COLLECTION`: Enables multiple document collections
- `OPENAI_API_KEY`: Required for embeddings and LLM

### Development Workflow

#### Adding New Document Types
1. Extend `load_pdfs()` in `ingest.py` for new file formats
2. Adjust chunking strategy in `chunk_docs()` if needed
3. Update Milvus schema in `milvus_index.py` for new metadata fields

#### Scaling Considerations
- **Memory**: FAISS index size = `num_chunks * embedding_dim * 4 bytes`
- **Milvus**: Uses IVF_FLAT index with `nlist=1024` (suitable for millions of vectors)
- **API**: Stateless design enables horizontal scaling

#### Local vs Production
- **Local**: Uses `./faiss_store` directory and Docker Compose Milvus
- **Production**: Kubernetes deployment with persistent volumes and external Milvus cluster

### Environment Variables
Key configuration from `.env`:
- `OPENAI_API_KEY`: Required for embeddings and LLM
- `EMBED_MODEL`: Defaults to `text-embedding-3-large` (3072 dimensions)
- `MILVUS_HOST/PORT`: Database connection settings
- `MILVUS_COLLECTION`: Collection name for document vectors

### Dependencies
- **Core**: FastAPI, LangChain, OpenAI, FAISS, Milvus
- **Document Processing**: unstructured, pdfplumber, tiktoken
- **Observability**: starlette-exporter, prometheus-client
- **Testing**: pytest

## Project-Specific Notes

### Document Ingestion Requirements
The system requires documents to be ingested before the API can serve queries. The FAISS index must exist at `./faiss_store` for the API to start successfully.

### Embedding Model Consistency
The embedding model must be consistent between ingestion and query time. Changing `EMBED_MODEL` requires re-ingesting all documents.

### Milvus Collection Management
Collections are auto-created with the schema defined in `milvus_index.py`. The embedding dimension is hardcoded to 3072 for OpenAI's text-embedding-3-large model.

### Memory Requirements
FAISS loads the entire vector index into memory at startup. For large document collections (>100k chunks), consider memory allocation for the API pods.
