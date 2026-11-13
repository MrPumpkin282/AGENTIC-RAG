# Agentic RAG: Self-Corrective AI Assistant

An intelligent **Agentic RAG** system built with **LangGraph**, **NVIDIA NIM**, **Qdrant Cloud**, and **Tavily Web Search**.

Unlike traditional RAG systems that blindly trust vector search, this agent **fact-checks retrieved evidence**, automatically **falls back to web search** when internal documents are insufficient, and **rewrites queries** to find accurate answers with exact page citations.

---

## Architecture Flow

![Architecture Workflow Diagram](docs/workflow_diagram.png)

The system operates across two coordinated pipelines:

### 1. Document Ingestion & Index Synchronization Pipeline
* **Upload Verification & Security**: Enforces regex path sanitization against directory traversal, streaming file-size limits (max 25 MB), and MIME / magic-byte verification (`%PDF` for PDF, `PK\x03\x04` for DOCX, binary null-byte check for TXT/MD).
* **Hierarchical Chunking**:
  * **Parent chunks (2,000 chars)**: Preserved for generation to provide broad context and prevent hallucinations.
  * **Child chunks (400 chars)**: Prefixed with metadata injection headers (`[File: ... | Page: ...]`) for pinpoint vector search.
* **Vector Storage**: Generates dense embeddings using NVIDIA NIM (`nvidia/nemotron-3.5-embed-1b`) and stores child vectors and parent payloads in Qdrant Cloud.
* **Hybrid Lexical Sync**: Automatically invalidates the in-memory BM25Okapi index to trigger an immediate rebuild on the next query.
* **Automated Cache Invalidation**: Automatically purges the Qdrant semantic cache namespace upon ingestion to eliminate stale cached responses.
* **Non-Blocking Concurrency**: Document parsing, chunking, and indexing execute as background worker jobs (`asyncio.to_thread`), keeping the FastAPI event loop responsive.

### 2. Self-Corrective Query Execution Workflow
1. **Semantic Cache Check**: Intercepts greetings (zero-cost bypass) and checks cosine similarity against previously answered questions in Qdrant before invoking models (7-day TTL, 500-item collection cap).
2. **Intent Routing**: Classifies the query as casual conversation (`direct`) or requiring internal documentation (`kb`).
3. **Hybrid Retrieval**: Queries child chunks using dense Qdrant vector search and lexical BM25Okapi keyword matching, fusing candidates via Reciprocal Rank Fusion (`RRF k=60`).
4. **NVIDIA Reranking**: Fetches parent documents and re-scores them using `nvidia/llama-nemotron-rerank-vl-1b-v2` (`top_k=5`) to prioritize the most relevant context, with graceful fallback to fusion order.
5. **Evidence Grading**: Evaluates whether retrieved chunks contain sufficient facts to answer the question.
6. **Web Search Fallback**: Automatically invokes Tavily Live Search if private documents lack adequate evidence.
7. **Query Rewriter**: Reformulates search keywords if web evidence is also insufficient (capped at `max_retries=2` and `workflow_recursion_limit=25` to prevent infinite loops).
8. **Answer Generation**: Synthesizes grounded answers citing specific filenames and page numbers, and writes the response to the semantic cache.
9. **Audit Trail**: Records questions, routing decisions, citations, and diagnostic execution traces in SQLite (`data/audit.db`) using Write-Ahead Logging (WAL) and connection pooling.

---

## Key Features

- **Semantic Caching & Resource Safeguards**: Qdrant-backed cache stores previous answers based on cosine similarity, bypassing expensive LLM/embedding calls for repeat questions. Includes a zero-cost greeting interceptor, 7-day TTL expiration, 500-item size cap, and automated cache invalidation upon document ingestion.
- **Hybrid Search (BM25 + Dense + RRF)**: Combines keyword-based BM25Okapi retrieval with dense vector search via Reciprocal Rank Fusion, ensuring exact terms and rare technical entities are never missed.
- **Reranking Layer**: Uses NVIDIA NV-Rerank (`nvidia/llama-nemotron-rerank-vl-1b-v2`) to reorder retrieved parent candidates by semantic relevance.
- **Multi-Document Synthesis**: Supports answering cross-domain queries spanning multiple ingested documents within an expanded context window (`max_context_chars=18000`).
- **Hierarchical Chunking (Parent-Child)**:
  - **Child chunks (400 chars)**: Embedded with file & page metadata for pinpoint vector search.
  - **Parent chunks (2,000 chars)**: Retrieved for generation to preserve full context.
- **Self-Corrective Loop**: Validates evidence before generation; dynamically falls back to live Tavily web search when internal documentation is incomplete.
- **Concurrency & Storage Hardening**: SQLite configured with Write-Ahead Logging (`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;`) and thread-safe connection pooling to prevent database locking under concurrent load.
- **Dual Interface**:
  - **CLI Runner**: Interactive terminal streaming of reasoning steps, dynamic trace logs (showing exact filenames), and citations.
  - **Headless API**: FastAPI endpoints (`/api/chat`, `/api/ingest`, `/api/health`) with async offloading and MIME verification.
- **Production Test Suite**: Modular Pytest unit and integration tests (38 tests) verifying schemas, RRF, BM25, caching, audit concurrency, API routes, and PDF fixtures.

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd "Agentic RAG"

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux / Mac

# Install production dependencies
pip install -r requirements.txt

# (Optional) Install development and testing dependencies
pip install -r requirements-dev.txt
```

### 2. Environment Variables (`.env`)

Copy the template and fill in your API keys:

```bash
cp .env.example .env
```

Configuration parameters:

```ini
# ── Required API Keys ──────────────────────────────────────────────
NVIDIA_API_KEY=your_nvidia_api_key
TAVILY_API_KEY=your_tavily_api_key
QDRANT_API_KEY=your_qdrant_api_key

# ── Qdrant Cloud ──────────────────────────────────────────────────
CLUSTER_END_POINT=https://your-cluster.cloud.qdrant.io:6333
QDRANT_INDEX_NAME=STORAGE
QDRANT_NAME_SPACE=DEFAULT

# ── NVIDIA NIM Models ────────────────────────────────────────────
NVIDIA_CHAT_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NVIDIA_EMBEDDING_MODEL=nvidia/nemotron-3.5-embed-1b

# ── Hyperparameters & Tuning ─────────────────────────────────────
RRF_K=60
RERANK_MODEL=nvidia/llama-nemotron-rerank-vl-1b-v2
RERANK_TOP_K=5
MAX_CONTEXT_CHARS=18000
MAX_RETRIES=2
WORKFLOW_RECURSION_LIMIT=25

# ── Security & Limits ────────────────────────────────────────────
ADMIN_API_KEY=localhost
MAX_UPLOAD_SIZE_MB=25
CACHE_SIMILARITY_THRESHOLD=0.85
LOG_LEVEL=WARNING
```

---

## How to Use

### Option A: CLI Runner (Recommended)

```bash
# 1. Ingest documents (.pdf, .docx, .txt, .md)
python run.py --ingest "tests/fixtures/sample.pdf"

# 2. Ask a question directly
python run.py "What are the main paradigms of RAG?"

# 3. Launch interactive REPL mode
python run.py
```

Inside REPL mode:
- Type your question directly.
- `ingest <path>` — Ingest a document, sync hybrid indices, and invalidate cache.
- `clear-cache` — Manually clear cached answers for the current namespace.
- `flow` — Display the architecture flow banner.
- `exit` — Quit the CLI.

### Option B: FastAPI Microservice

```bash
# Start the server
uvicorn app.main:app --port 8080 --reload
```

- **Health Check**: `GET http://localhost:8080/api/health`
- **Chat Endpoint**: `POST http://localhost:8080/api/chat`
  ```json
  {
    "question": "Explain Modular RAG from the survey"
  }
  ```
- **Upload Document**: `POST http://localhost:8080/api/ingest`
  - Multipart form-data with `file` (supports `.pdf`, `.docx`, `.txt`, `.md`, max 25MB).
  - Header: `x-admin-key: localhost`

---

## Testing

Run the automated test suite with pytest:

```bash
# Run all unit and integration tests
pytest tests -v
```

The test suite covers:
- **`tests/unit/test_state.py`**: State schemas and Pydantic validation (`RouteDecision`, `EvidenceGrade`).
- **`tests/unit/test_rank_fusion.py`**: Reciprocal Rank Fusion math, deduplication, and edge cases.
- **`tests/unit/test_bm25.py`**: Tokenization, BM25Okapi corpus, scoring, and index invalidation.
- **`tests/unit/test_cache.py`**: Greeting interceptor, cosine similarity thresholding, and TTL eviction.
- **`tests/unit/test_audit.py`**: SQLite WAL mode, schema dynamic init, and concurrent write resilience.
- **`tests/unit/test_api.py`**: Endpoint auth, size limit enforcement (HTTP 413), MIME magic byte verification, and health check.
- **`tests/unit/test_cli.py`**: Banner formatting and execution trace printer rendering.
- **`tests/integration/test_pipeline.py`**: Deterministic integration tests using PDF fixtures (`tests/fixtures/sample.pdf` and `test.pdf`).

---

## Project Structure

```text
├── app/
│   ├── api/
│   │   └── routes.py         # REST endpoints (/api/chat, /api/ingest, /api/health)
│   ├── core/
│   │   ├── config.py         # Pydantic settings loader & externalized hyperparameters
│   │   └── logging.py        # Log formatting & noise suppression
│   ├── rag/
│   │   ├── state.py          # LangGraph state schema (AgentState)
│   │   ├── vector_stores.py  # Qdrant vector store & embedding singletons
│   │   ├── bm25.py           # In-memory BM25Okapi index & invalidation
│   │   ├── rank_fusion.py    # Reciprocal Rank Fusion (RRF)
│   │   ├── reranking.py      # NVIDIA NV-Rerank layer (Llama-Nemotron-Rerank)
│   │   ├── llm.py            # ChatNVIDIA & Tavily tool singletons
│   │   ├── cache_nodes.py    # Semantic cache lookup node & routing
│   │   ├── router.py         # Intent classification node (kb vs direct)
│   │   ├── retrieval.py      # Hybrid retrieval, web search, query rewrite
│   │   ├── grading.py        # Evidence grading nodes (KB & web)
│   │   ├── generation.py     # Grounded answer generation nodes
│   │   └── workflow.py       # LangGraph assembly (build_graph, ask)
│   └── services/
│       ├── audit.py          # SQLite audit logger with WAL mode & connection pooling
│       ├── cache.py          # Qdrant semantic cache with TTL/size eviction
│       └── ingestion.py      # Unified document loader, chunker & index synchronizer
├── cli/
│   ├── banner.py             # CLI banner and flow diagram text
│   ├── trace_printer.py      # Per-node trace and step printer
│   ├── pipeline.py           # Execution stream runner
│   ├── ingest.py             # CLI document ingestion handler
│   └── repl.py               # Interactive CLI shell
├── docs/
│   └── workflow_diagram.png  # Architecture workflow flowchart
├── tests/
│   └── test.py               # Integration smoke tests
├── run.py                    # CLI entrypoint (argparse + bootstrap)
├── requirements.txt          # Pinned production dependencies
├── requirements-dev.txt      # Testing and development dependencies
├── .env.example              # Environment variables template
└── .env                      # Local configuration (git-ignored)
```

*(Note: SQLite database in `data/` and uploaded documents in `uploads/` are generated at runtime and strictly ignored in version control).*

---

## Roadmap

- [x] **Semantic Caching** — Cosine similarity caching with 7-day TTL, 500-item cap, and zero-cost greeting fast-path.
- [x] **Hybrid Search (BM25 + Vector)** — Dense vector search combined with BM25Okapi keyword matching via Reciprocal Rank Fusion (`rrf_k=60`).
- [x] **Reranking Layer** — NVIDIA NV-Rerank (`nvidia/llama-nemotron-rerank-vl-1b-v2`) reorders parent candidates prior to evidence grading.
- [x] **Multi-Document Synthesis** — Grounded generation across multiple ingested source files in a unified response (`max_context_chars=18000`).
- [x] **FastAPI Backend Improvements** — Non-blocking async worker thread offloading, streaming upload size limits (25MB), and MIME magic-byte verification.
- [x] **Concurrency & Observability Hardening** — SQLite Write-Ahead Logging (WAL mode), connection pooling, and unified diagnostic telemetry.
- [x] **Modular Test Suite** — Comprehensive 38-test unit and integration test coverage with fixtures.
- [ ] **Interactive Web Frontend** — Modern web interface for chat, document upload, and real-time reasoning trace visualization.
- [ ] **RAG Evaluation** — Automated evaluation metrics for faithfulness, answer relevance, and context precision.
- [ ] **Minor Improvements** — Code polish, expanded file format loaders, and performance tuning.

---

## License

Apache 2.0
