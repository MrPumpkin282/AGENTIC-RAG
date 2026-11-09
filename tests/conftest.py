import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.documents import Document

# Set test environment variables before importing app modules
os.environ["APP_ENV"] = "testing"
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY", "test-nvidia-key")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "test-tavily-key")
os.environ["ADMIN_API_KEY"] = "test-admin-secret"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"
os.environ["QDRANT_INDEX_NAME"] = "test_agentic_rag"
os.environ["QDRANT_NAME_SPACE"] = "test_namespace"


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Antigravity Agentic RAG provides hybrid retrieval combining dense vectors with BM25 keyword search.",
            metadata={"source": "sample.pdf", "page": 0, "parent_id": "p0", "child_id": "p0-c0"},
        ),
        Document(
            page_content="NVIDIA NIM provides high-throughput inference for embedding models and Nemotron LLMs.",
            metadata={"source": "sample.pdf", "page": 1, "parent_id": "p1", "child_id": "p1-c0"},
        ),
        Document(
            page_content="Qdrant is an open-source vector database supporting payload filtering and cosine similarity.",
            metadata={"source": "sample.pdf", "page": 2, "parent_id": "p2", "child_id": "p2-c0"},
        ),
    ]


@pytest.fixture
def mock_embeddings():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 1024
    embedder.embed_documents.side_effect = lambda texts: [[0.1] * 1024 for _ in texts]
    return embedder
