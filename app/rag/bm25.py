# app/rag/bm25.py
import re
import threading

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.rag.vector_stores import _namespace_filter, get_client

settings = get_settings()

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())

class _BM25Index:
    """In-memory BM25 index over child chunks for one namespace, lazily built
    and cached. Call invalidate() after ingesting new documents."""

    def __init__(self):
        self._lock = threading.Lock()
        self._bm25 = None
        self._docs: list[Document] = []   # child chunks, parallel to _bm25 corpus
        self._built_for_namespace = None

    def _build(self):
        client = get_client()
        docs, next_offset = [], None
        while True:
            points, next_offset = client.scroll(
                collection_name=settings.qdrant_index_name,
                scroll_filter=_namespace_filter(),
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_offset,
            )
            for p in points:
                docs.append(Document(
                    page_content=p.payload["page_content"],
                    metadata=p.payload["metadata"],
                ))
            if next_offset is None:
                break

        corpus = [_tokenize(d.page_content) for d in docs]
        self._docs = docs
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def ensure_built(self):
        with self._lock:
            if self._bm25 is None:
                self._build()

    def invalidate(self):
        with self._lock:
            self._bm25 = None
            self._docs = []

    def search(self, query: str, k: int):
        self.ensure_built()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in ranked if scores[i] > 0]


_index = _BM25Index()

def bm25_search(query: str, k: int = 10):
    """Returns list[(Document, score)] of top-k child chunks by BM25."""
    return _index.search(query, k)

def invalidate_bm25_index():
    """Call this after add_documents() in ingestion so new docs are searchable."""
    _index.invalidate()