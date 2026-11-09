from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from app.rag.bm25 import _tokenize, _BM25Index, bm25_search, invalidate_bm25_index


def test_tokenize():
    text = "Hello, World! Testing BM-25 tokenization... 123"
    tokens = _tokenize(text)
    assert tokens == ["hello", "world", "testing", "bm", "25", "tokenization", "123"]


def test_bm25_index_search_and_invalidate():
    index = _BM25Index()

    # Manually populate corpus
    d1 = Document(page_content="Python is an awesome programming language for AI")
    d2 = Document(page_content="Cooking pasta requires boiling salted water")
    d3 = Document(page_content="Artificial intelligence and Python neural networks")

    index._docs = [d1, d2, d3]
    corpus = [_tokenize(d.page_content) for d in index._docs]
    index._bm25 = BM25Okapi(corpus)

    # Search query
    results = index.search("Python AI", k=2)
    assert len(results) >= 1
    top_doc, score = results[0]
    assert score > 0
    assert "Python" in top_doc.page_content

    # Invalidate
    index.invalidate()
    assert index._bm25 is None
    assert index._docs == []


def test_global_bm25_search_mocked():
    with patch("app.rag.bm25._index") as mock_idx:
        mock_idx.search.return_value = [(Document(page_content="hit"), 1.5)]
        hits = bm25_search("query", k=5)
        assert len(hits) == 1
        assert hits[0][1] == 1.5
        mock_idx.search.assert_called_once_with("query", 5)

        invalidate_bm25_index()
        mock_idx.invalidate.assert_called_once()
