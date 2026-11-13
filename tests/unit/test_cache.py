import time
from unittest.mock import MagicMock, patch

from app.services.cache import clear_namespace, lookup


def test_lookup_empty_question():
    assert lookup("") is None
    assert lookup("   ") is None


def test_lookup_greeting_fast_path():
    for g in ["hello", "Hi", "HEY!"]:
        result = lookup(g)
        assert result is not None
        assert "Hello! How can I help you" in result["answer"]


@patch("app.services.cache.ensure_cache_collection")
@patch("app.services.cache.get_embeddings")
@patch("app.services.cache.get_client")
def test_lookup_cache_hit(mock_client, mock_embeddings, mock_ensure):
    mock_embeddings.return_value.embed_query.return_value = [0.1] * 1024

    mock_hit = MagicMock()
    mock_hit.score = 0.95
    mock_hit.id = "point-123"
    mock_hit.payload = {
        "namespace": "default",
        "question": "What is Python?",
        "answer": "Python is a programming language.",
        "source_used": "cache",
        "citations": [{"title": "Doc1"}],
        "cached_at": time.time(),
    }

    mock_query_res = MagicMock()
    mock_query_res.points = [mock_hit]
    mock_client.return_value.query_points.return_value = mock_query_res

    result = lookup("Explain Python programming")
    assert result is not None
    assert result["answer"] == "Python is a programming language."
    assert result["score"] == 0.95
    assert result["source_used"] == "cache"


@patch("app.services.cache.ensure_cache_collection")
@patch("app.services.cache.get_embeddings")
@patch("app.services.cache.get_client")
def test_lookup_cache_miss_below_threshold(mock_client, mock_embeddings, mock_ensure):
    mock_embeddings.return_value.embed_query.return_value = [0.1] * 1024

    mock_hit = MagicMock()
    mock_hit.score = 0.70  # Below 0.88 default threshold
    mock_hit.id = "point-456"
    mock_hit.payload = {
        "namespace": "default",
        "question": "Unrelated topic",
        "answer": "Unrelated answer",
        "source_used": "kb",
        "cached_at": time.time(),
    }

    mock_query_res = MagicMock()
    mock_query_res.points = [mock_hit]
    mock_client.return_value.query_points.return_value = mock_query_res

    result = lookup("Explain quantum computing")
    assert result is None


@patch("app.services.cache.get_client")
def test_clear_namespace(mock_client):
    with patch("app.services.cache.ensure_cache_collection"):
        clear_namespace()
        mock_client.return_value.delete.assert_called_once()
