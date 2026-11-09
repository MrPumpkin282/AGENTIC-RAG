import logging
import time
import uuid

from qdrant_client import models
from app.core.config import get_settings
from app.rag.vector_stores import get_client, get_embeddings, _namespace

logger = logging.getLogger(__name__)
settings = get_settings()

# Define common zero-cost conversational triggers
GREETINGS = {"hi", "hey", "hello", "ping", "sup", "greetings", "hey there", "hi there"}

# --- FREE TIER SAFEGUARDS ---
MAX_CACHE_SIZE = 500                 # Maximum number of cached questions per collection
TTL_SECONDS = 7 * 24 * 60 * 60       # Cache expiration time: 7 days


def _cache_name():
    return f"{settings.qdrant_index_name}_cache"


def _cache_point_id(namespace: str, question: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cache:{namespace}:{question.strip().lower()}"))


def ensure_cache_collection():
    client = get_client()
    name = _cache_name()
    if client.collection_exists(name):
        return
    dimension = len(get_embeddings().embed_query("dimension probe"))
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=name,
        field_name="namespace",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _namespace_filter(namespace: str):
    return models.Filter(
        must=[models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))]
    )


def lookup(question: str):
    """
    Look up a semantically similar cached question with TTL validation.
    """
    if not question or not question.strip():
        return None

    cleaned_question = question.strip().lower()

    # 1. Zero-Cost Fast-Path Interceptor for Greetings
    if cleaned_question in GREETINGS:
        logger.debug("Exact match greeting intercepted. Bypassing embedding API.")
        return {
            "namespace": _namespace(),
            "question": question,
            "answer": "Hello! How can I help you with your documents today?",
            "source_used": "direct",
            "citations": [],
            "cached_at": time.time(),
        }

    # 2. Semantic Lookup via Qdrant Cloud
    ensure_cache_collection()
    client = get_client()
    namespace = _namespace()
    query_vector = get_embeddings().embed_query(question)

    hits = client.query_points(
        collection_name=_cache_name(),
        query=query_vector,
        query_filter=_namespace_filter(namespace),
        limit=1,
        with_payload=True,
    ).points

    if not hits:
        return None

    hit = hits[0]
    
    # 3. TTL Expiration Check
    cached_at = hit.payload.get("cached_at", 0)
    if time.time() - cached_at > TTL_SECONDS:
        logger.debug("Semantic cache expired (Age > 7 days). Deleting stale point.")
        client.delete(
            collection_name=_cache_name(),
            points_selector=models.PointIdsList(points=[hit.id]),
        )
        return None

    # 4. Similarity Threshold Check
    if hit.score < settings.cache_similarity_threshold:
        logger.debug("Semantic cache miss (best score %.4f < threshold %.4f)", hit.score, settings.cache_similarity_threshold)
        return None

    logger.debug("Semantic cache hit (score %.4f) for: %s", hit.score, question[:80])
    return hit.payload


def _enforce_size_limit(client, collection_name: str, namespace: str):
    """
    Enforces a strict size limit by evicting the oldest cached entries 
    if the collection exceeds MAX_CACHE_SIZE.
    """
    try:
        # Count current points in this namespace
        count_result = client.count(
            collection_name=collection_name,
            count_filter=_namespace_filter(namespace)
        )
        current_count = count_result.count

        if current_count >= MAX_CACHE_SIZE:
            logger.debug("Cache size limit reached (%d/%d). Evicting oldest entries...", current_count, MAX_CACHE_SIZE)
            
            # Scroll through points to find the oldest ones
            records, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=_namespace_filter(namespace),
                with_payload=True,
                with_vectors=False,
                limit=MAX_CACHE_SIZE
            )
            
            if records:
                # Sort records by 'cached_at' timestamp ascending (oldest first)
                sorted_records = sorted(records, key=lambda r: r.payload.get("cached_at", 0))
                
                # Determine how many to delete to drop comfortably below the limit
                excess_count = (current_count - MAX_CACHE_SIZE) + 10  
                points_to_delete = [r.id for r in sorted_records[:excess_count]]
                
                if points_to_delete:
                    client.delete(
                        collection_name=collection_name,
                        points_selector=models.PointIdsList(points=points_to_delete),
                    )
                    logger.debug("Evicted %d oldest cache entries to free up Qdrant space.", len(points_to_delete))
    except Exception as exc:
        logger.warning("Failed to enforce cache size limit: %s", exc)


def store(question: str, answer: str, source_used: str, citations=None):
    """
    Cache a question/answer pair with automatic size-based eviction.
    """
    if source_used not in ("kb", "web"):
        return
    if not question or not answer:
        return

    ensure_cache_collection()
    client = get_client()
    namespace = _namespace()

    # Enforce size constraints before adding new entries
    _enforce_size_limit(client, _cache_name(), namespace)

    vector = get_embeddings().embed_query(question)
    point_id = _cache_point_id(namespace, question)

    point = models.PointStruct(
        id=point_id,
        vector=vector,
        payload={
            "namespace": namespace,
            "question": question,
            "answer": answer,
            "source_used": source_used,
            "citations": citations or [],
            "cached_at": time.time(),
        },
    )
    client.upsert(collection_name=_cache_name(), points=[point])
    logger.debug("Semantic cache store for: %s", question[:80])


def clear_namespace():
    """Delete all cached entries for the current namespace."""
    ensure_cache_collection()
    get_client().delete(
        collection_name=_cache_name(),
        points_selector=models.FilterSelector(filter=_namespace_filter(_namespace())),
    )
    logger.debug("Semantic cache cleared for namespace: %s", _namespace())
