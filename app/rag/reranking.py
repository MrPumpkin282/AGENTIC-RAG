import logging

from langchain_nvidia_ai_endpoints import NVIDIARerank

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is missing for reranker")
        _reranker = NVIDIARerank(
            model=settings.rerank_model,
            api_key=settings.nvidia_api_key,
            top_n=settings.rerank_top_k,
        )
    return _reranker


def rerank(query: str, docs: list):
    """Rerank documents using NVIDIA NV-Rerank. Falls back to original order on error."""
    if not docs:
        return docs
    try:
        return get_reranker().compress_documents(documents=docs, query=query)
    except Exception as exc:
        logger.warning("Reranking failed (%s), using original retrieval order", exc)
        return docs
