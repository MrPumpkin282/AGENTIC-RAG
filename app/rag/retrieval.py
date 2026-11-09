import logging

from app.core.config import get_settings
from app.rag.state import AgentState
from app.rag.llm import llm, web_search_tool, add_trace, MAX_QUERY_CHARS
from app.rag.vector_stores import get_vectorstore, get_parents, _namespace_filter
from app.rag.bm25 import bm25_search
from app.rag.rank_fusion import reciprocal_rank_fusion
from app.rag.reranking import rerank

logger = logging.getLogger(__name__)
settings = get_settings()


def retrieve_kb(state: AgentState):
    """Hybrid retrieval: dense vector search + BM25 keyword search, fused via RRF, then reranked."""
    query = state.get("current_query", state["question"])[:MAX_QUERY_CHARS]

    # 1. Dense vector search (child chunks)
    dense_hits = get_vectorstore().similarity_search(
        query, k=settings.top_k * 2, filter=_namespace_filter()
    )

    # 2. BM25 keyword search (child chunks)
    try:
        bm25_hits = [doc for doc, _score in bm25_search(query, k=settings.top_k * 2)]
    except Exception as exc:
        logger.warning("BM25 search failed (%s), using dense-only retrieval", exc)
        bm25_hits = []

    # 3. Reciprocal Rank Fusion to merge both ranked lists
    if bm25_hits:
        fused = reciprocal_rank_fusion([dense_hits, bm25_hits])
    else:
        fused = dense_hits

    # 4. Resolve parent chunks from top-k fused child hits
    parent_ids = []
    for h in fused:
        pid = h.metadata.get("parent_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)
    docs = get_parents(parent_ids[: settings.top_k])

    # 5. Rerank parent documents for final ordering
    if docs:
        try:
            docs = list(rerank(query, docs))
        except Exception as exc:
            logger.warning("Reranking step failed (%s), using fusion order", exc)

    retrieval_method = "hybrid (dense+BM25+RRF)" if bm25_hits else "dense-only"
    return {
        "kb_docs": docs,
        "trace": add_trace(state, f"Private KB retrieval ({retrieval_method}) -> {len(docs)} chunks"),
    }


def search_web(state: AgentState):
    query = state.get("current_query", state["question"])[:MAX_QUERY_CHARS]
    result = web_search_tool().invoke({"query": query})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback -> Tavily search"),
    }


def rewrite_query(state: AgentState):
    raw = llm().invoke(f"""
Rewrite this question to optimize retrieval against the knowledge base and web search.
Extract core concepts, technical keywords, and topic terms. Remove meta phrasing (like 'in the pdf', 'tell me about').
Preserve original intent, do not answer, return only the search query on one line, under 30 words.
Question: {state['question']}
""").content.strip()
    rewritten = raw.splitlines()[-1].strip()[:300] if raw else ""
    rewritten = rewritten or state["question"]
    return {
        "current_query": rewritten,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": add_trace(state, f"Query rewrite -> {rewritten}"),
    }
