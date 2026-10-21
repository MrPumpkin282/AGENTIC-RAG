import logging
import warnings
from typing import Literal

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vector_stores import get_vectorstore, get_parents, _namespace_filter

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None

MAX_CONTEXT_CHARS = 6000
MAX_QUERY_CHARS = 1000


def llm():
    global _llm
    if _llm is None:
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is missing")
        if not settings.nvidia_chat_model:
            raise RuntimeError("NVIDIA_CHAT_MODEL is missing")
        _llm = ChatNVIDIA(
            model=settings.nvidia_chat_model,
            temperature=0,
            max_completion_tokens=2048,
            api_key=settings.nvidia_api_key,
        )
    return _llm

def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]

def doc_label(d):
    name = d.metadata.get("file_name", "unknown")
    page = d.metadata.get("page")
    return f"{name}, page {page + 1}" if page is not None else name

def build_context(docs):
    parts, total = [], 0
    for d in docs:
        text = f"[Source: {doc_label(d)}]\n{d.page_content}"
        if total + len(text) > MAX_CONTEXT_CHARS:
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)

def _classify_route(question: str) -> str:
    prompt = f"""You route user questions for an intelligent knowledge assistant.
Use 'kb' for any questions seeking facts, concepts, summaries, troubleshooting, or information from ingested documents, research papers, technical manuals, or company documentation.
Use 'direct' ONLY for simple greetings (hello, hi), expressions of thanks, or casual conversational pleasantries that require no reference material.
Question: {question}
Return ONLY 'kb' or 'direct'."""
    try:
        decision = llm().with_structured_output(RouteDecision).invoke(prompt)
        if decision and hasattr(decision, "route") and decision.route in ("kb", "direct"):
            return decision.route
    except Exception as exc:
        logger.warning("Structured route failed (%s), falling back to text parsing", exc)
    raw = llm().invoke(prompt).content.strip().lower()
    return "direct" if "direct" in raw and "kb" not in raw else "kb"

def _grade_evidence(question: str, evidence: str, evidence_type: str = "Knowledge Base") -> str:
    prompt = f"""You grade retrieved evidence against the user question.
Question: {question}
{evidence_type} evidence:
{evidence}
Return 'good' if the evidence contains relevant and sufficient facts to answer the question specifically.
Otherwise return 'weak'.
Return ONLY 'good' or 'weak'."""
    try:
        grade = llm().with_structured_output(EvidenceGrade).invoke(prompt)
        if grade and hasattr(grade, "grade") and grade.grade in ("good", "weak"):
            return grade.grade
    except Exception as exc:
        logger.warning("Structured grade failed (%s), falling back to text parsing", exc)
    raw = llm().invoke(prompt).content.strip().lower()
    return "good" if "good" in raw and "weak" not in raw else "weak"

def route_question(state: AgentState):
    route = _classify_route(state["question"])
    return {"source_used": route, "trace": add_trace(state, f"Router -> {route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

def retrieve_kb(state: AgentState):
    query = state.get("current_query", state["question"])[:MAX_QUERY_CHARS]
    hits = get_vectorstore().similarity_search(
        query, k=settings.top_k * 2, filter=_namespace_filter()
    )
    parent_ids = []
    for h in hits:
        pid = h.metadata.get("parent_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)
    docs = get_parents(parent_ids[: settings.top_k])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval -> {len(docs)} chunks")}

def grade_kb(state: AgentState):
    docs = state.get("kb_docs", [])
    if not docs:
        return {"kb_grade": "weak", "trace": add_trace(state, "KB evidence grade -> WEAK (no chunks)")}
    context = build_context(docs)
    grade = _grade_evidence(state["question"], context, "Private company KB")
    return {"kb_grade": grade, "trace": add_trace(state, f"KB evidence grade -> {grade.upper()}")}

def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"

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

def grade_web(state: AgentState):
    web_text = state.get("web_results", "")[:MAX_CONTEXT_CHARS]
    if not web_text.strip():
        return {"web_grade": "weak", "trace": add_trace(state, "Web evidence grade -> WEAK (no results)")}
    grade = _grade_evidence(state["question"], web_text, "Web")
    return {"web_grade": grade, "trace": add_trace(state, f"Web evidence grade -> {grade.upper()}")}

def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state.get("retry_count", 0) < settings.max_retries:
        return "rewrite_query"
    return "insufficient"

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

def generate_from_kb(state: AgentState):
    docs = state.get("kb_docs", [])
    context = build_context(docs)
    answer = llm().invoke(f"""
You are an expert AI knowledge assistant. Answer the user question accurately and comprehensively using ONLY the provided knowledge base context below.
Be concise, well-structured, and factual. Do not fabricate facts outside the context.
Question: {state['question']}\n\nKnowledge Base:\n{context}
""").content
    citations, seen = [], set()
    for d in docs:
        name = d.metadata.get("file_name", "Knowledge Base")
        page = d.metadata.get("page")
        if (name, page) in seen:
            continue
        seen.add((name, page))
        title = f"{name} (page {page + 1})" if page is not None else name
        c = {"source": name, "title": title}
        if page is not None:
            c["page"] = page + 1
        citations.append(c)
    return {"answer": answer, "source_used": "kb", "citations": citations, "trace": add_trace(state, "Answer generation -> KNOWLEDGE BASE")}

def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are an expert AI knowledge assistant. The private knowledge base was insufficient to answer this question.
Answer the user question using ONLY the external web evidence below. Note clearly in your answer that this information is synthesized from external web sources.
Question: {state['question']}\n\nWeb evidence:\n{state.get('web_results', '')[:MAX_CONTEXT_CHARS]}
""").content
    return {"answer": answer, "source_used": "web", "trace": add_trace(state, "Answer generation -> WEB SEARCH")}

def direct_answer(state: AgentState):
    answer = llm().invoke(f"Respond briefly and naturally to: {state['question']}").content
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response -> no retrieval")}

def insufficient(state: AgentState):
    return {
        "answer": "I could not find enough reliable evidence in the knowledge base or external search to answer your question with confidence. Please refine your question or provide more details.",
        "source_used": "none",
        "citations": [],
        "trace": add_trace(state, "Stopped -> insufficient reliable evidence"),
    }

def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()

agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)