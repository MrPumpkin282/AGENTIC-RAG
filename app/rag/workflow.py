import logging
from typing import Literal
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

def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision)
    decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.
Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, or technology support.
Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
Question: {state['question']}
""")
    return {"source_used": decision.route, "trace": add_trace(state, f"Router → {decision.route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

def retrieve_kb(state: AgentState):
    query = state.get("current_query", state["question"])[:MAX_QUERY_CHARS]
    hits = get_vectorstore().similarity_search(
        query, k=settings.top_k * 2, filter=_namespace_filter()
    )
    parent_ids = []
    for h in hits:
        pid = h.metadata["parent_id"]
        if pid not in parent_ids:
            parent_ids.append(pid)
    docs = get_parents(parent_ids[: settings.top_k])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks")}

def grade_kb(state: AgentState):
    docs = state.get("kb_docs", [])
    if not docs:
        return {"kb_grade": "weak", "trace": add_trace(state, "KB evidence grade → WEAK (no chunks)")}
    grader = llm().with_structured_output(EvidenceGrade)
    context = build_context(docs)
    grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.
Question: {state['question']}
Private company KB evidence:\n{context}
Return good only if the evidence is sufficient to answer confidently and specifically.
Otherwise return weak.
""")
    return {"kb_grade": grade.grade, "trace": add_trace(state, f"KB evidence grade → {grade.grade.upper()}")}

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
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }

def grade_web(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade)
    grade = grader.invoke(f"""
Question: {state['question']}
Web evidence:\n{state.get('web_results', '')[:MAX_CONTEXT_CHARS]}
Return good if the evidence is sufficient and directly relevant; otherwise weak.
""")
    return {"web_grade": grade.grade, "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}")}

def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state.get("retry_count", 0) < settings.max_retries:
        return "rewrite_query"
    return "insufficient"

def rewrite_query(state: AgentState):
    raw = llm().invoke(f"""
Rewrite this IT support question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful technical keywords, do not answer, return only the query on one line, under 30 words.
Question: {state['question']}
""").content.strip()
    rewritten = raw.splitlines()[-1].strip()[:300] if raw else ""
    rewritten = rewritten or state["question"]
    return {
        "current_query": rewritten,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }

def generate_from_kb(state: AgentState):
    docs = state.get("kb_docs", [])
    context = build_context(docs)
    answer = llm().invoke(f"""
You are an enterprise IT support copilot. Answer ONLY from the private company KB below.
Be concise, actionable, and safe. If steps are present, present them clearly.
Do not invent policy details. Mention that the answer is based on the company's private knowledge base.
Question: {state['question']}\n\nPrivate KB:\n{context}
""").content
    citations, seen = [], set()
    for d in docs:
        name = d.metadata.get("file_name", "Private KB")
        page = d.metadata.get("page")
        if (name, page) in seen:
            continue
        seen.add((name, page))
        c = {"source": name, "title": name}
        if page is not None:
            c["page"] = page + 1
        citations.append(c)
    return {"answer": answer, "source_used": "kb", "citations": citations, "trace": add_trace(state, "Answer generation → PRIVATE KB")}

def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are an enterprise IT support copilot. The private company KB was insufficient.
Answer ONLY from the web evidence below. Clearly say this is external web information and may need IT validation before changing company-managed systems.
Question: {state['question']}\n\nWeb evidence:\n{state.get('web_results', '')[:MAX_CONTEXT_CHARS]}
""").content
    return {"answer": answer, "source_used": "web", "trace": add_trace(state, "Answer generation → WEB SEARCH")}

def direct_answer(state: AgentState):
    answer = llm().invoke(f"Respond briefly and naturally to: {state['question']}").content
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response → no retrieval")}

def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the IT help desk or provide more details.",
        "source_used": "none",
        "citations": [],
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
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