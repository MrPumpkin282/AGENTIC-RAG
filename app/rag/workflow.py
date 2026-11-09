import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState
from app.rag.cache_nodes import check_cache, route_after_cache
from app.rag.router import route_question, route_after_router
from app.rag.retrieval import retrieve_kb, search_web, rewrite_query
from app.rag.grading import grade_kb, after_kb, grade_web, after_web
from app.rag.generation import generate_from_kb, generate_from_web, direct_answer, insufficient

settings = get_settings()


def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "check_cache": check_cache,
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

    graph.add_edge(START, "check_cache")
    graph.add_conditional_edges("check_cache", route_after_cache, {
        "route_question": "route_question",
        END: END
    })
    
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
    config = {"recursion_limit": settings.workflow_recursion_limit}
    return agent_graph.invoke(initial, config=config)
