from typing import Literal
from langgraph.graph import END
from app.rag.state import AgentState
from app.rag.llm import add_trace
from app.services.cache import lookup


# =====================================================================
# SEMANTIC CACHE NODES & ROUTING
# =====================================================================

def check_cache(state: AgentState):
    """Check the semantic cache and greeting interceptor before hitting the router."""
    question = state["question"]
    cached = lookup(question)
    if cached:
        return {
            "answer": cached.get("answer"),
            "source_used": cached.get("source_used", "cache"),
            "citations": cached.get("citations", []),
            "trace": add_trace(state, "Semantic cache -> HIT"),
            "cache_hit": True
        }
    return {"cache_hit": False, "trace": add_trace(state, "Semantic cache -> MISS")}

def route_after_cache(state: AgentState) -> Literal["route_question", "__end__"]:
    """Route directly to END if cache hits, otherwise proceed to normal intent routing."""
    return END if state.get("cache_hit") else "route_question"