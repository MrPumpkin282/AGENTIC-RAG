import logging
from typing import Literal
from app.rag.state import AgentState, RouteDecision
from app.rag.llm import llm, add_trace

logger = logging.getLogger(__name__)


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

def route_question(state: AgentState):
    route = _classify_route(state["question"])
    return {"source_used": route, "trace": add_trace(state, f"Router -> {route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"