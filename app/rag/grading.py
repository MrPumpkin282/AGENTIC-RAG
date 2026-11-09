import logging
from typing import Literal
from app.core.config import get_settings
from app.rag.state import AgentState, EvidenceGrade
from app.rag.llm import llm, add_trace, build_context, MAX_CONTEXT_CHARS

logger = logging.getLogger(__name__)
settings = get_settings()


def _grade_evidence(question: str, evidence: str, evidence_type: str = "Knowledge Base") -> str:
    prompt = f"""You grade retrieved evidence against the user question.
Question: {question}
{evidence_type} evidence:
{evidence}
Return 'good' if the evidence is relevant to at least one part of the question AND contains facts that help form an answer to that part.
Return 'weak' only if the evidence is unrelated to the question or contains no useful facts.
Return ONLY 'good' or 'weak'."""
    try:
        grade = llm().with_structured_output(EvidenceGrade).invoke(prompt)
        if grade and hasattr(grade, "grade") and grade.grade in ("good", "weak"):
            return grade.grade
    except Exception as exc:
        logger.warning("Structured grade failed (%s), falling back to text parsing", exc)
    raw = llm().invoke(prompt).content.strip().lower()
    return "good" if "good" in raw and "weak" not in raw else "weak"

def grade_kb(state: AgentState):
    docs = state.get("kb_docs", [])
    if not docs:
        return {"kb_grade": "weak", "trace": add_trace(state, "KB evidence grade -> WEAK (no chunks)")}
    context = build_context(docs)
    grade = _grade_evidence(state["question"], context, "Private company KB")
    return {"kb_grade": grade, "trace": add_trace(state, f"KB evidence grade -> {grade.upper()}")}

def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"

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