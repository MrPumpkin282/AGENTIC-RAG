import pytest
from pydantic import ValidationError
from app.rag.state import AgentState, RouteDecision, EvidenceGrade


def test_route_decision_valid():
    r1 = RouteDecision(route="kb")
    assert r1.route == "kb"

    r2 = RouteDecision(route="direct")
    assert r2.route == "direct"


def test_route_decision_invalid():
    with pytest.raises(ValidationError):
        RouteDecision(route="invalid_route")  # type: ignore


def test_evidence_grade_valid():
    g1 = EvidenceGrade(grade="good")
    assert g1.grade == "good"

    g2 = EvidenceGrade(grade="weak")
    assert g2.grade == "weak"


def test_evidence_grade_invalid():
    with pytest.raises(ValidationError):
        EvidenceGrade(grade="unacceptable")  # type: ignore


def test_agent_state_keys():
    state: AgentState = {
        "question": "What is hybrid RAG?",
        "current_query": "hybrid RAG",
        "retry_count": 0,
        "trace": ["cache_miss", "router_kb"],
        "citations": [{"title": "Doc", "url": "https://example.com"}],
        "source_used": "kb",
        "answer": "Hybrid RAG combines dense and sparse search.",
    }
    assert state["question"] == "What is hybrid RAG?"
    assert state["retry_count"] == 0
    assert len(state["trace"]) == 2
    assert len(state["citations"]) == 1
