
from typing import List, Literal
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field
from langchain_core.documents import Document


# 1. Defines the routing decision made by the LLM
class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description=(
            "Choose 'kb' for IT/support questions requiring knowledge-base "
            "retrieval, or 'direct' for greetings and simple chat."
        )
    )


# 2. Defines the evidence-quality grading output
class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description=(
            "Use 'good' when the evidence sufficiently answers the question; "
            "use 'weak' when it is insufficient or irrelevant."
        )
    )


# 3. Defines the structure of a citation
class Citation(TypedDict, total=False):
    source: str
    title: str
    url: str
    page: int


# 4. Defines the shared state of the LangGraph workflow
class AgentState(TypedDict):
    # Required input
    question: str

    # Optional fields populated as the workflow executes
    current_query: NotRequired[str]
    kb_docs: NotRequired[List[Document]]
    web_results: NotRequired[str]

    kb_grade: NotRequired[Literal["good", "weak"]]
    web_grade: NotRequired[Literal["good", "weak"]]

    answer: NotRequired[str]
    source_used: NotRequired[Literal["kb", "web", "direct", "none"]]

    retry_count: NotRequired[int]
    trace: NotRequired[List[str]]
    citations: NotRequired[List[Citation]]
