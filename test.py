import traceback
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

from app.core.config import get_settings
from app.rag.state import RouteDecision, EvidenceGrade

settings = get_settings()


def check(name, fn):
    print(f"\n--- {name} ---")
    try:
        result = fn()
        print("PASS:", result)
        return True
    except Exception:
        print("FAIL:")
        traceback.print_exc()
        return False


def test_chat():
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    llm = ChatNVIDIA(
        model=settings.nvidia_chat_model,
        temperature=0,
        api_key=settings.nvidia_api_key,
    )
    return llm.invoke("Say hello in one short sentence.").content


def test_route_structured():
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    llm = ChatNVIDIA(
        model=settings.nvidia_chat_model,
        temperature=0,
        api_key=settings.nvidia_api_key,
    )
    router = llm.with_structured_output(RouteDecision)
    return router.invoke("How do I reset my VPN password?")


def test_grade_structured():
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    llm = ChatNVIDIA(
        model=settings.nvidia_chat_model,
        temperature=0,
        api_key=settings.nvidia_api_key,
    )
    grader = llm.with_structured_output(EvidenceGrade)
    return grader.invoke(
        "Question: What is the capital of France?\n"
        "Evidence: Paris is the capital of France.\n"
        "Return good if the evidence answers the question, otherwise weak."
    )


def test_embeddings():
    from app.rag.vector_stores import get_embeddings
    vec = get_embeddings().embed_query("how do I connect to the VPN")
    return f"vector size = {len(vec)}"


def test_qdrant():
    from app.rag.vector_stores import get_client
    names = [c.name for c in get_client().get_collections().collections]
    return f"connected, collections = {names}"


def test_ensure_collections():
    from app.rag.vector_stores import ensure_collections
    ensure_collections()
    return "collections ready"


def test_tavily():
    from langchain_tavily import TavilySearch
    tool = TavilySearch(tavily_api_key=settings.tavily_api_key, max_results=1)
    out = tool.invoke({"query": "latest Microsoft Teams outage"})
    return f"got {len(out.get('results', []))} result(s)"


if __name__ == "__main__":
    print("Chat model     :", settings.nvidia_chat_model)
    print("Embedding model:", settings.nvidia_embedding_model)

    results = {
        "chat model": check("1. Chat model (plain)", test_chat),
        "structured route": check("2. Structured output: RouteDecision", test_route_structured),
        "structured grade": check("3. Structured output: EvidenceGrade", test_grade_structured),
        "embeddings": check("4. Embedding model", test_embeddings),
        "qdrant": check("5. Qdrant connection", test_qdrant),
        "collections": check("6. ensure_collections()", test_ensure_collections),
        "tavily": check("7. Tavily web search", test_tavily),
    }

    print("\n========== SUMMARY ==========")
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")