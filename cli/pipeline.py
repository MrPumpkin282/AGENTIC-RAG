from app.rag.workflow import agent_graph
from app.rag.state import AgentState
from app.services.audit import write_audit
from cli.trace_printer import print_node_update


def run_pipeline(question: str, settings) -> dict:
    print("\n" + "=" * 79)
    print(f" QUESTION: {question}")
    print("=" * 79)

    initial: AgentState = {
        "question": question,
        "current_query": question,
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    final_state: dict = dict(initial)
    config = {"recursion_limit": settings.workflow_recursion_limit}

    for chunk in agent_graph.stream(initial, stream_mode="updates", config=config):
        for node_name, updates in chunk.items():
            final_state.update(updates)
            print_node_update(node_name, updates, final_state, question, settings)

    answer = final_state.get("answer", "No answer generated.")
    source_used = final_state.get("source_used", "none").upper()
    citations = final_state.get("citations", [])

    print("\n" + "=" * 79)
    print(" FINAL ANSWER")
    print("=" * 79)
    print(f"Source Used: {source_used}")
    print("-" * 79)
    print(answer)
    print("-" * 79)

    if citations:
        print("Citations & Sources:")
        for idx, cite in enumerate(citations, start=1):
            title = cite.get("title") or cite.get("source") or "Reference"
            url = cite.get("url")
            page = cite.get("page")
            details = []
            if page and f"page {page}" not in title.lower():
                details.append(f"page {page}")
            if url:
                details.append(url)
            extra = f" ({', '.join(details)})" if details else ""
            print(f"  [{idx}] {title}{extra}")
    print("=" * 79 + "\n")

    # Record into SQLite audit database
    try:
        write_audit(
            question,
            final_state.get("source_used", "none"),
            final_state.get("trace", []),
            citations,
        )
    except Exception:
        pass

    return final_state
