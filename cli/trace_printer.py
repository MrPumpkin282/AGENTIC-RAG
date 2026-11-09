def print_step(title: str, details: list[str]) -> None:
    print(f"\n┌── {title}")
    for line in details:
        print(f"│   {line}")
    print("└──" + "─" * 60)


def print_node_update(node_name: str, updates: dict, final_state: dict, question: str, settings) -> None:
    """Pretty-print the trace output for a single LangGraph node update,
    mirroring the per-node branches from the original CLI runner."""

    if node_name == "check_cache":
        cache_hit = updates.get("cache_hit", False)
        if cache_hit:
            print_step(
                "[STEP 0: SEMANTIC CACHE CHECK]",
                [
                    "Result       : HIT — returning cached answer",
                    "Action       : Skipping router, retrieval, and LLM calls",
                ],
            )
        else:
            print_step(
                "[STEP 0: SEMANTIC CACHE CHECK]",
                [
                    "Result       : MISS — proceeding to intent routing",
                ],
            )

    elif node_name == "route_question":
        route = updates.get("source_used", "kb")
        desc = (
            "Private Knowledge Base (requires company documentation)"
            if route == "kb"
            else "Direct Conversational (greeting or general conversation)"
        )
        print_step(
            "[STEP 1: INTENT ROUTING]",
            [
                f"Target Route : {route.upper()}",
                f"Routing Logic: {desc}",
            ],
        )

    elif node_name == "retrieve_kb":
        docs = updates.get("kb_docs", [])
        query_used = final_state.get("current_query", question)
        file_names = list({d.metadata.get("file_name", "Unknown") for d in docs})
        # Extract retrieval method from trace message
        trace_list = updates.get("trace", [])
        last_trace = trace_list[-1] if trace_list else ""
        method = "hybrid (dense+BM25+RRF)" if "hybrid" in last_trace else "dense-only"
        print_step(
            "[STEP 2: PRIVATE KB RETRIEVAL (Qdrant)]",
            [
                f"Search Query : \"{query_used}\"",
                f"Namespace    : {settings.qdrant_name_space or 'default'}",
                f"Method       : {method}",
                f"Hits Found   : {len(docs)} parent document chunk(s)",
                f"Sources      : {', '.join(file_names) if file_names else 'None (Empty retrieval)'}",
            ],
        )

    elif node_name == "grade_kb":
        grade = updates.get("kb_grade", "weak")
        eval_desc = (
            "GOOD -> Private KB contains sufficient facts to answer."
            if grade == "good"
            else "WEAK -> Private KB lacks required facts; triggering web search fallback."
        )
        print_step(
            "[STEP 3: KB EVIDENCE GRADING]",
            [
                f"Evaluation   : {grade.upper()}",
                f"Action       : {eval_desc}",
            ],
        )

    elif node_name == "search_web":
        query_used = final_state.get("current_query", question)
        citations = updates.get("citations", [])
        print_step(
            "[STEP 4: WEB SEARCH FALLBACK (Tavily)]",
            [
                f"Search Query : \"{query_used}\"",
                f"Web Sources  : {len(citations)} external source(s) retrieved",
            ]
            + [f" • {c.get('title', 'Web source')[:60]} ({c.get('url', '')})" for c in citations[:3]],
        )

    elif node_name == "grade_web":
        grade = updates.get("web_grade", "weak")
        retry_count = final_state.get("retry_count", 0)
        eval_desc = (
            "GOOD -> External web evidence is relevant and sufficient."
            if grade == "good"
            else f"WEAK -> Web results insufficient (Retry {retry_count}/{settings.max_retries})."
        )
        print_step(
            "[STEP 5: WEB EVIDENCE GRADING]",
            [
                f"Evaluation   : {grade.upper()}",
                f"Action       : {eval_desc}",
            ],
        )

    elif node_name == "rewrite_query":
        rewritten = updates.get("current_query", "")
        retries = updates.get("retry_count", 1)
        print_step(
            f"[STEP 6: QUERY REWRITER & RETRY (Loop #{retries})]",
            [
                f"Original Query   : \"{question}\"",
                f"Reformulated     : \"{rewritten}\"",
                "Action           : Looping back to Step 2 with refined technical keywords...",
            ],
        )

    elif node_name == "generate_from_kb":
        citations = updates.get("citations", [])
        # Dynamically extract the last trace message from the state updates
        trace_list = updates.get("trace", [])
        dynamic_mode = trace_list[-1] if trace_list else "Grounded generation from uploaded document."
        
        print_step(
            "[STEP 7: ANSWER GENERATION (Private KB)]",
            [
                f"Mode     : {dynamic_mode}",
                f"Citations: {len(citations)} source document(s)",
            ],
        )

    elif node_name == "generate_from_web":
        print_step(
            "[STEP 7: ANSWER GENERATION (External Web)]",
            [
                "Mode     : Synthesized response from external web evidence (KB was insufficient).",
                "Notice   : Marked as external vendor information.",
            ],
        )

    elif node_name == "direct_answer":
        print_step(
            "[STEP 2: DIRECT RESPONSE]",
            [
                "Mode     : Responding directly without vector retrieval.",
            ],
        )

    elif node_name == "insufficient":
        print_step(
            "[STOPPED: INSUFFICIENT EVIDENCE]",
            [
                "Status   : Retry limit reached with weak evidence.",
                "Action   : Advising user to contact internal IT help desk.",
            ],
        )
