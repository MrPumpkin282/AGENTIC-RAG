from app.rag.llm import MAX_CONTEXT_CHARS, add_trace, build_context, llm
from app.rag.state import AgentState
from app.services.cache import store


def generate_from_kb(state: AgentState):
    docs = state.get("kb_docs", [])
    context = build_context(docs)
    
    # Extract file names from document metadata dynamically
    file_names = list(dict.fromkeys(d.metadata.get("file_name", "uploaded doc") for d in docs if d.metadata.get("file_name")))
    source_label = ", ".join(f'"{f}"' for f in file_names) if file_names else '"uploaded doc"'

    answer = llm().invoke(f"""
You are an expert AI knowledge assistant. Answer the user question accurately and comprehensively using ONLY the provided knowledge base context below.
If the question asks about multiple topics (for example both RAG and Operating Systems), address all topics covered in the context.
Be concise, well-structured, and factual. Do not fabricate facts outside the context.
Question: {state['question']}

Knowledge Base:
{context}
""").content
    
    citations, seen = [], set()
    for d in docs:
        name = d.metadata.get("file_name", "Knowledge Base")
        page = d.metadata.get("page")
        if (name, page) in seen:
            continue
        seen.add((name, page))
        title = f"{name} (page {page + 1})" if page is not None else name
        c = {"source": name, "title": title}
        if page is not None:
            c["page"] = page + 1
        citations.append(c)
        
    # Automatically store grounded response in semantic cache
    store(
        question=state["question"],
        answer=answer,
        source_used="kb",
        citations=citations
    )

    trace_message = f'Grounded generation from {source_label}'
    return {
        "answer": answer, 
        "source_used": "kb", 
        "citations": citations, 
        "trace": add_trace(state, trace_message)
    }

def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are an expert AI knowledge assistant. The private knowledge base was insufficient to answer this question.
Answer the user question using ONLY the external web evidence below. Note clearly in your answer that this information is synthesized from external web sources.
Question: {state['question']}

Web evidence:
{state.get('web_results', '')[:MAX_CONTEXT_CHARS]}
""").content
    citations = state.get("citations", [])
    
    store(
        question=state["question"],
        answer=answer,
        source_used="web",
        citations=citations
    )

    return {"answer": answer, "source_used": "web", "trace": add_trace(state, "Answer generation -> WEB SEARCH")}

def direct_answer(state: AgentState):
    answer = llm().invoke(f"Respond briefly and naturally to: {state['question']}").content
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response -> no retrieval")}

def insufficient(state: AgentState):
    return {
        "answer": "I could not find enough reliable evidence in the knowledge base or external search to answer your question with confidence. Please refine your question or provide more details.",
        "source_used": "none",
        "citations": [],
        "trace": add_trace(state, "Stopped -> insufficient reliable evidence"),
    }
