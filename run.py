import sys
import os
import argparse
from pathlib import Path

import warnings

# Suppress known harmless warnings from langchain_nvidia_ai_endpoints
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

# Ensure UTF-8 output encoding on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.rag.workflow import agent_graph
from app.rag.state import AgentState
from app.services.audit import init_db, write_audit
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.rag.vector_stores import add_documents

configure_logging()
settings = get_settings()
init_db()

BANNER = """
===============================================================================
                     AGENTIC RAG PIPELINE - CLI RUNNER
===============================================================================
 Architecture: LangGraph State Graph + Qdrant Vector Store + NVIDIA NIM + Tavily
 Namespace:    {namespace}
 Qdrant Index: {index}
 Chat Model:   {chat_model}
 Embed Model:  {embed_model}
===============================================================================
 Pipeline Flow:
   [User Question]
          │
          ▼
   [1. Intent Router] ────(Direct)───► [Direct Answer] ───► [Done]
          │ (KB)
          ▼
   [2. Private KB Retrieval]
          │
          ▼
   [3. Grade KB Evidence] ───(Good)──► [Generate from KB] ──► [Done]
          │ (Weak)
          ▼
   [4. Web Search Fallback (Tavily)]
          │
          ▼
   [5. Grade Web Evidence] ──(Good)──► [Generate from Web] ─► [Done]
          │ (Weak & attempts < {max_retries})
          ▼
   [6. Query Rewrite & Retry Loop] ──► Back to Step 2
          │ (Max retries exceeded)
          ▼
   [Insufficient Evidence] ──────────► [Done]
===============================================================================
""".format(
    namespace=settings.qdrant_name_space or "default",
    index=settings.qdrant_index_name,
    chat_model=settings.nvidia_chat_model,
    embed_model=settings.nvidia_embedding_model,
    max_retries=settings.max_retries,
)


def print_step(title: str, details: list[str]) -> None:
    print(f"\n┌── {title}")
    for line in details:
        print(f"│   {line}")
    print("└──" + "─" * 60)


def run_pipeline(question: str) -> dict:
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

    for chunk in agent_graph.stream(initial, stream_mode="updates"):
        for node_name, updates in chunk.items():
            final_state.update(updates)

            if node_name == "route_question":
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
                print_step(
                    "[STEP 2: PRIVATE KB RETRIEVAL (Qdrant)]",
                    [
                        f"Search Query : \"{query_used}\"",
                        f"Namespace    : {settings.qdrant_name_space or 'default'}",
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
                print_step(
                    "[STEP 7: ANSWER GENERATION (Private KB)]",
                    [
                        "Mode     : Grounded generation from private company documents.",
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


def handle_ingest(file_path_str: str) -> None:
    path = Path(file_path_str.strip('\'"'))
    if not path.is_file():
        print(f"\n[Error] File not found: {path}\n")
        return

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        print(f"\n[Error] Unsupported file format '{suffix}'. Supported formats: {', '.join(sorted(SUPPORTED))}\n")
        return

    print("\n" + "=" * 79)
    print(f" DOCUMENT INGESTION PIPELINE: {path.name}")
    print("=" * 79)
    print(f"File Path    : {path.resolve()}")
    print(f"File Size    : {path.stat().st_size} bytes")
    print(f"Target Store : Qdrant ({settings.qdrant_index_name})")
    print(f"Namespace    : {settings.qdrant_name_space or 'default'}")
    print("-" * 79)

    try:
        print("[1/3] Loading document...")
        docs = load_file(path)
        print(f"      Loaded {len(docs)} raw document section(s).")

        print("[2/3] Chunking hierarchically (Parent 2000 chars, Child 400 chars)...")
        children, parents = chunk_documents(docs)
        print(f"      Generated {len(parents)} parent chunk(s) and {len(children)} searchable child chunk(s).")

        print("[3/3] Generating embeddings & indexing into Qdrant...")
        ids = add_documents(children, parents)
        print(f"      Successfully indexed {len(ids)} child vector(s) into '{settings.qdrant_index_name}'.")

        print("-" * 79)
        print(" [SUCCESS] Document is now live and retrievable in the Agentic RAG pipeline!")
        print("=" * 79 + "\n")
    except Exception as exc:
        print(f"\n[Error] Ingestion failed: {exc}\n")


def print_help() -> None:
    print("""
Commands:
  <question>          Ask any IT support question to run the Agentic RAG pipeline.
  ingest <filepath>   Index a document (.pdf, .txt, .md, .docx) into Qdrant.
  flow                Show the Agentic RAG pipeline architecture diagram.
  help                Show this help message.
  exit | quit         Exit the CLI.
""")


def interactive_cli() -> None:
    print(BANNER)
    print("Type an IT question to run the pipeline, 'ingest <path>' to add documents, or 'exit'.\n")

    while True:
        try:
            raw_input = input("Agentic-RAG> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not raw_input:
            continue

        cmd_lower = raw_input.lower()
        if cmd_lower in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        if cmd_lower == "help":
            print_help()
            continue

        if cmd_lower == "flow":
            print(BANNER)
            continue

        if cmd_lower.startswith("ingest "):
            file_arg = raw_input[7:].strip()
            handle_ingest(file_arg)
            continue

        # Run pipeline on input query
        try:
            run_pipeline(raw_input)
        except Exception as exc:
            print(f"\n[Error] Pipeline execution failed: {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG CLI Runner")
    parser.add_argument("query", nargs="?", help="Question to ask the Agentic RAG pipeline")
    parser.add_argument("--ingest", help="Path to document (.pdf, .txt, .md, .docx) to ingest into Qdrant")

    args = parser.parse_args()

    if args.ingest:
        handle_ingest(args.ingest)
    elif args.query:
        run_pipeline(args.query)
    else:
        interactive_cli()


if __name__ == "__main__":
    main()