from cli.ingest import handle_ingest
from cli.pipeline import run_pipeline


def print_help() -> None:
    print("""
Commands:
  <question>          Ask any IT support question to run the Agentic RAG pipeline.
  ingest <filepath>   Index a document (.pdf, .txt, .md, .docx) into Qdrant.
  clear-cache         Clear the semantic cache for the current namespace.
  flow                Show the Agentic RAG pipeline architecture diagram.
  help                Show this help message.
  exit | quit         Exit the CLI.
""")


def interactive_cli(banner: str, settings) -> None:
    print(banner)
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
            print(banner)
            continue

        if cmd_lower == "clear-cache":
            try:
                from app.services.cache import clear_namespace
                clear_namespace()
                print("\n[OK] Semantic cache cleared for current namespace.\n")
            except Exception as exc:
                print(f"\n[Error] Failed to clear cache: {exc}\n")
            continue

        if cmd_lower.startswith("ingest "):
            file_arg = raw_input[7:].strip()
            handle_ingest(file_arg, settings)
            continue

        # Run pipeline on input query
        try:
            run_pipeline(raw_input, settings)
        except Exception as exc:
            print(f"\n[Error] Pipeline execution failed: {exc}\n")
