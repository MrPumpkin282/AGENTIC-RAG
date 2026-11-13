import argparse
import sys
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
from app.services.audit import init_db
from cli.banner import build_banner
from cli.ingest import handle_ingest
from cli.pipeline import run_pipeline
from cli.repl import interactive_cli

configure_logging()
settings = get_settings()
init_db()

BANNER = build_banner(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG CLI Runner")
    parser.add_argument("query", nargs="?", help="Question to ask the Agentic RAG pipeline")
    parser.add_argument("--ingest", help="Path to document (.pdf, .txt, .md, .docx) to ingest into Qdrant")

    args = parser.parse_args()

    if args.ingest:
        handle_ingest(args.ingest, settings)
    elif args.query:
        run_pipeline(args.query, settings)
    else:
        interactive_cli(BANNER, settings)


if __name__ == "__main__":
    main()