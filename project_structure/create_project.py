from pathlib import Path

# Current working directory (or repository root)
root = Path(".")

# Directory structure for Agentic RAG
folders = [
    "app/api",
    "app/core",
    "app/rag",
    "app/services",
    "cli",
    "data",
    "uploads",
    "tests",
    "project_structure",
]

# Source files and configuration templates
files = [
    # Root configuration & entrypoints
    "run.py",
    "requirements.txt",
    "README.md",
    ".env.example",
    ".gitignore",

    # FastAPI application
    "app/__init__.py",
    "app/main.py",
    "app/api/__init__.py",
    "app/api/routes.py",

    # Core configuration & logging
    "app/core/__init__.py",
    "app/core/config.py",
    "app/core/logging.py",

    # RAG pipeline nodes & workflow
    "app/rag/__init__.py",
    "app/rag/state.py",
    "app/rag/llm.py",
    "app/rag/vector_stores.py",
    "app/rag/bm25.py",
    "app/rag/rank_fusion.py",
    "app/rag/reranking.py",
    "app/rag/cache_nodes.py",
    "app/rag/router.py",
    "app/rag/retrieval.py",
    "app/rag/grading.py",
    "app/rag/generation.py",
    "app/rag/workflow.py",

    # Services (Audit, Semantic Cache, Ingestion)
    "app/services/__init__.py",
    "app/services/audit.py",
    "app/services/cache.py",
    "app/services/ingestion.py",

    # CLI package
    "cli/__init__.py",
    "cli/banner.py",
    "cli/trace_printer.py",
    "cli/pipeline.py",
    "cli/ingest.py",
    "cli/repl.py",

    # Tests
    "tests/__init__.py",
    "tests/test.py",

    # Project structure tooling
    "project_structure/create_project.py",
]


def create_structure() -> None:
    # Create folders
    for folder in folders:
        (root / folder).mkdir(parents=True, exist_ok=True)

    # Create files
    for file in files:
        target = root / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)

    print(f"Project structure created successfully ({len(folders)} directories, {len(files)} files).")


if __name__ == "__main__":
    create_structure()
