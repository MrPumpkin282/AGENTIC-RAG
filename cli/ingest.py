from pathlib import Path
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.rag.vector_stores import add_documents
from app.rag.bm25 import invalidate_bm25_index
from app.services.cache import clear_namespace


def handle_ingest(file_path_str: str, settings) -> None:
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
        invalidate_bm25_index()
        clear_namespace()
        print(f"      Successfully indexed {len(ids)} child vector(s) into '{settings.qdrant_index_name}'.")

        print("-" * 79)
        print(" [SUCCESS] Document is now live and retrievable in the Agentic RAG pipeline!")
        print("=" * 79 + "\n")
    except Exception as exc:
        print(f"\n[Error] Ingestion failed: {exc}\n")
