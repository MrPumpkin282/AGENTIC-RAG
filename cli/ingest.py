from pathlib import Path
from app.services.ingestion import ingest_document, SUPPORTED


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
        print("[1/3] Loading & parsing document...")
        print("[2/3] Chunking hierarchically (Parent 2000 chars, Child 400 chars)...")
        print("[3/3] Generating embeddings, indexing Qdrant, syncing BM25 & invalidating cache...")

        result = ingest_document(path)

        print(f"      Generated {result['parents']} parent chunk(s) and {result['chunks']} searchable child chunk(s).")
        print(f"      Successfully indexed {result['ids_created']} child vector(s) into '{settings.qdrant_index_name}'.")
        print(f"      Hybrid lexical index invalidated & semantic cache purged.")

        print("-" * 79)
        print(" [SUCCESS] Document is now live and retrievable in the Agentic RAG pipeline!")
        print("=" * 79 + "\n")
    except Exception as exc:
        print(f"\n[Error] Ingestion failed: {exc}\n")
