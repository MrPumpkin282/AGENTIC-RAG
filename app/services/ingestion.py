import hashlib
import logging
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from docx import Document as DocxDocument

from app.rag.vector_stores import add_documents
from app.rag.bm25 import invalidate_bm25_index
from app.services.cache import clear_namespace

logger = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}

PARENT_SIZE, PARENT_OVERLAP = 2000, 200
CHILD_SIZE, CHILD_OVERLAP = 400, 60


def load_file(path: Path | str) -> list[Document]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), autodetect_encoding=True).load()
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Document(page_content=text, metadata={"source": str(path)})]
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_documents(docs: list[Document]) -> tuple[list[Document], dict[str, Document]]:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_SIZE, chunk_overlap=PARENT_OVERLAP, add_start_index=True
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE, chunk_overlap=CHILD_OVERLAP, add_start_index=True
    )

    parents: dict[str, Document] = {}
    children: list[Document] = []

    for parent_idx, parent in enumerate(parent_splitter.split_documents(list(docs))):
        source = parent.metadata.get("source", "")
        file_name = Path(source).name
        page = parent.metadata.get("page")  # PDFs only, 0-indexed
        parent_start = parent.metadata.get("start_index", 0)
        parent_id = f"{hashlib.md5(source.encode()).hexdigest()[:8]}-p{parent_idx}"

        # --- parent metadata ---
        parent.metadata.update({
            "parent_id": parent_id,
            "file_name": file_name,
            "chunk_level": "parent",
        })
        parents[parent_id] = parent

        # --- metadata injection header (embedded together with the child text) ---
        header_parts = [f"File: {file_name}"]
        if page is not None:
            header_parts.append(f"Page: {page + 1}")
        header = "[" + " | ".join(header_parts) + "]\n"

        # --- children ---
        for child_idx, child in enumerate(child_splitter.split_documents([parent])):
            if not child.page_content.strip():
                continue
            child.metadata.update({
                "child_id": f"{parent_id}-c{child_idx}",
                "child_index": child_idx,
                "chunk_level": "child",
                "start_index": parent_start + child.metadata.get("start_index", 0),
            })
            child.page_content = header + child.page_content
            children.append(child)

    return children, parents


def resolve_parents(child_hits: list[Document], parents: dict[str, Document]) -> list[Document]:
    seen, result = set(), []
    for c in child_hits:
        pid = c.metadata.get("parent_id")
        if pid and pid not in seen and pid in parents:
            seen.add(pid)
            result.append(parents[pid])
    return result


def ingest_document(file_path: Path | str) -> dict[str, Any]:
    """Unified ingestion pipeline:
    1. Loads and parses the document (PDF, DOCX, TXT, MD)
    2. Hierarchically chunks into parents (2000 chars) and children (400 chars)
    3. Generates dense embeddings & stores vectors in Qdrant
    4. Invalidates in-memory BM25 lexical index to trigger re-sync
    5. Invalidates semantic cache in Qdrant to purge stale cached responses
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED))}")

    logger.info("Starting ingestion for %s (size: %d bytes)", path.name, path.stat().st_size)

    # 1. Load document
    docs = load_file(path)

    # 2. Hierarchical chunking
    children, parents = chunk_documents(docs)

    # 3. Dense embeddings & vector storage
    ids = add_documents(children, parents)

    # 4. Lexical BM25 index invalidation (synchronize hybrid indices)
    try:
        invalidate_bm25_index()
    except Exception as exc:
        logger.warning("BM25 invalidation warning: %s", exc)

    # 5. Semantic cache invalidation (automate cache invalidation upon ingestion)
    try:
        clear_namespace()
    except Exception as exc:
        logger.warning("Semantic cache clear warning: %s", exc)

    logger.info("Ingestion complete for %s: %d parents, %d children", path.name, len(parents), len(children))

    return {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "file_size": path.stat().st_size,
        "parents": len(parents),
        "chunks": len(children),
        "ids_created": len(ids),
    }
