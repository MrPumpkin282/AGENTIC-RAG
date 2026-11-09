import asyncio
import logging
import re
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.services.ingestion import ingest_document, SUPPORTED
from app.services.audit import write_audit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")
settings = get_settings()

MAGIC_SIGNATURES = {
    ".pdf": b"%PDF",
    ".docx": b"PK\x03\x04",
}


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks and unwanted characters."""
    base = Path(filename).name
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    if not cleaned or cleaned.startswith("."):
        suffix = Path(base).suffix
        cleaned = f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    return cleaned


def verify_file_content(path: Path, suffix: str) -> None:
    """Verify magic bytes and content integrity to prevent MIME spoofing."""
    if suffix in MAGIC_SIGNATURES:
        expected = MAGIC_SIGNATURES[suffix]
        with open(path, "rb") as f:
            header = f.read(len(expected))
        if header != expected:
            raise HTTPException(
                status_code=400,
                detail=f"File signature mismatch: file content does not match expected format for {suffix}",
            )
    elif suffix in {".txt", ".md"}:
        with open(path, "rb") as f:
            sample = f.read(4096)
        if b"\x00" in sample:
            raise HTTPException(
                status_code=400,
                detail=f"File contains invalid binary data for text format {suffix}",
            )


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
async def chat(payload: ChatRequest):
    try:
        result = await asyncio.to_thread(ask, payload.question)

        trace = result.get("trace", [])
        citations = result.get("citations", [])
        source_used = result.get("source_used", "none")

        logger.info(
            "Query processed: '%s' | Source: %s | Steps: %d | Citations: %d",
            payload.question[:80],
            source_used,
            len(trace),
            len(citations),
        )

        write_audit(
            payload.question,
            source_used,
            trace,
            citations,
        )

        return {
            "answer": result.get("answer", ""),
            "source_used": source_used,
            "trace": trace,
            "citations": citations,
            "rewritten_query": result.get("current_query", payload.question),
        }

    except Exception as exc:
        logger.exception("Chat processing failed for query: %s", payload.question)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    raw_filename = file.filename or ""
    clean_name = sanitize_filename(raw_filename)
    suffix = Path(clean_name).suffix.lower()

    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED))}",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / clean_name

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    total_bytes = 0

    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(65536):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size limit of {settings.max_upload_size_mb} MB",
                    )
                f.write(chunk)
    except HTTPException:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    if total_bytes == 0:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        verify_file_content(dest, suffix)
    except HTTPException:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise

    try:
        result = await asyncio.to_thread(ingest_document, dest)
        logger.info(
            "API Ingestion complete: %s | Chunks: %d | Parents: %d | IDs: %d",
            dest.name,
            result["chunks"],
            result["parents"],
            result["ids_created"],
        )
        return {
            "message": "Document indexed successfully",
            "file": result["file_name"],
            "file_size": result["file_size"],
            "chunks": result["chunks"],
            "parents": result["parents"],
            "ids_created": result["ids_created"],
        }
    except Exception as exc:
        logger.exception("Ingestion failed for %s", dest.name)
        raise HTTPException(status_code=500, detail=f"Ingestion processing failed: {exc}") from exc
