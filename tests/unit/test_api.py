from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

client = TestClient(app)
settings = get_settings()


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_api_chat_validation_error():
    # Question too short (< 2 characters)
    response = client.post("/api/chat", json={"question": "a"})
    assert response.status_code == 422


@patch("app.api.routes.ask")
@patch("app.api.routes.write_audit")
def test_api_chat_success(mock_audit, mock_ask):
    mock_ask.return_value = {
        "answer": "This is a test response.",
        "source_used": "kb",
        "trace": ["cache_miss", "router_kb"],
        "citations": [{"title": "Test Doc", "url": "https://example.com"}],
        "current_query": "test query",
    }

    response = client.post("/api/chat", json={"question": "What is hybrid search?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a test response."
    assert data["source_used"] == "kb"
    assert len(data["trace"]) == 2
    mock_audit.assert_called_once()


def test_api_ingest_auth_failure():
    response = client.post(
        "/api/ingest",
        headers={"x-admin-key": "wrong-key"},
        files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
    )
    assert response.status_code == 401
    assert "Invalid admin key" in response.json()["detail"]


def test_api_ingest_unsupported_format():
    response = client.post(
        "/api/ingest",
        headers={"x-admin-key": settings.admin_api_key},
        files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_api_ingest_empty_file():
    response = client.post(
        "/api/ingest",
        headers={"x-admin-key": settings.admin_api_key},
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_api_ingest_magic_signature_mismatch():
    # File named .pdf but does not start with %PDF
    response = client.post(
        "/api/ingest",
        headers={"x-admin-key": settings.admin_api_key},
        files={"file": ("spoofed.pdf", b"This is plain text pretending to be PDF", "application/pdf")},
    )
    assert response.status_code == 400
    assert "signature mismatch" in response.json()["detail"]


@patch("app.api.routes.ingest_document")
def test_api_ingest_success(mock_ingest):
    mock_ingest.return_value = {
        "file_name": "valid.pdf",
        "file_path": "/tmp/valid.pdf",
        "file_size": 100,
        "parents": 2,
        "chunks": 6,
        "ids_created": 6,
    }

    valid_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    response = client.post(
        "/api/ingest",
        headers={"x-admin-key": settings.admin_api_key},
        files={"file": ("valid.pdf", valid_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Document indexed successfully"
    assert data["chunks"] == 6
    assert data["parents"] == 2
