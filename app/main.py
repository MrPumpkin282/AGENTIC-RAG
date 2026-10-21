from fastapi import FastAPI
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.audit import init_db

configure_logging()
settings = get_settings()
init_db()

app = FastAPI(title=settings.app_name, version="1.0.0")
app.include_router(router)

@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "status": "online",
        "mode": "agentic-rag-headless",
        "endpoints": ["/api/health", "/api/chat", "/api/ingest"],
    }