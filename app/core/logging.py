import logging
import warnings

from app.core.config import get_settings

# Suppress known harmless warnings from langchain_nvidia_ai_endpoints
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")


def configure_logging():
    warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.WARNING)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | [%(levelname)s] | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
        force=True,
    )

    # Explicitly silence noisy HTTP and client library loggers
    for logger_name in (
        "httpx",
        "httpcore",
        "urllib3",
        "qdrant_client",
        "langchain",
        "langchain_core",
        "langchain_community",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
