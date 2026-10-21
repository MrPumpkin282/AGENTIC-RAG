import logging
import warnings

# Suppress known harmless warnings from langchain_nvidia_ai_endpoints
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

def configure_logging():
    warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | [%(levelname)s] | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )