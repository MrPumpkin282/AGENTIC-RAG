import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_tavily import TavilySearch
from app.core.config import get_settings
from app.rag.state import AgentState

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_CONTEXT_CHARS = settings.max_context_chars
MAX_QUERY_CHARS = 1000

_llm = None
_web_search = None


def llm():
    global _llm
    if _llm is None:
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is missing")
        if not settings.nvidia_chat_model:
            raise RuntimeError("NVIDIA_CHAT_MODEL is missing")
        _llm = ChatNVIDIA(
            model=settings.nvidia_chat_model,
            temperature=0,
            max_completion_tokens=4096,
            api_key=settings.nvidia_api_key,
        )
    return _llm

def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]

def doc_label(d):
    name = d.metadata.get("file_name", "unknown")
    page = d.metadata.get("page")
    return f"{name}, page {page + 1}" if page is not None else name

def build_context(docs):
    parts, total = [], 0
    for d in docs:
        file_name = d.metadata.get("file_name", "Document")
        page = d.metadata.get("page")
        page_str = f" (Page {page + 1})" if page is not None else ""
        header = f"[Source: {file_name}{page_str}]"
        text = f"{header}\n{d.page_content}"
        if total + len(text) > MAX_CONTEXT_CHARS:
            if parts:
                break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)
