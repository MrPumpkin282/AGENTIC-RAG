"""Embedding model connectivity and dimension compatibility checker for NVIDIA NIM."""

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from app.core.config import get_settings

warnings.filterwarnings("ignore")
settings = get_settings()

TIMEOUT_SECONDS = 30

KNOWN_MODELS = [
    "nvidia/nemotron-3.5-embed-1b",
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/nv-embedqa-mistral-7b-v2",
    "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "baai/bge-m3",
]


def discover_models():
    found = []
    try:
        for m in NVIDIAEmbeddings.get_available_models(api_key=settings.nvidia_api_key):
            name = getattr(m, "id", None) or getattr(m, "model_name", None)
            if name and "embed" in name.lower():
                found.append(name)
    except Exception as e:
        print(f"Automatic model discovery unavailable ({type(e).__name__}). Using fallback list.")
    return found


def try_model(model_name: str):
    emb = NVIDIAEmbeddings(
        model=model_name,
        api_key=settings.nvidia_api_key,
        truncate="END",
    )
    query_vec = emb.embed_query("dimension probe")
    doc_vecs = emb.embed_documents(["Verification chunk for embedding dimensions."])
    return len(query_vec), len(doc_vecs[0])


def check_model(model_name: str):
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(try_model, model_name)
    try:
        q_dim, d_dim = future.result(timeout=TIMEOUT_SECONDS)
        seconds = round(time.time() - start, 1)
        if q_dim != d_dim:
            return "FAIL", None, seconds, f"query size {q_dim} != doc size {d_dim}"
        return "OK", q_dim, seconds, ""
    except FutureTimeout:
        return "SLOW", None, TIMEOUT_SECONDS, f"Timeout after {TIMEOUT_SECONDS}s"
    except Exception as exc:
        return "FAIL", None, round(time.time() - start, 1), str(exc).splitlines()[0][:90]
    finally:
        pool.shutdown(wait=False)


def run_checks():
    if not settings.nvidia_api_key:
        print("ERROR: NVIDIA_API_KEY is not set.")
        return

    print("Configured embedding model in settings:", settings.nvidia_embedding_model)
    print("Testing model availability with active API key...\n")

    candidates = []
    for candidate in [settings.nvidia_embedding_model] + discover_models() + KNOWN_MODELS:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    results = []
    for model_name in candidates:
        print(f"Testing {model_name} ...", end=" ", flush=True)
        status, dim, sec, note = check_model(model_name)
        print(f"[{status}] {sec}s {note}")
        results.append((model_name, status, dim, sec, note))

    print("\n" + "=" * 70)
    print(f"{'MODEL':48} {'STATUS':8} {'DIM':>6} {'SEC':>6}")
    print("=" * 70)
    for model_name, status, dim, sec, note in results:
        print(f"{model_name:48} {status:8} {dim or '-'!s:>6} {sec:>6}")
    print("=" * 70)


if __name__ == "__main__":
    run_checks()
