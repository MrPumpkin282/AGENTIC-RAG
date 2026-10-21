import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from app.core.config import get_settings

warnings.filterwarnings("ignore")
settings = get_settings()

TIMEOUT_SECONDS = 30

# Well-known models, tested even if the discovery step finds nothing
KNOWN_MODELS = [
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/nv-embedqa-mistral-7b-v2",
    "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "nvidia/nv-embed-v1",
    "snowflake/arctic-embed-l",
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
        print(f"Could not list models automatically ({type(e).__name__}). Using the known list only.")
    return found


def try_model(model_name):
    emb = NVIDIAEmbeddings(
        model=model_name,
        api_key=settings.nvidia_api_key,
        truncate="END",
    )
    query_vec = emb.embed_query("how do I connect to the VPN")
    doc_vecs = emb.embed_documents(["Open the VPN client and sign in with MFA."])
    return len(query_vec), len(doc_vecs[0])


def check(model_name):
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(try_model, model_name)
    try:
        q_dim, d_dim = future.result(timeout=TIMEOUT_SECONDS)
        seconds = round(time.time() - start, 1)
        if q_dim != d_dim:
            return "FAIL", None, seconds, f"query size {q_dim} != document size {d_dim}"
        return "OK", q_dim, seconds, ""
    except FutureTimeout:
        return "SLOW", None, TIMEOUT_SECONDS, f"no answer in {TIMEOUT_SECONDS}s"
    except Exception as e:
        return "FAIL", None, round(time.time() - start, 1), str(e).splitlines()[0][:90]
    finally:
        pool.shutdown(wait=False)


if __name__ == "__main__":
    if not settings.nvidia_api_key:
        raise SystemExit("NVIDIA_API_KEY is missing in .env")

    print("Current model in .env:", settings.nvidia_embedding_model)
    print("Finding embedding models for your key...\n")

    models = []
    for name in [settings.nvidia_embedding_model] + discover_models() + KNOWN_MODELS:
        if name and name not in models:
            models.append(name)

    results = []
    for name in models:
        print(f"Testing {name} ...", flush=True)
        status, dim, seconds, note = check(name)
        print(f"   {status}  {seconds}s  {note}")
        results.append((name, status, dim, seconds, note))

    print("\n========== SUMMARY ==========")
    print(f"{'MODEL':50} {'STATUS':7} {'DIM':>5} {'SEC':>6}")
    for name, status, dim, seconds, note in results:
        print(f"{name:50} {status:7} {str(dim or '-'):>5} {seconds:>6}")

    working = sorted([r for r in results if r[1] == "OK"], key=lambda r: r[3])
    print()
    if working:
        best = working[0]
        print(f"Fastest working model: {best[0]} (dimension {best[2]}, {best[3]}s)")
        print("\nPut this in .env:")
        print(f"NVIDIA_EMBEDDING_MODEL={best[0]}")
        print(f"QDRANT_INDEX_NAME=STORAGE_{best[2]}   # new name, because the vector size changes")
    else:
        print("No model worked. Check NVIDIA_API_KEY and your internet connection.")