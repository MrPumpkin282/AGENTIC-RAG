def build_banner(settings) -> str:
    return """
===============================================================================
                     AGENTIC RAG PIPELINE - CLI RUNNER
===============================================================================
 Architecture: LangGraph State Graph + Qdrant Vector Store + NVIDIA NIM + Tavily
 Namespace:    {namespace}
 Qdrant Index: {index}
 Chat Model:   {chat_model}
 Embed Model:  {embed_model}
===============================================================================
 Pipeline Flow:
   [User Question]
          │
          ▼
   [0. Semantic Cache Check] ──(HIT)──► [Cached Answer] ──► [Done]
          │ (MISS)
          ▼
   [1. Intent Router] ────(Direct)───► [Direct Answer] ───► [Done]
          │ (KB)
          ▼
   [2. Hybrid Retrieval (Dense + BM25 + RRF + Rerank)]
          │
          ▼
   [3. Grade KB Evidence] ───(Good)──► [Generate from KB] ──► [Done]
          │ (Weak)
          ▼
   [4. Web Search Fallback (Tavily)]
          │
          ▼
   [5. Grade Web Evidence] ──(Good)──► [Generate from Web] ─► [Done]
          │ (Weak & attempts < {max_retries})
          ▼
   [6. Query Rewrite & Retry Loop] ──► Back to Step 2
          │ (Max retries exceeded)
          ▼
   [Insufficient Evidence] ──────────► [Done]
===============================================================================
""".format(
        namespace=settings.qdrant_name_space or "default",
        index=settings.qdrant_index_name,
        chat_model=settings.nvidia_chat_model,
        embed_model=settings.nvidia_embedding_model,
        max_retries=settings.max_retries,
    )
