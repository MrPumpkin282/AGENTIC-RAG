from collections import defaultdict
from app.core.config import get_settings

settings = get_settings()


def reciprocal_rank_fusion(
    ranked_lists: list[list],
    k: int | None = None,
    key=lambda d: d.metadata.get("child_id", getattr(d, "id", None) or d.page_content),
):
    """
    ranked_lists: list of ranked document lists (best first), e.g. [dense_hits, bm25_hits]
    k: RRF constant (centralized in Settings.rrf_k, default 60)
    key: function to uniquely identify a document across lists
    Returns: list[Document] sorted by fused score, best first.
    """
    rrf_k = k if k is not None else settings.rrf_k
    scores = defaultdict(float)
    doc_by_key = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            k_id = key(doc)
            doc_by_key[k_id] = doc
            scores[k_id] += 1.0 / (rrf_k + rank + 1)

    ordered_keys = sorted(scores, key=lambda kid: scores[kid], reverse=True)
    results = []
    for kid in ordered_keys:
        doc = doc_by_key[kid]
        doc.metadata["rrf_score"] = scores[kid]
        results.append(doc)
    return results
