# app/rag/rank_fusion.py
from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60, key=lambda d: d.metadata["child_id"]):
    """
    ranked_lists: list of ranked document lists (best first), e.g. [dense_hits, bm25_hits]
    k: RRF constant (60 is the standard default from the original paper)
    key: function to uniquely identify a document across lists
    Returns: list[Document] sorted by fused score, best first.
    """
    scores = defaultdict(float)
    doc_by_key = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            k_id = key(doc)
            doc_by_key[k_id] = doc
            scores[k_id] += 1.0 / (k + rank + 1)

    ordered_keys = sorted(scores, key=lambda kid: scores[kid], reverse=True)
    return [doc_by_key[kid] for kid in ordered_keys]
