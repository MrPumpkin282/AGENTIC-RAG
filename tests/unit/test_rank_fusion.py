from langchain_core.documents import Document
from app.rag.rank_fusion import reciprocal_rank_fusion
from app.core.config import get_settings


def test_rrf_empty_lists():
    result = reciprocal_rank_fusion([])
    assert result == []

    result2 = reciprocal_rank_fusion([[], []])
    assert result2 == []


def test_rrf_single_list():
    d1 = Document(page_content="Doc 1", metadata={"id": 1})
    d2 = Document(page_content="Doc 2", metadata={"id": 2})

    fused = reciprocal_rank_fusion([[d1, d2]])
    assert len(fused) == 2
    assert fused[0].page_content == "Doc 1"
    assert fused[1].page_content == "Doc 2"


def test_rrf_combining_two_lists():
    d1 = Document(page_content="Common Doc", metadata={"id": "common"})
    d2 = Document(page_content="Dense only", metadata={"id": "dense"})
    d3 = Document(page_content="Sparse only", metadata={"id": "sparse"})

    dense_ranked = [d1, d2]
    sparse_ranked = [d3, d1]

    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], k=60)

    # d1 appears in both lists (ranks 0 and 1), so it should score highest:
    # 1/(60+1) + 1/(60+2) > 1/(60+1)
    assert len(fused) == 3
    assert fused[0].page_content == "Common Doc"
    assert "rrf_score" in fused[0].metadata
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]


def test_rrf_respects_settings_default():
    settings = get_settings()
    assert settings.rrf_k == 60

    d1 = Document(page_content="A")
    d2 = Document(page_content="B")
    fused = reciprocal_rank_fusion([[d1], [d2]])
    assert len(fused) == 2
