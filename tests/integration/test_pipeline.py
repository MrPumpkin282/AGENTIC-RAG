from pathlib import Path
from unittest.mock import patch, MagicMock
from app.services.ingestion import load_file, chunk_documents, resolve_parents
from app.rag.rank_fusion import reciprocal_rank_fusion
from app.rag.grading import after_kb, after_web
from app.rag.cache_nodes import check_cache, route_after_cache
from app.rag.state import AgentState

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
TEST_PDF = FIXTURES_DIR / "test.pdf"


def test_fixture_files_exist():
    assert SAMPLE_PDF.exists(), f"Missing fixture: {SAMPLE_PDF}"
    assert TEST_PDF.exists(), f"Missing fixture: {TEST_PDF}"


def test_load_sample_pdf_fixture():
    docs = load_file(SAMPLE_PDF)
    assert len(docs) > 0
    assert any(len(d.page_content.strip()) > 0 for d in docs)
    # Check page metadata
    assert "page" in docs[0].metadata
    assert docs[0].metadata["source"] == str(SAMPLE_PDF)


def test_load_test_pdf_fixture():
    docs = load_file(TEST_PDF)
    assert len(docs) > 0
    assert any(len(d.page_content.strip()) > 0 for d in docs)


def test_hierarchical_chunking_sample_pdf():
    raw_docs = load_file(SAMPLE_PDF)
    children, parents = chunk_documents(raw_docs)

    assert len(parents) > 0
    assert len(children) > len(parents)

    # Check parent structure
    first_parent_id, first_parent = next(iter(parents.items()))
    assert first_parent.metadata["chunk_level"] == "parent"
    assert first_parent.metadata["parent_id"] == first_parent_id
    assert first_parent.metadata["file_name"] == "sample.pdf"

    # Check child structure & metadata injection header
    first_child = children[0]
    assert first_child.metadata["chunk_level"] == "child"
    assert first_child.metadata["parent_id"] in parents
    assert "[File: sample.pdf" in first_child.page_content

    # Check parent resolution
    resolved = resolve_parents(children[:5], parents)
    assert len(resolved) > 0
    assert len(resolved) <= len(parents)
    # Deduplication check
    resolved_ids = [d.metadata["parent_id"] for d in resolved]
    assert len(resolved_ids) == len(set(resolved_ids))


def test_retrieval_and_rank_fusion_deterministic():
    raw_docs = load_file(SAMPLE_PDF)
    children, parents = chunk_documents(raw_docs[:2])

    if len(children) >= 4:
        # Simulate two ranked lists
        dense_results = [children[0], children[1], children[2]]
        bm25_results = [children[1], children[3], children[0]]

        fused = reciprocal_rank_fusion([dense_results, bm25_results], k=60)
        assert len(fused) == 4
        # children[1] appears in rank 0 (sparse) and rank 1 (dense) -> highest score
        assert fused[0].metadata["child_id"] == children[1].metadata["child_id"]

        # Resolve parents
        final_parents = resolve_parents(fused, parents)
        assert len(final_parents) > 0


def test_grading_transitions():
    # KB Grade good -> generate_from_kb
    state_kb_good: AgentState = {"kb_grade": "good"}  # type: ignore
    assert after_kb(state_kb_good) == "generate_from_kb"

    # KB Grade weak -> search_web
    state_kb_weak: AgentState = {"kb_grade": "weak"}  # type: ignore
    assert after_kb(state_kb_weak) == "search_web"

    # Web Grade good -> generate_from_web
    state_web_good: AgentState = {"web_grade": "good"}  # type: ignore
    assert after_web(state_web_good) == "generate_from_web"

    # Web Grade weak with retry < max_retries -> rewrite_query
    state_retry: AgentState = {"web_grade": "weak", "retry_count": 0}  # type: ignore
    assert after_web(state_retry) == "rewrite_query"

    # Web Grade weak with retry >= max_retries -> insufficient
    state_exhausted: AgentState = {"web_grade": "weak", "retry_count": 2}  # type: ignore
    assert after_web(state_exhausted) == "insufficient"


def test_cache_node_fast_path():
    greeting_state: AgentState = {
        "question": "hello there!",
        "current_query": "hello there!",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    result = check_cache(greeting_state)
    assert result.get("cache_hit") is True
    assert "answer" in result
    assert result.get("source_used") in ("direct", "cache")

    # Verify route_after_cache returns END on cache hit
    full_state = {**greeting_state, **result}
    from langgraph.graph import END
    assert route_after_cache(full_state) == END
