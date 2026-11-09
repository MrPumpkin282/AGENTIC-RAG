from app.core.config import get_settings
from cli.trace_printer import print_step, print_node_update
from cli.banner import build_banner
from langchain_core.documents import Document

settings = get_settings()


def test_cli_print_step(capsys):
    print_step("TEST STEP", ["detail line 1", "detail line 2"])
    captured = capsys.readouterr()
    assert "TEST STEP" in captured.out
    assert "detail line 1" in captured.out
    assert "detail line 2" in captured.out


def test_cli_print_node_updates(capsys):
    # check_cache miss
    print_node_update("check_cache", {"cache_hit": False}, {}, "question", settings)
    captured = capsys.readouterr()
    assert "SEMANTIC CACHE CHECK" in captured.out
    assert "MISS" in captured.out

    # check_cache hit
    print_node_update("check_cache", {"cache_hit": True}, {}, "question", settings)
    captured = capsys.readouterr()
    assert "HIT" in captured.out

    # route_question
    print_node_update("route_question", {"source_used": "kb"}, {}, "question", settings)
    captured = capsys.readouterr()
    assert "INTENT ROUTING" in captured.out
    assert "KB" in captured.out

    # retrieve_kb
    doc = Document(page_content="content", metadata={"file_name": "test.pdf"})
    print_node_update(
        "retrieve_kb",
        {"kb_docs": [doc], "trace": ["Private KB retrieval (hybrid (dense+BM25+RRF)) -> 1 chunks"]},
        {},
        "question",
        settings,
    )
    captured = capsys.readouterr()
    assert "PRIVATE KB RETRIEVAL" in captured.out
    assert "test.pdf" in captured.out


def test_cli_banner():
    banner = build_banner(settings)
    assert "AGENTIC RAG" in banner
    assert settings.qdrant_index_name in banner
