"""
Lightweight unit tests that don't require a GROQ_API_KEY or network access --
they exercise the pure-Python routing logic and the chunking function.

Run with:  pytest tests/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.nodes import decide_to_generate, decide_after_hallucination_check
from ingest import _chunk_text


def test_decide_to_generate_routes_to_generate_when_relevant_docs_exist():
    state = {"graded_documents": [{"content": "x"}], "retry_count": 0, "max_retries": 2}
    assert decide_to_generate(state) == "generate"


def test_decide_to_generate_routes_to_rewrite_when_retries_remain():
    state = {"graded_documents": [], "retry_count": 0, "max_retries": 2}
    assert decide_to_generate(state) == "rewrite"


def test_decide_to_generate_gives_up_when_retries_exhausted_and_no_web_search():
    state = {"graded_documents": [], "retry_count": 2, "max_retries": 2}
    # Note: whether this is "web_search" or "give_up" depends on settings
    # (TAVILY_API_KEY / ENABLE_WEB_SEARCH_FALLBACK), both are valid terminal
    # non-retry routes for this state.
    assert decide_to_generate(state) in ("web_search", "give_up")


def test_decide_after_hallucination_check_ends_when_grounded():
    state = {"grounded": True, "regenerated": False}
    assert decide_after_hallucination_check(state) == "end"


def test_decide_after_hallucination_check_regenerates_once():
    state = {"grounded": False, "regenerated": False}
    assert decide_after_hallucination_check(state) == "regenerate"


def test_decide_after_hallucination_check_stops_after_one_regeneration():
    state = {"grounded": False, "regenerated": True}
    assert decide_after_hallucination_check(state) == "end"


def test_chunking_produces_multiple_chunks_with_source_metadata():
    text = "# Title\n\n" + ("This is a sentence about FastAPI. " * 200)
    chunks = _chunk_text(text, source="test.md")
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "test.md" for c in chunks)
    assert all(len(c.page_content) <= 1000 for c in chunks)  # chunk_size + slack
