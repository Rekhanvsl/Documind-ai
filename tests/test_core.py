from documind.pdf_processor import chunk_pages
from documind.retriever import BM25Index


def test_chunks_keep_page_numbers():
    pages = [(1, "Hello world. " * 200), (2, ""), (3, "Revenue grew by 18 percent.")]
    chunks = chunk_pages(pages, chunk_size=300, overlap=50)
    assert {c.page for c in chunks} == {1, 3}
    assert all(len(c.text) <= 300 for c in chunks)


def test_chunking_terminates_on_long_word():
    chunks = chunk_pages([(1, "x" * 5000)], chunk_size=500, overlap=100)
    assert len(chunks) > 5


def test_search_finds_right_page():
    pages = [
        (1, "The company leave policy gives employees 24 days of annual leave."),
        (2, "Revenue increased by 18 percent in 2025 compared with 2024."),
        (3, "The office is located in Hyderabad near the main road."),
    ]
    index = BM25Index(chunk_pages(pages))
    hits = index.search("What was the revenue growth in 2025?", k=2)
    assert hits[0][0].page == 2


def test_search_no_match_returns_empty():
    index = BM25Index(chunk_pages([(1, "apples and oranges")]))
    assert index.search("quantum chromodynamics") == []


def test_empty_index_is_safe():
    assert BM25Index([]).search("anything") == []
