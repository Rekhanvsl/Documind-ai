"""Step 1 + 2 of RAG: Extract text from a PDF and split it into chunks."""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    page: int   # 1-based page number (used for citations)
    text: str


def extract_pages(path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for every page of the PDF."""
    import fitz  # PyMuPDF (imported here so tests don't need it)

    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, page.get_text("text")))
    return pages


def chunk_pages(pages, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Split each page into overlapping chunks. Chunks never cross pages,
    so every chunk has exactly one page number to cite."""
    chunks: list[Chunk] = []
    for page_no, text in pages:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):  # try not to cut a word in half
                space = text.rfind(" ", start + chunk_size // 2, end)
                if space != -1:
                    end = space
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(page_no, piece))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def process_pdf(path: str, chunk_size: int = 1000, overlap: int = 150):
    """Read a PDF and return (page_count, chunks)."""
    pages = extract_pages(path)
    return len(pages), chunk_pages(pages, chunk_size, overlap)
