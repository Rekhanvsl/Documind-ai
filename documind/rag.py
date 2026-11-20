"""Steps 6 + 7 of RAG: Generate an answer from retrieved chunks and cite pages."""
from dataclasses import dataclass

from .config import TOP_K
from .llm import LLMClient
from .retriever import BM25Index

QA_SYSTEM = (
    "You are DocuMind AI, an assistant that answers questions about a document.\n"
    "Rules:\n"
    "1. Use ONLY the provided context excerpts. Do not use outside knowledge.\n"
    "2. If the answer is not in the context, say clearly that the document "
    "does not seem to contain it.\n"
    "3. Cite the supporting pages in the text like (Page 12).\n"
    "4. Be concise and accurate."
)

SUMMARY_SYSTEM = (
    "You summarize documents accurately. Mention page numbers like (Page 4) "
    "for key points. Do not invent facts."
)


@dataclass
class Source:
    page: int
    snippet: str


@dataclass
class Answer:
    text: str
    sources: list[Source]


def answer_question(index: BM25Index, llm: LLMClient, question: str, k: int = TOP_K) -> Answer:
    hits = index.search(question, k=k)
    if not hits:
        return Answer(
            "I couldn't find anything related to that question in this document. "
            "Try different keywords.",
            [],
        )
    context = "\n\n".join(f"[Page {c.page}]\n{c.text}" for c, _ in hits)
    prompt = f"Context excerpts:\n{context}\n\nQuestion: {question}"
    text = llm.complete(QA_SYSTEM, prompt, max_tokens=900)

    sources, seen = [], set()
    for chunk, _ in hits:  # best match first, one entry per page
        if chunk.page not in seen:
            seen.add(chunk.page)
            sources.append(Source(chunk.page, chunk.text[:180] + "..."))
    return Answer(text, sources)


def summarize(chunks, llm: LLMClient, max_batches: int = 3, batch_chars: int = 300000) -> str:
    """Map-reduce summary: summarize parts, then summarize the summaries."""
    batches, cur, size = [], [], 0
    for c in chunks:
        line = f"[Page {c.page}] {c.text}"
        if size + len(line) > batch_chars and cur:
            batches.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line)
    if cur:
        batches.append("\n".join(cur))
    if not batches:
        return "This document has no text to summarize."

    sampled = len(batches) > max_batches
    if sampled:  # very long document: sample evenly to control cost
        step = len(batches) / max_batches
        batches = [batches[int(i * step)] for i in range(max_batches)]

    partials = [
        llm.complete(SUMMARY_SYSTEM, f"Summarize this part of a document:\n\n{b}", max_tokens=600)
        for b in batches
    ]
    if len(partials) == 1:
        result = partials[0]
    else:
        joined = "\n\n".join(f"Part {i}:\n{p}" for i, p in enumerate(partials, 1))
        result = llm.complete(
            SUMMARY_SYSTEM,
            "Combine these partial summaries into one clear summary with a short "
            f"overview followed by key points:\n\n{joined}",
            max_tokens=900,
        )
    if sampled:
        result += "\n\n(Note: this document was very long, so the summary samples sections evenly.)"
    return result
