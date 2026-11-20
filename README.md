# DocuMind AI

A Windows desktop app that lets you **chat with your PDFs**. Pick a document, ask questions in plain English, and get answers with **clickable page references**.

Built with **Python, PySide6, PyMuPDF and a Retrieval-Augmented Generation (RAG) pipeline** powered by the free Google Gemini API (Claude is also supported).

<!-- Add a screenshot here: ![DocuMind AI](docs/screenshot.png) -->

## Features (Version 1)

- Add PDFs from your computer and preview pages inside the app
- Ask questions and get grounded answers with page citations
- Click a citation to jump to that page
- One-click document summarization (map-reduce for long documents)
- Document history, saved locally in SQLite

## How it works (RAG)

```
PDF -> extract text (PyMuPDF) -> chunk (per page) -> index (BM25)
Question -> retrieve top chunks -> LLM answers using ONLY those chunks -> cite pages
```

| Module | Job |
|---|---|
| `documind/pdf_processor.py` | Extract text and split into page-aware chunks |
| `documind/retriever.py` | BM25 search for relevant chunks |
| `documind/rag.py` | Prompting, answers, citations, summarization |
| `documind/llm.py` | AI provider wrapper (Gemini default, Claude optional) |
| `documind/database.py` | SQLite document history |
| `documind/ui/main_window.py` | PySide6 interface |

## Setup (Windows)

1. Install [Python 3.11+](https://www.python.org/downloads/) (tick **Add Python to PATH**).
2. Clone and enter the project:
   ```
   git clone https://github.com/YOUR-USERNAME/documind-ai.git
   cd documind-ai
   ```
3. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Get a **free** Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then copy `.env.example` to `.env` and paste the key after `GEMINI_API_KEY=`.
5. Run:
   ```
   python main.py
   ```

## Tests

```
pip install pytest
pytest
```

## Roadmap

- [x] **V1** PDF upload, viewer, chat with citations, summaries, history
- [ ] **V2** Compare documents, extract names/dates/amounts, export to PDF/TXT, DOCX/TXT support, collections
- [ ] **V3** Local models, OCR for scanned PDFs, semantic search (embeddings + FAISS), encrypted storage, Windows installer

## Privacy

Your PDFs stay on your computer. Only the text excerpts relevant to your question (or the document text, when summarizing) are sent to the AI API. On Gemini's free tier, Google may use submitted content to improve its products, so **do not use confidential documents** with the free tier. Never commit your `.env` file.

## License

MIT
