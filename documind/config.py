"""Central settings for DocuMind AI."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_NAME = "DocuMind AI"

# All app data (history database) lives in your home folder, e.g. C:\Users\you\.documind
DATA_DIR = Path.home() / ".documind"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "documind.db"

# --- AI provider ---
PROVIDER = os.getenv("DOCUMIND_PROVIDER", "gemini").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.getenv("DOCUMIND_MODEL", "claude-sonnet-5-5").strip()

# --- RAG settings ---
CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 6             # how many chunks we send to the LLM
