"""SQLite storage: remembers documents, their chunks and saved summaries."""
import sqlite3
from datetime import datetime

from .pdf_processor import Chunk


class Database:
    def __init__(self, path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                added_at TEXT NOT NULL,
                summary TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def add_document(self, path: str, name: str, page_count: int, chunks) -> int:
        self.conn.execute("DELETE FROM documents WHERE path = ?", (path,))
        cur = self.conn.execute(
            "INSERT INTO documents (path, name, page_count, added_at) VALUES (?,?,?,?)",
            (path, name, page_count, datetime.now().isoformat(timespec="seconds")),
        )
        doc_id = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO chunks (doc_id, page, text) VALUES (?,?,?)",
            [(doc_id, c.page, c.text) for c in chunks],
        )
        self.conn.commit()
        return doc_id

    def list_documents(self):
        return self.conn.execute(
            "SELECT id, name, page_count FROM documents ORDER BY added_at DESC, id DESC"
        ).fetchall()

    def get_document(self, doc_id: int):
        return self.conn.execute(
            "SELECT id, path, name, page_count, summary FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()

    def get_chunks(self, doc_id: int) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT page, text FROM chunks WHERE doc_id = ? ORDER BY id", (doc_id,)
        ).fetchall()
        return [Chunk(p, t) for p, t in rows]

    def save_summary(self, doc_id: int, summary: str):
        self.conn.execute("UPDATE documents SET summary = ? WHERE id = ?", (summary, doc_id))
        self.conn.commit()

    def delete_document(self, doc_id: int):
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
