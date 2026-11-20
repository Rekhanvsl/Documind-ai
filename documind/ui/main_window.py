"""The DocuMind AI desktop window (PySide6)."""
import html
import os
import re

import fitz  # PyMuPDF - used here to render page images
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSplitter, QTextBrowser,
    QVBoxLayout, QWidget,
)

from .. import config, rag
from ..database import Database
from ..llm import LLMClient
from ..pdf_processor import process_pdf
from ..retriever import BM25Index

STYLE = """
QMainWindow, QWidget { font-size: 13px; }
QPushButton { padding: 6px 12px; border-radius: 4px; background: #2d6cdf; color: white; }
QPushButton:disabled { background: #9aa7bd; }
QPushButton#secondary { background: #e6e9ef; color: #222; }
QPushButton#secondary:disabled { color: #999; }
QLineEdit { padding: 7px; border: 1px solid #c5cad3; border-radius: 4px; }
QTextBrowser { border: 1px solid #d5d9e0; border-radius: 4px; background: #fafbfc; }
QListWidget { border: 1px solid #d5d9e0; border-radius: 4px; }
"""


# ---------- run slow work (PDF reading, AI calls) off the UI thread ----------
class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.finished.emit(self.fn(*self.args))
        except Exception as e:  # shown to the user, not a crash
            self.signals.error.emit(str(e))


def to_html(text: str) -> str:
    """Escape text, then support **bold** and line breaks."""
    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return safe.replace("\n", "<br>")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.resize(1400, 850)
        self.setStyleSheet(STYLE)

        self.db = Database(config.DB_PATH)
        self.llm = LLMClient()
        self.pool = QThreadPool.globalInstance()

        self.doc_id = None
        self.index = None
        self.chunks = []
        self.fitz_doc = None
        self.page_index = 0
        self.zoom = 1.4
        self._workers = set()  # keep references so workers aren't garbage-collected

        self._build_ui()
        self._refresh_history()
        self.statusBar().showMessage("Add a PDF to begin.")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # left: document history
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>Documents</b>"))
        self.history = QListWidget()
        self.history.itemClicked.connect(lambda item: self.open_document(item.data(Qt.UserRole)))
        lv.addWidget(self.history)
        self.btn_add = QPushButton("+ Add PDF")
        self.btn_add.clicked.connect(self.add_pdf)
        self.btn_remove = QPushButton("Remove selected")
        self.btn_remove.setObjectName("secondary")
        self.btn_remove.clicked.connect(self.remove_selected)
        lv.addWidget(self.btn_add)
        lv.addWidget(self.btn_remove)

        # center: PDF viewer
        center = QWidget()
        cv = QVBoxLayout(center)
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        self.btn_zout = QPushButton("−")
        self.btn_zin = QPushButton("+")
        for b in (self.btn_prev, self.btn_next, self.btn_zout, self.btn_zin):
            b.setObjectName("secondary")
            b.setFixedWidth(40)
        self.page_label = QLabel("No document")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.btn_prev.clicked.connect(lambda: self.show_page(self.page_index - 1))
        self.btn_next.clicked.connect(lambda: self.show_page(self.page_index + 1))
        self.btn_zout.clicked.connect(lambda: self.set_zoom(self.zoom - 0.2))
        self.btn_zin.clicked.connect(lambda: self.set_zoom(self.zoom + 0.2))
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.page_label, 1)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.btn_zout)
        nav.addWidget(self.btn_zin)
        cv.addLayout(nav)
        self.page_image = QLabel("Open a PDF to preview it here.")
        self.page_image.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setWidget(self.page_image)
        cv.addWidget(scroll, 1)

        # right: chat
        right = QWidget()
        rv = QVBoxLayout(right)
        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Ask DocuMind</b>"), 1)
        self.btn_summary = QPushButton("Summarize document")
        self.btn_summary.clicked.connect(self.summarize_document)
        top.addWidget(self.btn_summary)
        rv.addLayout(top)
        self.chat = QTextBrowser()
        self.chat.setOpenLinks(False)
        self.chat.anchorClicked.connect(self._on_link)
        rv.addWidget(self.chat, 1)
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask a question about this document...")
        self.input.returnPressed.connect(self.ask)
        self.btn_ask = QPushButton("Ask")
        self.btn_ask.clicked.connect(self.ask)
        row.addWidget(self.input, 1)
        row.addWidget(self.btn_ask)
        rv.addLayout(row)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([240, 620, 540])
        self.setCentralWidget(splitter)
        self._update_controls()

    def _update_controls(self):
        has_doc = self.doc_id is not None
        busy = getattr(self, "busy", False)
        for w in (self.btn_ask, self.btn_summary, self.input):
            w.setEnabled(has_doc and not busy)
        for w in (self.btn_add, self.btn_remove):
            w.setEnabled(not busy)
        for w in (self.btn_prev, self.btn_next, self.btn_zin, self.btn_zout):
            w.setEnabled(has_doc)

    def _set_busy(self, busy: bool, message: str = ""):
        self.busy = busy
        self.statusBar().showMessage(message)
        self._update_controls()

    # ---------------------------------------------------------- background
    def _run(self, fn, args, on_done, busy_message):
        self._set_busy(True, busy_message)
        worker = Worker(fn, *args)
        self._workers.add(worker)

        def done(result):
            self._workers.discard(worker)
            self._set_busy(False, "Ready.")
            on_done(result)

        def failed(msg):
            self._workers.discard(worker)
            self._set_busy(False, "Something went wrong.")
            QMessageBox.warning(self, config.APP_NAME, msg)

        worker.signals.finished.connect(done)
        worker.signals.error.connect(failed)
        self.pool.start(worker)

    # ------------------------------------------------------------ documents
    def _refresh_history(self, select_id=None):
        self.history.clear()
        for doc_id, name, pages in self.db.list_documents():
            item = QListWidgetItem(f"{name}  ({pages} pages)")
            item.setData(Qt.UserRole, doc_id)
            self.history.addItem(item)
            if doc_id == select_id:
                self.history.setCurrentItem(item)

    def add_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a PDF", "", "PDF files (*.pdf)")
        if not path:
            return
        self._run(process_pdf, (path, config.CHUNK_SIZE, config.CHUNK_OVERLAP),
                  lambda r: self._on_processed(path, r), "Reading and indexing the PDF...")

    def _on_processed(self, path, result):
        page_count, chunks = result
        if not chunks:
            QMessageBox.warning(
                self, config.APP_NAME,
                "No text could be extracted. This looks like a scanned PDF.\n"
                "OCR support is planned for Version 3.")
            return
        doc_id = self.db.add_document(path, os.path.basename(path), page_count, chunks)
        self._refresh_history(select_id=doc_id)
        self.open_document(doc_id)

    def open_document(self, doc_id):
        row = self.db.get_document(doc_id)
        if not row:
            return
        _, path, name, _, _ = row
        if not os.path.exists(path):
            QMessageBox.warning(self, config.APP_NAME,
                                f"The file was moved or deleted:\n{path}")
            return
        try:
            if self.fitz_doc:
                self.fitz_doc.close()
            self.fitz_doc = fitz.open(path)
        except Exception as e:
            QMessageBox.warning(self, config.APP_NAME, f"Could not open the PDF:\n{e}")
            return
        self.doc_id = doc_id
        self.chunks = self.db.get_chunks(doc_id)
        self.index = BM25Index(self.chunks)
        self.chat.clear()
        self._say("DocuMind", f"<b>{html.escape(name)}</b> is indexed "
                              f"({len(self.chunks)} sections). Ask me anything about it.")
        self.show_page(0)
        self._update_controls()

    def remove_selected(self):
        item = self.history.currentItem()
        if not item:
            return
        doc_id = item.data(Qt.UserRole)
        self.db.delete_document(doc_id)
        if doc_id == self.doc_id:
            self.doc_id = self.index = None
            self.chunks = []
            if self.fitz_doc:
                self.fitz_doc.close()
                self.fitz_doc = None
            self.chat.clear()
            self.page_image.clear()
            self.page_image.setText("Open a PDF to preview it here.")
            self.page_label.setText("No document")
        self._refresh_history()
        self._update_controls()

    # --------------------------------------------------------------- viewer
    def show_page(self, index: int):
        if not self.fitz_doc:
            return
        index = max(0, min(index, len(self.fitz_doc) - 1))
        self.page_index = index
        pix = self.fitz_doc[index].get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom), alpha=False)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
        self.page_image.setPixmap(QPixmap.fromImage(img))
        self.page_label.setText(f"Page {index + 1} of {len(self.fitz_doc)}")

    def set_zoom(self, zoom: float):
        self.zoom = max(0.6, min(zoom, 3.0))
        self.show_page(self.page_index)

    def _on_link(self, url):
        s = url.toString()
        if s.startswith("page:"):
            self.show_page(int(s.split(":")[1]) - 1)

    # ----------------------------------------------------------------- chat
    def _say(self, who: str, body_html: str):
        color = "#2d6cdf" if who == "You" else "#1b7f4b"
        self.chat.append(f"<p><span style='color:{color}'><b>{who}</b></span><br>{body_html}</p>")

    def ask(self):
        question = self.input.text().strip()
        if not question or self.index is None:
            return
        self.input.clear()
        self._say("You", to_html(question))
        self._run(rag.answer_question, (self.index, self.llm, question),
                  self._show_answer, "Searching the document and thinking...")

    def _show_answer(self, answer):
        body = to_html(answer.text)
        if answer.sources:
            body += "<br><br><b>Source references</b>"
            for s in answer.sources:
                body += (f"<br>• <a href='page:{s.page}'>Page {s.page}</a> "
                         f"<span style='color:#666'>– {html.escape(s.snippet)}</span>")
        self._say("DocuMind", body)

    def summarize_document(self):
        if self.doc_id is None:
            return
        cached = self.db.get_document(self.doc_id)[4]
        if cached:
            self._say("DocuMind", "<b>Summary</b><br>" + to_html(cached))
            return
        doc_id = self.doc_id

        def done(summary):
            self.db.save_summary(doc_id, summary)
            self._say("DocuMind", "<b>Summary</b><br>" + to_html(summary))

        self._run(rag.summarize, (self.chunks, self.llm), done,
                  "Summarizing (long documents can take a minute)...")
