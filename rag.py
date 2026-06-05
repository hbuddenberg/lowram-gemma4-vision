"""
RAG module for Gemma 4 API server.
FAISS vector store + sentence-transformers embeddings on CPU.
Stores chat history, retrieves relevant context within VRAM budget.
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

log = logging.getLogger("gemma4-rag")

# ── Config ────────────────────────────────────────────────────────────────────
RAG_DIR = Path.home() / ".gemma4api" / "rag"
RAG_DB_PATH = RAG_DIR / "history.db"
FAISS_INDEX_PATH = RAG_DIR / "faiss.index"
EMBEDDING_MODEL = os.environ.get("GEMMA4_RAG_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension
MAX_STORED_PER_SESSION = 500  # Max messages stored per session
TOP_K_RETRIEVAL = 5  # How many chunks to retrieve
MAX_RETRIEVAL_CHARS = 2000  # Max chars from retrieval injection

# ── Thread-safe singleton ─────────────────────────────────────────────────────
_lock = threading.Lock()
_instance: Optional["RAGStore"] = None


class RAGStore:
    """FAISS + SQLite store for chat history with semantic retrieval."""

    def __init__(self):
        RAG_DIR.mkdir(parents=True, exist_ok=True)
        self._embedding_model = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._next_id = 0
        self._init_db()
        self._init_faiss()

    # ── Lazy-load embedding model (CPU only) ───────────────────────────────
    @property
    def embedding_model(self):
        if self._embedding_model is None:
            log.info(f"Loading embedding model '{EMBEDDING_MODEL}' on CPU...")
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            log.info("Embedding model loaded.")
        return self._embedding_model

    # ── SQLite: stores messages + FAISS vector IDs ────────────────────────
    def _init_db(self):
        with sqlite3.connect(RAG_DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY,
                    faiss_id   INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    timestamp  REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, timestamp)")
            # Get highest faiss_id to resume
            row = conn.execute("SELECT COALESCE(MAX(faiss_id), -1) FROM messages").fetchone()
            self._next_id = row[0] + 1

    # ── FAISS index ───────────────────────────────────────────────────────
    def _init_faiss(self):
        if FAISS_INDEX_PATH.exists():
            self._index = faiss.read_index(str(FAISS_INDEX_PATH))
            # Sync _next_id with actual index size
            self._next_id = max(self._next_id, self._index.ntotal)
            log.info(f"Loaded FAISS index with {self._index.ntotal} vectors.")
        else:
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
            log.info("Created new FAISS index.")

    def _save_index(self):
        faiss.write_index(self._index, str(FAISS_INDEX_PATH))

    # ── Store messages ───────────────────────────────────────────────────
    def store_messages(self, session_id: str, messages: list[dict]):
        """Store messages from a chat completion request into FAISS + SQLite."""
        model = self.embedding_model

        to_embed = []
        to_store = []
        now = time.time()

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from multimodal content
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                content = " ".join(parts)

            if not content or not content.strip():
                continue

            # Dedup by content hash (avoid storing identical messages)
            content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

            # Check for duplicate
            with sqlite3.connect(RAG_DB_PATH) as conn:
                existing = conn.execute(
                    "SELECT id FROM messages WHERE session_id=? AND substr(content,1,64)=?",
                    (session_id, content[:64])
                ).fetchone()
                if existing:
                    continue

            to_embed.append(content)
            to_store.append((session_id, role, content, now))

        if not to_embed:
            return

        # Compute embeddings on CPU
        embeddings = model.encode(to_embed, show_progress_bar=False, normalize_embeddings=True)
        vectors = np.array(embeddings, dtype=np.float32)

        # Add to FAISS
        self._index.add(vectors)

        # Store in SQLite
        start_faiss_id = self._next_id
        with sqlite3.connect(RAG_DB_PATH) as conn:
            for i, (session_id, role, content, now) in enumerate(to_store):
                faiss_id = start_faiss_id + i
                conn.execute(
                    "INSERT INTO messages (faiss_id, session_id, role, content, timestamp) VALUES (?,?,?,?,?)",
                    (faiss_id, session_id, role, content, now),
                )
            # Trim old messages per session
            conn.execute("""
                DELETE FROM messages WHERE session_id=? AND id NOT IN (
                    SELECT id FROM messages WHERE session_id=? ORDER BY timestamp DESC LIMIT ?
                )
            """, (session_id, session_id, MAX_STORED_PER_SESSION))

        self._next_id = start_faiss_id + len(to_embed)
        self._save_index()
        log.debug(f"Stored {len(to_embed)} messages for session {session_id}")

    # ── Retrieve relevant context ─────────────────────────────────────────
    def retrieve(self, query: str, session_id: Optional[str] = None, top_k: int = TOP_K_RETRIEVAL) -> str:
        """Retrieve relevant past messages based on query. Returns formatted context string."""
        if self._index.ntotal == 0:
            return ""

        model = self.embedding_model
        query_vec = model.encode([query], show_progress_bar=False, normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        # Search FAISS
        scores, indices = self._index.search(query_vec, min(top_k, self._index.ntotal))

        results = []
        total_chars = 0

        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            if score < 0.3:  # Minimum similarity threshold
                continue

            with sqlite3.connect(RAG_DB_PATH) as conn:
                row = conn.execute(
                    "SELECT role, content FROM messages WHERE faiss_id=?", (int(idx),)
                ).fetchone()
                if row:
                    role, content = row
                    entry = f"[{role}]: {content}"
                    if total_chars + len(entry) > MAX_RETRIEVAL_CHARS:
                        break
                    results.append(entry)
                    total_chars += len(entry)

        if not results:
            return ""

        context = "\n".join(results)
        return context

    # ── Get session stats ─────────────────────────────────────────────────
    def session_stats(self) -> dict:
        with sqlite3.connect(RAG_DB_PATH) as conn:
            total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0]
            return {
                "total_messages": total,
                "total_sessions": sessions,
                "faiss_vectors": self._index.ntotal,
            }


def get_rag_store() -> RAGStore:
    """Get or create the singleton RAGStore."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RAGStore()
    return _instance
