#!/usr/bin/env python3
"""
Gemma 4 E4B Heretic — OpenAI-compatible API server
FastAPI server wrapping bitsandbytes NF4 quantized Gemma 4 for text + vision.

Endpoints:
  POST   /v1/chat/completions          — OpenAI-compatible (streaming + non-streaming)
  GET    /v1/models                    — List available models
  GET    /v1/models/{model}            — Get model details
  GET    /health                       — Health check + VRAM
  POST   /v1/keys                      — Generate API key (admin-key protected)
  GET    /v1/keys                      — List API keys (admin-key protected)
  DELETE /v1/keys/{key_prefix}         — Delete API key (admin-key protected)
  GET    /v1/keys/{key_prefix}/usage   — Per-key usage stats (admin-key protected)
  POST   /v1/keys/{key_prefix}/rotate  — Rotate key secret (admin-key protected)
  GET    /v1/usage                     — Aggregated usage analytics (admin-key protected)
  GET    /v1/config                    — Server config (admin-key protected)

Usage:
  python server.py [--host 0.0.0.0] [--port 8080] [--model-path ~/models/gemma4-heretic]
"""

import argparse
import asyncio
import base64
import gc
import hashlib
import ipaddress
import io
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

# Configure CUDA allocator to reduce fragmentation (critical for 12GB VRAM)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import requests as _requests
import torch
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse
from transformers import (
    BitsAndBytesConfig,
    Gemma4ForConditionalGeneration,
    Gemma4Processor,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from rag import get_rag_store

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gemma4-server")

# ── Globals ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "GEMMA4_MODEL_PATH", os.path.expanduser("~/models/gemma4-heretic")
)
MODEL_ID = "gemma-4-e4b-heretic"
MODEL_OBJ: Optional[Gemma4ForConditionalGeneration] = None
PROCESSOR: Optional[Gemma4Processor] = None
LOAD_LOCK = threading.Lock()
INFERENCE_SEMAPHORE: Optional[asyncio.Semaphore] = None
SERVER_START_TIME = time.time()
LAST_REQUEST_TIME = time.time()
REQUESTS_IN_FLIGHT = 0
REQUESTS_IN_FLIGHT_LOCK = threading.Lock()

# ── Configurable defaults ─────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 128  # Further reduced for 12GB VRAM
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
MAX_CONTEXT_MESSAGES = 4  # Max 2 turns (user-assistant-user-assistant) to avoid OOM
# Legacy single-key backward compat (used when MASTER_KEY is unset)
API_KEY = os.environ.get("GEMMA4_API_KEY", "gemma4-local")
# Accept both env var names for smooth migration
MASTER_KEY = os.environ.get("GEMMA4_MASTER_KEY") or os.environ.get("GEMMA4_ADMIN_KEY", "")
MAX_IMAGE_SIZE_MB = int(os.environ.get("GEMMA4_MAX_IMAGE_MB", "20"))
VALID_ROLES = {"system", "user", "assistant", "tool"}
DEFAULT_RATE_LIMIT = 30  # requests per minute per key (single GPU default)
RAG_ENABLED = os.environ.get("GEMMA4_RAG", "1").lower() in ("1", "true", "yes")

# ── Input/generation limits (critical for 12GB VRAM) ─────────────────────────
# Max input tokens the server will accept. Gemma 4 E4B with NF4 on 12GB VRAM
# starts thrashing past ~6K tokens. This is a hard reject boundary.
MAX_INPUT_TOKENS = int(os.environ.get("GEMMA4_MAX_INPUT_TOKENS", "4096"))
# Soft limit: messages beyond this get summarized/chunked before reaching the model
CHUNK_THRESHOLD_TOKENS = int(os.environ.get("GEMMA4_CHUNK_THRESHOLD", "2048"))
# Max wall-clock seconds for a single generation (prevents 15-min hangs)
GENERATION_TIMEOUT = int(os.environ.get("GEMMA4_GEN_TIMEOUT", "120"))  # 2 min

# ── Storage paths ─────────────────────────────────────────────────────────────
CONFIG_DIR = Path.home() / ".gemma4api"
DB_PATH = CONFIG_DIR / "keys.db"
# Legacy path — only used for one-time migration from old JSON storage
_LEGACY_KEYS_FILE = Path.home() / ".config" / "gemma4-api" / "keys.json"

# ── CORS configuration ────────────────────────────────────────────────────────
_CORS_ORIGINS = [o.strip() for o in os.environ.get("GEMMA4_CORS_ORIGINS", "*").split(",")]

# ── Security: allowed image directory (empty = deny all file:// paths) ────────
ALLOWED_IMAGE_DIR = os.environ.get("GEMMA4_IMAGE_DIR", "")

# ── Default system prompt (injected when client sends none) ───────────────────
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "GEMMA4_SYSTEM_PROMPT",
    "Eres un asistente útil y conciso. Responde SIEMPRE en el mismo idioma que el usuario. "
    "Si el usuario escribe en español, responde en español. "
    "Si el usuario escribe en inglés, responde en inglés.",
)

# ── Session management ────────────────────────────────────────────────────────
# Sessions allow RAG to track conversations across requests and cache system prompts.
# Session ID is derived from: X-Session-ID header > message content hash > "default"
SESSION_IDLE_TIMEOUT = int(os.environ.get("GEMMA4_SESSION_TIMEOUT", "3600"))  # 1 hour


# ── Database ──────────────────────────────────────────────────────────────────

def _init_db_sync() -> None:
    """Create ~/.gemma4api/, schema, indexes. Migrate legacy JSON if present."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash     TEXT UNIQUE NOT NULL,
                key_prefix   TEXT NOT NULL,
                name         TEXT NOT NULL,
                created_at   REAL NOT NULL,
                expires_at   REAL,
                last_used_at REAL,
                is_active    INTEGER NOT NULL DEFAULT 1,
                usage_count  INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                rate_limit   INTEGER NOT NULL DEFAULT 30
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         REAL NOT NULL,
                key_prefix        TEXT NOT NULL,
                model             TEXT NOT NULL,
                prompt_tokens     INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens      INTEGER NOT NULL,
                latency_ms        REAL NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_req_log_ts  ON request_log(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_req_log_key ON request_log(key_prefix)"
        )
        # ── Sessions table: caches system prompts per conversation ──────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                system_prompt TEXT,
                created_at   REAL NOT NULL,
                last_used_at REAL NOT NULL
            )
        """)
        conn.commit()

    if _LEGACY_KEYS_FILE.exists():
        try:
            _migrate_legacy_keys_sync()
        except Exception as exc:
            log.warning(f"Legacy key migration failed (non-fatal): {exc}")


def _migrate_legacy_keys_sync() -> None:
    """Import keys from old JSON file into SQLite (idempotent via INSERT OR IGNORE)."""
    with open(_LEGACY_KEYS_FILE) as fh:
        old_keys: dict = json.load(fh)

    with sqlite3.connect(DB_PATH) as conn:
        for prefix, data in old_keys.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO api_keys
                    (key_hash, key_prefix, name, created_at, expires_at,
                     is_active, usage_count, total_tokens, rate_limit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("key_hash", ""),
                    prefix,
                    data.get("name", "migrated"),
                    data.get("created_at", time.time()),
                    data.get("expires_at"),
                    1 if data.get("enabled", True) else 0,
                    data.get("total_requests", 0),
                    data.get("total_tokens", 0),
                    data.get("rate_limit", DEFAULT_RATE_LIMIT),
                ),
            )
        conn.commit()

    log.info(f"Migrated {len(old_keys)} keys from legacy JSON storage")
    _LEGACY_KEYS_FILE.rename(_LEGACY_KEYS_FILE.with_suffix(".json.migrated"))


def _hash_key(key: str) -> str:
    """Return SHA-256 hex digest of key for secure storage comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


def _validate_key_sync(key: str) -> Optional[str]:
    """Look up key in cache; on match update last_used_at + usage_count. Return key_prefix."""
    key_hash = _hash_key(key)
    now = time.time()
    with _KEYS_CACHE_LOCK:
        # Fast O(1) lookup via index
        if key_hash not in _KEY_HASH_INDEX:
            return None
        key_prefix = _KEY_HASH_INDEX[key_hash]
        key_data = _KEYS_CACHE.get(key_prefix)
        if not key_data:
            return None
        # Check active and expiry
        if key_data.get("is_active") != 1:
            return None
        expires_at = key_data.get("expires_at")
        if expires_at and expires_at <= now:
            return None
        # Update in cache and persist
        key_data["last_used_at"] = now
        key_data["usage_count"] = key_data.get("usage_count", 0) + 1
        _persist_key(key_prefix, key_data)
    return key_prefix


def _get_key_rate_limit_sync(key_prefix: str) -> int:
    """Return configured RPM for a key prefix; DEFAULT_RATE_LIMIT for legacy keys."""
    if key_prefix == "legacy":
        return DEFAULT_RATE_LIMIT
    key_data = _get_key(key_prefix)
    return key_data.get("rate_limit", DEFAULT_RATE_LIMIT) if key_data else DEFAULT_RATE_LIMIT


def _log_usage_sync(
    key_prefix: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
) -> None:
    """Insert into request_log and update key's total_tokens in cache."""
    tokens = prompt_tokens + completion_tokens
    now = time.time()

    # Update cache if not legacy key
    if key_prefix != "legacy":
        with _KEYS_CACHE_LOCK:
            if key_prefix in _KEYS_CACHE:
                _KEYS_CACHE[key_prefix]["total_tokens"] = (
                    _KEYS_CACHE[key_prefix].get("total_tokens", 0) + tokens
                )
                _persist_key(key_prefix, _KEYS_CACHE[key_prefix])

    # Log to request_log (analytics, append-only)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO request_log
                (timestamp, key_prefix, model, prompt_tokens, completion_tokens,
                 total_tokens, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now, key_prefix, model, prompt_tokens, completion_tokens, tokens, latency_ms),
        )
        conn.commit()


# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Per-key sliding-window rate limiter (in-memory, resets on restart)."""

    def __init__(self) -> None:
        self.requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key_prefix: str, limit: int = DEFAULT_RATE_LIMIT) -> bool:
        """Record and gate request; return False if over RPM limit."""
        now = time.time()
        window_start = now - 60
        with self._lock:
            bucket = self.requests.setdefault(key_prefix, [])
            self.requests[key_prefix] = [t for t in bucket if t > window_start]
            if len(self.requests[key_prefix]) >= limit:
                return False
            self.requests[key_prefix].append(now)
        return True


RATE_LIMITER = RateLimiter()


# ── In-memory key cache ───────────────────────────────────────────────────────

_KEYS_CACHE: dict[str, dict] = {}
_KEY_HASH_INDEX: dict[str, str] = {}  # Maps key_hash -> key_prefix for O(1) validation
_KEYS_CACHE_LOCK = threading.RLock()


def _load_keys_startup() -> None:
	"""Load all keys from DB into cache at startup."""
	global _KEYS_CACHE, _KEY_HASH_INDEX
	with sqlite3.connect(DB_PATH) as conn:
		conn.row_factory = sqlite3.Row
		rows = conn.execute("SELECT * FROM api_keys").fetchall()

	with _KEYS_CACHE_LOCK:
		for row in rows:
			key_data = dict(row)
			key_prefix = row["key_prefix"]
			_KEYS_CACHE[key_prefix] = key_data
			_KEY_HASH_INDEX[row["key_hash"]] = key_prefix
	log.info(f"Loaded {len(_KEYS_CACHE)} API keys into memory cache")


def _persist_key(key_prefix: str, key_data: dict) -> None:
	"""Write a single key to SQLite. MUST be called under _KEYS_CACHE_LOCK."""
	with sqlite3.connect(DB_PATH) as conn:
		conn.execute(
			"""
			INSERT INTO api_keys
				(key_hash, key_prefix, name, created_at, expires_at, last_used_at,
				 is_active, usage_count, total_tokens, rate_limit)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(key_hash) DO UPDATE SET
				key_hash = excluded.key_hash,
				name = excluded.name,
				created_at = excluded.created_at,
				expires_at = excluded.expires_at,
				last_used_at = excluded.last_used_at,
				is_active = excluded.is_active,
				usage_count = excluded.usage_count,
				total_tokens = excluded.total_tokens,
				rate_limit = excluded.rate_limit
			""",
			(
				key_data.get("key_hash"),
				key_prefix,
				key_data.get("name"),
				key_data.get("created_at"),
				key_data.get("expires_at"),
				key_data.get("last_used_at"),
				key_data.get("is_active", 1),
				key_data.get("usage_count", 0),
				key_data.get("total_tokens", 0),
				key_data.get("rate_limit", DEFAULT_RATE_LIMIT),
			),
		)
		conn.commit()


def _get_key(key_prefix: str) -> Optional[dict]:
	"""Get a key from cache (no I/O). Returns copy to prevent external mutations."""
	with _KEYS_CACHE_LOCK:
		data = _KEYS_CACHE.get(key_prefix)
		return dict(data) if data else None


def _get_all_keys() -> dict:
	"""Get all keys from cache (no I/O). Returns deep copy."""
	with _KEYS_CACHE_LOCK:
		return {k: dict(v) for k, v in _KEYS_CACHE.items()}


def _update_key(key_prefix: str, updates: dict) -> None:
	"""Update a key in cache and persist to SQLite atomically."""
	with _KEYS_CACHE_LOCK:
		if key_prefix not in _KEYS_CACHE:
			raise KeyError(key_prefix)
		_KEYS_CACHE[key_prefix].update(updates)
		_persist_key(key_prefix, _KEYS_CACHE[key_prefix])


def _create_key(key_prefix: str, key_data: dict) -> None:
	"""Create a new key in cache and persist."""
	with _KEYS_CACHE_LOCK:
		_KEYS_CACHE[key_prefix] = key_data
		_KEY_HASH_INDEX[key_data["key_hash"]] = key_prefix
		# Insert into DB (cache already in memory)
		with sqlite3.connect(DB_PATH) as conn:
			conn.execute(
				"""
				INSERT INTO api_keys
					(key_hash, key_prefix, name, created_at, expires_at, is_active, rate_limit)
				VALUES (?, ?, ?, ?, ?, ?, ?)
				""",
				(
					key_data["key_hash"],
					key_prefix,
					key_data["name"],
					key_data["created_at"],
					key_data.get("expires_at"),
					1,
					key_data.get("rate_limit", DEFAULT_RATE_LIMIT),
				),
			)
			conn.commit()


def _delete_key(key_prefix: str) -> None:
	"""Delete a key from cache and SQLite."""
	with _KEYS_CACHE_LOCK:
		if key_prefix not in _KEYS_CACHE:
			raise KeyError(key_prefix)
		key_hash = _KEYS_CACHE[key_prefix].get("key_hash")
		del _KEYS_CACHE[key_prefix]
		if key_hash and key_hash in _KEY_HASH_INDEX:
			del _KEY_HASH_INDEX[key_hash]
		# Delete from SQLite directly
		with sqlite3.connect(DB_PATH) as conn:
			conn.execute("DELETE FROM api_keys WHERE key_prefix = ?", (key_prefix,))
			conn.commit()


# ── Session management helpers ────────────────────────────────────────────────

# In-memory session cache: session_id -> {system_prompt, created_at, last_used_at}
_SESSIONS_CACHE: dict[str, dict] = {}
_SESSIONS_CACHE_LOCK = threading.RLock()


def _get_or_create_session(session_id: str, system_prompt: Optional[str] = None) -> dict:
    """Get or create a session, optionally caching its system prompt. Returns session data."""
    now = time.time()
    with _SESSIONS_CACHE_LOCK:
        if session_id in _SESSIONS_CACHE:
            session = _SESSIONS_CACHE[session_id]
            session["last_used_at"] = now
            # Update system prompt if provided and different
            if system_prompt and system_prompt != session.get("system_prompt"):
                session["system_prompt"] = system_prompt
                _persist_session(session_id, session)
            return session

        # Try loading from SQLite
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

        if row:
            session = dict(row)
            session["last_used_at"] = now
            if system_prompt:
                session["system_prompt"] = system_prompt
            _SESSIONS_CACHE[session_id] = session
            _persist_session(session_id, session)
            return session

        # Create new session
        session = {
            "session_id": session_id,
            "system_prompt": system_prompt,
            "created_at": now,
            "last_used_at": now,
        }
        _SESSIONS_CACHE[session_id] = session
        _persist_session(session_id, session)
        return session


def _persist_session(session_id: str, session: dict) -> None:
    """Write session to SQLite. Must be called under _SESSIONS_CACHE_LOCK."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, system_prompt, created_at, last_used_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                system_prompt = excluded.system_prompt,
                last_used_at = excluded.last_used_at
            """,
            (session_id, session.get("system_prompt"), session.get("created_at"), session.get("last_used_at")),
        )
        conn.commit()


def _derive_session_id(messages: "list[ChatMessage]") -> str:
    """Derive a stable session ID from the first user message content.
    
    This ensures the same conversation gets the same session ID across requests,
    enabling RAG to retrieve context from earlier messages in the same conversation.
    """
    for msg in messages:
        if msg.role == "user":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content:
                # Hash first 200 chars of first user message for stable session ID
                return f"s-{hashlib.sha256(content[:200].encode()).hexdigest()[:16]}"
    return "s-default"


def _cleanup_idle_sessions() -> None:
    """Remove sessions idle longer than SESSION_IDLE_TIMEOUT."""
    cutoff = time.time() - SESSION_IDLE_TIMEOUT
    with _SESSIONS_CACHE_LOCK:
        to_remove = [
            sid for sid, s in _SESSIONS_CACHE.items()
            if s.get("last_used_at", 0) < cutoff
        ]
        for sid in to_remove:
            del _SESSIONS_CACHE[sid]
    if to_remove:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "DELETE FROM sessions WHERE session_id IN (%s) AND last_used_at < ?"
                % ",".join("?" * len(to_remove)),
                to_remove + [cutoff],
            )
            conn.commit()
        log.debug(f"Cleaned up {len(to_remove)} idle sessions")


# ── URL auto-fix for tool calls ───────────────────────────────────────────────

# Regex: bare domain like "todorelatos.com", "example.org/path" — no scheme
_BARE_DOMAIN_RE = re.compile(
    r"^(?![a-zA-Z][a-zA-Z0-9+.\-]*://)"  # Not already a URL with scheme
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?\.)+"  # Domain parts
    r"[a-zA-Z]{2,}"  # TLD
    r"(?:/[^\s]*)?$"  # Optional path
)


def _fix_tool_call_urls(tool_calls: list[dict]) -> list[dict]:
    """Auto-prepend https:// to bare domains in tool call string arguments.
    
    Many models generate tool calls with bare domains like "todorelatos.com"
    instead of "https://todorelatos.com". Client apps often validate URL format
    with Pydantic (format: "url") which rejects bare domains.
    """
    for tc in tool_calls:
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")
        if isinstance(args_str, str):
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                continue
            modified = False
            for key, value in args.items():
                if isinstance(value, str) and _BARE_DOMAIN_RE.match(value):
                    args[key] = f"https://{value}"
                    modified = True
            if modified:
                func["arguments"] = json.dumps(args, ensure_ascii=False)
    return tool_calls


# ── Input token estimation & chunking ────────────────────────────────────────

def _estimate_tokens(messages: "list[ChatMessage]") -> int:
    """Rough token count estimation for messages before tokenization.
    
    Uses a simple heuristic: ~4 chars per token for mixed text.
    This is fast (no tokenizer call) and used for early rejection.
    After trimming/chunking, the actual tokenizer count is used.
    """
    total_chars = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    total_chars += len(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        total_chars += 256  # Approximate image token count
        # Role + formatting overhead
        total_chars += 20
    return total_chars // 4  # ~4 chars per token for multilingual text


def _chunk_long_message(text: str, max_chunk_tokens: int = 800) -> list[str]:
    """Split a long text into chunks of approximately max_chunk_tokens each.
    
    Tries to split on sentence boundaries (period, newline) first.
    Falls back to fixed-size chunks if no good split points found.
    """
    # Target chars per chunk (~4 chars/token)
    max_chars = max_chunk_tokens * 4
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    remaining = text
    
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        
        # Try to split at sentence boundary within the window
        split_point = max_chars
        
        # Look backwards for sentence boundaries
        for i in range(min(len(remaining), max_chars), max(max_chars // 2, 0), -1):
            if i < len(remaining) and remaining[i] in ".\n!?\r":
                split_point = i + 1
                break
        
        chunk = remaining[:split_point].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_point:].strip()
    
    return chunks


def _summarize_message_chunk(text: str, session_id: str) -> str:
    """Use RAG to extract a summary of a long message chunk.
    
    Instead of running the model for summarization (expensive), we store
    the full text in RAG and return a truncated version with a marker.
    The full text is still searchable via FAISS for later retrieval.
    """
    # Store the full text in RAG for retrieval
    try:
        rag = get_rag_store()
        rag.store_messages(session_id, [
            {"role": "user", "content": text}
        ])
    except Exception:
        pass
    
    # Return truncated version (first ~500 chars ≈ 125 tokens)
    max_preview = 500
    if len(text) <= max_preview:
        return text
    
    # Find a good cutoff point
    cutoff = text[:max_preview]
    last_period = cutoff.rfind(".")
    if last_period > max_preview // 2:
        cutoff = cutoff[:last_period + 1]
    
    return f"{cutoff} [...] [Texto largo almacenado en RAG — {len(text)} chars totales]"


# ── Stopping criteria ─────────────────────────────────────────────────────────

class StopSequenceCriteria(StoppingCriteria):
    """Stop generation when any configured stop sequence appears in output."""

    def __init__(self, tokenizer, stop_sequences: list[str]) -> None:
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self._last_len = 0
        self._accumulated = ""

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        current_len = input_ids.shape[1]
        if current_len > self._last_len:
            new_tokens = self.tokenizer.decode(
                input_ids[0, self._last_len:], skip_special_tokens=False
            )
            self._accumulated += new_tokens
            self._last_len = current_len
        return any(seq in self._accumulated for seq in self.stop_sequences)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str | list | None = ""
    # Tool calling fields (OpenAI-compatible, all optional)
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in VALID_ROLES:
            raise ValueError(f"Invalid role '{v}'. Must be one of: {VALID_ROLES}")
        return v


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = None
    n: int = Field(default=1, ge=1, le=1)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logprobs: bool = False
    top_logprobs: int | None = None
    stream_options: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | None = None
    enable_thinking: bool = False
    reasoning_effort: str | None = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list) -> list:
        if not v:
            raise ValueError("messages must contain at least one message")
        return v


def _chunk_messages(messages: list[ChatMessage], session_id: str) -> list[ChatMessage]:
    """Process messages with long text: store full text in RAG, replace with summaries.
    
    Strategy:
    1. For each message with text > 3200 chars (~800 tokens), store in RAG and replace
       with a truncated summary
    2. Keep the last 2 user messages intact (most relevant context)
    3. Preserve system messages and tool messages unchanged
    
    This ensures long texts (like documents, articles) are:
    - Fully stored in FAISS for semantic retrieval
    - Replaced with short previews in the context window
    - Still accessible when the model needs specific details
    """
    # Find the indices of the last 2 user messages to keep intact
    user_indices = [i for i, m in enumerate(messages) if m.role == "user"]
    keep_intact = set(user_indices[-2:]) if len(user_indices) >= 2 else set(user_indices)
    
    result = []
    for i, msg in enumerate(messages):
        # Never modify system, tool, or the last 2 user messages
        if msg.role in ("system", "tool") or i in keep_intact:
            result.append(msg)
            continue
        
        content = msg.content
        if not isinstance(content, str) or len(content) <= 3200:
            result.append(msg)
            continue
        
        # Long text: store in RAG and summarize
        summary = _summarize_message_chunk(content, session_id)
        log.info(f"Chunked {msg.role} msg: {len(content)} → {len(summary)} chars (session={session_id})")
        result.append(ChatMessage(role=msg.role, content=summary))
    
    return result


class ChatCompletionChoice(BaseModel):
    index: int
    message: dict
    finish_reason: str | None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    rate_limit: int = Field(default=DEFAULT_RATE_LIMIT, ge=1, le=1000)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


class ConfigResponse(BaseModel):
    model: str
    version: str
    vram_used_gb: float
    vram_total_gb: float
    active_keys: int
    uptime_seconds: float


# ── Error response helper ─────────────────────────────────────────────────────

def _error_response(
    message: str,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
) -> dict:
    """Format error in OpenAI style."""
    err: dict = {"message": message, "type": error_type}
    if code:
        err["code"] = code
    return {"error": err}


# ── Security helpers ──────────────────────────────────────────────────────────

def _is_private_url(url: str) -> bool:
    """Return True if URL resolves to a private/loopback/link-local address."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return True
        if hostname in ("localhost", "127.0.0.1", "::1"):
            return True
        import socket
        for _fam, _typ, _pro, _can, sockaddr in socket.getaddrinfo(hostname, None):
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return True
    except Exception:
        return True
    return False


def _validate_image_path(path: str) -> bool:
    """Return True only if path is a real file inside the configured image dir."""
    if not ALLOWED_IMAGE_DIR:
        return False
    real = os.path.realpath(path)
    allowed = os.path.realpath(ALLOWED_IMAGE_DIR)
    return (real.startswith(allowed + os.sep) or real == allowed) and os.path.isfile(real)


async def _check_api_key(request: Request) -> Optional[str]:
    """Extract and validate API key header; return key_prefix or None."""
    auth = request.headers.get("Authorization", "")
    key = auth[7:] if auth.startswith("Bearer ") else request.headers.get("X-API-Key", "")
    if not key:
        return None
    # Backward compat: single env-var key bypasses DB lookup
    if key == API_KEY:
        return "legacy"
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _validate_key_sync, key)


def _check_master_key(request: Request) -> bool:
    """Return True if request carries the configured master/admin key."""
    if not MASTER_KEY:
        return False
    auth = request.headers.get("Authorization", "")
    candidate = auth[7:] if auth.startswith("Bearer ") else request.headers.get("X-API-Key", "")
    return candidate == MASTER_KEY


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load Gemma 4 with NF4 text + fp16 vision (thread-safe)."""
    global MODEL_OBJ, PROCESSOR

    with LOAD_LOCK:
        if MODEL_OBJ is not None:
            return MODEL_OBJ, PROCESSOR

        log.info(f"Loading model from {model_path}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=[
                "vision_tower", "model.vision_tower",
                "multi_modal_projector",
                "embed_vision", "model.embed_vision",
                "audio_tower", "model.audio_tower",
                "embed_audio", "model.embed_audio",
                "lm_head",
            ],
        )

        gc.collect()
        torch.cuda.empty_cache()

        processor = Gemma4Processor.from_pretrained(model_path)
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            offload_buffers=True,
        )
        model.eval()

        MODEL_OBJ = model
        PROCESSOR = processor

        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info(f"Model loaded. VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")

    return model, processor


def unload_model() -> None:
    """Gracefully unload model and free GPU memory."""
    global MODEL_OBJ, PROCESSOR
    if MODEL_OBJ is not None:
        log.info("Unloading model...")
        del MODEL_OBJ
        del PROCESSOR
        MODEL_OBJ = None
        PROCESSOR = None
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Model unloaded, GPU memory freed.")


# ── Inference helpers ─────────────────────────────────────────────────────────

# Image preprocessing: max resolution before passing to Gemma vision encoder.
# Gemma 4 vision internally resizes to 896×896, so anything larger wastes VRAM.
IMAGE_MAX_DIMENSION = int(os.environ.get("GEMMA4_IMAGE_MAX_DIM", "896"))
# JPEG quality for in-memory compression (lower = less VRAM, slight quality loss)
IMAGE_QUALITY = int(os.environ.get("GEMMA4_IMAGE_QUALITY", "85"))

# VRAM budget: reduce context when vision/audio is present to reserve memory
CONTEXT_MESSAGES_TEXT = 4     # Text-only: up to 4 messages
CONTEXT_MESSAGES_VISION = 2  # With image: only 2 messages (reserve ~500 MB)
CONTEXT_MESSAGES_AUDIO = 3  # With audio: 3 messages (future-proofing)


def _preprocess_image(img: Image.Image) -> Image.Image:
    """Resize and optionally compress image to reduce VRAM during vision encoding.

    - Resizes to IMAGE_MAX_DIMENSION (longest side) maintaining aspect ratio
    - Converts to RGB (drops alpha channel which wastes memory)
    - Uses LANCZOS resampling for quality
    """
    w, h = img.size
    max_dim = max(w, h)

    if max_dim > IMAGE_MAX_DIMENSION:
        scale = IMAGE_MAX_DIMENSION / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        log.debug(f"Image resized: {w}x{h} → {new_w}x{new_h}")

    return img.convert("RGB")


def _get_context_budget(messages: list) -> int:
    """Determine MAX_CONTEXT_MESSAGES based on multimodal content presence.

    When images or audio are in the message history, reduce the context window
    to reserve VRAM for the vision/audio encoder.
    """
    has_vision = False
    has_audio = False

    for msg in messages:
        content = msg.content if hasattr(msg, "content") else msg
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    t = item.get("type", "")
                    if t == "image_url":
                        has_vision = True
                    elif t == "input_audio":
                        has_audio = True

    if has_vision:
        return CONTEXT_MESSAGES_VISION
    if has_audio:
        return CONTEXT_MESSAGES_AUDIO
    return CONTEXT_MESSAGES_TEXT


def _parse_content(content: str | list | None) -> tuple[str, list[Image.Image]]:
    """Extract text and images from OpenAI-format message content.

    Images are resized to reduce VRAM usage during vision encoding.
    Gemma 4 vision encoder processes at fixed internal resolution, so
    oversized images just waste memory during PIL→processor conversion.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []

    text_parts: list[str] = []
    images: list[Image.Image] = []

    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    b64 = url.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64)
                    if len(img_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                        raise ValueError(f"Image exceeds {MAX_IMAGE_SIZE_MB}MB limit")
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(_preprocess_image(img))
                elif url.startswith(("http", "ftp")):
                    if _is_private_url(url):
                        raise ValueError("Fetching internal/private URLs is blocked")
                    resp = _requests.get(url, timeout=15, stream=True, allow_redirects=False)
                    resp.raise_for_status()
                    size = 0
                    chunks: list[bytes] = []
                    for chunk in resp.iter_content(8192):
                        size += len(chunk)
                        if size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                            raise ValueError(f"Image exceeds {MAX_IMAGE_SIZE_MB}MB limit")
                        chunks.append(chunk)
                    img = Image.open(io.BytesIO(b"".join(chunks))).convert("RGB")
                    images.append(_preprocess_image(img))
                else:
                    if _validate_image_path(url):
                        with open(url, "rb") as fh:
                            img = Image.open(fh).convert("RGB")
                        images.append(_preprocess_image(img))

    return " ".join(text_parts), images


def _build_messages(messages: list[ChatMessage]) -> list[dict]:
    """Convert OpenAI ChatMessage list to the format expected by the Gemma 4 chat template.

    The Gemma 4 chat_template.jinja expects:
    - Assistant messages with `tool_calls`: [{function: {name, arguments (dict|str)}}]
    - Tool result messages with `tool_responses`: [{name, response}]
      or alternatively, `tool` role messages are accepted by apply_chat_template

    OpenAI clients send:
    - Assistant messages with `tool_calls`: [{id, type, function: {name, arguments (str)}}]
    - Tool messages with `role: "tool"`, `tool_call_id`, `name`, `content`
    """
    result = []
    for msg in messages:
        text, images = _parse_content(msg.content)
        entry: dict = {"role": msg.role}

        # Handle images in content
        if images:
            content_list: list[dict] = [{"type": "image"}]
            if text:
                content_list.append({"type": "text", "text": text})
            entry["content"] = content_list
        elif text:
            entry["content"] = text

        # ── Assistant messages with tool_calls ────────────────────────
        # OpenAI: {"role": "assistant", "tool_calls": [{id, type, function: {name, arguments: str}}]}
        # Gemma:  {"role": "assistant", "tool_calls": [{function: {name, arguments: dict|str}}]}
        # parse_response returns arguments as dict → our OpenAI output serializes to JSON str.
        # The chat template accepts BOTH dict and string arguments (lines 193-201).
        if msg.role == "assistant" and msg.tool_calls:
            gemma_tool_calls = []
            for tc in msg.tool_calls:
                func = tc.get("function", {})
                # Arguments from OpenAI client are a JSON string — try to parse to dict
                # so the chat template can use its own formatting (line 198)
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        pass  # Keep as string, template handles both
                gemma_tool_calls.append({
                    "function": {"name": func.get("name", ""), "arguments": args}
                })
            entry["tool_calls"] = gemma_tool_calls

        # ── Tool result messages ──────────────────────────────────────
        # OpenAI: {"role": "tool", "tool_call_id": "call_abc", "content": "...", "name": "fn"}
        # Convert to a standard dict the chat template can process.
        if msg.role == "tool":
            result.append(entry)
            continue

        result.append(entry)
    return result


def _prepare_inputs(
    model,
    processor,
    messages: list[ChatMessage],
    tools: list[dict] | None = None,
    enable_thinking: bool = False,
    session_id: Optional[str] = None,
):
    """Tokenize messages with apply_chat_template; support tools, thinking, vision.

    When RAG is enabled and messages exceed MAX_CONTEXT_MESSAGES, older messages
    are stored in FAISS and relevant context is retrieved and injected as a
    system message instead of being silently dropped.
    """
    # ── Inject default system prompt if none present ───────────────────────
    has_system = any(m.role == "system" for m in messages)
    if not has_system and DEFAULT_SYSTEM_PROMPT:
        # Try to load cached system prompt for this session
        session = _get_or_create_session(session_id or "default")
        cached_prompt = session.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        messages = [ChatMessage(role="system", content=cached_prompt)] + list(messages)
    elif has_system:
        # Cache the system prompt for this session
        for m in messages:
            if m.role == "system" and isinstance(m.content, str) and m.content.strip():
                _get_or_create_session(session_id or "default", system_prompt=m.content)
                break

    # ── Hard limit: reject requests exceeding MAX_INPUT_TOKENS ────────────
    estimated_tokens = _estimate_tokens(messages)
    if estimated_tokens > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Input too large: ~{estimated_tokens} tokens exceeds limit of {MAX_INPUT_TOKENS}. "
            f"Reduce message count or shorten long texts."
        )

    # ── Chunk long messages: store full text in RAG, use summary ──────────
    if estimated_tokens > CHUNK_THRESHOLD_TOKENS:
        messages = _chunk_messages(messages, session_id or "default")

    # ── RAG: store overflow messages and retrieve relevant context ────────
    rag_context = ""
    budget = _get_context_budget(messages)  # Dynamic: fewer msgs when images present

    if RAG_ENABLED and len(messages) > budget:
        try:
            rag = get_rag_store()
            # Extract the latest user query for retrieval
            last_user_msg = ""
            for msg in reversed(messages):
                if msg.role == "user":
                    content = msg.content if isinstance(msg.content, str) else ""
                    last_user_msg = content
                    break

            # Store older messages into RAG (using stable session_id)
            overflow = messages[:-(budget - 1)]
            rag.store_messages(session_id or "default", [
                {"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)}
                for m in overflow if m.content
            ])

            # Retrieve relevant context
            if last_user_msg:
                rag_context = rag.retrieve(last_user_msg, session_id=session_id or "default")

            # Trim to window: first (system) + last N-1
            if messages[0].role == "system":
                messages = [messages[0]] + messages[-(budget - 1):]
            else:
                messages = messages[-(budget):]
            log.info(f"RAG: stored {len(overflow)} msgs, retrieved {len(rag_context)} chars (budget={budget})")
        except Exception as exc:
            log.warning(f"RAG failed (non-fatal): {exc}")
            # Fallback: simple truncation
            if messages[0].role == "system":
                messages = [messages[0]] + messages[-(budget - 1):]
            else:
                messages = messages[-(budget):]
    elif len(messages) > budget:
        # RAG disabled: simple truncation (original behavior)
        if messages[0].role == "system":
            messages = [messages[0]] + messages[-(budget - 1):]
        else:
            messages = messages[-(budget):]
        log.info(f"Trimmed context to {len(messages)} messages (budget={budget}, has_vision={budget < CONTEXT_MESSAGES_TEXT})")

    # Inject RAG context as system message
    if rag_context:
        rag_msg = ChatMessage(
            role="system",
            content=f"Relevant context from earlier conversation:\n{rag_context}"
        )
        # Insert after existing system prompt or at position 0
        if messages and messages[0].role == "system":
            messages = [messages[0], rag_msg] + messages[1:]
        else:
            messages = [rag_msg] + messages

    has_images = False
    all_images: list[Image.Image] = []
    for msg in messages:
        _, imgs = _parse_content(msg.content)
        if imgs:
            has_images = True
            all_images.extend(imgs)

    chat_messages = _build_messages(messages)

    # use apply_chat_template with tools and thinking support
    text = processor.apply_chat_template(
        chat_messages,
        tools=tools,
        enable_thinking=enable_thinking,
        tokenize=False,
        add_generation_prompt=True,
    )

    if has_images:
        inputs = processor(text=[text], images=all_images[:1], return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")

    inputs_gpu = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    input_len = inputs_gpu["input_ids"].shape[-1]
    
    # ── Final token count check (after chunking, with real tokenizer) ────
    if input_len > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Tokenized input ({input_len} tokens) exceeds limit ({MAX_INPUT_TOKENS}). "
            f"The text was too long even after chunking. Shorten your message."
        )
    if input_len > CHUNK_THRESHOLD_TOKENS:
        log.warning(
            f"Large input: {input_len} tokens (threshold={CHUNK_THRESHOLD_TOKENS}). "
            f"Generation may be slow on 12GB VRAM."
        )
    
    return inputs_gpu, input_len, has_images


def generate_sync(
    messages: list[ChatMessage],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    stop: list[str] | None = None,
    tools: list[dict] | None = None,
    enable_thinking: bool = False,
    session_id: Optional[str] = None,
) -> tuple[str, int, int, str | None, list[dict] | None]:
    """Run synchronous generation; return (text, prompt_tokens, completion_tokens, reasoning, tool_calls)."""
    model, processor = load_model(MODEL_PATH)
    inputs_gpu, input_len, _ = _prepare_inputs(
        model, processor, messages, tools=tools, enable_thinking=enable_thinking,
        session_id=session_id,
    )

    gen_kwargs: dict = {
        "max_new_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "do_sample": temperature > 0,
    }
    if stop:
        criteria = StopSequenceCriteria(processor.tokenizer, stop)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])

    with torch.no_grad():
        outputs = model.generate(**inputs_gpu, **gen_kwargs)

    # Decode full output WITH special tokens (parse_response needs them)
    full_text = processor.decode(outputs[0], skip_special_tokens=False)

    # ── Use transformers' official Gemma 4 response parser ──────────────
    # parse_response() reads the response_schema from tokenizer_config.json
    # which contains the official Google regex for tool_calls, thinking, content.
    # It handles <|\"|> → JSON, unquoted keys, and all Gemma 4 markers.
    # IMPORTANT: pass only the generated tokens (not the prompt) to avoid
    # the parser matching the tool declarations in the prompt.
    generated_ids = outputs[0][input_len:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=False)
    parsed = processor.parse_response(generated_text)

    # Extract parsed fields
    reasoning_content = parsed.get("thinking")
    response_text = parsed.get("content") or ""
    # Strip remaining model artifacts
    for marker in ["<eos>", "<pad>", "<bos>", "<turn|>"]:
        response_text = response_text.replace(marker, "").strip()

    # Convert parsed tool_calls to OpenAI format
    tool_calls = None
    raw_tool_calls = parsed.get("tool_calls")
    if raw_tool_calls:
        tool_calls = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            # Arguments come as parsed dict — serialize back to JSON string (OpenAI spec)
            args_str = json.dumps(func.get("arguments", {}), ensure_ascii=False)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args_str,
                },
            })
        # Auto-fix bare domains in URLs (e.g. "todorelatos.com" → "https://todorelatos.com")
        tool_calls = _fix_tool_call_urls(tool_calls)

    # Stop sequence truncation
    if stop:
        for seq in stop:
            idx = response_text.find(seq)
            if idx != -1:
                response_text = response_text[:idx]
                break

    comp_tokens = len(processor.tokenizer.encode(response_text, add_special_tokens=False))
    return response_text, input_len, comp_tokens, reasoning_content, tool_calls


# ── FastAPI lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB + load keys cache + load model on startup; release GPU on shutdown."""
    global INFERENCE_SEMAPHORE, SERVER_START_TIME
    SERVER_START_TIME = time.time()
    INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

    if not MASTER_KEY:
        log.warning(
            "GEMMA4_MASTER_KEY / GEMMA4_ADMIN_KEY not set — "
            "admin endpoints disabled, only GEMMA4_API_KEY accepted"
        )

    loop = asyncio.get_event_loop()

    log.info("Initializing database...")
    await loop.run_in_executor(None, _init_db_sync)

    log.info("Loading API keys into cache...")
    await loop.run_in_executor(None, _load_keys_startup)

    log.info("Pre-loading model...")
    await loop.run_in_executor(None, load_model, MODEL_PATH)
    log.info("Server ready!")

    try:
        yield
    finally:
        log.info("Shutting down...")
        await loop.run_in_executor(None, unload_model)
        log.info("Cleanup complete.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gemma 4 E4B Heretic — OpenAI-Compatible API",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ── Request logging middleware ────────────────────────────────────────────────

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach X-Request-ID to every response."""
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Request-ID"] = request_id
    log.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"{response.status_code} {duration:.2f}s"
    )
    return response


# ── Timestamp formatter ───────────────────────────────────────────────────────

def _fmt_ts(ts: Optional[float]) -> Optional[str]:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) if ts else None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    loaded = MODEL_OBJ is not None
    vram: dict = {}
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram["used_gb"]  = round(used, 2)
        vram["free_gb"]  = round(total - used, 2)
        vram["total_gb"] = round(total, 2)
    return {
        "status": "ok" if loaded else "loading",
        "model_loaded": loaded,
        "model_id": MODEL_ID,
        "vram": vram,
        "last_request_at": LAST_REQUEST_TIME,
        "requests_in_flight": REQUESTS_IN_FLIGHT,
        "uptime_seconds": time.time() - SERVER_START_TIME,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(SERVER_START_TIME),
                "owned_by": "local",
                "permission": [],
            }
        ],
    }


@app.get("/v1/models/{model}")
async def get_model(model: str):
    """Get model details (OpenAI-compatible)."""
    if model != MODEL_ID:
        return JSONResponse(
            status_code=404,
            content=_error_response(f"Model '{model}' not found", "invalid_request_error"),
        )
    return {
        "id": MODEL_ID,
        "object": "model",
        "created": int(SERVER_START_TIME),
        "owned_by": "local",
        "permission": [],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, raw_request: Request):
    global LAST_REQUEST_TIME, REQUESTS_IN_FLIGHT
    LAST_REQUEST_TIME = time.time()
    with REQUESTS_IN_FLIGHT_LOCK:
        REQUESTS_IN_FLIGHT += 1

    try:
        key_prefix = await _check_api_key(raw_request)
        if not key_prefix:
            return JSONResponse(
                status_code=401,
                content=_error_response("Invalid or missing API key", "authentication_error"),
            )

        req_id = raw_request.state.request_id
        created = int(time.time())
        req_start = time.time()

        log.info(
            f"[{req_id}] request: {len(req.messages)} msgs, stream={req.stream}, "
            f"max_tokens={req.max_tokens}"
        )

        # ── Derive stable session ID for RAG ───────────────────────────────
        # Priority: X-Session-ID header > hash of first user message > "default"
        session_id = raw_request.headers.get("X-Session-ID") or _derive_session_id(req.messages)

        loop = asyncio.get_event_loop()
        rate_limit = await loop.run_in_executor(None, _get_key_rate_limit_sync, key_prefix)

        if not RATE_LIMITER.is_allowed(key_prefix, rate_limit):
            log.warning(f"[{req_id}] rate limit exceeded for {key_prefix}")
            return JSONResponse(
                status_code=429,
                content=_error_response(
                    f"Rate limit exceeded: {rate_limit} requests per minute", "rate_limit_error"
                ),
            )

        if INFERENCE_SEMAPHORE is None:
            return JSONResponse(
                status_code=503,
                content=_error_response("Server not ready", "server_error"),
            )

        if req.stream:
            # Acquire before returning generator; released in generator's finally block
            await INFERENCE_SEMAPHORE.acquire()
            return EventSourceResponse(
                _stream_generator(req, req_id, created, key_prefix, req_start, INFERENCE_SEMAPHORE, session_id),
                media_type="text/event-stream",
            )

        async with INFERENCE_SEMAPHORE:
            try:
                # Determine thinking: reasoning_effort takes precedence
                enable_thinking = req.enable_thinking
                if req.reasoning_effort:
                    enable_thinking = req.reasoning_effort != "low"

                response_text, prompt_tokens, completion_tokens, reasoning, tool_calls = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, generate_sync,
                        req.messages, req.max_tokens, req.temperature, req.top_p, req.stop,
                        req.tools, enable_thinking, session_id,
                    ),
                    timeout=GENERATION_TIMEOUT,
                )
                latency_ms = (time.time() - req_start) * 1000
                await loop.run_in_executor(
                    None, _log_usage_sync,
                    key_prefix, req.model, prompt_tokens, completion_tokens, latency_ms,
                )
                log.info(
                    f"[{req_id}] completed: {prompt_tokens}+{completion_tokens} tokens, "
                    f"{latency_ms:.0f}ms"
                )

                # Build response message with optional reasoning and tool_calls
                message: dict = {"role": "assistant"}
                if tool_calls:
                    message["content"] = None  # OpenAI: null content when tool_calls present
                    message["tool_calls"] = tool_calls
                else:
                    message["content"] = response_text
                if reasoning:
                    message["reasoning_content"] = reasoning

                # Determine finish_reason
                finish_reason = "tool_calls" if tool_calls else "stop"

                return ChatCompletionResponse(
                    id=req_id,
                    created=created,
                    model=req.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=message,
                            finish_reason=finish_reason,
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                log.error(f"[{req_id}] CUDA OOM!")
                return JSONResponse(
                    status_code=503,
                    content=_error_response("GPU out of memory — request too large", "server_error"),
                )
            except ValueError as exc:
                log.warning(f"[{req_id}] validation error: {exc}")
                return JSONResponse(
                    status_code=400,
                    content=_error_response(str(exc), "invalid_request_error"),
                )
            except asyncio.TimeoutError:
                log.error(f"[{req_id}] generation timeout ({GENERATION_TIMEOUT}s)")
                torch.cuda.empty_cache()
                gc.collect()
                return JSONResponse(
                    status_code=504,
                    content=_error_response(
                        f"Generation timed out after {GENERATION_TIMEOUT}s — input too large or model too slow. "
                        f"Reduce message length and try again.",
                        "timeout_error",
                    ),
                )
            except Exception as exc:
                log.exception(f"[{req_id}] inference failed")
                return JSONResponse(
                    status_code=500,
                    content=_error_response(f"Generation failed: {str(exc)}", "server_error"),
                )
    finally:
        # Aggressive memory cleanup after every request (critical for 12GB VRAM)
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: (gc.collect(), torch.cuda.empty_cache())
        )
        with REQUESTS_IN_FLIGHT_LOCK:
            REQUESTS_IN_FLIGHT -= 1


async def _stream_generator(
    req: ChatCompletionRequest,
    req_id: str,
    created: int,
    key_prefix: str,
    req_start: float,
    semaphore: asyncio.Semaphore,
    session_id: str = "default",
) -> AsyncIterator[dict]:
    """Queue-bridged streaming: inference thread -> asyncio -> SSE."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    error_holder: list[Optional[Exception]] = [None]

    def _run_inference() -> None:
        try:
            # Determine thinking: reasoning_effort takes precedence
            enable_thinking = req.enable_thinking
            if req.reasoning_effort:
                enable_thinking = req.reasoning_effort != "low"

            model, processor = load_model(MODEL_PATH)
            inputs_gpu, input_len, _ = _prepare_inputs(
                model, processor, req.messages, tools=req.tools, enable_thinking=enable_thinking,
                session_id=session_id,
            )

            streamer = TextIteratorStreamer(
                processor.tokenizer, skip_prompt=True, skip_special_tokens=False
            )
            gen_kwargs: dict = {
                **inputs_gpu,
                "max_new_tokens": req.max_tokens,
                "temperature": req.temperature if req.temperature > 0 else None,
                "top_p": req.top_p,
                "do_sample": req.temperature > 0,
                "streamer": streamer,
            }
            if req.stop:
                criteria = StopSequenceCriteria(processor.tokenizer, req.stop)
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])

            thread = threading.Thread(
                target=lambda: model.generate(**gen_kwargs), daemon=True
            )
            thread.start()

            accumulated = ""
            in_thinking = False
            in_tool = False
            asyncio.run_coroutine_threadsafe(queue.put(("role", input_len)), loop)

            for chunk in streamer:
                if chunk:
                    accumulated += chunk

                    # ── Thinking blocks: stream thinking tokens to client ──
                    if "<|think|>" in chunk:
                        in_thinking = True
                        asyncio.run_coroutine_threadsafe(queue.put(("thinking_start",)), loop)
                    elif "<|turn|>" in chunk and in_thinking:
                        in_thinking = False
                        asyncio.run_coroutine_threadsafe(queue.put(("thinking_end",)), loop)
                    elif in_thinking:
                        asyncio.run_coroutine_threadsafe(queue.put(("thinking", chunk)), loop)

                    # ── Tool call blocks: suppress from content stream ──
                    elif "<|tool_call>" in chunk:
                        in_tool = True
                    elif "<tool_call|>" in chunk:
                        in_tool = False
                    elif in_tool:
                        pass  # Suppress tool call tokens from content stream

                    # ── Content: only stream clean text ──
                    elif not in_thinking:
                        # Skip <|turn|> markers (model EOS token)
                        if "<turn|>" not in chunk:
                            asyncio.run_coroutine_threadsafe(queue.put(("content", chunk)), loop)

            thread.join()

            # ── Parse final output using transformers' official Gemma 4 parser ──
            parsed = processor.parse_response(accumulated)
            reasoning_content = parsed.get("thinking")

            tool_calls = None
            raw_tool_calls = parsed.get("tool_calls")
            if raw_tool_calls:
                tool_calls = []
                for tc in raw_tool_calls:
                    func = tc.get("function", {})
                    args_str = json.dumps(func.get("arguments", {}), ensure_ascii=False)
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": func.get("name", ""), "arguments": args_str},
                    })
                # Auto-fix bare domains in URLs
                tool_calls = _fix_tool_call_urls(tool_calls)

            # Clean accumulated for token count
            clean_accumulated = parsed.get("content") or ""
            for marker in ["<eos>", "<pad>", "<bos>"]:
                clean_accumulated = clean_accumulated.replace(marker, "").strip()

            comp_tokens = len(
                processor.tokenizer.encode(clean_accumulated, add_special_tokens=False)
            )
            asyncio.run_coroutine_threadsafe(
                queue.put(("done", input_len, comp_tokens, reasoning_content, tool_calls)), loop
            )
        except Exception as exc:
            error_holder[0] = exc
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop)

    loop.run_in_executor(None, _run_inference)

    total_prompt = 0
    total_completion = 0
    current_reasoning = ""
    current_tool_calls = None

    try:
        while True:
            item = await queue.get()
            tag = item[0]

            if tag == "role":
                total_prompt = item[1]
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                        ],
                    }),
                }

            elif tag == "thinking_start":
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {"index": 0, "delta": {"reasoning_content": ""}, "finish_reason": None}
                        ],
                    }),
                }

            elif tag == "thinking":
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {"index": 0, "delta": {"reasoning_content": item[1]}, "finish_reason": None}
                        ],
                    }),
                }

            elif tag == "thinking_end":
                pass  # no-op, next chunk type will be content

            elif tag == "content":
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {"index": 0, "delta": {"content": item[1]}, "finish_reason": None}
                        ],
                    }),
                }

            elif tag == "done":
                total_prompt = item[1]
                total_completion = item[2]
                final_reasoning = item[3]
                final_tool_calls = item[4]
                latency_ms = (time.time() - req_start) * 1000

                await asyncio.get_running_loop().run_in_executor(
                    None, _log_usage_sync,
                    key_prefix, req.model, total_prompt, total_completion, latency_ms,
                )

                # Build final delta
                final_delta: dict = {}
                if final_tool_calls:
                    final_delta["tool_calls"] = final_tool_calls

                # Determine finish_reason
                finish_reason = "tool_calls" if final_tool_calls else "stop"

                # Check stream_options.include_usage
                include_usage = req.stream_options and req.stream_options.get("include_usage", False)

                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": final_delta, "finish_reason": finish_reason}],
                }
                if include_usage:
                    final_chunk["usage"] = {
                        "prompt_tokens": total_prompt,
                        "completion_tokens": total_completion,
                        "total_tokens": total_prompt + total_completion,
                    }

                yield {
                    "event": "message",
                    "data": json.dumps(final_chunk),
                }
                yield {"event": "message", "data": "[DONE]"}
                log.info(
                    f"[{req_id}] stream completed: "
                    f"{total_prompt}+{total_completion} tokens, {latency_ms:.0f}ms"
                )
                break

            elif tag == "error":
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {"index": 0, "delta": {"content": f"\n[Error: {item[1]}]"}, "finish_reason": "stop"}
                        ],
                    }),
                }
                yield {"event": "message", "data": "[DONE]"}
                break
    finally:
        # Aggressive memory cleanup after streaming (critical for 12GB VRAM)
        gc.collect()
        torch.cuda.empty_cache()
        semaphore.release()


# ── Admin: key management ─────────────────────────────────────────────────────

@app.post("/v1/keys")
async def create_key(body: CreateKeyRequest, raw_request: Request):
    """Generate a new managed API key."""
    if not _check_master_key(raw_request):
        return JSONResponse(
            status_code=401,
            content=_error_response("Invalid or missing admin key", "authentication_error"),
        )

    req_id = raw_request.state.request_id
    key = f"g4h-{secrets.token_hex(16)}"
    key_hash = _hash_key(key)
    key_prefix = key[:8]
    created_ts = time.time()
    expires_at = created_ts + (body.expires_in_days * 86400) if body.expires_in_days else None

    key_data = {
        "id": None,  # Set by DB autoincrement; not tracked in cache
        "key_hash": key_hash,
        "key_prefix": key_prefix,
        "name": body.name,
        "created_at": created_ts,
        "expires_at": expires_at,
        "last_used_at": None,
        "is_active": 1,
        "usage_count": 0,
        "total_tokens": 0,
        "rate_limit": body.rate_limit,
    }

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _create_key, key_prefix, key_data)
    log.info(f"[{req_id}] Created API key: {key_prefix} (name={body.name!r})")

    resp: dict = {
        "key_prefix": key_prefix,
        "key": key,
        "name": body.name,
        "created_at": _fmt_ts(created_ts),
        "expires_at": _fmt_ts(expires_at),
        "last_used_at": None,
        "is_active": True,
        "usage_count": 0,
        "total_tokens": 0,
        "rate_limit": body.rate_limit,
    }
    return resp


@app.get("/v1/keys")
async def list_keys(raw_request: Request):
    """List all managed API keys (prefix + metadata, never the secret)."""
    if not _check_master_key(raw_request):
        return JSONResponse(
            status_code=401,
            content=_error_response("Invalid or missing admin key", "authentication_error"),
        )

    all_keys = _get_all_keys()
    # Sort by created_at descending
    sorted_keys = sorted(all_keys.values(), key=lambda k: k.get("created_at", 0), reverse=True)

    return {
        "keys": [
            {
                "key_prefix": k["key_prefix"],
                "name": k["name"],
                "created_at": _fmt_ts(k.get("created_at")),
                "expires_at": _fmt_ts(k.get("expires_at")),
                "last_used_at": _fmt_ts(k.get("last_used_at")),
                "is_active": bool(k.get("is_active", 1)),
                "usage_count": k.get("usage_count", 0),
                "total_tokens": k.get("total_tokens", 0),
                "rate_limit": k.get("rate_limit", DEFAULT_RATE_LIMIT),
            }
            for k in sorted_keys
        ]
    }


@app.get("/v1/keys/{key_prefix}/usage")
async def get_key_usage(key_prefix: str, raw_request: Request):
    """Return per-key usage statistics."""
    if not _check_master_key(raw_request):
        return JSONResponse(
            status_code=401,
            content=_error_response("Invalid or missing admin key", "authentication_error"),
        )

    def _query():
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM api_keys WHERE key_prefix = ?", (key_prefix,)
            ).fetchone()
            if not row:
                return None, None
            stats = conn.execute(
                """
                SELECT COUNT(*) AS requests,
                       COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
                  FROM request_log WHERE key_prefix = ?
                """,
                (key_prefix,),
            ).fetchone()
            return row, stats

    loop = asyncio.get_event_loop()
    row, stats = await loop.run_in_executor(None, _query)

    if not row:
        return JSONResponse(
            status_code=404,
            content=_error_response(f"Key '{key_prefix}' not found", "invalid_request_error"),
        )

    return {
        "key_prefix": key_prefix,
        "name": row["name"],
        "is_active": bool(row["is_active"]),
        "created_at": _fmt_ts(row["created_at"]),
        "expires_at": _fmt_ts(row["expires_at"]),
        "last_used_at": _fmt_ts(row["last_used_at"]),
        "usage_count": row["usage_count"],
        "total_tokens": row["total_tokens"],
        "avg_latency_ms": round(stats[1], 2) if stats else 0,
    }


@app.post("/v1/keys/{key_prefix}/rotate")
async def rotate_key(key_prefix: str, raw_request: Request):
    """Regenerate the secret for an existing key slot."""
    if not _check_master_key(raw_request):
        return JSONResponse(
            status_code=401,
            content=_error_response("Invalid or missing admin key", "authentication_error"),
        )

    new_key = f"g4h-{secrets.token_hex(16)}"
    new_hash = _hash_key(new_key)
    rotated_ts = time.time()

    with _KEYS_CACHE_LOCK:
        if key_prefix not in _KEYS_CACHE:
            return JSONResponse(
                status_code=404,
                content=_error_response(f"Key '{key_prefix}' not found", "invalid_request_error"),
            )
        old_hash = _KEYS_CACHE[key_prefix].get("key_hash")
        # Update cache
        _KEYS_CACHE[key_prefix]["key_hash"] = new_hash
        # Update hash index
        if old_hash and old_hash in _KEY_HASH_INDEX:
            del _KEY_HASH_INDEX[old_hash]
        _KEY_HASH_INDEX[new_hash] = key_prefix
        # Persist single key to SQLite
        _persist_key(key_prefix, _KEYS_CACHE[key_prefix])

    req_id = raw_request.state.request_id
    log.info(f"[{req_id}] Rotated API key: {key_prefix}")

    # Get data from cache for response (already updated)
    key_data = _get_key(key_prefix)

    return {
        "key_prefix": key_prefix,
        "key": new_key,
        "name": key_data["name"] if key_data else "unknown",
        "created_at": _fmt_ts(key_data["created_at"]) if key_data else None,
        "rotated_at": _fmt_ts(rotated_ts),
    }


@app.delete("/v1/keys/{key_prefix}")
async def delete_key(key_prefix: str, raw_request: Request):
    """Permanently delete an API key (cache + SQLite)."""
    if not _check_master_key(raw_request):
        return JSONResponse(
            status_code=401,
            content=_error_response("Invalid or missing admin key", "authentication_error"),
        )

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _delete_key, key_prefix)
    except KeyError:
        return JSONResponse(
            status_code=404,
            content=_error_response(f"Key '{key_prefix}' not found", "invalid_request_error"),
        )

    req_id = raw_request.state.request_id
    log.info(f"[{req_id}] Deleted API key: {key_prefix}")
    return {"status": "ok", "key_prefix": key_prefix}


# ── Admin: usage analytics ────────────────────────────────────────────────────

@app.get("/v1/usage")
async def usage_stats(
    raw_request: Request,
    key_id: Optional[str] = Query(None, description="Filter by key_prefix"),
    from_ts: Optional[float] = Query(None, alias="from", description="Start Unix timestamp"),
    to_ts: Optional[float] = Query(None, alias="to", description="End Unix timestamp"),
):
    """Return aggregated request statistics from the request log."""
    if not _check_master_key(raw_request):
        return JSONResponse(
            status_code=401,
            content=_error_response("Invalid or missing admin key", "authentication_error"),
        )

    def _query():
        conditions: list[str] = []
        params: list = []
        if key_id:
            conditions.append("key_prefix = ?")
            params.append(key_id)
        if from_ts is not None:
            conditions.append("timestamp >= ?")
            params.append(from_ts)
        if to_ts is not None:
            conditions.append("timestamp <= ?")
            params.append(to_ts)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with sqlite3.connect(DB_PATH) as conn:
            summary = conn.execute(
                f"""
                SELECT
                    COUNT(*)                           AS total_requests,
                    COALESCE(SUM(prompt_tokens),     0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(total_tokens),      0) AS total_tokens,
                    COALESCE(AVG(latency_ms),        0) AS avg_latency_ms
                FROM request_log {where}
                """,
                params,
            ).fetchone()

            by_key = conn.execute(
                f"""
                SELECT key_prefix,
                       COUNT(*)          AS requests,
                       SUM(total_tokens) AS tokens,
                       AVG(latency_ms)   AS avg_latency_ms
                  FROM request_log {where}
                 GROUP BY key_prefix
                 ORDER BY requests DESC
                """,
                params,
            ).fetchall()

        return summary, by_key

    loop = asyncio.get_event_loop()
    summary, per_key = await loop.run_in_executor(None, _query)

    return {
        "total_requests": summary[0],
        "total_prompt_tokens": summary[1],
        "total_completion_tokens": summary[2],
        "total_tokens": summary[3],
        "avg_latency_ms": round(summary[4], 2),
        "by_key": [
            {
                "key_prefix": r[0],
                "requests": r[1],
                "total_tokens": r[2],
                "avg_latency_ms": round(r[3], 2),
            }
            for r in per_key
        ],
    }


# ── Admin: server config ──────────────────────────────────────────────────────

@app.get("/v1/config")
async def config(raw_request: Request):
    """Return live server configuration."""
    if MASTER_KEY and not _check_master_key(raw_request):
        return JSONResponse(
            status_code=401,
            content=_error_response("Invalid or missing admin key", "authentication_error"),
        )

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_used  = round(torch.cuda.memory_allocated() / 1024**3, 2)
        vram_total = round(props.total_memory / 1024**3, 2)
    else:
        vram_used = vram_total = 0.0

    def _count_active() -> int:
        with sqlite3.connect(DB_PATH) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM api_keys WHERE is_active = 1"
            ).fetchone()[0]

    loop = asyncio.get_event_loop()
    active_keys = await loop.run_in_executor(None, _count_active)

    return ConfigResponse(
        model=MODEL_ID,
        version="2.1.0",
        vram_used_gb=vram_used,
        vram_total_gb=vram_total,
        active_keys=active_keys,
        uptime_seconds=round(time.time() - SERVER_START_TIME, 1),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 4 OpenAI-compatible API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=300,
    )
