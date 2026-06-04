# Software Design Document: Gemma 4 E4B Heretic API Server

> **Version:** 2.1.0 | **Commit:** b0df36c | **Date:** 2026-06-04

---

## 1. Overview

### 1.1 Purpose
Provide a production-grade, OpenAI-compatible API endpoint for running the Gemma 4 E4B Heretic model on consumer hardware (RTX 3060 12GB via Thunderbolt 3). Enables integration with any OpenAI-compatible client (Hermes, Open WebUI, custom scripts).

### 1.2 Design Goals
| Goal | Implementation |
|------|---------------|
| OpenAI API compatibility | Same request/response format as `/v1/chat/completions` |
| Single-GPU efficiency | `asyncio.Semaphore(1)` prevents VRAM OOM |
| Low-latency auth | In-memory cache + hash index (O(1) lookup) |
| Non-blocking I/O | SQLite writes via `run_in_executor` |
| Key management | Create, rotate, revoke, expire, rate-limit per key |
| Analytics | Per-request logging (tokens, latency) |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Client (curl/API/…)                │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP/SSE
                       ▼
┌──────────────────────────────────────────────────────┐
│              FastAPI (uvicorn)                       │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │ Auth MW  │→ │ Rate Limit│→ │ Request Handler  │  │
│  └──────────┘  └───────────┘  └────────┬─────────┘  │
│                                        │             │
│  ┌─────────────────┐     ┌─────────────▼──────────┐  │
│  │ _KEYS_CACHE     │     │ asyncio.Semaphore(1)   │  │
│  │ (in-memory)     │     │ (GPU concurrency)       │  │
│  │ _KEY_HASH_INDEX │     └─────────────┬──────────┘  │
│  │ threading.RLock │                   │             │
│  └────────┬────────┘                   ▼             │
│           │                     ┌──────────────┐     │
│           ▼                     │ Transformers  │     │
│  ┌────────────────┐            │ Model Pipeline│     │
│  │ SQLite (keys.db)│            │ (GPU: RTX 3060)│    │
│  │ api_keys       │            └──────────────┘     │
│  │ request_log    │                                 │
│  └────────────────┘                                 │
└──────────────────────────────────────────────────────┘
```

### 2.1 Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **FastAPI Routes** | HTTP request handling, SSE streaming, response formatting |
| **Auth Middleware** | Bearer token extraction, key hash validation via `_KEY_HASH_INDEX` |
| **Key Manager** | Create/rotate/delete keys in cache + SQLite under `RLock` |
| **Rate Limiter** | Per-key sliding window (tokens/min) via cache |
| **Inference Engine** | `model.generate()` / `StopSequenceCriteria` on GPU |
| **Usage Logger** | Async SQLite insert via `run_in_executor` |

### 2.2 Concurrency Model

| Mechanism | Purpose | Scope |
|-----------|---------|-------|
| `asyncio.Semaphore(1)` | GPU inference queue (1 at a time) | Async |
| `threading.RLock()` | Cache read-modify-write atomicity | Sync |
| `threading.Lock()` | `REQUESTS_IN_FLIGHT` counter | Sync |
| `run_in_executor()` | SQLite writes, `_log_usage_sync` | Thread pool |

---

## 3. Data Model

### 3.1 SQLite Schema

**Table: `api_keys`**
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `key_hash` | TEXT UNIQUE | SHA-256 hash of the API key |
| `key_prefix` | TEXT | Short identifier (first 8 chars hex) |
| `name` | TEXT | Human-readable label |
| `created_at` | REAL | Unix timestamp |
| `expires_at` | REAL NULL | Expiration timestamp (NULL = never) |
| `last_used_at` | REAL NULL | Last successful request |
| `is_active` | INTEGER | 1=active, 0=revoked |
| `usage_count` | INTEGER | Total requests served |
| `total_tokens` | INTEGER | Cumulative token usage |
| `rate_limit` | INTEGER | Max requests/minute (default: 30) |

**Table: `request_log`**
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `timestamp` | REAL | Unix timestamp |
| `key_prefix` | TEXT | Key that made the request |
| `model` | TEXT | Model ID used |
| `prompt_tokens` | INTEGER | Input tokens |
| `completion_tokens` | INTEGER | Output tokens |
| `total_tokens` | INTEGER | Sum of both |
| `latency_ms` | REAL | End-to-end latency |

### 3.2 In-Memory Structures

```python
_KEYS_CACHE: dict[str, dict]         # key_prefix → full row data
_KEY_HASH_INDEX: dict[str, str]      # key_hash → key_prefix (O(1) validation)
_KEYS_CACHE_LOCK = threading.RLock() # Guards both structures
```

### 3.3 Key Hashing
```
API Key: "g4h-a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
Hash:    SHA-256(key) → stored in SQLite + _KEY_HASH_INDEX
Prefix:  "a1b2c3d4" (first 8 hex chars, used as ID)
```

---

## 4. API Reference

### 4.1 Inference Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/chat/completions` | API Key | OpenAI-compatible chat completions (stream/non-stream) |
| `GET` | `/v1/models` | API Key | List available models |
| `GET` | `/v1/models/{model}` | API Key | Get model details |
| `GET` | `/health` | None | Health check (model loaded, VRAM, uptime, requests in flight) |

### 4.2 Admin Endpoints (require `GEMMA4_MASTER_KEY` or `GEMMA4_ADMIN_KEY`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/keys` | Create new API key |
| `GET` | `/v1/keys` | List all keys |
| `GET` | `/v1/keys/{key_prefix}/usage` | Per-key usage stats |
| `POST` | `/v1/keys/{key_prefix}/rotate` | Regenerate key secret |
| `DELETE` | `/v1/keys/{key_prefix}` | Permanently delete key |
| `GET` | `/v1/usage` | Global usage analytics |
| `GET` | `/v1/config` | Live server configuration |

### 4.3 Chat Completion Request Fields

| Field | Type | Supported | Notes |
|-------|------|-----------|-------|
| `model` | string | ✅ | Mapped to local model |
| `messages` | array | ✅ | Roles: system, user, assistant, tool |
| `max_tokens` | int | ✅ | Default: 512 |
| `temperature` | float | ✅ | Default: 0.7 |
| `top_p` | float | ✅ | Default: 0.9 |
| `stream` | bool | ✅ | SSE via `EventSourceResponse` |
| `frequency_penalty` | float | ✅ | Mapped to `repetition_penalty` |
| `presence_penalty` | float | ✅ | Combined with frequency |
| `stop` | array | ✅ | Via `StopSequenceCriteria` |
| `images` | array | ✅ | Vision input (base64, URL, local path) |
| `stream_options` | object | ✅ | `include_usage: true` adds usage to final stream chunk |
| `n` | int | ❌ | Constrained to 1 (single GPU) |
| `logprobs` | bool | ❌ | Not supported by transformers |
| `top_logprobs` | int | ❌ | Not supported by transformers |
---

## 5. Security

### 5.1 Authentication Flow
1. Client sends `Authorization: Bearer <key>` or `X-API-Key: <key>`
2. Server computes `SHA-256(key)` and looks up in `_KEY_HASH_INDEX` → O(1)
3. If found: checks `is_active`, `expires_at`, rate limit
4. If not found: falls back to legacy `GEMMA4_API_KEY` env var

### 5.2 SSRF Protection
Image URLs are validated against private IP ranges before fetching. Blocking: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`, `::1`, `localhost`. Local file paths are restricted to `GEMMA4_IMAGE_DIR`.

### 5.3 Request Tracing
Every response includes `X-Request-ID` header (format: `req-{hex12}`) generated via middleware for log correlation.

### 5.4 Rate Limiting
Per-key sliding window via `usage_count` + `last_used_at` in cache. Default: 30 req/min. Configurable per key. **Note:** Rate limit state is in-memory only — resets on server restart. Persistent usage totals (`total_tokens`, `usage_count`) survive restarts via SQLite.

---

## 6. Key Lifecycle

```
CREATE ──→ ACTIVE ──→ ROTATE ──→ ACTIVE (new secret)
  │            │
  │            └──→ DELETE ──→ removed (cache + SQLite)
  │
  └──→ EXPIRED ──→ rejected (checked on each request)
```

- **Create**: `POST /v1/keys` → generates `g4h-{64hex}`, stores hash in DB + cache
- **Rotate**: `POST /v1/keys/{prefix}/rotate` → new secret, updates cache + hash index atomically under `RLock`
- **Delete**: `DELETE /v1/keys/{prefix}` → removes from cache + hash index + SQLite DELETE under `RLock`
- **Expire**: `expires_at` checked in `_validate_key_sync()` every request

---

## 7. Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Key validation | O(1) | Hash index lookup |
| Persistence | O(1) per key | Single-row upsert |
| Concurrent requests | 1 | GPU semaphore |
| Cold start | ~70s | Model load into VRAM |
| Event loop blocking | 0ms | All I/O in executor |
| VRAM usage | 9.3GB / 11.6GB | NF4 + fp16 vision tower |

---

## 8. Deployment

### 8.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMMA4_API_KEY` | `gemma4-local` | Legacy single-key auth |
| `GEMMA4_MASTER_KEY` | (none) | Admin key for management endpoints |
| `GEMMA4_ADMIN_KEY` | (none) | Alias for MASTER_KEY |
| `GEMMA4_MODEL_PATH` | `~/models/gemma4-heretic` | Model directory |
| `GEMMA4_MAX_IMAGE_MB` | `20` | Max image size for vision |
| `GEMMA4_CORS_ORIGINS` | `*` | Allowed CORS origins |
| `GEMMA4_IMAGE_DIR` | (none) | Allowed local image directory |

### 8.2 Filesystem Paths

| Path | Purpose |
|------|---------|
| `~/.gemma4api/keys.db` | SQLite database |
| `~/.config/gemma4-api/usage.jsonl` | Legacy request log (migrated) |
| `~/.config/systemd/user/gemma4-api.service` | systemd unit |
| `~/models/gemma4-heretic/` | Model weights |

### 8.3 systemd Service
```ini
[Unit]
Description=Gemma 4 E4B Heretic API Server
After=network.target

[Service]
ExecStart=/path/to/.venv/bin/python server.py --host 0.0.0.0 --port 8080
WorkingDirectory=/home/hbuddenberg/lowram-gemma4-vision
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TimeoutStartSec=300

[Install]
WantedBy=default.target
```

---

## 9. Limitations & Future Work

### Known Limitations
- **Single concurrent request**: GPU semaphore limits to 1 (VRAM constraint)
- **No `logprobs`/`top_logprobs`**: Not supported by HuggingFace generate()
- **No `n > 1`**: Constrained to 1 response per request
- **No `stream_options`**: SSE delta-only, no `include_usage` support
- **SQLite single-writer**: Not suitable for multi-node deployment

### Future Work
- [x] `stream_options.include_usage` for streaming token counts
- [ ] Multi-model support (serve different models per endpoint)
- [ ] Web UI for key management
- [ ] Prometheus metrics endpoint
- [ ] PostgreSQL backend for multi-node deployments
- [ ] Request queuing with priority levels
- [ ] Vision: video frame support
