#!/usr/bin/env python3
"""
Gemma 4 E4B Heretic — OpenAI-compatible API server
FastAPI server wrapping bitsandbytes NF4 quantized Gemma 4 for text + vision.

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible (streaming + non-streaming)
  GET  /v1/models            — List available models
  GET  /health               — Health check + VRAM
  POST /v1/keys              — Generate/list/revoke API keys (master-key protected)
  GET  /v1/config            — Server config (master-key protected)

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
import secrets
import signal
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

import requests as _requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
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

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gemma4-server")

# ── Globals ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "GEMMA4_MODEL_PATH", os.path.expanduser("~/models/gemma4-heretic")
)
MODEL_ID = "gemma-4-e4b-heretic"
MODEL_OBJ: Optional[Gemma4ForConditionalGeneration] = None
PROCESSOR: Optional[Gemma4Processor] = None
LOAD_LOCK = threading.Lock()
INFERENCE_SEMAPHORE: Optional[asyncio.Semaphore] = None
SERVER_START_TIME = time.time()

# ── Configurable defaults ────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
API_KEY = os.environ.get("GEMMA4_API_KEY", "gemma4-local")
MASTER_KEY = os.environ.get("GEMMA4_MASTER_KEY", "")
MAX_IMAGE_SIZE_MB = int(os.environ.get("GEMMA4_MAX_IMAGE_MB", "20"))
VALID_ROLES = {"system", "user", "assistant", "tool"}
DEFAULT_RATE_LIMIT = 10  # requests per minute per key

# ── Storage paths ────────────────────────────────────────────────────────────
CONFIG_DIR = Path.home() / ".config" / "gemma4-api"
KEYS_FILE = CONFIG_DIR / "keys.json"
USAGE_LOG = CONFIG_DIR / "usage.jsonl"

# ── Security: allowed image hosts (empty = allow all public) ─────────────────
ALLOWED_IMAGE_DIR = os.environ.get("GEMMA4_IMAGE_DIR", "")


# ── API Key Management ──────────────────────────────────────────────────────


def _init_config_dir():
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_keys() -> dict:
    """Load API keys from storage."""
    if not KEYS_FILE.exists():
        return {}
    try:
        with open(KEYS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_keys(keys: dict):
    """Save API keys to storage."""
    _init_config_dir()
    with open(KEYS_FILE, "w", opener=lambda path, flags: os.open(path, flags, 0o600)) as f:
        json.dump(keys, f, indent=2)


def _hash_key(key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def _validate_key(key: str) -> Optional[str]:
    """Validate API key, return key_prefix if valid."""
    if not key:
        return None

    # Allow legacy key (no key management)
    if key == API_KEY:
        return "legacy"

    # Check against stored keys
    keys = _load_keys()
    key_hash = _hash_key(key)

    for prefix, data in keys.items():
        if data.get("key_hash") == key_hash and data.get("enabled"):
            return prefix

    return None


class RateLimiter:
    """Per-key sliding window rate limiter."""

    def __init__(self):
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, key_prefix: str, limit: int = DEFAULT_RATE_LIMIT) -> bool:
        """Check if request is allowed under rate limit (requests per minute)."""
        now = time.time()
        window_start = now - 60  # 1 minute window

        if key_prefix not in self.requests:
            self.requests[key_prefix] = []

        # Remove old requests outside window
        self.requests[key_prefix] = [t for t in self.requests[key_prefix] if t > window_start]

        # Check limit
        if len(self.requests[key_prefix]) >= limit:
            return False

        # Record request
        self.requests[key_prefix].append(now)
        return True


RATE_LIMITER = RateLimiter()


def _log_usage(key_prefix: str, model: str, prompt_tokens: int,
               completion_tokens: int, latency_ms: float):
    """Log request usage to JSONL file."""
    _init_config_dir()
    usage = {
        "timestamp": time.time(),
        "key_prefix": key_prefix,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_ms": latency_ms,
    }
    with open(USAGE_LOG, "a") as f:
        f.write(json.dumps(usage) + "\n")


# ── Stopping criteria for stop sequences ─────────────────────────────────────


class StopSequenceCriteria(StoppingCriteria):
    """Stop generation when any stop sequence appears in decoded output."""

    def __init__(self, tokenizer, stop_sequences: list[str]):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self._last_len = 0
        self._accumulated = ""

    def __call__(self, input_ids, scores, **kwargs):
        current_len = input_ids.shape[1]
        if current_len > self._last_len:
            new_tokens = self.tokenizer.decode(input_ids[0, self._last_len:], skip_special_tokens=False)
            self._accumulated += new_tokens
            self._last_len = current_len
        return any(seq in self._accumulated for seq in self.stop_sequences)


# ── Pydantic models (OpenAI-compatible) ──────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str | list = ""

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
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

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages must contain at least one message")
        return v


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


# ── API Key Management Models ────────────────────────────────────────────────


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    rate_limit: int = Field(default=DEFAULT_RATE_LIMIT, ge=1, le=1000)


class KeyResponse(BaseModel):
    key_prefix: str
    key: Optional[str] = None  # Only returned on creation
    name: str
    created_at: str
    rate_limit: int
    enabled: bool


class ConfigResponse(BaseModel):
    model: str
    version: str
    vram_used_gb: float
    vram_total_gb: float
    active_keys: int
    uptime_seconds: float


# ── Error Response Helper ────────────────────────────────────────────────────


def _error_response(message: str, error_type: str = "invalid_request_error",
                   code: Optional[str] = None) -> dict:
    """Format error response in OpenAI style."""
    error_dict = {"message": message, "type": error_type}
    if code:
        error_dict["code"] = code
    return {"error": error_dict}


# ── Security helpers ─────────────────────────────────────────────────────────


def _is_private_url(url: str) -> bool:
    """Check if a URL points to a private/internal network."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return True
        if hostname in ("localhost", "127.0.0.1", "::1"):
            return True
        import socket

        addr = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in addr:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return True
    except Exception:
        return True
    return False


def _validate_image_path(path: str) -> bool:
    """Ensure file path is within allowed directory (no traversal)."""
    if not ALLOWED_IMAGE_DIR:
        return False  # No file path loading without explicit config
    real = os.path.realpath(path)
    allowed = os.path.realpath(ALLOWED_IMAGE_DIR)
    return (real.startswith(allowed + os.sep) or real == allowed) and os.path.isfile(real)


def _check_api_key(request: Request) -> Optional[str]:
    """Validate API key from Authorization header, return key_prefix if valid."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        key = auth[7:]
        prefix = _validate_key(key)
        if prefix:
            return prefix

    # Also check X-API-Key header
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        prefix = _validate_key(api_key)
        if prefix:
            return prefix

    return None


def _check_master_key(request: Request) -> bool:
    """Validate master key for admin endpoints."""
    if not MASTER_KEY:
        return False
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:] == MASTER_KEY
    api_key = request.headers.get("X-API-Key", "")
    return api_key == MASTER_KEY


# ── Model loading ────────────────────────────────────────────────────────────


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


def unload_model():
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


# ── Inference helpers ────────────────────────────────────────────────────────


def _parse_content(content: str | list) -> tuple[str, list[Image.Image]]:
    """Extract text prompt and images from OpenAI-format content."""
    if isinstance(content, str):
        return content, []

    text_parts = []
    images = []

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
                        raise ValueError(
                            f"Image exceeds {MAX_IMAGE_SIZE_MB}MB limit"
                        )
                    images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                elif url.startswith("http"):
                    if _is_private_url(url):
                        raise ValueError("Fetching internal/private URLs is blocked")
                    resp = _requests.get(url, timeout=15, stream=True, allow_redirects=False)
                    resp.raise_for_status()
                    size = 0
                    chunks = []
                    for chunk in resp.iter_content(8192):
                        size += len(chunk)
                        if size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                            raise ValueError(
                                f"Image exceeds {MAX_IMAGE_SIZE_MB}MB limit"
                            )
                        chunks.append(chunk)
                    images.append(
                        Image.open(io.BytesIO(b"".join(chunks))).convert("RGB")
                    )
                else:
                    # File path — only if GEMMA4_IMAGE_DIR is configured
                    if _validate_image_path(url):
                        with open(url, "rb") as f:
                            images.append(Image.open(f).convert("RGB"))

    return " ".join(text_parts), images


def _build_messages(messages: list[ChatMessage]):
    """Convert ChatMessage list to the format processor expects."""
    result = []
    for msg in messages:
        text, images = _parse_content(msg.content)
        if images:
            content_list = [{"type": "image"}]
            if text:
                content_list.append({"type": "text", "text": text})
            result.append({"role": msg.role, "content": content_list})
        else:
            result.append({"role": msg.role, "content": text})
    return result


def _prepare_inputs(model, processor, messages: list[ChatMessage]):
    """Prepare tokenized inputs (shared between sync and stream)."""
    has_images = False
    all_images = []
    for msg in messages:
        _, imgs = _parse_content(msg.content)
        if imgs:
            has_images = True
            all_images.extend(imgs)

    chat_messages = _build_messages(messages)

    if has_images:
        text = processor.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=all_images[:1], return_tensors="pt"
        )
    else:
        inputs = processor.apply_chat_template(
            chat_messages, tokenize=True, return_tensors="pt", return_dict=True
        )

    inputs_gpu = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    input_len = inputs_gpu["input_ids"].shape[-1]
    return inputs_gpu, input_len, has_images


def generate_sync(
    messages: list[ChatMessage],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    stop: list[str] | None = None,
) -> tuple[str, int, int]:
    """Run generation, return (response_text, prompt_tokens, completion_tokens)."""
    model, processor = load_model(MODEL_PATH)
    inputs_gpu, input_len, _ = _prepare_inputs(model, processor, messages)

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "do_sample": temperature > 0,
    }

    # Use StoppingCriteria for early termination on stop sequences
    if stop:
        criteria = StopSequenceCriteria(processor.tokenizer, stop)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])

    with torch.no_grad():
        outputs = model.generate(**inputs_gpu, **gen_kwargs)

    new_tokens = outputs.shape[-1] - input_len
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Final truncation on stop sequences (in case StoppingCriteria missed)
    if stop:
        for seq in stop:
            idx = response.find(seq)
            if idx != -1:
                response = response[:idx]
                break

    # Accurate completion token count
    comp_tokens = len(processor.tokenizer.encode(response, add_special_tokens=False))

    return response, input_len, comp_tokens


# ── FastAPI lifespan ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global INFERENCE_SEMAPHORE, SERVER_START_TIME
    SERVER_START_TIME = time.time()
    INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

    # Warn if MASTER_KEY is not set
    if not MASTER_KEY:
        log.warning("GEMMA4_MASTER_KEY not set: /v1/keys and /v1/config endpoints are publicly accessible")

    loop = asyncio.get_event_loop()
    log.info("Pre-loading model...")
    await loop.run_in_executor(None, load_model, MODEL_PATH)
    log.info("Server ready!")

    # Handle graceful shutdown on SIGTERM
    def _on_sigterm():
        log.info("SIGTERM received, starting graceful shutdown...")

    try:
        yield  # Server runs here
    finally:
        # Shutdown: cleanup GPU
        log.info("Shutting down...")
        await loop.run_in_executor(None, unload_model)
        log.info("Cleanup complete.")


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gemma 4 E4B Heretic — OpenAI-Compatible API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ───────────────────────────────────────────────


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add X-Request-ID to all responses."""
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


# ── Auth dependency ──────────────────────────────────────────────────────────


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    loaded = MODEL_OBJ is not None
    vram = {}
    if torch.cuda.is_available():
        vram["used_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 1)
        vram["free_gb"] = round(
            (torch.cuda.get_device_properties(0).total_memory
             - torch.cuda.memory_allocated()) / 1024**3, 1
        )
        vram["total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
    return {
        "status": "ok" if loaded else "loading",
        "model_loaded": loaded,
        "vram": vram,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, raw_request: Request):
    # Auth check
    key_prefix = _check_api_key(raw_request)
    if not key_prefix:
        return JSONResponse(
            status_code=401,
            content=_error_response("Invalid or missing API key", "authentication_error")
        )

    req_id = raw_request.state.request_id
    created = int(time.time())
    req_start = time.time()

    log.info(f"[{req_id}] request: {len(req.messages)} msgs, stream={req.stream}, "
             f"max_tokens={req.max_tokens}")

    # Rate limiting check
    keys = _load_keys()
    key_data = keys.get(key_prefix, {})
    rate_limit = key_data.get("rate_limit", DEFAULT_RATE_LIMIT)

    if not RATE_LIMITER.is_allowed(key_prefix, rate_limit):
        log.warning(f"[{req_id}] rate limit exceeded for {key_prefix}")
        return JSONResponse(
            status_code=429,
            content=_error_response(
                f"Rate limit exceeded: {rate_limit} requests per minute",
                "rate_limit_error"
            )
        )

    # Acquire semaphore — serialize GPU access (prevent OOM)
    if INFERENCE_SEMAPHORE is None:
        return JSONResponse(
            status_code=503,
            content=_error_response("Server not ready", "server_error")
        )

    if req.stream:
        # For streaming, manually acquire semaphore before returning generator.
        # Release happens in generator's finally block after streaming completes.
        await INFERENCE_SEMAPHORE.acquire()
        return EventSourceResponse(
            _stream_generator(req, req_id, created, key_prefix, req_start, INFERENCE_SEMAPHORE),
            media_type="text/event-stream",
        )

    # Non-streaming: use async with for automatic release
    async with INFERENCE_SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            response_text, prompt_tokens, completion_tokens = await loop.run_in_executor(
                None,
                generate_sync,
                req.messages,
                req.max_tokens,
                req.temperature,
                req.top_p,
                req.stop,
            )

            latency_ms = (time.time() - req_start) * 1000
            _log_usage(key_prefix, req.model, prompt_tokens, completion_tokens, latency_ms)

            log.info(
                f"[{req_id}] completed: {prompt_tokens}+{completion_tokens} tokens, "
                f"{latency_ms:.0f}ms"
            )

            return ChatCompletionResponse(
                id=req_id,
                created=created,
                model=req.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message={"role": "assistant", "content": response_text},
                        finish_reason="stop",
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
                content=_error_response("GPU out of memory — request too large", "server_error")
            )
        except ValueError as e:
            log.warning(f"[{req_id}] validation error: {e}")
            return JSONResponse(
                status_code=400,
                content=_error_response(str(e), "invalid_request_error")
            )
        except Exception as e:
            log.exception(f"[{req_id}] inference failed")
            return JSONResponse(
                status_code=500,
                content=_error_response(f"Generation failed: {str(e)}", "server_error")
            )


async def _stream_generator(
    req: ChatCompletionRequest, req_id: str, created: int,
    key_prefix: str, req_start: float, semaphore: asyncio.Semaphore
) -> AsyncIterator[dict]:
    """Real streaming via asyncio.Queue bridge from thread to async SSE."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    error_holder: list[Exception | None] = [None]

    def _run_inference():
        """Runs in a thread — produces tokens via TextIteratorStreamer."""
        try:
            model, processor = load_model(MODEL_PATH)
            inputs_gpu, input_len, _ = _prepare_inputs(model, processor, req.messages)

            streamer = TextIteratorStreamer(
                processor.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            gen_kwargs = {
                **inputs_gpu,
                "max_new_tokens": req.max_tokens,
                "temperature": req.temperature if req.temperature > 0 else None,
                "top_p": req.top_p,
                "do_sample": req.temperature > 0,
                "streamer": streamer,
            }

            # StoppingCriteria for stop sequences
            if req.stop:
                criteria = StopSequenceCriteria(processor.tokenizer, req.stop)
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])

            thread = threading.Thread(
                target=lambda: model.generate(**gen_kwargs), daemon=True
            )
            thread.start()

            prompt_tokens = input_len
            accumulated_text = ""

            # First chunk: send role
            asyncio.run_coroutine_threadsafe(
                queue.put(("role", prompt_tokens)), loop
            )

            for text_chunk in streamer:
                if text_chunk:
                    accumulated_text += text_chunk
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("content", text_chunk)), loop
                    )

            thread.join()

            # Count actual tokens from accumulated text
            completion_tokens = len(
                processor.tokenizer.encode(accumulated_text, add_special_tokens=False)
            )

            # Signal completion
            asyncio.run_coroutine_threadsafe(
                queue.put(("done", prompt_tokens, completion_tokens)), loop
            )

        except Exception as e:
            error_holder[0] = e
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

    # Start inference in background (don't await)
    loop.run_in_executor(None, _run_inference)

    # Consume queue and yield SSE events
    total_prompt = 0
    total_completion = 0

    try:
        while True:
            item = await queue.get()
            tag = item[0]

            if tag == "role":
                total_prompt = item[1]
                # First delta: include role
                data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield {"event": "message", "data": json.dumps(data)}

            elif tag == "content":
                text_chunk = item[1]
                data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text_chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield {"event": "message", "data": json.dumps(data)}

            elif tag == "done":
                total_prompt = item[1]
                total_completion = item[2]

                # Log usage
                latency_ms = (time.time() - req_start) * 1000
                _log_usage(key_prefix, req.model, total_prompt, total_completion, latency_ms)

                # Final chunk with finish_reason + usage
                final = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": total_prompt,
                        "completion_tokens": total_completion,
                        "total_tokens": total_prompt + total_completion,
                    },
                }
                yield {"event": "message", "data": json.dumps(final)}
                # OpenAI [DONE] sentinel
                yield {"event": "message", "data": "[DONE]"}
                log.info(
                    f"[{req_id}] stream completed: "
                    f"{total_prompt}+{total_completion} tokens, {latency_ms:.0f}ms"
                )
                break

            elif tag == "error":
                error_data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"\n[Error: {item[1]}]"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield {"event": "message", "data": json.dumps(error_data)}
                yield {"event": "message", "data": "[DONE]"}
                break
    finally:
        semaphore.release()


@app.post("/v1/keys")
async def manage_keys(raw_request: Request):
    """Create/list/revoke API keys (master-key protected)."""
    if not _check_master_key(raw_request):
        return JSONResponse(
            status_code=401,
            content=_error_response("Invalid or missing master key", "authentication_error")
        )

    req_id = raw_request.state.request_id
    method = raw_request.method

    try:
        body = await raw_request.json() if raw_request.method in ["POST", "PUT"] else {}
    except Exception:
        body = {}

    if method == "POST":
        # Create new key
        if "name" not in body:
            return JSONResponse(
                status_code=400,
                content=_error_response("Missing 'name' field", "invalid_request_error")
            )

        name = body["name"]
        rate_limit = body.get("rate_limit", DEFAULT_RATE_LIMIT)

        # Generate key
        key = f"g4k-{secrets.token_hex(16)}"
        key_hash = _hash_key(key)
        key_prefix = key[:12]

        # Store key
        keys = _load_keys()
        keys[key_prefix] = {
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "name": name,
            "created_at": time.time(),
            "rate_limit": rate_limit,
            "enabled": True,
        }
        _save_keys(keys)

        log.info(f"[{req_id}] Created API key: {key_prefix}")

        return {
            "key_prefix": key_prefix,
            "key": key,  # Only returned once
            "name": name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rate_limit": rate_limit,
            "enabled": True,
        }

    elif method == "GET":
        # List keys
        keys = _load_keys()
        result = []
        for prefix, data in keys.items():
            result.append({
                "key_prefix": prefix,
                "name": data.get("name", "Unknown"),
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(data.get("created_at", 0))
                ),
                "rate_limit": data.get("rate_limit", DEFAULT_RATE_LIMIT),
                "enabled": data.get("enabled", True),
            })
        return {"keys": result}

    elif method == "PUT":
        # Revoke/update key
        if "key_prefix" not in body:
            return JSONResponse(
                status_code=400,
                content=_error_response("Missing 'key_prefix' field", "invalid_request_error")
            )

        key_prefix = body["key_prefix"]
        action = body.get("action", "revoke")

        keys = _load_keys()
        if key_prefix not in keys:
            return JSONResponse(
                status_code=404,
                content=_error_response(f"Key '{key_prefix}' not found", "invalid_request_error")
            )

        if action == "revoke":
            keys[key_prefix]["enabled"] = False
            log.info(f"[{req_id}] Revoked API key: {key_prefix}")
        elif action == "enable":
            keys[key_prefix]["enabled"] = True
            log.info(f"[{req_id}] Enabled API key: {key_prefix}")
        elif "rate_limit" in body:
            keys[key_prefix]["rate_limit"] = body["rate_limit"]
            log.info(f"[{req_id}] Updated rate limit for {key_prefix}")

        _save_keys(keys)
        return {"status": "ok"}

    return JSONResponse(
        status_code=405,
        content=_error_response(f"Method {method} not allowed", "invalid_request_error")
    )


@app.get("/v1/config")
async def config(raw_request: Request):
    """Get server configuration (master-key protected)."""
    if MASTER_KEY and not _check_master_key(raw_request):
        return JSONResponse(
            status_code=401,
            content=_error_response("Invalid or missing master key", "authentication_error")
        )

    vram = {}
    if torch.cuda.is_available():
        vram["used_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        vram["total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
    else:
        vram["used_gb"] = 0
        vram["total_gb"] = 0

    uptime = time.time() - SERVER_START_TIME
    keys = _load_keys()
    active_keys = sum(1 for k in keys.values() if k.get("enabled"))

    return ConfigResponse(
        model=MODEL_ID,
        version="2.0.0",
        vram_used_gb=vram["used_gb"],
        vram_total_gb=vram["total_gb"],
        active_keys=active_keys,
        uptime_seconds=uptime,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

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
