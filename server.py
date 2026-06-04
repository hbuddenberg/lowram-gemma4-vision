#!/usr/bin/env python3
"""
Gemma 4 E4B Heretic — OpenAI-compatible API server
FastAPI server wrapping bitsandbytes NF4 quantized Gemma 4 for text + vision.

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible (streaming + non-streaming)
  GET  /v1/models            — List available models
  GET  /health               — Health check + VRAM

Usage:
  python server.py [--host 0.0.0.0] [--port 8080] [--model-path ~/models/gemma4-heretic]
"""

import argparse
import asyncio
import base64
import gc
import ipaddress
import io
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

import requests as _requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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

# ── Configurable defaults ────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
API_KEY = os.environ.get("GEMMA4_API_KEY", "gemma4-local")
MAX_IMAGE_SIZE_MB = int(os.environ.get("GEMMA4_MAX_IMAGE_MB", "20"))
VALID_ROLES = {"system", "user", "assistant", "tool"}

# ── Security: allowed image hosts (empty = allow all public) ─────────────────
ALLOWED_IMAGE_DIR = os.environ.get("GEMMA4_IMAGE_DIR", "")


# ── Stopping criteria for stop sequences ─────────────────────────────────────


class StopSequenceCriteria(StoppingCriteria):
    """Stop generation when any stop sequence appears in decoded output."""

    def __init__(self, tokenizer, stop_sequences: list[str]):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self._decoded = ""

    def __call__(self, input_ids, scores, **kwargs):
        new_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(seq in new_text for seq in self.stop_sequences)


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
    return real.startswith(allowed) and os.path.isfile(real)


def _check_api_key(request: Request):
    """Validate API key from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        if token == API_KEY:
            return True
    # Also check X-API-Key header
    api_key = request.headers.get("X-API-Key", "")
    if api_key == API_KEY:
        return True
    return False


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
                    resp = _requests.get(url, timeout=15, stream=True)
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
    global INFERENCE_SEMAPHORE
    INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

    loop = asyncio.get_event_loop()
    log.info("Pre-loading model...")
    await loop.run_in_executor(None, load_model, MODEL_PATH)
    log.info("Server ready!")

    yield  # Server runs here

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
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    log.info(
        f"{request.method} {request.url.path} "
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
        "semaphore_available": INFERENCE_SEMAPHORE._value if INFERENCE_SEMAPHORE else 0,
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
    if not _check_api_key(raw_request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    log.info(f"[{req_id}] request: {len(req.messages)} msgs, stream={req.stream}, "
             f"max_tokens={req.max_tokens}")

    # Acquire semaphore — serialize GPU access (prevent OOM)
    if INFERENCE_SEMAPHORE is None:
        raise HTTPException(503, "Server not ready")

    if not INFERENCE_SEMAPHORE.locked() and INFERENCE_SEMAPHORE._value <= 0:
        raise HTTPException(503, "Server busy — another request is being processed")

    async with INFERENCE_SEMAPHORE:
        try:
            if req.stream:
                return EventSourceResponse(
                    _stream_generator(req, req_id, created),
                    media_type="text/event-stream",
                )

            # Non-streaming
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

            log.info(
                f"[{req_id}] completed: {prompt_tokens}+{completion_tokens} tokens"
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
            raise HTTPException(503, "GPU out of memory — request too large")
        except ValueError as e:
            log.warning(f"[{req_id}] validation error: {e}")
            raise HTTPException(400, str(e))
        except Exception as e:
            log.exception(f"[{req_id}] inference failed")
            raise HTTPException(500, f"Generation failed: {e}")


async def _stream_generator(
    req: ChatCompletionRequest, req_id: str, created: int
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
            completion_tokens = 0

            # First chunk: send role
            asyncio.run_coroutine_threadsafe(
                queue.put(("role", prompt_tokens)), loop
            )

            for text_chunk in streamer:
                if text_chunk:
                    completion_tokens += 1
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("content", text_chunk, completion_tokens)), loop
                    )

                    # Check stop sequences in streaming
                    if req.stop:
                        # We need accumulated text for stop check
                        # StoppingCriteria handles it at generate level
                        pass

            thread.join()

            # Signal completion
            asyncio.run_coroutine_threadsafe(
                queue.put(("done", prompt_tokens, completion_tokens)), loop
            )

        except Exception as e:
            error_holder[0] = e
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

    # Start inference in thread pool
    await loop.run_in_executor(None, _run_inference)

    # Consume queue and yield SSE events
    total_prompt = 0
    total_completion = 0

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
            total_completion = item[2]
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
                f"{total_prompt}+{total_completion} tokens"
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
