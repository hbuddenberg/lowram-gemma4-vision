#!/usr/bin/env python3
"""
Gemma 4 E4B Heretic — OpenAI-compatible API server
FastAPI server wrapping bitsandbytes NF4 quantized Gemma 4 for text + vision.

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible (streaming + non-streaming)
  GET  /v1/models            — List available models
  GET  /health               — Health check

Usage:
  python server.py [--host 0.0.0.0] [--port 8080] [--model-path ~/models/gemma4-heretic]
"""

import argparse
import asyncio
import base64
import gc
import io
import json
import logging
import os
import time
import uuid
from typing import AsyncIterator, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import (
    BitsAndBytesConfig,
    Gemma4ForConditionalGeneration,
    Gemma4Processor,
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
LOAD_LOCK = asyncio.Lock()

# ── Configurable defaults ────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
API_KEY = os.environ.get("GEMMA4_API_KEY", "gemma4-local")

# ── Pydantic models (OpenAI-compatible) ──────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str | list = ""  # str for text, list for vision


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = None


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


# ── Model loading ────────────────────────────────────────────────────────────


def load_model(model_path: str):
    """Load Gemma 4 with NF4 text + fp16 vision."""
    global MODEL_OBJ, PROCESSOR

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
                    # data:image/png;base64,AAAA...
                    b64 = url.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64)
                    images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                elif url.startswith("http"):
                    import requests

                    resp = requests.get(url, timeout=30)
                    images.append(Image.open(io.BytesIO(resp.content)).convert("RGB"))
                else:
                    # File path
                    if os.path.exists(url):
                        images.append(Image.open(url).convert("RGB"))

    return " ".join(text_parts), images


def _build_messages(messages: list[ChatMessage]):
    """Convert ChatMessage list to the format processor expects."""
    result = []
    for msg in messages:
        text, images = _parse_content(msg.content)
        if images:
            # Vision message
            content_list = [{"type": "image"}]
            if text:
                content_list.append({"type": "text", "text": text})
            result.append({"role": msg.role, "content": content_list})
        else:
            result.append({"role": msg.role, "content": text})
    return result


def _count_tokens(text: str) -> int:
    """Rough token count estimation."""
    if PROCESSOR is None:
        return len(text) // 4
    try:
        tokens = PROCESSOR.tokenizer.encode(text)
        return len(tokens)
    except Exception:
        return len(text) // 4


def generate_sync(
    messages: list[ChatMessage],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    stop: list[str] | None = None,
) -> tuple[str, int, int]:
    """Run generation, return (response_text, prompt_tokens, completion_tokens)."""
    model, processor = load_model(MODEL_PATH)

    # Check if any message has images
    has_images = False
    all_images = []
    for msg in messages:
        _, imgs = _parse_content(msg.content)
        if imgs:
            has_images = True
            all_images.extend(imgs)

    chat_messages = _build_messages(messages)

    if has_images:
        # Vision mode
        text = processor.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=all_images[:1], return_tensors="pt"
        )
    else:
        # Text-only mode
        inputs = processor.apply_chat_template(
            chat_messages, tokenize=True, return_tensors="pt", return_dict=True
        )

    inputs_gpu = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    input_len = inputs_gpu["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
    }
    if temperature == 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True

    with torch.no_grad():
        outputs = model.generate(**inputs_gpu, **gen_kwargs)

    new_tokens = outputs.shape[-1] - input_len
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Handle stop sequences
    if stop:
        for seq in stop:
            idx = response.find(seq)
            if idx != -1:
                response = response[:idx]
                new_tokens = len(
                    processor.tokenizer.encode(response, add_special_tokens=False)
                )

    return response, input_len, new_tokens


def generate_stream(
    messages: list[ChatMessage],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
):
    """Generator that yields text chunks for streaming."""
    model, processor = load_model(MODEL_PATH)

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

    # Use TextIteratorStreamer for token-by-token streaming
    from transformers import TextIteratorStreamer
    import threading

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs_gpu,
        "max_new_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "streamer": streamer,
    }
    if temperature == 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True

    def _generate():
        with torch.no_grad():
            model.generate(**gen_kwargs)

    thread = threading.Thread(target=_generate)
    thread.start()

    completion_tokens = 0
    for text_chunk in streamer:
        if text_chunk:
            completion_tokens += 1
            yield text_chunk, input_len, completion_tokens

    thread.join()


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gemma 4 E4B Heretic — OpenAI-Compatible API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Pre-load model on startup."""
    loop = asyncio.get_event_loop()
    log.info("Pre-loading model...")
    await loop.run_in_executor(None, load_model, MODEL_PATH)
    log.info("Server ready!")


@app.get("/health")
async def health():
    loaded = MODEL_OBJ is not None
    vram = {}
    if torch.cuda.is_available():
        vram["used_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 1)
        vram["total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
    return {"status": "ok" if loaded else "loading", "model_loaded": loaded, "vram": vram}


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
async def chat_completions(req: ChatCompletionRequest):
    # Optional API key check
    # if req.api_key != API_KEY: ...

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        return EventSourceResponse(
            _stream_generator(req, req_id, created),
            media_type="text/event-stream",
        )

    # Non-streaming: run in thread pool to not block event loop
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


async def _stream_generator(
    req: ChatCompletionRequest, req_id: str, created: int
) -> AsyncIterator[dict]:
    """SSE event generator for streaming responses."""
    loop = asyncio.get_event_loop()

    def _sync_gen():
        return list(generate_stream(
            req.messages, req.max_tokens, req.temperature, req.top_p
        ))

    # Run the sync generator in a thread
    chunks = await loop.run_in_executor(None, _sync_gen)

    prompt_tokens = chunks[0][1] if chunks else 0
    total_completion = 0

    for text_chunk, pt, ct in chunks:
        total_completion = ct
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

    # Final chunk with finish_reason
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion,
            "total_tokens": prompt_tokens + total_completion,
        },
    }
    yield {"event": "message", "data": json.dumps(final)}


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
