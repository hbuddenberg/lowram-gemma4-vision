# TRD — lowram-gemma4-vision Phase 2: Rust Migration

**Version:** 1.0.0  
**Date:** 2026-06-05  
**Author:** Hans-Dieter Buddenberg Blamey  
**Status:** Draft

---

## 1. Arquitectura

### 1.1 Visión General

Migración del motor de inferencia de Python/PyTorch a Rust (mistral.rs), manteniendo una capa API Python (FastAPI) para backwards compatibility.

```
                     ┌─────────────────────┐
   Discord App  ←──→ │  API (FastAPI)       │  ←── Hermes Gateway
   Mobile App   ←──→ │  :11434             │
   curl         ←──→ │  /v1/chat/completions│
                     └────────┬────────────┘
                              │
                     ┌────────▼────────────┐
                     │  mistralrs (Python) │  ← Runner API
                     └────────┬────────────┘
                              │ FFI
                     ┌────────▼────────────┐
                     │  mistral.rs (Rust)   │  ← Inference engine
                     │  ├── Gemma 4 LLM    │
                     │  ├── SigLIP Vision  │
                     │  └── Audio Encoder  │
                     └────────┬────────────┘
                              │ CUDA
                     ┌────────▼────────────┐
                     │  RTX 3060 12GB      │
                     │  (Thunderbolt 3)    │
                     └─────────────────────┘

   CPU (en paralelo):
   ┌─────────────────────────────┐
   │  FAISS-CPU (vector search)  │
   │  sentence-transformers      │
   │  SQLite (keys, history)     │
   └─────────────────────────────┘
```

### 1.2 Componentes

| Componente | Tecnología | Rol | VRAM/CPU |
|---|---|---|---|
| API Layer | FastAPI (Python) | HTTP, auth, rate limit, SSE streaming | CPU |
| Runner | `mistralrs` (PyPI) | Bridge Python ↔ Rust | CPU |
| Inference | mistral.rs (Rust) | LLM + vision + audio | GPU |
| RAG | FAISS-CPU + sentence-transformers | Historical context retrieval | CPU |
| Storage | SQLite | API keys, request log, RAG metadata | CPU/RAM |

### 1.3 Comparación Arquitecturas

```
ANTES (Phase 1):
  FastAPI → transformers → bitsandbytes → PyTorch → CUDA
  VRAM: ~11.6 GB (sin margen para audio)

DESPUÉS (Phase 2):
  FastAPI → mistralrs → mistral.rs → CUDA
  VRAM: ~4-6 GB (margen para vision + audio + KV cache)
```

## 2. Decisiones Técnicas

### 2.1 mistral.rs vs Alternativas

| Criterio | mistral.rs | llama.cpp | candle |
|---|---|---|---|
| Gemma 4 support | ✅ Nativo | ✅ (vía GGUF) | ❌ |
| Vision | ✅ Image+Video | ✅ | ❌ |
| Audio | ✅ WAV/MP3/FLAC/OGG | ✅ | ❌ |
| Formato modelo | Safetensors + GGUF | GGUF solo | Safetensors |
| Cuantización | In-situ 4-bit | Requiere GGUF pre-quant | Manual |
| Bindings Python | ✅ `mistralrs` | ✅ `llama-cpp-python` | ❌ (Rust solo) |
| Overhead memoria | Bajo (~0.3 GB) | Bajo (~0.3 GB) | Medio |
| Streaming SSE | ✅ Nativo | ✅ | Manual |
| Compilación | Cargo o PyPI | CMake o PyPI | Cargo |

**Decisión:** mistral.rs — carga safetensors directo (sin conversión GGUF), multimodal completo, Python bindings.

### 2.2 Cuantización: In-Situ vs Pre-conversión

**In-situ (mistral.rs):** Cuantiza en memoria al cargar.
- ✅ Sin archivo GGUF intermedio
- ✅ Mismo safetensors que ya tenemos
- ⚠️ Tiempo de carga ligeramente mayor (one-time)

**Pre-conversión (GGUF):** Convertir con llama.cpp, luego cargar GGUF.
- ✅ Carga más rápida (ya cuantizado)
- ❌ Requiere proceso de conversión
- ❌ Archivo extra (~7.5 GB)
- ❌ Puede perder metadatos de abliteration

**Decisión:** In-situ quant (`in_situ_quant="4"`). Mantenemos un solo modelo (safetensors).

### 2.3 API Layer: FastAPI vs Mistral.rs HTTP

mistral.rs tiene su propio servidor HTTP (`mistral-rs-server`), pero:
- No tiene auth por API key
- No tiene rate limiting
- No tiene request logging
- No tiene RAG integration

**Decisión:** Mantener FastAPI como API layer, usar `mistralrs` Python bindings como runner.

### 2.4 RAG: Mantener vs Reescribir

El módulo RAG (FAISS + sentence-transformers) funciona perfectamente en CPU.
No toca GPU, no necesita cambios.

**Decisión:** Migrar tal cual desde Phase 1. Solo adaptar la interfaz.

## 3. Diseño Detallado

### 3.1 Modelo de Datos (Request Flow)

```
Request JSON (OpenAI format)
    │
    ▼
FastAPI endpoint (/v1/chat/completions)
    │
    ├── Auth (API key validation)      ← SQLite
    ├── Rate limit                      ← In-memory
    ├── RAG check                       ← Si msgs > budget → FAISS
    │
    ▼
mistralrs.Runner
    │
    ├── ChatCompletionRequest
    │   ├── messages (text + image + audio)
    │   ├── max_tokens
    │   ├── temperature
    │   └── stream (bool)
    │
    ▼
mistral.rs (Rust, CUDA)
    │
    ├── Tokenizer (HuggingFace)
    ├── Model forward pass
    │   ├── Text embeddings (4-bit quant)
    │   ├── Image patches (SigLIP, fp16)
    │   └── Audio encoding (native)
    ├── KV cache management
    └── Sampling
    │
    ▼
Response (text / SSE stream)
    │
    ▼
FastAPI → JSON response / SSE events
```

### 3.2 Configuración

```yaml
# config.yaml
model:
  id: "igorls/gemma-4-E4B-it-heretic"
  arch: "Gemma4"
  quant: "4"                    # In-situ 4-bit
  max_seq_len: 8192
  kv_cache_mem_fraction: 0.25   # Reserve VRAM for vision/audio

server:
  host: "0.0.0.0"
  port: 11434

rag:
  enabled: true
  model: "all-MiniLM-L6-v2"
  top_k: 5
  max_retrieval_chars: 2000

budget:
  text_only: 4
  with_vision: 2
  with_audio: 3

image:
  max_dimension: 896
  max_size_mb: 20
```

### 3.3 VRAM Management

```
mistral.rs KV cache:
  kv_cache_mem_fraction = 0.25
  → Reserva 25% de VRAM libre para KV cache
  → El resto disponible para vision/audio encoding dinámico

Image preprocessing:
  → Resize a 896px antes de pasar al engine
  → Reduce activaciones del vision encoder

Audio:
  → FFmpeg decodea a WAV raw
  → Encoder procesa en chunks
```

### 3.4 Error Handling

| Error | HTTP | Origen |
|---|---|---|
| API key inválida | 401 | FastAPI |
| Rate limit | 429 | FastAPI |
| Modelo no encontrado | 404 | mistral.rs |
| VRAM insuficiente | 503 | mistral.rs |
| Imagen inválida | 400 | FastAPI |
| Audio inválido | 400 | FastAPI |

### 3.5 Streaming

mistral.rs soporta streaming nativo via Python bindings:

```python
# Non-streaming
response = runner.send_chat_completion_request(request)
print(response.choices[0].message.content)

# Streaming
for chunk in runner.send_chat_completion_request_stream(request):
    print(chunk.choices[0].delta.content, end="")
```

FastAPI SSE bridge:
```python
async def stream_response(request):
    for chunk in runner.send_chat_completion_request_stream(request):
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
```

## 4. Migración desde Phase 1

### 4.1 Componentes a Migrar

| Componente Phase 1 | Acción Phase 2 |
|---|---|
| `server.py` (FastAPI) | **Mantener** — reemplazar solo `generate_sync()` y `_prepare_inputs()` |
| `rag.py` (FAISS) | **Mantener tal cual** |
| `inference.py` (standalone) | **Eliminar** — funcionalidad absorbida por mistral.rs |
| `BitsAndBytesConfig` | **Eliminar** — reemplazado por in-situ quant |
| `load_model()` | **Reemplazar** — por `mistralrs.Runner()` |
| `_prepare_inputs()` | **Eliminar** — mistral.rs maneja tokenización |
| Image preprocessing | **Mantener** — resize antes de pasar a engine |
| Auth (API keys) | **Mantener** |
| Rate limiting | **Mantener** |
| Request logging | **Mantener** |

### 4.2 Estructura de Archivos

```
lowram-gemma4-vision/
├── server.py           # FastAPI API (modificado: mistral.rs runner)
├── rag.py              # RAG con FAISS (sin cambios)
├── config.py           # Nueva: carga config.yaml + env vars
├── image.py            # Nueva: preprocessing de imágenes
├── audio.py            # Nueva: preprocessing de audio (si aplica)
├── requirements.txt     # mistralrs + fastapi + faiss-cpu + ...
├── config.yaml         # Nueva: configuración centralizada
├── gemma4-api.service  # systemd (actualizar ExecStart)
├── docs/
│   └── PLANNING/
│       ├── PRD.md      # Este documento
│       ├── TRD.md      # Este documento
│       └── IMPLEMENTATION.md
└── tests/
    ├── test_text.py    # Inferencia texto
    ├── test_vision.py  # Inferencia con imagen
    ├── test_audio.py   # Inferencia con audio
    ├── test_rag.py     # RAG retrieval
    └── test_api.py     # API endpoints
```

## 5. Performance Targets

| Métrica | Phase 1 (actual) | Phase 2 (target) |
|---|---|---|
| VRAM idle | 9.3 GB | ~3.0 GB |
| VRAM inferencia text | ~10.0 GB | ~4.0 GB |
| VRAM inferencia vision | OOM | ~6.0 GB |
| VRAM inferencia audio | N/A | ~5.5 GB |
| Latencia text (128 tok) | ~2.5s | ≤ 3.0s |
| Latencia vision (img+64 tok) | ~4.0s | ≤ 5.0s |
| Tokens/segundo | ~50 tok/s | ~40-60 tok/s |
| Cold start | ~45s | ~30s |
| VRAM overhead (Python) | ~1.3 GB | ~0.3 GB (Rust) |

## 6. Seguridad

Sin cambios respecto a Phase 1:
- API key validation (SQLite backend)
- Rate limiting per key
- CORS configurable
- Image size limits
- Bloqueo de URLs privadas
- Sin tool calling expuesto a red externa
