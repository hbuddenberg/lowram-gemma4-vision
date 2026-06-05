# PRD — lowram-gemma4-vision Phase 2: Rust Migration

**Version:** 1.0.0  
**Date:** 2026-06-05  
**Author:** Hans-Dieter Buddenberg Blamey  
**Status:** Draft

---

## 1. Resumen Ejecutivo

Migrar el motor de inferencia de Python (transformers + bitsandbytes) a **mistral.rs** con Python bindings, manteniendo la API OpenAI-compatible existente. El objetivo es reducir el footprint de VRAM para habilitar **inferencia multimodal real (texto + visión + audio)** en una GPU RTX 3060 de 12 GB sin OOM.

## 2. Problema

### 2.1 Situación Actual

| Componente | Stack | VRAM |
|---|---|---|
| Modelo LLM | transformers + bitsandbytes NF4 | ~9.3 GB |
| Vision encoder | SigLIP (fp16) | ~0.5 GB |
| KV cache (4 msgs) | PyTorch | ~0.5 GB |
| Overhead Python/PyTorch | runtime | ~1.3 GB |
| **Total** | | **~11.6 GB (al límite)** |

**Consecuencias:**
- CUDA OOM cuando se envían imágenes con contexto largo
- No hay espacio para audio (Gemma 4 soporta audio nativo)
- `vLLM` bloqueado (requiere CUDA toolkit nvcc no disponible)
- Cada request con imagen requiere `torch.cuda.empty_cache()` + `gc.collect()`

### 2.2 Requisitos de Hardware

- **GPU:** NVIDIA RTX 3060 12 GB (2x via Thunderbolt 3 eGPU)
- **CPU:** Intel i5-7260U (NUC7i5BNB)
- **RAM:** 16 GB DDR4
- **OS:** Arch Linux (kernel 7.0.10-zen1)
- **Modelo:** `igorls/gemma-4-E4B-it-heretic` (safetensors, descensurizado)

## 3. Objetivos

### 3.1 Meta Principal

Correr Gemma 4 E4B multimodal (texto + visión + audio) en 12 GB VRAM con margen seguro.

### 3.2 Objetivos Específicos

| # | Objetivo | Métrica |
|---|---|---|
| O1 | Reducir VRAM del modelo LLM a ≤ 4 GB | `nvidia-smi` |
| O2 | Habilitar procesamiento de imágenes sin OOM | Request exitosa con imagen + 4 msgs |
| O3 | Habilitar procesamiento de audio | Request exitosa con audio WAV/MP3 |
| O4 | Mantener API OpenAI-compatible (mismo endpoint) | `curl` test pasa |
| O5 | Tiempo de respuesta ≤ 15s (text-only, 128 tokens) | Benchmark |
| O6 | Preservar RAG con FAISS (historial semántico) | Test 10 msgs recupera contexto |
| O7 | Sin servidor externo — todo local en NUC | Solo localhost/LAN |

## 4. Alcance

### 4.1 Incluido

- Motor de inferencia: mistral.rs con bindings Python (`mistralrs`)
- Cuantización in-situ a 4-bit (desde safetensors, sin conversión GGUF)
- Vision: imágenes (base64, URL, archivo local)
- Audio: WAV, MP3, FLAC, OGG
- API: mismo endpoint `/v1/chat/completions` (OpenAI-compatible)
- RAG: migrar FAISS existente
- Configuración: env vars + YAML (reemplaza bitsandbytes config)
- systemd service

### 4.2 Excluido

- vLLM (bloqueado por falta de nvcc)
- llama.cpp / GGUF (fallback, no ruta principal)
- candle (no soporta Gemma 4 multimodal)
- UI/CLI de chat
- Modelos GGUF pre-cuantizados
- Heretic abliteration on-device (ya tenemos modelo descensurizado)

## 5. Stack Propuesto

```
┌──────────────────────────────────────────┐
│  FastAPI (API layer, sin cambios majors)  │
├──────────────────────────────────────────┤
│  mistralrs (Python bindings)             │
├──────────────────────────────────────────┤
│  mistral.rs (Rust inference engine)       │
│  ├── Gemma 4 LLM (in-situ 4-bit quant)  │
│  ├── Vision encoder (SigLIP)             │
│  └── Audio encoder (native)              │
├──────────────────────────────────────────┤
│  CUDA 12 (nvidia-open-dkms)              │
└──────────────────────────────────────────┘
         ↕
┌──────────────────────────────────────────┐
│  FAISS-CPU (RAG, embeddings, sin VRAM)   │
│  sentence-transformers (all-MiniLM-L6-v2) │
│  SQLite (metadatos, API keys, historial)  │
└──────────────────────────────────────────┘
```

## 6. Presupuesto VRAM Objetivo

| Componente | VRAM Estimado |
|---|---|
| Gemma 4 E4B @ 4-bit (mistral.rs in-situ) | ~2.5–3.0 GB |
| Vision encoder (SigLIP) | ~0.3–0.5 GB |
| Audio encoder | ~0.3–0.5 GB |
| KV cache (8K tokens) | ~0.5–1.0 GB |
| Overhead (Rust, CUDA runtime) | ~0.3 GB |
| **Total** | **~4.0–5.5 GB** |
| **Headroom** | **~6.5–8.0 GB** |

vs. actual: 11.6 GB usados → **50-65% de reducción.**

## 7. Hitos

| Hito | Fecha Estimada | Entregable |
|---|---|---|
| M1: PoC mistral.rs | Semana 1 | Modelo carga y genera texto |
| M2: API OpenAI | Semana 1 | `/v1/chat/completions` funcional |
| M3: Vision | Semana 2 | Procesa imágenes sin OOM |
| M4: Audio | Semana 2 | Procesa audio WAV/MP3 |
| M5: RAG | Semana 2 | FAISS integrado |
| M6: Benchmark | Semana 3 | Métricas y optimización |
| M7: Release | Semana 3 | Tag en GitHub |

## 8. Riesgos

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| mistral.rs in-situ quant falla con modelo heretic | Media | Alto | Fallback: convertir a GGUF → llama.cpp |
| Compilación mistral.rs con CUDA falla en Arch | Baja | Alto | Usar PyPI `mistralrs` (pre-compiled) |
| Audio no soportado en Gemma 4 E4B | Media | Medio | Verificar con mistral.rs docs, desactivar si no |
| VRAM mayor al estimado | Baja | Medio | Reducir max-seq-len, ajustar kv-cache-mem-fraction |
| Breaking changes en mistral.rs API | Baja | Medio | Pinnear versión exacta en requirements.txt |

## 9. Criterios de Aceptación

- [ ] `curl localhost:11434/v1/chat/completions` con texto responde 200 OK
- [ ] Request con imagen (base64) responde sin OOM
- [ ] Request con audio (WAV) responde (si soportado)
- [ ] VRAM total ≤ 6 GB durante inferencia text-only
- [ ] VRAM total ≤ 8 GB durante inferencia con imagen
- [ ] RAG recupera contexto de mensajes anteriores
- [ ] Streaming SSE funciona
- [ ] systemd service arranca y para correctamente
- [ ] Test con 115 mensajes (historial largo) no OOM

## 10. Open Questions

1. ¿mistral.rs soporta `igorls/gemma-4-E4B-it-heretic` directamente o necesita el modelo base `google/gemma-4-12b-it`?
2. ¿La cuantización in-situ de mistral.rs es compatible con modelos que ya pasaron por abliteration (Heretic)?
3. ¿El audio de Gemma 4 funciona en el modelo E4B o solo en 12B/27B?
4. ¿mistral.rs tiene soporte para tool calling nativo de Gemma 4?
