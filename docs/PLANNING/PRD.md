# PRD — lowram-gemma4-vision Phase 2: Migración a Rust

**Version:** 2.1.0  
**Fecha:** 2026-06-05  
**Autor:** Hans-Dieter Buddenberg Blamey  
**Status:** Approved  
**Branch:** phase2  
**Decisión:** mistral.rs + nginx (consenso unánime AGY + OC + CC)

---

## 1. Resumen Ejecutivo

Migrar completamente el motor de inferencia de Python (transformers + PyTorch, ~2221 líneas) a **mistral.rs** (Rust puro) con **nginx** como reverse proxy. Objetivo: eliminar Python del path de inferencia, reduciendo RAM de 1.6 GB a ~100 MB y habilitando inferencia multimodal real (texto + visión + audio) en una RTX 3060 de 12 GB con margen seguro.

**Cambio fundamental:** El cuello de botella es **RAM** (380 MB libres de 7.6 GB, worst-case medido con sistema bajo carga), no VRAM. Python+PyTorch consume 1.6 GB RAM. mistral.rs consume ~50-100 MB. Eliminar Python libera ~1.5 GB RAM.

---

## 2. Problema

### 2.1 Situación Actual (Phase 1)

| Componente | Stack | VRAM | RAM |
|---|---|---|---|
| Modelo LLM E4B | transformers + PyTorch | ~9.3 GB | ~800 MB |
| Vision encoder (Gemma4 Vision) | transformers (fp16) | ~0.5 GB | ~200 MB |
| KV cache (4 msgs) | PyTorch | ~0.5 GB | ~100 MB |
| Overhead Python/PyTorch | runtime | ~1.3 GB | ~500 MB |
| **Total** | | **~11.6 GB** | **~1.6 GB** |

**Consecuencias:**
- CUDA OOM al enviar imágenes con contexto largo
- Sin espacio para audio (Gemma 4 soporta audio nativo)
- Swap zram saturado (3.7 GB de 3.8 GB usados)
- Cada request con imagen requiere `torch.cuda.empty_cache()` + `gc.collect()`
- Sistema al borde del OOM general con solo 380 MB RAM libre (worst-case)

### 2.2 Hardware Real (verificado por 3 agentes independientes)

| Recurso | Valor | Estado |
|---|---|---|
| **CPU** | Intel i5-7260U @ 2.20GHz (2C/4T) | Active |
| **RAM total** | 7,807 MB | ⚠️ Constraint principal |
| **RAM libre** | **380 MB** (worst-case) | 🔴 Crítico |
| **GPU** | NVIDIA RTX 3060 12 GB, compute_cap 8.6 | Active |
| **VRAM en uso** | 9,793 MB (server.py) | Alto |
| **VRAM libre** | ~2,100 MB | Ajustado |
| **CUDA** | 13.3.33, nvcc en /opt/cuda/bin/ | ✅ Disponible |
| **Rust** | 1.96.0 + Cargo 1.96.0 | ✅ Disponible |
| **cudnn** | NO instalado | ⚠️ Requiere `pacman -S cudnn` |
| **nginx** | NO instalado | ⚠️ Requiere `pacman -S nginx` |
| **FFmpeg** | Instalado | ✅ Ya disponible |
| **Swap** | 3.8 GB zram (3.7 GB en uso) | 🔴 Saturado (zram comprimido en RAM) |
| **Kernel** | Arch Linux 7.0.10-zen1 | Current |
| **Modelo** | ~/models/gemma4-heretic/ (16 GB safetensors, 1 shard) | ✅ Disponible |

### 2.3 Modelo: Gemma 4 E4B Heretic

- **Arquitectura:** Gemma4ForConditionalGeneration
- **Capas:** 42 texto (dim 2560) + 16 visión (dim 768) + 12 audio (dim 1024)
- **Origen:** `igorls/gemma-4-E4B-it-heretic` (descensurizado via abliteration)
- **tie_word_embeddings:** `true` (embed_tokens y lm_head comparten pesos)
- **Formato:** safetensors directo (NO GGUF)
- **Tamaño en disco:** 16 GB (1 shard)
- **⚠️ Componente crítico:** `embed_tokens_per_layer` — tensor [262144, 10752] = **5.64 GB fp16**, 35% del modelo

---

## 3. Objetivos

### 3.1 Meta Principal

Correr Gemma 4 E4B multimodal (texto + visión + audio) en RTX 3060 12 GB con margen seguro, eliminando Python del inference path.

### 3.2 Objetivos Específicos y Métricas

| # | Objetivo | Métrica | Target | Verificación |
|---|---|---|---|---|
| O1 | Reducir VRAM del modelo | `nvidia-smi` | ≤ 8.5 GB total en carga | Comparar idle vs Phase 1 |
| O2 | Reducir consumo RAM | `free -h` | ≤ 150 MB para inference engine | Medir proceso mistralrs |
| O3 | Habilitar visión sin OOM | curl + imagen base64 | Request exitosa con imagen + 4 msgs contexto | `nvidia-smi` durante request |
| O4 | Habilitar audio | curl + WAV | Request exitosa con audio + pregunta | Respuesta coherente |
| O5 | API OpenAI-compatible | curl test | `/v1/chat/completions` responde 200 | Formato OpenAI válido |
| O6 | Latencia aceptable | timed curl | ≤ 15s text-only, 128 tokens output | Benchmark |
| O7 | Streaming SSE funcional | curl -N | Chunks incrementales en tiempo real | EventSource |
| O8 | Todo local (sin cloud) | Verificar puertos | Solo localhost/LAN accesible | `ss -tlnp` |
| O9 | RAG post-migración | Test retrieval | FAISS recupera contexto relevante | Integration test |

---

## 4. Alcance

### 4.1 Incluido (IN SCOPE)

- **Motor de inferencia:** mistral.rs compilado con CUDA + flash-attn + cudnn
- **Cuantización:** ISQ Q4K in-situ desde safetensors (sin conversión GGUF)
- **Visión:** Imágenes via base64, URL, o archivo local (resize a ≤896px)
- **Audio:** WAV, MP3, FLAC, OGG via FFmpeg (ya instalado)
- **API:** mistral.rs HTTP server nativo (OpenAI-compatible, puerto 8080)
- **Reverse proxy:** nginx con autenticación API key, rate limiting, SSL opcional
- **Service:** systemd user service para mistral.rs (`gemma4-rs.service`)
- **RAG:** Integración post-migración con FAISS-CPU (servicio separado)
- **Configuración:** TOML para mistral.rs, nginx.conf para proxy
- **Monitoreo:** nvidia-smi logging, journalctl

### 4.2 Excluido (OUT OF SCOPE)

- ~~vLLM~~ (descartado: requiere 3-5 GB RAM, 380 MB libre = OOM garantizado)
- ~~llama.cpp / GGUF~~ (descartado: sin visión/audio para Gemma 4)
- ~~bitsandbytes~~ (descartado: Python-heavy, mismo problema de RAM)
- ~~FastAPI / Python API layer~~ (eliminado: zero Python en inference path)
- ~~UI/CLI de chat~~ (usar API directamente)
- ~~Heretic abliteration on-device~~ (ya tenemos modelo descensurizado)
- ~~Tool calling~~ (evaluar en Phase 3)
- ~~Fine-tuning~~ (no aplicable con ISQ)

---

## 5. Stack Tecnológico Final

```
┌─────────────────────────────────────────────────────────┐
│  nginx (reverse proxy)                                   │
│  ├── Autenticación API key (auth_request / lua)          │
│  ├── Rate limiting (limit_req)                           │
│  ├── SSL/TLS opcional                                    │
│  └── Puerto :443 (externo) → :8080 (interno)            │
├─────────────────────────────────────────────────────────┤
│  mistralrs serve (binario Rust, HTTP server nativo)      │
│  ├── /v1/chat/completions (OpenAI-compatible)            │
│  ├── /v1/models                                          │
│  ├── /health                                             │
│  ├── ISQ Q4K in-situ desde safetensors                   │
│  ├── Gemma 4 E4B (texto + visión + audio)                │
│  └── Puerto :8080 (localhost only)                        │
├─────────────────────────────────────────────────────────┤
│  CUDA 13.3 + cuDNN + Flash Attention                     │
│  └── RTX 3060 12 GB (compute_cap 8.6)                   │
├─────────────────────────────────────────────────────────┤
│  [POST-MIGRACIÓN] RAG Service (separado)                 │
│  ├── FAISS-CPU (búsqueda vectorial)                      │
│  ├── Embeddings (ONNX o pre-computados, ~130 MB)         │
│  └── Puerto :8081 o middleware nginx                     │
└─────────────────────────────────────────────────────────┘
```

**Cero Python en el inference path.** Un solo binario Rust (`mistralrs`) hace todo.

---

## 6. Presupuesto VRAM Objetivo (ISQ Q4K)

> **⚠️ DATOS REALES del safetensors** — verificado por análisis de tensores individuales.
> El componente `embed_tokens_per_layer` (5.64 GB fp16) es el **35% del modelo** y estaba ausente en la planificación original. Su cuantización determina la viabilidad del proyecto.

### 6.1 Escenario A: ISQ cuantiza embed_tokens_per_layer (objetivo)

| Componente | VRAM (Q4K) | Notas |
|---|---|---|
| Text layers Q4K (42 capas) | ~2.0 GB | Pesos 4-bit de capas de texto |
| **embed_tokens_per_layer** Q4K | **~1.41 GB** | Tensor [262144, 10752] cuantizado a 4-bit |
| Embedding/LM Head (shared, fp16) | ~1.34 GB | tie_word_embeddings=true: 1 tensor compartido |
| Per-layer projections Q4K | ~0.03 GB | Proyecciones por capa, muy pequeñas |
| Vision encoder (fp16) | ~0.34 GB | 16 capas, dim 768, on-demand |
| Audio encoder (fp16) | ~0.61 GB | 12 capas, dim 1024, on-demand |
| Projectors (visión + audio) | ~0.07 GB | 67 MB reales del safetensors |
| KV cache (4K tokens, GQA 4:1) | ~0.19 GB | Con KV sharing habilitado |
| CUDA overhead | ~0.3 GB | Rust FFI + runtime |
| **TOTAL texto-only** | **~5.3 GB** | ✅ Cabe holgado en 12 GB |
| **TOTAL multimodal (todo activo)** | **~6.2 GB** | ✅ Cabe con 5.8 GB headroom |

**Headroom:** ~5.8 GB — suficiente para batch, contexto largo y picos.

### 6.2 Escenario B: ISQ NO cuantiza embed_tokens_per_layer (riesgo)

| Componente | VRAM (fp16) | Notas |
|---|---|---|
| Text layers Q4K (42 capas) | ~2.0 GB | Pesos 4-bit |
| **embed_tokens_per_layer** fp16 | **~5.64 GB** | ⚠️ 35% del modelo SIN cuantizar |
| Embedding/LM Head (shared, fp16) | ~1.34 GB | |
| Per-layer projections Q4K | ~0.03 GB | |
| CUDA overhead | ~0.3 GB | |
| **TOTAL texto-only** | **~9.3 GB** | 🔴 Solo 2.7 GB headroom |
| + Vision encoder | ~0.34 GB | |
| + Audio encoder | ~0.61 GB | |
| + Projectors | ~0.07 GB | |
| + KV cache | ~0.19 GB | |
| **TOTAL multimodal** | **~10.8 GB** | 🔴 Sin margen, riesgo OOM alto |

### 6.3 Decisión Clave

**Verificar en Fase 2** si ISQ cuantiza `embed_tokens_per_layer`. Si no lo hace:
1. Confirmar si se puede forzar cuantización manual (ej: convertir a GGUF solo ese tensor)
2. Evaluar si el modelo funciona sin ese tensor en fp16 (offload a CPU/RAM)
3. Si ningún workaround funciona, operar en modo texto-only (sin encoders) con ~9.3 GB VRAM

### Presupuesto RAM Objetivo

| Componente | RAM Estimado |
|---|---|
| Proceso mistralrs (Rust) | 50–100 MB |
| nginx worker | 5–20 MB |
| CUDA driver userspace | ~30 MB |
| **TOTAL** | **~100–150 MB** |

**vs Phase 1:** 1.6 GB → 0.15 GB = **90% de reducción de RAM.**

**Swap:** zram (comprimido en RAM). Al liberar 1.5 GB de Python, zram tendrá más espacio disponible.

---

## 7. Criterios de Aceptación

### Obligatorios (Must Have)

- [ ] `mistralrs serve` arranca sin error y carga modelo heretic
- [ ] `curl localhost:8080/v1/chat/completions` con texto responde 200 OK
- [ ] Request con imagen (base64) responde sin CUDA OOM
- [ ] Request con audio (WAV) responde correctamente
- [ ] VRAM total ≤ 8.5 GB durante inferencia multimodal
- [ ] RAM del proceso mistralrs ≤ 150 MB
- [ ] Streaming SSE funciona (`curl -N`)
- [ ] nginx reverse proxy funciona con auth
- [ ] systemd service `gemma4-rs.service` arranca, para, y reinicia correctamente
- [ ] `nvidia-smi` confirma VRAM dentro de budget

### Deseables (Nice to Have)

- [ ] Latencia text-only ≤ 10s para 128 tokens
- [ ] RAG integrado con FAISS-CPU
- [ ] SSL/TLS configurado en nginx
- [ ] Benchmark documentado vs Phase 1

---

## 8. Riesgos

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|---|
| R1 | ISQ no cuantiza embed_tokens_per_layer (5.64 GB) | **Alta** | **Crítico** | ⚠️ Verificar en Fase 2. Si falla: modo texto-only con ~9.3 GB, evaluar offload CPU/RAM de ese tensor |
| R2 | Compilación mistral.rs falla en Arch con CUDA 13.3 | Media | Alto | Usar features específicas: `cuda flash-attn cudnn`. Fallback: sin flash-attn. |
| R3 | VRAM excede 8.5 GB con encoders activos | Media | Medio | Reducir max_seq_len, ajustar kv-cache. Verificar en Fase 2. |
| R4 | cudnn conflictúa con CUDA 13.3 | Baja | Alto | `pacman -S cudnn` versionada para CUDA. Verificar versiones. |
| R5 | Audio no funciona en E4B (solo 12B/27B) | Baja | Medio | audio_tower presente en config.json de heretic (12 capas). Debería funcionar. |
| R6 | Swap zram insuficiente durante compilación Rust | Alta | Medio | Compilación usa ~2-3 GB RAM. Cerrar server.py antes de compilar. zram es más flexible que swap file. |
| R7 | RAG post-migración complejo de integrar | Media | Bajo | Servicio separado o pre-computed embeddings. No bloquea release. |

---

## 9. Dependencias de Sistema

### Requisitos Previos (verificar/instalar antes de empezar)

```bash
# 1. cudnn (requerido por mistral.rs CUDA)
echo "PASSWORD" | sudo -S pacman -S cudnn

# 2. nginx
echo "PASSWORD" | sudo -S pacman -S nginx

# 3. FFmpeg (VERIFICAR — ya instalado en el sistema)
ffmpeg -version   # Debe mostrar versión instalada

# 4. Verificar toolchain Rust
rustc --version   # Debe ser >= 1.96.0
cargo --version   # Debe ser >= 1.96.0

# 5. Verificar CUDA
/opt/cuda/bin/nvcc --version  # Debe mostrar 13.3.33
nvidia-smi                     # Debe mostrar driver 610.43.02, RTX 3060 12GB
```

### Compilación mistral.rs

```bash
# Instalar mistralrs-cli desde crates.io
cargo install mistralrs-cli --features "cuda flash-attn cudnn"

# Verificar instalación
mistralrs --version
```

---

## 10. Outcomes Esperados

| Métrica | Phase 1 (actual) | Phase 2 (target) | Mejora |
|---|---|---|---|
| VRAM idle | 9.3 GB | ~5.0 GB | -46% |
| VRAM inferencia text | ~10.0 GB | ~5.3 GB | -47% |
| VRAM inferencia vision | OOM 🔴 | ~6.2 GB ✅ | De OOM a funcional |
| VRAM inferencia audio | N/A | ~6.2 GB | Nuevo capability |
| RAM inference engine | ~1.6 GB | ~0.1 GB | -94% |
| RAM libre sistema | 380 MB (worst-case) | ~1.8 GB | +370% |
| Archivos de código | server.py 2221 líneas | 1 config TOML + nginx.conf | -95% |
| Lenguaje inference | Python | Rust | Zero Python overhead |
