# lowram-gemma4-vision — Implementation Plan

## Estado: Phase 1 ✅ Completada

---

## Phase 1 — Foundation ✅ (Completada)

### Setup TB3 + GPU
- [x] Verificar Alpine Ridge TB3 en PCI
- [x] Configurar BIOS: Thunderbolt = Legacy Mode
- [x] Cold boot + verificar TB3 devices
- [x] Instalar nvidia-open-dkms + linux-zen-headers
- [x] Blacklist nouveau
- [x] Verificar nvidia-smi: RTX 3060 12GB

### Modelo + Inference
- [x] Instalar Python deps (transformers, bitsandbytes, torch, torchvision)
- [x] Descargar modelo igorls/gemma-4-E4B-it-heretic (15GB safetensors)
- [x] Configurar vm.overcommit_memory=1
- [x] Script inference.py (texto + visión)
- [x] Fix embed_vision quantization bug
- [x] Verificar visión funcional (test con abeja + anime)

### Documentación + Repo
- [x] README.md completo
- [x] GitHub repo: hbuddenberg/lowram-gemma4-vision
- [x] PRD.md
- [x] TRD.md
- [x] Este documento (IMPLEMENTATION.md)

---

## Phase 2 — API Layer (Próximo)

### Semana 1: FastAPI Server
- [ ] Crear `server.py` con FastAPI
- [ ] Endpoint `POST /text` — generación de texto
- [ ] Endpoint `POST /vision` — análisis de imagen (multipart)
- [ ] Endpoint `GET /health` — GPU status + VRAM
- [ ] Modelo se carga al iniciar, persiste en memoria
- [ ] Timeout handling para requests largos

### Semana 1: OpenAI-Compatible Wrapper
- [ ] Endpoint `POST /v1/chat/completions`
- [ ] Soporte para content type `image_url`
- [ ] Endpoint `GET /v1/models`
- [ ] Streaming de tokens (SSE)
- [ ] Compatible con Hermes Gateway como provider

---

## Phase 3 — Optimization

### Benchmark Suite
- [ ] Script `benchmark.py` con métricas:
  - tok/s por tipo (texto, visión, mixed)
  - VRAM pico por tipo de input
  - Latencia first-token (TTFT)
  - Throughput con distintos max_new_tokens
- [ ] Resultados en `docs/benchmarks/`
- [ ] Comparación: Q4_K_M GGUF (Ollama) vs NF4 safetensors

### Batch Processing
- [ ] Endpoint `POST /vision/batch` — procesar N imágenes
- [ ] Queue con asyncio para no saturar VRAM
- [ ] Resultados en JSON con metadata

### Performance Tuning
- [ ] Evaluar `torch.compile()` para speedup
- [ ] Probar diferentes bnb_4bit_compute_dtype
- [ ] KV cache optimization
- [ ] Evaluar Flash Attention 2

---

## Phase 4 — Distribution

### Docker
- [ ] Dockerfile con CUDA base + Python deps
- [ ] Model download como build step opcional
- [ ] docker-compose con GPU passthrough
- [ ] Publicar en ghcr.io

### Script de Setup
- [ ] `scripts/setup.sh` automatizado:
  - Detectar TB3 devices
  - Instalar nvidia-open-dkms
  - Configurar sysctl + blacklist
  - Descargar modelo
  - Verificar inference
- [ ] Soporte para diferentes distros (Arch, Ubuntu)

### Documentación Extendida
- [ ] Guía de troubleshooting (TB3 no detectado, OOM, nouveau, etc.)
- [ ] Configuraciones alternativas (RTX 4060, RX 7600, etc.)
- [ ] Guía para otros modelos multimodales (LLaVA, Qwen-VL)

---

## Resumen de Milestones

| Hito | Estado | Feature Principal |
|------|--------|------------------|
| Phase 1 | ✅ | TB3 + GPU + Texto + Visión |
| Phase 2 | 🔲 | FastAPI + OpenAI-compatible API |
| Phase 3 | 🔲 | Benchmarks + Batch + torch.compile |
| Phase 4 | 🔲 | Docker + Setup script + Docs extendidas |

## Dependencies

```
Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4
  ✅          Próximo
```

Phase 3 y 4 son independientes entre sí — pueden paralelizarse si se usa delegate_task.
