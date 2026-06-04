# nuc-gemma4-vision — Product Requirements Document

## 1. Executive Summary

Plataforma de inferencia multimodal (texto + visión) que corre un modelo Gemma 4 E4B de 15GB en una GPU externa RTX 3060 12GB conectada via Thunderbolt 3 a un Intel NUC7i5BNB con solo 7.6GB de RAM. El sistema utiliza quantización 4-bit NF4 para el backbone de texto y mantiene la visión tower en fp16, logrando ~5-7 tok/s con visión funcional.

**Diferenciador:** Setup completo y reproducible para correr modelos multimodales grandes en hardware consumer via eGPU, incluyendo el fix crítico del `embed_vision` que causa ceguera en la visión.

## 2. Tech Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Host OS | Arch Linux (kernel zen 7.x) | Low latency, nvidia-open-dkms support |
| Thunderbolt | Alpine Ridge 2C+4C, Legacy Mode | Único modo compatible con Linux |
| GPU Driver | nvidia-open-dkms 610.43.02 | Kernel zen requiere open variant |
| Modelo | igorls/gemma-4-E4B-it-heretic | Multimodal (texto+visión+audio), fine-tune heretic |
| Inference | transformers 5.10+ + bitsandbytes 0.49+ | 4-bit quant con skip de módulos de visión |
| Quantización | NF4 4-bit texto, bfloat16 visión | 9.3GB VRAM total en RTX 3060 12GB |
| Runtime | Python 3.11, PyTorch 2.x, CUDA 13.3 | Estándar de transformers |
| Package Manager | uv | Fast, venv-free, reproducible |

## 3. Features

| # | Feature | Priority | Description |
|---|---------|----------|-------------|
| 1 | Text inference | P0 | Generación de texto en español/inglés, ~4 tok/s |
| 2 | Vision inference | P0 | Análisis de imágenes con descripción detallada, ~5-7 tok/s |
| 3 | CLI inference | P0 | Script `inference.py` con args para texto/imagen |
| 4 | Reproducible setup | P0 | README paso a paso con verificaciones |
| 5 | embed_vision fix | P0 | Skip list completo para bitsandbytes |
| 6 | API server | P1 | FastAPI wrapper para inference como servicio |
| 7 | Ollama integration | P1 | Modelo GGUF para texto-only via Ollama |
| 8 | Batch processing | P2 | Procesar múltiples imágenes secuencialmente |
| 9 | Benchmark suite | P2 | Medir tok/s, VRAM, latencia por tipo de input |
| 10 | Docker/Podman | P2 | Container reproducible con todo incluido |

## 4. User Stories

> "Como usuario con hardware limitado, quiero correr modelos multimodales sin comprar una PC nueva."

> "Como investigador, quiero documentar exactamente cómo configurar TB3 + eGPU en Linux para que otros repliquen."

> "Como desarrollador, quiero una API local de visión para integrar en mis proyectos sin depender de servicios cloud."

## 5. What's NOT in v1 (No-Alcance)

- Fine-tuning o training
- Audio inference (no soportado por el pipeline actual)
- Multi-GPU setup
- Streaming de tokens en tiempo real
- Web UI / chat interface
- Model serving en producción (MPS, TensorRT)
- GPU sharing con otras apps durante inference

## 6. Hardware Requirements

### Mínimo (verificado)

| Componente | Spec |
|-----------|------|
| CPU | Intel i5-7260U (2C/4T) |
| RAM | 7.6GB DDR4 |
| GPU | NVIDIA RTX 3060 12GB |
| eGPU Enclosure | Razer Core X Chroma o similar |
| Thunderbolt | TB3 (Alpine Ridge o superior) |
| Storage | 30GB libres (modelo + deps) |

### Recomendado

| Componente | Spec |
|-----------|------|
| CPU | Intel i7 o superior |
| RAM | 16GB+ |
| GPU | NVIDIA RTX 3060 Ti 16GB / RTX 4060 |
| Storage | NVMe SSD |

## 7. KPIs & Success Criteria

| KPI | Target | Actual |
|-----|--------|--------|
| Text speed | ≥3 tok/s | 4.2 tok/s ✅ |
| Vision speed | ≥3 tok/s | 5.6-7.5 tok/s ✅ |
| VRAM usage | <11GB | 9.3GB ✅ |
| Vision accuracy | Describe imagen correctamente | ✅ Identifica bee, anime, colores |
| Setup time | <60 min | ~45 min documentado |

## 8. Risks

| Risk | Probabilidad | Impacto | Mitigación |
|------|-------------|---------|------------|
| TB3 no detectado en reboot | Media | Alto | Legacy Mode + cold boot obligatorio |
| OOM al cargar safetensors | Alta | Alto | `vm.overcommit_memory=1` |
| Vision ve gris | Alta (si no se fix) | Alto | Skip list completo en BitsAndBytesConfig |
| nouveau conflicta | Alta | Medio | Blacklist en modprobe.d |
| Disk full (55GB total) | Media | Alto | Limpiar cache HF, usar /tmp para temporales |
| Kernel update rompe nvidia | Media | Alto | DKMS auto-rebuild, pin kernel version |

## 9. Roadmap

### Phase 1 — Foundation (Completado ✅)
- TB3 setup + NVIDIA driver
- Modelo descargado y funcionando
- Text + Vision inference
- embed_vision bug fix
- GitHub repo

### Phase 2 — API Layer (Próximo)
- FastAPI wrapper con endpoints `/text` y `/vision`
- Health check + GPU status endpoint
- OpenAI-compatible API wrapper

### Phase 3 — Optimization
- Benchmark suite automatizada
- Batch processing de imágenes
- Torch.compile() para acelerar inference

### Phase 4 — Distribution
- Dockerfile reproducible
- Documentación de configuraciones alternativas (diferentes GPUs, modelos)
- Script de setup automatizado
