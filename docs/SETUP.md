# SETUP.md — Infraestructura LowRAM Gemma 4 Vision

> Última actualización: 2026-06-06

## Hardware

| Componente | Detalle |
|------------|---------|
| **Host** | NUC7i5BNB (Intel i5-7260U, 7.6GB RAM) |
| **GPU** | NVIDIA RTX 3060 12GB via Thunderbolt 3 (Legacy Mode) |
| **eGPU** | 2× Razer Core X |
| **Driver** | nvidia-open-dkms 610.43.02 |

## Software

| Componente | Versión | Path |
|------------|---------|------|
| **OS** | Arch Linux (kernel 7.0.10-zen1-1-zen) | - |
| **gcc** | 16.1.1 | /usr/bin/gcc |
| **CUDA Toolkit** | 13.3.33 | `/usr/local/cuda-13.3` → `/usr/local/cuda` |
| **CUDA Driver** | 13.3 (via nvidia 610.43.02) | - |
| **Ollama** | 0.30.3 | `/usr/local/bin/ollama` (systemd --user) |
| **Hermes** | Gateway (systemd --user) | `~/.hermes/hermes-agent/` |
| **Python** | 3.11.15 | sistema |
| **PyTorch** | venv project | `~/lowram-gemma4-vision/venv/` |

## Modelo Gemma 4 12B

| Parámetro | Valor |
|-----------|-------|
| **Modelo** | `igorls/gemma-4-12B-it-heretic-GGUF` |
| **Arquitectura** | gemma4 (48 layers, 16 attention heads, GQA) |
| **Parámetros** | 11.9B |
| **Quantización** | Q4_K_M (4-bit) |
| **Context length** | 131,072 tokens (128K) |
| **Embedding dims** | 3,840 |
| **RoPE base** | 1,000,000 |
| **Sliding window** | 1,024 (SWA local) |
| **Multimodal** | ✅ Visión + Audio via mmproj |
| **Disco** | 7.4 GB |

### Archivos GGUF

```
~/models/gemma4-12b-gguf/
├── gemma-4-12B-it-Q4_K_M.gguf          # 6.9 GB — Modelo texto
└── mmproj-gemma-4-12B-it-Q8_0.gguf     # 152 MB — Proyección visión
```

### Archivos Safetensors (Phase 1)

```
~/models/gemma4-heretic/                  # igorls/gemma-4-E4B-it-heretic
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── tokenizer.json
└── config.json
```

## Servicios

### Ollama (`systemctl --user`)

```bash
# Status
systemctl --user status ollama.service

# Restart
systemctl --user restart ollama.service

# Logs
journalctl --user -u ollama.service -f
```

**Config:** `~/.config/systemd/user/ollama.service`
- `OLLAMA_HOST=127.0.0.1:11434`
- `OLLAMA_MODELS=~/.ollama/models`

### Hermes Gateway (`systemctl --user`)

```bash
systemctl --user restart hermes-gateway.service
```

### API Endpoints

```
GET  http://localhost:11434/api/version         → {"version":"0.30.3"}
GET  http://localhost:11434/api/tags             → Modelos disponibles
POST http://localhost:11434/api/chat             → Chat nativo (funciona)
POST http://localhost:11434/v1/chat/completions  → OpenAI-compatible (⚠️ contenido vacío con Gemma 4 GGUF)
```

## Hermes Config (provider ollama-launch)

```yaml
model:
  api_key: ollama
  base_url: http://127.0.0.1:11434/v1
  default: igorls/gemma-4-12B-it-heretic-GGUF
  provider: ollama-launch

providers:
  ollama-launch:
    api: http://127.0.0.1:11434/v1
    default_model: igorls/gemma-4-12B-it-heretic-GGUF
    models:
      - igorls/gemma-4-12B-it-heretic-GGUF
      - igorls/gemma-4-12B-it-heretic-GGUF:latest
```

## Workaround gcc 16.1.1 + CUDA 13.3

gcc 16.x introdujo cambios en headers C++ estándar que rompen compatibilidad con headers CUDA. Solución:

```bash
# Flag para nvcc
-DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler"

# Verificado: llama.cpp compila con este flag
# Rust frameworks (candle, mistral.rs via cudarc) siguen fallando
```

## VM Overcommit

```bash
# Requerido para mmap de modelos grandes
sysctl vm.overcommit_memory=1

# Persistente
echo "vm.overcommit_memory=1" | sudo tee /etc/sysctl.d/99-overcommit.conf
```

## RAG (Phase 1 — Gemma 4 E4B)

```
Backend: faiss-cpu + sentence-transformers (all-MiniLM-L6-v2, CPU)
Store: ~/.gemma4api/rag/
Dynamic budget: 4 text / 2 vision / 3 audio
Image resize: 896px
```

## Troubleshooting

### Ollama API v1 retorna contenido vacío
- **Problema:** `/v1/chat/completions` retorna `content: ""` con modelos GGUF custom
- **Solución:** Usar API nativa `/api/chat` o `/api/generate`
- **Hermes:** Provider `ollama-launch` maneja esto internamente

### CUDA compilation error `is_void undefined`
- **Causa:** gcc 16.1.1 + CUDA headers incompatibles
- **Fix:** `--allow-unsupported-compiler` en nvcc flags
- **Alternativa:** Docker con gcc 12-13 (no testeado)

### /tmp lleno durante compilación
- **Causa:** `/tmp` es tmpfs (RAM) con 3.9GB, CUDA extrae 4GB+
- **Fix:** `TMPDIR=~/Downloads/tmp sudo ...` o aumentar tmpfs

## Historial de Decisiones

| Fecha | Decisión | Razón |
|-------|----------|-------|
| 2026-06-03 | Phase 1: BitsAndBytes NF4 + fp16 visión | Fix crítico: embed_vision en llm_int8_skip_modules |
| 2026-06-05 | Ollama como runtime | Pre-built binario, sin compilación gcc |
| 2026-06-05 | CUDA 13.3 toolkit en /usr/local/cuda | Path estándar, sin symlinks manuales |
| 2026-06-06 | Modelo 12B Q4_K_M via Ollama | Balance calidad/VRAM (7.4GB de 12GB) |
