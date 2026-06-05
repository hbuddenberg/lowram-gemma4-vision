# Consenso Multi-Agente: Revision Phase 2

**Fecha:** 2026-06-05  
**Agentes:** AGY (Gemini), OC (OpenCode), CC (Claude Code)  
**Metodo:** Revision independiente de PRD, TRD e IMPLEMENTATION.md, luego sintesis

---

## Hallazgo Critico: nvcc SI Esta Instalado

**Los tres agentes descubrieron independently que CUDA 13.3.0-1 con nvcc esta disponible en `/opt/cuda/bin/nvcc`.**

Esto INVALIDA la premisa del PRD de que vLLM esta bloqueado. Se abre una cuarta opcion.

```
nvcc --version: release 13.3, V13.3.33
pacman -Q cuda: cuda 13.3.0-1
```

---

## Tabla de Consenso

| Aspecto | AGY (Gemini) | OC (OpenCode) | CC (Claude) | **CONSENSO** |
|---|---|---|---|---|
| **mistral.rs es la mejor opcion?** | Si, con HTTP server nativo | Si, unico con todo integrado | Si | **SI** |
| **Eliminar Python FastAPI?** | Si, usar `mistralrs serve` | Si, pero proxy para auth/RAG | Si, Rust puro | **SI (con matices)** |
| **ISQ vs GGUF?** | ISQ (sin conversion) | ISQ (GGUF no incluye vision/audio) | ISQ | **ISQ** |
| **Audio en E4B?** | Si, nativo | Si, audio_tower existe | Si | **SI** |
| **Heretic + ISQ funciona?** | Si (weights no afectan arch) | Si (verificar config.json) | Si | **SI (verificar)** |
| **VRAM budget 4-6GB?** | ~5.5-6.5GB (realista) | 7-8.5GB (irrealistico) | Realista con ajustes | **6-8GB (revisar)** |
| **vLLM viable ahora?** | Si (nvcc existe!) | Si, nueva opcion | Si | **OPCION C** |
| **cudnn necesario?** | Si (`pacman -S cudnn`) | No mencionado | Si | **Instalar** |
| **Zero Python posible?** | Si con mistral.rs HTTP | Proxy nginx para auth | Rust puro + auth separada | **SI con proxy** |

---

## Las Tres Opciones (consensuadas)

### Opcion A: mistral.rs HTTP Server (PURISTA RUST)

```
Dispositivo -> nginx (auth/SSL) -> mistralrs serve (Rust, CUDA) -> GPU
                                   |
                               RAG: FAISS (CPU, separado)
```

**Pros:**
- Zero Python en inference path
- Un solo binario Rust
- OpenAI-compatible HTTP server built-in
- ISQ 4-bit, multimodal completo
- VRAM ~6-8GB estimado

**Contras:**
- Auth basica (necesita proxy nginx para API keys)
- RAG requiere servicio separado o plugin
- Tool calling? (verificar soporte)
- Compilar con CUDA requiere cudnn

**Setup:**
```bash
pacman -S cudnn
cargo install mistralrs-cli --features cuda,flash-attn
mistralrs serve \
  -m ~/models/gemma4-heretic \
  --isq Q4K \
  --port 11434 \
  --arch Gemma4
```

### Opcion B: mistral.rs Backend + Proxy Ligero

```
App -> FastAPI (auth, RAG, rate limit) -> mistralrs Python bindings -> Rust -> GPU
```

Igual que el plan original pero con la correccion de VRAM y nvcc.

**Pros:**
- Auth/RAG/rate-limit ya existentes
- Migracion incremental
- RAG integrado en request flow

**Contras:**
- Sigue teniendo Python en el path
- Overhead de Python (~0.3-0.5GB extra)
- Mas complejo

### Opcion C: vLLM (NUEVA - gracias a nvcc)

```
App -> nginx -> vLLM server (Python, CUDA JIT) -> GPU
```

**Pros:**
- Motor de inference SOTA para throughput
- OpenAI-compatible nativo
- PagedAttention (KV cache eficiente)
- Cuantizacion AWQ/GPTQ

**Contras:**
- Python (no Rust puro)
- Compilacion JIT al inicio (~5-10 min)
- Mas pesado que mistral.rs
- Gemma 4 multimodal? (verificar soporte en vLLM)
- Requiere mas VRAM base

---

## Correcciones al PRD/TRD

| Item Original | Correccion |
|---|---|
| "vLLM bloqueado (nvcc no disponible)" | **FALSO.** nvcc 13.3 instalado en /opt/cuda/ |
| VRAM target: 4-6GB total | **Revisar a 6-8GB** (projectors + KV cache pesan) |
| "Phase 1 usa 11.6GB" | Confirmado. Aun asi, 6-8GB es mejora significativa |
| FastAPI como API layer | **Eliminar.** Usar mistral.rs HTTP + nginx proxy |
| "Audio puede no funcionar en E4B" | **Funciona.** audio_tower presente en E4B |
| cudnn no mencionado | **Requerido** para mistral.rs con CUDA features |

---

## Recomendacion Final del Consenso

### **Opcion A: mistral.rs HTTP Server (Rust Puro)**

Los tres agentes coinciden que mistral.rs con su HTTP server integrado es el camino correcto para Rust puro.

**Razones:**
1. Unico engine Rust con soporte completo Gemma 4 multimodal
2. ISQ 4-bit sin conversion (mantiene heretic safetensors)
3. HTTP server OpenAI-compatible built-in (elimina FastAPI)
4. Audio nativo en E4B
5. Compilacion con nvcc disponible

**Arquitectura final propuesta:**
```
                    ┌─────────────────┐
  Discord App  <--> │  nginx/Caddy    │  Auth + SSL + Rate Limit
  Mobile App   <--> │  :443 / :80    │
  curl         <--> │                 │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  mistralrs      │  Rust binary, CUDA
                    │  serve :11434   │  OpenAI-compatible
                    │  ISQ Q4K        │  Text + Vision + Audio
                    └────────┬────────┘
                             │ CUDA
                    ┌────────▼────────┐
                    │  RTX 3060 12GB   │
                    └─────────────────┘

  CPU (paralelo):
  ┌──────────────────────────┐
  │  FAISS-CPU (RAG search)  │  Endpoint separado o plugin
  │  sentence-transformers   │
  └──────────────────────────┘
```

**VRAM estimado realista:**
| Componente | Estimado |
|---|---|
| LLM E4B @ ISQ Q4K | ~3.0-3.5 GB |
| Vision encoder (SigLIP) | ~0.5 GB |
| Audio encoder | ~0.3-0.5 GB |
| Projectors (vision+audio) | ~1.0-1.5 GB |
| KV cache (4K tokens) | ~1.0-1.5 GB |
| Overhead Rust/CUDA | ~0.3 GB |
| **TOTAL** | **~6-8 GB** |
| **Headroom** | **~4-6 GB** |

Suficiente para vision + audio simultaneos + contexto razonable.

---

## Proximos Pasos

1. `pacman -S cudnn` (requerido para mistral.rs CUDA)
2. `cargo install mistralrs-cli --features cuda,flash-attn`
3. Probar: `mistralrs serve -m ~/models/gemma4-heretic --isq Q4K --port 11434 --arch Gemma4`
4. Benchmark VRAM real
5. Configurar nginx como reverse proxy (auth + rate limit)
6. Integrar FAISS como servicio separado o middleware
7. Actualizar PRD/TRD/IMPLEMENTATION con correcciones
