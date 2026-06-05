# Decision Final: Multi-Agent Validation CLI

**Fecha:** 2026-06-05  
**Agentes:** AGY, OC, CC — ejecucion con comandos reales del sistema  
**Resultado:** CONSENSO UNANIME — Opcion A (mistral.rs + nginx)

---

## Datos Reales del Sistema (confirmados por los 3 agentes)

| Dato | Valor | Verificado por |
|------|-------|---------------|
| CPU | i5-7260U @ 2.20GHz (2C/4T) | CC |
| RAM total | 7,807 MB | CC |
| **RAM libre** | **380 MB** | CC |
| GPU | RTX 3060 12GB, compute cap 8.6 | AGY, OC, CC |
| VRAM en uso | 9,793 MB (server.py) | CC |
| CUDA | 13.3.33, nvcc en /opt/cuda/bin/ | AGY, OC, CC |
| Rust | 1.96.0 + Cargo 1.96.0 | AGY |
| cudnn | NO instalado (disponible en pacman) | AGY |
| nginx | NO instalado | CC |
| Modelo | 19GB safetensors, 1 shard | OC, CC |
| Model arch | Gemma4ForConditionalGeneration | AGY, OC, CC |
| Layers | 42 texto (2560d) + 16 vision (768d) + 12 audio (1024d) | CC |

---

## Hallazgo Critico: RAM, no VRAM

Los 3 agentes coinciden: **el cuello de botella real es RAM (380MB libre), no VRAM (2.1GB libre).**

Python + PyTorch consume 1.6GB RAM. mistral.rs consume ~50-100MB. Eliminar Python libera ~1.5GB RAM.

---

## Decision por Agente

### AGY (Gemini): mistral.rs SI ✅
- Compila con CUDA: SI (cargo 1.96.0 + nvcc 13.3.33)
- ISQ Q4K: SI (cuantiza safetensors directo, sin conversion)
- VRAM estimado: 7-8.5GB (3.5GB margen en 12GB)
- cudnn: `pacman -S cudnn` antes de compilar
- Heretic compatible: SI (misma arch, pesos abliterated no afectan)

### OC (OpenCode): vLLM NO ❌, mistral.rs SI ✅
- vLLM Gemma 4 multimodal: SI soportado (gemma4_mm)
- PERO: vLLM necesita 3-5GB RAM → 380MB libre = OOM garantizado
- VRAM: 8.5-11GB estimado (muy justo con vision+audio)
- Downgrade torch 2.12→2.11 (rompe setup)
- JIT compilation 5-15 min en i5-7260U
- Necesita pre-cuantizacion AWQ/GPTQ
- llama.cpp: NO viable (sin vision/audio para Gemma 4)

### CC (Claude): Opcion A — mistral.rs + nginx ✅
- Razon fundamental: RAM es el recurso mas escaso
- Eliminar Python libera 1.5GB RAM
- nginx suficiente para auth personal (1-5 usuarios)
- RAG viable post-migracion con embeddings ONNX (~130MB) o pre-computed
- Plan: 4 fases, 7-10 dias habiles

---

## Consenso Final

```
🥇 mistral.rs  → UNANIME (3/3)
🥈 bitsandbytes → Funciona pero Python-heavy (RAM)
🥉 vLLM         → OC rechazo (RAM insuficiente)
💀 llama.cpp    → Sin vision/audio
```

---

## Arquitectura Aprobada

```
Cliente → HTTPS :443 → nginx (auth, rate-limit, SSL)
                        ↓
              mistral.rs serve :8080 (Rust, CUDA)
                        ↓
              ~/models/gemma4-heretic/ (safetensors, ISQ Q4K)
              VRAM: ~7-8.5GB / 12GB
              RAM: ~100MB (vs 1.6GB actuales)
```

---

## Proximos Pasos

1. `echo 7907 | sudo -S pacman -S cudnn nginx`
2. `cargo install mistralrs --features cuda`
3. `mistralrs run --quant 4 -m ~/models/gemma4-heretic` (verificar carga)
4. `mistralrs serve --quant 4 -m ~/models/gemma4-heretic --port 8080` (verificar API)
5. Test vision con curl
6. Benchmark VRAM/RAM
7. Configurar nginx reverse proxy
8. Crear systemd service
