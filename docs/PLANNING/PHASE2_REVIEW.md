# Revisión Técnica: Phase 2 — lowram-gemma4-vision

> **Revisor:** CC (Senior ML Infrastructure Engineer)  
> **Fecha:** 2026-06-05  
> **Rama:** `phase2`

---

## Resumen Ejecutivo

Los tres documentos (PRD, TRD, IMPLEMENTATION) describen un sistema Phase 1 funcional basado en **Python/transformers/bitsandbytes** que corre Gemma 4 E4B Heretic en una RTX 3060 12GB. Phase 2 propone agregar una capa FastAPI + OpenAI-compatible API sobre esta base.

**El problema fundamental:** El contexto de la tarea menciona "migrar a Rust" y "pure Rust inference", pero **los documentos actuales NO proponen ninguna migración a Rust**. Los tres docs son 100% Python/transformers. La "Phase 2" actual es simplemente "poner FastAPI delante del inference.py existente" — lo cual **ya está implementado** en `server.py` (2221 líneas con auth, rate limiting, streaming, RAG, key management).

**Hay dos realidades que chocan:**
1. El `server.py` actual YA ES un servidor FastAPI OpenAI-compatible completo y funcional (corriendo en PID 93607, consumiendo 9.7GB VRAM).
2. El deseo declarado de "pure Rust" contradice completamente la arquitectura existente.

---

## Hallazgo Crítico #1: nvcc SÍ está instalado

El contexto dice: *"nvidia-open-dkms (NO nvcc/CUDA toolkit)"* y *"vLLM is BLOCKED (needs nvcc)"*.  
**Esto es FALSO en el sistema actual:**

```
$ /opt/cuda/bin/nvcc --version
Cuda compilation tools, release 13.3, V13.3.33

$ pacman -Qs cuda
local/cuda 13.3.0-1  (NVIDIA's GPU programming toolkit)
local/cccl 3.3.4-1   (CUDA Core Compute Libraries)

$ ldconfig -p | grep cublas
libcublas.so.13 => /opt/cuda/lib64/libcublas.so.13
libcublasLt.so.13 => /opt/cuda/lib64/libcublasLt.so.13
```

El sistema tiene CUDA 13.3 completo, nvcc incluido. vLLM **NO está bloqueado** por falta de nvcc. Cualquier decisión arquitectónica basada en "no tengo nvcc" debe ser reevaluada.

---

## Hallazgo Crítico #2: server.py ya existe y es masivo

El `server.py` tiene **2221 líneas** e implementa:
- OpenAI-compatible `/v1/chat/completions` (streaming + non-streaming)
- Auth con API keys (SHA-256 hash index, O(1) lookup)
- Rate limiting per-key (sliding window)
- Key lifecycle (create/rotate/delete/expire)
- SQLite persistence + in-memory cache
- RAG integration (`rag.py`)
- SSRF protection para image URLs
- Usage analytics
- systemd service (`gemma4-api.service`)
- Web UI ready

**Phase 2 de IMPLEMENTATION.md dice "Crear server.py con FastAPI" — ya existe.** El plan de implementación está desactualizado respecto al código real.

---

## Revisión por Punto de Análisis

### 1. ¿mistral.rs es la mejor opción?

**Sí, pero con matices importantes.**

mistral.rs tiene soporte nativo para:
- ✅ Gemma 4 multimodal (texto + visión + audio) vía `MultimodalArchitecture::Gemma4`
- ✅ In-situ quantization (ISQ) a 4-bit desde safetensors
- ✅ Servidor HTTP OpenAI-compatible integrado (`mistralrs serve`)
- ✅ Soporte CUDA vía cublas (el sistema lo tiene)
- ✅ Carga de modelos GGUF pre-cuantizados
- ✅ Web UI integrada en `/ui`

**¿Alternativas?**
- **candle** (Hugging Face): Demasiado bajo nivel. No tiene servidor HTTP ni soporte multimodal nativo para Gemma 4. Requeriría escribir todo desde cero.
- **llama.cpp**: No soporta Gemma 4 multimodal (visión/audio). Solo texto.
- **vLLM**: Ya no está bloqueado (nvcc disponible), pero su soporte multimodal para Gemma 4 es experimental y consumiría más VRAM que bitsandbytes.
- **Ollama**: Solo texto GGUF. No maneja vision natively para Gemma 4.

**Conclusión:** mistral.rs es la mejor opción si se quiere eliminar Python del inference path. Es el único engine que ofrece multimodal Gemma 4 + quantization + HTTP server en un solo binario Rust.

### 2. ¿Se puede eliminar FastAPI y usar el HTTP server de mistral.rs?

**SÍ, parcialmente.** Pero hay tradeoffs significativos:

**Lo que mistral.rs server reemplaza directamente:**
- ✅ `POST /v1/chat/completions` (streaming + non-streaming)
- ✅ `GET /v1/models`
- ✅ Vision input (`image_url`)
- ✅ Audio input (`audio_url`)
- ✅ Web UI en `/ui`
- ✅ OpenAI SDK compatible

**Lo que mistral.rs NO tiene y el server.py actual sí:**
- ❌ API key authentication / key management
- ❌ Rate limiting per-key
- ❌ Usage analytics con SQLite
- ❌ Key rotation / lifecycle
- ❌ SSRF protection configurable
- ❌ RAG integration
- ❌ Request tracing con X-Request-ID
- ❌ Session management

**Recomendación:** Si el objetivo es "pure Rust inference", se puede poner un reverse proxy (nginx/caddy) delante de mistral.rs para auth/rate-limiting, o implementar un middleware Rust ligero. Pero la funcionalidad de management del server.py actual se perdería o necesitaría reescribirse en Rust.

### 3. ¿In-situ quantization es confiable? ¿Pre-convertir a GGUF?

**In-situ quant (ISQ) en mistral.rs funciona, pero tiene limitaciones:**

```bash
# ISQ: cuantiza al cargar (RAM spike temporal)
mistralrs serve --quant 4 -m igorls/gemma-4-E4B-it-heretic
```

**Riesgos de ISQ con safetensors:**
- **RAM spike:** Necesita cargar el modelo completo en fp16/bf16 ANTES de cuantizar. Para un modelo de 15GB en safetensors con 7.6GB RAM + overcommit, esto es riesgoso pero factible (usa mmap + cuantiza por capas).
- **Tiempo de startup:** ISQ añade ~30-60s al cold start comparado con GGUF pre-cuantizado.
- **El bug de embed_vision:** Con bitsandbytes, `embed_vision` se cuantizaba accidentalmente causando "ceguera visual". Con mistral.rs ISQ, **este bug específico no aplica** porque mistral.rs maneja la arquitectura multimodal nativamente y sabe qué módulos excluir de la quantización del language model. Pero es CRÍTICO verificar.

**Pre-conversión a GGUF (recomendado):**
```bash
# Pre-cuantizar una vez, cargar rápido después
mistralrs quantize -m igorls/gemma-4-E4B-it-heretic --quant q4k --out gemma4-heretic-q4k.gguf
```

**Ventajas de GGUF pre-cuantizado:**
- Carga 3-5x más rápida (sin ISQ)
- Sin RAM spike temporal
- Archivo auto-contenido (~4-5GB vs 15GB safetensors)
- Soporte para mmap con memoria virtual controlada
- Portable entre sistemas

**Desventajas de GGUF para multimodal:**
- ⚠️ **El soporte multimodal en GGUF es limitado.** Los archivos GGUF estándar no incluyen la vision tower. mistral.rs puede cargar safetensors para vision + GGUF para text, pero es una configuración híbrida.
- ⚠️ Para audio, similar problema: el modelo base en GGUF no incluye audio tower.

**Recomendación final:** Usar ISQ con safetensors es más seguro para multimodal. Si solo se necesita texto, GGUF es superior.

### 4. ¿El modelo heretic (abliterated) funciona con mistral.rs ISQ?

**Sí, pero hay que verificar.**

El modelo `igorls/gemma-4-E4B-it-heretic` es un fine-tune de Gemma 4 E4B que usa "abliteration" (remoción de refusal via ortogonalización de activaciones). Esto modifica los pesos del language model pero:
- No cambia la arquitectura del modelo
- No modifica la vision tower ni embed_vision
- No introduce capas nuevas ni tipos de operación no estándar

**Riesgo principal:** mistral.rs espera un config.json compatible con la arquitectura `Gemma4`. Si el fine-tune heretic no modificó el config (solo los pesos), funcionará sin problemas. Si cambió el config (ej: vocab size, hidden dims), puede fallar.

**Verificación necesaria:**
```bash
# Comparar config del modelo base vs heretic
diff <(curl -s https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json) \
     <(cat ~/models/gemma4-heretic/config.json)
```

Si los configs son idénticos (excepto metadata), ISQ funcionará. Los pesos abliterated se cuantizan igual que pesos normales — la abliteration es una transformación lineal de los pesos existentes, no introduce nada que NF4 no pueda representar.

**Nota sobre calidad:** La cuantización NF4 puede degradar ligeramente el efecto de la abliteration porque la ortogonalización depende de presición en las direcciones de activación. En practice, Q4_K_M o NF4 debería preservar >95% del comportamiento "uncensored". Vale la pena comparar outputs entre bitsandbytes NF4 y mistral.rs ISQ para verificar.

### 5. ¿Audio en Gemma 4 E4B existe?

**Sí, según la documentación de mistral.rs.**

La doc de mistral.rs muestra explícitamente soporte para audio con `google/gemma-4-E4B-it` usando `audio_url`:
```python
runner = Runner(
    which=Which.MultimodalPlain(
        model_id="google/gemma-4-E4B-it",
        arch=MultimodalArchitecture.Gemma4,
    ),
    in_situ_quant="4",
)
# audio_url content type funciona
```

Sin embargo:
- El modelo E4B (4B effective params) es el más pequeño de la familia Gemma 4
- El TRD menciona `audio_tower` y `embed_audio` en el skip list, lo que confirma que la arquitectura del modelo incluye componentes de audio
- Pero en la PRD Phase 1 se dice explícitamente: *"Audio inference (no soportado por el pipeline actual)"*
- **El inference.py actual NO procesa audio** — solo texto y visión

**Conclusión:** La arquitectura Gemma 4 E4B incluye audio tower. mistral.rs lo soporta nativamente. El pipeline Python actual no lo usa. Con mistral.rs, audio debería funcionar automáticamente si el modelo tiene los pesos de audio. **Verificar si el fine-tune heretic preservó los pesos de audio.**

### 6. ¿El budget de VRAM es realista (4-6GB total)?

**NO. Es irrealmente optimista.**

**VRAM actual verificado (bitsandbytes NF4):**
| Componente | VRAM |
|-----------|------|
| Language Model NF4 | ~3.5GB |
| Vision Tower fp16 | ~0.5GB |
| embed_vision fp16 | ~0.5GB |
| lm_head fp16 | ~0.7GB |
| KV Cache + overhead | ~4.1GB |
| **Total** | **~9.3GB** |

El TRD dice "9.3GB VRAM total" — esto es lo que realmente usa. `nvidia-smi` confirma 9.8GB en uso.

mistral.rs con ISQ a 4-bit debería ser comparable o ligeramente mejor (KV cache más eficiente), pero **no va a bajar a 4-6GB**. El KV cache por sí solo crece con la secuencia:
- ~0.5MB/token para Gemma 4 E4B con KV cache fp16
- 2048 tokens de contexto → ~1GB solo en KV cache
- Con texto + imagen (2520 vision patches) → KV cache crece significativamente

**Presupuesto realista para mistral.rs ISQ:**
| Componente | VRAM estimada |
|-----------|--------------|
| Language Model ISQ 4-bit | ~3.0-3.5GB |
| Vision Tower fp16 | ~0.5GB |
| Projectors + lm_head fp16 | ~1.2GB |
| KV Cache (2048 ctx) | ~1.5-2.0GB |
| CUDA overhead | ~0.5-1.0GB |
| **Total estimado** | **7.0-8.5GB** |

Eso deja 3.5-5GB de margen en 12GB, que es mucho mejor que los 2.3GB actuales con bitsandbytes. Pero 4-6GB total es imposible sin sacrificar calidad drásticamente.

### 7. ¿Hay alternativas mejores que nos estamos perdiendo?

**Opciones a considerar:**

#### A. Qwen3-VL-4B-Instruct (alternativa al modelo)
- 4B params, multimodal nativo
- Soporte vision excelente en mistral.rs
- No requiere fine-tune "heretic" — ya es menos restrictivo por diseño
- Probablemente más rápido que Gemma 4 para inferencia
- **Pero:** Si el usuario quiere específicamente Gemma 4 heretic, esto es irrelevante

#### B. TensorRT-LLM (NVIDIA)
- Máximo rendimiento en RTX 3060
- Soporte multimodal creciente
- **Pero:** Requiere nvcc (disponible), compilation compleja, soporte Gemma 4 puede ser limitado
- Probablemente overkill para un solo usuario

#### C. Llama.cpp + multimodal adapter
- El ecosistema más maduro para inferencia cuantizada
- **Pero:** No soporta Gemma 4 multimodal nativamente. Solo texto GGUF.

#### D. Lo que recomiendo: mistral.rs standalone (sin Python)
```bash
# Un solo comando, sin Python
mistralrs serve \
  --host 0.0.0.0 \
  --port 8080 \
  --quant 4 \
  -m igorls/gemma-4-E4B-it-heretic \
  --arch gemma4
```

Esto da API OpenAI-compatible, vision, audio, quantization, web UI — todo en un binario Rust.

### 8. ¿Se puede lograr ZERO Python en el inference path?

**Sí, absolutamente.**

```bash
# 1. Instalar mistral.rs (pre-built o compilar)
cargo install mistralrs

# 2. Servir el modelo
mistralrs serve --quant 4 -m igorls/gemma-4-E4B-it-heretic

# 3. Consultar con cualquier cliente OpenAI
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}},
      {"type": "text", "text": "Describe esta imagen"}
    ]}]
  }'
```

**Pero se pierde:**
- El sistema de API keys / auth
- Rate limiting
- Analytics
- RAG
- Session management

**Compensación:** Poner un reverse proxy (nginx/caddy/Traefik) delante para auth y rate limiting, o escribir un middleware Rust ligero usando actix-web/axum que proxy a mistral.rs.

---

## Recomendaciones Concretas

### Opción A: mistral.rs standalone (recomendado si ZERO Python es prioridad)

**Pasos:**
1. Compilar mistral.rs con feature `cuda` (el sistema tiene CUDA 13.3 + cublas)
2. `mistralrs serve --quant 4 -m igorls/gemma-4-E4B-it-heretic`
3. nginx/caddy delante para auth básica y rate limiting
4. Migrar la lógica de keys a un servicio externo o simplificar

**Pros:** Zero Python, un solo binario, startup rápido, OpenAI-compatible  
**Contras:** Pierdes el richness del server.py actual (keys, RAG, analytics)  
**Timeline:** 1-2 días para setup básico

### Opción B: mistral.rs como backend, Python thin proxy (recomendado si quieres mantener features)

**Pasos:**
1. mistral.rs como backend inference (reemplaza transformers+bitsandbytes)
2. FastAPI delgado que proxy a mistral.rs y agrega auth/keys/analytics
3. Eliminar la dependencia de transformers, torch, bitsandbytes

**Pros:** Mantienes toda la infra de management, eliminas 90% del overhead Python  
**Contras:** Aún necesitas Python para el proxy (pero liviano)  
**Timeline:** 3-5 días

### Opción C: Quedarse con lo que funciona (no migrar)

**Razón:** El server.py actual funciona. 9.3GB VRAM, 4-7 tok/s, OpenAI-compatible, auth, RAG. Si no hay un pain point claro, no migres.

**Cuando sí migrar:**
- Si necesitas audio (mistral.rs lo tiene, tu pipeline no)
- Si necesitas más velocidad (mistral.rs es más rápido que transformers)
- Si necesitas reducir VRAM (mistral.rs KV cache es más eficiente)
- Si el overhead de Python (2GB RAM) es insostenible

---

## Veredicto Final

| Pregunta | Respuesta |
|----------|-----------|
| ¿mistral.rs es la mejor opción? | **Sí.** Es el único engine que ofrece Gemma 4 multimodal + quant + HTTP en un binario |
| ¿Eliminar FastAPI por mistral.rs HTTP? | **Posible pero pierdes features.** Opción B (thin proxy) es más pragmático |
| ¿ISQ vs GGUF? | **ISQ para multimodal.** GGUF no incluye vision/audio tower |
| ¿Heretic con ISQ funciona? | **Probablemente sí.** Verificar config.json vs base model |
| ¿Audio en E4B? | **Sí existe.** mistral.rs lo soporta. Tu pipeline actual no |
| ¿4-6GB VRAM realista? | **NO.** Realista es 7-8.5GB con ISQ |
| ¿Alternativas mejores? | **No para este caso de uso.** mistral.rs es el mejor fit |
| ¿Zero Python posible? | **Sí, con tradeoffs en features de management** |

### Acción inmediata recomendada:

1. **Verificar que el modelo heretic funciona con mistral.rs:**
   ```bash
   cargo install mistralrs --features cuda
   mistralrs run --quant 4 -m ~/models/gemma4-heretic
   ```
2. **Si funciona, benchmark comparativo:** tok/s, VRAM, calidad de vision (¿ve gris o ve correctamente?)
3. **Decidir entre Opción A o B** basado en si necesitas el key management
4. **Actualizar los documentos** — Phase 2 está completamente desactualizado respecto al código real

---

## Apéndice: Estado Real vs Documentado

| Item | Documentado | Real |
|------|------------|------|
| nvcc disponible | "NO nvcc" | ✅ CUDA 13.3 completo en /opt/cuda |
| Phase 2 status | "Próximo - Crear server.py" | ✅ server.py existe (2221 líneas, funcional) |
| VRAM usage | "9.3GB" | 9.8GB (nvidia-smi confirma) |
| Audio support | "No soportado" | Arquitectura lo incluye, mistral.rs lo soporta |
| vLLM blocked | "Needs nvcc" | nvcc disponible, no está bloqueado |
