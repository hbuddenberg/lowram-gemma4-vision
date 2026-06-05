# Plan de Implementacion - lowram-gemma4-vision Phase 2

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Migrar inferencia de Python/transformers/bitsandbytes a mistral.rs para habilitar multimodal (text+vision+audio) en 12 GB VRAM.

**Architecture:** FastAPI (API layer) -> mistralrs (Python bindings) -> mistral.rs (Rust, CUDA). RAG con FAISS se migra sin cambios.

**Tech Stack:** Python 3.11, mistralrs (PyPI), FastAPI, FAISS-CPU, sentence-transformers, SQLite, Rust/CUDA (mistral.rs)

---

## Task 1: Setup - Entorno y dependencias

**Objective:** Crear entorno de trabajo con mistralrs instalado y verificado.

**Files:**
- Create: `requirements.txt`
- Create: `config.py`

**Step 1: Crear requirements.txt**

```
# Inference (mistral.rs Python bindings)
mistralrs>=0.4.0

# API
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
sse-starlette>=2.0
pydantic>=2.0

# RAG
faiss-cpu>=1.7.0
sentence-transformers>=3.0.0

# Image preprocessing
Pillow>=10.0.0

# Audio
soundfile>=0.12.0

# Utils
requests>=2.28.0
pyyaml>=6.0
```

**Step 2: Crear config.py**

Archivo de configuracion centralizada con dataclasses (ModelConfig, ServerConfig, RAGConfig, BudgetConfig, ImageConfig). Carga desde YAML + env vars.

**Step 3: Instalar dependencias**

```bash
cd ~/lowram-gemma4-vision
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Step 4: Verificar mistralrs**

```python
from mistralrs import Runner, Which, MultimodalArchitecture
print("mistralrs imported successfully")
```

**Step 5: Commit**

```bash
git add requirements.txt config.py
git commit -m "chore: setup phase2 deps and config"
```

---

## Task 2: PoC - Cargar modelo y generar texto

**Objective:** Verificar que mistral.rs carga el modelo heretic y genera texto correctamente.

**Files:**
- Create: `tests/test_text.py`

**Step 1: Escribir test de carga**

```python
from mistralrs import Runner, Which, MultimodalArchitecture, ChatCompletionRequest, SamplingParams

def test_load_and_generate():
    runner = Runner(
        which=Which.MultimodalPlain(
            model_id="~/models/gemma4-heretic",
            arch=MultimodalArchitecture.Gemma4,
        ),
        in_situ_quant="4",
    )
    request = ChatCompletionRequest(
        model="gemma-4-e4b-heretic",
        messages=[{"role": "user", "content": "Hola, dime un haiku sobre Python"}],
        max_tokens=30,
        sampling_params=SamplingParams(temperature=0.7, top_p=0.9),
    )
    response = runner.send_chat_completion_request(request)
    assert response.choices[0].message.content
    print(f"Response: {response.choices[0].message.content}")
```

**Step 2: Ejecutar y verificar VRAM**

```bash
python3 tests/test_text.py
nvidia-smi  # Debe mostrar < 5 GB VRAM usados
```

**Expected:** VRAM <= 5 GB, respuesta coherente en espanol.

**Step 3: Commit**

```bash
git add tests/test_text.py
git commit -m "test: PoC mistral.rs text inference"
```

---

## Task 3: API Layer - FastAPI con mistral.rs runner

**Objective:** Reemplazar generate_sync() y _prepare_inputs() con mistral.rs Runner.

**Files:**
- Create: `server.py` (nuevo, limpio)
- Create: `tests/test_api.py`

**Step 1: Crear server.py minimal con FastAPI + mistralrs Runner + lifespan + /v1/chat/completions**

**Step 2: Probar con curl**

```bash
curl -s http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"gemma-4-e4b-heretic","messages":[{"role":"user","content":"Hola"}],"max_tokens":10}'
```

**Expected:** 200 OK con respuesta del modelo.

**Step 3: Commit**

```bash
git add server.py tests/test_api.py
git commit -m "feat: FastAPI server with mistral.rs runner"
```

---

## Task 4: Streaming SSE

**Objective:** Implementar Server-Sent Events para streaming.

**Files:**
- Modify: `server.py`

**Step 1: Agregar endpoint streaming con EventSourceResponse**

```python
async def _stream(req):
    for chunk in runner.send_chat_completion_request_stream(request):
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
```

**Step 2: Probar con curl -N**

**Step 3: Commit**

```bash
git commit -am "feat: SSE streaming support"
```

---

## Task 5: Vision - Procesamiento de imagenes

**Objective:** Habilitar inferencia con imagenes sin OOM.

**Files:**
- Create: `image.py` (preprocessing: resize a 896px)
- Modify: `server.py` (soporte multimodal en messages)
- Create: `tests/test_vision.py`

**Step 1: Crear image.py con preprocess_image()**

**Step 2: Formato multimodal para mistral.rs:**
```python
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "Describe esta imagen"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}
]
```

**Step 3: Probar con curl + base64 image**

**Step 4: Verificar VRAM < 8 GB con imagen**

**Step 5: Commit**

```bash
git add image.py server.py tests/test_vision.py
git commit -m "feat: vision inference with image preprocessing"
```

---

## Task 6: Audio

**Objective:** Probar si audio funciona con Gemma 4 E4B + mistral.rs.

**Riesgo:** Gemma 4 E4B puede no tener encoder de audio (solo 12B/27B).

**Step 1: Verificar soporte con WAV simple**

**Step 2: Si funciona -> integrar. Si no -> documentar como Future.**

**Step 3: Commit**

```bash
git commit -m "feat: audio inference (if supported)"
```

---

## Task 7: RAG Integration

**Objective:** Migrar modulo RAG (FAISS) desde Phase 1.

**Files:**
- Copy: `rag.py` (desde master)
- Modify: `server.py` (integrar RAG)

**Step 1: Copiar rag.py desde master: `git checkout master -- rag.py`**

**Step 2: Integrar en server.py - aplicar RAG antes de pasar a runner**

**Step 3: Probar con 10 mensajes (overflow + retrieval)**

**Step 4: Commit**

```bash
git add rag.py server.py
git commit -m "feat: RAG integration with FAISS"
```

---

## Task 8: Auth + Rate Limit + Logging

**Objective:** Migrar seguridad y monitoreo desde Phase 1.

**Files:**
- Modify: `server.py`
- Copy: SQLite schema (desde master)

**Step 1:** Copiar logica de auth, rate limiting, request logging desde master/server.py.

**Step 2:** Adaptar a nueva estructura (misma interfaz, diferente runner).

**Step 3: Probar todos los endpoints de auth.

**Step 4: Commit**

```bash
git commit -m "feat: auth, rate limiting, request logging"
```

---

## Task 9: systemd Service

**Objective:** Crear/actualizar servicio systemd.

**Files:**
- Modify: `gemma4-api.service`

**Step 1: Actualizar ExecStart para usar nuevo server.py**

**Step 2: Reload + restart**

```bash
systemctl --user daemon-reload
systemctl --user restart gemma4-api.service
```

**Step 3: Commit**

```bash
git commit -am "chore: update systemd service for phase2"
```

---

## Task 10: Benchmark + Validation

**Objective:** Medir metricas y comparar con Phase 1.

**Files:**
- Create: `tests/benchmark.py`

**Step 1: Medir VRAM en cada modo (idle, text, vision, rag)**

**Step 2: Comparar contra targets**

| Metrica | Phase 1 | Phase 2 Target |
|---|---|---|
| VRAM idle | 9.3 GB | <= 3.0 GB |
| VRAM text | ~10.0 GB | <= 4.0 GB |
| VRAM vision | OOM | <= 6.0 GB |
| Latencia text (128 tok) | ~2.5s | <= 3.0s |

**Step 3: Commit**

```bash
git add tests/benchmark.py
git commit -m "test: benchmark suite phase2"
```

---

## Task 11: Cleanup + Release

**Objective:** Limpiar, documentar, y taggear release.

**Step 1:** Actualizar README.md con instrucciones Phase 2.

**Step 2:** Verificar todos los tests pasan.

**Step 3:** Tag release.

```bash
git tag -a v2.0.0 -m "Phase 2: mistral.rs multimodal inference"
git push origin phase2 --tags
```

---

## Timeline

| Semana | Tasks |
|---|---|
| 1 | T1-T4: Setup, PoC, API, Streaming |
| 2 | T5-T8: Vision, Audio, RAG, Auth |
| 3 | T9-T11: Service, Benchmark, Release |

## Open Questions (resolver en T2)

1. mistralrs PyPI wheel incluye CUDA o solo CPU? Si CPU, necesitamos compilar desde fuente.
2. `in_situ_quant="4"` es compatible con modelo abliterated (heretic)?
3. Audio funciona en E4B o solo en 12B/27B?
4. Tool calling de Gemma 4 funciona via mistral.rs?
