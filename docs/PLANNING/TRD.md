# TRD — lowram-gemma4-vision Phase 2: Diseño Técnico

**Version:** 2.1.0  
**Fecha:** 2026-06-05  
**Autor:** Hans-Dieter Buddenberg Blamey  
**Status:** Approved  
**Branch:** phase2  
**Arquitectura:** mistral.rs (Rust puro) + nginx reverse proxy

---

## 1. Arquitectura del Sistema

### 1.1 Visión General

Arquitectura completamente nueva, sin Python. Un binario Rust (`mistralrs`) maneja inferencia multimodal completa. nginx provee autenticación, rate limiting y SSL como reverse proxy.

```
                     INTERNET / LAN
                          │
                    ┌─────▼──────┐
                    │   nginx     │  :443 (SSL) / :80
                    │   proxy     │
                    │ ┌─────────┐ │
                    │ │auth key │ │  Validación API key
                    │ │rate lim │ │  limit_req_zone
                    │ │SSL/TLS  │ │  Certbot opcional
                    │ └─────────┘ │
                    └─────┬──────┘
                          │ proxy_pass http://127.0.0.1:8080
                          │
                    ┌─────▼──────────────────────────┐
                    │  mistralrs serve                │  :8080 (localhost only)
                    │  (binario Rust, HTTP nativo)     │
                    │                                  │
                    │  Endpoints OpenAI-compatible:    │
                    │  POST /v1/chat/completions       │
                    │  GET  /v1/models                 │
                    │  GET  /health                    │
                    │                                  │
                    │  Interno:                        │
                    │  ├── Tokenizer (HF tokenizers)   │
                    │  ├── ISQ Q4K (in-situ quant)     │
                    │  ├── Gemma 4 LLM (4-bit)         │
                    │  ├── Gemma4 Vision Encoder        │
                    │  ├── Audio Encoder (native)       │
                    │  ├── KV Cache Manager             │
                    │  └── Sampling (temperature, etc.)  │
                    └─────┬──────────────────────────┘
                          │ CUDA 13.3 + cuDNN
                    ┌─────▼──────────────────────────┐
                    │  NVIDIA RTX 3060 12 GB           │
                    │  compute_cap 8.6                 │
                    │  Driver 610.43.02                │
                    └─────────────────────────────────┘

     CPU (paralelo, post-migración):
     ┌─────────────────────────────────────┐
     │  RAG Service (separado)              │  :8081
     │  ├── FAISS-CPU (vector search)       │
     │  ├── Embeddings (ONNX, ~130 MB)      │
     │  └── SQLite (metadatos)              │
     └─────────────────────────────────────┘
```

### 1.2 Flujo de Request

```
Cliente (curl / Discord bot / app)
  │
  │ HTTPS POST /v1/chat/completions
  │ Headers: Authorization: Bearer ***
  │ Body: { model, messages, max_tokens, temperature, stream }
  │
  ▼
nginx (:443)
  │
  ├── 1. Auth: validar API key contra archivo/SQLite
  │     └── 401 si inválida
  ├── 2. Rate limit: limit_req zone por IP/key
  │     └── 429 si excedido
  ├── 3. Proxy pass: http://127.0.0.1:8080/v1/chat/completions
  │
  ▼
mistralrs serve (:8080)
  │
  ├── 4. Parsear request JSON (OpenAI format)
  ├── 5. Tokenizar messages (texto + imagen + audio)
  ├── 6. ISQ Q4K: pesos ya cuantizados en memoria
  ├── 7. Forward pass:
  │     ├── Text tokens → embed_tokens_per_layer + LLM decoder (4-bit)
  │     ├── Image URL → Gemma4 Vision encoder (fp16) → projector
  │     ├── Audio URL → Audio encoder (fp16) → projector
  │     └── Concatenar → LLM decoder → KV cache
  ├── 8. Sampling (temperature, top_p, top_k)
  ├── 9. Response:
  │     ├── stream=false: JSON completo
  │     └── stream=true: SSE chunks
  │
  ▼
nginx → Cliente
  ├── JSON response (200 OK)
  └── o SSE stream (200 OK, text/event-stream)
```

---

## 2. Componentes y Responsabilidades

### 2.1 mistral.rs (Motor de Inferencia)

| Subcomponente | Responsabilidad | Tecnología |
|---|---|---|
| HTTP Server | Endpoints OpenAI-compatible | Actix-web (built-in) |
| Tokenizer | Encode/decode texto | HF tokenizers crate |
| ISQ Engine | Cuantización in-situ Q4K | candle (Rust ML) |
| LLM Backbone | Forward pass 42 capas texto | CUDA kernels |
| **embed_tokens_per_layer** | **Embedding per-capa [262144, 10752]** | **CUDA, 5.64 GB fp16** |
| Embedding/LM Head (shared) | Token embedding + output proj (1 tensor) | CUDA, 1.34 GB fp16 |
| Vision Encoder | Gemma4 Vision, 16 capas dim 768 | CUDA + flash-attn |
| Audio Encoder | 12 capas dim 1024 | CUDA + FFmpeg decode |
| Projectors | Vision/Audio → espacio LLM | Cross-attention CUDA, 67 MB total |
| KV Cache | Atención caching | CUDA managed memory |
| Sampling | Temperature, top_p, top_k | CPU |
| Model Loader | Carga safetensors desde disco | memmap2 + safetensors crate |

**Binario:** `~/.cargo/bin/mistralrs`  
**Fuente:** `cargo install mistralrs-cli --features "cuda flash-attn cudnn"`

### 2.2 nginx (Reverse Proxy)

| Subcomponente | Responsabilidad |
|---|---|
| SSL Termination | HTTPS → HTTP (certbot opcional) |
| Authentication | Validar API key (auth_request o mapas) |
| Rate Limiting | limit_req_zone por IP/key |
| Proxy Pass | Forward a mistralrs en localhost:8080 |
| Logging | access_log + error_log |
| Connection Timeout | Protección contra requests colgados |

### 2.3 systemd (Service Management)

| Aspecto | Detalle |
|---|---|
| Tipo | `Type=simple` |
| Nombre | `gemma4-rs.service` |
| Usuario | `hbuddenberg` (user service) |
| Restart | `on-failure` con `RestartSec=10` |
| Environment | `CUDA_VISIBLE_DEVICES=0`, `RUST_LOG=info` |
| Pre-start | Esperar GPU disponible |
| Post-stop | Limpiar VRAM |

---

## 3. ISQ Q4K: Cuantización In-Situ

### 3.1 Cómo Funciona

ISQ (In-Situ Quantization) es el mecanismo de mistral.rs para cuantizar modelos **al cargarlos en memoria**, sin necesidad de pre-conversión a GGUF.

**Proceso:**
1. mistral.rs lee el archivo safetensors directamente desde disco
2. Carga los pesos en fp16 a memoria GPU
3. Aplica cuantización Q4K (4-bit con bloques K) a las capas lineales
4. Descarta los pesos fp16 originales
5. Mantiene solo los pesos cuantizados (4-bit) en VRAM

**Formato Q4K:**
- Bloques de 256 pesos cuantizados a 4 bits
- Escalas y zeros cuantizados por bloque
- ~4.5 bits/parámetro efectivo (incluyendo overhead de escalas)
- Compatible con CUDA y CPU

**⚠️ Componente crítico `embed_tokens_per_layer`:**
- Tensor [262144, 10752] = 5.64 GB en fp16
- Si ISQ lo cuantiza: ~1.41 GB (Q4K) → cabe cómodamente
- Si ISQ NO lo cuantiza: 5.64 GB fp16 → consume casi todo el VRAM libre
- **Verificar comportamiento en Fase 2** (carga del modelo)
- Este tensor representa el **35% del modelo** y es determinante para la viabilidad

### 3.2 Por Qué No GGUF

| Aspecto | ISQ (elegido) | GGUF (descartado) |
|---|---|---|
| Conversión previa | No necesaria | Requiere proceso de conversión |
| Archivos extra | No | ~7.5 GB archivo GGUF |
| Multimodal | Vision + Audio nativo | Solo texto para Gemma 4 |
| Modelo heretic | Mismo safetensors | Puede perder metadatos |
| Complejidad | Un comando | Dos pasos (convertir + cargar) |

### 3.3 Uso en CLI

```bash
# Ejecución directa (chat interactivo)
mistralrs run --quant q4k -m ~/models/gemma4-heretic

# Servidor HTTP (producción)
mistralrs serve --quant q4k -p 8080 -m ~/models/gemma4-heretic

# Con TOML config (avanzado)
mistralrs --config-file config.toml
```

El flag `--quant q4k` activa ISQ Q4K automáticamente al cargar.

---

## 4. Presupuesto VRAM Detallado

> **Datos reales del safetensors** — cada tensor verificado individualmente.

### 4.1 Desglose por Componente (DATOS REALES)

| Componente | Dimensiones | Precision | VRAM | Notas |
|---|---|---|---|---|
| **Text layers** (42 capas) | ~3.8B params | Q4K (4-bit) | ~2.0 GB | Capas de texto principales |
| **embed_tokens_per_layer** | [262144, 10752] | Q4K o fp16 | ~1.41 GB / **5.64 GB** | ⚠️ **CRÍTICO** — 35% del modelo |
| **Embedding/LM Head (shared)** | [262144, 2560] | fp16 | ~1.34 GB | tie_word_embeddings=true, 1 tensor |
| Per-layer projections | ~110 MB fp16 | Q4K | ~0.03 GB | Muy pequeñas cuantizadas |
| **Vision Encoder** (Gemma4, 16 capas) | ~400M params | fp16 | ~0.34 GB | Se carga al procesar imagen |
| **Audio Encoder** (12 capas) | ~300M params | fp16 | ~0.61 GB | Se carga al procesar audio |
| **Projectors** (visión + audio) | ~67 MB | fp16 | ~0.07 GB | 67 MB reales del safetensors |
| **KV Cache** (4096 tokens, GQA 4:1) | - | fp16 | ~0.19 GB | Con KV sharing |
| **CUDA Runtime + Overhead** | - | - | ~0.3 GB | Rust FFI, muy bajo |

### 4.2 Escenarios de VRAM

**Escenario A: ISQ cuantiza embed_tokens_per_layer (objetivo) ✅**

| Modo | VRAM | Headroom |
|---|---|---|
| Texto solo (idle) | ~5.3 GB | ~6.7 GB |
| Texto + visión | ~5.6 GB | ~6.4 GB |
| Texto + audio | ~5.9 GB | ~6.1 GB |
| Texto + visión + audio | ~6.2 GB | ~5.8 GB |
| Máximo (con KV cache lleno) | ~6.4 GB | ~5.6 GB |

**Escenario B: ISQ NO cuantiza embed_tokens_per_layer (riesgo) 🔴**

| Modo | VRAM | Headroom |
|---|---|---|
| Texto solo (idle) | ~9.5 GB | ~2.5 GB |
| Texto + visión | ~9.8 GB | ~2.2 GB |
| Texto + audio | ~10.1 GB | ~1.9 GB |
| Texto + visión + audio | ~10.8 GB | ~1.2 GB |
| Máximo (con KV cache lleno) | ~11.0 GB | ~1.0 GB |

### 4.3 Estrategia de Gestión de VRAM

```
VRAM Layout (12 GB total) — Escenario A (ISQ cuantiza embed_tokens_per_layer):

┌─────────────────────────────────────────┐ 0 GB
│ Text layers Q4K (42 capas)              │ ~2.0 GB
├─────────────────────────────────────────┤ 2.0 GB
│ embed_tokens_per_layer Q4K              │ ~1.41 GB  ⚠️ CRÍTICO
├─────────────────────────────────────────┤ 3.41 GB
│ Embedding/LM Head (shared, fp16)        │ ~1.34 GB
│ Per-layer projections Q4K               │ ~0.03 GB
├─────────────────────────────────────────┤ 4.78 GB
│ CUDA Runtime Overhead                    │ ~0.3 GB
├─────────────────────────────────────────┤ 5.08 GB
│ KV Cache (reservado, 4K tokens)          │ ~0.19 GB
├─────────────────────────────────────────┤ 5.27 GB
│ [DISPONIBLE] Vision Encoder              │ ~0.34 GB  ← on-demand
│ [DISPONIBLE] Audio Encoder               │ ~0.61 GB  ← on-demand
│ [DISPONIBLE] Projectors                  │ ~0.07 GB  ← on-demand
│ [HEADROOM]                               │ ~5.5 GB
└─────────────────────────────────────────┘ 12 GB
```

**Nota:** Los encoders de visión y audio se activan bajo demanda cuando el request los necesita. En modo texto-only, esa VRAM queda libre.

---

## 5. Configuración mistral.rs

### 5.1 TOML de Configuración (config.toml)

```toml
command = "serve"

[server]
host = "127.0.0.1"        # Solo localhost (nginx hace proxy)
port = 8080

[[models]]
kind = "multimodal"
model_id = "/home/hbuddenberg/models/gemma4-heretic"
# No se especifica arch: mistral.rs lo detecta de config.json

[models.quantization]
in_situ_quant = "q4k"     # ISQ Q4K: cuantiza al cargar
```

### 5.2 Línea de Comandos Alternativa

```bash
# Sin TOML, directo en CLI:
mistralrs serve \
  --quant q4k \
  --port 8080 \
  -m ~/models/gemma4-heretic
```

### 5.3 Variables de Entorno

```bash
export CUDA_VISIBLE_DEVICES=0
export RUST_LOG=info             # Logging: error, warn, info, debug, trace
export RUST_BACKTRACE=1          # Stack traces en errores
```

---

## 6. Configuración nginx

### 6.1 nginx.conf (Arch Linux)

```nginx
# /etc/nginx/conf.d/gemma4-rs.conf
# En Arch Linux, los configs van en conf.d/ (no sites-available/ como Debian)

# Rate limiting zone: 10 requests/segundo por IP
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

upstream mistralrs_backend {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;
    # listen 443 ssl http2;  # Descomentar SOLO cuando SSL esté configurado con certbot
    server_name _;

    # SSL (opcional, habilitar con certbot)
    # ssl_certificate /etc/letsencrypt/live/DOMAIN/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/DOMAIN/privkey.pem;

    # Auth: API key via header
    # Método simple con map para uso personal (1-5 usuarios)
    set $api_key "";
    if ($http_authorization ~ "^Bearer (.+)$") {
        set $api_key $1;
    }

    # Validar contra archivo de keys
    # Para uso personal: validación directa en nginx
    location /v1/ {
        # Rate limiting
        limit_req zone=api_limit burst=20 nodelay;

        # Max body size para imágenes base64 (hasta 50 MB)
        client_max_body_size 50m;

        # Auth básico para uso personal
        auth_basic "Gemma 4 API";
        auth_basic_user_file /etc/nginx/.htpasswd;

        # Proxy a mistral.rs
        proxy_pass http://mistralrs_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        # SSE streaming support
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
    }

    # Health check (sin auth)
    location /health {
        proxy_pass http://mistralrs_backend;
        proxy_http_version 1.1;
    }

    # Bloquear todo lo demás
    location / {
        return 403;
    }

    # Logs
    access_log /var/log/nginx/gemma4-access.log;
    error_log /var/log/nginx/gemma4-error.log;
}
```

### 6.2 Setup Auth Básico

```bash
# Crear archivo de passwords (uso personal, 1-5 usuarios)
echo "PASSWORD" | sudo -S htpasswd -c /etc/nginx/.htpasswd hbuddenberg

# Testear configuración
echo "PASSWORD" | sudo -S nginx -t

# Recargar
echo "PASSWORD" | sudo -S systemctl reload nginx
```

---

## 7. systemd Service

### 7.1 Service File

```ini
# ~/.config/systemd/user/gemma4-rs.service

[Unit]
Description=Gemma 4 API - mistral.rs inference server
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
Environment=CUDA_VISIBLE_DEVICES=0
Environment=RUST_LOG=info
Environment=RUST_BACKTRACE=1
Environment=HOME=/home/hbuddenberg

ExecStart=/home/hbuddenberg/.cargo/bin/mistralrs serve --quant q4k --port 8080 -m /home/hbuddenberg/models/gemma4-heretic

Restart=on-failure
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=30

# Recursos
LimitNOFILE=65536

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=gemma4-rs

[Install]
WantedBy=default.target
```

### 7.2 Comandos de Gestión

```bash
# Instalar service
systemctl --user daemon-reload
systemctl --user enable gemma4-rs.service

# Iniciar
systemctl --user start gemma4-rs.service

# Ver estado
systemctl --user status gemma4-rs.service

# Ver logs
journalctl --user -u gemma4-rs.service -f

# Parar
systemctl --user stop gemma4-rs.service

# Reiniciar
systemctl --user restart gemma4-rs.service
```

---

## 8. Endpoints API

### 8.1 POST /v1/chat/completions

**Request:**

```json
{
  "model": "gemma4-heretic",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe esta imagen"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
      ]
    }
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

**Tipos de contenido soportados en messages:**

| Tipo | Formato | Ejemplo URL |
|---|---|---|
| Texto | `{"type": "text", "text": "..."}` | N/A |
| Imagen (local) | `{"type": "image_url", ...}` | `file:///path/to/image.jpg` |
| Imagen (base64) | `{"type": "image_url", ...}` | `data:image/jpeg;base64,...` |
| Imagen (URL) | `{"type": "image_url", ...}` | `https://example.com/img.jpg` |
| Audio (local) | `{"type": "audio_url", ...}` | `file:///path/to/audio.wav` |
| Video (local) | `{"type": "video_url", ...}` | `file:///path/to/video.mp4` |

**Response (non-streaming):**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "gemma4-heretic",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 50, "completion_tokens": 128, "total_tokens": 178}
}
```

**Response (streaming):**

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hola"}}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":" mundo"}}]}

data: [DONE]
```

### 8.2 GET /v1/models

```json
{
  "object": "list",
  "data": [{"id": "gemma4-heretic", "object": "model", "owned_by": "local"}]
}
```

### 8.3 GET /health

```json
{"status": "ok"}
```

---

## 9. Manejo de Multimodalidad

### 9.1 Visión (Imágenes)

**Flujo interno en mistral.rs:**
1. Recibir `image_url` en el mensaje
2. Decodificar imagen (JPEG/PNG/WebP)
3. Resize a dimensiones compatibles (≤896px, manteniendo aspect ratio)
4. Pasar por Gemma4 Vision encoder (16 capas, dim 768, fp16)
5. Vision projector: adapter al espacio del LLM (dim 2560)
6. Concatenar tokens de imagen con tokens de texto
7. Forward pass por el LLM backbone

**VRAM bajo demanda:** El vision encoder solo ocupa VRAM cuando hay una imagen en el request. En modo texto-only, esos ~0.34 GB quedan libres.

### 9.2 Audio

**Flujo interno en mistral.rs:**
1. Recibir `audio_url` en el mensaje
2. FFmpeg decodifica a WAV raw (16kHz mono)
3. Audio encoder (12 capas, dim 1024, fp16) procesa espectrograma
4. Audio projector: adapter al espacio del LLM
5. Concatenar tokens de audio con tokens de texto
6. Forward pass por el LLM backbone

**Formatos soportados:** WAV, MP3, FLAC, OGG (via FFmpeg).

### 9.3 Consideraciones de VRAM Multimodal

```
Escenario A (ISQ cuantiza embed_tokens_per_layer):
  Modo texto-only:           ~5.3 GB VRAM
  Modo texto+visión:         ~5.6 GB VRAM  (+0.34 GB)
  Modo texto+audio:          ~5.9 GB VRAM  (+0.61 GB)
  Modo texto+visión+audio:   ~6.2 GB VRAM  (+0.95 GB)
  Máximo (con KV cache):     ~6.4 GB VRAM
  → Todos los modos caben holgadamente con 5.6+ GB headroom ✅

Escenario B (embed_tokens_per_layer SIN cuantizar):
  Modo texto-only:           ~9.5 GB VRAM
  Modo texto+visión+audio:   ~10.8 GB VRAM
  → Riesgo OOM alto, margen mínimo 🔴
```

---

## 10. RAG: Estrategia Post-Migración

### 10.1 Estado

El RAG (FAISS + sentence-transformers) funciona en CPU y no requiere cambios inmediatos. Se migra como servicio separado después de que el inference engine esté operativo.

### 10.2 Opciones de Implementación

**Opción A: Servicio RAG separado (recomendado)**
```
Cliente → nginx → mistralrs :8080 (inference)
                  ↕
         nginx → rag-service :8081 (embeddings + search)
```
- Servicio ligero (Python o Rust) con FAISS-CPU
- Embeddings via ONNX runtime (~130 MB RAM) o pre-computados
- Cliente consulta RAG antes de enviar a inference

**Opción B: Pre-computed embeddings**
- Generar embeddings de todo el historial offline
- Almacenar en SQLite + FAISS index
- Búsqueda directa sin modelo de embeddings en runtime
- Menor footprint de RAM

### 10.3 Plan

1. Fase 1-4: Inference engine funcional sin RAG
2. Fase 5: Implementar RAG como servicio separado
3. Integrar via nginx routing o middleware del cliente

---

## 11. Seguridad

| Aspecto | Implementación | Capa |
|---|---|---|
| Autenticación | HTTP Basic Auth (nginx) o API key header | nginx |
| Rate Limiting | `limit_req_zone` 10 req/s por IP | nginx |
| Timeout | `proxy_read_timeout 300s` | nginx |
| Network isolation | mistralrs escucha solo en 127.0.0.1 | mistral.rs |
| Image size limit | nginx `client_max_body_size 50m` | nginx |
| SSL/TLS | Certbot + Let's Encrypt (opcional) | nginx |
| CORS | No configurado (solo localhost/LAN) | nginx |
| URL blocking | No aplica (archivos locales solo) | mistral.rs |

---

## 12. Monitoreo y Observabilidad

```bash
# VRAM usage en tiempo real
watch -n 1 nvidia-smi

# RAM del proceso mistralrs
ps aux | grep mistralrs

# Logs del servicio
journalctl --user -u gemma4-rs.service -f

# Logs nginx
tail -f /var/log/nginx/gemma4-access.log

# Health check curl
curl -s http://localhost:8080/health | jq .

# Estadísticas de VRAM durante inferencia
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

---

## 13. Estructura de Archivos Final

```
lowram-gemma4-vision/              (branch: phase2)
├── docs/
│   └── PLANNING/
│       ├── PRD.md                 # Este documento
│       ├── TRD.md                 # Este documento
│       ├── IMPLEMENTATION.md      # Plan de implementación
│       ├── FINAL_DECISION.md      # Decisión multi-agente
│       └── CONSENSUS.md           # Consenso validación
├── config.toml                    # Config mistral.rs (ISQ Q4K, puerto, modelo)
├── nginx/
│   └── gemma4-rs.conf             # Config nginx reverse proxy (para conf.d/)
├── systemd/
│   └── gemma4-rs.service          # systemd user service
├── scripts/
│   ├── install.sh                 # Setup completo (deps + compile + config)
│   ├── benchmark.sh               # Benchmark VRAM/RAM/latencia
│   └── test_vision.sh             # Test visión con imagen de prueba
├── tests/
│   ├── test_text.sh               # curl test texto
│   ├── test_vision.sh             # curl test imagen base64
│   ├── test_audio.sh              # curl test audio WAV
│   └── test_streaming.sh          # curl test SSE
├── rag/                           # [POST-MIGRACIÓN]
│   ├── embeddings/                # ONNX model o pre-computados
│   └── faiss_index/               # Índice FAISS
└── README.md                      # Instrucciones de deployment
```
