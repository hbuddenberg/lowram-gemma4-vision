# SPEC — lowram-gemma4-vision Phase 2: Especificación Técnica

**Version:** 1.1.0  
**Fecha:** 2026-06-05  
**Autor:** Hans-Dieter Buddenberg Blamey  
**Status:** Approved (Cycle 2 — OC ✅ AGY ✅)  
**Branch:** phase2  

---

## 1. API Specification

### 1.1 Visión General

mistral.rs expone un servidor HTTP OpenAI-compatible en `127.0.0.1:8080`. nginx actúa como reverse proxy en `0.0.0.0:80` (o `443` con SSL) con autenticación y rate limiting.

**Model ID:** `gemma4-heretic`  
**Base URL (directo):** `http://127.0.0.1:8080/v1`  
**Base URL (vía nginx):** `http://0.0.0.0:80/v1` (o `https://HOST:443/v1`)

---

### 1.2 POST /v1/chat/completions

**Descripción:** Genera una respuesta de chat (texto, vision o audio).

#### Request Headers

| Header | Requerido | Valor |
|--------|-----------|-------|
| `Content-Type` | Sí | `application/json` |
| `Authorization` | Sí (vía nginx) | `Basic <base64(user:pass)>` o `Bearer <api_key>` |
| `Accept` | No | `text/event-stream` (para streaming) |

> ⚠️ **Nota:** La autenticación la maneja **nginx**, no mistral.rs. mistral.rs escucha en `127.0.0.1:8080` sin auth. Un 401 significa que nginx rechazó la request (ver §3.1 — Errores de nginx vs mistral.rs).

#### Request Body

```json
{
  "model": string,              // Requerido. SIEMPRE "gemma4-heretic"
  "messages": [Message[]],      // Requerido. Array de mensajes (min 1, max 100)
  "max_tokens": integer,       // Opcional. Default: 512. Range: [1, 8192]
  "temperature": number,       // Opcional. Default: 0.7. Range: [0.0, 2.0]
  "top_p": number,             // Opcional. Default: 1.0. Range: [0.0, 1.0]
  "top_k": integer,           // Opcional. Default: 32. Range: [1, 256]
  "stream": boolean,          // Opcional. Default: false
  "stop": string | string[],  // Opcional. Sequencias de parada
  "frequency_penalty": number, // Opcional. Default: 0.0. Range: [-2.0, 2.0]
  "presence_penalty": number,  // Opcional. Default: 0.0. Range: [-2.0, 2.0]
  "seed": integer             // Opcional. Para reproducibilidad
}
```

#### Mensajes (Message object)

```json
{
  "role": "system" | "user" | "assistant",
  "content": string | ContentPart[]
}
```

**Si `content` es string:** mensaje de texto plano.

**Si `content` es array (multimodal):** cada elemento es un `ContentPart`:

```json
// Texto
{
  "type": "text",
  "text": string  // Texto del mensaje
}

// Imagen (base64)
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/jpeg;base64,<base64_data>"
  }
}

// Imagen (archivo local)
{
  "type": "image_url",
  "image_url": {
    "url": "file:///path/to/image.jpg"
  }
}

// Imagen (URL remota)
{
  "type": "image_url",
  "image_url": {
    "url": "https://example.com/image.jpg"
  }
}

// Audio (archivo local)
{
  "type": "audio_url",
  "image_url": {
    "url": "file:///path/to/audio.wav"
  }
}

// Video (archivo local)
{
  "type": "video_url",
  "image_url": {
    "url": "file:///path/to/video.mp4"
  }
}
```

> ⚠️ **Nota sobre ContentPart en mistral.rs:** El campo para audio/video usa `image_url` como key en la implementación actual de mistral.rs (herencia de la estructura OpenAI). Los tipos `audio_url` y `video_url` pueden cambiar en futuras versiones. Verificar con la documentación de mistral.rs al momento del deploy.

#### Response (stream=false) — 200 OK

```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": integer,           // Unix timestamp
  "model": "gemma4-heretic",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": string
      },
      "finish_reason": "stop" | "length" | "error"
    }
  ],
  "usage": {
    "prompt_tokens": integer,
    "completion_tokens": integer,
    "total_tokens": integer
  }
}
```

#### Response (stream=true) — 200 OK

Content-Type: `text/event-stream`

```
data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":<ts>,"model":"gemma4-heretic","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":<ts>,"model":"gemma4-heretic","choices":[{"index":0,"delta":{"content":"Hola"},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":<ts>,"model":"gemma4-heretic","choices":[{"index":0,"delta":{"content":" mundo"},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":<ts>,"model":"gemma4-heretic","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

#### Errores de mistral.rs

| Status | Tipo | Condiciones | Body |
|--------|------|-------------|------|
| 400 | `invalid_request_error` | JSON inválido, `model` vacío, `messages` vacío | `{"error": {"message": "...", "type": "invalid_request_error", "code": "..."}}` |
| 413 | `invalid_request_error` | Body > 50 MB (imagen base64 grande) | `{"error": {"message": "...", "type": "invalid_request_error", "code": "request_too_large"}}` |
| 500 | `server_error` | CUDA OOM durante inferencia | `{"error": {"message": "...", "type": "server_error", "code": "internal_error"}}` |

#### Errores de nginx (respuestas HTML por defecto)

| Status | Condiciones | Respuesta nginx |
|--------|-------------|-----------------|
| 401 | Sin header `Authorization` o credenciales inválidas | HTML `401 Authorization Required` |
| 429 | Rate limit excedido (>10 req/s) | HTML `429 Too Many Requests` |
| 413 | Body excede `client_max_body_size` | HTML `413 Request Entity Too Large` |

> ⚠️ **Nota:** nginx devuelve **HTML**, no JSON, para 401/429/413. Para obtener respuestas JSON, se puede agregar `error_page` con template JSON en nginx.conf, pero para uso personal el HTML nativo es suficiente.

---

### 1.3 GET /v1/models

**Descripción:** Lista modelos disponibles.

#### Response — 200 OK

```json
{
  "object": "list",
  "data": [
    {
      "id": "gemma4-heretic",
      "object": "model",
      "owned_by": "local",
      "created": integer,         // Unix timestamp de carga
      "permission": []
    }
  ]
}
```

#### Errores

| Status | Condiciones |
|--------|-------------|
| 401 | Sin credenciales |
| 503 | Modelo no cargado |

---

### 1.4 GET /health

**Descripción:** Health check del servidor. **Sin autenticación** (nginx lo pasa directo).

> ⚠️ **Nota:** mistral.rs `/health` devuelve **siempre 200** si el servidor está escuchando. **No verifica si el modelo está cargado.** Para verificar readiness del modelo, usar `GET /v1/models`.

#### Response — 200 OK

```json
{
  "status": "ok"
}
```

**Solo este estado.** No hay 503 ni otros códigos para este endpoint. Si el servidor no responde, significa que no está corriendo (verificar con `systemctl --user status gemma4-rs.service`).

---

### 1.5 OpenAI API Compatibility Matrix

| Feature | OpenAI API | mistral.rs | Compatibilidad |
|---------|-----------|------------|----------------|
| `POST /v1/chat/completions` | ✅ | ✅ | Completa |
| `GET /v1/models` | ✅ | ✅ | Completa |
| `GET /v1/models/{id}` | ✅ | ❌ | No implementado |
| `POST /v1/embeddings` | ✅ | ❌ | No aplicable (sin embedding endpoint) |
| `POST /v1/images/generations` | ✅ | ❌ | No aplicable |
| `POST /v1/audio/transcriptions` | ✅ | ❌ | No aplicable |
| `POST /v1/moderations` | ✅ | ❌ | No implementado |
| Streaming SSE | ✅ | ✅ | Completa |
| Function calling / tools | ✅ | ❌ | No soportado (Phase 3 potencial) |
| Logprobs | ✅ | ❌ | No soportado |
| `response_format: json_object` | ✅ | ❌ | No soportado |
| `seed` parameter | ✅ | ✅ | Parcial |
| `frequency_penalty` | ✅ | ✅ | Parcial (varía por modelo) |
| `presence_penalty` | ✅ | ✅ | Parcial (varía por modelo) |
| Vision (image_url) | ✅ (GPT-4V) | ✅ (Gemma4) | Completa (base64, file://, URL) |
| Audio input | ✅ (GPT-4o) | ✅ (Gemma4) | Completa (WAV, MP3, FLAC, OGG) |
| Multiple images per message | ✅ | ✅ | Completa |
| Multiple modalities per message | ✅ | ✅ | Completa (texto + imagen + audio) |

---

## 2. Config Specification

### 2.1 TOML Config (mistral.rs)

**Archivo:** `config.toml` en el directorio del proyecto.

```toml
# config.toml — mistral.rs configuration
# Para uso con: mistralrs --config-file config.toml

command = "serve"  # "serve" | "run"

[server]
host = "127.0.0.1"        # Solo localhost (nginx hace proxy público)
port = 8080                 # Puerto interno (1-65535)

[[models]]
kind = "multimodal"         # "text" | "multimodal" | "xlora" | "lora"
model_id = "/home/hbuddenberg/models/gemma4-heretic"
# tokenizer_id = ""         # Heredado de model_id por defecto
# arch = ""                 # Detectado de config.json automáticamente

[models.quantization]
# ISQ (In-Situ Quantization) — aplica al cargar
in_situ_quant = "4"          # "4" (Q4K) | "2" | "3" | "5" | "6" | "8" | "fp16" | "none"
# Nota: CLI acepta tanto "4" como "q4k", pero TOML usa formato numérico
```

> ⚠️ **Nota sobre Flash Attention:** Se habilita como **feature de compilación** Cargo, NO como opción TOML. Se compila con `--features "cuda flash-attn cudnn"`. No existe `[models.quantization.flash_attention]` en la configuración TOML.

#### Parámetros Clave

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `command` | string | `"run"` | Modo: `"serve"` para HTTP server, `"run"` para interactivo |
| `server.host` | string | `"127.0.0.1"` | Bind address. Siempre `127.0.0.1` (nginx expone) |
| `server.port` | integer | `8080` | Puerto. 8080 interno, nginx expone 80/443 |
| `models[].kind` | string | — | `"multimodal"` para Gemma 4 con visión+audio |
| `models[].model_id` | string | — | Ruta absoluta al directorio del modelo (contiene config.json + safetensors) |
| `models[].quantization.in_situ_quant` | string | `"none"` | Tipo ISQ: `"4"` para Q4K (4-bit con bloques K) |

> ⚠️ La cuantización ISQ se aplica en **tiempo de carga** del modelo, no en tiempo de compilación. Los encoders de visión y audio permanecen en fp16 — ISQ solo cuantiza los pesos del LLM backbone.

---

### 2.2 CLI Args (alternativa a TOML)

```bash
mistralrs serve \
  --port 8080 \
  --host 127.0.0.1 \
  --quant q4k \
  -m ~/models/gemma4-heretic
```

| Flag | Tipo | Default | Equivalente TOML |
|------|------|---------|-----------------|
| `--port` | int | `8080` | `server.port` |
| `--host` | string | `127.0.0.1` | `server.host` |
| `--quant` | string | `"none"` | `models[].quantization.in_situ_quant` |
| `-m, --model` | string | — | `models[].model_id` |
| `--config-file` | string | — | Archivo TOML (overrides flags) |

---

### 2.3 nginx.conf

**Archivo:** `/etc/nginx/conf.d/gemma4-rs.conf` (Arch Linux)

```nginx
# Rate limiting: 10 requests/segundo por IP
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

upstream mistralrs_backend {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;              # Cambiar a 0.0.0.0:80 para LAN access
    # listen 443 ssl http2;  # SSL opcional (descomentar con certbot)
    server_name _;          # Accept any hostname

    # SSL (opcional)
    # ssl_certificate /etc/letsencrypt/live/DOMAIN/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/DOMAIN/privkey.pem;

    # Para uso personal: HTTP Basic Auth
    # auth_basic "Gemma 4 API";
    # auth_basic_user_file /etc/nginx/.htpasswd;

    location /v1/ {
        # Rate limiting
        limit_req zone=api_limit burst=20 nodelay;

        # Max body size para imágenes base64
        client_max_body_size 50m;

        # Auth (descomentar una opción)
        # Opción A: HTTP Basic Auth
        # auth_basic "Gemma 4 API";
        # auth_basic_user_file /etc/nginx/.htpasswd;
        
        # Opción B: API key header
        # set $api_key "";
        # if ($http_authorization ~ "^Bearer (.+)$") {
        #     set $api_key $1;
        # }
        # if ($api_key != "YOUR_API_KEY") {
        #     return 401;
        # }

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

#### Directivas nginx Documentadas

| Directiva | Valor | Descripción |
|-----------|-------|-------------|
| `limit_req_zone` | `$binary_remote_addr zone=api_limit:10m rate=10r/s` | Rate limit: 10 req/s por IP, 10MB de estado |
| `limit_req` | `zone=api_limit burst=20 nodelay` | Burst de 20, sin delay |
| `client_max_body_size` | `50m` | Máximo 50 MB (imágenes base64 ~20-30 MB) |
| `proxy_connect_timeout` | `10s` | Timeout de conexión al backend |
| `proxy_read_timeout` | `300s` | Timeout de lectura (inferencia puede ser lenta) |
| `proxy_send_timeout` | `300s` | Timeout de envío al backend |
| `proxy_buffering` | `off` | Desactivado para SSE streaming |
| `keepalive` | `32` | Conexiones keepalive al backend |

---

## 3. Error Specification

### 3.1 Fuentes de Error

El sistema tiene 2 fuentes de error distintas con comportamientos diferentes:

| Fuente | Headers | Body Format | HTTP Codes |
|--------|---------|-------------|------------|
| **nginx** | (auth, rate-limit, body size) | **HTML** por defecto | 401, 413, 429 |
| **mistral.rs** | (validación, inference) | **JSON** OpenAI-compatible | 400, 500 |

> ⚠️ Esta distinción es importante: nginx no devuelve JSON por defecto. Un cliente que espere JSON para un 401/429/413 no lo recibirá, a menos que se agreguen `error_page` custom en nginx.conf.

### 3.2 Errores de mistral.rs (JSON)

| Code | Type | Condition | Resolution |
|------|------|-----------|------------|
| **400** | `invalid_request_error` | JSON parse error | Verificar formato JSON |
| **400** | `invalid_request_error` | `model` field missing/empty | Incluir `"model": "gemma4-heretic"` |
| **400** | `invalid_request_error` | `messages` empty o > 100 items | Min 1, max 100 mensajes |
| **400** | `invalid_request_error` | `max_tokens` fuera de [1, 8192] | Ajustar valor |
| **400** | `invalid_request_error` | `temperature` fuera de [0.0, 2.0] | Ajustar valor |
| **400** | `invalid_request_error` | Content type no soportado | Usar JPEG/PNG/WebP para imágenes |
| **500** | `server_error` | CUDA OOM durante inferencia | Reducir contexto o desactivar modalidades |
| **500** | `server_error` | Error en forward pass | Verificar journalctl |

### 3.3 Errores de nginx (HTML por defecto)

| Code | Condition | HTML Response |
|------|-----------|---------------|
| **401** | Sin header `Authorization` o credenciales inválidas | `401 Authorization Required` |
| **413** | Body excede `client_max_body_size` (50 MB) | `413 Request Entity Too Large` |
| **429** | Rate limit excedido (>10 req/s, burst >20) | `429 Too Many Requests` |

**Para devolver JSON en vez de HTML** (opcional para clientes que esperen JSON):

```nginx
# En server block de nginx
error_page 401 = @error401_json;
error_page 429 = @error429_json;

location @error401_json {
    default_type application/json;
    return 401 '{"error":{"message":"Unauthorized","type":"authentication_error","code":"invalid_api_key"}}';
}

location @error429_json {
    default_type application/json;
    return 429 '{"error":{"message":"Rate limit exceeded","type":"rate_limit_error","code":"rate_limit_exceeded"}}';
}
```

---

## 4. Content Type Specification

### 4.1 Imágenes

| Formato | Extensiones | MIME Types | Decodificador | Max Dimension | Max File Size |
|---------|-------------|------------|---------------|---------------|---------------|
| JPEG | `.jpg`, `.jpeg` | `image/jpeg` | Native | 896 × 896 px | 30 MB (base64 ~40 MB) |
| PNG | `.png` | `image/png` | Native | 896 × 896 px | 30 MB |
| WebP | `.webp` | `image/webp` | Native | 896 × 896 px | 30 MB |

**Resize:** Las imágenes se redimensionan automáticamente a ≤ 896px manteniendo aspect ratio (preservar ancho o alto, el otro se ajusta). No se aplica padding.

**Encoding en request:**

```json
// Base64 (recomendado para imágenes locales)
{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}

// Archivo local (mistral.rs lo lee del disco)
{"type": "image_url", "image_url": {"url": "file:///home/user/photo.jpg"}}

// URL remota (mistral.rs lo descarga)
{"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
```

**Conversión base64:**
```bash
base64 -w0 /path/to/image.jpg  # Sin line breaks (recomendado)
base64 /path/to/image.jpg       # Con line breaks de 76 chars (funciona también)
```

### 4.2 Audio

| Formato | Extensiones | MIME Types | Decodificador | Sample Rate | Max File Size |
|---------|-------------|------------|---------------|-------------|---------------|
| WAV | `.wav` | `audio/wav`, `audio/x-wav` | FFmpeg | 16 kHz (resample) | 20 MB |
| MP3 | `.mp3` | `audio/mpeg` | FFmpeg | 16 kHz (resample) | 20 MB |
| FLAC | `.flac` | `audio/flac` | FFmpeg | 16 kHz (resample) | 20 MB |
| OGG | `.ogg` | `audio/ogg` | FFmpeg | 16 kHz (resample) | 20 MB |

**Decodificación:** FFmpeg decodifica a WAV raw (16 kHz mono) antes de pasar al audio encoder. Si el audio es estéreo, se convierte a mono. Si el sample rate es diferente, se resamplea a 16 kHz.

**Encoding en request:**

```json
// Archivo local
{"type": "audio_url", "image_url": {"url": "file:///home/user/audio.wav"}}

// Base64 (si es pequeño)
{"type": "audio_url", "image_url": {"url": "data:audio/wav;base64,UklGRi..."}}
```

### 4.3 Video (limitado)

| Formato | Extensiones | Decodificador | Notas |
|---------|-------------|---------------|-------|
| MP4 | `.mp4` | FFmpeg | Se extraen frames como imágenes |
| WebM | `.webm` | FFmpeg | Soporte limitado |

> ⚠️ Video se procesa extrayendo frames (no procesa audio del video). Para audio + video en un solo request, enviar audio y video como content parts separados.

---

## 5. SSE Protocol Specification

### 5.1 Formato de Stream

Cada chunk es un Server-Sent Event con prefijo `data: `:

```
data: <json_payload>\n\n
```

### 5.2 Secuencia de Eventos

```
1. Role chunk (primer evento):
   data: {"id":"chatcmpl-xxx","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}

2. Content chunks (N eventos):
   data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"token"},"finish_reason":null}]}

3. Final chunk (último evento):
   data: {"id":"chatcmpl-xxx","choices":[{"delta":{},"finish_reason":"stop"}]}

4. Done signal:
   data: [DONE]
```

### 5.3 Propiedades de cada Chunk

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id` | string | ID de la completación (uuid, consistente entre chunks) |
| `object` | string | `"chat.completion.chunk"` |
| `created` | integer | Unix timestamp (consistente entre chunks) |
| `model` | string | `"gemma4-heretic"` |
| `choices[].index` | integer | Siempre `0` (single choice) |
| `choices[].delta` | object | Contenido incremental. Primer chunk: `{"role":"assistant"}`. Content chunks: `{"content":"text"}`. Último chunk: `{}` |
| `choices[].finish_reason` | string\|null | `null` durante contenido. `"stop"` en último chunk. `"length"` si max_tokens alcanzado. |

### 5.4 Configuración nginx para SSE

Las siguientes directivas son CRÍTICAS para streaming:

```nginx
proxy_buffering off;       # No acumular chunks en buffer
proxy_cache off;           # No cachear respuestas
chunked_transfer_encoding on;  # Transferencia chunked
```

Sin `proxy_buffering off`, nginx acumula toda la respuesta antes de enviarla, anulando el streaming.

### 5.5 Error en Streaming

Si ocurre un error durante el stream, se envía un chunk con `finish_reason: "error"`:

```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Error: CUDA OOM"},"finish_reason":"error"}]}

data: [DONE]
```

---

## 6. Security Specification

### 6.1 Capas de Seguridad

```
Internet/LAN
    │
    ▼
nginx (:80/:443)
    ├── Capa 1: Rate limiting (10 req/s, burst 20)
    ├── Capa 2: Autenticación (Basic Auth o API key header)
    ├── Capa 3: Body size limit (50 MB)
    ├── Capa 4: Timeout (connect 10s, read 300s)
    └── Capa 5: SSL/TLS (opcional, certbot)
    │
    ▼
mistral.rs (:8080, localhost only)
    ├── Capa 6: Network isolation (solo 127.0.0.1)
    └── Capa 7: No expone endpoints sensibles
```

### 6.2 Autenticación

**Opción A: HTTP Basic Auth (recomendada para uso personal)**

```bash
# Crear archivo de passwords
echo "PASSWORD" | sudo -S htpasswd -c /etc/nginx/.htpasswd hbuddenberg
```

Cliente envía:
```
Authorization: Basic <base64(user:password)>
```

**Opción B: API Key Header**

```nginx
set $api_key "";
if ($http_authorization ~ "^Bearer (.+)$") {
    set $api_key $1;
}
if ($api_key != "YOUR_SECRET_KEY") {
    return 401;
}
```

Cliente envía:
```
Authorization: Bearer YOUR_SECRET_KEY
```

### 6.3 Rate Limiting

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `rate` | `10r/s` | 10 requests por segundo por IP |
| `burst` | `20` | Permite hasta 20 requests instantáneos |
| `nodelay` | — | Procesa burst inmediatamente, sin delay |
| `zone size` | `10m` | ~160K IPs en estado |

**Respuesta 429 de nginx (HTML nativo):**
```html
<html>
<head><title>429 Too Many Requests</title></head>
<body><center><h1>429 Too Many Requests</h1></center><hr><center>nginx</center></body>
</html>
```

> ⚠️ Para que el cliente reciba JSON en vez de HTML, agregar las directivas `error_page` del §3.3 en nginx.conf.

### 6.4 Variables de Entorno del Servicio

```bash
# ~/.config/systemd/user/gemma4-rs.service
Environment=CUDA_VISIBLE_DEVICES=0    # Solo GPU 0
Environment=RUST_LOG=info             # No exponer internals en logs
Environment=RUST_BACKTRACE=1          # Stack traces para debugging (cambiar a 0 en prod)
Environment=HOME=/home/hbuddenberg
```

---

## 7. Performance Specification

### 7.1 Latency Budgets

| Tipo de Request | Latency Target (P50) | Latency Max (P99) | Notas |
|----------------|---------------------|-------------------|-------|
| Texto (128 tokens output) | ≤ 30s | ≤ 60s | ~3-5 tok/s en RTX 3060 12GB, ISQ Q4K |
| Texto (512 tokens output) | ≤ 120s | ≤ 180s | Generación larga |
| Texto + Imagen | ≤ 40s | ≤ 60s | Overhead de vision encoder (~2s) |
| Texto + Audio | ≤ 40s | ≤ 60s | Overhead de audio encoder (~3s) |
| Texto + Imagen + Audio | ≤ 45s | ≤ 90s | Overhead combinado |
| Streaming (first token) | ≤ 10s | ≤ 20s | Time to first token |
| `/v1/models` | ≤ 100ms | ≤ 500ms | Listado estático |
| `/health` | ≤ 50ms | ≤ 200ms | Sin GPU |

> 🎯 **Target aspiracional:** En hardware con más compute (RTX 4070+), P50 de texto (128 tok) podría alcanzar ≤ 15s. Para RTX 3060 12GB via TB3, ≤ 30s es el budget realista.

### 7.2 Throughput Targets

| Métrica | Target | Notas |
|---------|--------|-------|
| Tokens/segundo (texto) | ≥ 3 tok/s | RTX 3060, compute_cap 8.6 |
| Concurrent requests | ≥ 2 | nginx keepalive + mistral.rs queue |
| Requests/segundo | ≥ 0.5 | Limitado por velocidad de inferencia |

### 7.3 VRAM Thresholds

| Estado | VRAM Target | Acción si excede |
|--------|-------------|-----------------|
| Idle (texto-only) | ≤ 5.5 GB (A) / ≤ 9.5 GB (B) | Verificar ISQ del embed tensor |
| Inferencia texto | ≤ 6.0 GB (A) / ≤ 10.0 GB (B) | Reducir contexto |
| Inferencia visión | ≤ 7.0 GB (A) / ≤ 11.0 GB (B) | Desactivar audio simultáneo |
| Inferencia multimodal | ≤ 7.0 GB (A) / N/A (B) | Escenario B no soporta multimodal seguro |

### 7.4 RAM Thresholds

| Componente | RAM Target |
|-----------|------------|
| Proceso mistralrs | ≤ 200 MB |
| nginx worker | ≤ 20 MB |
| CUDA userspace | ≤ 30 MB |
| **Total inference stack** | **≤ 250 MB** |
| **RAM libre sistema** | **≥ 1.5 GB** (vs 380 MB en Phase 1) |

---

## 8. Integration Specification

### 8.1 Hermes Gateway Config

```yaml
# ~/.hermes/config.yaml — custom provider para Gemma 4 Local
custom_providers:
  - name: Gemma4-Local
    base_url: http://localhost/v1   # Via nginx (con auth)
    # base_url: http://127.0.0.1:8080/v1  # Directo (sin auth)
    api_key: gemma4-local
    models:
      - gemma4-heretic
```

> **Nota:** Cuando se usa vía nginx, Hermes necesita enviar las credenciales de auth en el header `Authorization`. Verificar que el provider de Hermes soporte HTTP Basic Auth o Bearer token.

### 8.2 curl Test Suite

```bash
# --- VARIABLES ---
API_URL="http://localhost"  # Via nginx (con auth)
# API_URL="http://127.0.0.1:8080"  # Directo
AUTH="-u hbuddenberg:PASSWORD"
MODEL="gemma4-heretic"

# 1. Health check (sin auth)
curl -s $API_URL/health

# 2. List models
curl -s $AUTH $API_URL/v1/models | jq .

# 3. Text completion
curl -s $AUTH $API_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Di OK\"}],\"max_tokens\":10}" | jq .

# 4. Vision (base64)
IMG_B64=$(base64 -w0 /tmp/test_img.jpg)
curl -s $AUTH $API_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"¿De qué color es?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$IMG_B64\"}}]}],\"max_tokens\":50}" | jq .

# 5. Streaming
curl -N $AUTH $API_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Cuenta del 1 al 5\"}],\"max_tokens\":50,\"stream\":true}"

# 6. Audio
AUDIO_B64=$(base64 -w0 /tmp/test_audio.wav)
curl -s $AUTH $API_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"¿Qué escuchas?\"},{\"type\":\"audio_url\",\"image_url\":{\"url\":\"data:audio/wav;base64,$AUDIO_B64\"}}]}],\"max_tokens\":50}" | jq .
```

### 8.3 WebSocket / Long Polling

**No soportado.** mistral.rs no expone endpoints WebSocket. Usar SSE (stream=true) para respuestas progresivas.

---

## 9. File Structure Reference

```
lowram-gemma4-vision/              (branch: phase2)
├── docs/
│   └── PLANNING/
│       ├── PRD.md                 # Product Requirements v2.1.0
│       ├── TRD.md                 # Technical Design v2.1.0
│       ├── SPEC.md                # ⬅️ Este documento (Technical Specification)
│       ├── IMPLEMENTATION.md      # Implementation Plan v2.1.0
│       ├── FINAL_DECISION.md      # Multi-agent decision
│       ├── CONSENSUS.md           # Validation consensus
│       ├── REVIEW_CYCLE1.md       # Review feedback ciclo 1
│       └── REVIEW_SPEC_CYCLE1.md  # Review feedback SPEC ciclo 1
├── config.toml                    # mistral.rs config (ISQ Q4K, :8080)
├── nginx/
│   └── gemma4-rs.conf             # nginx config (para /etc/nginx/conf.d/)
├── systemd/
│   └── gemma4-rs.service          # systemd user service
├── scripts/
│   ├── install.sh                 # Setup: deps + compile + config
│   ├── benchmark.sh               # VRAM/RAM/latency benchmarks
│   └── test_vision.sh             # Vision test con curl
├── tests/
│   ├── test_text.sh               # curl test texto
│   ├── test_vision.sh             # curl test imagen base64
│   ├── test_audio.sh              # curl test audio WAV
│   └── test_streaming.sh          # curl test SSE
├── rag/                           # [POST-MIGRACIÓN]
│   ├── embeddings/                # ONNX o pre-computados
│   └── faiss_index/               # Índice FAISS
└── README.md
```

---

## 10. Version History

| Version | Fecha | Cambios |
|---------|-------|---------|
| 1.1.0 | 2026-06-05 | Ciclo 1 review: eliminadas secciones TOML fabricadas (flash_attention, sequence), unificado in_situ_quant a "4", corregido /health (siempre 200), separados errores nginx vs mistral.rs, ajustado latency P50 a ≤30s, unificado model ID |
| 1.0.0 | 2026-06-05 | Draft inicial |
