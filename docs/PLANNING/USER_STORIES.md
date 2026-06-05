# User Stories y Use Cases — lowram-gemma4-vision Phase 2

**Versión:** 1.2.0 (CC+OC+AGY review — 11 fixes aplicados)  
**Fecha:** 2026-06-05  
**Autores:** CC + OC + AGY (consolidación de 3 perspectivas)  
**Proyecto:** lowram-gemma4-vision Phase 2 — Migración Python → Rust (mistral.rs + nginx)

---

## 1. User Stories

### US-01: Chat de texto vía curl

> **Como** Hans (desarrollador),  
> **quiero** enviar un prompt de texto vía `curl` a `/v1/chat/completions` y recibir una respuesta coherente del modelo Gemma 4 E4B,  
> **para** validar que el motor de inferencia en Rust funciona correctamente sin Python.

**Criterios de aceptación:**
- `curl -X POST localhost:8080/v1/chat/completions` con `messages: [{"role":"user","text":"Hola"}]` responde HTTP 200.
- La respuesta tiene formato OpenAI-compatible (`id`, `choices`, `usage`).
- Latencia ≤ 15 segundos para 128 tokens de output (target aspiracional; ≤30s es aceptable para RTX 3060).
- RAM del proceso `mistralrs` ≤ 150 MB medido con `ps aux`.
- VRAM total ≤ 8.5 GB durante inferencia texto-only (`nvidia-smi`).

**Prioridad:** Must Have  
**Rol:** Hans (curl client)  
**Trazabilidad:** PRD O1 (VRAM), O2 (RAM), O5 (API), O6 (Latency)

---

### US-02: Inferencia con imagen (visión)

> **Como** Hans (desarrollador),  
> **quiero** enviar una imagen codificada en base64 junto con un prompt de texto al endpoint `/v1/chat/completions`,  
> **para** obtener una descripción o análisis de la imagen sin que el sistema haga CUDA OOM.

**Criterios de aceptación:**
- Request con `image_url` (base64 JPEG/PNG) + texto responde 200 OK.
- Respuesta describe correctamente el contenido de la imagen.
- `nvidia-smi` muestra VRAM ≤ 8.5 GB durante el request (con ISQ Q4K).
- No se produce `torch.cuda.empty_cache()` ni `gc.collect()` (no hay Python).
- Vision encoder se carga on-demand y libera VRAM al terminar.

**Prioridad:** Must Have  
**Rol:** Hans (curl client)  
**Trazabilidad:** PRD O1, O3 (Visión sin OOM)

---

### US-03: Inferencia con audio

> **Como** Hans (desarrollador),  
> **quiero** enviar un archivo de audio (WAV/MP3/FLAC) junto con un prompt de texto,  
> **para** que el modelo transcriba o responda preguntas sobre el contenido del audio.

**Criterios de aceptación:**
- Request con `audio_url` (file:/// o data:audio) responde 200 OK.
- Audio se decodifica correctamente vía FFmpeg (16kHz mono).
- Respuesta es coherente con el contenido del audio.
- VRAM ≤ 8.5 GB durante inferencia audio+texto.
- Audio encoder se carga on-demand.

**Prioridad:** Must Have  
**Rol:** Hans (curl client)  
**Trazabilidad:** PRD O1, O4 (Audio)

---

### US-04: Streaming SSE en tiempo real

> **Como** desarrollador de mobile app,  
> **quiero** recibir tokens incrementalmente vía Server-Sent Events (SSE) con `stream=true`,  
> **para** mostrar la respuesta progresivamente en la UI de la app (como hace Poe) y no tener una pantalla de carga de 30 segundos.

**Criterios de aceptación:**
- `curl -N http://localhost/v1/chat/completions` con `"stream":true` devuelve chunks SSE.
- Cada chunk tiene formato `data: {"id":"...","choices":[{"delta":{"content":"token"}}]}`.
- Chunks llegan en tiempo real (delay < 5s entre chunks).
- Último chunk tiene `finish_reason:"stop"` seguido de `data: [DONE]`.
- nginx no bufferiza la respuesta (`proxy_buffering off`).

**Prioridad:** Must Have  
**Rol:** Mobile app developer  
**Trazabilidad:** PRD O7 (Streaming), SPEC §5 (SSE Protocol)

---

### US-05: Discord bot responde en canal

> **Como** Hans (usuario de Discord),  
> **quiero** que el bot de Hermes en Discord responda a mis mensajes en el canal `#lowram-gemma4` con texto generado por Gemma 4,  
> **para** tener una interfaz conversacional sin abrir una terminal.

**Criterios de aceptación:**
- Bot recibe mensaje de Discord → envía a mistral.rs → responde en el canal.
- Soporta mensajes con imagen adjunta (visión).
- Bot ignora sus propios mensajes (no loop infinito).
- Bot no requiere intervención manual.

**Prioridad:** High  
**Rol:** Hans (Discord user)  
**Trazabilidad:** TRD §7 (Hermes Gateway Integration)

---

### US-06: Mobile app Poe-like con chat multimodal

> **Como** usuario de mobile app (Poe-like),  
> **quiero** enviar prompts de texto, imágenes y notas de voz desde mi celular y recibir respuestas en tiempo real,  
> **para** tener una experiencia de chat multimodal similar a ChatGPT/Poe pero con mi propio modelo local.

**Criterios de aceptación:**
- App conecta a `http://192.168.1.5:80/v1` (LAN).
- Soporta texto, imagen (cámara/galería) y audio (micrófono).
- Muestra respuestas progresivas con streaming.
- Funciona sin internet (todo local).
- Maneja timeouts de red (reintento automático).

**Prioridad:** High  
**Rol:** Mobile app user  
**Trazabilidad:** PRD O3, O4, O7

---

### US-07: API segura con autenticación y rate limiting

> **Como** administrador de sistema,  
> **quiero** que el endpoint `/v1` esté protegido con autenticación (HTTP Basic Auth o API key) y rate limiting (10 req/s),  
> **para** evitar uso abusivo o accesos no autorizados a mi servidor local desde la LAN.

**Criterios de aceptación:**
- Sin header `Authorization` → 401 Unauthorized.
- Con credenciales correctas → 200 OK.
- >10 req/s promedio → 429 Too Many Requests.
- Burst hasta 20 requests → 429 solo después.
- Rate limiting por IP (no global).

**Prioridad:** High  
**Rol:** System admin  
**Trazabilidad:** TRD §6 (nginx reverse proxy), SPEC §6 (Security)

---

### US-08: Service systemd con auto-restart

> **Como** administrador de sistema,  
> **quiero** que `gemma4-rs.service` inicie automáticamente en el boot y se reinicie si el proceso crash,  
> **para** no tener que reiniciar el servicio manualmente después de un error o reinicio del sistema.

**Criterios de aceptación:**
- `systemctl --user enable gemma4-rs.service` → auto-inicio en boot (requiere `loginctl enable-linger hans` para persistir sin sesión activa).
- Matar el proceso → systemd lo reinicia automáticamente (RestartSec=10).
- Logs accesibles via `journalctl --user -u gemma4-rs.service -f`.
- Service carga el modelo al inicio (no lazy load).

**Prioridad:** High  
**Rol:** System admin  
**Trazabilidad:** TRD §7 (systemd service), IMPLEMENTATION Fase 5

---

### US-09: Consumo de recursos dentro de presupuesto

> **Como** Hans (desarrollador con recursos limitados),  
> **quiero** que el motor de inferencia ocupe ≤ 8.5 GB VRAM y ≤ 150 MB RAM,  
> **para** no saturar mi sistema (RAM 380 MB libre worst-case) y tener margen para otras aplicaciones.

**Criterios de aceptación:**
- `nvidia-smi` idle (modelo cargado) → ≤ 8.5 GB VRAM.
- `ps aux | grep mistralrs` → RSS ≤ 150 MB.
- Durante inferencia texto+visión+audio → ≤ 8.5 GB VRAM (Escenario A: ISQ cuantiza `embed_tokens_per_layer`).
- `free -h` → RAM libre ≥ 1 GB después del boot (vs 380 MB en Phase 1).

**Prioridad:** Must Have  
**Rol:** Hans (developer)  
**Trazabilidad:** PRD O1, O2, HW §2.2 (RAM bottleneck)

---

### US-10: RAG con búsqueda vectorial

> **Como** Hans (usuario con memoria conversacional),  
> **quiero** que el sistema recupere contexto relevante del historial de conversaciones vía FAISS,  
> **para** que el modelo recuerde temas discutidos anteriormente sin tener que repetir información.

**Criterios de aceptación:**
- RAG service consulta índice FAISS y devuelve top-K chunks.
- Contexto recuperado se inyecta en el prompt del modelo.
- RAG funciona en CPU (no consume VRAM).
- Búsqueda es relevante (mismatch < 10% en test set).

**Prioridad:** Medium (Post-Phase 2)  
**Rol:** Hans (end user)  
**Trazabilidad:** PRD O9, TRD §10 (RAG post-migración)

---

### US-11: Health check y monitoreo

> **Como** administrador de sistema,  
> **quiero** consultar `/health` para verificar si el servidor está vivo,  
> **para** integrarlo en un sistema de monitoreo (uptime checks) y saber si debo intervenir.

**Criterios de aceptación:**
- `curl http://localhost/health` → 200 OK si el servidor escucha.
- Endpoint no requiere autenticación (nginx lo pasa directo).
- Response JSON simple: `{"status":"ok"}`.
- Para verificar modelo cargado, usar `GET /v1/models` (incluye `created` timestamp).

**Prioridad:** Medium  
**Rol:** System admin  
**Trazabilidad:** SPEC §1.4 (/health endpoint)

---

### US-12: Despliegue reproducible con scripts

> **Como** Hans (desarrollador),  
> **quiero** ejecutar un solo script `install.sh` que instale dependencias, compile mistral.rs, configure nginx y cree el servicio systemd,  
> **para** poder replicar el deployment en otra máquina o restaurar el sistema desde cero sin errores manuales.

**Criterios de aceptación:**
- `./install.sh` → instala cudnn, nginx, compila mistral.rs, configura todo.
- Script es idempotente (puede ejecutarse varias veces sin romper).
- Script valida cada paso (fail fast si algo falla).
- Script funciona en Arch Linux (kernel 7.0.10-zen1).

**Prioridad:** High  
**Rol:** Hans (developer)  
**Trazabilidad:** IMPLEMENTATION Fase 0-1, scripts/install.sh

---

## 2. Use Cases

### UC-01: Chat de texto básico vía curl

**Actor:** Hans (desarrollador)  
**Precondiciones:**
- mistral.rs compilado con CUDA + flash-attn + cudnn
- Servicio `gemma4-rs.service` running
- Modelo `gemma4-heretic` cargado (VRAM idle ~8.5 GB)

**Flujo principal:**
1. Hans abre terminal y ejecuta:
   ```bash
   curl -s http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"gemma4-heretic","messages":[{"role":"user","content":"¿De qué color es el cielo?"}],"max_tokens":50}'
   ```
2. mistral.rs recibe request, tokeniza prompt, ejecuta forward pass.
3. Modelo genera respuesta: "El cielo es azul durante el día y se oscurece..."
4. mistral.rs devuelve HTTP 200 con formato OpenAI:
   ```json
   {"id":"chatcmpl-xxx","choices":[{"message":{"content":"El cielo es azul...","role":"assistant"}}],"usage":{"total_tokens":78}}
   ```
5. Hans ve la respuesta en terminal.

**Postcondiciones:**
- VRAM ≤ 8.5 GB durante toda la transacción
- RAM proceso ≤ 150 MB
- Latencia ≤ 30s para 50 tokens output

**Flujos alternativos:**
- 3A. Modelo no cargado → 503 Service Unavailable → verificar `systemctl --user status gemma4-rs.service`
- 3B. CUDA OOM → 500 Server Error → reducir contexto o desactivar modalidades

---

### UC-02: Inferencia visión con imagen

**Actor:** Hans (desarrollador)  
**Precondiciones:**
- Servicio running
- Imagen JPEG disponible en `/tmp/test_img.jpg`

**Flujo principal:**
1. Hans codifica imagen en base64:
   ```bash
   IMG_B64=$(base64 -w0 /tmp/test_img.jpg)
   ```
2. Envía request con imagen:
   ```bash
   curl -s http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d "{\"model\":\"gemma4-heretic\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"¿Qué hay en esta imagen?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$IMG_B64\"}}]}],\"max_tokens\":100}"
   ```
3. mistral.rs decodifica imagen (JPEG → bitmap).
4. Vision encoder (16 capas, 768d) procesa imagen → embeddings.
5. Vision projector adapta embeddings al espacio del LLM.
6. LLM genera respuesta: "La imagen muestra un gato..."
7. Hans recibe respuesta 200 OK.

**Postcondiciones:**
- VRAM ≤ 8.5 GB (incluyendo vision encoder on-demand)
- Latencia ≤ 40s (2s overhead de encoder + ~38s generación)

**Flujos alternativos:**
- 4A. Imagen corrupta → 400 Invalid Request
- 4B. CUDA OOM → 500 Server Error

---

### UC-03: Inferencia audio

**Actor:** Hans (desarrollador)  
**Precondiciones:**
- FFmpeg instalado
- Audio WAV disponible en `/tmp/audio.wav`

**Flujo principal:**
1. Hans envía request con audio:
   ```bash
   AUDIO_B64=$(base64 -w0 /tmp/audio.wav)
   curl -s http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d "{\"model\":\"gemma4-heretic\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Transcribe esto\"},{\"type\":\"audio_url\",\"image_url\":{\"url\":\"data:audio/wav;base64,$AUDIO_B64\"}}]}],\"max_tokens\":200}"
   ```
2. mistral.rs decodifica audio vía FFmpeg (WAV → 16kHz mono).
3. Audio encoder (12 capas, 1024d) procesa audio → embeddings.
4. LLM genera transcripción/respuesta.

**Postcondiciones:**
- Transcripción coherente
- VRAM ≤ 8.5 GB

**Flujos alternativos:**
- 3A. Audio corrupto o formato inválido → 400 Invalid Request
- 3B. FFmpeg no instalado → 500 Server Error
- 3C. CUDA OOM durante audio encoder → 500 Server Error
- 3D. Archivo > 50 MB → 413 Request Entity Too Large (nginx)

---

### UC-04: Streaming SSE para mobile app

**Actor:** Mobile app (iOS/Android)  
**Precondiciones:**
- App conectada a LAN (192.168.1.x)
- nginx configurado con `proxy_buffering off`

**Flujo principal:**
1. App envía request con `"stream":true`.
2. mistral.rs comienza a generar tokens.
3. Cada token se envía como chunk SSE:
   ```
   data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hola"}}]}

   data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" mundo"}}]}

   data: {"id":"chatcmpl-xxx","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

   data: [DONE]
   ```
4. App recibe chunks en tiempo real, actualiza UI progresivamente.

**Postcondiciones:**
- Delay entre chunks < 5s
- nginx no bufferiza

---

### UC-05: Deployment completo desde cero

**Actor:** Hans (desarrollador)  
**Precondiciones:**
- Arch Linux con kernel 7.0.10-zen1
- CUDA 13.3.33 instalado
- Rust 1.96.0 instalado
- Modelo safetensors en `~/models/gemma4-heretic/`

**Flujo principal:**
1. Hans clona repo:
   ```bash
   git clone https://github.com/hbuddenberg/lowram-gemma4-vision.git
   cd lowram-gemma4-vision
   git checkout phase2
   ```
2. Ejecuta script de instalación:
   ```bash
   ./scripts/install.sh
   ```
3. Script instala cudnn, nginx.
4. Script compila mistral.rs (`cargo install mistralrs-cli --features "cuda flash-attn cudnn"`).
5. Script crea config.toml y lo copia a `~/lowram-gemma4-vision/`.
6. Script configura nginx (copia `.conf` a `/etc/nginx/conf.d/`).
7. Script crea `gemma4-rs.service` en `~/.config/systemd/user/`.
8. Script inicia servicio: `systemctl --user enable --now gemma4-rs.service`.
9. Hans ejecuta test: `curl http://localhost/v1/models` → 200 OK.

**Postcondiciones:**
- Servicio running
- Modelo cargado
- VRAM ≤ 8.5 GB
- nginx accesible en LAN

---

### UC-06: Recuperación automática ante crash

**Actor:** System (systemd)  
**Precondiciones:**
- Servicio running
- `loginctl enable-linger hans` ejecutado (para persistir sin sesión activa)
- Service configurado con `Restart=on-failure`, `RestartSec=10`, `StartLimitBurst=5`, `StartLimitIntervalSec=300`

**Flujo principal:**
1. Proceso mistralrs crash (CUDA OOM, segfault).
2. systemd detecta crash.
3. Espera 10 segundos (RestartSec).
4. Reinicia proceso automáticamente.
5. Servicio vuelve a running.

**Postcondiciones:**
- Servicio se recupera sin intervención
- Logs muestran crash y restart

**Flujos alternativos:**
- 6A. Restart limit agotado (5 crashes en 300s) → servicio marcado `failed`, ejecutar `systemctl reset-failed gemma4-rs.service`
- 6B. GPU no disponible post-crash → reinicio falla, esperar recuperación GPU o reiniciar sistema
- 6C. VRAM no liberada tras crash → segundo inicio hace OOM, limpiar con `nvidia-smi --gpu-reset`

---

### UC-07: Monitoreo salud en producción

**Actor:** Hans (administrador)  
**Precondiciones:**
- Servidor en producción

**Flujo principal:**
1. Hans configura cron job para health check cada 5 min.
2. Script hace `curl http://localhost/health`.
3. Si curl falla (connection refused) → envía alerta.
4. Script hace `curl http://localhost/v1/models` para verificar modelo cargado.
5. Hans revisa `journalctl --user -u gemma4-rs.service` para diagnóstico.
6. Hans revisa `nvidia-smi` para VRAM.
7. Hans reinicia servicio si necesario.

**Postcondiciones:**
- Alerta enviada si servicio caído o modelo no cargado
- Logs accesibles para diagnóstico
- VRAM documentada
- Servicio restaurado si fue necesario

**Flujos alternativos:**
- 7A. `/health` devuelve 200 pero `/v1/models` falla → modelo descargado, reiniciar servicio
- 7B. Health check timeout → verificar servidor vivo (connection refused)
- 7C. VRAM excede 8.5 GB → acción correctiva (restart o reducir contexto)

---

### UC-08: Acceso app móvil vía LAN

**Actor:** Mobile app (iOS)  
**Precondiciones:**
- Móvil conectado a misma red Wi-Fi (192.168.1.x)
- App configurada con host `192.168.1.5:80`
- nginx configurado con `listen 0.0.0.0:80` (no solo 127.0.0.1)

**Flujo principal:**
1. App envía request a `http://192.168.1.5/v1/chat/completions`.
2. nginx recibe request en `0.0.0.0:80`.
3. nginx valida auth (Basic Auth).
4. nginx hace proxy a `127.0.0.1:8080`.
5. mistral.rs procesa request.
6. Respuesta vuelve a través de nginx.
7. App muestra respuesta.

**Postcondiciones:**
- App recibe respuesta 200 OK
- Auth validada correctamente
- Contenido de respuesta correcto
- Tráfico permanece en LAN (no sale a internet)

**Flujos alternativos:**
- 8A. Auth falla (credenciales incorrectas) → 401 Unauthorized
- 8B. Rate limit excedido → 429 Too Many Requests
- 8C. mistral.rs caído → 502 Bad Gateway
- 8D. Timeout de inferencia > 300s → 504 Gateway Timeout
- 8E. Request body > 50 MB → 413 Request Entity Too Large

---

## 3. Matriz de Trazabilidad

| User Story | Use Case | Rol | PRD Objective | Implementación Fase |
|------------|----------|-----|---------------|---------------------|
| US-01 | UC-01 | Hans (curl) | O1, O2, O5, O6 | Fase 2 |
| US-02 | UC-02 | Hans (curl) | O1, O3 | Fase 3 |
| US-03 | UC-03 | Hans (curl) | O1, O4 | Fase 3 |
| US-04 | UC-04 | Mobile app dev | O7 | Fase 3 |
| US-05 | - | Hans (Discord) | - | Post-Fase 5 |
| US-06 | UC-08 | Mobile app user | O3, O4, O7 | Fase 4 |
| US-07 | - | System admin | - | Fase 4 |
| US-08 | UC-06 | System admin | - | Fase 5 |
| US-09 | - | Hans (dev) | O1, O2 | Todas |
| US-10 | - | Hans (end user) | O9 | Post-Phase 2 |
| US-11 | UC-07 | System admin | - | Fase 6 |
| US-12 | UC-05 | Hans (dev) | - | Fase 0-1 |

---

## 4. Priorización por Fase

**Fase 2A (MVP):** US-01, US-02, US-03, US-04 (texto + visión básico + streaming)  
**Fase 2B (Multimodal):** US-06, US-08 (mobile + audio)  
**Fase 2C (Production):** US-07, US-11, US-12 (auth + monitoreo + deployment)  
**Post-Phase 2:** US-05, US-10 (Discord + RAG)

---

## 5. Notas de Consenso

**Consolidado desde 3 perspectivas:**
- **CC:** Enfoque en user stories completas con criterios de aceptación medibles
- **OC:** Enfoque en infrastructure/ops (deployment, crash recovery, monitoring)
- **AGY:** Enfoque en multimodal (visión + audio simultáneo) y constraints de RAM (380MB libre)

**Decisiones clave:**
- US-09 (Recursos ≤ presupuesto) es MUST HAVE dado el bottleneck de RAM
- US-02 (Visión) priorizada sobre US-03 (Audio) — audio es más experimental
- US-10 (RAG) marcada como Post-Phase 2 — fuera del MVP de migración
- UC-05 (Deployment) documenta el script `install.sh` que OC enfatizó

---

## 6. Referencias

- PRD.md v2.1.0 (Product Requirements)
- TRD.md v2.1.0 (Technical Design)
- SPEC.md v1.1.0 (Technical Specification)
- IMPLEMENTATION.md v2.1.0 (Implementation Plan)
