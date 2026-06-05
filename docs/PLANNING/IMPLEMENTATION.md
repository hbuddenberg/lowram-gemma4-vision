# Plan de Implementación — lowram-gemma4-vision Phase 2

**Version:** 2.1.0  
**Fecha:** 2026-06-05  
**Autor:** Hans-Dieter Buddenberg Blamey  
**Branch:** phase2  
**Duración estimada:** 7–10 días hábiles  

---

## Fase 0: Prerrequisitos (Día 1, ~1 hora)

### Objetivo
Verificar dependencias del sistema y toolchain.

### Tareas

| # | Tarea | Comando | Verificación |
|---|-------|---------|-------------|
| 0.1 | Instalar cudnn | `echo PASSWORD \| sudo -S pacman -S cudnn` | `pacman -Q cudnn` → versión |
| 0.2 | Instalar nginx | `echo PASSWORD \| sudo -S pacman -S nginx` | `pacman -Q nginx` → versión |
| 0.3 | **Verificar** FFmpeg (ya instalado) | `ffmpeg -version` | Muestra versión |
| 0.4 | Verificar Rust | `rustc --version && cargo --version` | >= 1.96.0 |
| 0.5 | Verificar CUDA | `/opt/cuda/bin/nvcc --version` | 13.3.33 |
| 0.6 | Verificar GPU | `nvidia-smi` | RTX 3060, 12288 MiB |
| 0.7 | Detener server.py | `systemctl --user stop gemma4-api.service` | `systemctl --user status gemma4-api.service` → inactive |
| 0.8 | Verificar VRAM libre | `nvidia-smi --query-gpu=memory.free --format=csv,noheader` | > 10000 MiB (sin modelo cargado) |

### Gate
- [ ] cudnn, nginx instalados; FFmpeg verificado
- [ ] Rust >= 1.96.0
- [ ] CUDA 13.3.33
- [ ] server.py detenido (libera VRAM para compilación)

---

## Fase 1: Compilar mistral.rs (Día 1-2, ~30-60 min compilación)

### Objetivo
Compilar `mistralrs-cli` con soporte CUDA + flash-attn + cudnn.

### Tareas

| # | Tarea | Comando | Verificación |
|---|-------|---------|-------------|
| 1.1 | Compilar mistral.rs | `cargo install mistralrs-cli --features "cuda flash-attn cudnn"` | `~/.cargo/bin/mistralrs --version` |
| 1.2 | Verificar instalación | `mistralrs --help 2>&1 \| head -10` | Muestra opciones serve, run, etc. |
| 1.3 | Verificar CUDA detectado | `mistralrs run --help 2>&1 \| grep -i cuda` | Menciona CUDA |

### Posibles errores y fixes

| Error | Causa | Fix |
|-------|-------|-----|
| `cargo install` OOM | Compilación usa ~3GB RAM, zram saturado | Cerrar Chrome/otras apps antes de compilar. zram comprime en RAM, liberar Python da más espacio |
| `cudnn` link error | cudnn no encontrado en paths | `export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH` |
| flash-attn compile fail | No compatible con CUDA 13.3 | Quitar feature: `--features "cuda cudnn"` |
| nvcc no encontrado | No está en PATH | `export PATH=/opt/cuda/bin:$PATH` |

### Gate
- [ ] `mistralrs --version` funciona
- [ ] Binario en `~/.cargo/bin/mistralrs`

---

## Fase 2: Verificar Carga del Modelo (Día 2, ~5 min)

### Objetivo
Confirmar que mistral.rs carga el modelo heretic con ISQ Q4K y responde a texto. **Verificar si ISQ cuantiza `embed_tokens_per_layer`** (5.64 GB fp16, 35% del modelo).

### Tareas

| # | Tarea | Comando | Verificación |
|---|-------|---------|-------------|
| 2.1 | Cargar modelo (interactive) | `mistralrs run --quant q4k -m ~/models/gemma4-heretic` | Carga sin error, prompt interactivo disponible |
| 2.2 | **⚠️ Verificar embed_tokens_per_layer** | Observar logs de carga o `nvidia-smi` durante carga | Si VRAM ~5.3 GB → ISQ lo cuantizó ✅. Si VRAM ~9.5 GB → no lo cuantizó 🔴 |
| 2.3 | Test texto | Escribir: "Hola, ¿cómo estás?" | Respuesta coherente en español |
| 2.4 | Verificar VRAM tras carga | `nvidia-smi --query-gpu=memory.used --format=csv,noheader` | ≤ 5.5 GB (si ISQ cuantiza embed_tokens_per_layer) |
| 2.5 | Verificar RAM del proceso | `ps aux \| grep mistralrs \| grep -v grep` | RSS ≤ 200 MB |

### Gate
- [ ] Modelo carga sin error
- [ ] Responde texto coherentemente
- [ ] VRAM ≤ 5.5 GB idle (Escenario A) — **si VRAM > 9 GB, documentar como Escenario B**
- [ ] RAM ≤ 200 MB

### ⚠️ Si esta fase falla
- Verificar que config.json tiene `model_type: "gemma4"` y `architectures: ["Gemma4ForConditionalGeneration"]`
- Probar sin ISQ: `mistralrs run -m ~/models/gemma4-heretic` (fp16, usará más VRAM pero confirma compatibilidad)
- Verificar logs: `RUST_LOG=debug mistralrs run --quant q4k -m ~/models/gemma4-heretic 2>&1 | tee /tmp/mistralrs-debug.log`
- **Si embed_tokens_per_layer no se cuantiza:** evaluar offload a CPU/RAM de ese tensor, o operar en modo texto-only con headroom limitado

---

## Fase 3: Verificar Multimodal (Día 2, ~15 min)

### Objetivo
Confirmar que visión y audio funcionan con el modelo heretic.

### Tareas

| # | Tarea | Comando | Verificación |
|---|-------|---------|-------------|
| 3.1 | Iniciar servidor | `mistralrs serve --quant q4k --port 8080 -m ~/models/gemma4-heretic &` | Escuchando en :8080 |
| 3.2 | Test texto via HTTP | `curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"gemma4-heretic","messages":[{"role":"user","content":"Di OK"}],"max_tokens":10}'` | 200 OK, response contiene "OK" |
| 3.3 | Test texto + VRAM | `nvidia-smi --query-gpu=memory.used --format=csv,noheader` | ≤ 5.5 GB (Escenario A) |
| 3.4 | Generar imagen de test con FFmpeg | `ffmpeg -f lavfi -i color=c=red:s=200x200:d=0.1 -frames:v 1 /tmp/test_img.jpg` | Archivo existe |
| 3.5 | Test visión (base64) | Ver comando abajo | 200 OK, describe color rojo |
| 3.6 | Verificar VRAM con visión | `nvidia-smi --query-gpu=memory.used --format=csv,noheader` | ≤ 7.0 GB (Escenario A) |
| 3.7 | Test audio | Generar WAV + enviar request | 200 OK, respuesta coherente |
| 3.8 | Verificar VRAM con audio | `nvidia-smi --query-gpu=memory.used --format=csv,noheader` | ≤ 7.0 GB (Escenario A) |
| 3.9 | Test streaming | `curl -N http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"gemma4-heretic","messages":[{"role":"user","content":"Cuenta del 1 al 5"}],"max_tokens":50,"stream":true}'` | Chunks SSE incrementales |

### Comando test visión (3.5)
```bash
IMG_B64=$(base64 -w0 /tmp/test_img.jpg)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"gemma4-heretic\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"¿De qué color es esta imagen?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${IMG_B64}\"}}]}],\"max_tokens\":50}"
```

### Gate
- [ ] Texto: 200 OK, VRAM ≤ 5.5 GB (Escenario A)
- [ ] Visión: 200 OK, describe imagen, VRAM ≤ 7.0 GB (Escenario A)
- [ ] Audio: 200 OK, respuesta coherente, VRAM ≤ 7.0 GB (Escenario A)
- [ ] Streaming: chunks SSE funcionan
- [ ] Sin CUDA OOM en ningún test

### ⚠️ Si visión falla
- Puede ser que ISQ no cuantiza correctamente los projectors de visión
- Probar sin ISQ para aislar: es un problema de cuantización o de arquitectura
- Revisar `mistralrs` issues en GitHub sobre Gemma 4 vision + ISQ

---

## Fase 4: Configurar nginx (Día 3, ~30 min)

### Objetivo
Configurar reverse proxy con auth y rate limiting.

### Tareas

| # | Tarea | Comando | Verificación |
|---|-------|---------|-------------|
| 4.1 | Crear config nginx | Copiar config del TRD (Sección 6) | Archivo en `/etc/nginx/conf.d/gemma4-rs.conf` |
| 4.2 | Crear htpasswd | `echo PASSWORD \| sudo -S htpasswd -c /etc/nginx/.htpasswd hbuddenberg` | Archivo creado |
| 4.3 | Test config | `echo PASSWORD \| sudo -S nginx -t` | `syntax is ok`, `test is successful` |
| 4.4 | Iniciar nginx | `echo PASSWORD \| sudo -S systemctl enable --now nginx` | Running |
| 4.5 | Test auth correcta | `curl -u hbuddenberg:PASSWORD http://localhost/v1/models` | 200 OK |
| 4.6 | Test auth incorrecta | `curl http://localhost/v1/models` | 401 Unauthorized |
| 4.7 | Test proxy a mistralrs | `curl -s -u hbuddenberg:PASSWORD http://localhost/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"gemma4-heretic","messages":[{"role":"user","content":"OK"}],"max_tokens":5}'` | 200 OK con respuesta |
| 4.8 | Configurar LAN access | Cambiar `listen 80` → `listen 0.0.0.0:80` en nginx conf | Accesible desde 192.168.1.x |
| 4.9 | Test desde LAN | `curl -u hbuddenberg:PASSWORD http://192.168.1.XX/v1/models` (ejecutar desde otro dispositivo en la red, ej: móvil o laptop) | 200 OK |

### Gate
- [ ] Auth funciona (401 sin credenciales, 200 con)
- [ ] Proxy pasa requests a mistralrs
- [ ] Accesible desde LAN

---

## Fase 5: Systemd Service (Día 3, ~15 min)

### Objetivo
Crear servicio systemd para auto-inicio.

### Tareas

| # | Tarea | Comando | Verificación |
|---|-------|---------|-------------|
| 5.1 | Crear service file | Copiar del TRD (Sección 7) | `~/.config/systemd/user/gemma4-rs.service` |
| 5.2 | Reload daemon | `systemctl --user daemon-reload` | Sin error |
| 5.3 | Enable + start | `systemctl --user enable --now gemma4-rs.service` | Running |
| 5.4 | Verificar | `systemctl --user status gemma4-rs.service` | Active (running) |
| 5.5 | Test via nginx | curl al endpoint nginx | 200 OK |
| 5.6 | Test restart | `systemctl --user restart gemma4-rs.service` | Reinicia correctamente |
| 5.7 | Test crash recovery | Matar proceso, verificar auto-restart | `RestartSec=10` funciona |

### Gate
- [ ] Service arranca, para, reinicia correctamente
- [ ] Auto-restart en crash funciona

---

## Fase 6: Benchmarks (Día 4, ~30 min)

### Objetivo
Medir VRAM, RAM y latencia reales. Comparar con Phase 1.

### Tareas

| # | Métrica | Comando | Target |
|---|---------|---------|--------|
| 6.1 | VRAM idle | `nvidia-smi --query-gpu=memory.used --format=csv,noheader` | ≤ 5.5 GB (Escenario A) |
| 6.2 | VRAM inferencia texto | Medir durante curl text | ≤ 6.0 GB |
| 6.3 | VRAM inferencia visión | Medir durante curl con imagen | ≤ 7.0 GB |
| 6.4 | VRAM inferencia audio | Medir durante curl con audio | ≤ 7.0 GB |
| 6.5 | RAM proceso | `ps -p $(pgrep mistralrs) -o rss=` | ≤ 200 MB |
| 6.6 | Latencia text (128 tok) | `time curl ...` | ≤ 15s |
| 6.7 | Latencia vision | `time curl ... con imagen` | ≤ 20s |
| 6.8 | Tokens/segundo | Extraer del response | ≥ 3 tok/s |
| 6.9 | RAM libre sistema | `free -h` | ≥ 1.5 GB (vs 380 MB worst-case Phase 1) |

### Tabla de comparación (llenar con datos reales)

| Métrica | Phase 1 (server.py) | Phase 2 (mistral.rs) | Mejora |
|---------|---------------------|---------------------|--------|
| VRAM idle | 9.3 GB | _ | _ |
| VRAM text | ~10 GB | _ | _ |
| VRAM vision | OOM 🔴 | _ | _ |
| RAM engine | 1.6 GB | _ | _ |
| RAM libre | 380 MB (worst-case) | _ | _ |
| Latencia text (128 tok) | ~4.2s | _ | _ |
| Tokens/s | ~4.2 | _ | _ |

### Gate
- [ ] Todos los targets cumplidos
- [ ] Tabla de comparación completada

---

## Fase 7: Cleanup y Documentación (Día 5, ~1 hora)

### Objetivo
Limpiar Phase 1, documentar resultados, preparar release.

### Tareas

| # | Tarea | Detalle |
|---|-------|---------|
| 7.1 | Deshabilitar servicio viejo | `systemctl --user disable gemma4-api.service` (servicio Python Phase 1) |
| 7.2 | Actualizar Hermes config | Cambiar provider a `http://localhost/v1` (via nginx) |
| 7.3 | Commit docs + config | `git add -A && git commit -m "Phase 2: mistral.rs migration"` |
| 7.4 | Push a GitHub | `git push origin phase2` |
| 7.5 | Limpiar archivos viejos | Archivar o borrar server.py, rag.py, inference.py de master |
| 7.6 | README.md | Instrucciones de deploy: install, config, start, test |
| 7.7 | Screenshot VRAM | Captura de nvidia-smi post-migración |
| 7.8 | Test end-to-end Discord | Enviar mensaje desde Discord → Hermes → mistral.rs → respuesta |

### Gate
- [ ] Servicio viejo deshabilitado
- [ ] Hermes actualizado y funcionando con mistral.rs
- [ ] GitHub actualizado
- [ ] README con instrucciones completas

---

## Cronograma

| Día | Fase | Entregable |
|-----|------|-----------|
| **1** | 0: Prerrequisitos | Deps instaladas/verificadas, toolchain verificado |
| **1-2** | 1: Compilar mistral.rs | Binario compilado |
| **2** | 2: Verificar modelo | Carga + texto OK + **verificar embed_tokens_per_layer ISQ** |
| **2** | 3: Verificar multimodal | Visión + audio + streaming OK |
| **3** | 4: nginx | Reverse proxy con auth |
| **3** | 5: systemd | Auto-start service |
| **4** | 6: Benchmarks | Tabla comparativa |
| **5** | 7: Cleanup | Release listo |
| **6-10** | Buffer | RAG post-migración (opcional) |

---

## Riesgos y Mitigaciones

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|-------|-------------|---------|------------|
| R1 | **ISQ no cuantiza embed_tokens_per_layer (5.64 GB)** | **Alta** | **Crítico** | ⚠️ Verificar en Fase 2. Si falla: evaluar offload CPU/RAM, modo texto-only, o conversión selectiva a GGUF de ese tensor |
| R2 | Compilación OOM (RAM insuficiente, zram saturado) | Alta | Medio | Cerrar apps, zram es más flexible que swap file. Si falla: compilar en otra máquina y copiar binario |
| R3 | ISQ no funciona con heretic | Media | Alto | Probar sin ISQ (fp16) primero. Si fp16 funciona, ISQ debería también |
| R4 | Visión "ciega" con ISQ | Media | Alto | ISQ solo cuantiza LM, encoders permanecen fp16. Verificar en Fase 3 |
| R5 | flash-attn no compila con CUDA 13.3 | Media | Bajo | Quitar feature, usar standard attention (más lento pero funcional) |
| R6 | VRAM excede budget con audio+visión (Escenario B) | Baja-Media | Alto | Reducir max_seq_len o KV cache size. En Escenario B, headroom es muy limitado |
| R7 | nginx auth bloquea Hermes | Media | Alto | Hermes envía API key en header. Verificar config Hermes provider |
| R8 | Audio no soportado en E4B | Baja | Medio | Verificar en Fase 3. Si falla, documentar como "text+vision only" |

---

## Checklist de Verificación Final

- [ ] `mistralrs serve` arranca sin error
- [ ] `/v1/chat/completions` texto → 200 OK
- [ ] `/v1/chat/completions` con imagen → 200 OK, describe imagen
- [ ] `/v1/chat/completions` con audio → 200 OK (o documentar como no soportado)
- [ ] `/v1/chat/completions` con `stream:true` → SSE chunks
- [ ] **embed_tokens_per_layer cuantizado por ISQ** → VRAM ~5.3 GB (o documentar Escenario B)
- [ ] VRAM idle ≤ 5.5 GB (Escenario A) / ≤ 9.5 GB (Escenario B)
- [ ] VRAM multimodal ≤ 7.0 GB (Escenario A)
- [ ] RAM proceso ≤ 200 MB
- [ ] nginx auth funciona (401/200)
- [ ] systemd `gemma4-rs.service` auto-restart OK
- [ ] Accesible desde LAN (192.168.1.x)
- [ ] Hermes gateway conecta correctamente
- [ ] Benchmarks documentados
- [ ] README actualizado
- [ ] Git commit + push

---

## Comandos de Rollback

```bash
# Si todo falla, volver a Phase 1:
systemctl --user stop gemma4-rs.service
systemctl --user disable gemma4-rs.service
echo PASSWORD | sudo -S systemctl stop nginx
systemctl --user start gemma4-api.service  # Restaurar servidor Python

# Verificar restauración:
curl -s http://localhost:11434/v1/models  # Debería responder
```
