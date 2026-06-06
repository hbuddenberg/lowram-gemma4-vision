# Distribución de Implementación — Phase 2

**Duración total:** 5 días (Fases 0-7) + 5 días buffer (RAG opcional)  
**Estrategia:** Distribución por especialidades de agentes

---

## Agentes y Roles

| Agente | Especialidad | Fortalezas |
|--------|-------------|------------|
| **CC (Claude Code)** | Ingeniero ML senior | Modelos, inferencia, benchmarks, testeo |
| **OC (OpenCode)** | Ingeniero infraestructura | nginx, systemd, seguridad, deployment |
| **AGY (Gemini)** | Ingeniero Rust/ML senior | Compilación Rust, debugging CUDA, multimodal |

---

## Distribución por Fases

### 📅 DÍA 1

| Fase | Tareas | Agente Principal | Agente Secundario | Duración |
|------|--------|-----------------|-------------------|----------|
| **Fase 0** | Prerrequisitos (cudnn, nginx, verify CUDA/Rust/GPU) | **OC** | — | 1 hora |
| **Fase 1** | Compilar mistral.rs (cargo install con features) | **AGY** | CC | 30-60 min |

**Entregables Día 1:**
- Deps instaladas (cudnn, nginx)
- `mistralrs --version` funciona
- Binario en `~/.cargo/bin/mistralrs`

---

### 📅 DÍA 2

| Fase | Tareas | Agente Principal | Agente Secundario | Duración |
|------|--------|-----------------|-------------------|----------|
| **Fase 2** | Verificar carga modelo (ISQ Q4K, texto, **embed_tokens_per_layer**) | **CC** | AGY | 5 min |
| **Fase 3** | Verificar multimodal (visión, audio, streaming) | **AGY** | CC | 15 min |

**Entregables Día 2:**
- Modelo carga sin error
- **⚠️ Verificación crítica:** embed_tokens_per_layer cuantizado (VRAM ≤ 5.5 GB vs 9.5 GB)
- Visión: 200 OK, describe imagen
- Audio: 200 OK (o documentado como no soportado)
- Streaming: SSE chunks funcionan

**Gate de decisión:** Si embed_tokens_per_layer NO se cuantiza (VRAM > 9 GB), evaluar:
- Offload CPU/RAM de ese tensor
- Modo texto-only con headroom limitado
- Documentar como Escenario B (10.8 GB 🔴)

---

### 📅 DÍA 3

| Fase | Tareas | Agente Principal | Agente Secundario | Duración |
|------|--------|-----------------|-------------------|----------|
| **Fase 4** | Configurar nginx (auth, rate limiting, reverse proxy) | **OC** | — | 30 min |
| **Fase 5** | Systemd service (auto-start, crash recovery) | **OC** | — | 15 min |

**Entregables Día 3:**
- nginx reverse proxy configurado con auth
- `http://localhost/v1/*` → `127.0.0.1:8080`
- Auth funciona (401/200)
- `gemma4-rs.service` creado y running
- Auto-restart funciona (test crash + recovery)

---

### 📅 DÍA 4

| Fase | Tareas | Agente Principal | Agente Secundario | Duración |
|------|--------|-----------------|-------------------|----------|
| **Fase 6** | Benchmarks (VRAM, RAM, latencia, tokens/s) | **CC** | AGY | 30 min |

**Entregables Día 4:**
- Tabla comparativa Phase 1 vs Phase 2
- VRAM idle ≤ 5.5 GB (Escenario A) / ≤ 9.5 GB (Escenario B)
- VRAM multimodal ≤ 7.0 GB (Escenario A)
- RAM proceso ≤ 200 MB
- Latencia text ≤ 15s (128 tokens)
- RAM libre ≥ 1.5 GB (vs 380 MB Phase 1)

---

### 📅 DÍA 5

| Fase | Tareas | Agente Principal | Agente Secundario | Duración |
|------|--------|-----------------|-------------------|----------|
| **Fase 7** | Cleanup, docs, release (archivar Phase 1, actualizar Hermes, README) | **OC** | CC | 1 hora |

**Entregables Día 5:**
- Servicio viejo (`gemma4-api.service`) deshabilitado
- Hermes config actualizado (`http://localhost/v1`)
- GitHub commit + push
- README con instrucciones de deploy
- Screenshot VRAM post-migración
- Test end-to-end Discord

---

### 📅 DÍAS 6-10 (BUFFER — OPCIONAL)

| Fase | Tareas | Agente Principal | Duración |
|------|--------|-----------------|----------|
| **Post-Phase 2** | RAG post-migración (FAISS, búsqueda vectorial) | **CC** | 1-5 días |

---

## Comunicación y Coordinación

### Checkpoints Diarios (Cada fin de día)

| Día | Punto de Verificación | Responsable |
|-----|----------------------|-------------|
| **1** | Deps OK + mistral.rs compilado | AGY reporta a todos |
| **2** | **CRÍTICO:** embed_tokens_per_layer ISQ OK? | CC reporta decisión (Escenario A vs B) |
| **3** | nginx + systemd OK | OC reporta |
| **4** | Benchmarks cumplen targets | CC reporta tabla comparativa |
| **5** | Release listo, Hermes funcionando | OC reporta |

### Rollback Plan

Si algo falla, **OC** ejecuta rollback:
```bash
systemctl --user stop gemma4-rs.service
systemctl --user disable gemma4-rs.service
echo PASSWORD | sudo -S systemctl stop nginx
systemctl --user start gemma4-api.service  # Restaurar Phase 1
```

---

## Resumen de Responsabilidades

### CC (Claude Code) — 40% del trabajo
- **Fase 2:** Verificar carga modelo (ISQ, texto, embed_tokens_per_layer)
- **Fase 3:** Verificar multimodal (soporte AGY en visión/audio)
- **Fase 6:** Benchmarks completos + tabla comparativa
- **Fase 7:** Soporte en cleanup + test Discord

### OC (OpenCode) — 35% del trabajo
- **Fase 0:** Prerrequisitos (cudnn, nginx, verify deps)
- **Fase 4:** nginx completo (auth, rate limiting, reverse proxy, LAN access)
- **Fase 5:** systemd completo (service, auto-start, crash recovery)
- **Fase 7:** Cleanup principal (archivar Phase 1, actualizar Hermes, README)

### AGY (Gemini) — 25% del trabajo
- **Fase 1:** Compilar mistral.rs (cargo install, debugging CUDA/compilation)
- **Fase 2:** Soporte CC en verificación modelo (debugging si falla)
- **Fase 3:** Verificar multimodal principal (visión, audio, streaming)

---

## Comandos Críticos por Agente

### CC
```bash
# Fase 2: Verificar embed_tokens_per_layer
mistralrs run --quant q4k -m ~/models/gemma4-heretic
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # ≤ 5.5 GB = OK

# Fase 6: Benchmarks
time curl ... (latencia)
nvidia-smi (VRAM)
ps -p $(pgrep mistralrs) -o rss= (RAM)
```

### OC
```bash
# Fase 0: Prerrequisitos
echo PASSWORD | sudo -S pacman -S cudnn nginx
rustc --version  # ≥ 1.96.0
nvidia-smi  # RTX 3060, 12288 MiB

# Fase 4: nginx
echo PASSWORD | sudo -S htpasswd -c /etc/nginx/.htpasswd hbuddenberg
echo PASSWORD | sudo -S nginx -t
echo PASSWORD | sudo -S systemctl enable --now nginx

# Fase 5: systemd
systemctl --user daemon-reload
systemctl --user enable --now gemma4-rs.service

# Fase 7: Rollback
systemctl --user stop gemma4-rs.service
systemctl --user start gemma4-api.service  # Si falla
```

### AGY
```bash
# Fase 1: Compilar
export PATH=/opt/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH
cargo install mistralrs-cli --features "cuda flash-attn cudnn"
mistralrs --version

# Fase 2: Debug si falla carga
RUST_LOG=debug mistralrs run --quant q4k -m ~/models/gemma4-heretic 2>&1 | tee /tmp/mistralrs-debug.log

# Fase 3: Multimodal test
IMG_B64=$(base64 -w0 /tmp/test_img.jpg)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"gemma4-heretic\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"¿De qué color es esta imagen?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${IMG_B64}\"}}]}],\"max_tokens\":50}"
```

---

## Timeline Resumido

```
Día 1: Fase 0 (OC) + Fase 1 (AGY)          → mistral.rs compilado
Día 2: Fase 2 (CC) + Fase 3 (AGY)          → Modelo + multimodal OK
Día 3: Fase 4 (OC) + Fase 5 (OC)          → nginx + systemd OK
Día 4: Fase 6 (CC)                         → Benchmarks + tabla
Día 5: Fase 7 (OC)                         → Release listo
Días 6-10: Buffer (CC)                     → RAG opcional
```

---

## Gate Decision Crítico (Fin Día 2)

**Pregunta:** ¿ISQ cuantiza `embed_tokens_per_layer`?

| VRAM idle | Escenario | Decisión |
|-----------|-----------|-----------|
| ≤ 5.5 GB | **A** ✅ | Continuar Fase 3-7 (headroom 2.3 GB para multimodal) |
| 9.5 GB | **B** 🔴 | Evaluar offload CPU/RAM o modo texto-only |

Si Escenario B:
1. CC prueba offload de `embed_tokens_per_layer` a CPU/RAM
2. Si offload falla → modo texto-only (sin visión/audio)
3. Documentar en README como "text-only mode"

---

## ¿Listos para Empezar?

**Comando de inicio:**
```bash
# OC empieza Fase 0
cd ~/lowram-gemma4-vision
git checkout phase2
# Ejecutar tareas Fase 0 (IMPLEMENTATION.md líneas 11-33)
```

¿Arrancamos Fase 0? 🚀
