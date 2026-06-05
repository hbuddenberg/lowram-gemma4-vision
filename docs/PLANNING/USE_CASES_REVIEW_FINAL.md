# Use Cases Review — Phase 2 (FINAL)

**Fecha:** 2026-06-05  
**Revisores:** CC (Claude Code) → OC (OpenCode) → AGY (Gemini)  
**Veredicto:** 4 PASS, 4 FAIL → 4 PASS tras correcciones

---

## Proceso de Review

**Ciclo 1 (CC):** 6 CRITICAL + 3 IMPORTANT + 5 MINOR → 2 PASS (UC-01, UC-02), 6 FAIL (UC-03-08)  
**Ciclo 2 (OC):** Confirmó CC + agregó 3 CRITICAL + 3 IMPORTANT propios  
**Ciclo 3 (AGY):** Reclasificó varios findings, dio veredicto final

---

## Veredicto Final (post-correcciones)

| UC | Status | Severidad Máx. (antes) | Correcciones Aplicadas | Status Final |
|----|--------|------------------------|----------------------|--------------|
| **UC-01** | ✅ PASS | — | 0 | ✅ PASS |
| **UC-02** | ✅ PASS | — | 0 | ✅ PASS |
| **UC-03** | ❌ FAIL | IMPORTANT | 2 (Content-Type + alt flows) | ✅ PASS |
| **UC-04** | ✅ PASS* | MINOR | 0 | ✅ PASS (no bloquea) |
| **UC-05** | ✅ PASS* | MINOR | 0 | ✅ PASS (no bloquea) |
| **UC-06** | ❌ FAIL | IMPORTANT | 3 (StartLimitBurst + linger + alt flows) | ✅ PASS |
| **UC-07** | ❌ FAIL | IMPORTANT | 3 (postcond + /v1/models + alt flows) | ✅ PASS |
| **UC-08** | ❌ FAIL | IMPORTANT | 3 (postcond + listen 0.0.0.0 + alt flows) | ✅ PASS |

**UC-04 y UC-05 marcados con * tienen gaps de documentación (alt flows ausentes) pero no bloquean implementación.**

---

## Correcciones Aplicadas (11 total)

### UC-03 (2 correcciones)
1. **IMPORTANT:** Agregado `-H "Content-Type: application/json"` al curl
2. **MINOR:** Agregada sección Flujos Alternativos (audio corrupto → 400, FFmpeg missing → 500, OOM → 500, 413 tamaño)

### UC-06 (3 correcciones)
1. **IMPORTANT:** Agregado `StartLimitBurst=5` + `StartLimitIntervalSec=300` a precondiciones
2. **IMPORTANT:** Agregado `loginctl enable-linger hans` a precondiciones
3. **MINOR:** Agregada sección Flujos Alternativos (restart limit, GPU unavailable, VRAM leak)

### UC-07 (3 correcciones)
1. **IMPORTANT:** Agregada sección Postcondiciones (alerta enviada, logs accesibles, VRAM documentada, servicio restaurado)
2. **IMPORTANT:** Paso 3 mejorado: curl connection refused (no response != 200), + paso 4 con `/v1/models`
3. **MINOR:** Agregada sección Flujos Alternativos (modelo descargado, health timeout, VRAM excede)

### UC-08 (3 correcciones)
1. **IMPORTANT:** Agregada sección Postcondiciones (200 OK, auth validada, contenido correcto, LAN only)
2. **MINOR:** Precondición agregada: nginx con `listen 0.0.0.0:80`
3. **MINOR:** Agregada sección Flujos Alternativos (401 auth, 429 rate limit, 502 backend down, 504 timeout, 413 body size)

---

## Decisions Clave del Review

1. **Reclasificación de severity:** AGY reclasificó "flujos alternativos ausentes" de CRITICAL → IMPORTANT (es gap de documentación, no defecto de diseño)

2. **UC-03 `image_url` para audio:** OC lo reportó como CRITICAL, AGY verificó contra SPEC §1.2 y confirmó que es la API correcta de mistral.rs — reclasificado a MINOR (confuso pero correcto)

3. **`install.sh` no existe:** OC reportó como CRITICAL, AGY reclasificó a MINOR (es artefacto que se creará en implementación)

4. **UC-04/05 marcados como PASS* con asterisco:** Tienen MINOR gaps (alt flows ausentes) pero no bloquean implementación — los flujos principales son completos y correctos

---

## Resumen de Findings por Agente

### CC (6 CRITICAL + 3 IMPORTANT + 5 MINOR)
- Detectó correctamente todos los UC faltantes de flujos alternativos
- Encontró el único bug funcional real: UC-03 sin Content-Type header

### OC (confirmó CC + 3 CRITICAL + 3 IMPORTANT)
- Confirmó la mayoría de findings CC
- Agregó perspectiva infrastructure/ops: StartLimitBurst, linger, RAM constraints
- OC-CRIT-01 (`image_url` para audio) reclasificado por AGY tras verificar SPEC

### AGY (veredicto final)
- Reclasificó varios findings de CRITICAL → IMPORTANT/MINOR
- Verificó consistencia contra SPEC/TRD/PRD
- Dio veredicto final: 4 PASS, 4 FAIL con lista de 11 correcciones requeridas

---

## Próximos Pasos

1. **Commit USER_STORIES.md v1.2.0** con 11 correcciones aplicadas
2. **Todos los 8 UC están ahora listos para implementación**
3. **Comenzar Fase 0** (instalar deps + compilar mistral.rs)

---

## Referencias

- USER_STORIES.md v1.2.0 (12 US + 8 UC completos)
- PRD.md v2.2.0 (con §4 User Stories)
- TRD.md v2.1.0 (arquitectura técnica)
- SPEC.md v1.1.0 (especificación API)
