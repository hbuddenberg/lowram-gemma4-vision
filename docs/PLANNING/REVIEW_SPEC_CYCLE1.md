# Review Cycle 1 — SPEC.md

**Fecha:** 2026-06-05  
**Documentos:** SPEC.md v1.0.0 → v1.1.0  
**Estado:** ✅ APPROVED (Cycle 2) — Consenso unanime

---

## Cycle 1: OC + AGY Review

### OC: ❌ REJECTED — 3 CRITICAL + 5 IMPORTANT + 5 MINOR

**CRITICAL:**
- C1: Error type `auth_error` vs `authentication_error` inconsistente
- C2: `[models.quantization.flash_attention]` FABRICADO (es Cargo feature, no TOML)
- C3: `[models.sequence]` FABRICADO (no existe en docs)

**IMPORTANT:**
- I1: `in_situ_quant = "q4k"` → debería ser `"4"` en TOML
- I2: nginx 413/429 devuelven HTML no JSON (SPEC se contradecía)
- I3: `/health` 503 response FABRICADO (siempre 200)
- I4: P50 latency 15s optimista para RTX 3060
- I5: Model ID inconsistente (`gemma4-heretic` vs `gemma-4-e4b-heretic`)

### AGY: ❌ REJECTED — 3 CRITICAL + 7 IMPORTANT + 3 MINOR
- Confirmed todos los CRITICAL de OC
- Nuevos: I6 (tabla parámetros documenta campos fabricados), I7 (`kind = "quantized"` cuestionable)
- Nuevos MINOR: M2 (auth attribution a mistral.rs incorrecto), M3 (`isq_q_group_size` no documentado)

---

## Cycle 2: Verificación tras correcciones (v1.1.0)

### OC: ✅ APPROVED
- Verificó las 8 correcciones sistemáticamente
- Sin nuevos errores

### AGY: ✅ APPROVED
- Verificó los 13 items (C1-C3, I1-I7, M2-M3)
- Confirmó hallazgos propios corregidos
- Sin nuevos issues
