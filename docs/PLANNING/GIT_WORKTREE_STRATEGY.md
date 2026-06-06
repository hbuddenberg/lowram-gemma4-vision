# Git Worktree Strategy — Phase 2 Implementation

**Pregunta:** ¿Usar git worktree para paralelismo entre agentes?

---

## Análisis del Workflow

### Fases Secuenciales (Días 1-5)

```
Día 1: OC (Fase 0) → AGY (Fase 1)         → 1 agente a la vez
Día 2: CC (Fase 2) → AGY (Fase 3)         → 1-2 agentes en serie
Día 3: OC (Fase 4) → OC (Fase 5)          → 1 agente
Día 4: CC (Fase 6)                         → 1 agente
Día 5: OC (Fase 7)                         → 1 agente
```

**Veredicto:** **NO necesario git worktree** para Días 1-5. Son secuenciales por diseño.

---

### Buffer Days (Días 6-10) — OPORTUNIDAD DE PARALELISMO

```
Día 6-10: CC (RAG post-migración)
          Otros agentes: disponibles para otros tasks
```

**Aquí SÍ tiene sentido git worktree:**
- CC trabaja en RAG en worktree `phase2-rag`
- AGY podría explorar optimizaciones/metrics en worktree `phase2-metrics`
- OC podría preparar infraestructura futura en worktree `phase2-infra`

---

## Estrategia Recomendada

### Opción A: Simple (NO worktree para Días 1-5)

**Ventajas:**
- Workflow simple, un solo working directory
- Commits secuenciales, historia lineal
- Menos complejidad

**Desventajas:**
- Sin paralelismo real
- Si un agente necesita hotfix durante implementación, bloquea a otros

**Implementación:**
```bash
# Todos trabajan en ~/lowram-gemma4-vision/ (branch phase2)
# Commits secuenciales por fase
git add -A && git commit -m "Fase X completada"
```

---

### Opción B: Worktree para Buffer Days (RECOMENDADO)

**Ventajas:**
- Paralelismo real en Días 6-10
- CC en RAG sin bloquear a otros
- Otros agentes pueden explorar mejoras en paralelo
- Histories separadas por feature

**Desventajas:**
- Complejidad añadida
- Merge eventual necesario

**Implementación:**

#### Día 5 (post-release)
```bash
cd ~/lowram-gemma4-vision

# Crear worktrees para paralelismo
git worktree add ../lowram-gemma4-rag phase2-rag      # CC: RAG
git worktree add ../lowram-gemma4-metrics phase2-metrics  # AGY: metrics/opt
git worktree add ../lowram-gemma4-infra phase2-infra    # OC: infra futura
```

#### Días 6-10 (paralelo)
```bash
# CC trabaja en RAG
cd ~/lowram-gemma4-rag
# ... desarrolla RAG ...
git add -A && git commit -m "RAG: FAISS index + retrieval"

# AGY explora optimizaciones
cd ~/lowram-gemma4-metrics
# ... explora ISQ refinements, cuantización, etc ...
git add -A && git commit -m "metrics: VRAM profiling, ISQ refinements"

# OC prepara infra futura
cd ~/lowram-gemma4-infra
# ... explora monitoring, alerting, etc ...
git add -A && git commit -m "infra: monitoring stack + alerts"
```

#### Merge eventual (Día 10+)
```bash
cd ~/lowram-gemma4-vision  # main worktree (phase2)

# Merge cada worktree
git merge phase2-rag       # Merge RAG
git merge phase2-metrics   # Merge metrics
git merge phase2-infra     # Merge infra

# Resolver conflicts si hay
git push origin phase2
```

---

### Opción C: Worktree desde Día 1 (NO RECOMENDADO)

**Ventajas:**
- Máximo paralelismo desde el inicio

**Desventajas:**
- **Sobreingeniería** para fases secuenciales
- Complejidad innecesaria
- Merge conflicts probables
- Las fases 0-7 son INTENCIONALMENTE secuenciales

**NO recomendado.**

---

## Veredicto Final

| Fase | Estrategia Git | Razón |
|------|----------------|-------|
| **Días 1-5** | **NO worktree** | Fases secuenciales por diseño |
| **Días 6-10** | **SÍ worktree** | Paralelismo real para RAG + exploración |

---

## Workflow Recomendado

### Fases 0-7 (Días 1-5)
```bash
# Single worktree, workflow simple
cd ~/lowram-gemma4-vision  # branch phase2

# Cada fase:
git checkout phase2
# ... trabajar ...
git add -A && git commit -m "Fase X: [descripción]"
git push origin phase2
```

### Buffer Days (Días 6-10)
```bash
# Crear worktrees
cd ~/lowram-gemma4-vision
git worktree add ../lowram-gemma4-rag phase2-rag      # CC: RAG
git worktree add ../lowram-gemma4-metrics phase2-metrics  # AGY: metrics
git worktree add ../lowram-gemma4-infra phase2-infra    # OC: infra

# Trabajar en paralelo, merge al final
```

---

## Alternativa: Branches sin Worktree

Si prefieres evitar git worktree pero permitir paralelismo en buffer days:

```bash
# Día 5
git branch phase2-rag
git branch phase2-metrics
git branch phase2-infra

# Días 6-10 (cada agente en su branch)
git checkout phase2-rag       # CC
# ... work ...
git checkout phase2-metrics    # AGY
# ... work ...
git checkout phase2-infra      # OC
# ... work ...

# Merge al final (en phase2)
git checkout phase2
git merge phase2-rag
git merge phase2-metrics
git merge phase2-infra
```

**Ventaja:** Sin múltiples directorios
**Desventaja:** Cambio de branches manual (más propenso a errores)

---

## Resumen

**Recomendación: Opción B (worktree solo para buffer days)**

| Momento | Estrategia | Comando |
|---------|------------|---------|
| **Días 1-5** | NO worktree | Trabajar en `~/lowram-gemma4-vision/` (phase2) |
| **Día 5** | Crear worktrees | `git worktree add ../lowram-gemma4-rag phase2-rag` |
| **Días 6-10** | SÍ worktree | Paralelismo real en 3 directorios |
| **Día 10+** | Merge | `git merge phase2-rag && git merge phase2-metrics && git merge phase2-infra` |

---

**¿Prefieres Opción A (simple, sin worktree) o Opción B (worktree para buffer days)?**
