# Use Cases Review — Phase 2

**Objetivo:** Verificar que los 8 casos de uso (UC-01 a UC-08) en USER_STORIES.md son completos, ejecutables y técnicamente correctos.

**Orden de review:** CC → OC → AGY

**Criterios de aprobación:**
1. Cada UC tiene Actor, Precondiciones, Flujo Principal, Postcondiciones
2. Flujos Alternativos donde aplica
3. Comandos curl/configuraciones son técnicamente correctos
4. Consistentes con PRD/TRD/SPEC
5. Ejecutables y testables

---

## Checklist por UC

| UC | Actor | Precondiciones | Flujo | Postcondiciones | Alternativos | Status |
|----|-------|----------------|-------|-----------------|---------------|--------|
| UC-01 | Hans (curl) | ✓ | ✓ | ✓ | ✓ (3A, 3B) | Pending |
| UC-02 | Hans (visión) | ✓ | ✓ | ✓ | ✓ (4A, 4B) | Pending |
| UC-03 | Hans (audio) | ✓ | ✓ | ✓ | Missing | Pending |
| UC-04 | Mobile app | ✓ | ✓ | ✓ | Missing | Pending |
| UC-05 | Hans (deploy) | ✓ | ✓ | ✓ | Missing | Pending |
| UC-06 | System (systemd) | ✓ | ✓ | ✓ | Missing | Pending |
| UC-07 | Hans (admin) | ✓ | ✓ | ✓ | Missing | Pending |
| UC-08 | Mobile app | ✓ | ✓ | ✓ | Missing | Pending |

**Gap:** UC-03 a UC-08 no tienen Flujos Alternativos documentados.
