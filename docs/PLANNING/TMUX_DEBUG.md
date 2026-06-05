# Diagnosis: tmux "can't find pane" en agent-thread-dispatch

**Fecha:** 2026-06-05  
**Contexto:** Spawn de 3 agentes (AGY, OC, CC) via tmux para validation CLI

---

## Error

```
can't find pane: agy-phase2
can't find pane: oc-phase2
can't find pane: cc-phase2
```

Error de tmux: la sesion ya no existe cuando el loop de monitoreo intenta capturar output.

---

## Causa Raiz

tmux **destruye automaticamente** la sesion cuando el proceso principal sale. Los 3 agentes salieron antes del primer polling (15s), asi que `tmux capture-pane` encontro paneles inexistentes.

---

## Bug por Agente

### AGY (Gemini CLI)

**Sintaxis usada (incorrecta):**
```bash
tmux new-session -d -s agy-phase2 "gemini -p --yolo < /tmp/agy-task.md 2>&1 | tee /tmp/agy-output.log"
```

**Problema:** `gemini -p` requiere argumento posicional (string). NO lee de stdin. El redirect `< /tmp/agy-task.md` pasa el contenido del archivo como stdin del shell, NO como argumento del CLI.

```
Error: "Not enough arguments following: p"
```

**Fix:**
```bash
# Leer el prompt en una variable, pasar como argumento
PROMPT=$(cat /tmp/agy-task.md)
tmux new-session -d -s agy-phase2 "gemini -p '$PROMPT' 2>&1 | tee /tmp/agy-output.log"

# O mejor: prompt directo inline
tmux new-session -d -s agy-phase2 "gemini -p 'Tu tarea aqui...' 2>&1 | tee /tmp/agy-output.log"
```

---

### OC (OpenCode)

**Sintaxis usada:**
```bash
tmux new-session -d -s oc-phase2 "opencode run 'prompt...' 2>&1 | tee /tmp/oc-output.log"
```

**Problema:** OC funciono parcialmente pero denego permisos para directorios fuera del workspace:
```
! permission requested: external_directory (/home/hbuddenberg/models/gemma4-heretic/*); auto-rejecting
```

**Fix:** Incluir el directorio del modelo en el workspace de OC o usar `cd ~/lowram-gemma4-vision && opencode run --include ~/models/gemma4-heretic`.

---

### CC (Claude Code)

**Sintaxis usada:**
```bash
tmux new-session -d -s cc-phase2 "claude -p 'prompt...' --output-format json ... 2>&1 | tee /tmp/cc-output.log"
```

**Problema:** El log tiene 0 bytes. Probablemente:
1. El prompt inline era demasiado largo y quoting se rompio en tmux
2. Claude salio antes de que `tee` flusheara a disco
3. `--output-format json` requiere procesamiento extra que puede fallar con pipe

**Fix:**
```bash
# Redirigir stderr por separado para capturar errores
tmux new-session -d -s cc-phase2 "claude -p 'prompt' 2>&1 | tee /tmp/cc-output.log"

# O usar --output-format text (mas confiable con pipe)
```

---

## Fix General: Evitar "can't find pane"

### Opcion 1: `set-option remain-on-exit` (recomendado)

```bash
# Mantener sesion viva despues de que el proceso sale
tmux new-session -d -s agy-phase2
tmux set-option -t agy-phase2 remain-on-exit on
tmux send-keys -t agy-phase2 "gemini -p 'prompt' 2>&1 | tee /tmp/agy-output.log" Enter

# La sesion PERMANECE despues de salir el proceso
# tmux capture-pane seguira funcionando
```

### Opcion 2: Usar un wrapper script

```bash
# Crear wrapper que no sale hasta que tee flushee
cat > /tmp/run-agent.sh << 'WRAPPER'
#!/bin/bash
eval "$@" 2>&1 | tee /tmp/agent-output.log
sleep 1  # Dar tiempo a tee para flush
WRAPPER
chmod +x /tmp/run-agent.sh

tmux new-session -d -s agy-phase2 "/tmp/run-agent.sh gemini -p 'prompt'"
```

### Opcion 3: Verificar sesion antes de capture

```python
# En el loop de monitoreo
r = terminal(f"tmux has-session -t {sess_name} 2>&1", timeout=3)
if r.get("exit_code") != 0:
    # Session ended — read log file instead
    try:
        final = Path(f"/tmp/{sess_name}-output.log").read_text()
    except FileNotFoundError:
        final = "No output captured"
    # Send final result
    ...
    continue
```

---

## Recomendacion

Usar **Opcion 1** (remain-on-exit) + **verificar sesion** (Opcion 3) como defensa en profundidad. Actualizar el skill `agent-thread-dispatch` para incluir estos fixes.
