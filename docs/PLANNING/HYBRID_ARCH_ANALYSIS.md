# Hybrid Architecture Analysis: GGUF (llama.cpp) + Transformers for Gemma 4

> **Date:** 2026-06-04 | **Context:** RTX 3060 12GB (TB3) | **Model:** igorls/gemma-4-E4B-it-heretic

---

## 1. Executive Summary

**Verdict: NOT VIABLE in the short term. Partially viable mid-term with significant caveats.**

The proposed hybrid architecture — using GGUF/llama.cpp for text+tools and full transformers for vision/audio — is architecturally sound in concept but **blocked by a critical upstream dependency**: llama.cpp does not currently support the `gemma4` architecture (`model_type: gemma4`, `Gemma4ForConditionalGeneration`). Gemma 3 is supported, but Gemma 4's architecture (released ~May 2026) has not yet been added to llama.cpp's model converter or inference engine.

---

## 2. Current System Analysis

### 2.1 Existing Setup (Single Pipeline)

```
┌─────────────────────────────────────────────────────┐
│  FastAPI (Python)                                    │
│  ├── POST /v1/chat/completions                      │
│  ├── Auth + Rate Limit + Key Management (SQLite)    │
│  └── Transformers Pipeline (NF4 quantized)          │
│      ├── Language Model: 42 layers, 2560d → NF4    │
│      ├── Vision Tower: 16 layers, 768d → fp16      │
│      ├── embed_vision: 768→2560 projection → fp16  │
│      ├── Audio Tower: NOT LOADED                    │
│      └── lm_head: 2560→262144 → fp16               │
│                                                      │
│  VRAM: 9.3GB / 11.6GB (RTX 3060 12GB)              │
│  Throughput: ~6.8 tok/s text, ~7.2 tok/s vision     │
│  Context Limit: 128 tokens max (self-imposed OOM)   │
│  Concurrent Requests: 1 (asyncio.Semaphore)         │
└─────────────────────────────────────────────────────┘
```

### 2.2 Identified Problems

| Problem | Root Cause | Impact |
|---------|-----------|--------|
| 128 tokens max output | VRAM pressure (9.3/11.6GB) + context truncation to 4 messages | Unusable for most agentic tasks |
| Single concurrent request | `asyncio.Semaphore(1)` due to GPU memory | Queueing on burst traffic |
| No audio support | Audio tower skipped entirely | Missing multimodal capability |
| ~90s cold start | Full model load with quantization | Poor UX on restart |
| Python GIL | Sync inference blocks event loop (mitigated by executor) | Latency spikes |

---

## 3. Proposed Hybrid Architecture

### 3.1 Design

```
┌──────────────────────────────────────────────────────────────────┐
│                    Rust Router / API Gateway                      │
│  (axum | actix-web)                                              │
│                                                                   │
│  ┌──────────────────┐     ┌─────────────────────────────────┐   │
│  │  Request Router   │     │  Input Classifier               │   │
│  │  (content-type)   │     │  • has_images? → vision path    │   │
│  │                   │     │  • has_audio? → audio path       │   │
│  │                   │     │  • text-only? → fast path        │   │
│  │                   │     │  • tools? → fast path + grammar  │   │
│  └────────┬──────────┘     └──────────────┬──────────────────┘   │
│           │                               │                       │
│     ┌─────┴───────────────┐               │                       │
│     │                     │               │                       │
│     ▼                     ▼               ▼                       │
│ ┌──────────────┐  ┌────────────────────────────────────────┐     │
│ │  TEXT PATH   │  │  MULTIMODAL PATH                       │     │
│ │  (llama.cpp) │  │  (transformers full)                   │     │
│ │              │  │                                        │     │
│ │ GGUF Q4_K_M  │  │ NF4 text + fp16 vision/audio          │     │
│ │ ~3-4GB VRAM  │  │ 9.3GB VRAM (current)                  │     │
│ │ 32K context  │  │ Or: full fp16 if VRAM freed           │     │
│ │ tool grammars │  │                                        │     │
│ │ ~15-25 tok/s │  │ ~6.8-7.2 tok/s                        │     │
│ └──────────────┘  └────────────────────────────────────────┘     │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Shared Services (Rust)                                   │    │
│  │  ├── Auth (SQLite/key management)                        │    │
│  │  ├── Rate Limiting (token bucket)                        │    │
│  │  ├── Usage Analytics                                     │    │
│  │  ├── Health / Metrics (Prometheus)                       │    │
│  │  └── OpenAI-compatible response formatting               │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 VRAM Budget Analysis

| Configuration | VRAM Usage | Feasible? |
|---|---|---|
| Current (NF4 + fp16 vision) | 9.3GB / 11.6GB | ✅ (but 128 tok limit) |
| GGUF Q4_K_M text-only (~4B params) | ~2.5-3.5GB | ✅ |
| Transformers multimodal (when loaded) | 9.3GB | ✅ (unload GGUF first) |
| Both loaded simultaneously | ~12.8GB | ❌ OOM |
| **Hot-swap: load/unload per request** | ~3.5-9.3GB | ⚠️ ~30-60s model swap time |

### 3.3 The Critical Blocker

**llama.cpp does NOT support `model_type: gemma4` (Gemma4ForConditionalGeneration).**

Evidence:
- llama.cpp's `convert_hf_to_gguf.py` supports gemma, gemma2, and **gemma3** architectures
- Gemma 4 uses a new architecture (`Gemma4ForConditionalGeneration`) with a 262K vocabulary (SigLip-based)
- The model's `config.json` declares `model_type: gemma4` — unrecognized by llama.cpp's converter
- llama.cpp supports multimodal via separate mmproj files for LLaVA/LLaMA-based models, but Gemma 4's `embed_vision` projector architecture is custom

**Timeline estimate:** Based on historical patterns (Gemma 3 support took ~3-4 weeks after release), Gemma 4 support in llama.cpp would likely arrive **3-6 weeks after model release** if community demand is high.

---

## 4. Viability Analysis

### 4.1 Advantages (if unblocked)

| Advantage | Detail |
|-----------|--------|
| **2-4x text throughput** | llama.cpp GGUF Q4_K_M on RTX 3060 typically achieves 15-25 tok/s vs current 6.8 tok/s |
| **Longer context** | GGUF with flash attention can handle 8K-32K context vs current 128-token truncation |
| **Tool calling via grammars** | llama.cpp supports JSON grammars for structured tool output — no custom parsing needed |
| **Rust backend performance** | axum/actix async runtime eliminates Python GIL, better concurrent connection handling |
| **Lower baseline VRAM** | Text-only GGUF ~3.5GB → 8GB free for vision when needed |
| **Independent scaling** | Text and multimodal paths can be optimized independently |

### 4.2 Disadvantages

| Disadvantage | Detail |
|---|---|
| **❌ llama.cpp doesn't support Gemma 4** | FATAL: Cannot convert or run the model today |
| **Model hot-swap latency** | Swapping between GGUF and transformers takes 30-60s (load/unload VRAM) |
| **Dual model maintenance** | Need to keep both GGUF and safetensors versions synced |
| **Quality variance in quantization** | GGUF Q4_K_M ≠ bitsandbytes NF4 — different quantization artifacts |
| **Tool call format mismatch** | Gemma 4 uses custom `<|tool_call|>` markers that llama.cpp wouldn't handle natively |
| **Increased system complexity** | Router + 2 backends + swap logic + state management |
| **Audio support still missing** | Neither path supports audio today |

### 4.3 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Gemma 4 never supported in llama.cpp | Low | Critical | Use Gemma 3 4B GGUF as text-only fallback |
| Hot-swap causes VRAM fragmentation | High | Medium | Use CUDA `expandable_segments:True` (already set) |
| Tool call parsing differs between backends | High | Medium | Normalize via Rust router |
| Context window mismatch between paths | Medium | Low | Router enforces per-path limits |
| User sends mixed text+vision in same request | High | High | Clear routing rules: any image → multimodal path |

---

## 5. Implementation Strategy (Phased)

### Phase 0: Unblocking (Pre-requisite)

**Wait for or contribute Gemma 4 support in llama.cpp.**

```bash
# Check for Gemma 4 support in llama.cpp convert scripts
python convert_hf_to_gguf.py ~/models/gemma4-heretic --outtype f16 --outfile gemma4-e4b-f16.gguf
# Currently FAILS: "Model architecture 'gemma4' is not supported"

# When available:
python convert_hf_to_gguf.py ~/models/gemma4-heretic --outtype q4_k_m --outfile gemma4-e4b-q4km.gguf
```

**Alternative:** Use a Gemma 3 4B GGUF as text-only model while keeping Gemma 4 transformers for vision. This gives most of the text performance benefits with a slight quality tradeoff.

### Phase 1: Rust API Gateway (4-6 weeks)

Replace FastAPI with Rust (axum) router:

```
Rust Binary (single process)
├── axum HTTP server
│   ├── POST /v1/chat/completions
│   ├── GET  /v1/models
│   ├── GET  /health
│   ├── Admin endpoints (keys, usage)
│   └── SSE streaming support
├── Router module
│   ├── Input classifier (detect images/audio/text-only)
│   ├── Backend selector
│   └── Model lifecycle (load/unload/swap)
├── Auth module
│   ├── SQLite via rusqlite
│   ├── In-memory key cache + hash index (from current Python impl)
│   └── Rate limiter (token bucket)
└── Backend connectors
    ├── llama.cpp FFI (via llama-cpp-2 crate or C binding)
    └── Python subprocess (transformers via gRPC/Unix socket)
```

**Key Rust crates:**
- `axum` + `tokio` — async HTTP
- `rusqlite` — SQLite (drop-in replacement for current schema)
- `llama-cpp-2` — Rust bindings for llama.cpp
- `serde_json` — OpenAI-compatible response serialization
- `tower` — middleware layers (auth, rate limit, logging)

### Phase 2: Text-Only Fast Path (2-3 weeks after Phase 0)

```
Request flow (text-only):
Client → Rust Router → detect: text only
       → llama.cpp (GGUF Q4_K_M, ~3.5GB VRAM)
       → 15-25 tok/s, 8K+ context
       → Rust formats OpenAI response
```

### Phase 3: Multimodal Path with Model Swap (3-4 weeks)

```
Request flow (vision):
Client → Rust Router → detect: has images
       → Unload GGUF model (if loaded)
       → Load transformers model (NF4 + fp16 vision, ~9.3GB VRAM)
       → Run inference via Python subprocess or PyO3 bridge
       → Return result
       → Optionally: unload transformers, reload GGUF
```

**Model swap strategies:**
1. **Lazy swap (recommended):** Keep current backend loaded, only swap on different modality request
2. **Timer-based:** Swap back to GGUF after N minutes of no multimodal requests
3. **Priority queue:** Queue text requests during multimodal inference, batch-swap afterward

### Phase 4: Optimization (2-3 weeks)

- Continuous batching for text requests via llama.cpp
- KV cache reuse for repeated system prompts
- Vision model warm-keeping (keep vision tower in VRAM, only swap language model)
- Prometheus metrics endpoint
- Structured tool calling via llama.cpp grammars

---

## 6. Alternative Approaches

### 6.1 Gemma 3 4B GGUF + Gemma 4 Vision (Hybrid Different Models)

**Viable TODAY.** Use Gemma 3 4B-it GGUF (Q4_K_M) for text/tools, and Gemma 4 E4B transformers for vision.

| Aspect | Detail |
|---|---|
| Text backend | Gemma 3 4B GGUF Q4_K_M — ~2.5GB VRAM, 15-25 tok/s |
| Vision backend | Gemma 4 E4B NF4 + fp16 — 9.3GB VRAM |
| Quality gap | Gemma 3 4B < Gemma 4 4B for text, but still very capable |
| Complexity | Same as proposed hybrid but available now |
| Tool calls | llama.cpp supports tool calling natively for Gemma 3 |

### 6.2 Pure Rust with Candle

Use HuggingFace's `candle` framework for inference entirely in Rust.

**Status:** Candle has `quantized-gemma` example but only for Gemma 1. No Gemma 4 support. No vision model support. Not viable.

### 6.3 Optimized Current Pipeline (Incremental)

Keep transformers but optimize:

| Optimization | Expected Gain |
|---|---|
| Increase `max_tokens` to 512-1024 | 4-8x output capacity (test if VRAM allows) |
| Speculative decoding | ~1.5-2x throughput (if draft model fits) |
| Flash Attention 2 | Reduce VRAM, allow longer context |
| KV cache quantization | ~20% VRAM savings |
| Torch.compile() | ~10-30% speedup after warmup |

**This is the RECOMMENDED immediate path** — lower risk, addresses the 128-token problem, and can be done today.

### 6.4 Two-Process Architecture (Python + llama.cpp)

Run llama.cpp server and FastAPI as separate processes, route via nginx/envoy:

```
nginx (port 443)
├── /v1/chat/completions → inspect body → 
│   ├── text-only → llama.cpp server (port 8081)
│   └── multimodal → FastAPI transformers (port 8082)
└── /v1/models → either backend
```

**Pros:** No Rust needed, simpler to implement, can use existing llama.cpp server binary.
**Cons:** Still blocked on Gemma 4 GGUF support, routing at nginx layer is awkward (body inspection).

---

## 7. Recommended Path Forward

### Immediate (This Week)

1. **Test if `max_tokens` can be increased to 512+ without OOM** — the 128 limit may be overly conservative
2. **Enable Flash Attention 2** — `model = ...from_pretrained(attn_implementation="flash_attention_2")`
3. **Profile actual VRAM usage** at different context lengths to find the true ceiling

### Short-term (1-2 weeks)

4. **Implement Gemma 3 4B GGUF as text-only backend** (Alternative 6.1)
5. **Build Rust router prototype** with input classification and dual-backend support
6. **Keep existing FastAPI** as multimodal backend behind the router

### Medium-term (1-2 months)

7. **Monitor llama.cpp Gemma 4 support** — contribute or wait for upstream
8. **When GGUF available:** Convert Gemma 4 to Q4_K_M and benchmark
9. **Implement model hot-swap** in Rust gateway with lazy unloading strategy

### Long-term

10. Full Rust migration (candle or llama.cpp FFI for both paths)
11. Continuous batching for text requests
12. Audio tower support via transformers path

---

## 8. Cost-Benefit Summary

| Approach | Feasibility | Performance Gain | Complexity | Recommended |
|---|---|---|---|---|
| Status quo (optimize) | ✅ Today | 2-4x (longer output) | Low | ⭐⭐⭐ **Immediate** |
| Gemma 3 GGUF + G4 vision | ✅ Today | 2-3x text speed | Medium | ⭐⭐⭐ **Short-term** |
| Gemma 4 GGUF hybrid | ❌ Blocked | 3-4x text speed | High | ⭐ **When supported** |
| Full Rust rewrite | ⚠️ Partial | Better concurrency | Very High | ⭐⭐ **Gradual** |
| Candle pure Rust | ❌ Insufficient | Unknown | Very High | ❌ Not ready |

---

## 9. Key Technical Decisions

### 9.1 Router: Rust vs Python

**Rust is justified** if:
- You need >100 concurrent connections (unlikely for personal/local use)
- You want to use llama.cpp FFI directly (avoids Python subprocess overhead)
- You're building for future multi-model deployment

**Python is sufficient** if:
- Single-user or <10 concurrent connections
- Using llama-cpp-python bindings
- Faster iteration time matters more than raw throughput

### 9.2 Model Swap Strategy

**Recommendation: Lazy swap with cooldown.**

```python
# Pseudocode for swap logic
class ModelManager:
    current_backend: Literal["gguf", "transformers"]
    last_multimodal_request: float
    COOLDOWN_SECONDS = 300  # 5 min before swapping back to GGUF

    def route(self, request):
        if request.has_images_or_audio:
            if self.current_backend != "transformers":
                self.swap_to("transformers")
            self.last_multimodal_request = time.time()
            return self.transformers_infer(request)
        else:
            # Text-only: prefer GGUF, fall back to transformers if loaded
            if self.current_backend == "gguf":
                return self.gguf_infer(request)
            elif time.time() - self.last_multimodal_request > COOLDOWN_SECONDS:
                self.swap_to("gguf")
                return self.gguf_infer(request)
            else:
                return self.transformers_infer(request)  # Use what's loaded
```

### 9.3 Vision Model in VRAM Optimization

The vision tower (~1.5GB fp16) could potentially stay loaded while swapping only the language model portion:

```
GGUF text model:   ~2.5GB (language model only)
Vision tower:      ~1.5GB (fp16, stays loaded)
embed_vision:      ~0.1GB (fp16, stays loaded)
KV cache:          ~1.0GB
─────────────────────────
Total:             ~5.1GB → 6.9GB free!

vs. current:
NF4 language:      ~5.2GB
Vision tower:      ~1.5GB  
embed_vision:      ~0.1GB
lm_head:           ~2.5GB
KV cache:          ~1.0GB
─────────────────────────
Total:             ~9.3GB → 2.3GB free
```

This "vision-always-loaded" approach could eliminate the swap latency for mixed workloads but requires deep integration with llama.cpp's loading to skip vision modules.

---

## 10. Conclusion

The hybrid architecture is **conceptually sound and well-suited** to the hardware constraints (RTX 3060 12GB). The core insight — using a lightweight quantized text backend for the common case (text/tools) and reserving the heavy transformers pipeline for multimodal — is the right approach.

However, **it is currently blocked by llama.cpp's lack of Gemma 4 support**. The recommended path is:

1. **Now:** Optimize the existing pipeline (Flash Attention, test higher max_tokens)
2. **1-2 weeks:** Implement Gemma 3 GGUF text backend as a stepping stone
3. **Ongoing:** Track llama.cpp Gemma 4 support, migrate when available
4. **Parallel:** Build Rust router incrementally, starting with the proxy pattern

The Rust backend is a good long-term investment for infrastructure quality but is not the critical path item. The critical path is getting a GGUF-quantized version of the language model that can run in ~3GB VRAM.
