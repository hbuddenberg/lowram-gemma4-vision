# NUC Gemma 4 Vision

Intel NUC7i5BNB → Thunderbolt 3 → RTX 3060 12GB → Gemma 4 E4B multimodal inference with vision.

## Hardware

| Component | Detail |
|-----------|--------|
| Host | Intel NUC7i5BNB (i5-7260U, 7.6GB RAM) |
| eGPU enclosure | Razer Core X Chroma (×2) |
| GPU | NVIDIA RTX 3060 12GB (GA104) |
| Connection | Thunderbolt 3 via Alpine Ridge 2C+4C, 40 Gb/s |
| OS | Arch Linux, kernel 7.0.10-zen1-1-zen |

## Setup

### 1. Thunderbolt 3 (BIOS)

Set Thunderbolt Security Level to **Legacy Mode** in BIOS (BNKBL357.86A.0088). Unique ID mode does NOT work with Alpine Ridge on Linux. Requires cold boot after BIOS change.

```bash
# Verify TB3 appears
lspci -nn | grep -i thunderbolt
# Should show: Alpine Ridge 2C [8086:15da] and Alpine Ridge 4C [8086:15d3]

# Verify GPU appears
lspci -nn | grep -i nvidia
# Should show: NVIDIA RTX 3060 [10de:2504]
```

### 2. NVIDIA Driver

```bash
sudo pacman -S nvidia-open-dkms nvidia-utils nvidia-settings linux-zen-headers
sudo dkms install nvidia/610.43.02 -k 7.0.10-zen1-1-zen
```

Blacklist nouveau:
```bash
# /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0
```

Reboot, then verify:
```bash
nvidia-smi
# Should show RTX 3060, 12GB VRAM
```

### 3. System Config

Allow mmap for large safetensors (7.6GB RAM host needs overcommit):
```bash
sudo sysctl vm.overcommit_memory=1
# Persist: echo "vm.overcommit_memory=1" | sudo tee -a /etc/sysctl.d/99-overcommit.conf
```

### 4. Python Dependencies

```bash
# Using uv (recommended)
uv pip install transformers accelerate bitsandbytes torchvision Pillow torch
```

### 5. Download Model

```bash
# From HuggingFace (15GB)
wget -c "https://huggingface.co/igorls/gemma-4-E4B-it-heretic/resolve/main/model.safetensors" \
  -O ~/models/gemma4-heretic/model.safetensors

# Also need: config.json, tokenizer.json, preprocessor_config.json, processor_config.json,
# generation_config.json, tokenizer_config.json, chat_template.jinja
# Either wget each file or use:
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('igorls/gemma-4-E4B-it-heretic', local_dir='~/models/gemma4-heretic', allow_patterns=['*.json','*.jinja'])"
```

### 6. Run Inference

```bash
python3 inference.py
# Text: ~4.2 tok/s
# Vision: ~5.6-7.5 tok/s
# VRAM: 9.3GB / 11.6GB
```

## Key Finding: embed_vision Must Skip Quantization

**Critical bug fix**: The `embed_vision` module (Gemma4MultimodalEmbedder) maps vision embeddings to the language model's embedding space. If bitsandbytes quantizes it to 4-bit (uint8), all image information is destroyed — the model sees a uniform gray image regardless of input.

The `llm_int8_skip_modules` must include the **full module path** for the skip to work:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=[
        "vision_tower", "model.vision_tower",
        "multi_modal_projector",
        "embed_vision", "model.embed_vision",
        "audio_tower", "model.audio_tower",
        "embed_audio", "model.embed_audio",
        "lm_head",
    ],
)
```

### Why This Happens

- Gemma4's architecture uses `model.embed_vision` (Gemma4MultimodalEmbedder), NOT `multi_modal_projector`
- `llm_int8_skip_modules` does fuzzy matching but the full path is needed for nested modules
- Without the skip, `embed_vision.embedding_projection.weight` gets quantized to uint8, producing garbage embeddings
- The vision tower itself (bfloat16) works fine — the corruption happens at the projection step

### Symptoms

- Text inference works perfectly (4+ tok/s)
- Vision inference runs but always describes "a solid gray image"
- `pixel_values` are correct (min=0, max=0.98, 7.1% zeros)
- Vision tower output is correct (varied activations)
- `embed_vision` parameters show `torch.uint8` instead of `torch.bfloat16`

## Architecture

```
Gemma4ForConditionalGeneration
├── model (Gemma4Model)
│   ├── vision_tower (Gemma4VisionModel) — 16 layers, hidden=768, fp16
│   ├── language_model — 42 layers, hidden=2560, 4-bit NF4
│   ├── audio_tower (Gemma4AudioModel) — skipped
│   ├── embed_vision (Gemma4MultimodalEmbedder) — projection 768→2560, fp16
│   └── embed_audio — skipped
└── lm_head — Linear(2560, 262144), fp16
```

VRAM breakdown:
- Language model 4-bit: ~3.5GB
- Vision tower fp16: ~0.5GB
- embed_vision + lm_head fp16: ~0.8GB
- KV cache + overhead: ~4.5GB
- **Total: ~9.3GB / 11.6GB**

## Files

- `inference.py` — Main inference script (text + vision)
- `config.json` — Model config (reference, downloaded from HF)
- `requirements.txt` — Python dependencies

## License

Setup scripts and documentation: MIT
Model: See [igorls/gemma-4-E4B-it-heretic](https://huggingface.co/igorls/gemma-4-E4B-it-heretic) for model license.
