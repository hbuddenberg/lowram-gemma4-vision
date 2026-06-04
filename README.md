# LowRAM Gemma 4 Vision

Run Gemma 4 E4B multimodal inference (text + vision) on consumer hardware with limited RAM. Text at ~4 tok/s, vision at ~7 tok/s on a single RTX 3060 12GB via Thunderbolt 3 eGPU.

**Key innovation**: 4-bit NF4 quantization for the language model backbone while keeping the vision pipeline (tower + projector) in full bfloat16 — fitting a 15GB model into 9.3GB VRAM.

## The Problem

Gemma 4 E4B is a 15GB multimodal model (text + vision + audio). Running it requires:
- 15GB+ VRAM for unquantized inference, OR
- Clever quantization that preserves vision capabilities

Standard 4-bit quantization breaks vision because the vision-to-language projector (`embed_vision`) gets quantized too, destroying all visual information. This project documents the fix and provides a ready-to-use inference script.

## The Fix

The `embed_vision` module must be excluded from bitsandbytes quantization. Without this, the model describes every image as "a solid gray background":

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=[
        "vision_tower", "model.vision_tower",
        "embed_vision", "model.embed_vision",    # ← CRITICAL
        "audio_tower", "model.audio_tower",
        "embed_audio", "model.embed_audio",
        "lm_head",
    ],
)
```

**Why both names?** `llm_int8_skip_modules` does fuzzy matching, but the full module path (`model.embed_vision`) is needed for nested modules in Gemma4's architecture. Without it, `embed_vision.embedding_projection.weight` gets quantized to `uint8` and vision becomes blind.

## Setup

### 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 3060 12GB | RTX 4060 16GB |
| System RAM | 4GB (with overcommit) | 16GB+ |
| Storage | 30GB free | NVMe SSD |
| Connection | PCIe or Thunderbolt 3+ | Direct PCIe |

### 2. NVIDIA Driver

```bash
# Arch Linux (kernel zen)
sudo pacman -S nvidia-open-dkms nvidia-utils linux-zen-headers

# Other distros: use standard nvidia driver >= 550
```

Blacklist nouveau:
```bash
# /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0
```

### 3. Memory Config (Low-RAM Hosts)

If your system RAM is less than the model file size (15GB), enable overcommit:

```bash
sudo sysctl vm.overcommit_memory=1
# Persist:
echo "vm.overcommit_memory=1" | sudo tee -a /etc/sysctl.d/99-overcommit.conf
```

This allows mmap to virtually map the 15GB safetensors file even with only 4GB RAM. The actual memory pressure stays low because bitsandbytes quantizes on load.

### 4. Thunderbolt 3 eGPU (If Using External GPU)

Set Thunderbolt Security Level to **Legacy Mode** in BIOS. Cold boot after changing. Verify:

```bash
lspci -nn | grep -i -E "thunderbolt|nvidia"
nvidia-smi
```

### 5. Python Dependencies

```bash
# Using uv (recommended)
uv pip install transformers>=5.10 accelerate>=1.13 bitsandbytes>=0.49 \
               torch torchvision Pillow

# Or pip
pip install transformers accelerate bitsandbytes torch torchvision Pillow
```

### 6. Download Model

```bash
mkdir -p ~/models/gemma4-heretic && cd ~/models/gemma4-heretic

# Large file — use wget with resume support
wget -c "https://huggingface.co/igorls/gemma-4-E4B-it-heretic/resolve/main/model.safetensors"

# Config files (small)
for f in config.json tokenizer.json tokenizer_config.json preprocessor_config.json \
         processor_config.json generation_config.json chat_template.jinja; do
  wget -c "https://huggingface.co/igorls/gemma-4-E4B-it-heretic/resolve/main/$f"
done
```

### 7. Run

```bash
python inference.py                                    # Text + sample image
python inference.py path/to/your/image.jpg             # Text + your image
GEMMA4_MODEL_PATH=~/models/gemma4-heretic python inference.py  # Custom model path
```

## OpenAI-Compatible API Server

Run Gemma 4 as a local OpenAI-compatible API — use it from any tool that supports OpenAI endpoints (Hermes, curl, OpenAI SDK, Continue, etc.).

### Quick Start

```bash
# Start the server (loads model on startup, ~90s)
./start.sh --port 8080

# Or directly:
.venv/bin/python server.py --port 8080
```

### Endpoints

#### Inference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/chat/completions` | POST | API Key | OpenAI-compatible chat (streaming + non-streaming) |
| `/v1/models` | GET | API Key | List available models |
| `/v1/models/{model}` | GET | API Key | Model details |
| `/health` | GET | None | Health check + VRAM + uptime |

#### Admin (requires `GEMMA4_MASTER_KEY`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /v1/keys` | POST | Create API key |
| `GET /v1/keys` | GET | List all keys |
| `GET /v1/keys/{prefix}/usage` | GET | Per-key usage stats |
| `POST /v1/keys/{prefix}/rotate` | POST | Regenerate secret |
| `DELETE /v1/keys/{prefix}` | DELETE | Delete key |
| `GET /v1/usage` | GET | Global analytics |
| `GET /v1/config` | GET | Server configuration |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMMA4_API_KEY` | `gemma4-local` | Legacy single-key auth |
| `GEMMA4_MASTER_KEY` | (none) | Admin key for management |
| `GEMMA4_ADMIN_KEY` | (none) | Alias for master key |
| `GEMMA4_MODEL_PATH` | `~/models/gemma4-heretic` | Model directory |
| `GEMMA4_MAX_IMAGE_MB` | `20` | Max vision image size |
| `GEMMA4_CORS_ORIGINS` | `*` | CORS allowed origins |

### Usage Examples

**curl (text):**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b-heretic",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "stream": false
  }'
```

**curl (streaming):**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b-heretic",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'
```

**curl (vision with base64 image):**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b-heretic",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "Describe this image"}
      ]
    }],
    "max_tokens": 512
  }'
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="gemma4-local")
response = client.chat.completions.create(
    model="gemma-4-e4b-heretic",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

### Connect to Hermes Agent

Add as a custom provider in `~/.hermes/config.yaml`:

```yaml
custom_providers:
  - name: gemma4-local
    base_url: http://localhost:8080/v1
    api_key: gemma4-local
    model: gemma-4-e4b-heretic
```

Then set as default:

```bash
hermes config set model.default gemma-4-e4b-heretic
hermes config set model.provider custom:gemma4-local
```

### Systemd Service (auto-start)

```bash
# Install the user service
mkdir -p ~/.config/systemd/user/
cp gemma4-api.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gemma4-api.service

# Check status (takes ~90s to load model)
systemctl --user status gemma4-api.service
curl http://localhost:8080/health
```

## Performance

| Metric | Value |
|--------|-------|
| Text generation | ~4.2 tok/s |
| Vision analysis | ~5.6-7.5 tok/s |
| VRAM usage | 9.3GB / 11.6GB (RTX 3060 12GB) |
| Model load time | ~90 seconds |
| First token latency | ~5 seconds (vision) |

Tested on: Intel i5-7260U (7.6GB RAM) + RTX 3060 12GB via Thunderbolt 3.

## Architecture

```
Gemma4ForConditionalGeneration
├── model (Gemma4Model)
│   ├── vision_tower (Gemma4VisionModel)     — 16 layers, 768d, fp16
│   ├── language_model                        — 42 layers, 2560d, NF4 4-bit
│   ├── embed_vision (Gemma4MultimodalEmbedder) — 768→2560 projection, fp16
│   └── audio_tower                           — skipped (not used)
└── lm_head (Linear 2560→262144)             — fp16
```

## Troubleshooting

### Vision sees "solid gray"
→ `embed_vision` is being quantized. Verify:
```python
print(model.model.embed_vision.embedding_projection.weight.dtype)
# Must be torch.bfloat16, NOT torch.uint8
```

### OOM when loading model
→ Enable overcommit: `sudo sysctl vm.overcommit_memory=1`

### nouveau conflicts with nvidia
→ Ensure blacklist is active: `lsmod | grep nouveau` should return nothing.

### Thunderbolt not detecting GPU
→ BIOS: set Thunderbolt Security Level to Legacy Mode. Cold boot (full power off).

## Project Structure

```
lowram-gemma4-vision/
├── README.md
├── SDD.md                        # Software Design Document (v2.1.0)
├── LICENSE (MIT)
├── requirements.txt
├── inference.py                 # Standalone text + vision script
├── server.py                    # OpenAI-compatible API server (FastAPI + SQLite)
├── start.sh                     # Quick start script
├── gemma4-api.service           # systemd user service
├── .env.example
├── .gitignore
├── docs/PLANNING/
│   ├── PRD.md                  # Product requirements
│   ├── TRD.md                  # Technical architecture
│   └── IMPLEMENTATION.md       # Phased plan
├── scripts/                     # Future: setup.sh
└── examples/                    # Future: usage examples
```

## License

MIT for code and documentation. The Gemma 4 model is subject to [Google's license](https://huggingface.co/igorls/gemma-4-E4B-it-heretic).
