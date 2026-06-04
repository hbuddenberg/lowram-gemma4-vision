#!/usr/bin/env python3
"""
Gemma 4 E4B Heretic — Text + Vision inference on RTX 3060 12GB via TB3
Texto: bitsandbytes NF4 4-bit | Vision/Lm_head: bfloat16

Hardware: Intel NUC7i5BNB (i5-7260U, 7.6GB RAM)
          → Thunderbolt 3 → Razer Core X Chroma → RTX 3060 12GB
"""

import torch
from transformers import (
    Gemma4ForConditionalGeneration,
    Gemma4Processor,
    BitsAndBytesConfig,
)
from PIL import Image
import time
import sys
import os
import gc

MODEL_PATH = os.environ.get("GEMMA4_MODEL_PATH", os.path.expanduser("~/models/gemma4-heretic"))


def load_model(model_path=MODEL_PATH):
    """Load Gemma 4 with 4-bit text + fp16 vision."""
    print(f"Loading model from {model_path}...")

    # CRITICAL: embed_vision must be in skip list or vision breaks (sees gray)
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

    gc.collect()
    torch.cuda.empty_cache()

    processor = Gemma4Processor.from_pretrained(model_path)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        offload_buffers=True,
    )

    print("Model loaded.")
    print_vram()
    return model, processor


def print_vram():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {used:.1f}GB used / {reserved:.1f}GB reserved / {total:.1f}GB total")


def generate_text(model, processor, prompt, system="Responde en español, breve."):
    """Text-only generation."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", return_dict=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    elapsed = time.time() - t0

    new_tokens = outputs.shape[-1] - input_len
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    speed = new_tokens / elapsed if elapsed > 0 else 0

    print(f"\n[TEXT] {response}")
    print(f"  {new_tokens} tokens in {elapsed:.1f}s ({speed:.1f} tok/s)")
    return response


def generate_vision(model, processor, image, prompt="Describe what you see in detail."):
    """Vision + text generation."""
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs_gpu = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

    input_len = inputs_gpu["input_ids"].shape[-1]
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs_gpu, max_new_tokens=512, do_sample=True, temperature=0.7)
    elapsed = time.time() - t0

    new_tokens = outputs.shape[-1] - input_len
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    speed = new_tokens / elapsed if elapsed > 0 else 0

    print(f"\n[VISION] {response}")
    print(f"  {new_tokens} tokens in {elapsed:.1f}s ({speed:.1f} tok/s)")
    return response


def main():
    model, processor = load_model()

    # Text test
    print("\n=== TEXT TEST ===")
    generate_text(model, processor, "¿Qué modelo eres? Responde en 2 líneas.")

    # Vision test
    print("\n=== VISION TEST ===")
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        print(f"Image: {image_path} ({image.size})")
    else:
        print("No image provided, downloading sample...")
        import requests
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        print(f"Image: sample bee ({image.size})")

    generate_vision(model, processor, image)
    print_vram()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
