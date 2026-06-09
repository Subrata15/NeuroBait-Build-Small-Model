---
title: NeuroBait
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: An ADHD-friendly space and gentle boost for your everyday.
---

# NeuroBait

NeuroBait is an ADHD-friendly companion for task initiation: a warm space and a
gentle boost for the moment when starting feels heavier than the task itself.

It is built to avoid shame, streak pressure, and generic productivity advice.
The app uses a fine-tuned small model, not an external LLM API.

## Build Small Hackathon

- Primary track: **Backyard AI**
- Bonus quest fit: **Well-Tuned**
- Bonus quest fit: **Off-Brand**
- Sponsor fit: **Modal-powered**

NeuroBait was fine-tuned with Modal and deployed as a Gradio app on Hugging Face
ZeroGPU.

## Model

- Base model: `unsloth/gemma-3-12b-it`
- Adapter: `build-small-hackathon/NeuroBait`
- Method: 16-bit LoRA via Unsloth
- Runtime: `transformers` + `peft`
- Quantization: 4-bit bitsandbytes NF4 inside the `@spaces.GPU` window

## Runtime

Expected environment variables:

```text
BASE_MODEL=unsloth/gemma-3-12b-it
ADAPTER_ID=build-small-hackathon/NeuroBait
LOAD_IN_4BIT=1
MAX_NEW_TOKENS=220
PREWARM=1
```

Weights are pre-warmed to the Space cache on CPU at import so the GPU window can
focus on quantized loading and generation.

## Interface

The interface uses custom Gradio styling, dark earthy colors, anti-shame copy,
and a simple mood check-in:

- Calm
- Tired
- Anxious
- Focused

The mood choice lightly adapts the response style while keeping the same model
and safety scope.
