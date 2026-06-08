---
title: NeuroBait
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: ADHD-friendly task initiation with one tiny next move.
---

# NeuroBait

NeuroBait is an ADHD-friendly task-initiation assistant. It is designed to help
the user find one tiny next move without shame, streak pressure, or a full
productivity lecture.

Target URL:

```text
https://huggingface.co/spaces/build-small-hackathon/NeuroBait
```

This Space loads through Unsloth:

- Base model: `unsloth/gemma-4-26b-a4b-it`
- LoRA adapter: `build-small-hackathon/NeuroBait`

## Runtime

The default app path uses Unsloth 4-bit loading for the NeuroBait LoRA adapter.
The `@spaces.GPU` call requests ZeroGPU `xlarge` with a 60-second function
window because the model is a 26B MoE adapter stack.

Expected environment variables:

```text
BASE_MODEL=unsloth/gemma-4-26b-a4b-it
ADAPTER_ID=build-small-hackathon/NeuroBait
LOAD_IN_4BIT=1
MAX_NEW_TOKENS=160
LOAD_AT_STARTUP=0
```

If the base model requires authentication in the Space runtime, add `HF_TOKEN`
as a Space secret.
