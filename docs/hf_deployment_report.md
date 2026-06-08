# Hugging Face Deployment Report

Status date: 2026-06-08

## Public Targets

- Model repo: `https://huggingface.co/build-small-hackathon/NeuroBait`
- Space repo: `https://huggingface.co/spaces/build-small-hackathon/NeuroBait`
- Direct Space host: `https://build-small-hackathon-neurobait.hf.space`

The short slug keeps the hackathon URL readable while the app title and model
card carry the fuller NeuroBait context.

## Model Repo

The LoRA adapter was uploaded from the Modal Volume `/out/neurobait-lora-run3`
to `build-small-hackathon/NeuroBait`.

Uploaded artifacts include:

- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer files
- `chat_template.jinja`
- `train_summary.json`
- NeuroBait model card

## Space Runtime

The Space was created as a public Gradio Space in the `build-small-hackathon`
org. Runtime verification through the Hugging Face API showed:

```text
stage: RUNNING
hardware.current: zero-a10g
hardware.requested: zero-a10g
host: https://build-small-hackathon-neurobait.hf.space
```

The Gradio page and `/config` endpoint returned HTTP 200.

## Feasibility Notes

Three deployment paths were tested:

1. Transformers + PEFT 4-bit adapter loading.
   - Result: app startup reached model loading, then failed.
   - Reason: PEFT could not inject LoRA into `Gemma4ClippableLinear` on the
     Gemma 4 A4B stack.

2. Unsloth startup import/load.
   - Result: failed at import time.
   - Reason: ZeroGPU had no CUDA device during module import.

3. Unsloth lazy import/load inside `@spaces.GPU`.
   - Result: Space runs and event routing works.
   - Blocking condition during final test: ZeroGPU quota was exhausted:

```text
120s requested vs. 0s left
retry after roughly 23h44m from the 2026-06-08 test
```

The current Space source keeps the third path because it matches the loader used
successfully in Modal generation eval. The next direct feasibility check should
retry inference after ZeroGPU quota resets.

## Open Decision

If ZeroGPU lazy Unsloth still fails after quota reset, the pragmatic next choices
are:

- use a paid HF GPU tier for the Space after explicit approval, or
- produce a deployment-specific quantized/merged artifact that avoids PEFT
  adapter injection against `Gemma4ClippableLinear`.
