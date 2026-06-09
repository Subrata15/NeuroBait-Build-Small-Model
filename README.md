# NeuroBait

NeuroBait is an ADHD-friendly AI companion for task initiation. It is built for
the moment when a task is not intellectually hard, but still feels impossible to
start.

Instead of turning that friction into a productivity lecture, NeuroBait responds
with short, warm, agency-preserving language. It avoids shame, streak pressure,
diagnosis-like framing, and visible prompt labels such as `Micro-action`, `Hook`,
or `Stakes`.

## Live Demo

- Hugging Face Space: https://huggingface.co/spaces/build-small-hackathon/NeuroBait
- Model adapter: https://huggingface.co/build-small-hackathon/NeuroBait

## Hackathon Submission

NeuroBait is a Build Small Hackathon submission.

- Primary track: **Backyard AI**
- Why: the app is designed around a real everyday ADHD friction - starting the
  thing that already matters - rather than a generic chatbot or productivity
  template.
- Bonus quest fit: **Well-Tuned**, because NeuroBait uses a fine-tuned model
  published on Hugging Face.
- Bonus quest fit: **Off-Brand**, because the Space uses a custom Gradio
  interface instead of the default app look.
- Sponsor fit: **Modal-powered**, because fine-tuning and generation eval were
  run on Modal GPU infrastructure.

## What We Built

The model is a LoRA fine-tune of `unsloth/gemma-3-12b-it`, trained on a small
hand-curated bilingual Indonesian/English dataset built from real ADHD task
initiation friction.

The main lesson was simple: for a voice and behavior layer, dataset quality beat
model size. The fine-tune is intentionally narrow. It is not trying to become a
general therapist, planner, or productivity operating system.

## The Stack

A real model, trained on a real budget:

- **Base:** `unsloth/gemma-3-12b-it` (dense Gemma 3 12B,
  `Gemma3ForConditionalGeneration`)
- **Method:** 16-bit LoRA, not QLoRA, via Unsloth
- **LoRA:** `r=16`, `alpha=16`, `dropout=0`
- **Epochs:** `3` | **LR:** `2e-4` | **Batch:** `1 x grad_accum 8` |
  **Max sequence:** `2048`
- **Chat template:** `gemma-3`, response markers `<start_of_turn>user\n` and
  `<start_of_turn>model\n`
- `save_strategy="no"` to avoid the known Unsloth/TRL checkpoint pickle bug
- **Train/eval:** Modal H100 80GB GPU
- **Deploy:** Hugging Face Space on ZeroGPU using Gradio, `transformers`,
  `peft`, and 4-bit bitsandbytes NF4 runtime loading
- **Data:** small, hand-curated, synthetic, and grounded in real ADHD friction,
  not generic productivity tropes

## Results

Run #4 completed 102 training steps.

- Train conversations: 270
- Eval conversations: 30
- Train loss: 1.7501
- Eval loss: 1.8844

Generation eval over 8 held-out or novel prompts:

- Base persona average: 2.25 / 4
- Fine-tuned persona average: 4.0 / 4
- Base average words: 80.4
- Fine-tuned average words: 55.1
- Base label leaks: 5
- Fine-tuned label leaks: 0

The loss is only a weak diagnostic here. The important result is behavioral: the
fine-tuned model became shorter, warmer, more consistent, and stopped leaking
the internal recipe labels that the base model often exposed.

## Repository Map

- `train/`: Modal + Unsloth fine-tuning entrypoints
- `eval/`: local and Modal generation evaluation scripts
- `deploy/`: Hugging Face Space app source
- `model-card/`: README used for the Hugging Face model repo
- `docs/`: runbooks, deployment notes, and evaluation notes

The dataset and trained adapter artifacts are intentionally kept out of Git.

## Reproduce

Install the Modal CLI, authenticate, and create a Hugging Face secret:

```bash
pip install modal
modal token new
modal secret create huggingface HF_TOKEN=hf_xxx
```

Run a remote preflight:

```bash
modal run train/modal_train.py --run-preflight --no-run-train
```

Train:

```bash
modal run train/modal_train.py
```

Push the LoRA adapter and Space:

```bash
modal run train/modal_train.py --no-run-train --push-lora
modal run train/modal_train.py --no-run-train --push-space --set-hardware
```

## Safety Scope

NeuroBait is not a medical device, diagnostic tool, therapist, or emergency
support system. It is a small-model demo for gentle task-initiation support.
