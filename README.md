# NeuroBait Build Small Model

NeuroBait is a fine-tuned assistant for ADHD / neurodivergent task initiation.
This repo is the engineering source of truth for training, evaluation, and
deployment scaffolding. Dataset and model artifacts are intentionally kept out
of Git.

## Architecture

There are three separate operational artifacts:

1. GitHub repo: training, eval, deploy source code, and docs.
2. Hugging Face Model repo: LoRA adapter and/or merged model weights.
3. Hugging Face Space repo: Gradio app that loads the HF Model repo.

Flow:

```text
GitHub source code
  -> Modal.com fine-tune/eval/merge
  -> HF Model repo
  -> HF Space Gradio app
```

`github-hf.svg` contains the same architecture as a visual diagram.

## Local Development

Use the shared LLM development venv:

```bash
source /media/haris-subrata/Work/llm/agent_analytics/venv/bin/activate
python --version
```

Install only lightweight local tooling here when needed. GPU-heavy training
dependencies are owned by the Modal image in `train/modal_train.py`.

## Data

Copy the upstream final dataset out-of-band:

```text
data/train.jsonl
data/eval.jsonl
```

Expected run #3 dataset:

- `train.jsonl`: 270 conversations
- `eval.jsonl`: 30 conversations
- bilingual ID/EN
- each row has `messages[]` with the official NeuroBait system prompt prepended

`data/` and `*.jsonl` are gitignored.

## Training On Modal

One-time setup:

```bash
pip install modal
modal token new
modal secret create huggingface HF_TOKEN=hf_xxx
```

Run training:

```bash
modal run train/modal_train.py
```

Remote preflight before paid training:

```bash
modal run train/modal_train.py --run-preflight --no-run-train
```

Optional upload/merge:

```bash
modal run train/modal_train.py --push-lora
modal run train/modal_train.py --push-lora --merge
```

Locked training design:

- Base: `unsloth/gemma-4-26b-a4b-it`
- Method: 16-bit LoRA, not QLoRA
- LoRA: `r=16`, `alpha=16`, `dropout=0`
- Epochs: `3`
- LR: `2e-4`
- Batch: `1 x grad_accum 8`
- Max sequence: `2048`
- Chat template: `gemma-4`
- Response markers: `<|turn>user\n` and `<|turn>model\n`
- `save_strategy="no"` to avoid the known Unsloth/TRL checkpoint pickle bug

Expected step count for 270 train examples:

```text
ceil(270 / 8) * 3 = 102
```

## Evaluation

Local heuristic check:

```bash
python eval/eval_neurobait.py data/eval.jsonl --output outputs/eval_reference.json
python eval/make_report.py outputs/eval_reference.json
```

This is not a replacement for GPU qualitative eval. It gives a fast local check
for label leaks, rough response length, and action-cue coverage. Run #3 must
include separate English-transfer review on held-out EN examples.

## HF Space

`deploy/` is the source for the HF Space repo:

```text
deploy/app.py
deploy/requirements.txt
```

Set `MODEL_ID` in the Space environment to the merged HF Model repo, for example:

```text
USER/neurobait-gemma4-26b-a4b
```

The Space uses Gradio and `@spaces.GPU`. Hardware must be chosen based on the
final model format. A merged bf16 26B model needs a large GPU; a quantized or
4-bit loading path may be needed for smaller Space hardware.

## Repository Boundaries

Commit to GitHub:

- `train/`
- `eval/`
- `deploy/`
- docs and diagrams

Do not commit:

- dataset JSONL
- LoRA adapters
- merged models
- GGUF files
- generated eval outputs
- tokens or `.env`
