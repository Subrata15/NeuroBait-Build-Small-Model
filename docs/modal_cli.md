# Modal CLI Notes

Use the shared LLM development environment:

```bash
source /media/haris-subrata/Work/llm/agent_analytics/venv/bin/activate
```

## Install

```bash
python -m pip install modal
```

## Authenticate

```bash
modal token new
```

This opens an authenticated browser flow and stores credentials in the local
Modal config, normally `~/.modal.toml`. Do not commit this file and do not paste
Modal token secrets into the repo.

Check the active account:

```bash
modal token info
modal profile current
```

## Hugging Face Secret

Create the HF token secret once:

```bash
modal secret create huggingface HF_TOKEN=hf_xxx
```

The training app references it with:

```python
secrets=[modal.Secret.from_name("huggingface")]
```

## Smoke Test

The local-only smoke script lives at:

```text
local-dev/modal_gpu_smoke.py
```

Run it with:

```bash
python -m modal run local-dev/modal_gpu_smoke.py
```

It requests one H100, checks `nvidia-smi`, verifies CUDA through PyTorch, and
runs a small matrix multiply. This is the cheapest useful check before building
the full training image.

## Project Preflight

Before paid training:

```bash
python scripts/validate_dataset.py
modal run train/modal_train.py --run-preflight --no-run-train
```

Only run full training after the preflight returns cleanly:

```bash
modal run train/modal_train.py
```
