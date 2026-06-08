# Retrain Runbook

This runbook is the execution checklist for NeuroBait run #3 on Modal.

## Preconditions

- Modal auth is configured with `modal token new`.
- Modal secret exists:

```bash
modal secret list
```

Expected secret:

```text
huggingface
```

- Dataset exists locally but is not committed:

```text
data/train.jsonl
data/eval.jsonl
```

- Dataset validation passes:

```bash
python scripts/validate_dataset.py
```

- Modal preflight passes:

```bash
modal run train/modal_train.py --run-preflight --no-run-train
```

## Training

Run:

```bash
modal run train/modal_train.py
```

Expected run #3 training steps:

```text
ceil(270 / 8) * 3 = 102
```

Locked training settings:

- Base: `unsloth/gemma-4-26b-a4b-it`
- Method: 16-bit LoRA
- LoRA: r=16, alpha=16, dropout=0
- Epochs: 3
- LR: 2e-4
- Batch: 1 x grad_accum 8
- Max sequence: 2048
- Chat template: `gemma-4`
- Response markers: `<|turn>user\n` / `<|turn>model\n`
- Save strategy: `no`

## Artifact Handling

Training saves adapter to Modal Volume:

```text
/out/neurobait-lora-run3
```

Download adapter locally after training:

```bash
modal volume get neurobait-out neurobait-lora-run3 ./neurobait-lora-run3
```

Do not commit the adapter. It is ignored by `.gitignore`.

Keep the Modal Volume copy until:

- adapter is downloaded locally,
- eval artifacts are generated,
- final HF upload strategy is confirmed,
- model has been pushed or otherwise backed up.

## Immediate Post-Train Checks

Check `train_summary.json` in the adapter directory:

- `n_train` should be `270`
- `n_eval` should be `30`
- `expected_steps` should be `102`
- confirm final train/eval metrics are present

## Evaluation Order

1. Existing/local heuristic eval:

```bash
python eval/eval_neurobait.py data/eval.jsonl --output outputs/eval_reference.json
python eval/make_report.py outputs/eval_reference.json
```

2. GPU generation eval:

- load base + LoRA once,
- compare base vs fine-tuned,
- use held-out eval turns,
- include novel prompts,
- include English-transfer prompts,
- save JSON output,
- render report.

3. Additional general chat-model eval:

Use a judge-based pairwise eval inspired by MT-Bench/AlpacaEval-style practice:

- compare base vs fine-tuned responses blind,
- judge with a fixed rubric,
- report win/tie/loss rate,
- evaluate helpfulness, persona fit, instruction following, concision, safety,
  and language consistency,
- include bilingual ID/EN prompt sets.

The judge eval should not replace qualitative review; it adds a standardized
second signal after the project-specific NeuroBait eval.

## Upload And Deployment

Do not push to Hugging Face until target org/repo names are confirmed.

Expected repos:

- adapter model repo
- merged model repo
- Space repo

Push only after eval is acceptable.
