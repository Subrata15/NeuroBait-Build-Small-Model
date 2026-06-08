# NeuroBait Gemma 4 26B-A4B

Draft model card for the Hugging Face Model repo.

## Model Summary

NeuroBait is a LoRA fine-tune of `unsloth/gemma-4-26b-a4b-it` for ADHD and
neurodivergent task initiation conversations. It is designed to produce warm,
short, agency-preserving prose that helps users start a task without turning the
conversation into a to-do list.

## Base Model

- Base: `unsloth/gemma-4-26b-a4b-it`
- Method: 16-bit LoRA
- License inheritance: verify from the base model card before publishing

## Training Data

The run #3 dataset is bilingual Indonesian/English:

- 270 train conversations
- 30 eval conversations
- multi-turn format with `messages[]`
- official NeuroBait system prompt prepended to each example

Dataset files are not included in the source repo or model repo unless explicitly
published later.

## Intended Use

This model is intended for a Gradio chat demo and research/product exploration
around ADHD-friendly task initiation support.

## Limitations

- Not a medical device.
- Not a replacement for professional mental-health support.
- May fail to infer the right motivator or deadline anchor from sparse context.
- English transfer must be evaluated separately for run #3.
- App-initiated reminders should be handled by app logic, not generated
  spontaneously by the model.

## Training Configuration

- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0
- Epochs: 3
- Learning rate: 2e-4
- Effective batch size: 8
- Max sequence length: 2048
- Chat template: `gemma-4`
- Response-only markers: `<|turn>user\n` / `<|turn>model\n`

## Evaluation

Add final run #3 metrics and qualitative examples here after Modal eval.
