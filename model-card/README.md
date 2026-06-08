---
base_model: unsloth/gemma-4-26b-a4b-it
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - gemma-4
  - adhd
  - neurodivergent
  - task-initiation
  - build-small-hackathon
license: other
---

# NeuroBait

NeuroBait is a LoRA fine-tune of `unsloth/gemma-4-26b-a4b-it` for ADHD and
neurodivergent task-initiation conversations. It is designed to produce warm,
short, agency-preserving prose that helps a user start one tiny next move
without turning the conversation into a full to-do list.

This repo contains the LoRA adapter, tokenizer files, chat template, and the
training summary for run #3. It is intended to be used by the Gradio Space:

```text
https://huggingface.co/spaces/build-small-hackathon/NeuroBait
```

## Intended Behavior

NeuroBait should:

- respond in short, natural prose,
- avoid visible labels such as `Micro-action`, `Hook`, or `Stakes`,
- avoid guilt framing,
- preserve user agency,
- ask one light question when context is too sparse,
- offer one tiny concrete action when enough context exists.

It should not act as a medical device, diagnostic tool, therapist, emergency
support system, or replacement for professional care.

## Training Data

Run #3 used a bilingual Indonesian/English conversational dataset:

- 270 train conversations
- 30 eval conversations
- multi-turn `messages[]` format
- official NeuroBait system prompt prepended to each example

The dataset is not included in this model repo.

## Training Configuration

- Base: `unsloth/gemma-4-26b-a4b-it`
- Method: 16-bit LoRA
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0
- Target modules: q/k/v/o/gate/up/down projections
- Epochs: 3
- Learning rate: 2e-4
- Effective batch size: 8
- Max sequence length: 2048
- Scheduler: cosine
- Warmup ratio: 0.05
- Optimizer: adamw 8-bit
- Precision: bf16
- Chat template: `gemma-4`
- Response-only markers: `<|turn>user\n` / `<|turn>model\n`

Training ran on Modal with an H100 80GB GPU.

## Run #3 Results

Training completed 102 steps.

Training summary:

- train conversations: 270
- eval conversations: 30
- train loss: 0.242
- final `trainer.evaluate()` loss: 2.4044

The loss signal should be treated as a weak training diagnostic for this project.
NeuroBait is primarily evaluated through generated behavior against the base
model.

Generation eval summary over 8 held-out/novel prompts:

- base persona average: 3.625 / 4
- fine-tuned persona average: 4.0 / 4
- base average words: 72.0
- fine-tuned average words: 56.625
- base label leaks: 1
- fine-tuned label leaks: 0
- base action-cue responses: 3
- fine-tuned action-cue responses: 5

Qualitatively, the fine-tuned adapter produced shorter, more conversational
responses, maintained English on English prompts more reliably than the base
model in this sample, and avoided literal structure labels in the tested turns.

## Loading

Example adapter loading path:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

base_model = "unsloth/gemma-4-26b-a4b-it"
adapter_id = "build-small-hackathon/NeuroBait"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()
```

## Limitations

- The model is not a medical or crisis-support system.
- It may ask for a motivator or deadline when the user has already implied one.
- It may still be too brief or too playful for some users.
- Long-term personalization should be handled by product logic, not only by the
  model.
- App-initiated reminders should be handled by the app layer, not generated
  spontaneously by the model.

## Submission Context

This model is part of the Build Small Model Hackathon submission for NeuroBait.
The public GitHub source repo tracks training, evaluation, deployment scaffolding,
and agent development notes.
