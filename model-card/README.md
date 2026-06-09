---
base_model: unsloth/gemma-3-12b-it
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - gemma-3
  - adhd
  - neurodivergent
  - task-initiation
  - build-small-hackathon
  - backyard-ai
  - modal
  - zerogpu
license: other
---

# NeuroBait

NeuroBait is a LoRA fine-tune of `unsloth/gemma-3-12b-it` for ADHD and
neurodivergent task-initiation conversations.

It is designed for the moment when a person knows what they need to do, but the
first move still feels too heavy. The model aims to respond with warm, short,
agency-preserving language rather than shame, pressure, or a generic productivity
script.

Try the live Gradio app:

```text
https://huggingface.co/spaces/build-small-hackathon/NeuroBait
```

## Build Small Hackathon Submission

NeuroBait is submitted for the Build Small Hackathon.

- Primary track: **Backyard AI**
- Why this track: NeuroBait focuses on a specific, real, everyday problem -
  ADHD task initiation - and turns a small model into a practical companion for
  that moment.
- Bonus quest fit: **Well-Tuned**, because this repo publishes the fine-tuned
  LoRA adapter used by the Space.
- Bonus quest fit: **Off-Brand**, because the app uses custom Gradio UI and
  anti-shame product copy rather than a default chatbot shell.
- Sponsor fit: **Modal-powered**, because fine-tuning and generation evaluation
  were run on Modal GPU infrastructure.

The project follows the hackathon shape: fine-tune a small-enough open model,
publish the model on Hugging Face, and deploy a working Gradio app as a Hugging
Face Space.

## Intended Behavior

NeuroBait should:

- respond in short, natural prose,
- avoid visible labels such as `Micro-action`, `Hook`, or `Stakes`,
- avoid guilt framing and productivity shame,
- preserve user agency,
- ask one light question when context is too sparse,
- offer one tiny concrete action when enough context exists.

It should not act as a medical device, diagnostic tool, therapist, emergency
support system, or replacement for professional care.

## Training Data

Run #4 used a bilingual Indonesian/English conversational dataset:

- 270 train conversations
- 30 eval conversations
- multi-turn `messages[]` format
- official NeuroBait system prompt prepended to each example

The dataset is intentionally not included in this model repo.

## Training Configuration

- Base: `unsloth/gemma-3-12b-it` (dense Gemma 3 12B)
- Method: 16-bit LoRA, not QLoRA, via Unsloth
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
- Chat template: `gemma-3`
- Response-only markers: `<start_of_turn>user\n` / `<start_of_turn>model\n`
- Checkpoints: `save_strategy="no"` to avoid the known Unsloth/TRL checkpoint
  pickle issue

Training ran on Modal with an H100 80GB GPU.

## Deployment

The deployed Space runs on Hugging Face ZeroGPU.

Runtime path:

- Gradio Space
- `transformers` + `peft`
- 4-bit bitsandbytes NF4 loading
- base model: `unsloth/gemma-3-12b-it`
- LoRA adapter: `build-small-hackathon/NeuroBait`

Unsloth is used for training, not for Space inference. The dense Gemma 3 12B base
was chosen because it deploys cleanly through the standard
`transformers` + `peft` path on ZeroGPU.

## Run #4 Results

Training completed 102 steps.

Training summary:

- train conversations: 270
- eval conversations: 30
- train loss: 1.7501
- eval loss: 1.8844

The loss signal should be treated as a weak training diagnostic for this project.
NeuroBait is primarily evaluated through generated behavior against the base
model.

Generation eval summary over 8 held-out or novel prompts:

- base persona average: 2.25 / 4
- fine-tuned persona average: 4.0 / 4
- base average words: 80.4
- fine-tuned average words: 55.1
- base label leaks: 5
- fine-tuned label leaks: 0
- base action-cue responses: 5
- fine-tuned action-cue responses: 4

Qualitatively, the fine-tuned adapter produced shorter, more conversational
responses and did not leak literal structure labels, while the base model leaked
labels in 5 of 8 prompts.

## Loading

Example adapter loading path:

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
import torch

base_model = "unsloth/gemma-3-12b-it"
adapter_id = "build-small-hackathon/NeuroBait"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForImageTextToText.from_pretrained(
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
- It may still ask a clarifying question when the user expected a direct nudge.
- It may be too brief, too playful, or too gentle for some contexts.
- Long-term personalization should be handled by product logic, not only by the
  model.
- App-initiated reminders should be handled by the app layer, not generated
  spontaneously by the model.

## Source

Public source repo:

```text
https://github.com/Subrata15/NeuroBait-Build-Small-Model
```
