# NeuroBait HF Space

This directory is the source for the Hugging Face Space repo. The Space should
remain separate from the GitHub source repo and the HF Model repo.

## Files

- `app.py`: Gradio chat app
- `requirements.txt`: Space dependencies

## Space Configuration

Set this environment variable in the HF Space:

```text
MODEL_ID=USER/neurobait-gemma4-26b-a4b
```

The model repo should point to the merged model intended for inference.

## Hardware

A merged bf16 Gemma 4 26B-A4B model is large. Use a GPU tier that can actually
load the final model, or switch the app to a quantized loading path before using
smaller hardware.

## App-Initiated Openers

Proactive openers belong in application logic. If the app wants to start with an
assistant message, inject that opener into chat history before the next user
turn. Do not ask the model to decide when to initiate contact.
