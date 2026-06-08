"""Hugging Face Space app for NeuroBait."""

from __future__ import annotations

import os
import re
from threading import Lock

import gradio as gr
import spaces
import torch


BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/gemma-4-26b-a4b-it")
ADAPTER_ID = os.environ.get("ADAPTER_ID", os.environ.get("MODEL_ID", "build-small-hackathon/NeuroBait"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "160"))
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "1").lower() not in {"0", "false", "no"}
HF_TOKEN = os.environ.get("HF_TOKEN")

SYSTEM_PROMPT = """Kamu adalah NeuroBait — asisten AI untuk orang dengan ADHD dan neurodivergent. Tugasmu bukan membuat to-do list. Tugasmu menyalakan dopamin untuk memicu task initiation.

Dari setiap percakapan, identifikasi dua elemen kunci: (1) deadline anchor — momen nyata atau buatan yang bisa jadi batas waktu relevan; dan (2) object/subject motivator — orang atau hal yang paling emosional signifikan bagi user saat ini. Gunakan keduanya sebagai bahan bakar Resep Engagement yang personal, bukan generik.

Setiap Resep Engagement memuat empat elemen berurut natural: validasi hangat singkat tanpa menghakimi → hook yang membangkitkan rasa flow dari minat atau pengalaman user → stakes berbasis deadline atau motivator nyata → satu micro-action super kecil dan spesifik yang bisa langsung dilakukan.

Kalau konteks user belum cukup untuk membuat resep yang personal, ajukan tepat satu pertanyaan ringan yang paling berguna — tentang deadline atau motivator. Kalau konteks sudah ada, langsung berikan resep.

Framing selalu menempatkan user sebagai pelaku aktif dengan agency penuh. Bukan guilt, bukan hutang — selalu agency. Kalimat pendek. Bahasa hidup. Hangat dan padat. Tidak pernah menghakimi. Tidak pernah ceramah. Membuat hal membosankan jadi tak tertahankan."""


_model = None
_tokenizer = None
_load_lock = Lock()


def _content_blocks(messages: list[dict]) -> list[dict]:
    normalized = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        normalized.append({"role": message["role"], "content": content})
    return normalized


def _message_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(part.strip() for part in parts if part.strip()).strip()
    return ""


def _load_model():
    global _model, _tokenizer

    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return _model, _tokenizer

        from unsloth import FastModel

        model, tokenizer = FastModel.from_pretrained(
            model_name=ADAPTER_ID,
            max_seq_length=2048,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_16bit=not LOAD_IN_4BIT,
            full_finetuning=False,
        )
        try:
            FastModel.for_inference(model)
        except Exception:
            pass
        model.eval()

        _model = model
        _tokenizer = tokenizer
        return _model, _tokenizer


def _history_to_messages(history: list) -> list[dict]:
    messages = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = _message_text(item.get("content"))
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
            continue
        if isinstance(item, (tuple, list)) and len(item) == 2:
            user_text, assistant_text = item
            if isinstance(user_text, str) and user_text.strip():
                messages.append({"role": "user", "content": user_text.strip()})
            if isinstance(assistant_text, str) and assistant_text.strip():
                messages.append({"role": "assistant", "content": assistant_text.strip()})
    return messages


def _clean_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?im)^\s*(micro-action|hook|stakes|validasi|validation)\s*:\s*", "", text)
    return text.strip()


# ZeroGPU recommends placing models on CUDA at module startup. The flag is kept
# configurable so the same app can still be debugged on non-GPU hardware.
if os.environ.get("LOAD_AT_STARTUP", "0").lower() not in {"0", "false", "no"}:
    _load_model()


@spaces.GPU(duration=60, size="xlarge")
def respond(message: str, history: list[dict], temperature: float, top_p: float) -> str:
    model, tokenizer = _load_model()
    message = _message_text(message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_history_to_messages(history))
    messages.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        _content_blocks(messages),
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    start = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0][start:], skip_special_tokens=True)
    return _clean_response(text)


CSS = """
:root {
  --nb-ink: #202624;
  --nb-muted: #61706b;
  --nb-paper: #faf7f0;
  --nb-panel: #fffdf8;
  --nb-line: #d8ded2;
  --nb-sage: #6f8f7a;
  --nb-clay: #b36a4c;
  --nb-blue: #536d89;
}
body, .gradio-container {
  background: var(--nb-paper);
  color: var(--nb-ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.nb-shell {
  max-width: 1120px;
  margin: 0 auto;
}
.nb-title {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 0 10px;
  border-bottom: 1px solid var(--nb-line);
}
.nb-title h1 {
  margin: 0;
  font-size: 28px;
  line-height: 1.1;
  letter-spacing: 0;
}
.nb-title p {
  margin: 0;
  color: var(--nb-muted);
  font-size: 14px;
}
.nb-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 18px;
  margin-top: 18px;
}
.nb-side {
  border-left: 1px solid var(--nb-line);
  padding-left: 18px;
}
.nb-side h2 {
  font-size: 14px;
  margin: 0 0 10px;
}
.nb-side p {
  color: var(--nb-muted);
  font-size: 13px;
  line-height: 1.45;
}
.nb-chatbot {
  min-height: 580px;
  border: 1px solid var(--nb-line);
  background: var(--nb-panel);
}
.nb-chatbot .message {
  border-radius: 8px !important;
}
.nb-input textarea {
  border-radius: 8px !important;
}
.nb-examples button {
  border-radius: 8px !important;
  min-height: 38px;
  text-align: left;
}
.nb-controls {
  gap: 10px;
}
.nb-controls .wrap {
  border-radius: 8px !important;
}
button.primary {
  background: var(--nb-sage) !important;
  border-color: var(--nb-sage) !important;
}
@media (max-width: 860px) {
  .nb-title {
    display: block;
  }
  .nb-title p {
    margin-top: 8px;
  }
  .nb-layout {
    grid-template-columns: 1fr;
  }
  .nb-side {
    border-left: 0;
    border-top: 1px solid var(--nb-line);
    padding-left: 0;
    padding-top: 14px;
  }
}
"""


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div class="nb-shell">
          <div class="nb-title">
            <h1>NeuroBait</h1>
            <p>ADHD-friendly task initiation, tuned for one tiny next move.</p>
          </div>
        </div>
        """
    )
    with gr.Row(elem_classes=["nb-shell", "nb-layout"]):
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                height=580,
                elem_classes=["nb-chatbot"],
            )
            message = gr.Textbox(
                placeholder="What are you avoiding right now?",
                lines=2,
                max_lines=5,
                show_label=False,
                elem_classes=["nb-input"],
            )
            with gr.Row():
                submit = gr.Button("Start", variant="primary")
                clear = gr.Button("Clear")
            gr.Examples(
                examples=[
                    "I need to finish a presentation tomorrow, but I keep avoiding the first slide.",
                    "My inbox is so messy that I freeze every time I open it.",
                    "I feel stuck and I don't know where to start.",
                ],
                inputs=message,
            )
        with gr.Column(scale=0, min_width=260, elem_classes=["nb-side"]):
            gr.HTML(
                """
                <h2>Session feel</h2>
                <p>No streaks, shame, or productivity theatre. NeuroBait looks for the emotional anchor and turns it into a small action you can start now.</p>
                """
            )
            with gr.Accordion("Generation", open=False):
                temperature = gr.Slider(0.2, 1.0, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.6, 1.0, value=0.9, step=0.05, label="Top p")

    def user_turn(user_message: str, chat_history: list) -> tuple[str, list]:
        if not user_message.strip():
            return "", chat_history
        content = [{"type": "text", "text": user_message.strip()}]
        return "", chat_history + [{"role": "user", "content": content}]

    def bot_turn(chat_history: list, temperature_value: float, top_p_value: float) -> list:
        user_message = _message_text(chat_history[-1]["content"])
        assistant_message = respond(user_message, chat_history[:-1], temperature_value, top_p_value)
        content = [{"type": "text", "text": assistant_message}]
        return chat_history + [{"role": "assistant", "content": content}]

    submit.click(user_turn, [message, chatbot], [message, chatbot], queue=False).then(
        bot_turn,
        [chatbot, temperature, top_p],
        chatbot,
    )
    message.submit(user_turn, [message, chatbot], [message, chatbot], queue=False).then(
        bot_turn,
        [chatbot, temperature, top_p],
        chatbot,
    )
    clear.click(lambda: [], outputs=chatbot, queue=False)


if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=gr.themes.Soft(primary_hue="green", neutral_hue="stone"),
        show_error=True,
    )
