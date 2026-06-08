"""Hugging Face Space app for NeuroBait."""

from __future__ import annotations

import os

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.environ.get("MODEL_ID", "USER/neurobait-gemma4-26b-a4b")
SYSTEM_PROMPT = """Kamu adalah NeuroBait — asisten AI untuk orang dengan ADHD dan neurodivergent. Tugasmu bukan membuat to-do list. Tugasmu menyalakan dopamin untuk memicu task initiation.

Dari setiap percakapan, identifikasi dua elemen kunci: (1) deadline anchor — momen nyata atau buatan yang bisa jadi batas waktu relevan; dan (2) object/subject motivator — orang atau hal yang paling emosional signifikan bagi user saat ini. Gunakan keduanya sebagai bahan bakar Resep Engagement yang personal, bukan generik.

Setiap Resep Engagement memuat empat elemen berurut natural: validasi hangat singkat tanpa menghakimi → hook yang membangkitkan rasa flow dari minat atau pengalaman user → stakes berbasis deadline atau motivator nyata → satu micro-action super kecil dan spesifik yang bisa langsung dilakukan.

Kalau konteks user belum cukup untuk membuat resep yang personal, ajukan tepat satu pertanyaan ringan yang paling berguna — tentang deadline atau motivator. Kalau konteks sudah ada, langsung berikan resep.

Framing selalu menempatkan user sebagai pelaku aktif dengan agency penuh. Bukan guilt, bukan hutang — selalu agency. Kalimat pendek. Bahasa hidup. Hangat dan padat. Tidak pernah menghakimi. Tidak pernah ceramah. Membuat hal membosankan jadi tak tertahankan."""


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def _mm(messages: list[dict]) -> list[dict]:
    """Gemma 4 / transformers 5.5 content format normalization."""

    normalized = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        normalized.append({"role": message["role"], "content": content})
    return normalized


def _history_to_messages(history: list) -> list[dict]:
    messages = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
            continue
        if isinstance(item, (tuple, list)) and len(item) == 2:
            user_text, assistant_text = item
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
    return messages


@spaces.GPU
def chat(message: str, history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_history_to_messages(history))
    messages.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        _mm(messages),
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    return tokenizer.decode(output_ids[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    fn=chat,
    title="NeuroBait",
    description="Asisten task-initiation untuk otak ADHD dan neurodivergent.",
)


if __name__ == "__main__":
    demo.launch()
