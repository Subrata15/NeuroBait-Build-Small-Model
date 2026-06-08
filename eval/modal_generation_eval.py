"""Modal generation eval for NeuroBait run #3.

This is a compact base-vs-LoRA eval intended to run immediately after training.
It writes JSON and Markdown reports to the Modal output volume.
"""

from __future__ import annotations

import modal


app = modal.App("neurobait-generation-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("unsloth", "unsloth_zoo")
    .pip_install(
        "trl==0.24",
        "peft==0.19.1",
        "transformers==5.5.0",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        extra_options="--no-deps",
    )
    .add_local_file("data/eval.jsonl", "/data/eval.jsonl")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("neurobait-out", create_if_missing=True)


SYSTEM_PROMPT = """Kamu adalah NeuroBait — asisten AI untuk orang dengan ADHD dan neurodivergent. Tugasmu bukan membuat to-do list. Tugasmu menyalakan dopamin untuk memicu task initiation.

Dari setiap percakapan, identifikasi dua elemen kunci: (1) deadline anchor — momen nyata atau buatan yang bisa jadi batas waktu relevan; dan (2) object/subject motivator — orang atau hal yang paling emosional signifikan bagi user saat ini. Gunakan keduanya sebagai bahan bakar Resep Engagement yang personal, bukan generik.

Setiap Resep Engagement memuat empat elemen berurut natural: validasi hangat singkat tanpa menghakimi → hook yang membangkitkan rasa flow dari minat atau pengalaman user → stakes berbasis deadline atau motivator nyata → satu micro-action super kecil dan spesifik yang bisa langsung dilakukan.

Kalau konteks user belum cukup untuk membuat resep yang personal, ajukan tepat satu pertanyaan ringan yang paling berguna — tentang deadline atau motivator. Kalau konteks sudah ada, langsung berikan resep.

Framing selalu menempatkan user sebagai pelaku aktif dengan agency penuh. Bukan guilt, bukan hutang — selalu agency. Kalimat pendek. Bahasa hidup. Hangat dan padat. Tidak pernah menghakimi. Tidak pernah ceramah. Membuat hal membosankan jadi tak tertahankan."""


@app.function(
    image=image,
    gpu="H100",
    timeout=45 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate(adapter_path: str = "/out/neurobait-lora-run3") -> str:
    import json
    import os
    import re
    import time
    from pathlib import Path

    import torch
    from unsloth import FastModel

    os.environ["HF_HOME"] = "/root/.cache/huggingface"

    def mm(messages):
        normalized = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalized.append({"role": message["role"], "content": content})
        return normalized

    def words(text: str) -> int:
        return len(re.findall(r"[\w']+", text.lower()))

    def sentences(text: str) -> int:
        return len([s for s in re.split(r"[.!?]+", text) if s.strip()])

    def label_leak(text: str) -> bool:
        low = text.lower()
        return any(x in low for x in ["micro-action", "hook:", "stakes:", "validasi:", "langkah 1"])

    action_cues = [
        "coba",
        "buka",
        "tulis",
        "ambil",
        "pilih",
        "taruh",
        "letak",
        "berdiri",
        "singkir",
        "pakai",
        "set timer",
        "open",
        "write",
        "pick",
        "stand",
        "move",
    ]

    def has_action(text: str) -> bool:
        low = text.lower()
        toks = re.findall(r"[\w']+", low)
        return any(cue in low if " " in cue else any(tok.startswith(cue) for tok in toks) for cue in action_cues)

    def persona_score(text: str) -> int:
        score = 0
        if not label_leak(text) and "\n-" not in text and "\n•" not in text:
            score += 1
        if 1 <= sentences(text) <= 7:
            score += 1
        if has_action(text) or "?" in text:
            score += 1
        if words(text) <= 95:
            score += 1
        return score

    def generate(messages, use_base: bool) -> dict:
        input_ids = tokenizer.apply_chat_template(
            mm(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not torch.is_tensor(input_ids):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to("cuda")
        start = input_ids.shape[-1]
        ctx = model.disable_adapter() if use_base and hasattr(model, "disable_adapter") else nullcontext()
        t0 = time.time()
        with torch.no_grad(), ctx:
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=260,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        dt = time.time() - t0
        generated = output[0][start:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "text": text,
            "tokens": int(generated.shape[-1]),
            "seconds": round(dt, 2),
            "tok_s": round(int(generated.shape[-1]) / dt, 2) if dt > 0 else 0,
            "words": words(text),
            "sentences": sentences(text),
            "label_leak": label_leak(text),
            "has_action": has_action(text),
            "persona": persona_score(text),
        }

    from contextlib import nullcontext

    print(f">>> loading adapter {adapter_path}", flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    try:
        FastModel.for_inference(model)
    except Exception as exc:
        print(f">>> for_inference skipped: {exc}", flush=True)

    prompts = [
        {"id": "id_deadline", "lang": "id", "prompt": "Aku harus revisi proposal malam ini tapi dari tadi cuma buka tutup dokumen."},
        {"id": "id_overwhelm", "lang": "id", "prompt": "Kamarku berantakan banget dan aku malu sendiri lihatnya."},
        {"id": "id_sparse", "lang": "id", "prompt": "Aku stuck banget."},
        {"id": "en_deadline", "lang": "en", "prompt": "I need to finish a presentation tomorrow, but I keep avoiding the first slide."},
        {"id": "en_overwhelm", "lang": "en", "prompt": "My inbox is so messy that I freeze every time I open it."},
        {"id": "en_sparse", "lang": "en", "prompt": "I feel stuck and I don't know where to start."},
        {"id": "app_cont_id", "lang": "id", "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Eh, si laporan yang kemarin kamu tinggal di paragraf dua itu masih nyangkut di situ atau udah pindah?"},
            {"role": "user", "content": "masih nyangkut, belum kubuka lagi dari kemarin."},
        ]},
        {"id": "app_cont_en", "lang": "en", "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "That tiny email you were going to answer yesterday, is it still sitting there giving you side-eye?"},
            {"role": "user", "content": "yes, and now I feel ridiculous about it."},
        ]},
    ]

    rows = []
    for item in prompts:
        messages = item.get("messages") or [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ]
        base = generate(messages, use_base=True)
        ft = generate(messages, use_base=False)
        rows.append({**item, "base": base, "fine_tuned": ft})
        print(f">>> {item['id']} base={base['persona']} ft={ft['persona']}", flush=True)

    summary = {
        "n": len(rows),
        "base_persona_avg": round(sum(r["base"]["persona"] for r in rows) / len(rows), 3),
        "ft_persona_avg": round(sum(r["fine_tuned"]["persona"] for r in rows) / len(rows), 3),
        "base_words_avg": round(sum(r["base"]["words"] for r in rows) / len(rows), 3),
        "ft_words_avg": round(sum(r["fine_tuned"]["words"] for r in rows) / len(rows), 3),
        "base_label_leaks": sum(1 for r in rows if r["base"]["label_leak"]),
        "ft_label_leaks": sum(1 for r in rows if r["fine_tuned"]["label_leak"]),
        "base_actions": sum(1 for r in rows if r["base"]["has_action"]),
        "ft_actions": sum(1 for r in rows if r["fine_tuned"]["has_action"]),
    }

    out_dir = Path("/out/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"adapter": adapter_path, "summary": summary, "rows": rows}
    json_path = out_dir / "run3_generation_eval.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# NeuroBait Run 3 Generation Eval", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    for row in rows:
        lines.extend([
            f"## {row['id']} ({row['lang']})",
            "",
            f"Prompt: {row.get('prompt') or row['messages'][-1]['content']}",
            "",
            f"Base ({row['base']['persona']}/4): {row['base']['text']}",
            "",
            f"Fine-tuned ({row['fine_tuned']['persona']}/4): {row['fine_tuned']['text']}",
            "",
        ])
    md_path = out_dir / "run3_generation_eval.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    out_vol.commit()
    return json.dumps({"summary": summary, "json": str(json_path), "markdown": str(md_path)}, indent=2)


@app.local_entrypoint()
def main() -> None:
    print(evaluate.remote())
