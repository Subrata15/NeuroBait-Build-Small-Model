"""Core NeuroBait fine-tuning logic.

This module is intentionally import-safe: call ``train_adapter()`` from Modal or
from a GPU box. Unsloth and other heavy libraries are imported inside the
function so local development can still run syntax checks without the GPU stack.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BASE_MODEL = "unsloth/gemma-4-26b-a4b-it"
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"


@dataclass(frozen=True)
class TrainConfig:
    base_model: str = BASE_MODEL
    train_file: str = "/data/train.jsonl"
    eval_file: str = "/data/eval.jsonl"
    out_adapter: str = "/out/neurobait-lora-run3"
    out_dir: str = "/out/outputs"
    epochs: float = 3.0
    max_seq: int = 2048
    seed: int = 42

    @classmethod
    def from_env(cls) -> "TrainConfig":
        return cls(
            base_model=os.environ.get("BASE_MODEL", cls.base_model),
            train_file=os.environ.get("TRAIN_FILE", cls.train_file),
            eval_file=os.environ.get("EVAL_FILE", cls.eval_file),
            out_adapter=os.environ.get("OUT_ADAPTER", cls.out_adapter),
            out_dir=os.environ.get("OUT_DIR", cls.out_dir),
            epochs=float(os.environ.get("EPOCHS", str(cls.epochs))),
            max_seq=int(os.environ.get("MAX_SEQ", str(cls.max_seq))),
            seed=int(os.environ.get("SEED", str(cls.seed))),
        )


def train_adapter(config: TrainConfig | None = None) -> dict[str, Any]:
    """Fine-tune Gemma 4 26B-A4B with 16-bit LoRA and save the adapter."""

    cfg = config or TrainConfig.from_env()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.out_adapter).mkdir(parents=True, exist_ok=True)

    from unsloth import FastModel  # import before transformers / trl / peft

    print(f">>> loading base {cfg.base_model} with 16-bit LoRA", flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    from datasets import load_dataset

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text.removeprefix("<bos>")}

    train_ds = load_dataset("json", data_files=cfg.train_file, split="train").map(format_example)
    eval_ds = None
    if Path(cfg.eval_file).exists():
        eval_ds = load_dataset("json", data_files=cfg.eval_file, split="train").map(format_example)

    print(
        f">>> train={len(train_ds)}"
        + (f" eval={len(eval_ds)}" if eval_ds is not None else " eval=none"),
        flush=True,
    )

    sample = train_ds[0]["text"]
    if INSTRUCTION_PART not in sample or RESPONSE_PART not in sample:
        raise RuntimeError(
            "Gemma 4 train_on_responses_only markers were not found. "
            "Expected '<|turn>user\\n' and '<|turn>model\\n'."
        )
    print(">>> pre-flight marker OK", repr(INSTRUCTION_PART), repr(RESPONSE_PART), flush=True)

    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=cfg.epochs,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        optim="adamw_8bit",
        logging_steps=1,
        output_dir=cfg.out_dir,
        report_to="none",
        seed=cfg.seed,
        save_strategy="no",
    )
    if eval_ds is not None:
        training_args.update(per_device_eval_batch_size=1, eval_strategy="epoch")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq,
        args=TrainingArguments(**training_args),
    )

    from unsloth.chat_templates import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    stats = trainer.train()
    eval_metrics = trainer.evaluate() if eval_ds is not None else {}

    model.save_pretrained(cfg.out_adapter)
    tokenizer.save_pretrained(cfg.out_adapter)

    summary = {
        **asdict(cfg),
        "n_train": len(train_ds),
        "n_eval": len(eval_ds) if eval_ds is not None else 0,
        "train_loss": stats.metrics.get("train_loss"),
        "eval": eval_metrics,
        "expected_steps": ((len(train_ds) + 7) // 8) * int(cfg.epochs),
    }
    summary_path = Path(cfg.out_adapter) / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f">>> adapter saved to {cfg.out_adapter}", flush=True)
    return summary


def main() -> None:
    train_adapter()


if __name__ == "__main__":
    main()
