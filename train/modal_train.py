"""Modal entrypoints for NeuroBait training, evaluation, merge, and upload."""

from __future__ import annotations

import modal


APP_NAME = "neurobait-small-model"
DEFAULT_LORA_REPO = "USER/neurobait-gemma4-26b-a4b-lora"
DEFAULT_MERGED_REPO = "USER/neurobait-gemma4-26b-a4b"

app = modal.App(APP_NAME)

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
    .add_local_file("train/train_core.py", "/root/neurobait/train_core.py")
    .add_local_file("data/train.jsonl", "/data/train.jsonl")
    .add_local_file("data/eval.jsonl", "/data/eval.jsonl")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("neurobait-out", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=10 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def preflight() -> str:
    """Check the remote Modal runtime before spending a full training run."""

    import json
    import os
    import subprocess

    import unsloth
    import torch
    import datasets
    import peft
    import transformers
    import trl

    nvidia_smi = subprocess.run(
        ["nvidia-smi"],
        check=False,
        capture_output=True,
        text=True,
    )

    gpu = {}
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": f"sm_{capability[0]}{capability[1]}",
            "device_count": torch.cuda.device_count(),
        }

    result = {
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "unsloth": unsloth.__version__,
        "nvidia_smi": nvidia_smi.stdout[-4000:],
        "nvidia_smi_stderr": nvidia_smi.stderr[-1000:],
    }
    return json.dumps(result, indent=2)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train() -> dict:
    import os
    import sys

    sys.path.insert(0, "/root/neurobait")
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    os.environ.setdefault("TRAIN_FILE", "/data/train.jsonl")
    os.environ.setdefault("EVAL_FILE", "/data/eval.jsonl")
    os.environ.setdefault("OUT_ADAPTER", "/out/neurobait-lora-run3")
    os.environ.setdefault("OUT_DIR", "/out/outputs")

    from train_core import train_adapter

    summary = train_adapter()
    out_vol.commit()
    return summary


@app.function(
    image=image,
    timeout=30 * 60,
    volumes={"/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def push_lora_to_hub(repo_id: str = DEFAULT_LORA_REPO) -> str:
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        folder_path="/out/neurobait-lora-run3",
        repo_id=repo_id,
        repo_type="model",
    )
    return repo_id


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def merge_and_push(repo_id: str = DEFAULT_MERGED_REPO) -> str:
    import os

    os.environ["HF_HOME"] = "/root/.cache/huggingface"

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    from huggingface_hub import HfApi

    base_model = os.environ.get("BASE_MODEL", "unsloth/gemma-4-26b-a4b-it")
    adapter_path = os.environ.get("OUT_ADAPTER", "/out/neurobait-lora-run3")
    merged_path = "/out/merged-neurobait-gemma4-26b-a4b"

    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    HfApi().upload_folder(folder_path=merged_path, repo_id=repo_id, repo_type="model")
    out_vol.commit()
    return repo_id


@app.local_entrypoint()
def main(
    run_preflight: bool = False,
    run_train: bool = True,
    push_lora: bool = False,
    merge: bool = False,
    lora_repo: str = DEFAULT_LORA_REPO,
    merged_repo: str = DEFAULT_MERGED_REPO,
) -> None:
    if run_preflight:
        print("preflight:", preflight.remote())
    if run_train:
        summary = train.remote()
        print("train:", summary)
    if push_lora:
        print("lora repo:", push_lora_to_hub.remote(lora_repo))
    if merge:
        print("merged repo:", merge_and_push.remote(merged_repo))
