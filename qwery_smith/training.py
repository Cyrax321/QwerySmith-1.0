"""QLoRA training (plan §3.3): pinned config + T4/Colab trainer.

Local machines (no CUDA) only *prepare* runs: validate triples, write the
pinned config, emit the exact command. The Colab notebook/agent executes the
same config. Hyperparameters are FULLY PINNED (§3.3) — the only delta between
row 1 and row 2 of the matrix is the adapter produced here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import DatasetConfig
from .exceptions import HarnessError

QLORA_DEFAULTS = {
    "base_model": "Qwen/Qwen3-8B",
    "load_in_4bit": True,          # NF4 + double quantization
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
    "optim": {"lr": 1.0e-4, "schedule": "cosine", "warmup_ratio": 0.03, "epochs": 3},
    "batch": {"per_device": 1, "grad_accum": 16, "max_len": 4096},
    "packing": True,
    # decoding contract (§3.3): non-thinking mode, Qwen card defaults.
    # greedy is OFF the table for Qwen3 thinking; non-thinking T=0.7/top_p=0.8.
    "decoding": {"mode": "non-thinking", "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                 "max_new_tokens": 1024},
}


def qlora_config_yaml(cfg: DatasetConfig, seed: int, **overrides: Any) -> str:
    d = dict(QLORA_DEFAULTS)
    d["dataset"] = cfg.name
    d["seed"] = seed
    d.update({k: v for k, v in overrides.items() if k not in ("lora", "optim", "batch", "decoding")})
    for section in ("lora", "optim", "batch", "decoding"):
        if section in overrides:
            d[section] = {**d[section], **overrides[section]}
    return yaml.safe_dump(d, sort_keys=True)


@dataclass
class TrainInstance:
    prompt: str
    target: str

    @property
    def text(self) -> str:
        return f"{self.prompt}\n\nASSISTANT:\n{self.target}"


def prepare_training_run(
    cfg: DatasetConfig,
    triples_path: Path,
    seeds: list[int],
    output_dir: Path,
    epochs: int = 3,
    lr: float = 1.0e-4,
) -> dict[str, Any]:
    """Validate triples, write per-seed pinned configs + run manifest."""
    if not Path(triples_path).exists():
        raise HarnessError(f"triples not found: {triples_path}")

    n_triples = 0
    with open(triples_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                for k in ("prompt", "target", "kind"):
                    if k not in d:
                        raise HarnessError(f"malformed triple: missing {k}")
                n_triples += 1
    if n_triples == 0:
        raise HarnessError("no triples found")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"train_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_configs = []
    for seed in seeds:
        cfg_text = qlora_config_yaml(cfg, seed, epochs=epochs, lr=lr)
        cfg_path = run_dir / f"qlora_seed{seed}.yaml"
        cfg_path.write_text(cfg_text, encoding="utf-8")
        seed_configs.append({"seed": seed, "config": str(cfg_path)})

    manifest = {
        "dataset": cfg.name,
        "dataset_version": "1.0.0",
        "triples_path": str(triples_path),
        "n_triples": n_triples,
        "seeds": seeds,
        "hardware_target": "T4 16GB (Colab) — Unsloth fast path",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configs": seed_configs,
        "defaults": QLORA_DEFAULTS,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "n_triples": n_triples, "seeds": seeds,
            "config_path": seed_configs[0]["config"]}


# ---------------------------------------------------------------- trainer ---
# Executed on Colab where CUDA + unsloth exist. Kept in one function so the
# notebook cell is a single call: python -m qwery_smith.training --config X


def train_from_config(config_path: Path, triples_path: Path, adapter_out: Path) -> Path:
    """The actual QLoRA run. Import errors here are expected on CPU-only machines."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainingArguments

    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = cfg["seed"]

    torch.manual_seed(seed)
    import random as _r

    _r.seed(seed)

    # Colab/T4 path: unsloth is required for the fast 4-bit QLoRA route.
    # (Local CPU machines use `prepare_training_run` only — never this function.)
    from unsloth import FastLanguageModel

    lora = cfg["lora"]
    batch = cfg["batch"]
    optim = cfg["optim"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=batch["max_len"],
        dtype=None,               # auto (bf16 on T4 is emulated; unsloth handles)
        load_in_4bit=cfg["load_in_4bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    instances = []
    with open(triples_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                instances.append(TrainInstance(prompt=d["prompt"], target=d["target"]))

    def fmt(inst: TrainInstance):
        return {"text": inst.text}

    ds = Dataset.from_list([fmt(i) for i in instances])

    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(adapter_out),
            per_device_train_batch_size=batch["per_device"],
            gradient_accumulation_steps=batch["grad_accum"],
            num_train_epochs=optim["epochs"],
            learning_rate=optim["lr"],
            lr_scheduler_type=optim["schedule"],
            warmup_ratio=optim["warmup_ratio"],
            logging_steps=10,
            seed=seed,
            report_to=[],
            max_length=batch["max_len"],
            packing=cfg["packing"],
        ),
    )
    trainer.train()

    adapter_path = Path(adapter_out) / f"adapter_seed{seed}"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    return adapter_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--triples", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = train_from_config(Path(args.config), Path(args.triples), Path(args.out))
    print(f"adapter saved: {out}")