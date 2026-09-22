# QwerySmith v3.0 — T4 Training Notebook (Colab)
#
# Executes the pinned QLoRA config produced by `qwery_smith train --dry-run`.
# Runtime: T4 GPU. Everything is config-driven: the notebook reads the YAML,
# loads the triples JSONL, trains, and saves the adapter to Drive/HF.
#
# Cell 1 — setup (run first)
# Cell 2 — train one seed
# Cell 3 — train remaining seeds (loop)
# Cell 4 — sanity-check the adapter

# ============================== Cell 1 ==============================
# !pip install -q unsloth trl datasets peft transformers accelerate bitsandbytes
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# REPO = "/content/drive/MyDrive/qwerysmith_v3"   # repo synced to Drive
# import sys; sys.path.insert(0, REPO)
# from qwery_smith.training import train_from_config
#
# # verify CUDA + T4
# import torch
# assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
# print(torch.cuda.get_device_name(0), "| VRAM:", torch.cuda.get_device_properties(0).total_memory // 2**20, "MB")

# ============================== Cell 2 ==============================
# CONFIG = f"{REPO}/runs/olist/train_<TS>/qlora_seed1.yaml"
# TRIPLES = f"{REPO}/datasets/olist/prepared/triples_seed42.jsonl"
# OUT = f"{REPO}/runs/olist/adapters"
#
# adapter_path = train_from_config(CONFIG, TRIPLES, OUT)
# print("saved:", adapter_path)

# ============================== Cell 3 ==============================
# for seed in [2, 3]:
#     cfg = f"{REPO}/runs/olist/train_<TS>/qlora_seed{seed}.yaml"
#     train_from_config(cfg, TRIPLES, OUT)

# ============================== Cell 4 ==============================
# # contract sanity: load adapter, run one prompt, verify SQL:/ANSWER: shape
# from unsloth import FastLanguageModel
# import yaml, json
#
# cfg = yaml.safe_load(open(CONFIG))
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=cfg["base_model"],
#     adapter_name=f"{OUT}/adapter_seed1",
#     max_seq_length=cfg["batch"]["max_len"],
#     load_in_4bit=True,
# )
# FastLanguageModel.for_inference(model)
#
# triple = json.loads(open(TRIPLES).readline())
# inputs = tokenizer(triple["prompt"] + "\n\nASSISTANT:\n", return_tensors="pt").to("cuda")
# out = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.8)
# text = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
# print(text)
# assert "SQL:" in text, "adapter does not follow output contract"