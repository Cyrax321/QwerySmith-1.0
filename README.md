# QwerySmith: Production Text-to-SQL Model Family

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-QwerySmith--1.1-blue)](https://huggingface.co/Cyrax321/QwerySmith-1.1)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-QwerySmith--1.0-blue)](https://huggingface.co/Cyrax321/QwerySmith-1.0)
[![Model Quantization](https://img.shields.io/badge/Format-GGUF%20%7C%20Merged%2016bit-green)](https://huggingface.co/Cyrax321/QwerySmith-1.0-GGUF)
[![Framework](https://img.shields.io/badge/Fine--Tuning-Unsloth%20%2B%20TRL-orange)](https://github.com/unslothai/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QwerySmith** is an open-source Text-to-SQL research and production ecosystem built on **Qwen3-4B** using **QLoRA** via [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl). It encompasses the full lifecycle of specialized language modeling: multi-source curriculum fine-tuning, catastrophic forgetting diagnosis, leak-proof held-out evaluation, and an interactive self-healing database agent with execution verification.

---

## 📂 Repository Layout

```text
├── QwerySmith/
│   ├── Qwerysmith_V11.py       # v1.1 fine-tuning pipeline with multi-source mixing & ablation
│   ├── data_sources.py         # Multi-dataset registry, split carver & leak-proof sampler
│   └── QwerySmith.py           # v1.0 baseline training and evaluation pipeline
├── agent.py                    # Production Text-to-SQL agent with schema inspection & self-healing
├── qwerysmith_eval.py          # 3,600+ line statistical evaluation & calibration suite
├── tests/
│   └── make_synthetic_run.py   # Offline test harness (evaluates without GPU/model downloads)
└── README.md
```

---

## 🏛️ Model Family & Evolution: v1.0 vs v1.1

QwerySmith is developed across two complementary releases:

---

## 🏛️ Model Family & Evolution: v1.0 vs v1.1

QwerySmith is developed across two complementary releases:

| Feature / Aspect | **QwerySmith 1.0** (Baseline) | **QwerySmith 1.1** (Latest Production) |
|:---|:---|:---|
| **Base Foundation** | `unsloth/Qwen3-4B` | `unsloth/Qwen3-4B` |
| **Training Data** | 10,000 rows `b-mc2/sql-create-context` (single-source) | 10,000 rows balanced 50/50 mix (`sql-create-context` + `synthetic_text_to_sql`) |
| **Splitting Strategy** | Post-hoc random sampling | **Leak-proof pre-split carving** (zero contamination) |
| **LoRA Config** | $r=16, lpha=32$, dropout = 0, LR = `2e-4` | $r=16, lpha=32$, **dropout = 0.05**, **LR = `1e-4`** |
| **In-Dist Execution Acc** | **88.5%** | **88.5%** (Exact Match: **84.5%**, 13-0 record) |
| **Enterprise External Acc** | 34.6% *(Suffered single-source overfit)* | **55.7%** *(+21.1% over v1.0; 43 wins vs 18 losses vs base)* |
| **Syntactic SQL Validity** | High on simple queries | **80.9%** on noisy schemas (`heldout_sqale`), **98.0%** in-dist |
| **Held-Out Generalization** | Not evaluated | Evaluated against unobserved benchmarks (`sqale`, `large_schema`) |

---

## 📊 Comprehensive Benchmark Results

All evaluations use in-memory SQLite instances pre-populated with synthetic or gold `INSERT` rows to measure **execution correctness** (returning identical row sets) rather than mere superficial string matching.

### 1. The Head-to-Head Progression: Base Model vs v1.0 vs v1.1

| Benchmark Split | Base Model (3-Shot) | QwerySmith 1.0 | QwerySmith 1.1 | v1.1 vs Base (Head-to-Head) |
|:---|:---:|:---:|:---:|:---|
| **In-Distribution** (`sql-create-context`) | 67.2% | **88.5%** | **88.5%** *(84.5% EM)* | **13 Wins, 0 Losses** (+21.3% leap; $p < 0.05$) |
| **Enterprise Test** (`synthetic_text_to_sql`) | 47.3% | 34.6% | **55.7%** *(32.0% EM)* | **43 Wins, 18 Losses** (+25 net wins over base) |
| **Noisy Schemas** (`heldout_sqale`) | 40.9% | — | **45.5%** *(80.9% Valid)* | **4 Wins, 2 Losses** (Highest syntax resilience) |
| **Complex Schemas** (`heldout_large_schema`)| 19.7% | — | **17.6%** *(55.8% Valid)* | **10 Wins, 13 Losses** (Overlapping 95% CIs) |

### 2. Detailed QwerySmith 1.1 Metrics with 95% Confidence Intervals

| Test Split | System | Valid SQL | Exact Match | Execution Acc (95% CI) | Scored Items |
|:---|:---|:---:|:---:|:---:|:---:|
| **`in_dist`** | Base (Zero-Shot)<br>Base (3-Shot)<br>**QwerySmith 1.1** | 98.5%<br>98.0%<br>**98.0%** | 6.0%<br>7.0%<br>**84.5%** | 67.2% (54.7% to 77.7%)<br>67.2% (54.7% to 77.7%)<br>**88.5% (78.2% to 94.3%)** | 61/200 |
| **`gretel_test`** | Base (Zero-Shot)<br>Base (3-Shot)<br>**QwerySmith 1.1** | 92.7%<br>88.7%<br>**90.7%** | 26.0%<br>26.3%<br>**32.0%** | 52.3% (46.7% to 58.0%)<br>47.3% (41.7% to 53.0%)<br>**55.7% (50.0% to 61.2%)** | 298/300 |
| **`heldout_sqale`** | Base (Zero-Shot)<br>Base (3-Shot)<br>**QwerySmith 1.1** | 78.7%<br>70.2%<br>**80.9%** | 12.8%<br>12.8%<br>**10.6%** | 50.0% (35.8% to 64.2%)<br>40.9% (27.7% to 55.6%)<br>**45.5% (31.7% to 59.9%)** | 44/47 |
| **`heldout_large_schema`**| Base (Zero-Shot)<br>Base (3-Shot)<br>**QwerySmith 1.1** | 61.0%<br>53.9%<br>**55.8%** | 1.9%<br>2.6%<br>**1.9%** | 21.8% (15.8% to 29.3%)<br>19.7% (14.0% to 27.0%)<br>**17.6% (12.2% to 24.7%)** | 142/154 |

> **Key Research Finding — The Few-Shot Paradox:**
> Providing 3-shot prompt exemplars to the base model caused prompt dilution and degraded accuracy across real-world enterprise queries (dropping from 52.3% to 47.3% on Gretel, and 50.0% to 40.9% on SQaLe). Fine-tuning embedded SQL syntax rules permanently into the weights, achieving superior accuracy with **zero additional prompt tokens or latency overhead**.



---

## ⚡ Published Models on Hugging Face

All models are published in three optimized formats to support production, edge, and researcher workflows:

### QwerySmith 1.1 (Recommended)
| Format | Repository | Size | Ideal Use Case |
|---|---|---|---|
| **LoRA Adapter** | [`Cyrax321/QwerySmith-1.1`](https://huggingface.co/Cyrax321/QwerySmith-1.1) | ~132 MB | Fast fine-tuning & inference via Unsloth / PEFT |
| **Merged 16-Bit** | [`Cyrax321/QwerySmith-1.1-Merged`](https://huggingface.co/Cyrax321/QwerySmith-1.1-Merged) | ~8.06 GB | Standalone deployment via vLLM, TGI, or Hugging Face Pipelines |
| **Quantized GGUF** | [`Cyrax321/QwerySmith-1.1-GGUF`](https://huggingface.co/Cyrax321/QwerySmith-1.1-GGUF) | ~2.5 GB | Ultra-fast local execution with Ollama or `llama.cpp` |

### QwerySmith 1.0 (Baseline)
| Format | Repository | Size | Ideal Use Case |
|---|---|---|---|
| **LoRA Adapter** | [`Cyrax321/QwerySmith-1.0`](https://huggingface.co/Cyrax321/QwerySmith-1.0) | ~132 MB | Legacy baseline adapter |
| **Merged 16-Bit** | [`Cyrax321/QwerySmith-1.0-Merged`](https://huggingface.co/Cyrax321/QwerySmith-1.0-Merged) | ~8.06 GB | Standalone v1.0 checkpoint |
| **Quantized GGUF** | [`Cyrax321/QwerySmith-1.0-GGUF`](https://huggingface.co/Cyrax321/QwerySmith-1.0-GGUF) | ~2.5 GB | Local baseline GGUF |

---

## 💻 Quickstart & Inference

### 1. Fast 4-bit Inference with Unsloth (v1.1 or v1.0)

```python
from unsloth import FastLanguageModel

# Load QwerySmith 1.1 (or switch to "Cyrax321/QwerySmith-1.0")
model, tokenizer = FastLanguageModel.from_pretrained(
    "Cyrax321/QwerySmith-1.1",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

prompt = """<|im_start|>system
You are a text-to-SQL assistant. Given a database schema and a question, reply with exactly one SQL query and nothing else.<|im_end|>
<|im_start|>user
Schema: CREATE TABLE orders (order_id INT, customer_id INT, amount DECIMAL(10,2), status VARCHAR);
Question: What is the total revenue from completed orders?<|im_end|>
<|im_start|>assistant
<think>
</think>
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
print(tokenizer.batch_decode(outputs)[0])
```

### 2. Local Edge Inference with Ollama

```bash
# Run QwerySmith 1.1 locally (CPU or GPU)
ollama run Cyrax321/QwerySmith-1.1-GGUF

# Or run QwerySmith 1.0 baseline
ollama run Cyrax321/QwerySmith-1.0-GGUF
```

---

---

---

## ⚡ Hardware Resource Profile

Training and inference resource footprint measured on standard cloud hardware:

| Phase | Device | Precision | VRAM / RAM | Speed / Throughput |
|:---|:---|:---|:---|:---|
| **Training (QLoRA)** | 1x Tesla T4 (16GB) | 4-bit Base + 16-bit LoRA | 4.5 GB GPU / 4.1 GB RAM | 2.44 samples/sec (~68 min / epoch) |
| **Inference (Unsloth)** | 1x Tesla T4 (16GB) | 4-bit BitsAndBytes | ~3.1 GB GPU | ~45 tokens/sec |
| **Edge / Local (GGUF)** | CPU (Apple M-Series / x86) | Q4_K_M Quantized | ~2.6 GB System RAM | ~35 tokens/sec |

## 🛠️ Training Both Versions (Google Colab / Linux GPU)

Recommended runtime: **Tesla T4 GPU** (Google Colab free tier is fully sufficient).

### 1. Install Dependencies
```bash
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets matplotlib
```

### 2. Train QwerySmith 1.0 (Baseline Pipeline)
```bash
python QwerySmith/QwerySmith.py --stage all --out runs/qwerysmith-1.0
```

### 3. Train QwerySmith 1.1 (Multi-Source Production Pipeline)
```bash
python QwerySmith/Qwerysmith_V11.py --stage all --out runs/qwerysmith-1.1 \
    --mix sql_create_context:5000,gretel:5000 \
    --heldout sqale,large_schema \
    --lr 1e-4 --dropout 0.05
```

---

---

## 🤖 Interactive Self-Healing Agent (`agent.py`)

`agent.py` provides a production-ready interface connecting QwerySmith to SQLite or PostgreSQL databases:

```python
from agent import QwerySmithAgent

agent = QwerySmithAgent(db_path="company.db")
result = agent.query("Find all customers who made more than 3 purchases this year.")
print("Generated SQL:", result["sql"])
print("Query Result:", result["rows"])
```

### Agent Capabilities:
- **Schema Introspection**: Automatically reads database catalogs to extract table names, column types, foreign keys, and primary keys.
- **Self-Healing Loop**: If a generated query causes an execution error (e.g., column mislabeling or invalid join), the agent captures the database error traceback and prompts the model to self-correct.
- **Lenient Output Sanitizer**: Automatically cleans `<think>...</think>` tokens, markdown fences, and explanatory chatter.

---

## 📈 Research & Evaluation Toolkit (`qwerysmith_eval.py`)

`qwerysmith_eval.py` is an evaluation framework that produces publication-ready figures, confidence intervals, and statistical tests:

```bash
# Generate all figures, tables, and REPORT.md from cached predictions (CPU-only, seconds)
python qwerysmith_eval.py --stage figures --out runs/v11-C

# Run GPU-based calibration and confidence scoring
python qwerysmith_eval.py --stage confidence --out runs/v11-C

# Package everything into a downloadable zip
python qwerysmith_eval.py --stage all --out runs/v11-C
```

### Metrics Produced:
- **Execution-Based Scoring**: Result-set equality against gold SQL using in-memory SQLite instances.
- **Statistical Significance**: Paired bootstrap confidence intervals, Wilson score intervals, and McNemar test.
- **Inter-Rater Reliability**: Cohen's kappa, Fleiss' kappa, and Gwet's AC1.
- **Calibration & Uncertainty**: Expected Calibration Error (ECE), Brier score, and reliability diagrams.
- **AST Analysis**: Clause-level confusion matrices (WHERE, GROUP BY, HAVING, ORDER BY, JOIN) and schema-linking P/R/F1.

---

## ⚙️ CLI Reference (`Qwerysmith_V11.py`)

| Argument | Default | Description |
|---|---|---|
| `--stage` | `all` | Pipeline stage: `baseline`, `train`, `eval`, `report`, `export`, or `all` |
| `--model` | `unsloth/Qwen3-4B` | Base foundation model |
| `--mix` | `sql_create_context:5000,gretel:5000` | Comma-separated `source:count` pairs for training mix |
| `--heldout` | `sqale,large_schema` | Comma-separated sources held out strictly for generalization testing |
| `--n-test` | `200` | Number of in-distribution test samples |
| `--n-external` | `300` | Number of near-distribution Gretel test samples |
| `--n-heldout` | `300` | Number of test samples per held-out source |
| `--lr` | `1e-4` | Learning rate (lowered from 2e-4 to prevent overfitting) |
| `--dropout` | `0.05` | LoRA dropout probability |
| `--rank` | `16` | LoRA rank dimension (`lora_alpha = 32`) |
| `--batch-size` | `2` | Per-device batch size |
| `--grad-accum` | `8` | Gradient accumulation steps (effective batch size = 16) |
| `--save-steps` | `0` | Checkpoint frequency (`0` = save only at end) |
| `--smoke` | `False` | Fast sanity check with miniature data mix |
| `--merge` | `False` | Export merged 16-bit standalone model |
| `--gguf` | `False` | Export quantized GGUF (`q4_k_m`) |
| `--push` | `""` | Hugging Face repository ID to push adapter |

---

## 📜 License

This project is licensed under the MIT License.
