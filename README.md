# QwerySmith: Production Text-to-SQL Model Family

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-QwerySmith--1.0-blue)](https://huggingface.co/Cyrax321/QwerySmith-1.0)
[![Model Quantization](https://img.shields.io/badge/Format-GGUF%20%7C%20Merged%2016bit-green)](https://huggingface.co/Cyrax321/QwerySmith-1.0-GGUF)
[![Framework](https://img.shields.io/badge/Fine--Tuning-Unsloth%20%2B%20TRL-orange)](https://github.com/unslothai/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QwerySmith** is an end-to-end Text-to-SQL research and production system built on **Qwen3-4B** using **QLoRA** via [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl). It features an interactive self-healing database agent, a multi-source data-mixing training pipeline, and an enterprise evaluation suite with execution-based verification.

---

## 📂 Repository Structure

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

## 🔬 What's New in QwerySmith 1.1

### The Single-Source Overfitting Problem (v1.0 Diagnosis)
In **QwerySmith 1.0**, the model was trained exclusively on 10,000 rows of `b-mc2/sql-create-context`. While in-distribution execution accuracy surged to **88.5%**, the model **regressed by -18%** on external test queries relative to the untouched base model:

| Evaluation Split | Base Model (3-Shot) | QwerySmith 1.0 (Fine-Tuned) | Outcome |
|---|:---:|:---:|---|
| **In-Distribution** (`sql-create-context`) | 60.7% | **88.5%** | 🔥 +21% improvement |
| **External** (`gretelai/synthetic_text_to_sql`) | 52.3% | **34.6%** | ❌ **14 wins vs 52 losses** |

**Root Cause**: The model memorized narrow dataset artifacts (e.g. single tables named `table_name_XX`, string-quoted numbers like `rank = "31"`), causing catastrophic forgetting of multi-table joins and standard SQL data types.

### How v1.1 Solves It:
1. **Multi-Source Data Mixing (`--mix`)**: Draws a configurable mix across sources (`sql_create_context:5000,gretel:5000`), forcing the model to learn 100+ realistic business schemas, correct column linking, and real numeric types.
2. **Leak-Proof Evaluation Splits**: Evaluation sets (`in_dist`, `gretel_test`, `heldout_*`) are carved out **before** constructing the training mix. Any question present in any eval set is strictly excluded from training.
3. **True Out-of-Distribution Generalization (`--heldout`)**: Benchmarked against sources that **never** appear in training (`sqale`, `large_schema`).
4. **Regularization & Recipe Ablation**: Configurable LoRA dropout (`--dropout 0.05`), lowered learning rate (`--lr 1e-4`), and periodic checkpointing (`--save-steps`).

---

## ⚡ Published Models on Hugging Face

QwerySmith 1.0 is published in three formats for different deployment scenarios:

| Format | Repository | Size | Ideal Use Case |
|---|---|---|---|
| **LoRA Adapter** | [`Cyrax321/QwerySmith-1.0`](https://huggingface.co/Cyrax321/QwerySmith-1.0) | 132 MB | Lightweight fine-tuning weights for Unsloth / PEFT |
| **Merged 16-Bit** | [`Cyrax321/QwerySmith-1.0-Merged`](https://huggingface.co/Cyrax321/QwerySmith-1.0-Merged) | 8.06 GB | Standalone deployment via vLLM, TGI, or Transformers |
| **Quantized GGUF** | [`Cyrax321/QwerySmith-1.0-GGUF`](https://huggingface.co/Cyrax321/QwerySmith-1.0-GGUF) | 2.5 GB | Local CPU/GPU inference via `llama.cpp` or Ollama |

Run locally with Ollama:
```bash
ollama run Cyrax321/QwerySmith-1.0-GGUF
```

---

## 🚀 Quickstart & Training (Google Colab / Linux GPU)

Recommended runtime: **T4 GPU** (free tier is fully sufficient).

### 1. Install Dependencies
```bash
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets matplotlib
```

### 2. Run a 5-Minute Smoke Test
Verify model loading, schema extraction, training masking, and evaluation end-to-end:
```bash
!python QwerySmith/Qwerysmith_V11.py --smoke
```

### 3. Run Experimental Ablations

- **Run B (Data Mix Ablation)** — Isolates the effect of 50/50 data mixing:
  ```bash
  !python QwerySmith/Qwerysmith_V11.py --out runs/v11-B \
      --mix sql_create_context:5000,gretel:5000 \
      --lr 2e-4 --dropout 0 --heldout sqale,large_schema
  ```

- **Run C (Data Mix + Regularized Recipe)** — Recommended full pipeline with lower LR and LoRA dropout:
  ```bash
  !python QwerySmith/Qwerysmith_V11.py --out runs/v11-C \
      --mix sql_create_context:5000,gretel:5000 \
      --lr 1e-4 --dropout 0.05 --save-steps 100 --heldout sqale,large_schema
  ```

---

## 🤖 Interactive Self-Healing Agent (`agent.py`)

`agent.py` provides a production-ready wrapper that connects QwerySmith to any SQLite or PostgreSQL database:

```python
from agent import QwerySmithAgent

# Connect to any business database
agent = QwerySmithAgent(db_path="ecommerce.db")

# Ask questions in plain English
result = agent.query("What are the top 5 customers by revenue this month?")
print("Generated SQL:", result["sql"])
print("Query Result:", result["rows"])
```

### Agent Features:
- **Introspection**: Automatically extracts table schemas, foreign keys, and indexes.
- **Self-Healing Loop**: If a query fails execution (syntax error, missing join), the agent inspects the database error, feeds the traceback back to the LLM, and regenerates a corrected query.
- **Lenient Formatting**: Automatically strips thinking blocks (`<think>...</think>`), markdown fences, and explanatory chatter.

---

## 📊 Comprehensive Evaluation Suite (`qwerysmith_eval.py`)

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
