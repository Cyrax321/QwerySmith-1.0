# QwerySmith: Production Text-to-SQL Model Family

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-QwerySmith--1.1-blue)](https://huggingface.co/Cyrax321/QwerySmith-1.1)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-QwerySmith--1.0-blue)](https://huggingface.co/Cyrax321/QwerySmith-1.0)
[![Model Quantization](https://img.shields.io/badge/Format-GGUF%20%7C%20Merged%2016bit-green)](https://huggingface.co/Cyrax321/QwerySmith-1.0-GGUF)
[![Framework](https://img.shields.io/badge/Fine--Tuning-Unsloth%20%2B%20TRL-orange)](https://github.com/unslothai/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QwerySmith** is an open-source Text-to-SQL research and production ecosystem built on **Qwen3-4B** using **QLoRA** via [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl). It encompasses the full lifecycle of specialized language modeling: multi-source curriculum fine-tuning, catastrophic forgetting diagnosis, leak-proof held-out evaluation, and an interactive self-healing database agent with execution verification.

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

### The Single-Source Overfitting Problem (v1.0 Diagnosis)
In **QwerySmith 1.0**, the model was trained exclusively on 10,000 rows of `b-mc2/sql-create-context`. While in-distribution execution accuracy surged to **88.5%**, the model **regressed by -18%** on external test queries relative to the untouched base model:

| Evaluation Split | Base Model (3-Shot) | QwerySmith 1.0 (Fine-Tuned) | Outcome |
|---|:---:|:---:|---|
| **In-Distribution** (`sql-create-context`) | 60.7% | **88.5%** | 🔥 +21% improvement |
| **External** (`gretelai/synthetic_text_to_sql`) | 52.3% | **34.6%** | ❌ **14 wins vs 52 losses** |

**Root Cause**: The model memorized narrow dataset artifacts (e.g. single tables named `table_name_XX`, string-quoted numbers like `rank = "31"`), causing catastrophic forgetting of multi-table joins and standard SQL data types.

### 2. Detailed QwerySmith 1.1 Metrics with 95% Confidence Intervals

| Test Split | System | Valid SQL | Exact Match | Execution Acc (95% CI) | Scored Items |
|:---|:---|:---:|:---:|:---:|:---:|
| **`in_dist`** | Base (Zero-Shot)<br>Base (3-Shot)<br>**QwerySmith 1.1** | 98.5%<br>98.0%<br>**98.0%** | 6.0%<br>7.0%<br>**84.5%** | 67.2% (54.7% to 77.7%)<br>67.2% (54.7% to 77.7%)<br>**88.5% (78.2% to 94.3%)** | 61/200 |
| **`gretel_test`** | Base (Zero-Shot)<br>Base (3-Shot)<br>**QwerySmith 1.1** | 92.7%<br>88.7%<br>**90.7%** | 26.0%<br>26.3%<br>**32.0%** | 52.3% (46.7% to 58.0%)<br>47.3% (41.7% to 53.0%)<br>**55.7% (50.0% to 61.2%)** | 298/300 |
| **`heldout_sqale`** | Base (Zero-Shot)<br>Base (3-Shot)<br>**QwerySmith 1.1** | 78.7%<br>70.2%<br>**80.9%** | 12.8%<br>12.8%<br>**10.6%** | 50.0% (35.8% to 64.2%)<br>40.9% (27.7% to 55.6%)<br>**45.5% (31.7% to 59.9%)** | 44/47 |
| **`heldout_large_schema`**| Base (Zero-Shot)<br>Base (3-Shot)<br>**QwerySmith 1.1** | 61.0%<br>53.9%<br>**55.8%** | 1.9%<br>2.6%<br>**1.9%** | 21.8% (15.8% to 29.3%)<br>19.7% (14.0% to 27.0%)<br>**17.6% (12.2% to 24.7%)** | 142/154 |

> **Key Research Finding — The Few-Shot Paradox:**
> Providing 3-shot prompt exemplars to the base model caused prompt dilution and degraded accuracy across real-world enterprise queries (dropping from 52.3% to 47.3% on Gretel, and 50.0% to 40.9% on SQaLe). Fine-tuning embedded SQL syntax rules permanently into the weights, achieving superior accuracy with **zero additional prompt tokens or latency overhead**.

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
