# QwerySmith 1.0

**QwerySmith 1.0** is an end-to-end Text-to-SQL fine-tuning and evaluation pipeline leveraging **Qwen3-4B** and **QLoRA** via [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl).

---

## ⚡ Key Features

- **Multi-Stage Workflow (`--stage`)**:
  - `baseline`: Zero-shot and 3-shot evaluation of the untouched base model.
  - `train`: Memory-efficient QLoRA fine-tuning with prompt response loss masking.
  - `eval`: High-throughput batched inference on the fine-tuned adapter.
  - `report`: Automated scoring with Wilson score 95% confidence intervals, win/loss breakdowns, CSV outputs, and comparison charts.
  - `export`: Export to merged 16-bit weights, quantized GGUF (`q4_k_m`), or upload directly to Hugging Face Hub.
- **Dual Evaluation Suites**:
  - **In-Distribution (`b-mc2/sql-create-context`)**: 200 held-out test examples.
  - **External / Out-of-Distribution (`gretelai/synthetic_text_to_sql`)**: 300 test items with real `INSERT` data to verify query execution under real-world conditions.
- **Execution-Based Scoring**:
  - In-memory SQLite evaluation engine.
  - Synthetic table data population seeded with gold query literals for empty schema testing.
  - Compares canonical result tuples rather than brittle syntax matching.
- **Fault-Tolerant Caching**:
  - Caches intermediate predictions and model artifacts in `--out`.
  - Re-running the pipeline skips finished stages seamlessly.

---

## 🛠️ Installation & Requirements

Recommended: Run on GPU (e.g., Google Colab T4 / A100 or local NVIDIA GPU).

```bash
git clone https://github.com/Cyrax321/QwerySmith-1.0.git
cd QwerySmith-1.0
pip install -r requirements.txt
```

---

## 🚀 Quickstart

### 1. Smoke Test (5–10 min sanity check)
Run a quick end-to-end check with a smaller subset:
```bash
python QwerySmith.py --smoke
```

### 2. Full Training & Evaluation Pipeline
Run the complete pipeline across all stages:
```bash
python QwerySmith.py --stage all --n-train 10000
```

### 3. Running Specific Stages
You can execute individual stages independently:

- **Baseline evaluation only:**
  ```bash
  python QwerySmith.py --stage baseline
  ```

- **Train adapter only:**
  ```bash
  python QwerySmith.py --stage train --n-train 10000 --epochs 1 --lr 2e-4
  ```

- **Evaluate adapter:**
  ```bash
  python QwerySmith.py --stage eval
  ```

- **Generate report and comparison figures:**
  ```bash
  python QwerySmith.py --stage report
  ```

- **Export & Push to Hugging Face Hub:**
  ```bash
  export HF_TOKEN="your_hf_write_token"
  python QwerySmith.py --stage export --merge --gguf --push your-username/QwerySmith-1.0
  ```

---

## ⚙️ CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--stage` | `all` | Pipeline stage: `baseline`, `train`, `eval`, `report`, `export`, or `all` |
| `--model` | `unsloth/Qwen3-4B` | Base model repository |
| `--out` | `runs/qwerysmith-1.0` | Output directory for checkpoints, predictions, and reports |
| `--n-train` | `10000` | Number of training examples (`0` for all available) |
| `--n-test` | `200` | Number of in-distribution test samples |
| `--n-external` | `300` | Number of external test samples (`0` to disable) |
| `--epochs` | `1` | Training epochs |
| `--batch-size` | `2` | Per-device training batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--rank` | `16` | LoRA rank dimension |
| `--lr` | `2e-4` | Learning rate |
| `--max-len` | `2048` | Maximum sequence length |
| `--gen-batch` | `16` | Batch size for inference generation |
| `--smoke` | `False` | Run tiny smoke test to verify execution |
| `--merge` | `False` | Export merged 16-bit model |
| `--gguf` | `False` | Export quantized GGUF (`q4_k_m`) |
| `--push` | `""` | Hugging Face repo ID to upload adapter |

---

## 📊 Metrics & Reports

Outputs generated in `--out`:
- `results.md`: Markdown summary table with valid SQL %, exact match %, and execution accuracy (with 95% Wilson CI).
- `results.json`: Raw metric scores in JSON format.
- `predictions.csv`: Side-by-side predictions from `base_zeroshot`, `base_fewshot`, and `finetuned` systems.
- `comparison.png`: Visual bar chart comparing system accuracies across datasets.

---

## 📜 License

MIT License.
