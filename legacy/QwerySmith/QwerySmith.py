#!/usr/bin/env python3
"""
Qwerysmith 1.0: text-to-SQL fine-tuning pipeline (Qwen3-4B + QLoRA via Unsloth).

Stages (run one with --stage, or everything with --stage all):
  baseline  evaluate the untouched base model (zero-shot and 3-shot)
  train     QLoRA fine-tune, save the adapter
  eval      evaluate the fine-tuned adapter
  report    build the comparison table, CSV of predictions, and a chart
  export    optional: merged 16-bit model, GGUF, push to the Hugging Face Hub

Colab usage (T4 is enough):
  !pip install -q unsloth trl datasets
  !python QwerySmith.py --smoke            # 5-10 min sanity check first
  !python QwerySmith.py --n-train 10000    # real run

Everything is cached in --out, so if Colab dies you can re-run the same
command and finished steps are skipped. Point --out at Google Drive
(from google.colab import drive; drive.mount("/content/drive")) so the
results survive the session.

Two evaluation sets:
  in_dist   200 held-out rows from b-mc2/sql-create-context (same style as training)
  external  rows from gretelai/synthetic_text_to_sql test split (different data,
            uses its own INSERT rows when an example has them)

Library APIs change often. If a call errors, check the Unsloth / TRL docs first.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

SEED = 42
MODEL_NAME = "Qwerysmith-1.0"
MODEL_DEFAULT = "unsloth/Qwen3-4B"
IN_DIST_DATASET = "b-mc2/sql-create-context"      # columns: question, context, answer
EXTERNAL_DATASET = "gretelai/synthetic_text_to_sql"  # columns: sql_prompt, sql_context, sql

SYSTEM = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "reply with exactly one SQL query and nothing else."
)

# Qwen3 chat-format markers, used to (a) end each training example and (b) mask
# the loss so the model is only trained on the SQL answer. If you switch to a
# different model family, change these three.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
END = "<|im_end|>"

SYSTEMS = ["base_zeroshot", "base_fewshot", "finetuned"]


# --------------------------------------------------------------------------
# 0. Installation
# --------------------------------------------------------------------------
def ensure_installed() -> None:
    """Install dependencies if missing, then restart this script."""
    import importlib.util

    needed = ["unsloth", "trl", "datasets"]
    missing = [p for p in needed if importlib.util.find_spec(p) is None]
    if not missing:
        return
    if os.environ.get("_T2S_REEXEC"):
        sys.exit(f"Could not install: {missing}. Install them manually and re-run.")
    print(f"Installing missing packages: {missing} (takes a few minutes)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "unsloth", "trl", "datasets", "matplotlib"]
    )
    os.environ["_T2S_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


# --------------------------------------------------------------------------
# 1. SQLite helpers (used for execution-based scoring)
# --------------------------------------------------------------------------
def split_statements(sql: str) -> list[str]:
    out, buf = [], ""
    for part in sql.split(";"):
        buf += part
        if sqlite3.complete_statement(buf + ";"):
            if buf.strip():
                out.append(buf.strip())
            buf = ""
        else:
            buf += ";"
    if buf.strip():
        out.append(buf.strip().rstrip(";"))
    return out


def schema_only(context: str) -> str:
    """Keep only CREATE TABLE / CREATE VIEW statements (the model never sees INSERT rows)."""
    keep = [s + ";" for s in split_statements(context) if re.match(r"(?is)^create\s+(table|view)\b", s)]
    return "\n".join(keep) if keep else context.strip()


def _literals(sql: str):
    strs = re.findall(r"'([^']*)'", sql)
    nums = []
    for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", sql):
        nums.append(float(x) if "." in x else int(x))
    return strs, nums


def populate_empty_tables(conn: sqlite3.Connection, gold: str, seed: int, n_rows: int = 40) -> None:
    """Fill tables that have no rows with random data, seeded with the gold
    query's own literals so that WHERE filters and joins actually match rows."""
    rng = random.Random(seed)
    strs, nums = _literals(gold)
    text_pool = strs + ["alpha", "beta", "gamma", "delta", "north", "south", "x", "y"]
    date_pool = [s for s in strs if re.match(r"\d{4}-\d{2}", s)]
    int_pool = [n for n in nums if isinstance(n, int)] + list(range(1, 11))
    real_pool = [float(n) for n in nums] + [round(rng.uniform(1, 100), 2) for _ in range(10)]
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
    except sqlite3.Error:
        return
    for t in tables:
        try:
            if conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] > 0:
                continue
            cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
        except sqlite3.Error:
            continue
        rows = []
        for _ in range(n_rows):
            row = []
            for c in cols:
                ty = (c[2] or "").upper()
                if "INT" in ty:
                    row.append(rng.choice(int_pool))
                elif any(k in ty for k in ("REAL", "FLOA", "DOUB", "DEC", "NUM")):
                    row.append(rng.choice(real_pool))
                elif "DATE" in ty or "TIME" in ty:
                    if date_pool and rng.random() < 0.5:
                        row.append(rng.choice(date_pool))
                    else:
                        row.append(f"20{rng.randint(20, 24)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}")
                else:
                    row.append(rng.choice(text_pool))
            rows.append(row)
        try:
            conn.executemany(f'INSERT INTO "{t}" VALUES ({",".join("?" * len(cols))})', rows)
        except sqlite3.Error:
            pass


def make_db(context: str, gold: str, seed: int) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    for stmt in split_statements(context):
        try:
            conn.execute(stmt)
        except sqlite3.Error:
            pass
    populate_empty_tables(conn, gold, seed)
    return conn


def run_query(conn: sqlite3.Connection, sql: str, timeout: float = 3.0):
    if not re.match(r"(?is)^\s*(select|with)\b", sql or ""):
        return False, None  # empty output, chatter, or non-SELECT statements count as invalid
    start = time.time()
    conn.set_progress_handler(lambda: 1 if time.time() - start > timeout else 0, 10000)
    try:
        return True, conn.execute(sql).fetchmany(1000)
    except Exception:
        return False, None
    finally:
        conn.set_progress_handler(None, 0)


def _canon(rows) -> list[str]:
    return sorted(
        repr(tuple(round(v, 4) if isinstance(v, float) else v for v in r)) for r in rows
    )


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").lower()).strip()


def clean_sql(text: str) -> str:
    """Lenient extraction so the base model is not punished for chatty formatting."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    m = re.search(r"(?is)```(?:sql)?\s*(.*?)```", text)
    if m:
        text = m.group(1)
    text = text.strip()
    m = re.search(r"(?is)\b(select|with)\b", text)
    if m:
        text = text[m.start():]
    stmts = split_statements(text)
    return (stmts[0] if stmts else text).strip()


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def score(items: list[dict], preds: list[str]):
    """Returns (metrics, per_item). Execution accuracy only counts items whose
    gold query executes and returns at least one row."""
    valid = em = ex = scored = 0
    per = []
    for i, (it, pred) in enumerate(zip(items, preds)):
        conn = make_db(it["context"], it["gold"], seed=i)
        ok_p, rows_p = run_query(conn, pred)
        ok_g, rows_g = run_query(conn, it["gold"])
        conn.close()
        v = bool(ok_p)
        e = _norm(pred) == _norm(it["gold"])
        x = None
        if ok_g and rows_g:
            scored += 1
            x = bool(ok_p and _canon(rows_p) == _canon(rows_g))
            ex += int(x)
        valid += int(v)
        em += int(e)
        per.append({"valid": v, "em": e, "ex": x})
    n = len(items)
    metrics = {
        "n": n,
        "valid": valid / n if n else 0.0,
        "exact_match": em / n if n else 0.0,
        "exact_match_ci": wilson(em, n),
        "exec_acc": ex / scored if scored else 0.0,
        "exec_scored": scored,
        "exec_ci": wilson(ex, scored),
    }
    return metrics, per


# --------------------------------------------------------------------------
# 2. Data
# --------------------------------------------------------------------------
def load_in_dist(args):
    from datasets import load_dataset

    print(f"Loading {IN_DIST_DATASET} ...")
    ds = load_dataset(IN_DIST_DATASET, split="train")
    rows = [
        {
            "question": r["question"],
            "context": r["context"],
            "schema": schema_only(r["context"]),
            "gold": r["answer"].strip(),
        }
        for r in ds
    ]
    random.Random(SEED).shuffle(rows)
    test, rest = rows[: args.n_test], rows[args.n_test:]
    test_q = {r["question"] for r in test}
    train = [r for r in rest if r["question"] not in test_q]  # drop duplicate questions
    if args.n_train:
        train = train[: args.n_train]
    train_schemas = {r["schema"] for r in train}
    overlap = sum(r["schema"] in train_schemas for r in test) / max(1, len(test))
    print(
        f"train={len(train)}  in_dist test={len(test)}  "
        f"test schemas also seen in train: {overlap:.0%} (high = easy test, expected)"
    )
    return train, test


def load_external(args):
    from datasets import load_dataset

    print(f"Loading {EXTERNAL_DATASET} (external test) ...")
    try:
        ds = load_dataset(EXTERNAL_DATASET, split="test")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not load external set, skipping it ({e})")
        return []
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    items = []
    for i in idx:
        r = ds[i]
        it = {
            "question": r["sql_prompt"],
            "context": r["sql_context"],
            "schema": schema_only(r["sql_context"]),
            "gold": r["sql"].strip(),
        }
        conn = make_db(it["context"], it["gold"], seed=0)
        ok, rows = run_query(conn, it["gold"])
        conn.close()
        if ok and rows:  # keep only items whose gold query runs and returns rows
            items.append(it)
        if len(items) >= args.n_external:
            break
    with_rows = sum("insert into" in it["context"].lower() for it in items)
    print(f"external items usable: {len(items)} ({with_rows} ship with their own INSERT rows)")
    return items


def user_msg(it: dict) -> dict:
    return {"role": "user", "content": f"Schema:\n{it['schema']}\n\nQuestion: {it['question']}"}


def build_messages(it: dict, shots=()) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for s in shots:
        msgs += [user_msg(s), {"role": "assistant", "content": s["gold"]}]
    msgs.append(user_msg(it))
    return msgs


def render_prompt(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


# --------------------------------------------------------------------------
# 3. Model, generation, training
# --------------------------------------------------------------------------
def free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def load_model(args, path: str | None = None):
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=path or args.model,
        max_seq_length=args.max_len,
        load_in_4bit=True,
        dtype=None,
    )
    return model, tok


def generate(model, tok, messages_list, batch_size: int, max_new_tokens: int = 256) -> list[str]:
    import torch

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    t0 = time.time()
    for i in range(0, len(messages_list), batch_size):
        prompts = [render_prompt(tok, m) for m in messages_list[i : i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
        for row in gen:
            outs.append(tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"  generated {len(outs)}/{len(messages_list)}  ({time.time() - t0:.0f}s)")
    return outs


def run_system(name, model, tok, sets, shots, args) -> None:
    pdir = Path(args.out) / "preds"
    pdir.mkdir(parents=True, exist_ok=True)
    for sname, items in sets.items():
        f = pdir / f"{name}__{sname}.json"
        if f.exists() and not args.force:
            print(f"[{name}/{sname}] cached, skipping")
            continue
        print(f"[{name}/{sname}] generating {len(items)} ...")
        raw = generate(model, tok, [build_messages(it, shots) for it in items], args.gen_batch)
        f.write_text(json.dumps({"raw": raw, "pred": [clean_sql(r) for r in raw]}, indent=1))


def stage_baseline(args, sets, shots) -> None:
    pdir = Path(args.out) / "preds"
    todo = [
        n
        for n in ("base_zeroshot", "base_fewshot")
        if args.force or any(not (pdir / f"{n}__{s}.json").exists() for s in sets)
    ]
    if not todo:
        print("baselines already cached")
        return
    from unsloth import FastLanguageModel

    model, tok = load_model(args)
    FastLanguageModel.for_inference(model)
    if "base_zeroshot" in todo:
        run_system("base_zeroshot", model, tok, sets, (), args)
    if "base_fewshot" in todo:
        run_system("base_fewshot", model, tok, sets, shots, args)
    del model, tok
    free_gpu()


def stage_train(args, train_items):
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    out = Path(args.out)
    adapter_dir = out / "adapter"
    model, tok = load_model(args)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    # Training text = the exact same prompt used at inference + the SQL answer.
    # Matching train and inference formatting avoids the most common fine-tune bug.
    texts = [
        {"text": render_prompt(tok, build_messages(it)) + it["gold"] + END} for it in train_items
    ]
    ds = Dataset.from_list(texts)
    print("Example training text:\n" + "-" * 40 + "\n" + ds[0]["text"] + "\n" + "-" * 40)

    use_bf16 = torch.cuda.is_bf16_supported()
    cfg = dict(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="no",
        output_dir=str(out / "trainer"),
        report_to="none",
        seed=SEED,
        dataset_num_proc=2,
    )
    sft_cfg = None
    for len_key in ("max_seq_length", "max_length"):  # TRL renamed this argument
        try:
            sft_cfg = SFTConfig(**cfg, **{len_key: args.max_len})
            break
        except TypeError:
            continue
    if sft_cfg is None:
        raise RuntimeError("Could not build SFTConfig. Check your TRL version.")
    trainer = None
    for tok_key in ("processing_class", "tokenizer"):  # TRL renamed this too
        try:
            trainer = SFTTrainer(model=model, train_dataset=ds, args=sft_cfg, **{tok_key: tok})
            break
        except TypeError:
            continue
    if trainer is None:
        raise RuntimeError("Could not build SFTTrainer. Check your TRL version.")
    sample = ds[0]["text"]
    if INSTRUCTION_PART in sample and RESPONSE_PART in sample:
        trainer = train_on_responses_only(
            trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
        )
        print("Loss masked to the SQL answer only.")
    else:
        print("WARNING: chat markers not found, training on the full text (no loss masking).")

    t0 = time.time()
    trainer.train()
    print(f"Training took {(time.time() - t0) / 60:.1f} min")
    (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")
    return model, tok


def stage_export(args, model, tok) -> None:
    out = Path(args.out)
    token = os.environ.get("HF_TOKEN")
    if args.merge:
        model.save_pretrained_merged(str(out / "merged_16bit"), tok, save_method="merged_16bit")
        print("Merged 16-bit model saved.")
    if args.gguf:
        model.save_pretrained_gguf(str(out / "gguf"), tok, quantization_method="q4_k_m")
        print("GGUF saved (q4_k_m).")
    if args.push:
        if not token:
            print("Set the HF_TOKEN environment variable (write token) to push.")
        else:
            model.push_to_hub(args.push, token=token)
            tok.push_to_hub(args.push, token=token)
            print(f"Pushed adapter to https://huggingface.co/{args.push}")


# --------------------------------------------------------------------------
# 4. Report
# --------------------------------------------------------------------------
def stage_report(args, sets) -> None:
    out = Path(args.out)
    pdir = out / "preds"
    results, per_item, preds_by = {}, {}, {}
    for sname, items in sets.items():
        for sysname in SYSTEMS:
            f = pdir / f"{sysname}__{sname}.json"
            if not f.exists():
                continue
            preds = json.loads(f.read_text())["pred"]
            m, per = score(items, preds)
            results[(sname, sysname)] = m
            per_item[(sname, sysname)] = per
            preds_by[(sname, sysname)] = preds
    if not results:
        print("No predictions found. Run the baseline and eval stages first.")
        return

    def pct(x):
        return f"{100 * x:.1f}%"

    lines = [
        "| set | system | valid SQL | exact match | execution acc (95% CI) | scored |",
        "|---|---|---|---|---|---|",
    ]
    for (sname, sysname), m in results.items():
        lo, hi = m["exec_ci"]
        lines.append(
            f"| {sname} | {sysname} | {pct(m['valid'])} | {pct(m['exact_match'])} | "
            f"{pct(m['exec_acc'])} ({pct(lo)} to {pct(hi)}) | {m['exec_scored']}/{m['n']} |"
        )
    table = "\n".join(lines)

    extra = []
    for sname in sets:
        a, b = per_item.get((sname, "finetuned")), per_item.get((sname, "base_fewshot"))
        if a and b:
            wins = sum(1 for x, y in zip(a, b) if x["ex"] is True and y["ex"] is False)
            losses = sum(1 for x, y in zip(a, b) if x["ex"] is False and y["ex"] is True)
            extra.append(
                f"- {sname}: fine-tuned vs 3-shot base, execution-correct only on one side: "
                f"{wins} wins, {losses} losses"
            )

    notes = """
How to read this
- valid SQL: the query runs in SQLite without error.
- exact match: normalized string equality with the gold query. Punishes correct queries written differently.
- execution acc: predicted and gold queries return the same rows. Only items whose gold query returns
  at least one row are scored (the "scored" column). in_dist has no real data, so tables are filled with
  random rows seeded from the gold query's literals; that can occasionally make two different queries
  look equal. external uses each example's own INSERT rows where present (see below).
- If the confidence intervals overlap, do not claim one system beats the other.
- SQLite is not Postgres/MySQL: a few correct queries in other dialects will be marked wrong.
- Base-model output is parsed leniently (code fences and chatter stripped) so it is not punished for formatting.
"""
    ext_note = ""
    if "external" in sets:
        n_ext = max(1, len(sets["external"]))
        frac = sum("insert into" in it["context"].lower() for it in sets["external"]) / n_ext
        ext_note = f"- external: {frac:.0%} of items ship with their own INSERT rows; the rest use random filler rows.\n"
    md = "# Results\n\n" + table + "\n\n" + "\n".join(extra) + "\n" + notes + ext_note
    (out / "results.md").write_text(md)
    (out / "results.json").write_text(
        json.dumps({f"{s}/{n}": m for (s, n), m in results.items()}, indent=1, default=list)
    )
    print("\n" + table + "\n" + "\n".join(extra))

    # per-example predictions
    with open(out / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["set", "question", "gold"]
        for n in SYSTEMS:
            header += [f"{n}_pred", f"{n}_exec_correct"]
        w.writerow(header)
        for sname, items in sets.items():
            for i, it in enumerate(items):
                row = [sname, it["question"], it["gold"]]
                for n in SYSTEMS:
                    p = preds_by.get((sname, n))
                    q = per_item.get((sname, n))
                    row += [p[i] if p else "", q[i]["ex"] if q else ""]
                w.writerow(row)

    # chart
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        set_names = list(sets)
        width = 0.8 / len(SYSTEMS)
        fig, ax = plt.subplots(figsize=(7, 4))
        for j, n in enumerate(SYSTEMS):
            vals, errs = [], [[], []]
            for s in set_names:
                m = results.get((s, n))
                vals.append(100 * m["exec_acc"] if m else 0)
                lo, hi = m["exec_ci"] if m else (0, 0)
                errs[0].append(max(0, vals[-1] - 100 * lo))
                errs[1].append(max(0, 100 * hi - vals[-1]))
            xs = [k + j * width for k in range(len(set_names))]
            ax.bar(xs, vals, width, yerr=errs, capsize=3, label=n)
        ax.set_xticks([k + width for k in range(len(set_names))])
        ax.set_xticklabels(set_names)
        ax.set_ylabel("execution accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "comparison.png", dpi=150)
        print(f"Chart saved to {out / 'comparison.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(chart skipped: {e})")
    print(f"Report written to {out / 'results.md'}")


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=["all", "baseline", "train", "eval", "report", "export"])
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--out", default=f"runs/{MODEL_NAME.lower()}")
    p.add_argument("--n-train", type=int, default=10000, help="0 = use all rows (slow on a T4)")
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-external", type=int, default=300, help="0 = skip the external set")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max-steps", type=int, default=-1, help="overrides epochs when > 0")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--force", action="store_true", help="ignore cached predictions")
    p.add_argument("--smoke", action="store_true", help="tiny run to check everything works")
    p.add_argument("--merge", action="store_true", help="export merged 16-bit model")
    p.add_argument("--gguf", action="store_true", help="export GGUF (q4_k_m)")
    p.add_argument("--push", default="", help=f"HF repo id to push the adapter to, e.g. your-username/{MODEL_NAME} (needs HF_TOKEN env)")
    args, _ = p.parse_known_args()
    if args.smoke:
        args.n_train, args.n_test, args.n_external, args.max_steps = 500, 30, 30, 30
        args.out = args.out.rstrip("/") + "-smoke"
    return args


def main() -> None:
    ensure_installed()
    args = parse_args()
    print(f"=== {MODEL_NAME} | base model: {args.model} ===")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    if args.stage != "report":
        import unsloth  # noqa: F401  (must be imported before transformers/trl)
        import torch

        if not torch.cuda.is_available():
            sys.exit("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")
        print("GPU:", torch.cuda.get_device_name(0))

    train_items, in_test = load_in_dist(args)
    sets = {"in_dist": in_test}
    if args.n_external:
        ext = load_external(args)
        if ext:
            sets["external"] = ext
    shots = train_items[:3]

    stage = args.stage
    model = tok = None

    if stage in ("baseline", "all"):
        stage_baseline(args, sets, shots)

    if stage in ("train", "all"):
        model, tok = stage_train(args, train_items)

    if stage in ("eval", "all"):
        from unsloth import FastLanguageModel

        if model is None:
            model, tok = load_model(args, path=str(out / "adapter"))
        FastLanguageModel.for_inference(model)
        run_system("finetuned", model, tok, sets, (), args)

    if stage in ("report", "all"):
        stage_report(args, sets)

    if stage in ("export", "all") and (args.merge or args.gguf or args.push):
        if model is None:
            model, tok = load_model(args, path=str(out / "adapter"))
        stage_export(args, model, tok)

    print("\nDone. Outputs in:", out.resolve())


if __name__ == "__main__":
    main()