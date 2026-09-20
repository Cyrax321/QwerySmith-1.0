#!/usr/bin/env python3
"""
agent.py -- QwerySmith Interactive Autonomous Database Agent

Connects your fine-tuned QwerySmith (1.0 or 1.1) model to any SQLite database,
translates English queries into SQL in a live chat loop, executes them,
and renders tabular results with execution timings and automatic error healing.

Usage:
    python agent.py
    python agent.py --model /content/drive/MyDrive/qwerysmith-1.1/adapter --db custom.db
    
In Google Colab:
    import agent
    agent.chat_loop()
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from pathlib import Path


# --------------------------------------------------------------------------
# 1. Sample Enterprise Database Generator
# --------------------------------------------------------------------------
def init_sample_db(db_path: str | Path = "company_store.db") -> Path:
    """Creates a rich, multi-table e-commerce enterprise database for testing."""
    path = Path(db_path).resolve()
    if path.exists() and path.stat().st_size > 1000:
        return path

    print(f"📦 Initializing enterprise demo database: {path.name} ...")
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS reviews;
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            country TEXT NOT NULL,
            loyalty_tier TEXT CHECK(loyalty_tier IN ('Bronze', 'Silver', 'Gold', 'Platinum')),
            signup_date DATE
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            cost REAL NOT NULL,
            stock_quantity INTEGER NOT NULL
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date DATE NOT NULL,
            total_amount REAL NOT NULL,
            status TEXT CHECK(status IN ('Completed', 'Shipped', 'Pending', 'Cancelled')),
            payment_method TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );

        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        CREATE TABLE reviews (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            rating INTEGER CHECK(rating BETWEEN 1 AND 5),
            comment TEXT,
            review_date DATE,
            FOREIGN KEY (product_id) REFERENCES products(id),
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
    """)

    # Seed Customers
    customers = [
        (1, "Sophia Chen", "sophia@example.com", "US", "Platinum", "2023-01-15"),
        (2, "Marcus Vance", "marcus@example.com", "UK", "Gold", "2023-02-20"),
        (3, "Elena Rostova", "elena@example.com", "DE", "Silver", "2023-03-10"),
        (4, "Kenji Sato", "kenji@example.com", "JP", "Platinum", "2023-04-05"),
        (5, "Liam O'Connor", "liam@example.com", "IE", "Bronze", "2023-05-12"),
        (6, "Amara Diop", "amara@example.com", "FR", "Gold", "2023-06-18"),
        (7, "Carlos Mendoza", "carlos@example.com", "MX", "Bronze", "2023-07-22"),
        (8, "Aisha Khan", "aisha@example.com", "AE", "Platinum", "2023-08-30"),
    ]
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?);", customers)

    # Seed Products
    products = [
        (101, "MacBook Pro 16", "Electronics", 2499.00, 1800.00, 15),
        (102, "iPhone 15 Pro", "Electronics", 1199.00, 750.00, 45),
        (103, "Dell 32-inch 4K Monitor", "Electronics", 650.00, 420.00, 20),
        (104, "Ergonomic Mesh Chair", "Furniture", 420.00, 210.00, 8),
        (105, "Motorized Standing Desk", "Furniture", 750.00, 390.00, 12),
        (106, "Wireless Mechanical Keyboard", "Accessories", 160.00, 75.00, 60),
        (107, "Precision Bluetooth Mouse", "Accessories", 99.00, 40.00, 85),
        (108, "Noise-Cancelling Headphones", "Audio", 349.00, 160.00, 25),
        (109, "Studio Monitor Speakers", "Audio", 499.00, 260.00, 10),
        (110, "USB-C Multiport Hub", "Accessories", 79.00, 25.00, 110),
    ]
    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?);", products)

    # Seed Orders
    orders = [
        (1001, 1, "2024-01-10", 2578.00, "Completed", "Credit Card"),
        (1002, 2, "2024-01-15", 1359.00, "Completed", "PayPal"),
        (1003, 3, "2024-02-01", 420.00, "Completed", "Credit Card"),
        (1004, 4, "2024-02-14", 3249.00, "Completed", "Wire Transfer"),
        (1005, 1, "2024-02-28", 239.00, "Completed", "Credit Card"),
        (1006, 5, "2024-03-05", 99.00, "Completed", "Debit Card"),
        (1007, 6, "2024-03-12", 1170.00, "Completed", "PayPal"),
        (1008, 2, "2024-03-15", 499.00, "Shipped", "Credit Card"),
        (1009, 8, "2024-03-18", 2499.00, "Shipped", "Credit Card"),
        (1010, 7, "2024-03-20", 79.00, "Pending", "Debit Card"),
    ]
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?);", orders)

    # Seed Order Items
    order_items = [
        (1, 1001, 101, 1, 2499.00),
        (2, 1001, 110, 1, 79.00),
        (3, 1002, 102, 1, 1199.00),
        (4, 1002, 106, 1, 160.00),
        (5, 1003, 104, 1, 420.00),
        (6, 1004, 101, 1, 2499.00),
        (7, 1004, 105, 1, 750.00),
        (8, 1005, 106, 1, 160.00),
        (9, 1005, 110, 1, 79.00),
        (10, 1006, 107, 1, 99.00),
        (11, 1007, 105, 1, 750.00),
        (12, 1007, 104, 1, 420.00),
        (13, 1008, 109, 1, 499.00),
        (14, 1009, 101, 1, 2499.00),
        (15, 1010, 110, 1, 79.00),
    ]
    cur.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?);", order_items)

    # Seed Reviews
    reviews = [
        (1, 101, 1, 5, "Unbelievable processing power and display quality.", "2024-01-20"),
        (2, 102, 2, 4, "Great camera, battery life could be slightly better.", "2024-01-25"),
        (3, 104, 3, 5, "Completely cured my back pain during long coding sessions.", "2024-02-10"),
        (4, 105, 4, 5, "Whisper quiet motors, solid steel frame.", "2024-02-25"),
        (5, 108, 8, 5, "Best noise cancellation on long flights.", "2024-03-22"),
        (6, 107, 5, 3, "Decent ergonomics but clicks are a bit loud.", "2024-03-10"),
    ]
    cur.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?);", reviews)

    conn.commit()
    conn.close()
    print(f"✅ Demo database '{path.name}' created with 5 tables and populated data.\n")
    return path


# --------------------------------------------------------------------------
# 2. Text & Table Utilities
# --------------------------------------------------------------------------
def clean_sql(raw: str) -> str:
    """Strips thinking blocks, markdown fences, and explanatory prose."""
    s = raw or ""
    # Strip <think>...</think> tags if model produced reasoning
    s = re.sub(r"(?is)<think>.*?</think>", "", s)
    # Strip markdown ```sql ... ``` fences
    m = re.search(r"```(?:sql)?(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1)
    # Extract only until the first trailing semicolon or comment
    s = s.strip()
    if ";" in s:
        s = s[:s.find(";") + 1]
    return s.strip()


def format_table(columns: list[str], rows: list[tuple], max_rows: int = 50) -> str:
    """Renders a clean, auto-aligned ASCII table."""
    if not columns:
        return "  (Empty result set - 0 columns)"
    if not rows:
        header = " | ".join(columns)
        sep = "-" * len(header)
        return f"  {header}\n  {sep}\n  (0 rows returned)"

    display_rows = rows[:max_rows]
    str_rows = [[("" if v is None else str(v)) for v in row] for row in display_rows]
    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header = "| " + " | ".join(f"{c:<{widths[i]}}" for i, c in enumerate(columns)) + " |"
    lines = [sep, header, sep]
    for row in str_rows:
        line = "| " + " | ".join(f"{val:<{widths[i]}}" for i, val in enumerate(row)) + " |"
        lines.append(line)
    lines.append(sep)

    if len(rows) > max_rows:
        lines.append(f"  ... ({len(rows) - max_rows} additional rows omitted)")
    return "\n".join(lines)


GREETINGS = {
    "hi", "hey", "hello", "hola", "yo", "sup", "good morning", "good afternoon",
    "good evening", "who are you", "what are you", "what can you do", "help", ":help"
}


def is_conversational_greeting(text: str) -> bool:
    """Detects if user input is casual chit-chat or greeting rather than a database query."""
    clean = re.sub(r"[^\w\s]", "", text.strip().lower())
    if clean in GREETINGS:
        return True
    return clean.startswith(("hello ", "hey ", "hi there", "who are you", "what can you do"))


# --------------------------------------------------------------------------
# 3. Autonomous QwerySmith Agent
# --------------------------------------------------------------------------
class QwerySmithAgent:
    def __init__(self, model_path: str | None = None):
        """Initializes model, tokenizer, and fast inference pipeline."""
        self.model_path = self._resolve_model_path(model_path)
        self._load_model()

    def _resolve_model_path(self, requested: str | None) -> str:
        """Finds the most specific adapter or merged model path available."""
        if requested and (Path(requested).exists() or "/" in requested):
            return requested

        candidates = [
            "/content/drive/MyDrive/qwerysmith-1.1/adapter",
            "/content/drive/MyDrive/qwerysmith-1.1",
            "runs/qwerysmith-1.1/adapter",
            "/content/drive/MyDrive/qwerysmith-1.0/adapter",
            "runs/qwerysmith-1.0/adapter",
            "Cyrax321/QwerySmith-1.1",
            "Cyrax321/QwerySmith-1.1-Merged",
        ]
        for c in candidates:
            if Path(c).exists() and (Path(c) / "adapter_config.json").exists():
                return str(Path(c).resolve())
            if Path(c).exists() and (Path(c) / "config.json").exists():
                return str(Path(c).resolve())

        # Fallback to Hugging Face repository
        return "Cyrax321/QwerySmith-1.1"

    def _load_model(self):
        """Loads model into GPU VRAM using Unsloth if present, or Hugging Face PEFT."""
        print(f"📦 Loading QwerySmith from: {self.model_path}")
        t0 = time.time()
        try:
            from unsloth import FastLanguageModel
            self.model, self.tok = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            print(f"⚡ FastLanguageModel loaded in {time.time() - t0:.1f}s (4-bit optimized).")
        except Exception as e:
            print(f"  ⚠️ Unsloth fast loader fallback ({e}). Using standard Transformers...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            self.model.eval()
            print(f"⚡ Transformers model loaded in {time.time() - t0:.1f}s.")

    def get_schema(self, conn: sqlite3.Connection) -> str:
        """Extracts complete CREATE TABLE DDL definitions from active SQLite database."""
        cur = conn.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%';")
        tables = [row[0].strip() + ";" for row in cur.fetchall() if row[0]]
        return "\n".join(tables)

    def generate_sql(self, schema: str, question: str, error_feedback: str | None = None) -> str:
        """Translates natural language question and database schema into pure SQL."""
        system_msg = (
            "You are a text-to-SQL assistant. Given a database schema and a question, "
            "reply with exactly one SQL query and nothing else."
        )
        user_content = f"Schema:\n{schema}\n\nQuestion: {question}"
        if error_feedback:
            user_content += f"\n\nPrevious attempt failed with error:\n{error_feedback}\nPlease repair the query."

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        import torch
        import warnings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enc = self.tok([prompt], return_tensors="pt").to(device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
            warnings.filterwarnings("ignore", category=UserWarning)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                    use_cache=True,
                )

        raw = self.tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        return clean_sql(raw)

    def query(self, db_path: str | Path, question: str, auto_repair: bool = True) -> dict:
        """End-to-end execution: NL Question -> SQL -> DB Sandbox Execution -> Results."""
        conn = sqlite3.connect(str(db_path))
        schema = self.get_schema(conn)

        if not schema:
            conn.close()
            return {"error": "Database contains no tables.", "success": False}

        t_start = time.perf_counter()
        sql = self.generate_sql(schema, question)
        latency_gen = (time.perf_counter() - t_start) * 1000

        # Execution phase
        try:
            t_exec_start = time.perf_counter()
            cur = conn.cursor()
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchmany(100)
            latency_exec = (time.perf_counter() - t_exec_start) * 1000
            conn.close()
            return {
                "question": question,
                "sql": sql,
                "columns": columns,
                "rows": rows,
                "latency_gen_ms": latency_gen,
                "latency_exec_ms": latency_exec,
                "success": True,
            }
        except Exception as err:
            if auto_repair:
                # Attempt self-healing reflection
                repaired_sql = self.generate_sql(schema, question, error_feedback=str(err))
                try:
                    cur = conn.cursor()
                    cur.execute(repaired_sql)
                    columns = [desc[0] for desc in cur.description] if cur.description else []
                    rows = cur.fetchmany(100)
                    conn.close()
                    return {
                        "question": question,
                        "sql": repaired_sql,
                        "repaired_from": sql,
                        "columns": columns,
                        "rows": rows,
                        "latency_gen_ms": latency_gen,
                        "latency_exec_ms": 0.0,
                        "success": True,
                    }
                except Exception as err2:
                    conn.close()
                    return {
                        "question": question,
                        "sql": repaired_sql,
                        "error": str(err2),
                        "success": False,
                    }
            conn.close()
            return {"question": question, "sql": sql, "error": str(err), "success": False}

    # ----------------------------------------------------------------------
    # 4. Interactive Live Chat Loop
    # ----------------------------------------------------------------------
    def interactive_chat(self, db_path: str | Path = "company_store.db"):
        """Launches a full interactive command-line / Colab chat loop."""
        db_file = init_sample_db(db_path)
        conn = sqlite3.connect(str(db_file))
        schema = self.get_schema(conn)

        # Extract table names
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cur.fetchall()]
        conn.close()

        print("\n" + "=" * 72)
        print("💬 QWERYSMITH 1.1 INTERACTIVE DATABASE CHAT AGENT")
        print("=" * 72)
        print(f"📁 Connected Database : {db_file.name}")
        print(f"📊 Available Tables   : {', '.join(tables)}")
        print(f"🤖 Loaded Model       : {self.model_path}")
        print("💡 Special Commands   : :schema, :tables, :sample <table>, :db <path>, :exit")
        print("-" * 72)
        print("Try asking questions like:")
        print("  • Which customers spent more than $1,000 in total?")
        print("  • What is our top-selling product by revenue?")
        print("  • List all products with fewer than 15 items in stock.")
        print("  • What is the average customer order value per country?")
        print("=" * 72 + "\n")

        while True:
            try:
                user_input = input("💬 Ask a question (or ':exit'): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            # Command Handlers
            if user_input.lower() in [":exit", ":quit", "exit", "quit", ":q"]:
                print("👋 Session ended. Happy querying!")
                break

            if is_conversational_greeting(user_input):
                print(f"\n👋 Hello! I am QwerySmith 1.1, your autonomous Text-to-SQL database agent.")
                print(f"I am connected to '{db_file.name}' ({len(tables)} tables: {', '.join(tables)}).")
                print("Ask me questions in plain English to query your database, for example:")
                print("  • 'Which customers spent more than $1,000 in total?'")
                print("  • 'What is our top-selling product by revenue?'")
                print("  • 'List all products with stock quantity below 20.'")
                print("  • 'Show the average order value per country.'")
                print("  • 'Show all 5-star reviews along with customer name.'")
                print("\nCommands: :schema, :tables, :sample <table>, :db <path>, :exit\n")
                continue

            if user_input.lower() == ":schema":
                conn = sqlite3.connect(str(db_file))
                print("\n📋 DATABASE SCHEMA DDL:")
                print("-" * 50)
                print(self.get_schema(conn))
                print("-" * 50 + "\n")
                conn.close()
                continue

            if user_input.lower() == ":tables":
                conn = sqlite3.connect(str(db_file))
                cur = conn.cursor()
                print("\n📊 DATABASE SUMMARY:")
                for t in tables:
                    cur.execute(f"SELECT count(*) FROM {t};")
                    cnt = cur.fetchone()[0]
                    print(f"  • {t:<15} ({cnt} rows)")
                print()
                conn.close()
                continue

            if user_input.lower().startswith(":sample"):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: :sample <table_name>")
                    continue
                tname = parts[1]
                conn = sqlite3.connect(str(db_file))
                cur = conn.cursor()
                try:
                    cur.execute(f"SELECT * FROM {tname} LIMIT 3;")
                    cols = [d[0] for d in cur.description]
                    s_rows = cur.fetchall()
                    print(f"\n🔍 Sample from '{tname}':")
                    print(format_table(cols, s_rows))
                    print()
                except Exception as e:
                    print(f"Error reading table '{tname}': {e}")
                conn.close()
                continue

            if user_input.lower().startswith(":db"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: :db /path/to/database.db")
                    continue
                new_db = Path(parts[1]).resolve()
                if not new_db.exists():
                    print(f"❌ Error: Database file not found: {new_db}")
                    continue
                db_file = new_db
                conn = sqlite3.connect(str(db_file))
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                tables = [r[0] for r in cur.fetchall()]
                conn.close()
                print(f"✅ Switched active database to: {db_file.name} ({len(tables)} tables)")
                continue

            # Natural Language SQL Generation & Execution
            print("\n⚡ Synthesizing SQL query...")
            res = self.query(db_file, user_input)

            print(f"🧠 Generated SQL:")
            print(f"   \033[1;32m{res['sql']}\033[0m")

            if "repaired_from" in res:
                print(f"   \033[1;33m(Self-healed from previous syntax fault: {res['repaired_from']})\033[0m")

            if res["success"]:
                cols = res["columns"]
                rows = res["rows"]
                print(f"\n📊 Results ({len(rows)} rows, {res.get('latency_gen_ms', 0):.0f}ms gen):")
                print(format_table(cols, rows))
                print()
            else:
                print(f"\n❌ Execution Failed: {res.get('error', 'Unknown error')}\n")


# --------------------------------------------------------------------------
# 5. Top-Level Entry Points
# --------------------------------------------------------------------------
def chat_loop(model_path: str | None = None, db_path: str | Path = "company_store.db"):
    """One-click Python entry point for Colab, Jupyter, or terminal."""
    agent = QwerySmithAgent(model_path=model_path)
    agent.interactive_chat(db_path=db_path)


def main():
    parser = argparse.ArgumentParser(description="Live interactive chat agent for QwerySmith Text-to-SQL.")
    parser.add_argument("--model", default=None, help="Path to fine-tuned LoRA adapter or HuggingFace repo.")
    parser.add_argument("--db", default="company_store.db", help="Path to SQLite database.")
    args = parser.parse_args()

    chat_loop(model_path=args.model, db_path=args.db)


if __name__ == "__main__":
    main()
