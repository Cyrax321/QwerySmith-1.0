#!/usr/bin/env python3
"""
QwerySmith 1.0 - Interactive Autonomous Database Agent
Connects your fine-tuned QwerySmith model to any SQLite database,
translates English questions into SQL, executes them, and returns results.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


class QwerySmithAgent:
    def __init__(self, model_path: str = "runs/qwerysmith-1.0/adapter"):
        """Initialize the agent with your fine-tuned LoRA adapter."""
        from unsloth import FastLanguageModel

        print(f"Loading QwerySmith model from {model_path}...")
        self.model, self.tok = FastLanguageModel.from_pretrained(model_path)
        FastLanguageModel.for_inference(self.model)
        print("Model loaded and ready for queries.")

    def get_schema(self, conn: sqlite3.Connection) -> str:
        """Extract table schemas from a SQLite database connection."""
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type IN ('table', 'view');")
        tables = [row[0] + ";" for row in cursor.fetchall() if row[0]]
        return "\n".join(tables)

    def generate_sql(self, schema: str, question: str) -> str:
        """Generate pure SQL from schema and natural language question."""
        prompt = (
            f"<|im_start|>system\n"
            f"You are a text-to-SQL assistant. Given a database schema and a question, "
            f"reply with exactly one SQL query and nothing else.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Schema:\n{schema}\n\nQuestion: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        inputs = self.tok([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id,
            use_cache=True,
        )
        raw_sql = self.tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return raw_sql.strip()

    def query(self, db_path: str | Path, question: str) -> dict:
        """End-to-end query: English question -> SQL -> Database execution -> Results."""
        conn = sqlite3.connect(str(db_path))
        schema = self.get_schema(conn)

        if not schema:
            conn.close()
            return {"error": "Database has no tables or views."}

        sql = self.generate_sql(schema, question)

        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(100)
            conn.close()
            return {
                "question": question,
                "sql": sql,
                "columns": columns,
                "rows": rows,
                "success": True,
            }
        except Exception as e:
            conn.close()
            return {
                "question": question,
                "sql": sql,
                "error": str(e),
                "success": False,
            }


# Example quick demo when run standalone
if __name__ == "__main__":
    # Create sample in-memory company database
    conn = sqlite3.connect("sample_company.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            product TEXT,
            category TEXT,
            revenue INT,
            units INT
        );
    """)
    conn.execute("DELETE FROM sales;")
    conn.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?);",
        [
            ("MacBook Pro", "Electronics", 45000, 20),
            ("iPhone 15", "Electronics", 80000, 100),
            ("Office Chair", "Furniture", 12000, 60),
            ("Desk Lamp", "Furniture", 3000, 150),
        ],
    )
    conn.commit()
    conn.close()

    print("Created 'sample_company.db' with sales data.")
    print("\nTo query this database with your model, run:")
    print("  agent = QwerySmithAgent('/path/to/adapter')")
    print("  result = agent.query('sample_company.db', 'Which category made the highest revenue?')")
    print("  print(result)")
