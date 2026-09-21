#!/usr/bin/env python3
"""
harness/agent.py -- QwerySmith Autonomous Database & Conversational Agent

Connects fine-tuned QwerySmith models to SQLite databases, translates natural language
into SQL with sub-millisecond persistent memory, executes queries, provides natural
language synthesis, and self-heals syntax or schema errors via real-time reflection.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import HarnessConfig
from .conversation import DialogueStateTracker
from .decoding import ExecutionGuidedSelector
from .memory import AgentMemoryEngine, MemoryRetrievalResult
from .reflection import ErrorDiagnosis, MultiStepRepairTracker, SelfHealingEngine
from .schema import SchemaLinker, ValueGrounder
from .security import execute_sandboxed_query
from .telemetry import EventType, LatencyBreakdown, TelemetryDispatcher
from .tools import (
    classify_intent,
    clean_sql,
    execute_query,
    format_table,
    get_schema,
    get_table_counts,
    get_table_sample,
    get_tables,
    init_sample_db,
)


class QwerySmithAgent:
    """
    Autonomous Text-to-SQL Agent powered by the QwerySmith model family.

    Features:
    - Dual-Mode Inference: QLoRA SQL synthesis mode vs native base model chat reasoning.
    - Sub-Millisecond Persistent Memory: B-Tree short-term turn storage + FTS5 BM25 verified query cache.
    - Self-Healing Reflection Engine: Multi-step AST & error diagnosis with auto-repair retry.
    - Schema Linking & Value Grounding: Graph-expanded schema pruning and literal matching.
    - Sandboxed Execution: Strict read-only query execution with timeouts.
    - Real-Time Grounding: Dynamic system clock and schema catalog synchronization.
    """

    PAPER_URL = "https://drive.google.com/file/d/1sN1eVn7LpOi6cLEI1euxOT2cByBoXLlg/view?usp=sharing"
    CODEBASE_URL = "https://github.com/Cyrax321/QwerySmith-1.0"
    MODELS = {
        "1.1": {
            "lora": "https://huggingface.co/Cyrax321/QwerySmith-1.1/tree/main",
            "merged": "https://huggingface.co/Cyrax321/QwerySmith-1.1-Merged",
            "gguf": "https://huggingface.co/Cyrax321/QwerySmith-1.1-GGUF/tree/main",
            "hf_id": "Cyrax321/QwerySmith-1.1",
        },
        "1.0": {
            "lora": "https://huggingface.co/Cyrax321/QwerySmith-1.0",
            "merged": "https://huggingface.co/Cyrax321/QwerySmith-1.0-Merged",
            "gguf": "https://huggingface.co/Cyrax321/QwerySmith-1.0-GGUF",
            "hf_id": "Cyrax321/QwerySmith-1.0",
        },
    }

    def __init__(
        self,
        model_path: str | None = None,
        memory_path: str | Path | None = None,
        config: Optional[HarnessConfig] = None,
    ):
        """Initializes model, tokenizer, persistent memory, reflection engine, schema linker, and fast inference."""
        self.config = config or HarnessConfig()
        if memory_path:
            self.config.memory_path = memory_path

        self.model_path = self._resolve_model_path(model_path)
        self.memory = AgentMemoryEngine(storage_path=self.config.memory_path)
        self.healer = SelfHealingEngine()
        self.state_tracker = DialogueStateTracker(max_context_turns=self.config.max_dialogue_turns_context)
        self.selector = ExecutionGuidedSelector(timeout_sec=self.config.timeout_sec, max_rows=self.config.max_rows)
        self.telemetry = TelemetryDispatcher(verbose=self.config.verbose_telemetry)
        self._schema_linkers: Dict[str, SchemaLinker] = {}
        self._value_grounders: Dict[str, ValueGrounder] = {}
        self._load_model()

    def get_schema_linker(self, db_path: str | Path) -> SchemaLinker:
        """Cached schema linker for a given database."""
        key = str(Path(db_path).resolve())
        if key not in self._schema_linkers:
            self._schema_linkers[key] = SchemaLinker(db_path)
        return self._schema_linkers[key]

    def get_value_grounder(self, db_path: str | Path) -> ValueGrounder:
        """Cached value grounder for a given database."""
        key = str(Path(db_path).resolve())
        if key not in self._value_grounders:
            self._value_grounders[key] = ValueGrounder(
                db_path,
                max_distinct_per_col=self.config.max_distinct_values_per_col,
            )
        return self._value_grounders[key]

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
            "Cyrax321/QwerySmith-1.0",
            "Cyrax321/QwerySmith-1.0-Merged",
        ]
        for c in candidates:
            if Path(c).exists() and (Path(c) / "adapter_config.json").exists():
                return str(Path(c).resolve())
            if Path(c).exists() and (Path(c) / "config.json").exists():
                return str(Path(c).resolve())

        return "Cyrax321/QwerySmith-1.1"

    def _load_model(self):
        """Loads model into GPU VRAM using Unsloth if present, or Hugging Face PEFT."""
        print(f"[Loading] QwerySmith from: {self.model_path}")
        t0 = time.time()
        try:
            from unsloth import FastLanguageModel
            self.model, self.tok = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            print(f"[Model] FastLanguageModel loaded in {time.time() - t0:.1f}s (4-bit optimized).")
        except Exception as e:
            print(f"  [Warning] Unsloth fast loader fallback ({e}). Using standard Transformers...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            self.model.eval()
            print(f"[Model] Transformers model loaded in {time.time() - t0:.1f}s.")

    @contextmanager
    def disable_adapter_ctx(self):
        """Temporarily bypasses fine-tuned LoRA adapter to access base model's full conversational abilities."""
        if hasattr(self.model, "disable_adapter"):
            try:
                with self.model.disable_adapter():
                    yield
            except Exception:
                yield
        else:
            yield

    def generate_chat_text(self, messages: List[Dict[str, str]], max_tokens: int = 256) -> str:
        """Generates natural conversational text with LoRA adapter temporarily disabled."""
        import torch
        import warnings
        device = "cuda" if torch.cuda.is_available() else "cpu"

        prompt = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = self.tok([prompt], return_tensors="pt").to(device)

        with self.disable_adapter_ctx():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    gen = self.model.generate(
                        **enc,
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                        use_cache=True,
                    )

        raw = self.tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw

    def get_schema(self, conn: sqlite3.Connection) -> str:
        """Helper to extract schema via tools module."""
        return get_schema(conn)

    def generate_sql(
        self,
        schema: str,
        question: str,
        failed_sql: Optional[str] = None,
        error_feedback: Optional[str] = None,
        memory_context: Optional[str] = None,
        grounding_hints: Optional[str] = None,
        dialogue_context: Optional[str] = None,
        repair_prompt: Optional[str] = None,
    ) -> str:
        """Translates natural language question and database schema into pure SQL."""
        system_msg = (
            "You are an expert Text-to-SQL database engine. Given a database schema and a question, "
            "reply with exactly one executable SQLite query and nothing else.\n"
            "Critical Rules:\n"
            "1. Output ONLY the raw SQL query. Do not wrap in markdown quotes, comments, or prose.\n"
            "2. Always prefix column names with their table name or table alias (e.g., Track.TrackId or t.TrackId) "
            "whenever joining multiple tables to prevent ambiguous column errors.\n"
            "3. Ensure all table and column names strictly match the schema."
        )
        user_content = f"Schema:\n{schema}\n\nQuestion: {question}"

        if dialogue_context:
            user_content = f"{dialogue_context}\n\n" + user_content

        if grounding_hints:
            user_content += f"\n\n{grounding_hints}"

        if memory_context:
            user_content += f"\n\n{memory_context}"

        if repair_prompt:
            user_content += f"\n\n{repair_prompt}"
        elif error_feedback and failed_sql:
            diag = self.healer.diagnose_error(error_feedback, failed_sql)
            built_repair = self.healer.build_repair_prompt(question, failed_sql, error_feedback, diagnosis=diag)
            user_content += f"\n\n{built_repair}"
        elif error_feedback:
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
            warnings.simplefilter("ignore")
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

    def synthesize_human_response(self, question: str, sql: str, columns: list, rows: list) -> str:
        """Translates database query results into friendly, clear, natural human language."""
        if not rows:
            data_summary = "The query executed successfully, but returned 0 matching records."
        else:
            header = ", ".join(columns)
            sample_rows = "\n".join(str(r) for r in rows[:15])
            data_summary = f"Columns: {header}\nRows ({len(rows)} total):\n{sample_rows}"
            if len(rows) > 15:
                data_summary += f"\n... ({len(rows) - 15} additional rows)"

        now_str = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
        system_msg = (
            f"You are QwerySmith, an intelligent AI database assistant and business data analyst. "
            f"Current real-time timestamp: {now_str}. "
            "Given a user question, executed SQL query, and live database results, explain the findings directly and conversationally in natural human English. "
            "Be clear, concise, and helpful. Mention key numbers and takeaways."
        )
        user_msg = (
            f"User Question: {question}\n\n"
            f"Executed SQL: {sql}\n\n"
            f"Live Database Results:\n{data_summary}\n\n"
            "Please summarize this answer directly in friendly, professional natural human language:"
        )

        return self.generate_chat_text([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ], max_tokens=220)

    def chat_conversational(
        self,
        user_prompt: str,
        table_names: Optional[List[str]] = None,
        last_context: Optional[dict] = None,
    ) -> str:
        """Responds to general chit-chat, conceptual questions, coding help, and explanations."""
        now_str = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
        tables_str = ", ".join(table_names) if table_names else "none"
        system_msg = (
            f"You are QwerySmith, an advanced conversational AI assistant and expert SQL database engineer. "
            f"Real-time timestamp: {now_str}. "
            f"You are connected to a live database containing tables: [{tables_str}]. "
            "You can engage in natural conversation, explain database and SQL concepts, write code, "
            "or help the user analyze data and diagnose query errors. Reply naturally, warmly, and helpfully."
        )
        context_str = ""
        if last_context:
            if not last_context.get("success", True):
                context_str = (
                    f"\n\n[System Note: The user previously asked '{last_context.get('question')}'. "
                    f"The generated query was: '{last_context.get('sql')}'. "
                    f"It failed with SQLite error: '{last_context.get('error')}'. "
                    f"If the user asks why it failed, explain this exact error technically and describe how to correct it.]"
                )
            elif last_context.get("sql"):
                context_str = f"\n\n[System Note: The last query executed was: '{last_context.get('sql')}']"

        return self.generate_chat_text([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt + context_str},
        ], max_tokens=350)

    def query(
        self,
        db_path: str | Path,
        question: str,
        auto_repair: bool = True,
        session_id: str = "default_session",
    ) -> dict:
        """
        End-to-end multi-stage pipeline:
        1. Subgraph Schema Linking (Pruning irrelevant tables & finding FK paths)
        2. Column Value Grounding (Literal indexing & case-normalizing)
        3. Dialogue State Tracking (Anaphoric follow-up resolution & history)
        4. Sub-Millisecond Persistent Memory Recall (<1ms)
        5. Candidate Generation & Execution-Guided Selection
        6. Multi-Step AST Self-Healing Reflection
        7. Natural Language Data Synthesis & Telemetry
        """
        t_turn_start = time.perf_counter()
        db_name = Path(db_path).name
        latencies = LatencyBreakdown()

        self.telemetry.dispatch(EventType.TURN_START, session_id, db_name, question=question)

        # Step 1: Subgraph Schema Linking & Pruning
        t_link = time.perf_counter()
        if self.config.enable_schema_pruning:
            linker = self.get_schema_linker(db_path)
            pruned = linker.link(question, max_tables=self.config.schema_max_tables)
            schema = pruned.pruned_ddl
            available_cols = [c for t in pruned.selected_tables for c in linker.tables[t].columns]
            available_tbls = pruned.selected_tables
        else:
            schema = get_schema(db_path)
            available_cols = None
            available_tbls = None
        latencies.schema_linking_ms = (time.perf_counter() - t_link) * 1000

        if not schema:
            return {"error": "Database contains no tables.", "success": False}

        self.telemetry.dispatch(EventType.SCHEMA_LINKED, session_id, db_name, tables=available_tbls)

        # Step 2: Value Grounding
        t_val = time.perf_counter()
        grounding_hints = ""
        if self.config.enable_value_grounding:
            grounder = self.get_value_grounder(db_path)
            matches = grounder.ground(question)
            grounding_hints = grounder.format_grounding_hints(matches)
        latencies.value_grounding_ms = (time.perf_counter() - t_val) * 1000

        if grounding_hints:
            self.telemetry.dispatch(EventType.VALUE_GROUNDED, session_id, db_name, hints=grounding_hints)

        # Step 3: Dialogue State Tracking Context
        dst_context = ""
        if self.config.enable_dst:
            dst_context = self.state_tracker.build_context_prompt(session_id, question)

        # Step 4: Persistent Memory Recall (<1ms)
        t_mem = time.perf_counter()
        mem_res = self.memory.recall(session_id=session_id, db_name=db_name, question=question)
        latencies.memory_recall_ms = (time.perf_counter() - t_mem) * 1000

        # Step 5: SQL Synthesis
        t_gen = time.perf_counter()
        sql = self.generate_sql(
            schema=schema,
            question=question,
            memory_context=mem_res.prompt_context,
            grounding_hints=grounding_hints,
            dialogue_context=dst_context,
        )
        latencies.generation_ms = (time.perf_counter() - t_gen) * 1000

        self.telemetry.dispatch(EventType.SQL_GENERATED, session_id, db_name, sql=sql)

        # Step 6: Sandboxed Execution & Selection
        t_exec = time.perf_counter()
        sel_res = self.selector.select(
            conn_or_path=db_path,
            candidates=[sql],
            read_only=self.config.read_only,
        )
        latencies.execution_ms = (time.perf_counter() - t_exec) * 1000

        # Step 7: Multi-Step AST Self-Healing Reflection
        repaired_from = None
        was_repaired = False
        if not sel_res.success and auto_repair:
            t_rep = time.perf_counter()
            tracker = self.healer.start_repair_session(session_id, db_name, question, sql)
            curr_sql = sql
            curr_err = sel_res.error or "Unknown error"
            repaired_from = sql

            for attempt in range(1, self.config.max_repair_attempts + 1):
                print(f"  [Warning] Attempt {attempt} failed: {curr_err}")
                print(f"  [Reflect] Engaging self-healing reflection (Attempt {attempt}/{self.config.max_repair_attempts})...")

                diag = self.healer.diagnose_error(
                    curr_err,
                    curr_sql,
                    available_columns=available_cols,
                    available_tables=available_tbls,
                )
                self.healer.record_attempt(tracker, curr_sql, str(curr_err), diag)

                repair_prompt = self.healer.build_repair_prompt(
                    question,
                    curr_sql,
                    str(curr_err),
                    diagnosis=diag,
                    attempt=attempt,
                    max_attempts=self.config.max_repair_attempts,
                    history=tracker.attempts,
                )

                self.telemetry.dispatch(
                    EventType.REPAIR_ATTEMPTED,
                    session_id,
                    db_name,
                    attempt=attempt,
                    diagnosis=diag.error_type,
                    fix=diag.suggested_fix,
                )

                rep_sql = self.generate_sql(
                    schema=schema,
                    question=question,
                    failed_sql=curr_sql,
                    error_feedback=str(curr_err),
                    memory_context=mem_res.prompt_context,
                    grounding_hints=grounding_hints,
                    dialogue_context=dst_context,
                    repair_prompt=repair_prompt,
                )

                rep_sel = self.selector.select(
                    conn_or_path=db_path,
                    candidates=[rep_sql],
                    read_only=self.config.read_only,
                )

                if rep_sel.success:
                    self.healer.finalize_repair(tracker, rep_sql, success=True)
                    sql = rep_sql
                    sel_res = rep_sel
                    was_repaired = True
                    break
                else:
                    curr_sql = rep_sql
                    curr_err = rep_sel.error or "Unknown error"

            if not sel_res.success:
                self.healer.finalize_repair(tracker, curr_sql, success=False)
                sql = curr_sql

            latencies.repair_ms = (time.perf_counter() - t_rep) * 1000

        latencies.total_turn_ms = (time.perf_counter() - t_turn_start) * 1000

        # Step 8: Natural Language Synthesis & Commit
        if sel_res.success:
            human_ans = self.synthesize_human_response(question, sql, sel_res.columns, sel_res.rows)

            # Record in Dialogue State Tracker
            if self.config.enable_dst:
                self.state_tracker.record_turn(
                    session_id=session_id,
                    db_name=db_name,
                    question=question,
                    sql=sql,
                    columns=sel_res.columns,
                    rows=sel_res.rows,
                    human_summary=human_ans,
                )

            # Commit to Persistent Agent Memory
            self.memory.commit(
                session_id=session_id,
                db_name=db_name,
                question=question,
                sql=sql,
                columns=sel_res.columns,
                rows=sel_res.rows,
                human_summary=human_ans,
                success=True,
                repaired_from=repaired_from if was_repaired else None,
                latency_gen_ms=latencies.generation_ms,
                latency_exec_ms=sel_res.latency_exec_ms,
            )

            self.telemetry.dispatch(EventType.TURN_END, session_id, db_name, success=True, sql=sql)

            return {
                "question": question,
                "sql": sql,
                "repaired_from": repaired_from if was_repaired else None,
                "was_repaired": was_repaired,
                "columns": sel_res.columns,
                "rows": sel_res.rows,
                "human_answer": human_ans,
                "latency_gen_ms": latencies.generation_ms,
                "latency_exec_ms": sel_res.latency_exec_ms,
                "latency_breakdown": latencies.to_dict(),
                "memory_latency_ms": mem_res.retrieval_ms,
                "memory_recalled": bool(mem_res.prompt_context),
                "is_followup": mem_res.is_followup,
                "success": True,
            }
        else:
            if was_repaired:
                self.memory.commit(
                    session_id=session_id,
                    db_name=db_name,
                    question=question,
                    sql=sql,
                    success=False,
                    repaired_from=repaired_from,
                    error_msg=sel_res.error,
                )

            self.telemetry.dispatch(EventType.TURN_END, session_id, db_name, success=False, error=sel_res.error)

            return {
                "question": question,
                "sql": sql,
                "attempted_original": repaired_from,
                "error": sel_res.error,
                "latency_breakdown": latencies.to_dict(),
                "memory_latency_ms": mem_res.retrieval_ms,
                "success": False,
            }

    def interactive_chat(self, db_path: str | Path = "company_store.db"):
        """Launches an interactive live chat loop with memory, tools, and reflection."""
        db_file = init_sample_db(db_path)
        tables = get_tables(db_file)
        session_id = f"session_{int(time.time())}"

        print("\n" + "=" * 72)
        print("QWERYSMITH 1.1 INTERACTIVE DATABASE & CONVERSATIONAL AGENT")
        print("=" * 72)
        print(f"Connected Database : {db_file.name}")
        print(f"Available Tables   : {', '.join(tables)}")
        print(f"Loaded Model       : {self.model_path}")
        print("Special Commands   : :schema, :tables, :sample <table>, :memory, :clearmem, :db <path>, :exit")
        print("-" * 72)
        print("You can chat normally or ask live database queries:")
        print("  • 'Hey! How are you doing today?'")
        print("  • 'What is an inner join vs left join in SQL?'")
        print("  • 'Which customers spent more than $1,000 in total?'")
        print("  • 'What is our top-selling product by revenue?'")
        print("  • 'What is today's date and how many orders do we have?'")
        print("=" * 72 + "\n")

        last_interaction = None
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Command Handlers
            if user_input.lower() in [":exit", ":quit", "exit", "quit", ":q"]:
                print("Session ended. Happy querying!")
                break

            if user_input.lower() in [":memory", ":mem"]:
                st = self.memory.stats()
                print("\nPERSISTENT AGENTIC MEMORY STATUS:")
                print("-" * 50)
                print(f"  • Active Session ID       : {session_id}")
                print(f"  • Total Recorded Turns    : {st['total_turns']}")
                print(f"  • Unique Chat Sessions    : {st['total_sessions']}")
                print(f"  • Verified Past Queries   : {st['total_verified_queries']}")
                print(f"  • Self-Healed Experiences : {st['total_self_healed_patterns']}")
                print(f"  • Storage Database Path   : {st['storage_file']}")
                if st["queries_per_db"]:
                    print("  • Verified Queries by DB  :")
                    for db_k, cnt in st["queries_per_db"].items():
                        print(f"    - {db_k}: {cnt} queries")
                print("-" * 50 + "\n")
                continue

            if user_input.lower() in [":clearmem", ":clear_memory"]:
                cleared = self.memory.clear_session(session_id)
                print(f"Cleared {cleared} turns from active session memory.\n")
                continue

            if user_input.lower() == ":schema":
                print("\nDATABASE SCHEMA DDL:")
                print("-" * 50)
                print(get_schema(db_file))
                print("-" * 50 + "\n")
                continue

            if user_input.lower() == ":tables":
                counts = get_table_counts(db_file)
                print("\nDATABASE SUMMARY:")
                for t, cnt in counts.items():
                    print(f"  • {t:<15} ({cnt} rows)")
                print()
                continue

            if user_input.lower().startswith(":sample"):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: :sample <table_name>")
                    continue
                tname = parts[1]
                s_res = get_table_sample(db_file, tname)
                if s_res.get("error"):
                    print(f"Error reading table '{tname}': {s_res['error']}")
                else:
                    print(f"\nSample from '{tname}':")
                    print(format_table(s_res["columns"], s_res["rows"]))
                    print()
                continue

            if user_input.lower().startswith(":db"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: :db /path/to/database.db")
                    continue
                new_db = Path(parts[1]).resolve()
                if not new_db.exists():
                    print(f"[Error] Database file not found: {new_db}")
                    continue
                db_file = new_db
                tables = get_tables(db_file)
                print(f"[OK] Switched active database to: {db_file.name} ({len(tables)} tables)")
                continue

            # Classify Intent
            intent = classify_intent(user_input, tables)

            # Route 1: Real-time System Queries
            if intent == "REALTIME_SYS":
                now = datetime.now()
                print(f"\nReal-Time System Status:")
                print(f"  • Current Date & Time : {now.strftime('%A, %B %d, %Y - %I:%M:%S %p')}")
                print(f"  • Connected Database  : {db_file.name} ({len(tables)} tables active)\n")
                continue

            # Route 2: Database Metadata
            if intent == "DB_META":
                counts = get_table_counts(db_file)
                print(f"\nLive Database Overview ({db_file.name}):")
                for t, cnt in counts.items():
                    print(f"  • Table '{t}': {cnt} live records")
                print()
                continue

            # Route 3: General Chit-Chat / Concepts / Reasoning
            if intent == "CONVERSATIONAL":
                print("\nFormulating response...")
                reply = self.chat_conversational(user_input, tables, last_context=last_interaction)
                print(f"\nQwerySmith:\n  {reply}\n")
                continue

            # Route 4: Real-time Database Query & Natural Language Synthesis
            mem_preview = self.memory.recall(session_id, db_file.name, user_input)
            if mem_preview.is_followup and mem_preview.previous_turn:
                print(f"\n\033[1;36m[Memory] Follow-up detected. Injected prior turn context ({mem_preview.retrieval_ms:.2f}ms)\033[0m")
            elif mem_preview.exemplars:
                print(f"\n\033[1;36m[Memory] Recalled {len(mem_preview.exemplars)} verified schema exemplar(s) ({mem_preview.retrieval_ms:.2f}ms)\033[0m")

            print("\nSynthesizing SQL query...")
            res = self.query(db_file, user_input, session_id=session_id)
            last_interaction = res

            if res["success"]:
                print(f"\nQwerySmith:")
                print(f"  {res.get('human_answer', '')}\n")

                cols = res["columns"]
                rows = res["rows"]
                print(f"Live Data ({len(rows)} rows, {res.get('latency_gen_ms', 0):.0f}ms gen, {res.get('memory_latency_ms', 0):.2f}ms mem):")
                print(format_table(cols, rows))

                print(f"\nGenerated SQL:")
                print(f"   \033[1;32m{res['sql']}\033[0m")
                if "repaired_from" in res:
                    print(f"   \033[1;33m(Self-healed from: {res['repaired_from']})\033[0m")
                print()
            else:
                print(f"\n[Failed] Execution Failed: {res.get('error', 'Unknown error')}")
                print(f"   Attempted SQL: {res.get('sql', 'N/A')}")
                if "attempted_original" in res:
                    print(f"   Initial SQL:   {res.get('attempted_original')}")
                print()


def chat_loop(
    model_path: str | None = None,
    db_path: str | Path = "company_store.db",
    memory_path: str | Path | None = None,
):
    """One-click Python entry point for Colab, Jupyter, or terminal."""
    agent = QwerySmithAgent(model_path=model_path, memory_path=memory_path)
    agent.interactive_chat(db_path=db_path)


def main():
    parser = argparse.ArgumentParser(description="Live interactive chat agent for QwerySmith Text-to-SQL.")
    parser.add_argument("--model", default=None, help="Path to fine-tuned LoRA adapter or HuggingFace repo.")
    parser.add_argument("--db", default="company_store.db", help="Path to SQLite database.")
    parser.add_argument("--memory", default=None, help="Path to SQLite persistent agentic memory file.")
    args = parser.parse_args()

    chat_loop(model_path=args.model, db_path=args.db, memory_path=args.memory)


if __name__ == "__main__":
    main()
