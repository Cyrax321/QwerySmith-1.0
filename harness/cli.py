#!/usr/bin/env python3
"""
harness/cli.py -- Command-line runner for QwerySmith harness
"""
import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwerysmith-harness", description="QwerySmith Text-to-SQL Evaluation Harness")
    parser.add_argument("--db", type=str, help="Path to SQLite database file")
    parser.add_argument("--eval-file", type=str, help="Path to evaluation questions JSON")
    parser.add_argument("--timeout", type=float, default=3.0, help="Per-query execution timeout in seconds")
    parser.add_argument("--read-only", action="store_true", default=True, help="Enforce strict read-only execution sandbox")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
