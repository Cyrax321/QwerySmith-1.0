#!/usr/bin/env python3
"""
harness/__main__.py -- Package execution entrypoint (python -m harness)
"""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
