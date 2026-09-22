"""Adapter factory."""

from __future__ import annotations

from pathlib import Path

from .base import DatabaseAdapter


def open_adapter(uri: str, read_only: bool = True, **kwargs) -> DatabaseAdapter:
    if uri.startswith("sqlite"):
        # forms: sqlite:////abs/path (hostless absolute) or sqlite:///rel/path
        if uri.startswith("sqlite:////"):
            path = Path("/" + uri[len("sqlite:////"):].lstrip("/"))
        else:
            path = Path(uri.split("sqlite:///", 1)[-1])
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        from .sqlite_adapter import SQLiteAdapter

        return SQLiteAdapter(path, read_only=read_only)
    if uri.startswith("postgres"):
        from .postgres_adapter import PostgresAdapter

        return PostgresAdapter(uri, read_only=read_only, **kwargs)
    raise ValueError(f"unsupported uri scheme: {uri}")