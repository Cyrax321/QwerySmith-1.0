"""Adapter factory."""

from __future__ import annotations

from pathlib import Path

from .base import DatabaseAdapter


def open_adapter(uri: str, read_only: bool = True, **kwargs) -> DatabaseAdapter:
    if uri.startswith("sqlite"):
        # sqlite:////abs/path or sqlite:///rel/path (relative to cwd)
        raw = uri.split(":///", 1)[-1] if uri.startswith("sqlite:///") else uri
        path = Path("/" + raw) if uri.startswith("sqlite:////") else Path(raw)
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        from .sqlite_adapter import SQLiteAdapter

        return SQLiteAdapter(path, read_only=read_only)
    if uri.startswith("postgres"):
        from .postgres_adapter import PostgresAdapter

        return PostgresAdapter(uri, read_only=read_only, **kwargs)
    raise ValueError(f"unsupported uri scheme: {uri}")