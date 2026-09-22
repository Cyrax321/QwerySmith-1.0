"""Dataset configuration — the ONLY place dataset specifics may live (plan §3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .exceptions import ConfigError

DATASETS_ROOT = Path("datasets")


@dataclass(frozen=True)
class HoldoutConfig:
    column: str          # "table.column" the cutoff applies to
    months: int          # window size, e.g. 6

    @property
    def table(self) -> str:
        return self.column.split(".", 1)[0]

    @property
    def col(self) -> str:
        return self.column.split(".", 1)[1]


@dataclass(frozen=True)
class DatasourceConfig:
    """Raw-files -> database mapping (plan §4.1)."""

    csv_dir: Path
    csv_pattern: str                 # e.g. "olist_*.csv" (sqlite) — paired with table_map
    table_map: dict[str, str]       # csv filename stem -> table name
    uri: str                         # SQLAlchemy URI to the prepared database
    dialect: str                     # "postgres" | "sqlite"
    extra: dict[str, Any] = field(default_factory=dict)  # columns / primary_keys / foreign_keys


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    license: str
    source_url: str
    question_file: Path
    holdout: Optional[HoldoutConfig]
    datasource: DatasourceConfig
    seed: int = 42
    retrieval_top_k: int = 8
    statement_timeout_sec: float = 30.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, dataset: str, root: Path | None = None) -> "DatasetConfig":
        base = (root or Path.cwd()) / DATASETS_ROOT / dataset
        cfg_path = base / "config.yaml"
        if not cfg_path.exists():
            raise ConfigError(f"no config found at {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text())

        # sqlite URIs are dataset-relative: resolve against the dataset dir so
        # `python -m qwery_smith ingest olist` works from any cwd.
        # Absolute paths must keep the 4-slash hostless form.
        uri = raw["datasource"]["uri"]
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            rel = uri.split("sqlite:///", 1)[1]
            resolved = base / rel
            uri = f"sqlite:///{resolved}" if not str(resolved).startswith("/") else f"sqlite:////{resolved}"

        try:
            ds = raw["datasource"]
            ho = raw.get("holdout")
            return cls(
                name=raw["name"],
                license=raw["license"],
                source_url=raw["source_url"],
                question_file=base / raw["question_file"],
                holdout=HoldoutConfig(**ho) if ho else None,
                datasource=DatasourceConfig(
                    csv_dir=base / ds["csv_dir"],
                    csv_pattern=ds["csv_pattern"],
                    table_map=dict(ds["table_map"]),
                    uri=uri,
                    dialect=ds["dialect"],
                    extra=ds.get("extra", {}),
                ),
                seed=raw.get("seed", 42),
                retrieval_top_k=raw.get("retrieval_top_k", 8),
                statement_timeout_sec=raw.get("statement_timeout_sec", 30.0),
                extra=raw.get("extra", {}),
            )
        except KeyError as e:
            raise ConfigError(f"{cfg_path}: missing key {e}") from e