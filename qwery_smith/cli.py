"""CLI (plan §8.2): ingest → validate → retrieve → train → eval → report → all."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import DATASETS_ROOT, DatasetConfig
from .exceptions import HarnessError, StageNotImplementedError

app = typer.Typer(add_completion=False, help="QwerySmith v3.0 harness")


def _load(dataset: str, root: Optional[Path] = None) -> DatasetConfig:
    try:
        return DatasetConfig.load(dataset, root=root)
    except HarnessError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(2)


def _uri_for_stage(cfg: DatasetConfig, stage: str) -> str:
    """eval/validate run read-only; ingest needs write. SQLite: same file.
    Postgres: ingest URI may include a rw role — kept simple in v3.0.0."""
    return cfg.datasource.uri


@app.command()
def version() -> None:
    """Print harness version."""
    typer.echo(__version__)


@app.command()
def ingest(
    dataset: str = typer.Argument(..., help="dataset name under datasets/"),
    root: Optional[Path] = typer.Option(None, help="repo root (default cwd)"),
) -> None:
    """Stage 1: load raw CSVs into the database (plan §4.1)."""
    cfg = _load(dataset, root)
    from .adapters import open_adapter
    from .validate import ingest_dataset

    adapter = open_adapter(cfg.datasource.uri, read_only=False)
    try:
        counts = ingest_dataset(cfg, adapter)
        total = sum(counts.values())
        typer.secho(f"ingested {total} rows across {len(counts)} tables:", fg=typer.colors.GREEN)
        for t, n in sorted(counts.items()):
            typer.echo(f"  {t:45s} {n:>9,d}")
    finally:
        adapter.close()


@app.command()
def profile(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
) -> None:
    """Stage 1b: profile DB — row counts, date ranges, holdout cutoff (plan §3.1/§4.2)."""
    cfg = _load(dataset, root)
    from .adapters import open_adapter
    from .profiler import profile_database

    adapter = open_adapter(cfg.datasource.uri, read_only=True)
    try:
        prof = profile_database(adapter, holdout=cfg.holdout)
        out = (root or Path.cwd()) / DATASETS_ROOT / cfg.name / "prepared" / "profile.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        import yaml

        out.write_text(yaml.safe_dump(prof.to_yaml_dict(), sort_keys=True), encoding="utf-8")
        typer.secho(f"profile written: {out}", fg=typer.colors.GREEN)
        if prof.holdout:
            h = prof.holdout
            typer.echo(f"  holdout column: {h['column']}")
            typer.echo(f"  data max:      {h['data_max']}")
            typer.echo(f"  CUTOFF:        {h['cutoff']}  (frozen — record in manifest)")
    finally:
        adapter.close()


@app.command()
def validate(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
) -> None:
    """Stage 2: validate question set — scorable denominator + leak checks (plan §3.2)."""
    cfg = _load(dataset, root)
    from .adapters import open_adapter
    from .questions import QuestionSet
    from .validate import validate_question_set

    qs = QuestionSet.load(cfg.question_file)
    adapter = open_adapter(cfg.datasource.uri, read_only=True)
    try:
        res = validate_question_set(qs, adapter, holdout=cfg.holdout)
        typer.echo(res.summary())
        if res.ok:
            typer.secho(f"OK — {res.scorable} scorable questions", fg=typer.colors.GREEN)
        else:
            typer.secho(f"FAILED — {len(res.excluded)} blocking issues", fg=typer.colors.RED)
            raise typer.Exit(1)
    finally:
        adapter.close()


@app.command()
def retrieve(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
    row_caps: Optional[str] = typer.Option(
        None, help="per-table row caps for the index, e.g. 'geolocation:1000' (comma-separated)"
    ),
) -> None:
    """Stage 3: freeze evidence packs per question from question text only (plan §4.3)."""
    cfg = _load(dataset, root)
    from .adapters import open_adapter
    from .questions import QuestionSet
    from .retrieval import build_index, build_pack, freeze_pack
    from .schema_loader import load_schema

    caps = {}
    if row_caps:
        for part in row_caps.split(","):
            t, n = part.split(":")
            caps[t.strip()] = int(n)

    qs = QuestionSet.load(cfg.question_file)
    adapter = open_adapter(cfg.datasource.uri, read_only=True)
    try:
        schema = load_schema(adapter)
        typer.echo(f"building row-text index ({sum(t.row_count for t in schema.tables.values()):,} rows)...")
        index = build_index(adapter, schema, row_caps=caps or None)

        packs_dir = (root or Path.cwd()) / DATASETS_ROOT / cfg.name / "prepared" / "packs"
        registry = {}
        for q in qs.questions:
            pack = build_pack(q.id, q.question, index, schema, cfg.retrieval_top_k)
            registry[q.id] = freeze_pack(pack, packs_dir)

        reg_path = packs_dir / "_registry.json"
        import json

        reg_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        typer.secho(f"froze {len(registry)} packs -> {packs_dir}", fg=typer.colors.GREEN)
        if cfg.holdout:
            # summary of pack row-table distribution
            from collections import Counter

            tables = Counter()
            for qid in registry:
                from .retrieval import load_pack

                p = load_pack(packs_dir, qid)
                for r in p.rows:
                    tables[r["table"]] += 1
            typer.echo("  pack composition: " + ", ".join(f"{t}={n}" for t, n in sorted(tables.items())))
    finally:
        adapter.close()


@app.command()
def triples(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
    seed: int = typer.Option(0, help="distractor sampling seed (per-run; eval seeds are separate)"),
) -> None:
    """Stage 3b: build RAFT triples from train_ok questions only (plan §3.3)."""
    cfg = _load(dataset, root)
    import random

    from .adapters import open_adapter
    from .questions import QuestionSet
    from .retrieval import build_index, load_pack
    from .schema_loader import load_schema
    from .triples import build_triples, write_triples

    qs = QuestionSet.load(cfg.question_file)
    adapter = open_adapter(cfg.datasource.uri, read_only=True)
    try:
        schema = load_schema(adapter)
        index = build_index(adapter, schema)
        packs_dir = (root or Path.cwd()) / DATASETS_ROOT / cfg.name / "prepared" / "packs"
        eval_packs = {q.id: load_pack(packs_dir, q.id) for q in qs.questions if q.split == "train_ok"}
        rng = random.Random(cfg.seed + seed)

        built = build_triples(cfg, qs.questions, index, schema.ddl_text(), eval_packs, rng=rng)
        out = (root or Path.cwd()) / DATASETS_ROOT / cfg.name / "prepared" / f"triples_seed{cfg.seed + seed}.jsonl"
        meta = write_triples(built, out)
        typer.secho(f"built {meta['n']} triples -> {meta['triples']}", fg=typer.colors.GREEN)
    finally:
        adapter.close()


@app.command()
def train(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
    seeds: str = typer.Option("1", help="comma-separated training seeds, e.g. '1,2,3'"),
    triples_file: Optional[Path] = typer.Option(None, help="override triples path"),
    output_dir: Optional[Path] = typer.Option(None, help="adapter output dir (default runs/)"),
    epochs: int = typer.Option(3),
    lr: float = typer.Option(1e-4),
    dry_run: bool = typer.Option(False, help="validate inputs only; no GPU work"),
) -> None:
    """Stage 4: train QLoRA adapter on RAFT triples (plan §3.3).

    The GPU work (Unsloth/QLoRA) targets a T4 on Colab — this CLI emits the
    training command with fully pinned config and checks readiness.
    """
    cfg = _load(dataset, root)
    from .training import prepare_training_run, qlora_config_yaml

    base = root or Path.cwd()
    tpath = triples_file or (
        base / DATASETS_ROOT / cfg.name / "prepared" / f"triples_seed{cfg.seed}.jsonl"
    )
    if not tpath.exists():
        typer.secho(f"triples not found: {tpath} (run `triples` first)", fg=typer.colors.RED)
        raise typer.Exit(2)

    run = prepare_training_run(
        cfg=cfg,
        triples_path=tpath,
        seeds=[int(s) for s in seeds.split(",")],
        output_dir=output_dir or (base / "runs" / cfg.name),
        epochs=epochs,
        lr=lr,
    )
    typer.secho(f"training run prepared: {run['run_dir']}", fg=typer.colors.GREEN)
    typer.echo(f"  triples:   {run['n_triples']} instances")
    typer.echo(f"  seeds:     {run['seeds']}")
    typer.echo(f"  config:    {run['config_path']}")
    typer.echo("\nOn the T4 (Colab), run:")
    typer.echo(f"  uv run python -m qwery_smith.training --config {run['config_path']}")
    if dry_run:
        typer.secho("dry-run: stopping before GPU work", fg=typer.colors.YELLOW)
        return
    typer.echo("(GPU training executes on Colab; this machine has no CUDA)")


@app.command()
def eval(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
    systems_yaml: Optional[Path] = typer.Option(
        None, help="systems config (default datasets/<name>/systems.yaml)"
    ),
    split: str = typer.Option("heldout", help="which split to score"),
    out: Optional[Path] = typer.Option(None, help="run output dir (default runs/<name>/<ts>)"),
) -> None:
    """Stage 5: run eval matrix — identical questions + identical packs (plan §3.4)."""
    cfg = _load(dataset, root)
    import json
    from datetime import datetime

    import yaml

    from .adapters import open_adapter
    from .evaluate import run_system, write_run
    from .questions import QuestionSet
    from .retrieval import load_pack
    from .schema_loader import load_schema
    from .systems import SystemConfig, build_system_fn

    base = root or Path.cwd()
    sys_path = systems_yaml or (base / DATASETS_ROOT / cfg.name / "systems.yaml")
    if not sys_path.exists():
        typer.secho(f"systems config not found: {sys_path}", fg=typer.colors.RED)
        raise typer.Exit(2)
    specs = [SystemConfig.from_yaml(d) for d in yaml.safe_load(sys_path.read_text())["systems"]]

    qs = QuestionSet.load(cfg.question_file)
    questions = [q for q in qs.questions if q.split == split]
    if not questions:
        typer.secho(f"no '{split}' questions found", fg=typer.colors.RED)
        raise typer.Exit(2)

    adapter = open_adapter(cfg.datasource.uri, read_only=True)
    try:
        schema = load_schema(adapter)
        packs_dir = base / DATASETS_ROOT / cfg.name / "prepared" / "packs"
        packs = {q.id: load_pack(packs_dir, q.id) for q in questions}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out or (base / "runs" / cfg.name / ts)
        run_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"{len(questions)} '{split}' questions x {len(specs)} systems -> {run_dir}")
        all_results = {}
        for spec in specs:
            typer.secho(f"  running {spec.name} ({spec.role})...", bold=True)
            fn = build_system_fn(spec)
            run = run_system(
                spec, fn, questions, packs, schema.ddl_text(), adapter,
                timeout_sec=cfg.statement_timeout_sec,
            )
            summary = write_run(run, run_dir, questions)
            typer.echo(
                f"    EX={summary['ex_heldout']:.1%} flip={summary['flip_rate']:.3f} "
                f"refusal={summary['refusal_rate']:.1%} failures={summary['n_failures']}"
            )
            all_results[spec.name] = run.headline

        (run_dir / "headline_results.json").write_text(
            json.dumps({k: [r.to_json() for r in v] for k, v in all_results.items()}, indent=2),
            encoding="utf-8",
        )
        typer.secho(f"eval complete -> {run_dir}", fg=typer.colors.GREEN)
        typer.echo(f"next: python -m qwery_smith report {cfg.name} --run {run_dir}")
    finally:
        adapter.close()


@app.command()
def report(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
    run: Optional[Path] = typer.Option(None, help="run dir from `eval` (default latest runs/<name>/*)"),
    split: str = typer.Option("heldout"),
) -> None:
    """Stage 6: results table + gate verdict + failure folders (plan §3.5)."""
    cfg = _load(dataset, root)
    import json

    from .questions import QuestionSet
    from .report import evaluate_gate, render_report, write_failure_folders
    from .scoring import ScoredResult

    base = root or Path.cwd()
    run_dir = run
    if run_dir is None:
        candidates = sorted((base / "runs" / cfg.name).glob("*"))
        if not candidates:
            typer.secho("no runs found; run `eval` first", fg=typer.colors.RED)
            raise typer.Exit(2)
        run_dir = candidates[-1]

    summaries = []
    headline = {}
    for p in sorted(run_dir.glob("*__summary.json")):
        s = json.loads(p.read_text())
        summaries.append(s)
    hr_path = run_dir / "headline_results.json"
    if hr_path.exists():
        raw = json.loads(hr_path.read_text())
        for name, results in raw.items():
            headline[name] = [ScoredResult(**r) for r in results]

    if not summaries:
        typer.secho(f"no summaries in {run_dir}", fg=typer.colors.RED)
        raise typer.Exit(2)

    by_role = {s["role"]: s for s in summaries}
    gate = None
    if "candidate" in by_role and "reference" in by_role:
        gate = evaluate_gate(
            ex_row2=by_role["candidate"]["ex_heldout"],
            flip_row2=by_role["candidate"]["flip_rate"],
            ex_row4=by_role["reference"]["ex_heldout"],
            flip_row4=by_role["reference"]["flip_rate"],
        )

    qs = QuestionSet.load(cfg.question_file)
    md = render_report(cfg.name, summaries, headline, gate=gate)
    out_md = run_dir / "report.md"
    out_md.write_text(md, encoding="utf-8")
    typer.echo(md)

    if headline:
        counts = write_failure_folders(
            run_dir, {q.id: q for q in qs.questions},
            headline,
        )
        typer.secho(f"failure folders: {counts}", fg=typer.colors.GREEN)
    typer.secho(f"report written: {out_md}", fg=typer.colors.GREEN)


@app.command()
def all(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
    stages: str = typer.Option("ingest,profile,validate,retrieve,train,eval,report"),
) -> None:
    """One command rebuilds everything from raw data (plan §8.2)."""
    for stage in [s.strip() for s in stages.split(",") if s.strip()]:
        typer.secho(f"=== {stage} ===", bold=True)
        # dispatch through the same command functions
        {"ingest": ingest, "profile": profile, "validate": validate,
         "retrieve": retrieve, "triples": triples, "train": train,
         "eval": eval, "report": report}[stage](dataset, root=root)


if __name__ == "__main__":
    app()