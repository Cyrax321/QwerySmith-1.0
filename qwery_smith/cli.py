"""CLI (plan §8.2): ingest → validate → retrieve → train → eval → report → all."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import DATASETS_ROOT, DatasetConfig
from .exceptions import HarnessError

app = typer.Typer(add_completion=False, help="QwerySmith v3.0 harness")


def _load_env() -> None:
    """Auto-load a gitignored .env next to the repo root (HF_TOKEN etc.).
    No dependency: plain KEY=VALUE lines; comments ignored."""
    import os

    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:  # real environment wins over .env
            os.environ[k] = v


_load_env()


def _load(dataset: str, root: Optional[Path] = None) -> DatasetConfig:
    try:
        return DatasetConfig.load(dataset, root=root)
    except HarnessError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(2) from e


def _uri_for_stage(cfg: DatasetConfig, stage: str) -> str:
    """eval/validate run read-only; ingest needs write. SQLite: same file.
    Postgres: ingest URI may include a rw role - kept simple in v3.0.0."""
    return cfg.datasource.uri


@app.command()
def version() -> None:
    """Print harness version."""
    typer.echo(__version__)


@app.command()
def author(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
    n: int = typer.Option(20, help="number of candidates to generate"),
    seed: int = typer.Option(1, help="authoring seed"),
    out: Optional[Path] = typer.Option(None, help="draft output (default prepared/questions_draft.jsonl)"),
    force: bool = typer.Option(False, help="overwrite an existing draft"),
) -> None:
    """Stage 0: generate draft questions with validated gold + mechanical splits (plan §5.1).

    Splits are assigned by the clamped-shadow rule: a question is 'heldout'
    iff its gold result differs when the holdout window is removed. The author
    reviews/edits the file; `validate` re-checks everything afterwards.
    """
    cfg = _load(dataset, root)
    from datetime import date as _date

    from .adapters import open_adapter
    from .authoring import generate_candidates
    from .clamp import build_clamped, window_dependent
    from .profiler import compute_holdout_cutoff
    from .questions import ExpectedRows, Question, QuestionSet, canonical_rows_hash
    from .schema_loader import load_schema

    base = root or Path.cwd()
    out_path = out or (
        base / DATASETS_ROOT / cfg.name / "prepared" / "questions_draft.jsonl"
    )
    if out_path.exists() and not force:
        typer.secho(f"draft exists ({out_path}); use --force to overwrite", fg=typer.colors.YELLOW)
        raise typer.Exit(2)

    adapter = open_adapter(cfg.datasource.uri, read_only=True)
    try:
        schema = load_schema(adapter)
        candidates = generate_candidates(adapter, schema, cfg, n=n, seed=seed)
        if not candidates:
            typer.secho("no candidates generated - check schema/facts", fg=typer.colors.RED)
            raise typer.Exit(2)

        cutoff_iso = None
        clamped = None
        if cfg.holdout:
            _max, cutoff_iso = compute_holdout_cutoff(adapter, cfg.holdout)
            clamped = build_clamped(adapter, schema, cfg.holdout, cutoff_iso)

        qs = QuestionSet()
        n_heldout = 0
        for i, cand in enumerate(candidates):
            rows, cols, _ = adapter.safe_execute(cand.gold_sql)
            sha, n_rows = canonical_rows_hash(rows, cols)
            split = "train_ok"
            if clamped is not None:
                if window_dependent(cand.gold_sql, sha, clamped):
                    split = "heldout"
                    n_heldout += 1
            qs.add(Question(
                id=f"{cfg.name}-{i + 1:04d}",
                question=cand.question,
                gold_sql=cand.gold_sql,
                expected_rows=ExpectedRows(sha256=sha, n_rows=n_rows),
                date=_date.today().isoformat(),
                difficulty=cand.difficulty,
                category=cand.category,
                split=split,
                source="template+human-verified",  # downgraded to 'human' only after review
            ))
        qs.save(out_path)

        typer.secho(f"drafted {len(candidates)} questions -> {out_path}", fg=typer.colors.GREEN)
        if cfg.holdout:
            typer.echo(f"  holdout cutoff: {cutoff_iso}")
            typer.echo(f"  window-dependent (=> heldout): {n_heldout}")
            typer.echo(f"  window-independent (=> train_ok): {len(candidates) - n_heldout}")
        typer.echo(
            "\nNEXT: human review - rewrite questions for natural phrasing, verify gold"
            "\nSQL by eye, then set source='human' per question and move the file to"
            "\nquestions_v1.jsonl. `validate` re-checks everything."
        )
    finally:
        adapter.close()


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
    """Stage 1b: profile DB - row counts, date ranges, holdout cutoff (plan §3.1/§4.2)."""
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
            typer.echo(f"  CUTOFF:        {h['cutoff']}  (frozen - record in manifest)")
    finally:
        adapter.close()


@app.command()
def validate(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
) -> None:
    """Stage 2: validate question set - scorable denominator + leak checks (plan §3.2)."""
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
            typer.secho(f"OK - {res.scorable} scorable questions", fg=typer.colors.GREEN)
            # write the versioned manifest (plan §3.2): seed, counts, licence,
            # db_fingerprint, frozen holdout cutoff - the question-set's identity card
            from .profiler import compute_holdout_cutoff
            from .questions import write_manifest
            from .schema_loader import load_schema

            schema = load_schema(adapter)
            holdout_block = None
            if cfg.holdout is not None:
                _mx, cutoff = compute_holdout_cutoff(adapter, cfg.holdout)
                holdout_block = {"column": cfg.holdout.column,
                                 "months": cfg.holdout.months, "cutoff": cutoff}
            manifest_path = cfg.question_file.parent / "prepared" / "manifest.yaml"
            write_manifest(
                manifest_path,
                name=cfg.name,
                version="1.0.0",
                license_=cfg.license,
                source_url=cfg.source_url,
                seed=cfg.seed,
                questions_path=cfg.question_file,
                db_fingerprint=schema.fingerprint(),
                holdout=holdout_block,
                counts=qs.counts(),
            )
            typer.secho(f"manifest written: {manifest_path}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"FAILED - {len(res.excluded)} blocking issues", fg=typer.colors.RED)
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
    per_question: int = typer.Option(
        8, help="instances sampled per question (mix + distractors re-rolled per draw)"
    ),
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

        built = build_triples(cfg, qs.questions, index, schema.ddl_text(), eval_packs,
                              rng=rng, per_question=per_question)
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
    execute: bool = typer.Option(
        False,
        help="actually train (requires CUDA + unsloth; run this on the T4/Colab)",
    ),
) -> None:
    """Stage 4: train QLoRA adapter on RAFT triples (plan §3.3).

    Local (CPU): default mode prepares pinned configs + manifest, emits the
    Colab command. On the T4: pass --execute to run all seeds in sequence.
    """
    cfg = _load(dataset, root)
    from .training import prepare_training_run, train_from_config

    base = root or Path.cwd()
    tpath = triples_file or (
        base / DATASETS_ROOT / cfg.name / "prepared" / f"triples_seed{cfg.seed}.jsonl"
    )
    if not tpath.exists():
        typer.secho(f"triples not found: {tpath} (run `triples` first)", fg=typer.colors.RED)
        raise typer.Exit(2)

    seed_list = [int(s) for s in seeds.split(",")]
    run = prepare_training_run(
        cfg=cfg,
        triples_path=tpath,
        seeds=seed_list,
        output_dir=output_dir or (base / "runs" / cfg.name),
        epochs=epochs,
        lr=lr,
    )
    typer.secho(f"training run prepared: {run['run_dir']}", fg=typer.colors.GREEN)
    typer.echo(f"  triples:   {run['n_triples']} instances")
    typer.echo(f"  seeds:     {run['seeds']}")

    if dry_run:
        typer.secho("dry-run: stopping before GPU work", fg=typer.colors.YELLOW)
        return

    if not execute:
        typer.echo("\nOn the T4 (Colab), run either:")
        typer.echo(f"  python -m qwery_smith train {cfg.name} --seeds {seeds} --execute")
        typer.echo(f"  python -m qwery_smith.training --config <cfg> --triples {tpath} --out {run['run_dir']}/adapters")
        typer.echo("(this machine has no CUDA - preparation only)")
        return

    # --execute: the GPU path (Colab)
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False
    if not cuda_ok:
        typer.secho("--execute requires CUDA (run on the Colab T4)", fg=typer.colors.RED)
        raise typer.Exit(2)
    adapters_dir = Path(run["run_dir"]) / "adapters"
    for seed in seed_list:
        config_path = Path(run["run_dir"]) / f"qlora_seed{seed}.yaml"
        typer.secho(f"=== training seed {seed} ===", bold=True)
        adapter_path = train_from_config(config_path, tpath, adapters_dir)
        typer.secho(f"  adapter saved: {adapter_path}", fg=typer.colors.GREEN)
    typer.secho(f"all {len(seed_list)} seeds trained -> {adapters_dir}", fg=typer.colors.GREEN)


@app.command()
def eval(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
    systems_yaml: Optional[Path] = typer.Option(
        None, help="systems config (default datasets/<name>/systems.yaml)"
    ),
    split: str = typer.Option("heldout", help="which split to score"),
    out: Optional[Path] = typer.Option(None, help="run output dir (default runs/<name>/<ts>)"),
    roles: Optional[str] = typer.Option(
        None, help="comma-separated roles to run this pass, e.g. 'baseline,candidate_seed' "
        "(omit = all). Enables the 3-seed protocol: one pass per served adapter."
    ),
    tag: Optional[str] = typer.Option(None, help="label appended to the run dir name"),
) -> None:
    """Stage 5: run eval matrix - identical questions + identical packs (plan §3.4)."""
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
    if roles:
        wanted = {r.strip() for r in roles.split(",")}
        specs = [s for s in specs if s.role in wanted]
        if not specs:
            typer.secho(f"no systems with roles: {sorted(wanted)}", fg=typer.colors.RED)
            raise typer.Exit(2)

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
        run_dir = out or (base / "runs" / cfg.name / (ts + (f"_{tag}" if tag else "")))
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
    run: Optional[Path] = typer.Option(
        None,
        help="run dir from `eval`, or a parent dir containing multiple seed runs "
        "(3-seed protocol merges every child with *__summary.json)",
    ),
    split: str = typer.Option("heldout"),
) -> None:
    """Stage 6: results table + gate verdict + failure folders (plan §3.5/§7.1)."""
    cfg = _load(dataset, root)
    import json

    from .questions import QuestionSet
    from .report import aggregate_seeds, evaluate_gate, render_report, write_failure_folders
    from .scoring import ScoredResult

    base = root or Path.cwd()
    run_dir = run
    if run_dir is None:
        candidates = sorted((base / "runs" / cfg.name).glob("*"))
        if not candidates:
            typer.secho("no runs found; run `eval` first", fg=typer.colors.RED)
            raise typer.Exit(2)
        run_dir = candidates[-1]

    # collect summaries + per-question headlines: from one run dir, or from
    # every child run dir when the parent holds a multi-seed series
    summary_paths = sorted(run_dir.glob("*__summary.json"))
    headline_paths = [run_dir / "headline_results.json"]
    if not summary_paths:
        for child in sorted(run_dir.glob("*/*__summary.json")):
            summary_paths.append(child)
            headline_paths.append(child.parent / "headline_results.json")

    summaries = [json.loads(p.read_text()) for p in summary_paths]
    if not summaries:
        typer.secho(f"no summaries in {run_dir}", fg=typer.colors.RED)
        raise typer.Exit(2)

    headline: dict[str, list[ScoredResult]] = {}
    for hp in headline_paths:
        if not hp.exists():
            continue
        raw = json.loads(hp.read_text())
        for name, results in raw.items():
            headline[name] = [ScoredResult(**r) for r in results]

    # gate per §7.1: candidate EX = seed-mean when role='candidate_seed' rows exist
    candidate, rep_seed = aggregate_seeds(summaries, headline)
    gate = None
    ref = next((s for s in summaries if s.get("role") == "reference"), None)
    if candidate is not None and ref is not None:
        gate = evaluate_gate(
            ex_row2=candidate["ex_heldout"],
            flip_row2=candidate["flip_rate"],
            ex_row4=ref["ex_heldout"],
            flip_row4=ref["flip_rate"],
        )

    qs = QuestionSet.load(cfg.question_file)
    md = render_report(cfg.name, summaries, headline, gate=gate)
    out_md = run_dir / "report.md" if summary_paths and summary_paths[0].parent == run_dir else run_dir / "report.md"
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
def publish(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
    hf_user: str = typer.Option(..., help="HF username/org (e.g. Cyrax321)"),
    model_prefix: str = typer.Option("QwerySmith-2.0"),
    train_run: Optional[Path] = typer.Option(None, help="train run dir (default latest runs/<name>/train_*)"),
    report_path: Optional[Path] = typer.Option(None, help="report.md to embed in the card"),
    canonical_seed: Optional[int] = typer.Option(
        None, help="median-EX seed to also publish as the canonical repo (after eval)"
    ),
    gate_pass: Optional[bool] = typer.Option(None, help="gate verdict to state on the card"),
    dry_run: bool = typer.Option(False, help="stage locally, skip the upload"),
) -> None:
    """Stage 7: publish adapters to Hugging Face with full model cards + figures.

    Per seed: <prefix>-seed<N> repo (adapter, card, loss figure, records).
    With --canonical-seed: also <prefix> (the canonical release).
    Set HF_TOKEN in the environment; --dry-run stages without uploading.
    """
    cfg = _load(dataset, root)
    import json
    import os

    base = root or Path.cwd()
    train_dir = train_run
    if train_dir is None:
        candidates = sorted((base / "runs" / cfg.name).glob("train_*"))
        if not candidates:
            typer.secho("no train run found; run `train` first", fg=typer.colors.RED)
            raise typer.Exit(2)
        train_dir = candidates[-1]
    adapters_dir = train_dir / "adapters"
    if not adapters_dir.exists() or not list(adapters_dir.glob("adapter_seed*")):
        typer.secho(f"no adapters in {adapters_dir}", fg=typer.colors.RED)
        raise typer.Exit(2)

    # question counts for the card
    counts: dict = {}
    if cfg.question_file.exists():
        from .questions import QuestionSet

        counts = QuestionSet.load(cfg.question_file).counts()

    # cutoff from the frozen profile
    cutoff = None
    profile = base / DATASETS_ROOT / cfg.name / "prepared" / "profile.yaml"
    if profile.exists():
        import yaml

        prof = yaml.safe_load(profile.read_text())
        cutoff = (prof.get("holdout") or {}).get("cutoff")

    # report to embed
    report_md = report_path
    if report_md is None:
        latest = sorted((base / "runs" / cfg.name).glob("*/report.md"))
        report_md = latest[-1] if latest else None

    from .publish import build_model_card

    if dry_run:
        # stage cards locally for inspection, no upload
        out = base / "runs" / cfg.name / train_dir.name / "hf_cards_preview"
        out.mkdir(parents=True, exist_ok=True)
        import re as _re
        import yaml as _yaml

        for adapter in sorted(adapters_dir.glob("adapter_seed*")):
            m = _re.search(r"seed(\d+)", adapter.name)
            record_p = adapter / "train_record.json"
            if not m or not record_p.exists():
                continue
            seed = int(m.group(1))
            record = json.loads(record_p.read_text())
            qlora_cfg = _yaml.safe_load((train_dir / f"qlora_seed{seed}.yaml").read_text())
            card = build_model_card(
                model_name=f"{model_prefix}-seed{seed}",
                base_model=qlora_cfg["base_model"],
                dataset=cfg.name,
                dataset_url=cfg.source_url,
                dataset_license=cfg.license,
                question_counts=counts,
                cutoff=cutoff,
                qlora_config=qlora_cfg,
                train_record=record,
                report_md=(Path(report_md).read_text() if report_md and Path(report_md).exists() else None),
                gate_pass=gate_pass,
            )
            (out / f"card_seed{seed}.md").write_text(card, encoding="utf-8")
        typer.secho(f"dry-run: cards staged at {out} (no upload)", fg=typer.colors.GREEN)
        return

    if not os.environ.get("HF_TOKEN"):
        typer.secho("HF_TOKEN not set - export it (huggingface.co/settings/tokens, write access)", fg=typer.colors.RED)
        raise typer.Exit(2)

    from .publish import publish_all

    results = publish_all(
        adapters_dir=adapters_dir,
        train_run_dir=train_dir,
        hf_user=hf_user,
        model_prefix=model_prefix,
        dataset_cfg=cfg,
        question_counts=counts,
        cutoff=cutoff,
        report_md_path=Path(report_md) if report_md else None,
        gate_pass=gate_pass,
        canonical_seed=canonical_seed,
    )
    for r in results:
        typer.secho(f"published: https://huggingface.co/{r['repo_id']}", fg=typer.colors.GREEN)


@app.command()
def all(
    dataset: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None),
    stages: str = typer.Option(
        "ingest,profile,validate,retrieve,triples,train,eval,report",
        help="comma-separated stages to run (publish excluded: needs HF_TOKEN)",
    ),
) -> None:
    """One command rebuilds everything from raw data (plan §8.2)."""
    for stage in [s.strip() for s in stages.split(",") if s.strip()]:
        typer.secho(f"=== {stage} ===", bold=True)
        {"ingest": ingest, "profile": profile, "validate": validate,
         "retrieve": retrieve, "triples": triples, "train": train,
         "eval": eval, "report": report}[stage](dataset, root=root)


if __name__ == "__main__":
    app()