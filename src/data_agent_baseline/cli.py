from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.runner import (
    TaskRunArtifacts,
    create_run_output_dir,
    run_benchmark,
    run_single_task,
)
from data_agent_baseline.tools.filesystem import list_context_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()

PENALTY_LAMBDA = 0.2
NULL_STRINGS = {"", "null", "none", "nan", "nat", "<na>"}


def _normalize_cell(value: str) -> str:
    """Normalize a cell value per official eval rules."""
    stripped = value.strip().replace("\r\n", "\n").replace("\r", "")
    if stripped.lower() in NULL_STRINGS:
        return ""
    # Try numeric
    try:
        d = Decimal(stripped)
        if d.is_finite():
            return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):
        pass
    # Try date (YYYY-MM-DD or variants)
    import re
    date_match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", stripped)
    if date_match:
        y, m, d_val = date_match.groups()
        return f"{y}-{int(m):02d}-{int(d_val):02d}"
    # Try datetime with timezone → UTC
    from datetime import datetime, timezone
    try:
        if "T" in stripped:
            if stripped.endswith("Z"):
                dt = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            elif "+" in stripped[10:] or (stripped.count("-") > 2):
                dt = datetime.fromisoformat(stripped)
                if dt.tzinfo is not None:
                    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                return dt.isoformat()
    except (ValueError, TypeError):
        pass
    return stripped


def _column_signature(col_values: list[str]) -> tuple[str, ...]:
    """Build sorted column signature from normalized cell values."""
    return tuple(sorted(_normalize_cell(v) for v in col_values))


def _score_prediction(
    pred_cols: list[str], pred_rows: list[list[str]],
    gold_cols: list[str], gold_rows: list[list[str]],
) -> tuple[float, int, int, int]:
    """Score prediction against gold using official column-signature matching.

    Returns (score, matched, gold_count, pred_count).
    """
    if not gold_cols:
        return 0.0, 0, 0, len(pred_cols)
    if not pred_cols:
        return 0.0, 0, len(gold_cols), 0

    # Build column signatures for gold
    gold_sigs: list[tuple[str, ...]] = []
    for ci in range(len(gold_cols)):
        col_vals = [row[ci] if ci < len(row) else "" for row in gold_rows]
        gold_sigs.append(_column_signature(col_vals))

    # Build column signatures for prediction
    pred_sigs: list[tuple[str, ...]] = []
    for ci in range(len(pred_cols)):
        col_vals = [row[ci] if ci < len(row) else "" for row in pred_rows]
        pred_sigs.append(_column_signature(col_vals))

    # Count occurrences of each signature
    from collections import Counter
    gold_counter: Counter[tuple[str, ...]] = Counter(gold_sigs)
    pred_counter: Counter[tuple[str, ...]] = Counter(pred_sigs)

    # Match: for each signature, matched count = min(gold_count, pred_count)
    matched = 0
    for sig, g_count in gold_counter.items():
        p_count = pred_counter.get(sig, 0)
        matched += min(g_count, p_count)

    gold_count = len(gold_cols)
    pred_count = len(pred_cols)
    extra = pred_count - matched

    recall = matched / gold_count if gold_count > 0 else 0.0
    penalty = PENALTY_LAMBDA * (extra / pred_count) if pred_count > 0 else 0.0
    score = max(0.0, recall - penalty)

    return score, matched, gold_count, pred_count


def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
    score_avg: float = 0.0,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
        "score": f"score={score_avg:.3f}" if completed_count > 0 else "score=—",
    }


@app.callback()
def cli() -> None:
    """Utilities for working with the local DABench baseline project."""


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show the local project layout and public dataset presence."""
    app_config = load_app_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(app_config.dataset.root_path)

    table = Table(title="DABench Baseline Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row(
        "dataset_root",
        str(app_config.dataset.root_path),
        _status_value(app_config.dataset.root_path),
    )
    table.add_row("config_path", str(config_path), _status_value(config_path))

    console.print(table)

    if public_dataset.exists:
        console.print(f"Public tasks: {len(public_dataset.list_task_ids())}")
        counts = public_dataset.task_counts()
        if counts:
            rendered_counts = ", ".join(
                f"{difficulty}={count}" for difficulty, count in sorted(counts.items())
            )
            console.print(f"Public task counts: {rendered_counts}")


@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show task metadata and available context files."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the ReAct baseline on one task."""
    import csv

    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(
            app_config.run.output_dir, run_id=app_config.run.run_id
        )
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc

    start = perf_counter()
    artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)
    elapsed = perf_counter() - start

    console.print(f"\n[bold]Completed in {elapsed:.1f}s[/bold]")
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.failure_reason is not None:
        console.print(f"[red]Failure: {artifacts.failure_reason}[/red]")

    # Load prediction
    pred_cols: list[str] = []
    pred_rows: list[list[str]] = []
    if artifacts.prediction_csv_path and artifacts.prediction_csv_path.exists():
        with open(artifacts.prediction_csv_path, newline="") as f:
            reader = csv.reader(f)
            all_rows = list(reader)
        if all_rows:
            pred_cols = all_rows[0]
            pred_rows = [r for r in all_rows[1:] if r]

    # Load gold
    gold_path = PROJECT_ROOT / "data" / "public" / "output" / task_id / "gold.csv"
    gold_cols: list[str] = []
    gold_rows: list[list[str]] = []
    if gold_path.exists():
        with open(gold_path, newline="") as f:
            reader = csv.reader(f)
            all_rows = list(reader)
        if all_rows:
            gold_cols = all_rows[0]
            gold_rows = [r for r in all_rows[1:] if r]

    # Print prediction
    console.print("\n[bold cyan]── Prediction ──[/bold cyan]")
    if pred_cols:
        ptable = Table()
        for col in pred_cols:
            ptable.add_column(col)
        for row in pred_rows:
            ptable.add_row(*row)
        console.print(ptable)
    else:
        console.print("[dim](no prediction)[/dim]")

    # Print gold
    console.print("\n[bold green]── Gold Answer ──[/bold green]")
    if gold_cols:
        gtable = Table()
        for col in gold_cols:
            gtable.add_column(col)
        for row in gold_rows:
            gtable.add_row(*row)
        console.print(gtable)
    else:
        console.print("[dim](no gold answer available)[/dim]")

    # Score using official metric
    if gold_cols:
        score, matched, gold_count, pred_count = _score_prediction(
            pred_cols, pred_rows, gold_cols, gold_rows
        )
        extra = pred_count - matched
        if score == 1.0:
            console.print(f"\n[bold green]✓ PERFECT SCORE: {score:.4f}[/bold green]")
        elif score > 0.0:
            console.print(
                f"\n[bold yellow]~ PARTIAL SCORE: {score:.4f} "
                f"(matched={matched}/{gold_count}, extra={extra})[/bold yellow]"
            )
        elif not pred_cols:
            console.print("\n[bold red]✗ NO PREDICTION produced (score=0)[/bold red]")
        else:
            console.print(
                f"\n[bold red]✗ SCORE: {score:.4f} "
                f"(matched={matched}/{gold_count}, extra={extra})[/bold red]"
            )


@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
) -> None:
    """Run the ReAct baseline on multiple tasks from the config selection."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task_total = len(dataset.iter_tasks())
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = app_config.run.max_workers

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[bold magenta]{task.fields[score]}[/bold magenta]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        running_score_sum = 0.0
        scored_count = 0
        start_time = perf_counter()

        import csv as _csv_progress

        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            nonlocal running_score_sum, scored_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1

            # Score this task immediately
            pred_c: list[str] = []
            pred_r: list[list[str]] = []
            if artifact.prediction_csv_path and artifact.prediction_csv_path.exists():
                with open(artifact.prediction_csv_path, newline="") as f:
                    rows_all = list(_csv_progress.reader(f))
                if rows_all:
                    pred_c = rows_all[0]
                    pred_r = [r for r in rows_all[1:] if r]

            gold_p = PROJECT_ROOT / "data" / "public" / "output" / artifact.task_id / "gold.csv"
            if gold_p.exists():
                with open(gold_p, newline="") as f:
                    rows_all = list(_csv_progress.reader(f))
                if rows_all:
                    gold_c = rows_all[0]
                    gold_r = [r for r in rows_all[1:] if r]
                    s, _, _, _ = _score_prediction(pred_c, pred_r, gold_c, gold_r)
                    running_score_sum += s
                    scored_count += 1

            score_avg = running_score_sum / scored_count if scored_count > 0 else 0.0
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                    score_avg=score_avg,
                ),
            )

        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                progress_callback=on_task_complete,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
        final_score_avg = running_score_sum / scored_count if scored_count > 0 else 0.0
        progress.update(
            progress_task_id,
            completed=task_total,
            description="Benchmark",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
                score_avg=final_score_avg,
            ),
        )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")

    # Score all tasks against gold
    import csv as csv_mod
    scores: list[tuple[str, float]] = []
    for artifact in artifacts:
        task_id_str = artifact.task_id
        pred_c: list[str] = []
        pred_r: list[list[str]] = []
        if artifact.prediction_csv_path and artifact.prediction_csv_path.exists():
            with open(artifact.prediction_csv_path, newline="") as f:
                rows_all = list(csv_mod.reader(f))
            if rows_all:
                pred_c = rows_all[0]
                pred_r = [r for r in rows_all[1:] if r]

        gold_p = PROJECT_ROOT / "data" / "public" / "output" / task_id_str / "gold.csv"
        gold_c: list[str] = []
        gold_r: list[list[str]] = []
        if gold_p.exists():
            with open(gold_p, newline="") as f:
                rows_all = list(csv_mod.reader(f))
            if rows_all:
                gold_c = rows_all[0]
                gold_r = [r for r in rows_all[1:] if r]

        if gold_c:
            s, _, _, _ = _score_prediction(pred_c, pred_r, gold_c, gold_r)
            scores.append((task_id_str, s))

    if scores:
        total_score = sum(s for _, s in scores)
        avg_score = total_score / len(scores)
        perfect_count = sum(1 for _, s in scores if s == 1.0)
        partial_count = sum(1 for _, s in scores if 0.0 < s < 1.0)
        zero_count = sum(1 for _, s in scores if s == 0.0)

        console.print(f"\n[bold]━━━ Official Scoring ━━━[/bold]")

        # Per-task breakdown sorted by score first
        console.print(f"  [dim]Per-task scores:[/dim]")
        for tid, s in sorted(scores, key=lambda x: x[1]):
            if s == 1.0:
                console.print(f"    [green]✓ {tid}: {s:.4f}[/green]")
            elif s > 0.0:
                console.print(f"    [yellow]~ {tid}: {s:.4f}[/yellow]")
            else:
                console.print(f"    [red]✗ {tid}: {s:.4f}[/red]")

        # Summary stats last
        console.print(f"\n  Tasks scored: {len(scores)}")
        console.print(f"  [green]Perfect (1.0): {perfect_count}[/green]")
        console.print(f"  [yellow]Partial (>0): {partial_count}[/yellow]")
        console.print(f"  [red]Zero: {zero_count}[/red]")
        console.print(f"  [bold]Average Score: {avg_score:.4f}[/bold]")


@app.command()
def tui(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Launch the interactive task runner (simple CLI loop)."""
    import csv

    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    tasks = dataset.iter_tasks()

    if not tasks:
        console.print("[red]No tasks found.[/red]")
        raise typer.Exit(1)

    # Show task list
    console.print(f"\n[bold]{len(tasks)} tasks available[/bold]\n")
    for i, t in enumerate(tasks):
        q = t.question if len(t.question) <= 80 else t.question[:77] + "..."
        console.print(f"  {i+1:3d}. [{t.difficulty}] {t.task_id}: {q}")

    console.print("\n[dim]Enter task number, 'n' for next, 'p' for prev, 'q' to quit[/dim]\n")

    current_idx = 0
    results: list[tuple[str, str]] = []  # (task_id, verdict)

    while True:
        try:
            choice = input(f"[{current_idx+1}/{len(tasks)}] Run which task? ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "q":
            break
        elif choice == "n":
            current_idx = min(current_idx + 1, len(tasks) - 1)
        elif choice == "p":
            current_idx = max(current_idx - 1, 0)
        elif choice == "":
            pass  # just run current
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(tasks):
                current_idx = idx
            else:
                console.print(f"[red]Invalid number (1-{len(tasks)})[/red]")
                continue
        else:
            matched = [i for i, t in enumerate(tasks) if t.task_id == choice]
            if matched:
                current_idx = matched[0]
            else:
                console.print("[red]Unknown input. Enter number, n, p, or q.[/red]")
                continue

        task = tasks[current_idx]
        console.print(f"\n[bold cyan]━━━ Running {task.task_id} ({task.difficulty}) ━━━[/bold cyan]")
        console.print(f"Q: {task.question}\n")

        try:
            _, run_output_dir = create_run_output_dir(
                app_config.run.output_dir, run_id=app_config.run.run_id
            )
        except (ValueError, FileExistsError):
            run_output_dir = app_config.run.output_dir / "tui_run"
            run_output_dir.mkdir(parents=True, exist_ok=True)

        # Fresh model per task to avoid stale connections
        from data_agent_baseline.run.runner import build_model_adapter
        fresh_model = build_model_adapter(app_config)

        start = perf_counter()
        artifacts = run_single_task(
            task_id=task.task_id, config=app_config, run_output_dir=run_output_dir,
            model=fresh_model,
        )
        elapsed = perf_counter() - start

        console.print(f"\n[bold]Completed in {elapsed:.1f}s[/bold]")
        if artifacts.failure_reason:
            console.print(f"[red]Failure: {artifacts.failure_reason}[/red]")

        # Load prediction
        pred_cols: list[str] = []
        pred_rows: list[list[str]] = []
        if artifacts.prediction_csv_path and artifacts.prediction_csv_path.exists():
            with open(artifacts.prediction_csv_path, newline="") as f:
                reader = csv.reader(f)
                all_rows = list(reader)
            if all_rows:
                pred_cols = all_rows[0]
                pred_rows = [r for r in all_rows[1:] if r]

        # Load gold
        gold_path = PROJECT_ROOT / "data" / "public" / "output" / task.task_id / "gold.csv"
        gold_cols: list[str] = []
        gold_rows: list[list[str]] = []
        if gold_path.exists():
            with open(gold_path, newline="") as f:
                reader = csv.reader(f)
                all_rows = list(reader)
            if all_rows:
                gold_cols = all_rows[0]
                gold_rows = [r for r in all_rows[1:] if r]

        # Print comparison
        console.print("\n[bold cyan]── Prediction ──[/bold cyan]")
        if pred_cols:
            ptable = Table()
            for col in pred_cols:
                ptable.add_column(col)
            for row in pred_rows:
                ptable.add_row(*row)
            console.print(ptable)
        else:
            console.print("[dim](no prediction)[/dim]")

        console.print("\n[bold green]── Gold Answer ──[/bold green]")
        if gold_cols:
            gtable = Table()
            for col in gold_cols:
                gtable.add_column(col)
            for row in gold_rows:
                gtable.add_row(*row)
            console.print(gtable)
        else:
            console.print("[dim](no gold answer available)[/dim]")

        # Score using official metric
        verdict = "no_gold"
        task_score = 0.0
        if gold_cols:
            task_score, matched, gold_count, pred_count = _score_prediction(
                pred_cols, pred_rows, gold_cols, gold_rows
            )
            extra = pred_count - matched
            if task_score == 1.0:
                console.print(f"\n[bold green]✓ PERFECT SCORE: {task_score:.4f}[/bold green]")
                verdict = "perfect"
            elif task_score > 0.0:
                console.print(
                    f"\n[bold yellow]~ PARTIAL SCORE: {task_score:.4f} "
                    f"(matched={matched}/{gold_count}, extra={extra})[/bold yellow]"
                )
                verdict = "partial"
            elif not pred_cols:
                console.print("\n[bold red]✗ NO PREDICTION produced (score=0)[/bold red]")
                verdict = "no_pred"
            else:
                console.print(
                    f"\n[bold red]✗ SCORE: {task_score:.4f} "
                    f"(matched={matched}/{gold_count}, extra={extra})[/bold red]"
                )
                verdict = "wrong"

        results.append((task.task_id, verdict, task_score))
        console.print(f"\n[dim]n=next, p=prev, number=jump, q=quit[/dim]")

    # Summary report
    if results:
        console.print("\n[bold]━━━ Session Summary ━━━[/bold]")
        total = len(results)
        perfect = sum(1 for _, v, _ in results if v == "perfect")
        partial = sum(1 for _, v, _ in results if v == "partial")
        wrong = sum(1 for _, v, _ in results if v == "wrong")
        no_pred = sum(1 for _, v, _ in results if v == "no_pred")
        total_score = sum(s for _, _, s in results)
        avg_score = total_score / total if total > 0 else 0.0

        # Per-task breakdown first
        console.print("  [dim]Breakdown:[/dim]")
        for tid, v, s in results:
            icon = {"perfect": "✓", "partial": "~", "wrong": "✗", "no_pred": "✗", "no_gold": "?"}[v]
            color = {"perfect": "green", "partial": "yellow", "wrong": "red", "no_pred": "red", "no_gold": "dim"}[v]
            console.print(f"    [{color}]{icon} {tid} (score={s:.4f})[/{color}]")

        # Summary stats last
        console.print(f"\n  Tasks run: {total}")
        console.print(f"  [green]Perfect (1.0): {perfect}[/green]")
        console.print(f"  [yellow]Partial (>0): {partial}[/yellow]")
        console.print(f"  [red]Zero score: {wrong + no_pred}[/red]")
        console.print(f"  [bold]Average Score: {avg_score:.4f} (total={total_score:.4f}/{total})[/bold]")


def main() -> None:
    app()
