from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, RichLog, Static

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig, load_app_config
from data_agent_baseline.run.runner import (
    _run_single_task_core,
    _write_task_outputs,
    build_fast_model_adapter,
    build_model_adapter,
    create_run_output_dir,
)
from data_agent_baseline.tools.registry import create_default_tool_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_OUTPUT_DIR = PROJECT_ROOT / "data" / "public" / "output"


def _load_gold_csv(task_id: str) -> tuple[list[str], list[list[str]]] | None:
    gold_path = GOLD_OUTPUT_DIR / task_id / "gold.csv"
    if not gold_path.exists():
        return None
    with gold_path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return None
    return rows[0], [row for row in rows[1:] if row]


class TaskListScreen(Screen):
    BINDINGS = [
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("q", "quit_app", "Quit"),
    ]

    def action_cursor_down(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.action_cursor_down()

    def action_cursor_up(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.action_cursor_up()

    def __init__(self, dataset: DABenchPublicDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.tasks = dataset.iter_tasks()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f" {len(self.tasks)} tasks available — press Enter to run", id="task-hint")
        yield DataTable(id="task-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Task ID", "Difficulty", "Question")
        for task in self.tasks:
            q = task.question if len(task.question) <= 100 else task.question[:97] + "..."
            table.add_row(task.task_id, task.difficulty, q, key=task.task_id)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        task_id = str(event.row_key.value)
        self.app.push_screen(RunScreen(task_id))

    def action_quit_app(self) -> None:
        self.app.exit()


class RunScreen(Screen):
    BINDINGS = [
        Binding("escape", "go_back", "Back to list"),
        Binding("c", "cancel_run", "Cancel"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = task_id
        self._cancelled = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f" Running: {self.task_id}", id="run-label")
        yield RichLog(id="agent-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._run_agent()

    @work(thread=True)
    def _run_agent(self) -> None:
        app: DABenchTUI = self.app  # type: ignore[assignment]
        log_widget = self.query_one("#agent-log", RichLog)
        run_label = self.query_one("#run-label", Label)

        start = perf_counter()
        last_label_update = 0.0

        def write_log(msg: str) -> None:
            self.app.call_from_thread(log_widget.write, msg)

        def update_label(elapsed: float) -> None:
            run_label.update(f" Running: {self.task_id}  |  Elapsed: {elapsed:.0f}s")

        def log_cb(step: dict | str) -> None:
            nonlocal last_label_update
            elapsed = perf_counter() - start
            ts = f"[dim]{elapsed:6.1f}s[/dim]"
            if isinstance(step, dict):
                action = step.get("action", "")
                detail = step.get("detail", "")
                self.app.call_from_thread(log_widget.write, f"{ts} [bold cyan]\\[{action}][/bold cyan] {detail}")
            else:
                self.app.call_from_thread(log_widget.write, f"{ts} {step}")
            if elapsed - last_label_update > 5.0:
                last_label_update = elapsed
                self.app.call_from_thread(update_label, elapsed)

        write_log(f"[bold]Task: {self.task_id}[/bold]")
        write_log(f"Question: {app.dataset.get_task(self.task_id).question}")
        write_log("")
        # Fresh model per task to avoid connection reuse rate-limiting
        model = build_model_adapter(app.config)
        fast_model = build_fast_model_adapter(app.config)
        try:
            run_result = _run_single_task_core(
                task_id=self.task_id,
                config=app.config,
                model=model,
                tools=app.tools,
                fast_model=fast_model,
                log_callback=log_cb,
            )
        except Exception as exc:
            write_log(f"[bold red]Agent error: {exc}[/bold red]")
            return

        elapsed = perf_counter() - start
        write_log("")
        write_log(f"[bold]Completed in {elapsed:.1f}s[/bold]")
        self.app.call_from_thread(
            run_label.update,
            f" Done: {self.task_id}  |  Total: {elapsed:.1f}s",
        )

        try:
            _, run_output_dir = create_run_output_dir(
                app.config.run.output_dir, run_id=app.config.run.run_id
            )
        except (ValueError, FileExistsError):
            run_output_dir = app.config.run.output_dir / "tui_run"
            run_output_dir.mkdir(parents=True, exist_ok=True)
        _write_task_outputs(self.task_id, run_output_dir, run_result)

        pred_cols: list[str] = []
        pred_rows: list[list[str]] = []
        answer = run_result.get("answer")
        if isinstance(answer, dict):
            pred_cols = answer.get("columns", [])
            pred_rows = answer.get("rows", [])

        gold = _load_gold_csv(self.task_id)

        self.app.call_from_thread(
            self.app.push_screen,
            ResultsScreen(
                task_id=self.task_id,
                pred_cols=pred_cols,
                pred_rows=pred_rows,
                gold=gold,
                elapsed=elapsed,
                succeeded=run_result.get("succeeded", False),
            ),
        )

    def action_cancel_run(self) -> None:
        self._cancelled = True
        self.workers.cancel_all()
        log_widget = self.query_one("#agent-log", RichLog)
        log_widget.write("[bold yellow]Cancelled.[/bold yellow]")

    def action_go_back(self) -> None:
        self._cancelled = True
        self.workers.cancel_all()
        self.app.pop_screen()

    def action_quit_app(self) -> None:
        self.workers.cancel_all()
        self.app.exit()


class ResultsScreen(Screen):
    BINDINGS = [
        Binding("n", "run_next", "Next task"),
        Binding("p", "run_prev", "Prev task"),
        Binding("escape", "go_back", "Back to list"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(
        self,
        task_id: str,
        pred_cols: list[str],
        pred_rows: list[list[str]],
        gold: tuple[list[str], list[list[str]]] | None,
        elapsed: float,
        succeeded: bool,
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.pred_cols = pred_cols
        self.pred_rows = pred_rows
        self.gold = gold
        self.elapsed = elapsed
        self.succeeded = succeeded

    def compose(self) -> ComposeResult:
        yield Header()
        status = "[green]OK[/green]" if self.succeeded else "[red]FAIL[/red]"
        yield Label(
            f" Results: {self.task_id}  |  Status: {status}  |  Time: {self.elapsed:.1f}s",
            id="result-header",
        )
        with Horizontal(id="tables-row"):
            with Vertical(id="pred-panel"):
                yield Static("[bold]Agent Prediction[/bold]", id="pred-title")
                yield DataTable(id="pred-table")
            with Vertical(id="gold-panel"):
                yield Static("[bold]Gold Answer[/bold]", id="gold-title")
                yield DataTable(id="gold-table")
        match_label = self._compute_match()
        yield Label(match_label, id="match-label")
        yield Footer()

    def on_mount(self) -> None:
        pred_table = self.query_one("#pred-table", DataTable)
        if self.pred_cols:
            pred_table.add_columns(*self.pred_cols)
            for row in self.pred_rows:
                pred_table.add_row(*row)
        else:
            pred_table.add_column("(no prediction)")

        gold_table = self.query_one("#gold-table", DataTable)
        if self.gold:
            gold_cols, gold_rows = self.gold
            gold_table.add_columns(*gold_cols)
            for row in gold_rows:
                gold_table.add_row(*row)
        else:
            gold_table.add_column("(no gold answer available)")

    def _compute_match(self) -> str:
        if not self.gold:
            return " No gold answer to compare"
        gold_cols, gold_rows = self.gold

        if self.pred_cols == gold_cols and self.pred_rows == gold_rows:
            return " EXACT MATCH"

        norm_pred = {tuple(r) for r in self.pred_rows}
        norm_gold = {tuple(r) for r in gold_rows}
        if self.pred_cols == gold_cols and norm_pred == norm_gold:
            return " MATCH (row order differs)"

        if not self.pred_cols:
            return " NO PREDICTION produced"

        diffs: list[str] = []
        if self.pred_cols != gold_cols:
            diffs.append(f"columns differ: pred={self.pred_cols} vs gold={gold_cols}")
        if len(self.pred_rows) != len(gold_rows):
            diffs.append(f"row count: pred={len(self.pred_rows)} vs gold={len(gold_rows)}")
        return " MISMATCH: " + "; ".join(diffs) if diffs else " MISMATCH (values differ)"

    def _get_adjacent_task_id(self, direction: int) -> str | None:
        app: DABenchTUI = self.app  # type: ignore[assignment]
        task_ids = [t.task_id for t in app.dataset.iter_tasks()]
        try:
            idx = task_ids.index(self.task_id)
            new_idx = idx + direction
            if 0 <= new_idx < len(task_ids):
                return task_ids[new_idx]
        except ValueError:
            pass
        return None

    def action_run_next(self) -> None:
        next_id = self._get_adjacent_task_id(1)
        if next_id:
            self.app.switch_screen(RunScreen(next_id))

    def action_run_prev(self) -> None:
        prev_id = self._get_adjacent_task_id(-1)
        if prev_id:
            self.app.switch_screen(RunScreen(prev_id))

    def action_go_back(self) -> None:
        self.app.switch_screen(TaskListScreen(self.app.dataset))  # type: ignore[attr-defined]

    def action_quit_app(self) -> None:
        self.app.exit()


class DABenchTUI(App):
    CSS = """
    #task-hint {
        padding: 1;
        color: $text-muted;
    }
    #task-table {
        height: 1fr;
    }
    #run-label {
        padding: 1;
        color: $accent;
    }
    #agent-log {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    #result-header {
        padding: 1;
    }
    #tables-row {
        height: 1fr;
    }
    #pred-panel, #gold-panel {
        width: 1fr;
        padding: 0 1;
    }
    #pred-title, #gold-title {
        text-align: center;
        padding: 0 0 1 0;
    }
    #pred-table, #gold-table {
        height: 1fr;
        border: round $primary;
    }
    #match-label {
        padding: 1;
        text-align: center;
        text-style: bold;
    }
    """

    TITLE = "DABench Task Runner"

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.dataset = DABenchPublicDataset(config.dataset.root_path)
        self.model = build_model_adapter(config)
        self.tools = create_default_tool_registry()
        self.fast_model = build_fast_model_adapter(config)

    def on_mount(self) -> None:
        self.push_screen(TaskListScreen(self.dataset))


def run_tui(config_path: Path) -> None:
    config = load_app_config(config_path)
    app = DABenchTUI(config)
    app.run()
