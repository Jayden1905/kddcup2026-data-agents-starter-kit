# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

KDD Cup 2026 DataAgent-Bench starter kit — a data agent that reads tasks from `data/public/input/` and produces `prediction.csv` answers. Supports two agent modes (ReAct and Investigation) and two LLM backends (OpenAI-compatible and Azure OpenAI).

## Commands

```bash
uv sync                                                           # Install dependencies
uv run dabench status --config configs/react_baseline.example.yaml       # Check project layout and dataset
uv run dabench inspect-task task_1 --config configs/react_baseline.local.yaml  # View task metadata
uv run dabench run-task task_1 --config configs/react_baseline.local.yaml      # Run one task
uv run dabench run-benchmark --config configs/react_baseline.local.yaml        # Run all tasks
uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 5  # Run N tasks
uv run pytest                                                     # Run tests
uv run ruff check src                                             # Lint
uv run ruff format src                                            # Format
```

Copy `configs/react_baseline.example.yaml` to `configs/react_baseline.local.yaml` and fill in `agent.model`, `agent.api_base`, and `agent.api_key`. Local configs are gitignored.

## Architecture

The pipeline is: CLI (`cli.py`) -> Runner (`run/runner.py`) -> Agent (`react.py` or `investigation.py`) -> ModelAdapter + ToolRegistry. The runner selects agent type and LLM backend based on `agent_type` and `backend` config fields.

**ReAct agent** (`agents/react.py`): `ReActAgent.run()` iterates up to `max_steps`. Each step: build message history -> call LLM -> parse JSON response (thought/action/action_input) -> execute tool -> append observation. Terminates when the `answer` tool is called or steps are exhausted.

**Investigation agent** (`agents/investigation.py`): Closed-loop V2 agent. Each iteration: PLAN (LLM generates tool call steps) -> EXECUTE (run all planned tools, collect evidence) -> EVALUATE (LLM judges if evidence is sufficient, identifies gaps) -> loop or SYNTHESIZE (LLM produces final answer table from evidence). Up to `max_investigation_iterations` (default 5).

**Model layer** (`agents/model.py`): `ModelAdapter` protocol with `OpenAIModelAdapter` (any OpenAI-compatible API), `AzureOpenAIModelAdapter` (Azure OpenAI with endpoint normalization), and `ScriptedModelAdapter` (testing).

**Tool system** (`tools/registry.py`): `ToolRegistry` holds `ToolSpec` (schema for prompt) + `ToolHandler` (execution function). Tools receive a `PublicTask` and `action_input` dict, return `ToolExecutionResult`. The `answer` tool is terminal (`is_terminal=True`). Tools: `list_context`, `read_csv`, `read_json`, `read_doc`, `inspect_sqlite_schema`, `execute_context_sql`, `execute_python`, `answer`.

**Python execution** (`tools/python_exec.py`): Runs user code in a subprocess with fd-level stdout/stderr capture. 30-second timeout. Working directory is the task's `context/` dir.

**Task timeout** (`run/runner.py`): When `task_timeout_seconds > 0`, each task runs in a separate `multiprocessing.Process` with terminate/kill escalation. Benchmark parallelism uses `ThreadPoolExecutor` (each thread spawns a subprocess for timeout).

**Dataset** (`benchmark/dataset.py`, `benchmark/schema.py`): `DABenchPublicDataset` loads tasks from `data/public/input/task_<id>/`. Each task has `task.json` (task_id, difficulty, question) and a `context/` directory with data files. `PublicTask` is the core domain object passed throughout.

**Config** (`config.py`): YAML-based, loaded into frozen dataclasses (`AppConfig` -> `DatasetConfig`, `AgentConfig`, `RunConfig`). Relative paths resolve from project root.

**Prompt construction** (`agents/prompt.py`): System prompt includes ReAct rules, tool descriptions, and response format examples. LLM must return a single fenced JSON block with `thought`, `action`, `action_input`.

## Key Conventions

- Python 3.10+, managed with `uv` and `hatchling` build backend
- All dataclasses use `frozen=True, slots=True`
- Ruff for linting/formatting, line length 100
- All tool file paths are relative to the task's `context/` directory, enforced by `resolve_context_path` (path traversal guard)
- `data/`, `tests/`, `artifacts/`, and local configs are gitignored
- Output structure: `artifacts/runs/<run_id>/<task_id>/{trace.json, prediction.csv}`
