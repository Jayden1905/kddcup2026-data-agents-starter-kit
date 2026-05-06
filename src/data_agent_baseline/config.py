from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    agent_type: str = "react"
    backend: str = "openai"
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    api_version: str = ""
    deployment_name: str = ""
    max_steps: int = 16
    max_investigation_iterations: int = 5
    temperature: float = 0.0
    fast_model: str = ""
    fast_deployment_name: str = ""
    fast_backend: str = ""
    fast_api_base: str = ""
    fast_api_key: str = ""


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        agent_type=str(agent_payload.get("agent_type", agent_defaults.agent_type)),
        backend=str(agent_payload.get("backend", agent_defaults.backend)),
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        api_version=str(agent_payload.get("api_version", agent_defaults.api_version)),
        deployment_name=str(agent_payload.get("deployment_name", agent_defaults.deployment_name)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        max_investigation_iterations=int(
            agent_payload.get(
                "max_investigation_iterations", agent_defaults.max_investigation_iterations
            )
        ),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        fast_model=str(agent_payload.get("fast_model", agent_defaults.fast_model)),
        fast_deployment_name=str(
            agent_payload.get("fast_deployment_name", agent_defaults.fast_deployment_name)
        ),
        fast_backend=str(agent_payload.get("fast_backend", agent_defaults.fast_backend)),
        fast_api_base=str(agent_payload.get("fast_api_base", agent_defaults.fast_api_base)),
        fast_api_key=str(agent_payload.get("fast_api_key", agent_defaults.fast_api_key)),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(
            run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)
        ),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
