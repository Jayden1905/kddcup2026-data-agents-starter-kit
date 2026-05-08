"""Docker entrypoint: reads /input, writes /output/task_<id>/prediction.csv."""

import os
import shutil
from pathlib import Path

import yaml

from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.runner import run_benchmark


def run():
    input_dir = Path("/input")
    output_dir = Path("/output")

    model_api_url = os.environ.get("MODEL_API_URL", "")
    model_api_key = os.environ.get("MODEL_API_KEY", "")
    model_name = os.environ.get("MODEL_NAME", "")

    if not model_api_url or not model_api_key or not model_name:
        raise RuntimeError(
            "MODEL_API_URL, MODEL_API_KEY, and MODEL_NAME must be set."
        )

    config = {
        "dataset": {"root_path": str(input_dir)},
        "agent": {
            "agent_type": "question_driven",
            "backend": "openai",
            "model": model_name,
            "api_base": model_api_url,
            "api_key": model_api_key,
            "temperature": 0.0,
        },
        "run": {
            "output_dir": "/tmp/run_output",
            "max_workers": 1,
            "task_timeout_seconds": 0,
        },
    }

    config_path = Path("/tmp/docker_config.yaml")
    config_path.write_text(yaml.dump(config))

    app_config = load_app_config(config_path)
    _, artifacts = run_benchmark(config=app_config)

    for artifact in artifacts:
        if artifact.prediction_csv_path and artifact.prediction_csv_path.exists():
            out_task_dir = output_dir / artifact.task_id
            out_task_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact.prediction_csv_path, out_task_dir / "prediction.csv")


if __name__ == "__main__":
    run()
