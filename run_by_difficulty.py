#!/usr/bin/env python3
"""Run benchmark filtered by difficulty group with interactive selection."""

import json
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("data/public/input")


def get_difficulty_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for task_dir in sorted(DATA_DIR.iterdir()):
        task_json = task_dir / "task.json"
        if not task_json.exists():
            continue
        with open(task_json) as f:
            data = json.load(f)
        difficulty = data.get("difficulty", "unknown")
        groups.setdefault(difficulty, []).append(task_dir.name)
    return groups


def main() -> None:
    groups = get_difficulty_groups()
    if not groups:
        print("No tasks found in", DATA_DIR)
        sys.exit(1)

    # Parse optional CLI args
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark by difficulty group")
    parser.add_argument(
        "difficulty",
        nargs="?",
        help="Difficulty level (easy/medium/hard/extreme). Interactive if omitted.",
    )
    parser.add_argument("--config", default="configs/react_baseline.local.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks within the group")
    args = parser.parse_args()

    if args.difficulty:
        chosen = args.difficulty.lower()
    else:
        print("\nAvailable difficulty groups:")
        sorted_keys = sorted(groups.keys())
        for i, diff in enumerate(sorted_keys, 1):
            print(f"  {i}. {diff} ({len(groups[diff])} tasks)")
        print()
        choice = input("Select difficulty (name or number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(sorted_keys):
            chosen = sorted_keys[int(choice) - 1]
        else:
            chosen = choice.lower()

    if chosen not in groups:
        print(f"Unknown difficulty '{chosen}'. Available: {sorted(groups.keys())}")
        sys.exit(1)

    task_count = len(groups[chosen])
    effective = min(task_count, args.limit) if args.limit else task_count
    print(f"\nRunning {effective} '{chosen}' tasks...\n")

    cmd = [
        "uv", "run", "dabench", "run-benchmark",
        "--config", args.config,
        "--difficulty", chosen,
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
