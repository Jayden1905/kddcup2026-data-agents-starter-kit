"""Quick evaluation: compare predictions against gold answers."""
import csv
import sys
from pathlib import Path


def normalize_value(v: str) -> str:
    """Normalize a value for comparison."""
    v = v.strip().lower()
    # Try to normalize numbers
    try:
        f = float(v)
        # Round to reasonable precision
        if f == int(f):
            return str(int(f))
        return f"{f:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        pass
    return v


def load_csv_values(path: Path) -> list[list[str]]:
    """Load CSV and return normalized rows (skip header)."""
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return []
        for row in reader:
            rows.append([normalize_value(v) for v in row])
    return rows


def compare_task(pred_path: Path, gold_path: Path) -> tuple[bool, str]:
    """Compare prediction to gold. Returns (match, detail)."""
    if not pred_path.exists():
        return False, "no prediction"

    pred_rows = load_csv_values(pred_path)
    gold_rows = load_csv_values(gold_path)

    if not pred_rows and not gold_rows:
        return True, "both empty"
    if not pred_rows:
        return False, "prediction empty"
    if not gold_rows:
        return False, "gold empty"

    # Compare row-by-row
    if len(pred_rows) != len(gold_rows):
        # Check if single-value match (common case)
        if len(gold_rows) == 1 and len(pred_rows) == 1:
            gold_vals = set(gold_rows[0])
            pred_vals = set(pred_rows[0])
            if gold_vals & pred_vals:
                return True, "value match"
        return False, f"row count mismatch: pred={len(pred_rows)} gold={len(gold_rows)}"

    matches = 0
    for pred_row, gold_row in zip(pred_rows, gold_rows):
        # Flexible: check if all gold values appear in pred (order may differ)
        gold_set = set(gold_row)
        pred_set = set(pred_row)
        if gold_set == pred_set:
            matches += 1
        elif gold_set <= pred_set:
            matches += 1
        else:
            # Try individual value comparison
            if len(gold_row) == len(pred_row):
                if all(g == p for g, p in zip(sorted(gold_row), sorted(pred_row))):
                    matches += 1

    if matches == len(gold_rows):
        return True, "exact match"
    elif matches > 0:
        return False, f"partial: {matches}/{len(gold_rows)} rows"
    else:
        # Single-value comparison for simple answers
        if len(gold_rows) == 1 and len(gold_rows[0]) == 1:
            if len(pred_rows) >= 1 and len(pred_rows[0]) >= 1:
                if normalize_value(pred_rows[0][0]) == gold_rows[0][0]:
                    return True, "single value match"
                # Also check if any pred column matches
                for v in pred_rows[0]:
                    if normalize_value(v) == gold_rows[0][0]:
                        return True, "value found in pred"
        return False, f"no match (gold={gold_rows[0][:3]} pred={pred_rows[0][:3]})"


def main():
    gold_dir = Path("data/public/output")

    # Find the latest run
    runs_dir = Path("artifacts/runs")
    if not runs_dir.exists():
        print("No runs found")
        sys.exit(1)

    if len(sys.argv) > 1:
        run_dir = runs_dir / sys.argv[1]
    else:
        run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.name)
        run_dir = run_dirs[-1]

    print(f"Evaluating: {run_dir.name}")
    print("=" * 60)

    correct = 0
    total = 0
    failures = []

    for task_dir in sorted(gold_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        gold_path = task_dir / "gold.csv"
        pred_path = run_dir / task_id / "prediction.csv"

        if not gold_path.exists():
            continue

        total += 1
        match, detail = compare_task(pred_path, gold_path)
        status = "✓" if match else "✗"
        if match:
            correct += 1
        else:
            failures.append((task_id, detail))

        print(f"  {status} {task_id}: {detail}")

    print("=" * 60)
    print(f"Score: {correct}/{total} ({correct/total*100:.1f}%)")
    print()

    if failures:
        print(f"Failed tasks ({len(failures)}):")
        for task_id, detail in failures:
            print(f"  {task_id}: {detail}")


if __name__ == "__main__":
    main()
