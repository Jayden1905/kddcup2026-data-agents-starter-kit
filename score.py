"""Score predictions against gold answers using official KDD Cup 2026 evaluation rules.

Scoring: column-level content consistency matching (column signatures).
- Ignores column names, matches only by sorted cell values.
- Ignores row order (sorting constructs signature).
- Supports duplicate columns (signature count must match).
- Penalty for extra (unmatched) prediction columns.

Score = Recall - λ * (Extra Columns / Predicted Columns)
  Recall = Matched Columns / Gold Columns
  λ = 0.2
  Lower bound = 0
"""

import csv
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

PENALTY_LAMBDA = 0.2
NULL_STRINGS = {"", "null", "none", "nan", "nat", "<na>"}


def _normalize_cell(value: str) -> str:
    """Normalize a cell value per official eval rules.

    - Null values → ""
    - Numeric → 2 decimal places (ROUND_HALF_UP)
    - Date → YYYY-MM-DD (zero-padded)
    - DateTime with TZ → UTC ending with Z
    - String → strip whitespace/CRLF, case-sensitive
    """
    stripped = value.strip().replace("\r\n", "\n").replace("\r", "")
    if stripped.lower() in NULL_STRINGS:
        return ""
    # Numeric
    try:
        d = Decimal(stripped)
        if d.is_finite():
            return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):
        pass
    # Date: YYYY-M-D → YYYY-MM-DD
    date_match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", stripped)
    if date_match:
        y, m, d_val = date_match.groups()
        return f"{y}-{int(m):02d}-{int(d_val):02d}"
    # DateTime with timezone → UTC
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


def score_prediction(
    pred_cols: list[str],
    pred_rows: list[list[str]],
    gold_cols: list[str],
    gold_rows: list[list[str]],
) -> tuple[float, int, int, int]:
    """Score prediction against gold using column-signature matching.

    Returns (score, matched_columns, gold_columns, predicted_columns).
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

    # Match by signature count
    gold_counter: Counter[tuple[str, ...]] = Counter(gold_sigs)
    pred_counter: Counter[tuple[str, ...]] = Counter(pred_sigs)

    matched = 0
    for sig, g_count in gold_counter.items():
        p_count = pred_counter.get(sig, 0)
        matched += min(g_count, p_count)

    gold_count = len(gold_cols)
    pred_count = len(pred_cols)
    extra = pred_count - matched

    # Score = Recall - λ * (Extra / Predicted)
    recall = matched / gold_count if gold_count > 0 else 0.0
    penalty = PENALTY_LAMBDA * (extra / pred_count) if pred_count > 0 else 0.0
    score = max(0.0, recall - penalty)

    return score, matched, gold_count, pred_count


def load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Load CSV, return (columns, data_rows)."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python score.py <prediction_dir> [gold_dir]")
        print("  prediction_dir: folder containing task_*/prediction.csv")
        print("  gold_dir: folder containing task_*/gold.csv (default: data/public/output)")
        sys.exit(1)

    pred_dir = Path(sys.argv[1])
    gold_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/public/output")

    if not pred_dir.exists():
        print(f"Error: {pred_dir} does not exist")
        sys.exit(1)
    if not gold_dir.exists():
        print(f"Error: {gold_dir} does not exist")
        sys.exit(1)

    tasks = sorted(
        [d.name for d in pred_dir.iterdir() if d.is_dir() and d.name.startswith("task_")]
    )
    if not tasks:
        print(f"No task_* directories found in {pred_dir}")
        sys.exit(1)

    scores: list[tuple[str, float]] = []
    perfect: list[str] = []
    partial: list[tuple[str, float, int, int, int]] = []
    zero: list[tuple[str, str]] = []

    for task in tasks:
        pred_file = pred_dir / task / "prediction.csv"
        gold_file = gold_dir / task / "gold.csv"

        if not gold_file.exists():
            continue
        if not pred_file.exists():
            scores.append((task, 0.0))
            zero.append((task, "NO PREDICTION"))
            continue

        try:
            pred_cols, pred_rows = load_csv(pred_file)
            gold_cols, gold_rows = load_csv(gold_file)
            score, matched, g_count, p_count = score_prediction(
                pred_cols, pred_rows, gold_cols, gold_rows
            )
            scores.append((task, score))
            if score >= 1.0:
                perfect.append(task)
            elif score > 0:
                partial.append((task, score, matched, g_count, p_count))
            else:
                zero.append((task, f"matched=0/{g_count} pred_cols={p_count}"))
        except Exception as e:
            scores.append((task, 0.0))
            zero.append((task, str(e)))

    if not scores:
        print("No tasks with gold answers found.")
        sys.exit(1)

    avg = sum(s for _, s in scores) / len(scores)

    if perfect:
        print(f"--- PERFECT ({len(perfect)}) ---")
        for t in perfect:
            print(f"  {t}")
        print()

    if partial:
        print(f"--- PARTIAL ({len(partial)}) ---")
        for t, s, m, g, p in sorted(partial, key=lambda x: -x[1]):
            print(f"  {t}: score={s:.3f} (matched {m}/{g} gold cols, pred has {p} cols)")
        print()

    if zero:
        print(f"--- ZERO ({len(zero)}) ---")
        for t, reason in sorted(zero):
            print(f"  {t}: {reason}")
        print()

    print(f"{'='*60}")
    print(f"  KDD Cup 2026 - Column Signature Scoring")
    print(f"{'='*60}")
    print(f"  Tasks scored:    {len(scores)}")
    print(f"  Average Score:   {avg:.4f} ({avg*100:.1f}%)")
    print(f"  Perfect (1.0):   {len(perfect)}")
    print(f"  Partial (0<s<1): {len(partial)}")
    print(f"  Zero (0.0):      {len(zero)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
