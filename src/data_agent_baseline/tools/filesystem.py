from __future__ import annotations

import csv
import json
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }


def read_csv_preview(
    task: PublicTask, relative_path: str, *, max_rows: int = 20
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return {
                "path": relative_path,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "total_row_count": 0,
            }
        preview_rows: list[list[str]] = []
        for row in reader:
            if len(preview_rows) < max_rows:
                preview_rows.append(row)
            else:
                break
        has_more = len(preview_rows) == max_rows
        total_estimate: int | None = None
        if has_more:
            file_size = path.stat().st_size
            if file_size > 1_000_000:
                total_estimate = _estimate_csv_rows(path, file_size)
    return {
        "path": relative_path,
        "columns": header,
        "rows": preview_rows,
        "row_count": len(preview_rows),
        "has_more": has_more,
        **({"estimated_total_rows": total_estimate} if total_estimate else {}),
    }


def _estimate_csv_rows(path: Path, file_size: int) -> int:
    sample_size = 0
    sample_lines = 0
    with path.open(newline="") as handle:
        next(handle, None)
        for line in handle:
            sample_size += len(line.encode("utf-8", errors="replace"))
            sample_lines += 1
            if sample_lines >= 100:
                break
    if sample_lines == 0 or sample_size == 0:
        return 0
    avg_line_bytes = sample_size / sample_lines
    return int(file_size / avg_line_bytes)


def read_json_preview(
    task: PublicTask, relative_path: str, *, max_chars: int = 4000
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    file_size = path.stat().st_size
    if file_size <= max_chars * 3:
        payload = json.loads(path.read_text())
        preview = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        raw = path.read_text(errors="replace")[: max_chars * 3]
        try:
            payload = json.loads(raw)
            preview = json.dumps(payload, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            preview = raw
    return {
        "path": relative_path,
        "preview": preview[:max_chars],
        "truncated": file_size > max_chars * 3,
        "file_size_bytes": file_size,
    }


def read_doc_preview(
    task: PublicTask,
    relative_path: str,
    *,
    max_chars: int = 4000,
    offset: int = 0,
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    file_size = path.stat().st_size
    text = path.read_text(errors="replace")
    total_chars = len(text)
    chunk = text[offset : offset + max_chars]
    return {
        "path": relative_path,
        "preview": chunk,
        "offset": offset,
        "chars_returned": len(chunk),
        "total_chars": total_chars,
        "file_size_bytes": file_size,
        "truncated": (offset + len(chunk)) < total_chars,
    }
