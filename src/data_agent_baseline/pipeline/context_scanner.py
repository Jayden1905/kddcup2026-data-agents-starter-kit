"""Deterministic context scanner — inventories all data sources in a task.

No LLM calls. Catalogs files, reads schemas, provides previews.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StructuredSource:
    path: Path
    file_type: str  # "csv", "json", "sqlite"
    table_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DocSource:
    path: Path
    size_bytes: int
    preview: str  # first ~500 chars


@dataclass(slots=True)
class TaskContext:
    context_dir: Path
    structured_sources: list[StructuredSource] = field(default_factory=list)
    doc_sources: list[DocSource] = field(default_factory=list)
    knowledge_text: str = ""
    task_type: str = "structured_only"  # structured_only | doc_only | mixed


def scan_context(context_dir: Path) -> TaskContext:
    """Scan the task's context/ directory and catalog all data sources."""
    ctx = TaskContext(context_dir=context_dir)

    # CSVs
    csv_dir = context_dir / "csv"
    if csv_dir.is_dir():
        for f in sorted(csv_dir.glob("*.csv")):
            ctx.structured_sources.append(StructuredSource(
                path=f, file_type="csv", table_names=[f.stem]
            ))

    # JSONs
    json_dir = context_dir / "json"
    if json_dir.is_dir():
        for f in sorted(json_dir.glob("*.json")):
            ctx.structured_sources.append(StructuredSource(
                path=f, file_type="json", table_names=[f.stem]
            ))

    # SQLite DBs
    db_dir = context_dir / "db"
    if db_dir.is_dir():
        for f in sorted(db_dir.glob("*.db")) + sorted(db_dir.glob("*.sqlite")):
            ctx.structured_sources.append(StructuredSource(
                path=f, file_type="sqlite", table_names=_get_sqlite_tables(f)
            ))
    # Also check for .db files directly in context/
    for f in sorted(context_dir.glob("*.db")):
        if f.name != "_consolidated.db":
            ctx.structured_sources.append(StructuredSource(
                path=f, file_type="sqlite", table_names=_get_sqlite_tables(f)
            ))

    # Documents
    doc_dir = context_dir / "doc"
    if doc_dir.is_dir():
        for f in sorted(doc_dir.iterdir()):
            if f.suffix in (".md", ".txt", ".text"):
                size = f.stat().st_size
                preview = f.read_text(errors="replace")[:500]
                ctx.doc_sources.append(DocSource(path=f, size_bytes=size, preview=preview))

    # Knowledge.md
    knowledge_path = context_dir / "knowledge.md"
    if knowledge_path.exists():
        ctx.knowledge_text = knowledge_path.read_text(errors="replace")

    # Determine task type
    has_structured = len(ctx.structured_sources) > 0
    has_docs = len(ctx.doc_sources) > 0
    if has_structured and has_docs:
        ctx.task_type = "mixed"
    elif has_docs:
        ctx.task_type = "doc_only"
    else:
        ctx.task_type = "structured_only"

    return ctx


def _get_sqlite_tables(db_path: Path) -> list[str]:
    """Get table names from a SQLite file."""
    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception:
        return []
