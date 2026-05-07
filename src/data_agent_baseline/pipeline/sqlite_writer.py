"""Writes merged records to SQLite."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.pipeline.field_discoverer import DocumentSchema
from data_agent_baseline.pipeline.record_merger import MergedRecord, MergeResult


def _sanitize_column_name(name: str) -> str:
    """Replace non-alphanumeric characters with underscores."""
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Strip leading/trailing underscores
    return sanitized.strip("_").lower()


def _infer_column_type(values: list[Any]) -> str:
    """Infer column type: if all non-None values are float-like, use REAL; otherwise TEXT."""
    for val in values:
        if val is None:
            continue
        try:
            float(val)
        except (ValueError, TypeError):
            return "TEXT"
    return "REAL"


def _resolve_table_name(entity_name: str, protected_tables: set[str] | None) -> str:
    """Resolve table name, adding _doc suffix if it collides with protected tables."""
    table_name = _sanitize_column_name(entity_name)
    if protected_tables and table_name in protected_tables:
        table_name = f"{table_name}_doc"
    return table_name


def write_to_sqlite(
    db_path: Path,
    merge_result: MergeResult,
    schema: DocumentSchema,
    protected_tables: set[str] | None = None,
    log_fn: Callable[[str, str], None] | None = None,
) -> None:
    """Write merged records to SQLite.

    - Table name = schema.entity_name (or entity_name + "_doc" if name collides with protected)
    - Create table with ID TEXT PRIMARY KEY + discovered columns
    - Infer column types: if all values are float-like -> REAL, otherwise TEXT
    - INSERT OR REPLACE
    """
    if not merge_result.records:
        if log_fn:
            log_fn("sqlite_writer", "No records to write, skipping.")
        return

    table_name = _resolve_table_name(schema.entity_name, protected_tables)

    # Discover all columns from records
    all_columns: set[str] = set()
    for record in merge_result.records:
        all_columns.update(record.fields.keys())

    # Sanitize column names and build mapping (dedup on sanitized name)
    col_mapping: dict[str, str] = {}
    seen_sanitized: set[str] = set()
    for col in sorted(all_columns):
        sanitized = _sanitize_column_name(col)
        if sanitized not in seen_sanitized:
            col_mapping[col] = sanitized
            seen_sanitized.add(sanitized)
        else:
            col_mapping[col] = sanitized

    # Infer column types
    col_types: dict[str, str] = {}
    for original_col, sanitized_col in col_mapping.items():
        values = [r.fields.get(original_col) for r in merge_result.records]
        col_types[sanitized_col] = _infer_column_type(values)

    # Build CREATE TABLE statement (deduplicated)
    columns_sql = ["record_id TEXT PRIMARY KEY"]
    sanitized_cols = sorted(set(col_mapping.values()))
    for col in sanitized_cols:
        if col == "record_id":
            continue
        col_type = col_types.get(col, "TEXT")
        columns_sql.append(f"{col} {col_type}")

    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns_sql)})"

    # Build INSERT statement
    insert_cols = ["record_id"] + [c for c in sanitized_cols if c != "record_id"]
    placeholders = ", ".join(["?"] * len(insert_cols))
    insert_sql = (
        f"INSERT OR REPLACE INTO {table_name} ({', '.join(insert_cols)}) "
        f"VALUES ({placeholders})"
    )

    # Connect and write
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")

        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(create_sql)

        # Insert records
        for record in merge_result.records:
            row_values: list[Any] = [record.record_id]
            for col in sanitized_cols:
                if col == "record_id":
                    continue
                # Find original column name for this sanitized name
                original_col = None
                for orig, san in col_mapping.items():
                    if san == col:
                        original_col = orig
                        break
                val = record.fields.get(original_col) if original_col else None
                row_values.append(val)

            conn.execute(insert_sql, row_values)

        conn.commit()

        if log_fn:
            log_fn(
                "sqlite_writer",
                f"Wrote {len(merge_result.records)} records to table '{table_name}'",
            )
    finally:
        conn.close()
