"""Graph builder that runs concurrently with extraction agents.

Continuously consumes from the entity queue and writes to SQLite.
Creates tables on-the-fly as new entity types arrive.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from data_agent_baseline.pipeline.entity_queue import EntityQueue, Relationship


def _sanitize_name(name: str) -> str:
    """Sanitize a table or column name for SQLite."""
    name = re.sub(r"[^\w]", "_", name.strip())
    if name and name[0].isdigit():
        name = "_" + name
    return name or "_unknown"


class GraphBuilder:
    """Consumes entities from the queue and writes them to SQLite in real-time."""

    def __init__(
        self,
        db_path: Path,
        queue: EntityQueue,
        log_fn: Callable[[str, str], None] | None = None,
        protected_tables: set[str] | None = None,
    ) -> None:
        self.db_path = db_path
        self.queue = queue
        self.log_fn = log_fn
        self._protected_tables = protected_tables or set()
        self._stop_event = threading.Event()
        self._known_tables: dict[str, set[str]] = {}
        self._total_entities = 0
        self._total_rels = 0

    def run(self) -> None:
        """Main loop: pull from queue, write to SQLite. Runs until stopped."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")

        while not self._stop_event.is_set():
            if not self.queue.wait_for_data(timeout=0.5):
                continue
            self._flush(conn)

        # Final drain after stop signal
        self._flush(conn)
        conn.close()

    def stop(self) -> None:
        self._stop_event.set()

    def _flush(self, conn: sqlite3.Connection) -> None:
        entities = self.queue.drain_entities()
        relationships = self.queue.drain_relationships()

        if entities:
            self._write_entities(conn, entities)
            batch_count = sum(len(records) for records in entities.values())
            self._total_entities += batch_count
        if relationships:
            self._write_relationships(conn, relationships)
            self._total_rels += len(relationships)
        if entities or relationships:
            conn.commit()
            if self.log_fn:
                self.log_fn(
                    "graph_builder",
                    f"Flushed: {self._total_entities} entities, "
                    f"{self._total_rels} rels in {list(entities.keys()) if entities else []}",
                )

    def _write_entities(self, conn: sqlite3.Connection, entities_by_type: dict[str, list[dict[str, Any]]]) -> None:
        for entity_type, records in entities_by_type.items():
            table_name = _sanitize_name(entity_type)
            self._ensure_table(conn, table_name, records)
            self._upsert_records(conn, table_name, records)

    def _ensure_table(self, conn: sqlite3.Connection, table_name: str, records: list[dict[str, Any]]) -> None:
        all_columns: set[str] = set()
        for record in records:
            all_columns.update(k for k in record.keys() if k != "_id")

        if table_name not in self._known_tables:
            existing = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            existing_cols = {row[1] for row in existing}

            if existing_cols and table_name in self._protected_tables:
                # Structured table from CSV/JSON — add columns but never drop
                self._known_tables[table_name] = existing_cols
                for col in all_columns:
                    safe_col = _sanitize_name(col)
                    if safe_col not in existing_cols:
                        try:
                            conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_col}" TEXT')
                        except sqlite3.OperationalError:
                            pass
                        existing_cols.add(safe_col)
            elif not existing_cols:
                # New table — create with ID primary key
                cols_sql = ', '.join(
                    f'"{_sanitize_name(c)}" TEXT' for c in sorted(all_columns)
                )
                create_sql = (
                    f'CREATE TABLE IF NOT EXISTS "{table_name}" '
                    f'("ID" TEXT PRIMARY KEY, {cols_sql})'
                )
                try:
                    conn.execute(create_sql)
                except sqlite3.OperationalError:
                    pass
                self._known_tables[table_name] = {"ID"} | {_sanitize_name(c) for c in all_columns}
            else:
                # Existing non-protected table (shouldn't happen now) — recreate
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                cols_sql = ', '.join(
                    f'"{_sanitize_name(c)}" TEXT' for c in sorted(all_columns)
                )
                create_sql = (
                    f'CREATE TABLE "{table_name}" '
                    f'("ID" TEXT PRIMARY KEY, {cols_sql})'
                )
                try:
                    conn.execute(create_sql)
                except sqlite3.OperationalError:
                    pass
                self._known_tables[table_name] = {"ID"} | {_sanitize_name(c) for c in all_columns}
        else:
            known = self._known_tables[table_name]
            for col in all_columns:
                safe_col = _sanitize_name(col)
                if safe_col not in known:
                    try:
                        conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_col}" TEXT')
                    except sqlite3.OperationalError:
                        pass
                    known.add(safe_col)

    def _upsert_records(self, conn: sqlite3.Connection, table_name: str, records: list[dict[str, Any]]) -> None:
        for record in records:
            entity_id = record.get("_id", "")
            fields = {_sanitize_name(k): v for k, v in record.items() if k != "_id" and v is not None}
            fields["ID"] = entity_id

            columns = list(fields.keys())
            placeholders = ", ".join("?" for _ in columns)
            col_names = ", ".join(f'"{c}"' for c in columns)
            update_clause = ", ".join(f'"{c}" = excluded."{c}"' for c in columns if c != "ID")

            sql = (
                f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders}) '
                f'ON CONFLICT("ID") DO UPDATE SET {update_clause}'
            )
            try:
                conn.execute(sql, list(fields.values()))
            except sqlite3.OperationalError:
                pass

    def _write_relationships(self, conn: sqlite3.Connection, relationships: list[Relationship]) -> None:
        if not relationships:
            return

        conn.execute(
            'CREATE TABLE IF NOT EXISTS "_relationships" '
            '("source_type" TEXT, "source_id" TEXT, "rel_type" TEXT, '
            '"target_type" TEXT, "target_id" TEXT, '
            'UNIQUE("source_type", "source_id", "rel_type", "target_type", "target_id"))'
        )
        for rel in relationships:
            try:
                conn.execute(
                    'INSERT OR IGNORE INTO "_relationships" VALUES (?, ?, ?, ?, ?)',
                    (rel.source_type, rel.source_id, rel.rel_type, rel.target_type, rel.target_id),
                )
            except sqlite3.OperationalError:
                pass
