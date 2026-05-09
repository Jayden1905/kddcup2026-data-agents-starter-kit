from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


KNOWLEDGE_GRAPH_PROMPT = """
You are a semantic data analyst. Given the QUESTION, data schemas, and domain
knowledge, produce a structured grounding response that gives a downstream data
agent everything it needs to answer correctly without further schema discovery.

FILE TREE:
{file_tree}

DATA SCHEMA:
{schema_hint}

DOMAIN KNOWLEDGE:
{knowledge_text}

QUESTION:
{question}

CURRENT DATE: {current_date}
Use this for any age, duration, or time-elapsed calculations (e.g. strftime('%Y', '{current_date}') for year).

Return ONLY a JSON object (no markdown fences) with these keys:
{{
  "computation_steps": [
    {{
      "step": 1,
      "description": "what this step does",
      "sql": "SELECT DISTINCT d.number FROM qualifying q JOIN drivers d ON q.driverId = d.driverId WHERE q.raceId = 903 AND q.q3 LIKE '1:54%'",
      "tool": "execute_context_sql",
      "path": "_consolidated.db"
    }}
  ],
  "data_requirements": [
    {{"name": "descriptive name", "table": "table_name", "column": "column_name", "description": "what this value represents"}}
  ],
  "join_paths": [
    {{"from_table": "table_a", "to_table": "table_b", "on": "table_a.col = table_b.col"}}
  ],
  "filters": [
    {{"table": "table_name", "column": "column_name", "operator": "= or LIKE or > etc", "value": "the value or pattern"}}
  ],
  "domain_rules": [
    "constraint or rule from knowledge.md that affects the computation"
  ],
  "ambiguous_columns": [
    {{"column": "col_name", "wrong_source": "table.column (wrong meaning)", "correct_source": "table.column (correct meaning)"}}
  ],
  "reasoning": "Free-form explanation of the query strategy and why this approach is correct",
  "entity_schemas": [
    {{"entity_name": "name", "fields": [{{"name": "field", "field_type": "text|integer|real|date", "description": "what it is"}}]}}
  ]
}}

RULES:
- "computation_steps" is the primary output. Each step should be a COMPLETE,
  EXECUTABLE SQL query or Python code that the agent can run directly.
  The steps should together produce the final answer.
- For time/date values in the question, use PRECISION-AWARE matching:
  "0:01:54" in a question means the data likely stores "1:54.xxx" → use LIKE '1:54%'
  "June 2013" → use LIKE '2013-06%' or BETWEEN
  Always use LIKE prefix matching when the question's precision is lower than the data's.
- For filters, always use case-insensitive matching: LOWER(col) = LOWER('value')
  or col LIKE '%value%' COLLATE NOCASE.
- "join_paths" lists FK relationships needed to traverse from filter table to answer table.
- "ambiguous_columns" flags columns where the same name means different things in
  different tables. The "correct_source" is where the agent should SELECT from.
- "data_requirements" lists each table.column needed and what it means.
- "domain_rules" extracts constraints from knowledge.md (e.g. "abnormal creatinine means CRE > 1.5").
- "reasoning" explains WHY this query approach is correct.
- Table names in _consolidated.db match filenames without extension
  (e.g. qualifying.csv → "qualifying" table, drivers.json → "drivers" table).
- Use DISTINCT in SQL to avoid duplicate rows from joins.
- If the question asks for a property of entity X, SELECT from X's own table,
  not from a related table that happens to have a column with the same name.
- BIDIRECTIONAL RELATIONSHIP TABLES: If a table has two FK columns pointing to
  the same entity type (e.g. atom_id + atom_id2 in a "connected" table), the table
  likely stores EACH relationship TWICE (once per direction). To count relationships
  per entity: JOIN on ONLY ONE FK column (e.g. WHERE atom_id = X), NEVER sum
  counts from both columns. Joining on both sides DOUBLE-COUNTS every relationship.
  Formula for "average bonds per atom": COUNT(bond_id) / COUNT(DISTINCT atom_id)
  using a single JOIN on one FK column.
- TEXT VALUE MATCHING: The question may phrase values differently from the data.
  E.g. "Brazilian Portuguese" in the question may be stored as "Portuguese (Brazil)"
  in the data. When filtering text columns, prefer LIKE '%keyword%' COLLATE NOCASE
  over exact match. If the domain knowledge shows example values, use the data's
  format, not the question's phrasing.
- AVERAGE vs SUM: When the question asks for "average X", check the DOMAIN
  KNOWLEDGE for a formula definition. "Average monthly consumption" may mean
  AVG(consumption_per_record)/12 or SUM(consumption)/COUNT(customers)/12 —
  use the formula from knowledge.md if one is provided.
- RATIO vs COUNT: "How many times was X more than Y" or "How many times greater"
  means RATIO = X / Y (producing a decimal result), NOT a count of occurrences.
  Similarly, "how many times less" = Y / X. Use CAST(X AS REAL) / Y in SQL.
""".strip()


def _strip_json_fence(raw: str) -> str:
    raw = raw.strip()
    fence = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    generic = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return raw


def _validate_grounding(kg: dict[str, Any], db_path: Path | None) -> list[str]:
    """Validate the grounding response against the actual consolidated DB schema."""
    errors: list[str] = []
    if not db_path or not db_path.exists():
        return errors

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        tables_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        valid_tables = {r[0].lower() for r in tables_rows}

        # Collect all valid columns per table
        table_columns: dict[str, set[str]] = {}
        for (tbl,) in tables_rows:
            cols = conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()
            table_columns[tbl.lower()] = {r[1].lower() for r in cols}

        # Validate computation_steps SQL by running them
        for step in kg.get("computation_steps", []):
            sql = step.get("sql", "")
            if not sql:
                continue
            try:
                conn.execute(f"EXPLAIN {sql}")
            except sqlite3.OperationalError as e:
                errors.append(f"Step {step.get('step', '?')} SQL error: {e}")
                continue
            # Run the query and flag empty results on filter queries
            try:
                cursor = conn.execute(sql)
                rows = cursor.fetchall()
                if not rows or (len(rows) == 1 and rows[0][0] in (0, None)):
                    errors.append(
                        f"Step {step.get('step', '?')} returned empty/zero — "
                        f"the filter value may not match actual data format. "
                        f"Check sample values with: SELECT DISTINCT <col> FROM <table> LIMIT 10"
                    )
            except Exception:
                pass

        # Validate join paths reference real tables
        for jp in kg.get("join_paths", []):
            for key in ("from_table", "to_table"):
                tbl = jp.get(key, "").lower()
                if tbl and tbl not in valid_tables:
                    errors.append(f"join_paths references non-existent table '{tbl}'")

        # Validate data requirements
        for req in kg.get("data_requirements", []):
            tbl = req.get("table", "").lower()
            col = req.get("column", "").lower()
            if tbl and tbl not in valid_tables:
                errors.append(f"data_requirements references non-existent table '{tbl}'")
            elif tbl and col and col not in table_columns.get(tbl, set()):
                errors.append(f"data_requirements references non-existent column '{tbl}.{col}'")

        conn.close()
    except Exception:
        pass
    return errors


VALIDATION_RETRY_PROMPT = """
Your previous grounding response had validation errors when checked against the
actual database schema. Fix these errors and produce a corrected response.

ERRORS:
{errors}

ORIGINAL RESPONSE:
{original}

Fix the SQL, table names, and column references to match the actual schema.
Return the full corrected JSON object (same format as before).
""".strip()


def build_knowledge_graph(
    *,
    model: ModelAdapter,
    file_tree: str,
    schema_hint: str,
    knowledge_text: str,
    question: str,
    db_path: Path | None = None,
    max_retries: int = 5,
) -> dict[str, Any]:
    from datetime import datetime as _datetime

    prompt = KNOWLEDGE_GRAPH_PROMPT.format(
        file_tree=file_tree,
        schema_hint=schema_hint or "(no structured data files)",
        knowledge_text=knowledge_text or "(no knowledge.md found)",
        question=question,
        current_date=_datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    messages = [
        ModelMessage(role="system", content="You are a semantic data analyst."),
        ModelMessage(role="user", content=prompt),
    ]

    for attempt in range(1 + max_retries):
        try:
            raw = model.complete(messages)
            parsed = json.loads(_strip_json_fence(raw))
            if not isinstance(parsed, dict):
                return {}

            # Validate against actual DB
            if db_path and db_path.exists():
                errors = _validate_grounding(parsed, db_path)
                if errors and attempt < max_retries:
                    retry_prompt = VALIDATION_RETRY_PROMPT.format(
                        errors="\n".join(f"- {e}" for e in errors),
                        original=json.dumps(parsed, indent=2),
                    )
                    messages = [
                        ModelMessage(role="system", content="You are a semantic data analyst."),
                        ModelMessage(role="user", content=retry_prompt),
                    ]
                    continue
            return parsed
        except (json.JSONDecodeError, ValueError, Exception):
            if attempt == max_retries:
                return {}
    return {}


def render_knowledge_graph(kg: dict[str, Any]) -> str:
    if not kg:
        return ""
    lines: list[str] = []

    # Computation steps — the primary output, agent should execute these
    steps = kg.get("computation_steps", [])
    if steps:
        lines.append("COMPUTATION STEPS (execute these in order):")
        for s in steps:
            step_num = s.get("step", "?")
            desc = s.get("description", "")
            sql = s.get("sql", "")
            tool = s.get("tool", "execute_context_sql")
            path = s.get("path", "_consolidated.db")
            lines.append(f"  Step {step_num}: {desc}")
            if sql:
                lines.append(f"    Tool: {tool} | Path: {path}")
                lines.append(f"    SQL: {sql}")
            code = s.get("code", "")
            if code:
                lines.append(f"    Tool: execute_python")
                lines.append(f"    Code: {code}")
        lines.append("")

    # Join paths
    joins = kg.get("join_paths", [])
    if joins:
        lines.append("JOIN PATHS:")
        for j in joins:
            lines.append(f"  {j.get('from_table', '?')} -> {j.get('to_table', '?')} ON {j.get('on', '?')}")
        lines.append("")

    # Filters
    filters = kg.get("filters", [])
    if filters:
        lines.append("FILTERS:")
        for f in filters:
            lines.append(
                f"  {f.get('table', '?')}.{f.get('column', '?')} "
                f"{f.get('operator', '=')} {f.get('value', '?')}"
            )
        lines.append("")

    # Ambiguous columns
    ambiguous = kg.get("ambiguous_columns", [])
    if ambiguous:
        lines.append("AMBIGUOUS COLUMNS:")
        for a in ambiguous:
            lines.append(
                f"  '{a.get('column', '?')}': "
                f"wrong={a.get('wrong_source', '?')}, "
                f"correct={a.get('correct_source', '?')}"
            )
        lines.append("")

    # Domain rules
    rules = kg.get("domain_rules", [])
    if rules:
        lines.append("DOMAIN RULES:")
        for r in rules:
            lines.append(f"  - {r}")
        lines.append("")

    # Data requirements
    reqs = kg.get("data_requirements", [])
    if reqs:
        lines.append("DATA REQUIREMENTS:")
        for r in reqs:
            lines.append(f"  - {r.get('table', '?')}.{r.get('column', '?')}: {r.get('description', '')}")
        lines.append("")

    # Reasoning
    reasoning = kg.get("reasoning", "")
    if reasoning:
        lines.append(f"REASONING: {reasoning}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SQLite Consolidation
# ---------------------------------------------------------------------------

CONSOLIDATED_DB_NAME = "_consolidated.db"


def _sanitize_table_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if clean and clean[0].isdigit():
        clean = "t_" + clean
    return clean



def _infer_column_types(rows: list[list[str]], col_count: int) -> list[str]:
    """Infer column types from sample rows. Returns 'INTEGER', 'REAL', or 'TEXT'."""
    types: list[str] = ["INTEGER"] * col_count
    for row in rows:
        for i in range(min(len(row), col_count)):
            if types[i] == "TEXT":
                continue
            val = row[i].strip()
            if not val:
                continue
            if types[i] == "INTEGER":
                try:
                    int(val)
                    continue
                except ValueError:
                    pass
                try:
                    float(val)
                    types[i] = "REAL"
                    continue
                except ValueError:
                    types[i] = "TEXT"
            elif types[i] == "REAL":
                try:
                    float(val)
                    continue
                except ValueError:
                    types[i] = "TEXT"
    return types


def _cast_row(row: tuple[str, ...], types: list[str]) -> tuple[Any, ...]:
    """Cast row values to inferred types for proper SQLite storage."""
    result: list[Any] = []
    for i, val in enumerate(row):
        if i >= len(types):
            result.append(val)
            continue
        if not val.strip():
            result.append(None if types[i] != "TEXT" else val)
            continue
        if types[i] == "INTEGER":
            try:
                result.append(int(val))
            except ValueError:
                result.append(val)
        elif types[i] == "REAL":
            try:
                result.append(float(val))
            except ValueError:
                result.append(val)
        else:
            result.append(val)
    return tuple(result)


def _load_csv_into_db(conn: sqlite3.Connection, csv_path: Path, table_name: str) -> bool:
    try:
        with csv_path.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers:
                return False
            # Read sample rows to infer types
            sample_rows: list[list[str]] = []
            all_rows: list[list[str]] = []
            for row in reader:
                all_rows.append(row)
                if len(sample_rows) < 50:
                    sample_rows.append(row)

            col_types = _infer_column_types(sample_rows, len(headers))
            cols_def = ", ".join(
                f'"{h.strip()}" {col_types[i]}' for i, h in enumerate(headers)
            )
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols_def})')
            placeholders = ", ".join(["?"] * len(headers))
            batch: list[tuple[Any, ...]] = []
            for row in all_rows:
                if len(row) == len(headers):
                    batch.append(_cast_row(tuple(row), col_types))
                elif len(row) > len(headers):
                    batch.append(_cast_row(tuple(row[: len(headers)]), col_types))
                else:
                    padded = tuple(row) + ("",) * (len(headers) - len(row))
                    batch.append(_cast_row(padded, col_types))
                if len(batch) >= 1000:
                    conn.executemany(
                        f'INSERT INTO "{table_name}" VALUES ({placeholders})', batch
                    )
                    batch.clear()
            if batch:
                conn.executemany(
                    f'INSERT INTO "{table_name}" VALUES ({placeholders})', batch
                )
        conn.commit()
        return True
    except Exception:
        return False


def _infer_json_column_type(records: list[dict[str, Any]], key: str) -> str:
    """Infer SQLite type from JSON values."""
    seen_int = False
    seen_float = False
    for r in records[:50]:
        val = r.get(key)
        if val is None or val == "":
            continue
        if isinstance(val, bool):
            return "INTEGER"
        if isinstance(val, int):
            seen_int = True
        elif isinstance(val, float):
            seen_float = True
        elif isinstance(val, str):
            return "TEXT"
        else:
            return "TEXT"
    if seen_float:
        return "REAL"
    if seen_int:
        return "INTEGER"
    return "TEXT"


def _load_json_into_db(conn: sqlite3.Connection, json_path: Path, table_name: str) -> bool:
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        records: list[dict[str, Any]] = []
        if isinstance(raw, list):
            records = [r for r in raw if isinstance(r, dict)]
        elif isinstance(raw, dict):
            for key in ("records", "data", "items", "results"):
                candidate = raw.get(key)
                if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
                    records = candidate
                    break
            if not records and all(isinstance(v, (str, int, float, bool, type(None))) for v in raw.values()):
                records = [raw]
        if not records:
            return False

        all_keys: list[str] = []
        seen: set[str] = set()
        for r in records[:100]:
            for k in r:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        col_types = [_infer_json_column_type(records, k) for k in all_keys]
        cols_def = ", ".join(
            f'"{all_keys[i]}" {col_types[i]}' for i in range(len(all_keys))
        )
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols_def})')
        placeholders = ", ".join(["?"] * len(all_keys))
        batch: list[tuple[Any, ...]] = []
        for r in records:
            row_vals: list[Any] = []
            for i, k in enumerate(all_keys):
                val = r.get(k)
                if isinstance(val, (list, dict)):
                    row_vals.append(json.dumps(val, ensure_ascii=False))
                elif val is None:
                    row_vals.append(None)
                elif col_types[i] == "INTEGER" and isinstance(val, (int, float)):
                    row_vals.append(int(val))
                elif col_types[i] == "REAL" and isinstance(val, (int, float)):
                    row_vals.append(float(val))
                else:
                    row_vals.append(str(val) if val is not None else None)
            batch.append(tuple(row_vals))
            if len(batch) >= 1000:
                conn.executemany(
                    f'INSERT INTO "{table_name}" VALUES ({placeholders})', batch
                )
                batch.clear()
        if batch:
            conn.executemany(
                f'INSERT INTO "{table_name}" VALUES ({placeholders})', batch
            )
        conn.commit()
        return True
    except Exception:
        return False


def _attach_sqlite_db(conn: sqlite3.Connection, db_path: Path, alias: str) -> list[str]:
    try:
        conn.execute(f"ATTACH DATABASE ? AS \"{alias}\"", (str(db_path),))
        tables = conn.execute(
            f"SELECT name FROM \"{alias}\".sqlite_master "
            f"WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        table_names = []
        for (tbl,) in tables:
            conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{tbl}" AS SELECT * FROM "{alias}"."{tbl}"'
            )
            table_names.append(tbl)
        conn.execute(f'DETACH DATABASE "{alias}"')
        conn.commit()
        return table_names
    except Exception:
        return []


def consolidate_to_sqlite(context_dir: Path, output_dir: Path | None = None) -> Path | None:
    csv_files = sorted(
        p for p in context_dir.rglob("*.csv") if not p.name.startswith("_")
    )
    json_files = sorted(
        p for p in context_dir.rglob("*.json") if p.name != "task.json"
    )
    db_files = sorted(
        p for p in context_dir.rglob("*.db") if p.name != CONSOLIDATED_DB_NAME
    ) + sorted(context_dir.rglob("*.sqlite"))

    source_count = len(csv_files) + len(json_files) + len(db_files)
    if source_count == 0:
        return None

    # Try output_dir first, then context_dir, then /tmp
    target_dir = output_dir or context_dir
    db_path = target_dir / CONSOLIDATED_DB_NAME
    try:
        if db_path.exists():
            db_path.unlink()
    except OSError:
        pass
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=DELETE")
    except Exception:
        # context_dir is read-only — fall back to /tmp
        import tempfile
        db_path = Path(tempfile.gettempdir()) / f"_consolidated_{context_dir.name}.db"
        if db_path.exists():
            db_path.unlink()
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=DELETE")
        except Exception:
            return None

    loaded_tables: list[str] = []
    try:
        for csv_path in csv_files:
            rel = csv_path.relative_to(context_dir).as_posix()
            table_name = _sanitize_table_name(csv_path.stem)
            if _load_csv_into_db(conn, csv_path, table_name):
                loaded_tables.append(f"{table_name} (from {rel})")

        for json_path in json_files:
            rel = json_path.relative_to(context_dir).as_posix()
            table_name = _sanitize_table_name(json_path.stem)
            if _load_json_into_db(conn, json_path, table_name):
                loaded_tables.append(f"{table_name} (from {rel})")

        for i, existing_db in enumerate(db_files):
            alias = f"attached_{i}"
            tables = _attach_sqlite_db(conn, existing_db, alias)
            for t in tables:
                loaded_tables.append(f"{t} (from {existing_db.name})")

        conn.close()
    except Exception:
        conn.close()
        if db_path.exists():
            db_path.unlink()
        return None

    if not loaded_tables:
        if db_path.exists():
            db_path.unlink()
        return None

    return db_path


def get_consolidated_schema(db_path: Path) -> str:
    lines: list[str] = []
    all_columns: dict[str, list[tuple[str, list[str]]]] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (tbl,) in tables:
            cols = conn.execute(f"PRAGMA table_info(\"{tbl}\")").fetchall()
            col_descs = []
            for row in cols:
                col_name = row[1]
                col_type = row[2] or "TEXT"
                col_descs.append(f"{col_name} {col_type}")
                # Collect sample values for shared column detection
                all_columns.setdefault(col_name, [])
            count = conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
            prefix = ""
            if tbl.startswith("_extracted_"):
                prefix = "[FROM DOC - JOIN with structured tables] "
            lines.append(f"{prefix}TABLE {tbl} ({', '.join(col_descs)}) [{count} rows]")

            # Get sample values for each column to help disambiguate
            for row in cols:
                col_name = row[1]
                try:
                    samples = conn.execute(
                        f'SELECT DISTINCT "{col_name}" FROM "{tbl}" '
                        f'WHERE "{col_name}" IS NOT NULL AND "{col_name}" != "" LIMIT 5'
                    ).fetchall()
                    sample_vals = [str(s[0]) for s in samples]
                    all_columns[col_name].append((tbl, sample_vals))
                except Exception:
                    all_columns[col_name].append((tbl, []))
        conn.close()
    except Exception:
        pass

    shared = {col: entries for col, entries in all_columns.items() if len(entries) > 1}
    if shared:
        lines.append("")
        lines.append("SHARED COLUMN NAMES (DIFFERENT meanings — always use the correct table):")
        for col, entries in sorted(shared.items()):
            table_names = [t for t, _ in entries]
            lines.append(f"  '{col}' in: {', '.join(table_names)}")
            for tbl, samples in entries:
                if samples:
                    lines.append(f"    {tbl}.{col} samples: {samples[:4]}")
        lines.append(
            "  IMPORTANT: When the question asks for a property of an entity, "
            "SELECT it from that entity's own table (e.g. a driver's number comes "
            "from the drivers table, not from qualifying)."
        )

    return "\n".join(lines)
