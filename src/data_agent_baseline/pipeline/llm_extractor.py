"""LLM-based document extractor for tasks where regex extraction is insufficient.

Triggered when:
  - task_type == "doc_only" (no structured data at all)
  - OR regex extraction misses columns the question asks about

Pipeline:
  1. Schema discovery: sample doc text + question → column definitions
  2. Chunk doc by record boundaries
  3. Extract per chunk (1 LLM call each)
  4. Merge all records → write to SQLite
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


CHUNK_SIZE = 18000  # chars per LLM call

SCHEMA_PROMPT = """Analyze this document excerpt and determine what structured data can be extracted.

DOCUMENT EXCERPT:
{doc_sample}

QUESTION (what we need to answer):
{question}

Based on the document structure, identify ALL columns/fields that appear for each record.
Each record is identified by a numeric ID (patient number, case ID, etc.).

Return ONLY a JSON object:
{{
  "id_field": "the field name for the record identifier",
  "columns": ["col1", "col2", "col3"],
  "id_pattern": "regex pattern to find record IDs in the text"
}}

The columns should include ALL data fields you can see (dates, measurements, categories, etc.), not just what the question asks about. Use short lowercase names with underscores.
""".strip()

EXTRACT_PROMPT = """Extract structured records from this document chunk.

COLUMNS TO EXTRACT: {columns}
ID FIELD: {id_field}

DOCUMENT CHUNK:
{chunk}

Extract every record you find. Each record starts with an ID ({id_field}).
Return ONLY a JSON array of objects, one per record:
[
  {{"{id_field}": "12345", "col1": "value1", "col2": "value2"}},
  ...
]

RULES:
- Extract exact values as they appear (numbers, dates, text).
- If a field is not mentioned for a record, omit it (don't include null).
- For dates, preserve the original format.
- Return raw numeric values without units.
- Include ALL records in the chunk — don't skip any.
""".strip()


@dataclass
class LLMExtractedTable:
    name: str
    id_field: str
    columns: list[str]
    records: list[dict[str, Any]]


def should_use_llm_extraction(
    task_type: str,
    question: str,
    regex_tables: list[Any],
    db_path: Path | None = None,
) -> bool:
    """Decide whether LLM extraction is needed.

    Triggers only when task_type is "doc_only" — meaning there's no structured
    data (CSV/JSON/DB) at all, and we must extract everything from prose docs.

    For "mixed" tasks, the structured data already provides queryable tables and
    the regex extraction supplements it — LLM extraction adds cost without benefit.
    """
    return task_type == "doc_only"


def discover_schema(
    model: ModelAdapter,
    doc_paths: list[Path],
    question: str,
) -> dict[str, Any]:
    """Use LLM to discover the schema from a document sample."""
    # Take first ~4000 chars from each doc
    sample_parts = []
    for path in doc_paths[:3]:
        text = path.read_text(errors="replace")
        sample_parts.append(f"--- {path.name} ---\n{text[:4000]}")
    doc_sample = "\n\n".join(sample_parts)

    prompt = SCHEMA_PROMPT.format(doc_sample=doc_sample, question=question)
    messages = [ModelMessage(role="user", content=prompt)]
    raw = model.complete(messages)

    return _parse_json(raw) or {"id_field": "_id", "columns": [], "id_pattern": r"\d{4,}"}


def extract_from_docs(
    model: ModelAdapter,
    doc_paths: list[Path],
    schema: dict[str, Any],
) -> list[LLMExtractedTable]:
    """Extract structured records from documents using LLM."""
    id_field = schema.get("id_field", "_id")
    columns = schema.get("columns", [])
    id_pattern = schema.get("id_pattern", r"\d{4,}")

    if not columns:
        return []

    all_records: dict[str, dict[str, Any]] = {}  # id → merged record

    for doc_path in doc_paths:
        text = doc_path.read_text(errors="replace")
        chunks = _chunk_by_records(text, id_pattern, CHUNK_SIZE)

        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue

            prompt = EXTRACT_PROMPT.format(
                columns=json.dumps(columns),
                id_field=id_field,
                chunk=chunk,
            )
            messages = [ModelMessage(role="user", content=prompt)]

            try:
                raw = model.complete(messages)
            except RuntimeError:
                continue

            parsed = _parse_json(raw)
            if isinstance(parsed, list):
                for record in parsed:
                    if not isinstance(record, dict):
                        continue
                    rid = str(record.get(id_field, ""))
                    if not rid:
                        continue
                    if rid not in all_records:
                        all_records[rid] = {id_field: rid}
                    # Merge: later values overwrite earlier ones
                    for k, v in record.items():
                        if v is not None and v != "":
                            all_records[rid][k] = v

    if not all_records:
        return []

    # Build table per source doc (or one combined)
    records_list = list(all_records.values())
    table_name = doc_paths[0].stem if len(doc_paths) == 1 else "extracted"

    return [LLMExtractedTable(
        name=table_name,
        id_field=id_field,
        columns=[id_field] + columns,
        records=records_list,
    )]


def write_llm_extracted_table(db_path: Path, table: LLMExtractedTable) -> None:
    """Write LLM-extracted table to SQLite, replacing any existing regex version."""
    conn = sqlite3.connect(str(db_path))

    # Drop existing table if regex already created one
    conn.execute(f'DROP TABLE IF EXISTS "{table.name}"')

    # Determine all columns from records
    all_cols = list(table.columns)
    for rec in table.records:
        for k in rec.keys():
            if k not in all_cols:
                all_cols.append(k)

    col_defs = ", ".join(f'"{c}" TEXT' for c in all_cols)
    conn.execute(f'CREATE TABLE "{table.name}" ({col_defs})')

    placeholders = ", ".join(["?"] * len(all_cols))
    for rec in table.records:
        values = [str(rec.get(c, "")) if rec.get(c) is not None else None for c in all_cols]
        conn.execute(f'INSERT INTO "{table.name}" VALUES ({placeholders})', values)

    conn.commit()
    conn.close()


def _chunk_by_records(text: str, id_pattern: str, max_chars: int) -> list[str]:
    """Split text into chunks at record boundaries."""
    try:
        pattern = re.compile(id_pattern)
    except re.error:
        pattern = re.compile(r"\d{4,}")

    # Find all record start positions
    starts = [m.start() for m in pattern.finditer(text)]

    if not starts:
        # No records found — just split by size
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    chunks = []
    current_start = 0

    for i, pos in enumerate(starts):
        # Check if adding the next record would exceed chunk size
        if pos - current_start > max_chars and current_start < pos:
            chunks.append(text[current_start:pos])
            current_start = pos

    # Don't forget the last chunk
    if current_start < len(text):
        chunks.append(text[current_start:])

    return chunks


def _parse_json(raw: str) -> Any:
    """Parse JSON from LLM response."""
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    for start, end in [("{", "}"), ("[", "]")]:
        idx = raw.find(start)
        if idx >= 0:
            depth = 0
            for i in range(idx, len(raw)):
                if raw[i] == start:
                    depth += 1
                elif raw[i] == end:
                    depth -= 1
                    if depth == 0:
                        candidate = raw[idx:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                            try:
                                return json.loads(fixed)
                            except json.JSONDecodeError:
                                break
            break
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
