"""Hybrid document extraction: structure-aware parsing + LLM extraction.

Pipeline:
  1. Parse markdown structure (deterministic, fast)
  2. Discover schema from knowledge.md (deterministic)
  3. Batch sections into chunks (~8000 chars each)
  4. LLM extracts records per chunk (schema-guided)
  5. Merge records by ID+Date → write to SQLite
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.pipeline.field_discoverer import DocumentSchema, discover_schema
from data_agent_baseline.pipeline.md_parser import Section, filter_data_sections, parse_md_structure
from data_agent_baseline.pipeline.sqlite_writer import _sanitize_column_name


EXTRACT_PROMPT = """Extract ALL records from this text into a JSON array.

ENTITY: {entity_name}
ID FIELD: {id_field}
FIELDS: {field_list}

RULES:
1. Each object: {{"{id_field}": "...", "Date": "YYYY-MM-DD", ...only fields present...}}
2. CORRECTIONS: initially X, corrected/adjusted to Y → use ONLY Y.
3. NaN/unavailable → null. Not mentioned → omit.
4. Numbers only, no units.
5. If text explicitly describes a field as elevated/impaired/compromised/dysfunction → add "<field>_status": "abnormal". If normal/healthy/unremarkable → "normal". Only for explicitly described fields.
6. Extract EVERY record.

TEXT:
{text_chunk}

JSON array only:"""


def _build_field_list(schema: DocumentSchema) -> str:
    parts = []
    for f in schema.fields:
        desc = f.name
        if f.aliases:
            desc += f" (aka {', '.join(f.aliases[:3])})"
        parts.append(desc)
    return ", ".join(parts)


def _detect_section_fields(text: str, schema: DocumentSchema) -> str:
    """Detect which schema fields are actually mentioned in this section.

    Returns a comma-separated field list for the prompt. Only includes fields
    whose name or alias appears in the text, keeping output compact.
    """
    text_lower = text.lower()
    found: list[str] = []
    for f in schema.fields:
        names_to_check = [f.name.lower()] + [a.lower() for a in (f.aliases or [])]
        for name in names_to_check:
            if name in text_lower:
                desc = f.name
                if f.aliases:
                    desc += f" (aka {', '.join(f.aliases[:3])})"
                found.append(desc)
                break
    return ", ".join(found) if found else ""


def _parse_llm_json(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    bracket_start = raw.find("[")
    bracket_end = raw.rfind("]")
    if bracket_start == -1 or bracket_end == -1:
        return []
    json_str = raw[bracket_start:bracket_end + 1]
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            data = json.loads(fixed)
            if isinstance(data, list):
                return [r for r in data if isinstance(r, dict)]
        except json.JSONDecodeError:
            pass
    return []


def _chunk_sections(sections: list[Section], max_chars: int = 15000) -> list[str]:
    """Pack data-bearing paragraphs into chunks of ~max_chars."""
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for section in sections:
        for para in section.paragraphs:
            if len(para) < 30:
                continue
            if buffer and buffer_len + len(para) > max_chars:
                chunks.append("\n\n".join(buffer))
                buffer = []
                buffer_len = 0
            buffer.append(para)
            buffer_len += len(para)

    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def _chunk_by_section(sections: list[Section], max_chars: int = 30000) -> list[tuple[str, str]]:
    """Chunk by section groups, returning (section_heading, text) pairs.

    Groups consecutive sections into chunks up to max_chars.
    Returns the heading of the first section in each group for field detection.
    """
    chunks: list[tuple[str, str]] = []
    buffer: list[str] = []
    buffer_len = 0
    heading = ""

    for section in sections:
        section_text = "\n\n".join(p for p in section.paragraphs if len(p) >= 30)
        if not section_text:
            continue

        if buffer and buffer_len + len(section_text) > max_chars:
            chunks.append((heading, "\n\n".join(buffer)))
            buffer = []
            buffer_len = 0
            heading = ""

        if not heading:
            heading = section.heading
        buffer.append(section_text)
        buffer_len += len(section_text)

    if buffer:
        chunks.append((heading, "\n\n".join(buffer)))
    return chunks


def hybrid_extract_docs(
    doc_paths: list[Path],
    db_path: Path,
    model: ModelAdapter,
    knowledge_text: str = "",
    time_remaining_fn: Callable[[], float] = lambda: 300.0,
    log_fn: Callable[[str, str], None] | None = None,
    structured_tables: list[str] | None = None,
) -> int:
    """Hybrid extraction pipeline: structure parsing + LLM extraction.

    Drop-in replacement for deterministic_extract_docs().
    """
    if not doc_paths:
        return 0

    protected = {t.lower() for t in structured_tables} if structured_tables else set()
    total_records = 0

    for doc_path in doc_paths:
        if time_remaining_fn() < 60:
            break

        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) < 100:
            continue

        # Stage 1: Parse structure (deterministic)
        sections = parse_md_structure(text)
        data_sections = filter_data_sections(sections)
        if not data_sections:
            if log_fn:
                log_fn("extract_skip", f"{doc_path.stem}: no data-dense sections")
            continue

        # Stage 2: Discover schema (deterministic)
        schema = discover_schema(text, doc_path.stem, knowledge_text)
        if log_fn:
            log_fn("schema", f"{doc_path.stem}: {len(schema.fields)} fields, "
                   f"entity={schema.entity_name}")

        # Stage 3: Chunk by section groups
        section_chunks = _chunk_by_section(data_sections, max_chars=30000)
        if log_fn:
            log_fn("chunks", f"{doc_path.stem}: {len(section_chunks)} section chunks")

        # Stage 4: LLM extraction per section chunk (sequential)
        id_field = schema.id_field or "ID"
        all_records: list[dict[str, Any]] = []

        for i, (heading, chunk_text) in enumerate(section_chunks):
            if time_remaining_fn() < 40:
                if log_fn:
                    log_fn("time_bail", f"Stopping at chunk {i}/{len(section_chunks)}")
                break

            # Detect which fields are mentioned in this section
            section_fields = _detect_section_fields(chunk_text, schema)
            if not section_fields:
                section_fields = _build_field_list(schema)

            prompt = EXTRACT_PROMPT.format(
                entity_name=schema.entity_name,
                id_field=id_field,
                field_list=section_fields,
                text_chunk=chunk_text,
            )
            messages = [ModelMessage(role="user", content=prompt)]

            try:
                response = model.complete(messages, thinking=False)
            except Exception as e:
                if log_fn:
                    log_fn("llm_error", f"Chunk {i}: {str(e)[:100]}")
                continue

            records = _parse_llm_json(response)
            if log_fn:
                log_fn("chunk_done", f"Chunk {i}/{len(section_chunks)}: "
                       f"{len(records)} records ({len(chunk_text)} chars)")
            all_records.extend(records)

        if log_fn:
            log_fn("extracted", f"{doc_path.stem}: {len(all_records)} raw records")

        if not all_records:
            continue

        # Stage 5: Merge records by ID+Date and write to SQLite
        n_written = _merge_and_write(
            all_records, schema, db_path, protected, id_field, log_fn
        )
        total_records += n_written

    if log_fn:
        log_fn("pipeline_done", f"Total: {total_records} records from {len(doc_paths)} docs")

    return total_records


def _merge_and_write(
    records: list[dict[str, Any]],
    schema: DocumentSchema,
    db_path: Path,
    protected: set[str],
    id_field: str,
    log_fn: Callable[[str, str], None] | None = None,
) -> int:
    """Merge extracted records by composite key and write to SQLite."""
    merged: dict[str, dict[str, Any]] = {}

    for rec in records:
        rid = str(rec.get(id_field, rec.get("ID", rec.get("id", ""))))
        if not rid:
            continue
        date_val = rec.get("Date", rec.get("date", ""))
        if date_val:
            key = f"{rid}_{date_val}"
        else:
            key = rid

        if key not in merged:
            merged[key] = {"record_id": key, id_field.lower(): rid}
        for k, v in rec.items():
            if k in (id_field, "ID", "id"):
                continue
            col = _sanitize_column_name(k)
            if v is not None:
                merged[key][col] = v

    if not merged:
        return 0

    table_name = _sanitize_column_name(schema.entity_name)
    if protected and table_name in protected:
        table_name = f"{table_name}_doc"

    all_cols: set[str] = set()
    for rec in merged.values():
        all_cols.update(rec.keys())
    all_cols.discard("record_id")
    sorted_cols = sorted(all_cols)

    col_types: dict[str, str] = {}
    for col in sorted_cols:
        is_numeric = True
        for rec in merged.values():
            val = rec.get(col)
            if val is None:
                continue
            try:
                float(val)
            except (ValueError, TypeError):
                is_numeric = False
                break
        col_types[col] = "REAL" if is_numeric else "TEXT"

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        col_defs = ["record_id TEXT PRIMARY KEY"]
        for col in sorted_cols:
            col_defs.append(f"{col} {col_types.get(col, 'TEXT')}")
        create_sql = f"CREATE TABLE {table_name} ({', '.join(col_defs)})"
        conn.execute(create_sql)

        insert_cols = ["record_id"] + sorted_cols
        placeholders = ", ".join(["?"] * len(insert_cols))
        insert_sql = (
            f"INSERT OR REPLACE INTO {table_name} "
            f"({', '.join(insert_cols)}) VALUES ({placeholders})"
        )

        for rec in merged.values():
            row = [rec.get("record_id")]
            for col in sorted_cols:
                val = rec.get(col)
                if val is not None and col_types.get(col) == "REAL":
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
                row.append(val)
            conn.execute(insert_sql, row)

        conn.commit()
        if log_fn:
            log_fn("sqlite", f"Wrote {len(merged)} records to '{table_name}'")
    finally:
        conn.close()

    return len(merged)
