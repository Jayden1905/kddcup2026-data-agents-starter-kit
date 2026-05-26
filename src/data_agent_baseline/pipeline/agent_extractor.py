"""LLM Agent-based document extraction.

Full agent loop:
  1. LLM reads knowledge.md + existing DB schema + doc sample → proposes table schema
  2. LLM extracts records per chunk using discovered schema
  3. Validates extraction (row counts, key fields present)
  4. Retries failed chunks with adjusted prompts
  5. Merges records by ID → writes to SQLite
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.pipeline.md_parser import parse_md_structure, filter_data_sections


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SCHEMA_DISCOVERY_PROMPT = """You are a data extraction agent. Analyze this document and determine the table schema to extract.

DOCUMENT NAME: {doc_name}
DOCUMENT SAMPLE (first 3000 chars):
{doc_sample}

{knowledge_section}
{db_context}
TASK: Define the extraction schema for this document. The extracted table must integrate with the existing database above.

Return a JSON object:
{{
  "table_name": "snake_case name for the table",
  "id_field": "the field that uniquely identifies each record (look for IDs, codes, registry numbers in the text)",
  "fields": [
    {{"name": "field_name", "type": "TEXT|INTEGER|REAL", "description": "what this field contains"}}
  ]
}}

RULES:
- The id_field MUST match identifiers actually used in the document text (e.g. "TR001", registry number 72)
- Include FK fields that JOIN with existing tables (look at column names and grain columns above)
- Include ALL numeric attributes mentioned in the document (heights, weights, counts, scores, amounts, percentages)
- Include ALL categorical fields (status, type, category, label, alignment)
- Use snake_case for all field names
- Only include fields actually described in the document, not invented ones
- The table should complement the existing database — fill the gap that structured files don't cover

JSON only:"""

EXTRACT_CHUNK_PROMPT = """Extract ALL records from this text into a JSON array.

TABLE: {table_name}
SCHEMA:
{schema_text}
{memory_section}
RULES:
1. Each record MUST have "{id_field}" — the unique identifier from the text.
2. Extract ALL fields defined in the schema for EVERY record. A record with ONLY the ID field is INVALID — the text surrounding each ID contains the field values. Use null ONLY if a field is genuinely absent.
3. CORRECTIONS: if text says "initially X, corrected to Y" → use ONLY Y.
4. Numbers: plain numbers, no units (the field name indicates the unit).
5. If multiple records appear in the text, extract ALL of them.
6. Same entity mentioned in multiple paragraphs → merge into ONE record.

TEXT:
{text_chunk}

Return JSON array only:"""

RETRY_PROMPT = """The previous extraction returned {issue}. Try again more carefully.

TABLE: {table_name}
SCHEMA:
{schema_text}

Focus on finding the "{id_field}" for each record. Look for patterns like:
- "identifier N", "registry number N", "ID N", "code XYZ", "designated ABC"
- Compound codes like "TR001", "rec123ABC"

TEXT:
{text_chunk}

Return JSON array only:"""

RETRY_FIELDS_PROMPT = """Your previous extraction found the IDs but MISSED the field values. \
Each record had only "{id_field}" with no other fields populated. This is wrong.

TABLE: {table_name}
SCHEMA:
{schema_text}

The text describes entities with attributes. For EACH entity:
- Find its "{id_field}" (the unique identifier)
- Read the surrounding text to extract ALL other schema fields

Every field value is stated in the text near the ID. Do NOT return records with only the ID.

TEXT:
{text_chunk}

Return JSON array with ALL fields populated:"""


# ---------------------------------------------------------------------------
# Extraction Memory — accumulates context across chunks
# ---------------------------------------------------------------------------

class ExtractionMemory:
    """Tracks entities, patterns, and corrections across extraction chunks.

    Renders a compact summary injected into later chunk prompts so the LLM
    knows what was already extracted and can maintain consistency.
    """

    def __init__(self, max_ids: int = 50, max_patterns: int = 5) -> None:
        self.ids_found: list[str] = []
        self.field_patterns: dict[str, str] = {}  # field → observed format
        self.corrections: list[str] = []  # "ID: old → new"
        self._max_ids = max_ids
        self._max_patterns = max_patterns

    def ingest_records(self, records: list[dict[str, Any]], id_field: str) -> None:
        """Learn from extracted records."""
        for rec in records:
            rid = str(rec.get(id_field, ""))
            if rid and rid not in self.ids_found:
                self.ids_found.append(rid)

            # Detect value patterns per field (first non-null wins)
            for k, v in rec.items():
                if k == id_field or v is None or v == "":
                    continue
                if k not in self.field_patterns and len(self.field_patterns) < self._max_patterns:
                    sample = str(v)[:30]
                    if isinstance(v, (int, float)):
                        self.field_patterns[k] = f"numeric (e.g. {sample})"
                    elif len(sample) > 5:
                        self.field_patterns[k] = f"text (e.g. \"{sample}\")"

    def render(self) -> str:
        """Render memory as a prompt section for subsequent chunks."""
        if not self.ids_found and not self.field_patterns:
            return ""
        lines: list[str] = []
        lines.append("EXTRACTION CONTEXT (from prior chunks):")
        if self.ids_found:
            shown = self.ids_found[-self._max_ids:]
            lines.append(f"  IDs already extracted ({len(self.ids_found)} total): "
                         f"{', '.join(shown[:20])}" +
                         (f"... +{len(shown)-20} more" if len(shown) > 20 else ""))
            lines.append("  → Skip these if re-mentioned (already captured).")
            lines.append("  → If new info appears for an existing ID, still extract it (will merge).")
        if self.field_patterns:
            lines.append("  Value patterns observed:")
            for field, pattern in self.field_patterns.items():
                lines.append(f"    {field}: {pattern}")
        if self.corrections:
            lines.append("  Corrections found:")
            for c in self.corrections[-3:]:
                lines.append(f"    {c}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Agent Extractor
# ---------------------------------------------------------------------------

def agent_extract_docs(
    doc_paths: list[Path],
    db_path: Path,
    model: ModelAdapter,
    knowledge_text: str = "",
    time_remaining_fn: Callable[[], float] = lambda: 300.0,
    log_fn: Callable[[str, str], None] | None = None,
    structured_tables: list[str] | None = None,
    kg_context: str = "",
) -> int:
    """Full LLM agent loop for doc extraction."""
    if not doc_paths:
        return 0

    protected = {t.lower() for t in structured_tables} if structured_tables else set()
    total_records = 0

    if log_fn:
        log_fn("doc_extract_start",
               f"{len(doc_paths)} docs, db={db_path.name}, "
               f"protected={list(protected)[:5]}, "
               f"kg_context={len(kg_context)} chars, "
               f"time_remaining={time_remaining_fn():.0f}s")

    for doc_idx, doc_path in enumerate(doc_paths):
        time_left = time_remaining_fn()
        if time_left < 60:
            if log_fn:
                log_fn("doc_skip_time",
                       f"{doc_path.stem}: skipped ({time_left:.0f}s remaining < 60s)")
            break

        if log_fn:
            log_fn("doc_start",
                   f"[{doc_idx+1}/{len(doc_paths)}] {doc_path.stem} "
                   f"({time_left:.0f}s remaining)")

        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            if log_fn:
                log_fn("doc_read_error", f"{doc_path.stem}: {e}")
            continue

        if len(text) < 100:
            if log_fn:
                log_fn("doc_skip_short", f"{doc_path.stem}: only {len(text)} chars")
            continue

        if log_fn:
            log_fn("doc_loaded", f"{doc_path.stem}: {len(text)} chars")

        # Step 1: Schema Discovery (1 LLM call)
        if log_fn:
            log_fn("schema_discovery_start",
                   f"{doc_path.stem}: sending {min(len(text), 3000)} char sample to LLM")

        schema = _discover_schema_with_llm(
            doc_path, text, knowledge_text, db_path, model, log_fn, kg_context
        )
        if not schema:
            if log_fn:
                log_fn("schema_discovery_failed", f"{doc_path.stem}: no valid schema returned")
            continue

        table_name = schema["table_name"]
        id_field = schema["id_field"]
        fields = schema["fields"]

        if log_fn:
            field_names = [f.get("name", "?") for f in fields]
            log_fn("schema_discovered",
                   f"{doc_path.stem}: table={table_name}, id={id_field}, "
                   f"fields={field_names}")

        # Step 2: Chunk the document
        sections = parse_md_structure(text)
        data_sections = filter_data_sections(sections)
        if not data_sections:
            data_sections = sections
            if log_fn:
                log_fn("chunk_filter",
                       f"{doc_path.stem}: no data sections found, using all "
                       f"{len(sections)} sections")

        chunks = _chunk_paragraphs(data_sections, max_chars=20000)
        if log_fn:
            chunk_sizes = [len(c) for c in chunks]
            log_fn("chunks_created",
                   f"{doc_path.stem}: {len(chunks)} chunks, "
                   f"sizes={chunk_sizes}")

        # Step 3: Extract per chunk, write immediately (preserves partial progress)
        schema_text = _format_schema(fields, id_field)
        memory = ExtractionMemory()
        doc_records = 0

        for i, chunk_text in enumerate(chunks):
            time_left = time_remaining_fn()
            if time_left < 40:
                if log_fn:
                    log_fn("chunk_bail",
                           f"{doc_path.stem}: stopping at chunk {i}/{len(chunks)} "
                           f"({time_left:.0f}s remaining < 40s)")
                break

            if log_fn:
                mem_ids = len(memory.ids_found)
                log_fn("chunk_start",
                       f"{doc_path.stem} chunk {i+1}/{len(chunks)}: "
                       f"{len(chunk_text)} chars, memory has {mem_ids} IDs")

            records = _extract_chunk(
                chunk_text, table_name, id_field, schema_text, model, log_fn,
                i, len(chunks), memory,
            )

            # Retry if 0 records extracted from a non-trivial chunk
            if not records and len(chunk_text) > 500 and time_remaining_fn() > 40:
                if log_fn:
                    log_fn("chunk_retry_start",
                           f"{doc_path.stem} chunk {i+1}: 0 records from "
                           f"{len(chunk_text)} chars, retrying")
                records = _retry_chunk(
                    chunk_text, table_name, id_field, schema_text, model, log_fn, i
                )

            # Retry if records have only the ID field (missing all other values)
            if records and time_remaining_fn() > 40:
                id_only_count = sum(
                    1 for r in records
                    if all(
                        k == id_field or v is None or v == "" or v == "null"
                        for k, v in r.items()
                    )
                )
                if id_only_count > len(records) * 0.7:
                    if log_fn:
                        log_fn("chunk_retry_start",
                               f"{doc_path.stem} chunk {i+1}: {id_only_count}/{len(records)} "
                               f"records have only ID, retrying with fields prompt")
                    records = _retry_fields_chunk(
                        chunk_text, table_name, id_field, schema_text, model, log_fn, i
                    )

            if records:
                memory.ingest_records(records, id_field)
                if log_fn:
                    sample_ids = [str(r.get(id_field, "?")) for r in records[:5]]
                    log_fn("chunk_write_start",
                           f"{doc_path.stem} chunk {i+1}: writing {len(records)} records "
                           f"(IDs: {sample_ids})")

                n_written = _merge_and_write(
                    records, table_name, id_field, db_path, protected, log_fn
                )
                doc_records += n_written

                if log_fn:
                    log_fn("chunk_write_done",
                           f"{doc_path.stem} chunk {i+1}: "
                           f"{n_written}/{len(records)} written to DB "
                           f"(total so far: {doc_records})")
            else:
                if log_fn:
                    log_fn("chunk_empty",
                           f"{doc_path.stem} chunk {i+1}: 0 records after all attempts")

        if log_fn:
            log_fn("doc_done",
                   f"{doc_path.stem}: {doc_records} records written to "
                   f"table '{table_name}', memory tracked {len(memory.ids_found)} IDs")

        total_records += doc_records

    if log_fn:
        log_fn("doc_extract_done",
               f"Pipeline complete: {total_records} total records from "
               f"{len(doc_paths)} docs")

    return total_records


# ---------------------------------------------------------------------------
# Step 1: Schema Discovery
# ---------------------------------------------------------------------------

def _discover_schema_with_llm(
    doc_path: Path,
    text: str,
    knowledge_text: str,
    db_path: Path,
    model: ModelAdapter,
    log_fn: Callable[[str, str], None] | None = None,
    kg_context: str = "",
) -> dict[str, Any] | None:
    """Use LLM to discover the table schema from doc + context."""
    doc_sample = text[:3000]

    # Build knowledge section
    knowledge_section = ""
    if knowledge_text:
        entity = doc_path.stem.lower()
        relevant_lines = _extract_relevant_knowledge(knowledge_text, entity)
        if relevant_lines:
            knowledge_section = f"KNOWLEDGE.MD (relevant section):\n{relevant_lines}\n"
            if log_fn:
                log_fn("schema_knowledge",
                       f"{doc_path.stem}: found {len(relevant_lines)} chars of "
                       f"relevant knowledge for entity '{entity}'")

    # Use KG context if available, otherwise fall back to raw DB pragma
    db_context = ""
    if kg_context:
        db_context = kg_context
        if log_fn:
            log_fn("schema_context", f"{doc_path.stem}: using KG context ({len(kg_context)} chars)")
    else:
        try:
            conn = sqlite3.connect(str(db_path))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if tables:
                db_lines = ["EXISTING DB TABLES (your extracted table should JOIN with these):"]
                for t in tables:
                    cols = [f"{r[1]} ({r[2]})" for r in
                            conn.execute(f'PRAGMA table_info("{t}")').fetchall()]
                    db_lines.append(f"  {t}: {', '.join(cols)}")
                db_context = "\n".join(db_lines) + "\n"
                if log_fn:
                    log_fn("schema_context",
                           f"{doc_path.stem}: using DB pragma ({len(tables)} tables)")
            conn.close()
        except Exception:
            pass

    prompt = SCHEMA_DISCOVERY_PROMPT.format(
        doc_name=doc_path.stem,
        doc_sample=doc_sample,
        knowledge_section=knowledge_section,
        db_context=db_context,
    )

    if log_fn:
        log_fn("schema_llm_call",
               f"{doc_path.stem}: prompt={len(prompt)} chars")

    try:
        response = model.complete([ModelMessage(role="user", content=prompt)], thinking=False)
        if log_fn:
            log_fn("schema_llm_response",
                   f"{doc_path.stem}: response={len(response)} chars")
        schema = _parse_json_object(response)
        if schema and "table_name" in schema and "fields" in schema:
            if "id_field" not in schema:
                schema["id_field"] = "id"
            if log_fn:
                log_fn("schema_parsed",
                       f"{doc_path.stem}: table={schema['table_name']}, "
                       f"id={schema['id_field']}, "
                       f"{len(schema['fields'])} fields")
            return schema
        else:
            if log_fn:
                log_fn("schema_parse_failed",
                       f"{doc_path.stem}: JSON parsed but missing required keys "
                       f"(got keys: {list(schema.keys()) if schema else 'None'})")
    except Exception as e:
        if log_fn:
            log_fn("schema_error", f"{doc_path.stem}: {str(e)[:200]}")

    return None


def _extract_relevant_knowledge(knowledge_text: str, entity: str) -> str:
    """Extract the most relevant section of knowledge.md for this entity."""
    lines = knowledge_text.split("\n")
    result_lines: list[str] = []
    in_section = False
    section_depth = 0

    for line in lines:
        stripped = line.strip().lower()

        # Check if this line starts a section about our entity
        if entity in stripped and (stripped.startswith("#") or stripped.startswith("- **")):
            in_section = True
            section_depth = 0
            result_lines.append(line)
            continue

        if in_section:
            # Stop at next major heading that's not about our entity
            if stripped.startswith("##") and entity not in stripped:
                section_depth += 1
                if section_depth > 1:
                    break
            result_lines.append(line)

            # Limit to ~1500 chars
            if sum(len(rl) for rl in result_lines) > 1500:
                break

    return "\n".join(result_lines) if result_lines else ""


# ---------------------------------------------------------------------------
# Step 2-3: Chunk and Extract
# ---------------------------------------------------------------------------

def _chunk_paragraphs(sections: list, max_chars: int = 20000) -> list[str]:
    """Split sections into text chunks of ~max_chars."""
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


def _format_schema(fields: list[dict], id_field: str) -> str:
    """Format schema fields for the extraction prompt."""
    lines = [f"  - {id_field} (required, unique identifier)"]
    for f in fields:
        name = f.get("name", "")
        if name == id_field:
            continue
        ftype = f.get("type", "TEXT")
        desc = f.get("description", "")
        lines.append(f"  - {name} ({ftype}): {desc}")
    return "\n".join(lines)


def _extract_chunk(
    chunk_text: str,
    table_name: str,
    id_field: str,
    schema_text: str,
    model: ModelAdapter,
    log_fn: Callable[[str, str], None] | None,
    chunk_idx: int,
    total_chunks: int,
    memory: ExtractionMemory | None = None,
) -> list[dict[str, Any]]:
    """Extract records from a single chunk."""
    memory_section = memory.render() if memory else ""
    prompt = EXTRACT_CHUNK_PROMPT.format(
        table_name=table_name,
        id_field=id_field,
        schema_text=schema_text,
        text_chunk=chunk_text,
        memory_section=memory_section,
    )

    try:
        response = model.complete([ModelMessage(role="user", content=prompt)], thinking=False)
        records = _parse_json_array(response)
        if log_fn:
            log_fn("chunk_done", f"Chunk {chunk_idx}/{total_chunks}: "
                   f"{len(records)} records ({len(chunk_text)} chars)")
        return records
    except Exception as e:
        if log_fn:
            log_fn("chunk_error", f"Chunk {chunk_idx}: {str(e)[:100]}")
        return []


def _retry_chunk(
    chunk_text: str,
    table_name: str,
    id_field: str,
    schema_text: str,
    model: ModelAdapter,
    log_fn: Callable[[str, str], None] | None,
    chunk_idx: int,
) -> list[dict[str, Any]]:
    """Retry extraction with a more specific prompt."""
    prompt = RETRY_PROMPT.format(
        issue="0 records (no data found)",
        table_name=table_name,
        id_field=id_field,
        schema_text=schema_text,
        text_chunk=chunk_text[:15000],  # Trim for retry
    )

    try:
        response = model.complete([ModelMessage(role="user", content=prompt)], thinking=False)
        records = _parse_json_array(response)
        if log_fn:
            log_fn("retry_done", f"Chunk {chunk_idx} retry: {len(records)} records")
        return records
    except Exception as e:
        if log_fn:
            log_fn("retry_error", f"Chunk {chunk_idx}: {str(e)[:100]}")
        return []


def _retry_fields_chunk(
    chunk_text: str,
    table_name: str,
    id_field: str,
    schema_text: str,
    model: ModelAdapter,
    log_fn: Callable[[str, str], None] | None,
    chunk_idx: int,
) -> list[dict[str, Any]]:
    """Retry extraction when records had only IDs but no field values."""
    prompt = RETRY_FIELDS_PROMPT.format(
        table_name=table_name,
        id_field=id_field,
        schema_text=schema_text,
        text_chunk=chunk_text[:15000],
    )

    try:
        response = model.complete([ModelMessage(role="user", content=prompt)], thinking=False)
        records = _parse_json_array(response)
        if log_fn:
            log_fn("retry_done", f"Chunk {chunk_idx} retry: {len(records)} records")
        return records
    except Exception as e:
        if log_fn:
            log_fn("retry_error", f"Chunk {chunk_idx}: {str(e)[:100]}")
        return []


# ---------------------------------------------------------------------------
# Step 4: Merge and Write
# ---------------------------------------------------------------------------

def _merge_and_write(
    records: list[dict[str, Any]],
    table_name: str,
    id_field: str,
    db_path: Path,
    protected: set[str],
    log_fn: Callable[[str, str], None] | None = None,
) -> int:
    """Merge records by ID and write to SQLite."""
    # Detect actual ID key in the records
    actual_id_key = _detect_id_key(records, id_field)
    if log_fn:
        log_fn("merge_id_detect",
               f"expected id_field='{id_field}', actual key in records='{actual_id_key}'")

    merged: dict[str, dict[str, Any]] = {}
    skipped_no_id = 0

    for rec in records:
        rid = str(rec.get(actual_id_key, rec.get("ID", rec.get("id", ""))))
        if not rid:
            skipped_no_id += 1
            continue

        if rid not in merged:
            merged[rid] = {}
        for k, v in rec.items():
            if k == actual_id_key:
                continue
            col = _sanitize_col(k)
            if v is not None and v != "" and v != "null":
                if isinstance(v, (list, dict)):
                    v = json.dumps(v) if len(str(v)) < 200 else str(v)[:200]
                merged[rid][col] = v

    if log_fn:
        log_fn("merge_result",
               f"{len(records)} records -> {len(merged)} unique IDs "
               f"(skipped {skipped_no_id} without ID)")

    if not merged:
        return 0

    # Use table_name from schema, protect existing tables
    safe_name = _sanitize_col(table_name)
    if protected and safe_name.lower() in protected:
        safe_name = f"{safe_name}_doc"
        if log_fn:
            log_fn("merge_rename",
                   f"table '{table_name}' conflicts with protected, using '{safe_name}'")

    # Collect all columns across records
    all_cols: set[str] = set()
    for rec in merged.values():
        all_cols.update(rec.keys())
    sorted_cols = sorted(all_cols)

    # Infer types
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

    if log_fn:
        log_fn("merge_schema",
               f"table='{safe_name}', {len(sorted_cols)} columns, "
               f"types: {dict(list(col_types.items())[:5])}")

    # Write to SQLite — incremental upsert (preserves existing rows)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")

        # Check if table exists and get its current columns
        existing_cols: set[str] = set()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (safe_name,),
        )
        table_exists = cursor.fetchone() is not None
        if table_exists:
            for row_info in conn.execute(f'PRAGMA table_info("{safe_name}")').fetchall():
                existing_cols.add(row_info[1].lower())

        if not table_exists:
            col_defs = [f'"{id_field}" TEXT PRIMARY KEY']
            for col in sorted_cols:
                col_defs.append(f'"{col}" {col_types.get(col, "TEXT")}')
            create_sql = f'CREATE TABLE "{safe_name}" ({", ".join(col_defs)})'
            conn.execute(create_sql)
            if log_fn:
                log_fn("write_create_table",
                       f"created '{safe_name}' with PK='{id_field}', "
                       f"{len(sorted_cols)} cols")
        else:
            # Add any new columns that don't exist yet
            new_cols = []
            for col in sorted_cols:
                if col.lower() not in existing_cols and col.lower() != id_field.lower():
                    conn.execute(
                        f'ALTER TABLE "{safe_name}" ADD COLUMN '
                        f'"{col}" {col_types.get(col, "TEXT")}'
                    )
                    new_cols.append(col)
            if log_fn:
                log_fn("write_alter_table",
                       f"table '{safe_name}' exists ({len(existing_cols)} cols), "
                       f"added {len(new_cols)} new cols: {new_cols}")

        insert_cols = [f'"{id_field}"'] + [f'"{c}"' for c in sorted_cols]
        placeholders = ", ".join(["?"] * len(insert_cols))
        insert_sql = (
            f'INSERT OR REPLACE INTO "{safe_name}" '
            f'({", ".join(insert_cols)}) VALUES ({placeholders})'
        )

        for rid, rec in merged.items():
            row: list[Any] = [rid]
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
            log_fn("write_committed",
                   f"upserted {len(merged)} rows into '{safe_name}'")
    except Exception as e:
        if log_fn:
            log_fn("write_error", f"table='{safe_name}': {str(e)[:200]}")
        conn.rollback()
        conn.close()
        return 0

    conn.close()
    return len(merged)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _detect_id_key(records: list[dict[str, Any]], id_field: str) -> str:
    """Detect which key in the records is the actual ID field."""
    candidates = [id_field, "ID", "id", "identifier", "record_id",
                  "molecule_id", "registry_number", "hero_id", "race_id",
                  "superhero_id", "event_id", "member_id"]
    for key in candidates:
        hits = sum(1 for r in records[:20] if r.get(key))
        if hits > len(records[:20]) * 0.5:
            return key
    # Fallback: find any field with "id" in the name that's well-populated
    if records:
        for key in records[0]:
            if "id" in key.lower() or "identifier" in key.lower() or "number" in key.lower():
                hits = sum(1 for r in records[:20] if r.get(key))
                if hits > len(records[:20]) * 0.5:
                    return key
    return id_field


def _sanitize_col(name: str) -> str:
    """Convert a field name to a safe SQLite column name."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return name or "col"


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """Parse a JSON object from LLM response."""
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Remove markdown fences
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    # Find the JSON object
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start == -1 or brace_end == -1:
        return None
    json_str = raw[brace_start:brace_end + 1]
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        # Try fixing trailing commas
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            data = json.loads(fixed)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON array from LLM response."""
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
