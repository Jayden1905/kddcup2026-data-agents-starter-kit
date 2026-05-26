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
{db_context}
{field_guidance}
RULES:
1. Each record MUST have an ID field (numeric identifier, code, or registry number mentioned in text).
2. Extract ALL attributes: names, numeric values, categories, IDs, codes, statuses, labels.
3. CORRECTIONS: if text says "initially X, corrected/updated to Y" → use ONLY the corrected value Y.
4. Numbers: extract as plain numbers without units. Use the attribute name to indicate what it measures.
5. If a value is unavailable/unknown/not applicable → use null.
6. Use snake_case for field names (e.g. height_cm, full_name, publisher_id).
7. If the same entity appears multiple times (e.g. biometrics in a later paragraph), merge into one record.
8. Extract EVERY entity/record mentioned. Do not skip any.

TEXT:
{text_chunk}

Return a JSON array of objects. Each object is one record:"""


def _build_field_list(schema: DocumentSchema) -> str:
    parts = []
    for f in schema.fields:
        desc = f.name
        if f.aliases:
            desc += f" (aka {', '.join(f.aliases[:3])})"
        parts.append(desc)
    return ", ".join(parts)


def _get_db_context(db_path: Path, entity_name: str) -> str:
    """Get existing DB schema as context hints for extraction."""
    try:
        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if not tables:
            conn.close()
            return ""
        lines = [
            "EXISTING DB TABLES (extract fields that can JOIN with these):"
        ]
        for t in tables:
            cols_info = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
            cols = [f"{r[1]} ({r[2]})" for r in cols_info]
            lines.append(f"  {t}: {', '.join(cols)}")
        conn.close()
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def _get_knowledge_fields(knowledge_text: str, entity_name: str) -> str:
    """Extract field definitions from knowledge.md for the target entity."""
    if not knowledge_text:
        return ""
    lines = knowledge_text.split("\n")
    entity_lower = entity_name.lower()

    field_lines: list[str] = []

    # Strategy 1: Find "- **Entity**: description" followed by indented "  - **field**:"
    # This is the most precise format (entity definition with sub-fields)
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match: "- **Molecule**: ..." — entity name must be in the bold label
        if stripped.startswith("- **"):
            bold_end = stripped.find("**", 4)
            if bold_end > 0:
                bold_label = stripped[4:bold_end].lower()
                if entity_lower == bold_label or entity_lower in bold_label.split():
                    for j in range(i + 1, min(i + 20, len(lines))):
                        sub = lines[j]
                        if sub.startswith("  ") and "**" in sub and ":" in sub:
                            field_lines.append(sub.strip())
                        elif sub.strip() and not sub.startswith(" "):
                            break
                    if field_lines:
                        break

    # Strategy 2: Find a heading like "### Superhero" with field bullets below
    if not field_lines:
        best_start = -1
        best_specificity = 0
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if stripped.startswith("#") and entity_lower in stripped:
                level = len(stripped) - len(stripped.lstrip("#"))
                exact = stripped.lstrip("#").strip() == entity_lower
                specificity = level + (10 if exact else 0)
                if specificity > best_specificity:
                    best_specificity = specificity
                    best_start = i

        if best_start >= 0:
            for line in lines[best_start + 1:]:
                stripped = line.strip()
                if stripped.startswith("#"):
                    if field_lines:
                        break
                    continue
                if stripped.startswith("- **") and ":" in stripped:
                    if "SELECT " in stripped or "Formula" in stripped:
                        continue
                    if "Explanation" in stripped:
                        continue
                    field_lines.append(stripped)

    if field_lines:
        return "SCHEMA FROM KNOWLEDGE:\n" + "\n".join(field_lines) + "\n"
    return ""


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
    If a single section exceeds max_chars, splits it by paragraphs.
    Returns the heading of the first section in each group for field detection.
    """
    chunks: list[tuple[str, str]] = []
    buffer: list[str] = []
    buffer_len = 0
    heading = ""

    for section in sections:
        paragraphs = [p for p in section.paragraphs if len(p) >= 30]
        section_text = "\n\n".join(paragraphs)
        if not section_text:
            continue

        # If this single section exceeds max_chars, split by paragraphs
        if len(section_text) > max_chars:
            # Flush current buffer first
            if buffer:
                chunks.append((heading, "\n\n".join(buffer)))
                buffer = []
                buffer_len = 0
                heading = ""

            para_buffer: list[str] = []
            para_len = 0
            for para in paragraphs:
                if para_buffer and para_len + len(para) > max_chars:
                    chunks.append((section.heading, "\n\n".join(para_buffer)))
                    para_buffer = []
                    para_len = 0
                para_buffer.append(para)
                para_len += len(para)
            if para_buffer:
                chunks.append((section.heading, "\n\n".join(para_buffer)))
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
        db_context = _get_db_context(db_path, doc_path.stem)
        knowledge_fields = _get_knowledge_fields(knowledge_text, doc_path.stem)
        if knowledge_fields:
            db_context = knowledge_fields + db_context
        if log_fn:
            log_fn("schema", f"{doc_path.stem}: {len(schema.fields)} fields, "
                   f"entity={schema.entity_name}")

        # Stage 3: Chunk by section groups
        section_chunks = _chunk_by_section(data_sections, max_chars=20000)
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

            field_guidance = ""
            if section_fields:
                field_guidance = f"KNOWN FIELDS: {section_fields}\n"
            elif schema.fields:
                field_guidance = f"KNOWN FIELDS: {_build_field_list(schema)}\n"

            prompt = EXTRACT_PROMPT.format(
                entity_name=schema.entity_name,
                db_context=db_context,
                field_guidance=field_guidance,
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


def _detect_id_key(records: list[dict[str, Any]], id_field: str) -> str:
    """Detect which key in the records is the actual ID field."""
    candidates = [id_field, "ID", "id", "identifier", "record_id",
                  "molecule_id", "registry_number", "hero_id", "race_id"]
    for key in candidates:
        hits = sum(1 for r in records[:20] if r.get(key))
        if hits > len(records[:20]) * 0.5:
            return key
    # Fallback: find a field that looks like an ID (present in most records, unique-ish)
    if records:
        for key in records[0]:
            if "id" in key.lower() or "identifier" in key.lower():
                hits = sum(1 for r in records[:20] if r.get(key))
                if hits > len(records[:20]) * 0.5:
                    return key
    return id_field


def _merge_and_write(
    records: list[dict[str, Any]],
    schema: DocumentSchema,
    db_path: Path,
    protected: set[str],
    id_field: str,
    log_fn: Callable[[str, str], None] | None = None,
) -> int:
    """Merge extracted records by composite key and write to SQLite."""
    actual_id_key = _detect_id_key(records, id_field)
    merged: dict[str, dict[str, Any]] = {}

    for rec in records:
        rid = str(rec.get(actual_id_key, rec.get("ID", rec.get("id", ""))))
        if not rid:
            continue
        date_val = rec.get("Date", rec.get("date", ""))
        if date_val:
            key = f"{rid}_{date_val}"
        else:
            key = rid

        if key not in merged:
            merged[key] = {"record_id": key}
        for k, v in rec.items():
            if k == actual_id_key:
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
            if isinstance(val, (list, dict)):
                is_numeric = False
                break
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
                # Flatten lists/dicts to strings
                if isinstance(val, (list, dict)):
                    val = str(val)
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
