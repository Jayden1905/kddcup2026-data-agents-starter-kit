"""Semantic extraction pipeline with context-window-aware batching.

Architecture:
  1. PLANNER — analyzes doc structure + question, determines schema
  2. BATCHER — splits by entity boundaries (paragraphs), packs into batches
     respecting a context budget (prompt + content + output headroom)
  3. WORKERS — extract records from batches in parallel (8 threads)
  4. VALIDATOR — checks completeness, finds IDs with missing fields
  5. REPAIR — retries gaps with few-shot examples from successful extractions

Batching is context-window aware: each batch fills up to `max_batch_content_chars`
(default 6000) of entity paragraphs, leaving room for prompt template (~2K) and
output (~2K). Short paragraphs get more entities per batch; long paragraphs fewer.
"""

from __future__ import annotations

import re
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """Analyze this document and determine what structured data to extract.

QUESTION the user needs to answer:
{question}

{db_context}

{knowledge_hint}

DOCUMENT SAMPLE (from different sections of the document):
---
{sample_text}
---

First, reason step by step inside <think> tags:
- What entity does each record represent? (One paragraph = one entity record)
- What column names appear in the EXISTING DATABASE TABLES or RELEVANT KNOWLEDGE above? List them ALL exactly as written.
- Which of those columns can be populated from this document?
- What does the _id look like in this document? Does it match any foreign key in the database tables? If yes, this doc is a LOOKUP TABLE — extract the entity's own attributes (name, label, properties), not the FK column name from the referencing table.
- Does the document contain CORRECTIONS (e.g. "initially X, corrected to Y")? If so, the FINAL value should be used.
- Are there NaN/missing values? Those become NULL.

Then return ONLY a JSON object:
{{"entity": "name of entity", "fields": ["_id", "field1", "field2", ...], "id_description": "how _id appears in text"}}

CRITICAL RULES:
- Field names MUST match the database schema / knowledge definitions exactly. Copy the exact column names character for character (e.g. "height_cm" not "height", "publisher_id" not "publisher").
- Include ALL columns that this document could populate, even if only some records have them.
- CATEGORICAL FIELDS: If the document assigns each record to a category/type/class (e.g. "categorized as X", "designated for Y", "classified as Z"), you MUST include that as a field. Name it to match the knowledge/schema (e.g. "category", "type").
- STATUS/QUALIFIER FIELDS: If the document assigns qualitative assessments to numeric values (e.g. "normal", "abnormal", "elevated", "impaired", "within range", "exceeds threshold", "below average"), include a corresponding status field named "<measurement>_status". This captures the document's own judgment rather than requiring threshold inference later.
- FOREIGN KEYS: If records reference entities from the existing database (e.g. event names, member IDs), include a link field (e.g. "link_to_event", "event_id") matching the schema convention.
- LOOKUP TABLE vs FK SOURCE: If the document describes the ENTITY ITSELF that other tables reference via FK (e.g. doc="major.md" and DB has member.link_to_major), then this doc is a LOOKUP TABLE. Extract the entity's OWN attributes (_id, name/label) — NOT the FK column name from the referencing table. The _id here IS the FK target value. Use knowledge-defined field names (e.g. "major_name") for the descriptive attribute.
- AMOUNTS: If the knowledge defines a field name for monetary values (e.g. "amount"), use that name — not synonyms like "allocation" or "budget_value".
- If a field has DATE values in prose (e.g. "tenth of February, 1986"), include it — workers will convert to ISO format.
- NAME/LABEL FIELDS: If each record in the document has a proper name, title, or label (e.g. "Belgium Jupiler League", "John Smith"), include a "name" field (or matching schema field like "team_long_name"). This is critical for filtering and joining later.
- Always include "_id" as the first field."""

WORKER_PROMPT = """Extract ALL records from this text as a JSON array.

ENTITY: {entity}
FIELDS TO EXTRACT: {fields}
ID FORMAT: {id_description}

{db_context}

{fk_lookup}

{schema_hint}

TEXT (each section separated by --- is ONE record):
---
{chunk_text}
---

RULES:
1. Return a JSON array. Each object = one {entity}. Each section between --- separators = one record.
2. Extract EVERY {entity} — do NOT skip any section.
3. ONLY use the field names listed above. Do NOT add extra fields.
4. "_id" must be the EXACT identifier from the text — preserve original format.
5. CORRECTIONS: When text says "initially X" / "first logged as X" then "corrected to Y" / "amended to Y" / "confirmed at Y" / "rectified to Y", use ONLY the FINAL corrected value Y.
6. MISSING DATA: If a value is "not available", "NaN", "not recorded", "unavailable", or entire panel is missing, use null for that field.
7. NOISE FILTERING: Ignore irrelevant sentences about hobbies, library books, parking, furniture, weather, travel, exercise, diet, or office logistics. Only extract the actual data fields.
8. Values should be clean atomic data (numbers, names, IDs), not prose.
9. For FK/link fields: if VALID REFERENCES are listed above, use the ID value (e.g. "recXYZ"), NOT the name.
10. For category/type fields: look for phrases like "categorized as X", "classified as X", "designated for X". Extract ONLY the short label (e.g. "Advertisement", "Food").
11. For numeric fields: extract the NUMBER only. Look for "amount of 75", "value of 67.81", "level at 28.0" etc.
15. For STATUS fields (any field ending in _status): derive from the text's qualitative assessment of that measurement. Positive/good assessments (e.g. "normal", "within range", "healthy", "unremarkable") → "normal". Negative/bad assessments (e.g. "elevated", "impaired", "compromised", "abnormal", "deficient") → "abnormal". Uncertain/edge assessments (e.g. "borderline", "upper limit") → "borderline". If the text makes no qualitative statement about that value, use null.
12. DATES: Convert natural language dates to ISO format (YYYY-MM-DD). "tenth of February, 1986" → "1986-02-10".
13. CRITICAL — PARTIAL RECORDS ARE MANDATORY: You MUST extract EVERY entity that has an _id, even if only 1-2 fields can be filled. Use null for all unknown fields. A record with just {{"_id": 7, "name": "X"}} and nulls for everything else is CORRECT output. NEVER return [] if the text mentions any entity IDs.
14. If the text truly contains NO entity identifiers at all, return: []

Return ONLY the JSON array, nothing else."""

REPAIR_PROMPT = """Some records are missing fields. Extract the missing data from this chunk.

RECORDS NEEDING REPAIR (these IDs exist but are missing some fields):
{missing_ids}

EXAMPLE OF A COMPLETE RECORD:
{example_record}

FIELDS EXPECTED: {fields}

TEXT CHUNK:
---
{chunk_text}
---

Return ONLY a JSON array with repaired records (include _id + the previously missing fields).
If none of these IDs appear in this chunk, return: []"""


# ---------------------------------------------------------------------------
# Document chunking — entity-boundary aware, context-window constrained
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split document into paragraphs. LLM decides what's an entity vs noise."""
    sections = re.split(r"(?=^#{1,4}\s)", text, flags=re.MULTILINE)

    paragraphs: list[str] = []
    for section in sections:
        parts = [p.strip() for p in re.split(r"\n\s*\n", section)]
        for p in parts:
            if len(p) >= 60:
                paragraphs.append(p)
    return paragraphs


def _batch_paragraphs(
    paragraphs: list[str], max_batch_content_chars: int = 4000,
    max_entities_per_batch: int = 15,
) -> list[str]:
    """Pack entity paragraphs into batches respecting context budget.

    Each batch contains multiple paragraphs separated by \\n\\n---\\n\\n
    so the LLM can clearly see entity boundaries. The budget ensures
    total content fits within the model's context window alongside
    prompt template (~2K chars) and output headroom (~2K chars).
    """
    if not paragraphs:
        return []

    batches: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 7  # account for \n\n---\n\n separator
        if current and (current_len + para_len > max_batch_content_chars
                        or len(current) >= max_entities_per_batch):
            batches.append("\n\n---\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len

    if current:
        batches.append("\n\n---\n\n".join(current))

    return batches


def _chunk_document(text: str, max_chunk_chars: int = 4000) -> list[str]:
    """Split document into context-aware batches of entity paragraphs.

    Falls back to simple paragraph grouping if no entity paragraphs found.
    """
    paragraphs = _split_paragraphs(text)
    if paragraphs:
        return _batch_paragraphs(paragraphs, max_batch_content_chars=max_chunk_chars)

    # Fallback: any paragraph over 60 chars, batched
    all_paras = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if len(p.strip()) >= 60
    ]
    if not all_paras:
        return []
    return _batch_paragraphs(all_paras, max_batch_content_chars=max_chunk_chars)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_knowledge_hint(question: str, knowledge_text: str) -> str:
    if not knowledge_text:
        return ""
    q_words = set(re.findall(r"[a-z]{3,}", question.lower()))
    relevant: list[str] = []
    for line in knowledge_text.split("\n"):
        if any(w in line.lower() for w in q_words):
            relevant.append(line.strip())
    if relevant:
        return "RELEVANT KNOWLEDGE:\n" + "\n".join(relevant[:10])
    return ""


def _get_existing_db_context(db_path: Path) -> str:
    if not db_path or not db_path.exists():
        return ""
    try:
        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute(
            r"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\_%' ESCAPE '\'"
        ).fetchall()]
        if not tables:
            conn.close()
            return ""

        parts: list[str] = ["EXISTING DATABASE TABLES (from structured files):"]
        for table in tables[:5]:
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
            rows = conn.execute(f'SELECT * FROM "{table}" LIMIT 3').fetchall()
            parts.append(f"TABLE: {table} ({', '.join(cols)})")
            for row in rows:
                parts.append(f"  {row}")
        parts.append("\nThe '_id' you extract must match the foreign key values in these tables exactly.")
        conn.close()
        return "\n".join(parts)
    except Exception:
        return ""


def _build_fk_lookup(db_path: Path) -> str:
    """Build a FK lookup reference: PK → human-readable name for each existing table.

    Workers use this to output proper FK values instead of prose descriptions.
    """
    if not db_path or not db_path.exists():
        return ""
    try:
        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute(
            r"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\_%' ESCAPE '\'"
        ).fetchall()]
        if not tables:
            conn.close()
            return ""

        parts: list[str] = []
        for table in tables[:5]:
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
            if not cols:
                continue
            pk_col = cols[0]
            # Find a name/label column
            name_col = None
            for c in cols[1:]:
                if any(n in c.lower() for n in ("name", "title", "label", "display")):
                    name_col = c
                    break
            if not name_col:
                continue
            rows = conn.execute(
                f'SELECT "{pk_col}", "{name_col}" FROM "{table}" WHERE "{name_col}" IS NOT NULL LIMIT 50'
            ).fetchall()
            if rows:
                parts.append(f"VALID {table} REFERENCES (use {pk_col} as FK value):")
                for pk_val, name_val in rows:
                    parts.append(f"  {pk_val} = {name_val}")
        conn.close()
        if parts:
            return "\n".join(parts)
        return ""
    except Exception:
        return ""


def _resolve_fk_post_extraction(
    records: list[dict[str, Any]],
    db_path: Path,
    plan_fields: list[str],
    log_fn: Callable[[str, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Post-extraction: resolve FK fields by fuzzy-matching text values to existing PKs.

    For fields that look like FKs (link_to_X, X_id, event_name when event table exists),
    match extracted text values against actual PK/name pairs in referenced tables.
    """
    if not db_path or not db_path.exists() or not records:
        return records

    try:
        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute(
            r"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\_%' ESCAPE '\'"
        ).fetchall()]
    except Exception:
        return records

    # Build lookup: table_name → {pk_val: name_val, name_val_lower: pk_val}
    table_lookups: dict[str, dict[str, str]] = {}
    table_pk_cols: dict[str, str] = {}
    for table in tables:
        try:
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
            if not cols:
                continue
            pk_col = cols[0]
            table_pk_cols[table] = pk_col
            name_col = None
            for c in cols[1:]:
                if any(n in c.lower() for n in ("name", "title", "label", "display")):
                    name_col = c
                    break
            if not name_col:
                continue
            rows = conn.execute(
                f'SELECT "{pk_col}", "{name_col}" FROM "{table}" WHERE "{name_col}" IS NOT NULL'
            ).fetchall()
            lookup: dict[str, str] = {}
            for pk_val, name_val in rows:
                lookup[str(name_val).lower()] = str(pk_val)
                lookup[str(pk_val)] = str(pk_val)  # identity mapping
            table_lookups[table] = lookup
        except Exception:
            continue
    conn.close()

    if not table_lookups:
        return records

    # Identify FK fields in plan: link_to_X, X_id, or field named after another table
    fk_fields: dict[str, str] = {}  # field_name → target_table
    for field in plan_fields:
        fl = field.lower()
        if fl == "_id":
            continue
        # link_to_event → event
        if fl.startswith("link_to_"):
            target = fl[8:]
            for t in tables:
                if t.lower() == target or t.lower().rstrip("s") == target:
                    fk_fields[field] = t
                    break
        # event_id → event
        elif fl.endswith("_id"):
            target = fl[:-3]
            for t in tables:
                if t.lower() == target or t.lower().rstrip("s") == target:
                    fk_fields[field] = t
                    break
        # event_name when entity != event → FK to event
        elif fl.endswith("_name"):
            target = fl[:-5]
            for t in tables:
                if t.lower() == target or t.lower().rstrip("s") == target:
                    fk_fields[field] = t
                    break

    if not fk_fields:
        return records

    resolved_count = 0
    for record in records:
        for fk_field, target_table in fk_fields.items():
            val = record.get(fk_field)
            if val is None:
                continue
            val_str = str(val).strip()
            val_lower = val_str.lower()
            lookup = table_lookups.get(target_table, {})
            if not lookup:
                continue
            # Already a valid PK?
            if val_str in lookup:
                continue
            # Exact name match
            if val_lower in lookup:
                record[fk_field] = lookup[val_lower]
                resolved_count += 1
                continue
            # Fuzzy: check if any lookup name is contained in the value or vice versa
            best_match = None
            best_len = 0
            for name_key, pk_val in lookup.items():
                if name_key == pk_val:  # skip identity entries
                    continue
                if name_key in val_lower or val_lower in name_key:
                    if len(name_key) > best_len:
                        best_match = pk_val
                        best_len = len(name_key)
            if best_match:
                record[fk_field] = best_match
                resolved_count += 1

    if log_fn and resolved_count > 0:
        log_fn("fk_resolved", f"Resolved {resolved_count} FK values across fields {list(fk_fields.keys())}")

    return records


def _parse_json_response(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [r for r in data if isinstance(r, dict)]
            except json.JSONDecodeError:
                pass
    return []


def _extract_knowledge_columns(knowledge_text: str) -> set[str]:
    """Extract column names mentioned in knowledge.md definitions."""
    cols: set[str] = set()
    for match in re.findall(r"`([a-z][a-z0-9_]+)`", knowledge_text):
        cols.add(match)
    for match in re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", knowledge_text):
        if not match.startswith(("http", "www")):
            cols.add(match)
    return cols


def _get_sample_text(text: str, max_chars: int = 4000) -> str:
    """Get representative sample from different parts of the document.

    Picks paragraphs at evenly spaced intervals to capture all sections
    (e.g., names section, appearance section, biometrics section).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 60]
    if not paragraphs:
        return text[:max_chars]

    # Pick 5-6 paragraphs evenly spaced through the document
    n_samples = min(6, len(paragraphs))
    step = max(1, len(paragraphs) // n_samples)
    indices = [i * step for i in range(n_samples) if i * step < len(paragraphs)]

    selected = [paragraphs[i][:800] for i in indices]
    return "\n\n".join(selected)[:max_chars]


# ---------------------------------------------------------------------------
# Agent 1: PLANNER
# ---------------------------------------------------------------------------

def _run_planner(
    model: ModelAdapter,
    text: str,
    question: str,
    db_context: str,
    knowledge_hint: str,
    log_fn: Callable[[str, str], None] | None,
    doc_name: str = "",
) -> dict[str, Any] | None:
    """Planner analyzes doc and determines extraction schema."""
    sample = _get_sample_text(text)
    if log_fn:
        log_fn("planner_input", f"question={question[:120]}")
        log_fn("planner_input", f"knowledge_hint={knowledge_hint[:200]}")
        log_fn("planner_input", f"db_context={db_context[:200]}")
        log_fn("planner_input", f"sample_text (first 300 chars)={sample[:300]}")

    doc_hint = ""
    if doc_name:
        doc_hint = f"\nDOCUMENT NAME: {doc_name}\nThe document filename indicates this file contains {doc_name} records. The entity name should be \"{doc_name}\" unless the content clearly represents something else.\n"

    prompt = PLANNER_PROMPT.format(
        question=question,
        sample_text=sample,
        db_context=db_context,
        knowledge_hint=knowledge_hint,
    ) + doc_hint
    messages = [ModelMessage(role="user", content=prompt)]

    try:
        raw = model.complete(messages)
    except Exception as e:
        if log_fn:
            log_fn("planner_error", str(e)[:200])
        return None

    if log_fn:
        log_fn("planner_raw_response", raw[:500])

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        plan = json.loads(raw)
        if isinstance(plan, dict) and "fields" in plan:
            # Strip table alias prefixes (e.g. "leagueData.name" → "name")
            fields = [f.split(".")[-1] if "." in f and f != "_id" else f for f in plan["fields"]]
            # Deduplicate while preserving order
            seen: set[str] = set()
            deduped: list[str] = []
            for f in fields:
                if f not in seen:
                    seen.add(f)
                    deduped.append(f)
            fields = deduped
            # Ensure _id is always the first field
            if "_id" not in fields:
                fields.insert(0, "_id")
            elif fields[0] != "_id":
                fields.remove("_id")
                fields.insert(0, "_id")
            # Drop FK-like fields only when they reference this entity itself
            # (e.g. extracting "event" entity → drop "link_to_event" since that's
            # a column from a REFERENCING table, not this entity's own attribute).
            # Keep link_to_X when X is a DIFFERENT entity (outgoing FK).
            entity_name = (plan.get("entity") or "").lower()
            fields = [
                f for f in fields
                if not f.lower().startswith("link_to_")
                or f.lower()[8:] != entity_name
            ]
            # Ensure a "name" field exists — every entity has a human-readable label
            has_name = any(
                "name" in f.lower() or "title" in f.lower() or "label" in f.lower()
                for f in fields if f != "_id"
            )
            if not has_name:
                fields.append("name")
            plan["fields"] = fields
            if log_fn:
                log_fn("planner_done", f"entity={plan.get('entity')}, fields={plan.get('fields')}, id_description={plan.get('id_description')}")
            return plan
    except json.JSONDecodeError:
        pass

    if log_fn:
        log_fn("planner_fail", f"Could not parse plan: {raw[:200]}")
    return None


# ---------------------------------------------------------------------------
# Agent 2: WORKERS (parallel)
# ---------------------------------------------------------------------------

def _run_workers(
    model: ModelAdapter,
    chunks: list[str],
    plan: dict[str, Any],
    db_context: str,
    fk_lookup: str,
    log_fn: Callable[[str, str], None] | None,
) -> list[dict[str, Any]]:
    """Workers extract records from chunks in parallel."""
    entity = plan.get("entity", "record")
    fields = plan.get("fields", ["_id"])
    id_description = plan.get("id_description", "unique identifier")
    fields_str = ", ".join(fields)

    all_records: list[dict[str, Any]] = []
    failed_chunks: list[int] = []

    def _extract_chunk(idx_chunk: tuple[int, str]) -> tuple[int, list[dict[str, Any]]]:
        i, chunk = idx_chunk
        schema_hint = ""
        if all_records:
            example = next((r for r in all_records if len(r) > 2), None)
            if example:
                schema_hint = f"EXAMPLE OUTPUT FORMAT:\n{json.dumps(example, default=str)}"

        prompt = WORKER_PROMPT.format(
            entity=entity,
            fields=fields_str,
            id_description=id_description,
            db_context=db_context,
            fk_lookup=fk_lookup,
            schema_hint=schema_hint,
            chunk_text=chunk,
        )
        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw_response = model.complete(messages)
            records = _parse_json_response(raw_response)
            if log_fn and not records:
                log_fn("worker_empty", f"chunk {i+1}: no records parsed. Response preview: {raw_response[:150]}")
            if log_fn and records and i > 0 and i % 10 == 0:
                log_fn("worker_sample", f"chunk {i+1}: first record={json.dumps(records[0], default=str)[:200]}")
            return i, records
        except Exception as e:
            if log_fn:
                log_fn("worker_error", f"chunk {i+1}: {str(e)[:150]}")
            return i, []

    # First batch sequentially to get an example for schema hint
    if chunks:
        expected_entities = chunks[0].count("\n\n---\n\n") + 1
        _, first_records = _extract_chunk((0, chunks[0]))
        if first_records:
            all_records.extend(first_records)
        else:
            failed_chunks.append(0)
        if log_fn:
            log_fn(
                "worker_done",
                f"batch 1/{len(chunks)}: {len(first_records)}/{expected_entities} records extracted"
                + (f" | sample: {json.dumps(first_records[0], default=str)[:200]}" if first_records else ""),
            )

    # Remaining batches in parallel
    remaining = [(i, chunk) for i, chunk in enumerate(chunks) if i > 0]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_extract_chunk, ic): ic for ic in remaining}
        for future in as_completed(futures):
            i, records = future.result()
            expected = chunks[i].count("\n\n---\n\n") + 1
            if records:
                all_records.extend(records)
            else:
                failed_chunks.append(i)
            if log_fn:
                log_fn("worker_done", f"batch {i+1}/{len(chunks)}: {len(records)}/{expected} records")

    if log_fn:
        log_fn(
            "workers_complete",
            f"{len(all_records)} total records from {len(chunks)} batches "
            f"({len(failed_chunks)} empty batches)",
        )

    # Normalize _id: strip text prefixes from IDs that contain numbers
    # BUT skip normalization if IDs look like opaque tokens (base62, UUIDs)
    id_values = [str(r.get("_id", "")) for r in all_records if r.get("_id")]
    numeric_count = sum(1 for v in id_values if v.isdigit())
    has_digits_count = sum(1 for v in id_values if re.search(r"\d", v))
    token_like = sum(
        1 for v in id_values
        if (len(v) >= 10 and re.match(r"^[a-zA-Z0-9]+$", v) and not v.isdigit())
        or (len(v) >= 4 and re.match(r"^[a-zA-Z]+\d+$", v))
    )
    if token_like > len(id_values) * 0.2:
        pass
    elif numeric_count > len(id_values) * 0.3 or has_digits_count > len(id_values) * 0.7:
        for r in all_records:
            rid = str(r.get("_id", ""))
            if not rid.isdigit():
                nums = re.findall(r"\d+", rid)
                if nums:
                    r["_id"] = nums[-1]
        if log_fn:
            normalized = sum(1 for r in all_records if str(r.get("_id", "")).isdigit())
            log_fn("id_normalize", f"{normalized}/{len(all_records)} IDs are now numeric")

    return all_records


# ---------------------------------------------------------------------------
# Agent 3: VALIDATOR
# ---------------------------------------------------------------------------

def _run_validator(
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    log_fn: Callable[[str, str], None] | None,
) -> dict[str, list[str]]:
    """Validator checks completeness — finds IDs with missing fields."""
    fields = set(plan.get("fields", []))
    if not fields or "_id" not in fields:
        return {}

    # Deduplicate first
    merged: dict[str, dict[str, Any]] = {}
    for r in records:
        rid = r.get("_id")
        if rid is None:
            continue
        rid_str = str(rid)
        if rid_str not in merged:
            merged[rid_str] = dict(r)
        else:
            for k, v in r.items():
                if v is not None:
                    merged[rid_str][k] = v

    # Find IDs missing expected fields
    incomplete: dict[str, list[str]] = {}
    for rid, record in merged.items():
        missing = [f for f in fields if f != "_id" and record.get(f) is None]
        if missing:
            incomplete[rid] = missing

    if log_fn:
        all_fields = set()
        for v in merged.values():
            all_fields.update(v.keys())
        log_fn("validator_summary", f"{len(merged)} unique IDs, {len(all_fields)} total fields: {sorted(all_fields)}")
        if incomplete:
            sample_incomplete = {k: v for k, v in list(incomplete.items())[:5]}
            log_fn("validator_gaps", f"{len(incomplete)} incomplete: {sample_incomplete}")
        else:
            log_fn("validator_pass", "All records have all expected fields")

    return incomplete


# ---------------------------------------------------------------------------
# Agent 4: REPAIR
# ---------------------------------------------------------------------------

def _run_repair(
    model: ModelAdapter,
    chunks: list[str],
    all_records: list[dict[str, Any]],
    incomplete: dict[str, list[str]],
    plan: dict[str, Any],
    log_fn: Callable[[str, str], None] | None,
) -> list[dict[str, Any]]:
    """Repair agent retries chunks to fill in missing fields."""
    if not incomplete:
        return []

    fields = plan.get("fields", [])
    fields_str = ", ".join(fields)

    # Find a good example record (one with most fields filled)
    example = max(all_records, key=lambda r: sum(1 for v in r.values() if v is not None), default={})
    example_str = json.dumps(example, default=str)

    # Build list of IDs needing repair
    incomplete_ids = list(incomplete.keys())[:20]  # cap to avoid huge prompts
    missing_summary = ", ".join(f"{rid} (needs: {', '.join(incomplete[rid][:3])})" for rid in incomplete_ids[:10])

    repair_records: list[dict[str, Any]] = []

    def _repair_chunk(chunk: str) -> list[dict[str, Any]]:
        prompt = REPAIR_PROMPT.format(
            missing_ids=missing_summary,
            example_record=example_str,
            fields=fields_str,
            chunk_text=chunk,
        )
        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = model.complete(messages)
            return _parse_json_response(raw)
        except Exception:
            return []

    # Only retry chunks that returned 0 records (likely the ones with missing data)
    empty_chunk_indices = []
    records_per_chunk: dict[int, int] = {}
    for i, chunk in enumerate(chunks):
        # Estimate: if chunk has an incomplete ID mentioned, retry it
        chunk_lower = chunk.lower()
        has_incomplete = any(str(rid) in chunk for rid in incomplete_ids[:20])
        if has_incomplete:
            empty_chunk_indices.append(i)

    if not empty_chunk_indices:
        return []

    if log_fn:
        log_fn("repair_start", f"Retrying {len(empty_chunk_indices)} chunks for {len(incomplete_ids)} incomplete IDs")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_repair_chunk, chunks[i]): i for i in empty_chunk_indices[:15]}
        for future in as_completed(futures):
            records = future.result()
            repair_records.extend(records)

    if log_fn:
        log_fn("repair_done", f"Recovered {len(repair_records)} records")

    return repair_records


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def chunked_extract(
    doc_path: Path,
    db_path: Path,
    model: ModelAdapter,
    question: str,
    knowledge_text: str = "",
    log_fn: Callable[[str, str], None] | None = None,
    protected_tables: set[str] | None = None,
) -> int:
    """Multi-agent extraction for a single document."""
    try:
        text = doc_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    if len(text) < 100:
        return 0

    # Entity-boundary splitting + context-aware batching
    entity_paras = _split_paragraphs(text)
    if entity_paras:
        chunks = _batch_paragraphs(entity_paras)
    else:
        chunks = _chunk_document(text)

    if not chunks:
        if log_fn:
            log_fn("chunk_skip", f"{doc_path.stem}: no entity paragraphs found")
        return 0

    if log_fn:
        avg_para_len = sum(len(p) for p in entity_paras) // max(1, len(entity_paras)) if entity_paras else 0
        batch_sizes = [batch.count("\n\n---\n\n") + 1 for batch in chunks]
        log_fn(
            "batching",
            f"{doc_path.stem}: {len(text)} chars, {len(entity_paras)} entities "
            f"(avg {avg_para_len} chars/entity) → {len(chunks)} batches "
            f"(entities/batch: min={min(batch_sizes)}, max={max(batch_sizes)}, "
            f"avg={sum(batch_sizes)//len(batch_sizes)})",
        )

    knowledge_hint = _build_knowledge_hint(question, knowledge_text)
    db_context = _get_existing_db_context(db_path)

    if log_fn:
        log_fn("orchestrator_context", f"knowledge_hint_len={len(knowledge_hint)}, db_context_len={len(db_context)}")

    # Agent 1: PLANNER
    if log_fn:
        log_fn("phase", "=== PLANNER PHASE ===")
    plan = _run_planner(model, text, question, db_context, knowledge_hint, log_fn, doc_name=doc_path.stem)
    # Guard: if planner chose an entity that already exists as a DB table, override to doc name.
    # The document exists to provide data NOT already in the structured DB.
    if plan and plan.get("entity") and protected_tables:
        planner_entity = plan["entity"].lower()
        if planner_entity in {t.lower() for t in protected_tables} and planner_entity != doc_path.stem.lower():
            if log_fn:
                log_fn("planner_entity_override", f"'{plan['entity']}' conflicts with existing DB table, using '{doc_path.stem}'")
            plan["entity"] = doc_path.stem
            plan["fields"] = ["_id"]
            plan["id_description"] = "unique identifier for each record"
    if not plan:
        # Build a better fallback using knowledge columns
        fallback_fields = ["_id"]
        known_cols = _extract_knowledge_columns(knowledge_text)
        # Add columns from knowledge that are likely entity attributes
        skip_words = {"select", "from", "where", "count", "group", "order", "join", "having", "case", "when", "then", "else", "end"}
        for col in sorted(known_cols):
            if col not in skip_words and len(col) > 2 and col != "_id":
                fallback_fields.append(col)
        if len(fallback_fields) < 3:
            fallback_fields.extend(["name", "category", "amount"])
        plan = {"entity": doc_path.stem, "fields": fallback_fields, "id_description": "unique identifier"}
        if log_fn:
            log_fn("planner_fallback", f"Using knowledge-based plan for {doc_path.stem}: {fallback_fields}")

    # Cross-reference: ensure planner fields include known columns from knowledge/DB
    known_cols = _extract_knowledge_columns(knowledge_text)
    if db_path and db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            for table in tables:
                cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
                known_cols.update(cols)
            conn.close()
        except Exception:
            pass
    # Augment: add columns from the same entity definition in knowledge.md
    if knowledge_text and plan.get("entity"):
        entity_name = plan["entity"].lower()
        plan_fields = set(f.lower() for f in plan.get("fields", []))
        # Find the entity's own definition block (between its **Entity**: and the next **Entity**:)
        entity_block = ""
        pattern = re.compile(
            r'-\s*\*\*' + re.escape(entity_name) + r'\*\*.*?(?=\n-\s*\*\*[A-Z]|\n###|\Z)',
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(knowledge_text)
        if match:
            entity_block = match.group(0)
        if entity_block:
            # Extract from `entityData.field` backtick patterns within this entity's definition
            backtick_fields = re.findall(r'`\w+\.(\w+)`', entity_block)
            knowledge_fields = re.findall(r"\*\*(\w+)\s*\(", entity_block)
            all_knowledge_fields = list(dict.fromkeys(knowledge_fields + backtick_fields))
            missing_added = []
            for col in all_knowledge_fields:
                if col.lower() not in plan_fields:
                    plan["fields"].append(col)
                    plan_fields.add(col.lower())
                    missing_added.append(col)
            if missing_added and log_fn:
                log_fn("planner_augment", f"Added from knowledge entity section: {missing_added}")

        # Fallback: if doc contains categorization patterns, ensure a type/category
        # field exists in the plan. This is a fundamental entity attribute.
        if "type" not in plan_fields and "category" not in plan_fields:
            # Check first ~5000 chars for categorization language
            sample_check = text[:5000].lower()
            has_categories = any(
                phrase in sample_check for phrase in
                ("categorized as", "classified as", "designated for", "category of", "categorized under")
            )
            if has_categories:
                plan["fields"].append("type")
                plan_fields.add("type")
                if log_fn:
                    log_fn("planner_augment", f"Added 'type' — doc contains categorization patterns")

    # Structural FK field injection: if the doc contains opaque IDs from a
    # structured table's PK, the doc entity has an outgoing FK. Ensure the field exists.
    # Skip if the IDs belong to the entity itself (lookup table pattern).
    if plan and db_path and db_path.exists():
        entity_name = (plan.get("entity") or "").lower()
        plan_fields_lower = {f.lower() for f in plan.get("fields", [])}
        try:
            conn = sqlite3.connect(str(db_path))
            tables = [r[0] for r in conn.execute(
                r"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\_%' ESCAPE '\'"
            ).fetchall()]
            for tbl in tables:
                if tbl.lower() == entity_name:
                    continue
                cols = conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()
                if not cols:
                    continue
                pk_col = cols[0][1]
                pk_type = (cols[0][2] or "").upper()
                if pk_type in ("INTEGER", "REAL", "NUMERIC"):
                    continue
                # Skip if the PK column references this entity (e.g. attendance.link_to_event
                # contains event IDs — these are the entity's OWN IDs, not outgoing FKs)
                if entity_name in pk_col.lower():
                    continue
                sample_rows = conn.execute(
                    f'SELECT DISTINCT "{pk_col}" FROM "{tbl}" WHERE "{pk_col}" IS NOT NULL LIMIT 20'
                ).fetchall()
                sample_vals = [str(r[0]) for r in sample_rows if r[0]]
                if not sample_vals:
                    continue
                is_opaque = all(
                    len(v) >= 5 and len(v) <= 50
                    and " " not in v and "@" not in v
                    and not v.replace(".", "").replace("-", "").isdigit()
                    and any(c.isalpha() for c in v)
                    and any(c.isdigit() for c in v)
                    for v in sample_vals
                )
                if not is_opaque:
                    continue
                matches = sum(1 for v in sample_vals if v in text)
                if matches >= 2 or (matches >= 1 and len(sample_vals) <= 2):
                    fk_field = f"link_to_{tbl}"
                    if fk_field.lower() not in plan_fields_lower:
                        plan["fields"].append(fk_field)
                        plan_fields_lower.add(fk_field.lower())
                        # Remove redundant <table>_name field — the name is
                        # accessible via FK join, and workers would fill it
                        # with wrong values since the doc has IDs, not names
                        redundant = f"{tbl}_name"
                        plan["fields"] = [
                            f for f in plan["fields"]
                            if f.lower() != redundant.lower()
                        ]
                        if log_fn:
                            log_fn("fk_field_injected", f"Added '{fk_field}' (replaced '{redundant}') — {tbl}.{pk_col} IDs found in doc ({matches}/{len(sample_vals)})")
            conn.close()
        except Exception:
            pass

    # Build FK lookup for workers
    fk_lookup = _build_fk_lookup(db_path)
    if fk_lookup and log_fn:
        log_fn("fk_lookup", f"{fk_lookup[:200]}")

    # Agent 2: WORKERS
    if log_fn:
        log_fn("phase", "=== WORKERS PHASE ===")
    all_records = _run_workers(model, chunks, plan, db_context, fk_lookup, log_fn)

    if not all_records:
        if log_fn:
            log_fn("orchestrator_empty", f"{doc_path.stem}: no records extracted from {len(chunks)} chunks")
        return 0

    # Post-extraction: resolve FK fields to actual PK values
    all_records = _resolve_fk_post_extraction(all_records, db_path, plan.get("fields", []), log_fn)

    # Agent 3: VALIDATOR
    if log_fn:
        log_fn("phase", "=== VALIDATOR PHASE ===")
    incomplete = _run_validator(all_records, plan, log_fn)

    # Agent 4: REPAIR (only if significant gaps)
    if incomplete and len(incomplete) > len(all_records) * 0.1:
        if log_fn:
            log_fn("phase", "=== REPAIR PHASE ===")
        repair_records = _run_repair(model, chunks, all_records, incomplete, plan, log_fn)
        if repair_records:
            all_records.extend(repair_records)
            if log_fn:
                log_fn("repair_merged", f"Total records after repair: {len(all_records)}")
    elif log_fn:
        log_fn("repair_skip", f"Skipped repair: {len(incomplete)} incomplete / {len(all_records)} total")

    # Write to SQLite
    if log_fn:
        log_fn("phase", "=== WRITE PHASE ===")
    table_name = re.sub(r"[^a-zA-Z0-9_]", "_", doc_path.stem).lower()
    if not table_name or table_name[0].isdigit():
        table_name = "t_" + table_name
    if protected_tables and table_name in protected_tables:
        table_name = f"{table_name}_doc"

    if log_fn:
        log_fn("write_target", f"table='{table_name}', records_to_write={len(all_records)}")

    return _write_records(db_path, table_name, all_records, log_fn)


# ---------------------------------------------------------------------------
# SQLite writer
# ---------------------------------------------------------------------------

def _write_records(
    db_path: Path,
    table_name: str,
    records: list[dict[str, Any]],
    log_fn: Callable[[str, str], None] | None = None,
) -> int:
    """Write extracted records to SQLite with deduplication."""
    if not records:
        return 0

    # Deduplicate by _id: merge records sharing an _id (last value wins)
    if any("_id" in r for r in records):
        merged: dict[str, dict[str, Any]] = {}
        for r in records:
            rid = r.get("_id")
            if rid is None:
                continue
            rid_str = str(rid)
            if rid_str not in merged:
                merged[rid_str] = dict(r)
            else:
                for k, v in r.items():
                    if v is not None and v != "":
                        merged[rid_str][k] = v
        if log_fn:
            all_fields = set()
            for v in merged.values():
                all_fields.update(v.keys())
            log_fn("dedup", f"{len(records)} raw -> {len(merged)} unique, {len(all_fields)} fields: {sorted(all_fields)}")
            sample_records = list(merged.values())[:3]
            for i, sr in enumerate(sample_records):
                log_fn("dedup_sample", f"record[{i}]: {json.dumps(sr, default=str)[:300]}")
            # Log merge stats for publisher_id specifically (diagnostic)
            has_pub = sum(1 for v in merged.values() if v.get("publisher_id") is not None)
            log_fn("dedup_fill_check", f"publisher_id filled: {has_pub}/{len(merged)}")
            # Count raw records with publisher_id set
            raw_with_pub = sum(1 for r in records if r.get("publisher_id") is not None)
            log_fn("dedup_fill_check", f"raw records with publisher_id: {raw_with_pub}/{len(records)}")
        records = list(merged.values())

    # Auto-fill <table>_id column from _id when it's the natural PK alias
    # (e.g., molecule table's "molecule_id" column should equal the _id value
    # since _id IS the molecule's identifier — workers often leave it null)
    pk_alias = f"{table_name}_id"
    if any(pk_alias in r for r in records):
        filled_count = sum(1 for r in records if r.get(pk_alias) not in (None, ""))
        if filled_count < len(records) * 0.5:
            for r in records:
                if r.get(pk_alias) in (None, "") and r.get("_id") not in (None, ""):
                    r[pk_alias] = str(r["_id"])
            if log_fn:
                new_filled = sum(1 for r in records if r.get(pk_alias) not in (None, ""))
                log_fn("pk_alias_fill", f"Filled {pk_alias} from _id: {new_filled}/{len(records)}")

    # Filter columns: keep only fields that appear in at least 10% of records
    col_counts: dict[str, int] = {}
    for r in records:
        for k, v in r.items():
            if v is not None:
                col_counts[k] = col_counts.get(k, 0) + 1

    threshold = max(1, len(records) // 10)
    frequent_cols = {k for k, count in col_counts.items() if count >= threshold}
    if log_fn:
        rare_cols = {k for k in col_counts if k not in frequent_cols}
        if rare_cols:
            log_fn("column_filter", f"Dropped {len(rare_cols)} rare columns (< {threshold} occurrences): {sorted(rare_cols)[:20]}")
        log_fn("column_filter", f"Keeping {len(frequent_cols)} columns: {sorted(frequent_cols)}")

    # Deduplicate case-insensitive column names (SQLite is case-insensitive)
    seen_lower: dict[str, str] = {}  # lowercase -> canonical name
    canonical_cols: set[str] = set()
    merge_map: dict[str, str] = {}  # duplicate -> canonical
    for col in sorted(frequent_cols):
        col_lower = col.lower()
        if col_lower in seen_lower:
            merge_map[col] = seen_lower[col_lower]
        else:
            seen_lower[col_lower] = col
            canonical_cols.add(col)
    if merge_map:
        # Merge duplicate columns in records
        for r in records:
            for dup, canonical in merge_map.items():
                if dup in r:
                    if canonical not in r or r[canonical] is None:
                        r[canonical] = r[dup]
                    del r[dup]
        frequent_cols = canonical_cols
        if log_fn:
            log_fn("column_dedup", f"Merged case-duplicates: {merge_map}")

    # Gather columns and infer types (only frequent cols)
    all_cols: dict[str, str] = {}
    for r in records:
        for k, v in r.items():
            if k not in all_cols and k in frequent_cols:
                if isinstance(v, (int, float)):
                    all_cols[k] = "REAL"
                else:
                    all_cols[k] = "TEXT"

    if not all_cols:
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        col_defs = ", ".join(f'"{c}" {t}' for c, t in all_cols.items())
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

        col_names = list(all_cols.keys())
        placeholders = ", ".join("?" * len(col_names))
        quoted_cols = ", ".join(f'"{c}"' for c in col_names)
        insert_sql = f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({placeholders})'

        written = 0
        for r in records:
            values = []
            for c in col_names:
                v = r.get(c)
                if v is not None and all_cols[c] == "REAL":
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        pass
                values.append(v)
            non_id = [v for k, v in r.items() if k != "_id" and v is not None]
            if not non_id:
                continue
            try:
                conn.execute(insert_sql, values)
                written += 1
            except Exception:
                continue

        conn.commit()
        if log_fn:
            log_fn("sqlite_written", f"Wrote {written} records to '{table_name}'")
        return written
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API (preserves interface for question_driven.py)
# ---------------------------------------------------------------------------

def compiled_extract_docs(
    doc_paths: list[Path],
    db_path: Path,
    model: ModelAdapter,
    question: str,
    knowledge_text: str = "",
    log_fn: Callable[[str, str], None] | None = None,
    structured_tables: list[str] | None = None,
) -> int:
    """Extract all docs using semantic entity-boundary extraction."""
    if not doc_paths:
        return 0

    import time

    protected = {t.lower() for t in structured_tables} if structured_tables else set()
    total = 0

    if log_fn:
        log_fn(
            "extraction_start",
            f"{len(doc_paths)} docs to extract: {[p.name for p in doc_paths]}",
        )

    for doc_path in doc_paths:
        t0 = time.time()

        n = chunked_extract(
            doc_path=doc_path,
            db_path=db_path,
            model=model,
            question=question,
            knowledge_text=knowledge_text,
            log_fn=log_fn,
            protected_tables=protected,
        )
        elapsed = time.time() - t0
        total += n
        if log_fn:
            log_fn(
                "doc_extracted",
                f"{doc_path.name}: {n} records in {elapsed:.1f}s",
            )

    if log_fn:
        log_fn("extraction_done", f"Total: {total} records from {len(doc_paths)} docs")

    return total
