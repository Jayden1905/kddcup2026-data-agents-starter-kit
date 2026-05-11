"""Multi-agent chunked extraction pipeline.

Architecture:
  1. PLANNER — analyzes doc structure + question, determines schema and chunking
  2. WORKERS — extract records from chunks in parallel (8 threads)
  3. VALIDATOR — checks completeness, finds IDs with missing fields
  4. REPAIR — retries gaps with few-shot examples from successful extractions

Each LLM call is small (~1-3KB). Robust because LLM handles format variations
natively. Multi-pass ensures high coverage.
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
- What entity does each record represent?
- What column names appear in the EXISTING DATABASE TABLES or RELEVANT KNOWLEDGE above? List them ALL exactly as written.
- Which of those columns can be populated from this document?
- What does the _id look like in this document? Does it match any foreign key in the database tables?

Then return ONLY a JSON object:
{{"entity": "name of entity", "fields": ["_id", "field1", "field2", ...], "id_description": "how _id appears in text"}}

CRITICAL RULES:
- Field names MUST match the database schema / knowledge definitions exactly. Copy the exact column names character for character (e.g. "height_cm" not "height", "publisher_id" not "publisher").
- Include ALL columns that this document could populate, even if only some records have them.
- CATEGORICAL FIELDS: If the document assigns each record to a category/type/class (e.g. "categorized as X", "designated for Y", "classified as Z"), you MUST include that as a field. Name it to match the knowledge/schema (e.g. "category", "type").
- FOREIGN KEYS: If records reference entities from the existing database (e.g. event names, member IDs), include a link field (e.g. "link_to_event", "event_id") matching the schema convention.
- AMOUNTS: If the knowledge defines a field name for monetary values (e.g. "amount"), use that name — not synonyms like "allocation" or "budget_value".
- Always include "_id" as the first field."""

WORKER_PROMPT = """Extract ALL records from this chunk as JSON array.

ENTITY: {entity}
FIELDS TO EXTRACT: {fields}
ID FORMAT: {id_description}

{db_context}

{fk_lookup}

{schema_hint}

TEXT CHUNK:
---
{chunk_text}
---

RULES:
1. Return a JSON array. Each object = one {entity}.
2. Extract EVERY {entity} — each paragraph typically describes one. Do NOT skip any.
3. ONLY use the field names listed above. Do NOT add extra fields.
4. "_id" must be the EXACT identifier from the text — preserve original format.
5. If text has corrections ("initially X, corrected/amended to Y"), use the FINAL corrected value only.
6. Values should be clean atomic data (numbers, names, IDs), not prose.
7. For FK/link fields: if VALID REFERENCES are listed above, use the ID value (e.g. "recXYZ"), NOT the name.
8. For category/type fields: look for phrases like "categorized as X", "classified as X", "designated for X", "type of X". Extract ONLY the short label (e.g. "Advertisement", "Food"), NOT surrounding prose.
9. For numeric fields (amount, cost, spent, remaining): extract the NUMBER only. Look for "amount of 75", "spent value of 67.81", "balance of 7.19" etc.
10. If a value is described as placeholder, missing, or 0.0 with a note it's inaccurate, use null.
11. If no records in this chunk, return: []

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
# Document chunking
# ---------------------------------------------------------------------------

def _chunk_document(text: str, max_chunk_chars: int = 3000) -> list[str]:
    """Split a markdown document into chunks, respecting structure."""
    sections = re.split(r"(?=^#{1,4}\s)", text, flags=re.MULTILINE)

    paragraphs: list[str] = []
    for section in sections:
        parts = [p.strip() for p in re.split(r"\n\s*\n", section)]
        for p in parts:
            if len(p) < 60:
                continue
            paragraphs.append(p)

    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chunk_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


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
) -> dict[str, Any] | None:
    """Planner analyzes doc and determines extraction schema."""
    sample = _get_sample_text(text)
    if log_fn:
        log_fn("planner_input", f"question={question[:120]}")
        log_fn("planner_input", f"knowledge_hint={knowledge_hint[:200]}")
        log_fn("planner_input", f"db_context={db_context[:200]}")
        log_fn("planner_input", f"sample_text (first 300 chars)={sample[:300]}")

    prompt = PLANNER_PROMPT.format(
        question=question,
        sample_text=sample,
        db_context=db_context,
        knowledge_hint=knowledge_hint,
    )
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
            # Ensure _id is always the first field
            fields = plan["fields"]
            if "_id" not in fields:
                fields.insert(0, "_id")
            elif fields[0] != "_id":
                fields.remove("_id")
                fields.insert(0, "_id")
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
            chunk_text=chunk[:3000],
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

    # First chunk sequentially to get an example
    if chunks:
        _, first_records = _extract_chunk((0, chunks[0]))
        if first_records:
            all_records.extend(first_records)
        else:
            failed_chunks.append(0)
        if log_fn:
            log_fn("worker_done", f"chunk 1/{len(chunks)}: {len(first_records)} records")

    # Remaining chunks in parallel
    remaining = [(i, chunk) for i, chunk in enumerate(chunks) if i > 0]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_extract_chunk, ic): ic for ic in remaining}
        for future in as_completed(futures):
            i, records = future.result()
            if records:
                all_records.extend(records)
            else:
                failed_chunks.append(i)
            if log_fn:
                log_fn("worker_done", f"chunk {i+1}/{len(chunks)}: {len(records)} records")

    if log_fn:
        log_fn("workers_complete", f"{len(all_records)} records, {len(failed_chunks)} empty chunks")

    # Normalize _id: if most IDs are numeric, strip surrounding text from non-numeric ones
    id_values = [str(r.get("_id", "")) for r in all_records if r.get("_id")]
    numeric_count = sum(1 for v in id_values if v.isdigit())
    if numeric_count > len(id_values) * 0.3:
        for r in all_records:
            rid = str(r.get("_id", ""))
            if not rid.isdigit():
                # Extract the numeric part
                nums = re.findall(r"\d+", rid)
                if nums:
                    r["_id"] = nums[-1]  # Use last number (most specific)
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
            chunk_text=chunk[:3000],
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

    chunks = _chunk_document(text)
    if not chunks:
        if log_fn:
            log_fn("chunk_skip", f"{doc_path.stem}: no paragraphs")
        return 0

    if log_fn:
        log_fn("orchestrator_start", f"{doc_path.stem}: {len(text)} chars, {len(chunks)} chunks")

    knowledge_hint = _build_knowledge_hint(question, knowledge_text)
    db_context = _get_existing_db_context(db_path)

    if log_fn:
        log_fn("orchestrator_context", f"knowledge_hint_len={len(knowledge_hint)}, db_context_len={len(db_context)}")

    # Agent 1: PLANNER
    if log_fn:
        log_fn("phase", "=== PLANNER PHASE ===")
    plan = _run_planner(model, text, question, db_context, knowledge_hint, log_fn)
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
    # Augment: add columns from the same entity section in knowledge.md
    # Only add columns that were defined under the same entity heading as the planner's entity
    if knowledge_text and plan.get("entity"):
        entity_name = plan["entity"].lower()
        plan_fields = set(plan.get("fields", []))
        # Parse knowledge.md for fields under the matching entity heading
        entity_section = ""
        for section in re.split(r"###\s+", knowledge_text):
            if entity_name in section[:50].lower():
                entity_section = section
                break
        if entity_section:
            # Extract field names from "- **field_name (...)**:" pattern
            knowledge_fields = re.findall(r"\*\*(\w+)\s*\(", entity_section)
            missing_added = []
            for col in knowledge_fields:
                if col not in plan_fields:
                    plan["fields"].append(col)
                    missing_added.append(col)
            if missing_added and log_fn:
                log_fn("planner_augment", f"Added from knowledge entity section: {missing_added}")

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
                    if v is not None:
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
    """Extract all docs using multi-agent approach."""
    if not doc_paths:
        return 0

    protected = {t.lower() for t in structured_tables} if structured_tables else set()
    total = 0

    for doc_path in doc_paths:
        n = chunked_extract(
            doc_path=doc_path,
            db_path=db_path,
            model=model,
            question=question,
            knowledge_text=knowledge_text,
            log_fn=log_fn,
            protected_tables=protected,
        )
        total += n

    if log_fn:
        log_fn("extraction_done", f"Total: {total} records from {len(doc_paths)} docs")

    return total
