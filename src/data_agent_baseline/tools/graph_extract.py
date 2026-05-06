from __future__ import annotations

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage

CHUNK_SIZE = 12000
CHUNK_OVERLAP = 2000
MAX_PARALLEL_WORKERS = 8
MAX_ENTITY_EXTRACTION_TOKENS = 60000
MAX_DISCOVERY_CHUNKS = 4
MAX_EXTRACTION_CHUNKS = 8

# ---------------------------------------------------------------------------
# Pass 1: Entity & Schema Discovery
# ---------------------------------------------------------------------------

DISCOVERY_SYSTEM = "You are a document analyst. You identify entity types and their schemas for database table creation."

DISCOVERY_PROMPT = """
Analyze the text below and identify the MINIMAL set of entity types needed to
represent ALL structured data as database tables.

QUESTION (this tells you what data matters): {question}

RULES:
- MINIMIZE entity types. Prefer ONE table with many columns over many tables.
- If the same record is described across multiple sections with different
  attributes, that is ONE entity type with all those fields combined.
- Synonyms and aliases for the same concept = ONE entity type, not many.
- Only create a SEPARATE entity type if records have genuinely different primary
  keys and different schemas.
- Foreign keys (references to other entities' IDs) are ATTRIBUTES, not separate
  entity types.
- If a value was corrected in the text, the schema should hold the corrected value.
- Include ALL fields observed: IDs, foreign keys, categories, statuses, dates, etc.
- Name each entity type based on what the RECORDS in this document represent
  (e.g. "legality_entry", "ruling", "result"), NOT based on what they reference
  via foreign keys. The name should describe the rows in THIS document.

Return ONLY a JSON object (no markdown fences):
{{
  "entity_types": [
    {{
      "name": "table_name_for_database",
      "id_field": "primary_key_field_name",
      "attributes": [
        {{"name": "field_name", "type": "INTEGER|REAL|TEXT|DATE", "description": "what it represents"}}
      ],
      "relationships": [
        {{"field": "foreign_key_field", "target_entity": "other_table", "description": "what it references"}}
      ],
      "sample_ids": [1655, 5924, 15846]
    }}
  ]
}}

TEXT:
{text}
""".strip()

# ---------------------------------------------------------------------------
# Pass 2: Focused Entity Extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = "You are a precise data extractor. You produce complete structured records."

EXTRACTION_PROMPT = """
Extract ALL records of type "{entity_type}" from the text below.

SCHEMA:
- ID field: {id_field} ({id_type})
- Attributes: {attributes}
- Relationships: {relationships}

QUESTION: {question}

Return ONLY a JSON object (no markdown fences):
{{
  "records": [
    {{"<id_field>": <value>, "<attr1>": <value>, ...}},
    ...
  ],
  "edges": [
    {{"from_id": <entity_id>, "relation": "<relationship_name>", "to_entity": "<target_type>", "to_id": <target_id>}},
    ...
  ]
}}

RULES:
- Extract EVERY record. Do not skip any.
- Include ALL numeric, categorical, or date fields mentioned for each record,
  even if they are NOT listed in the schema above. Use the field abbreviation or
  name from the text as the JSON key (e.g. "CRE": 1.5, "GOT": 28.0).
- If a value was corrected, use the CORRECTED/FINAL value.
- If a value is missing or unknown, use null.
- Parse dates into YYYY-MM-DD when possible.
- Ignore decorative filler unrelated to the data.
- For relationships: if entity A references entity B's ID, create an edge.
- QUALITATIVE LABELS: For each measurement or assessment, if the text describes it
  using qualitative language, add a corresponding "_status" field with the exact
  FULL label from the text (include severity modifiers). Examples:
  - "significantly elevated" → record "significantly elevated" (not just "elevated")
  - "markedly impaired" → record "markedly impaired" (not just "impaired")
  - "normal", "within range", "unremarkable" → record it
  For instance, "CRE: 3.5 (abnormal)" → add "CRE_status": "abnormal".
  ONLY add _status when the descriptor appears in the SAME sentence/clause as that
  field's value. A general category description does NOT apply to every field —
  only the field the descriptor explicitly modifies. If no adjacent descriptor
  exists for a field, do NOT add _status for it.

TEXT:
{text}
""".strip()

# ---------------------------------------------------------------------------
# Chunking with overlap
# ---------------------------------------------------------------------------


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        cut = text.rfind("\n", start, end)
        if cut <= start:
            cut = end
        chunks.append(text[start:cut])
        start = cut - overlap
        if start < 0:
            start = 0
    return chunks


def _paragraph_split(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _strip_json_fence(raw: str) -> str:
    raw = raw.strip()
    fence = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    generic = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return raw


def _parse_json_response(raw: str) -> dict[str, Any]:
    try:
        return json.loads(_strip_json_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Pass 1: Discover entity types from document chunks
# ---------------------------------------------------------------------------


def _discover_entities_in_chunk(
    model: ModelAdapter, chunk: str, question: str
) -> list[dict[str, Any]]:
    prompt = DISCOVERY_PROMPT.format(text=chunk, question=question)
    messages = [
        ModelMessage(role="system", content=DISCOVERY_SYSTEM),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        raw = model.complete(messages)
        data = _parse_json_response(raw)
        return data.get("entity_types", [])
    except Exception:
        return []


def _select_discovery_chunks(chunks: list[str]) -> list[str]:
    """Select evenly-spaced chunks for schema discovery (first, middle, last, + one more)."""
    if len(chunks) <= MAX_DISCOVERY_CHUNKS:
        return chunks
    # Always include first and last; fill remaining with evenly-spaced
    indices = [0, len(chunks) - 1]
    remaining = MAX_DISCOVERY_CHUNKS - 2
    if remaining > 0:
        step = len(chunks) / (remaining + 1)
        for i in range(1, remaining + 1):
            idx = int(i * step)
            if idx not in indices:
                indices.append(idx)
    indices = sorted(set(indices))[:MAX_DISCOVERY_CHUNKS]
    return [chunks[i] for i in indices]


def _discover_all_entities(
    model: ModelAdapter, chunks: list[str], question: str
) -> list[dict[str, Any]]:
    """Run discovery on sampled chunks in parallel, merge entity type schemas."""
    sampled = _select_discovery_chunks(chunks)
    all_discoveries: list[list[dict[str, Any]]] = []

    workers = min(MAX_PARALLEL_WORKERS, len(sampled))
    if workers <= 1:
        for chunk in sampled:
            all_discoveries.append(_discover_entities_in_chunk(model, chunk, question))
    else:
        indexed: list[tuple[int, list[dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_discover_entities_in_chunk, model, chunk, question): i
                for i, chunk in enumerate(sampled)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    indexed.append((idx, future.result()))
                except Exception:
                    indexed.append((idx, []))
        indexed.sort(key=lambda x: x[0])
        all_discoveries = [d for _, d in indexed]

    return _merge_entity_schemas(all_discoveries)


def _normalize_entity_name(name: str) -> str:
    """Normalize entity type names for deduplication."""
    name = name.lower().strip().replace("-", "_").replace(" ", "_")
    # Remove common prefixes/suffixes that are just noise
    for prefix in ("card_", "portfolio_", "strategic_"):
        if name.startswith(prefix) and len(name) > len(prefix) + 2:
            name = name[len(prefix):]
    return name


def _names_are_similar(a: str, b: str) -> bool:
    """Check if two entity names likely refer to the same thing."""
    na = _normalize_entity_name(a)
    nb = _normalize_entity_name(b)
    if na == nb:
        return True
    # One is substring of other
    if na in nb or nb in na:
        return True
    # Singular/plural
    if na.rstrip("s") == nb.rstrip("s"):
        return True
    # Check word overlap
    words_a = set(na.split("_"))
    words_b = set(nb.split("_"))
    if words_a and words_b:
        overlap = len(words_a & words_b)
        if overlap >= max(1, min(len(words_a), len(words_b)) - 1):
            return True
    return False


def _merge_entity_schemas(
    all_discoveries: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Merge entity type definitions across chunks into unified schemas."""
    merged: dict[str, dict[str, Any]] = {}

    for chunk_entities in all_discoveries:
        for entity in chunk_entities:
            name = entity.get("name", "").lower().strip()
            if not name:
                continue

            # Find existing entry this should merge with
            target_key = None
            for existing_key in merged:
                if _names_are_similar(name, existing_key):
                    target_key = existing_key
                    break

            if target_key is None:
                target_key = _normalize_entity_name(name)
                merged[target_key] = {
                    "name": target_key,
                    "id_field": entity.get("id_field", "id"),
                    "attributes": {},
                    "relationships": {},
                    "sample_ids": set(),
                }

            existing = merged[target_key]

            # Merge attributes (keep unique by name)
            for attr in entity.get("attributes", []):
                attr_name = attr.get("name", "")
                if attr_name and attr_name not in existing["attributes"]:
                    existing["attributes"][attr_name] = attr

            # Merge relationships
            for rel in entity.get("relationships", []):
                rel_field = rel.get("field", "")
                if rel_field and rel_field not in existing["relationships"]:
                    existing["relationships"][rel_field] = rel

            # Merge sample IDs
            for sid in entity.get("sample_ids", []):
                if sid is not None:
                    existing["sample_ids"].add(sid)

    # If we still have multiple entity types that share the same ID field, merge them
    by_id_field: dict[str, list[str]] = {}
    for key, ent in merged.items():
        id_field = ent["id_field"]
        by_id_field.setdefault(id_field, []).append(key)

    final_merged: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    for id_field, keys in by_id_field.items():
        if len(keys) > 1:
            # Merge all entities sharing the same ID field into one
            primary = keys[0]
            for other in keys[1:]:
                for attr_name, attr in merged[other]["attributes"].items():
                    if attr_name not in merged[primary]["attributes"]:
                        merged[primary]["attributes"][attr_name] = attr
                for rel_field, rel in merged[other]["relationships"].items():
                    if rel_field not in merged[primary]["relationships"]:
                        merged[primary]["relationships"][rel_field] = rel
                merged[primary]["sample_ids"].update(merged[other]["sample_ids"])
                consumed.add(other)
            final_merged[primary] = merged[primary]
        else:
            final_merged[keys[0]] = merged[keys[0]]

    for key in merged:
        if key not in final_merged and key not in consumed:
            final_merged[key] = merged[key]

    # Convert to list format
    result: list[dict[str, Any]] = []
    for entity in final_merged.values():
        result.append({
            "name": entity["name"],
            "id_field": entity["id_field"],
            "attributes": list(entity["attributes"].values()),
            "relationships": list(entity["relationships"].values()),
            "sample_ids": sorted(entity["sample_ids"], key=lambda x: (isinstance(x, str), x))[:50],
        })
    return result


# ---------------------------------------------------------------------------
# Pass 2: Collect paragraphs mentioning each entity type
# ---------------------------------------------------------------------------


def _collect_paragraphs_for_entity(
    paragraphs: list[str],
    entity_schema: dict[str, Any],
) -> str:
    """Collect all paragraphs that mention this entity type's IDs or name."""
    entity_name = entity_schema["name"]
    id_field = entity_schema.get("id_field", "id")
    sample_ids = entity_schema.get("sample_ids", [])
    attr_names = [a.get("name", "") for a in entity_schema.get("attributes", [])]
    rel_fields = [r.get("field", "") for r in entity_schema.get("relationships", [])]

    # Build search patterns
    name_words = entity_name.replace("_", " ").split()
    name_patterns = [entity_name.replace("_", " "), entity_name.replace("_", "_")]
    name_patterns.extend(name_words)
    name_patterns = [p for p in name_patterns if len(p) >= 3]

    id_strs = [str(sid) for sid in sample_ids[:30]]

    # Also look for attribute/relationship field names (like "format", "status", "cards_id")
    field_patterns = [f.lower() for f in attr_names + rel_fields if len(f) >= 3]

    relevant: list[str] = []
    for para in paragraphs:
        para_lower = para.lower()
        # Check if paragraph mentions entity name
        if any(pat in para_lower for pat in name_patterns):
            relevant.append(para)
            continue
        # Check for specific IDs
        if any(id_str in para for id_str in id_strs):
            relevant.append(para)
            continue
        # Check for ID field references (e.g. "ID 1655", "ruling_id 1655")
        if re.search(r"\bID\s+\d+", para, re.IGNORECASE):
            relevant.append(para)
            continue
        # Check for attribute mentions (format, status, etc.)
        if any(pat in para_lower for pat in field_patterns):
            relevant.append(para)
            continue

    return "\n\n".join(relevant)


def _collect_all_text_for_entity(
    text: str,
    entity_schema: dict[str, Any],
    num_entity_types: int = 1,
) -> str:
    """Collect text relevant to this entity type. If only 1 entity type, use full text."""
    # If there's only one entity type, the whole doc is relevant
    if num_entity_types <= 1:
        return text

    paragraphs = _paragraph_split(text)
    collected = _collect_paragraphs_for_entity(paragraphs, entity_schema)
    # If we got very little or most of the doc matched, use the full text
    if len(collected) < 500 and len(text) > 500:
        return text
    if len(collected) > len(text) * 0.7:
        return text
    return collected


# ---------------------------------------------------------------------------
# Pass 3: Extract complete records for each entity type
# ---------------------------------------------------------------------------


def _format_attributes(attrs: list[dict[str, Any]]) -> str:
    if not attrs:
        return "(none discovered)"
    parts = []
    for a in attrs:
        parts.append(f"{a.get('name', '?')} ({a.get('type', 'TEXT')}): {a.get('description', '')}")
    return "; ".join(parts)


def _format_relationships(rels: list[dict[str, Any]]) -> str:
    if not rels:
        return "(none)"
    parts = []
    for r in rels:
        parts.append(f"{r.get('field', '?')} -> {r.get('target_entity', '?')}: {r.get('description', '')}")
    return "; ".join(parts)


def _extract_one_chunk(
    model: ModelAdapter,
    chunk: str,
    entity_schema: dict[str, Any],
    question: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract records and edges from a single chunk."""
    prompt = EXTRACTION_PROMPT.format(
        entity_type=entity_schema["name"],
        id_field=entity_schema["id_field"],
        id_type="INTEGER",
        attributes=_format_attributes(entity_schema.get("attributes", [])),
        relationships=_format_relationships(entity_schema.get("relationships", [])),
        question=question,
        text=chunk,
    )
    messages = [
        ModelMessage(role="system", content=EXTRACTION_SYSTEM),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        raw = model.complete(messages)
        data = _parse_json_response(raw)
        records = data.get("records", [])
        edges = data.get("edges", [])
        r_list = [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []
        e_list = [e for e in edges if isinstance(e, dict)] if isinstance(edges, list) else []
        return r_list, e_list
    except Exception:
        return [], []


def _extract_entity_records(
    model: ModelAdapter,
    text_for_entity: str,
    entity_schema: dict[str, Any],
    question: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract records and edges for one entity type. Returns (records, edges).

    For large documents (>60K chars), uses record-segmented extraction which
    splits by record boundaries and processes in small batches for complete coverage.
    For smaller documents, uses the original chunk-based approach.
    """
    # For large docs, use segmented extraction for better record completeness
    if len(text_for_entity) > MAX_ENTITY_EXTRACTION_TOKENS:
        records = _extract_segmented(model, text_for_entity, entity_schema, question)
        return records, []

    # Small docs: single-chunk extraction
    all_records: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    recs, edges = _extract_one_chunk(model, text_for_entity, entity_schema, question)
    all_records.extend(recs)
    all_edges.extend(edges)

    merged_records = _resolve_entities(all_records, entity_schema["id_field"])
    return merged_records, all_edges


def _resolve_entities(
    records: list[dict[str, Any]], id_field: str
) -> list[dict[str, Any]]:
    """Deduplicate and merge records sharing the same composite key.

    Uses (id_field + Date) as composite key when records have a Date/date field,
    allowing multiple time-series records per entity. Falls back to id_field alone
    when no date field exists.
    """
    if not records:
        return records

    # Detect date field for composite key
    date_field = None
    for r in records[:10]:
        for k in r:
            if k.lower() in ("date", "examination_date", "test_date"):
                date_field = k
                break
        if date_field:
            break

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for r in records:
        id_val = r.get(id_field)
        if id_val is None:
            continue
        # Composite key: id + date (if available)
        if date_field and r.get(date_field):
            key = f"{id_val}|{r[date_field]}"
        else:
            key = str(id_val)
        if key not in merged:
            merged[key] = {}
            order.append(key)
        existing = merged[key]
        for k, v in r.items():
            if v is not None and v != "" and v != "null":
                existing[k] = v

    if not merged:
        return records
    return [merged[key] for key in order]


# ---------------------------------------------------------------------------
# Record-segmented extraction (for large docs with many records)
# ---------------------------------------------------------------------------

RECORDS_PER_BATCH = 15
MAX_BATCH_CHARS = 30000

SEGMENT_EXTRACTION_PROMPT = """
Extract structured records from the text segments below.

Each segment describes one or more records of type "{entity_type}".
Extract the ID and ALL numeric, categorical, date, or text fields for each record.

SCHEMA HINT (may be incomplete — extract all fields you see, not just these):
- ID field: {id_field}
- Known attributes: {attributes}

QUESTION (tells you what data matters): {question}

Return ONLY a JSON object (no markdown fences):
{{
  "records": [
    {{"{id_field}": <value>, "<field1>": <value>, "<field2>": <value>, ...}},
    ...
  ]
}}

RULES:
- Extract EVERY record mentioned in the segments below.
- If a value was corrected in the text, use the CORRECTED/FINAL value only.
- If a value is missing or unknown, use null.
- Parse dates into YYYY-MM-DD when possible.
- Include ALL fields mentioned for each record, even if not in the schema above.
- Use SHORT ABBREVIATIONS for field names consistently (e.g. "CRE" not "Creatinine",
  "GOT" not "glutamic_oxaloacetic_transaminase", "UA" not "uric_acid"). If the text
  provides an abbreviation in parentheses like "creatinine (CRE)", use "CRE".
- QUALITATIVE LABELS: For each measurement or assessment, if the text describes it
  using qualitative language, add a corresponding "_status" field with the exact
  label from the text. Examples of labels to capture:
  - "abnormal", "elevated", "impaired", "compromised", "markedly impaired",
    "beyond normal", "high", "low", "deficient", "critical" → record the label
  - "normal", "healthy", "within range", "unremarkable", "adequate", "stable" → record the label
  For instance, if "renal function: markedly impaired" appears, add
  "renal_function_status": "markedly impaired". If "CRE: 3.5 (abnormal)",
  add "CRE_status": "abnormal".
  STRICT RULES FOR _status FIELDS:
  - ONLY add a _status field when the descriptor appears in the SAME sentence or
    clause as that specific field's value.
  - Do NOT infer status from numbers alone.
  - A general description of a category (e.g. "compromised profile") does NOT
    apply to every individual field — only attach _status to the field that the
    descriptor explicitly modifies.
  - If a field has a value but NO adjacent qualitative descriptor, do NOT add
    a _status for it.
  - Capture the FULL qualifying phrase as the label, including severity modifiers.
    E.g. "significantly elevated" not just "elevated", "markedly impaired" not
    just "impaired".

TEXT SEGMENTS:
{text}
""".strip()


def _segment_by_records(text: str, id_field: str, sample_ids: list | None = None) -> list[str]:
    """Split text into segments, each containing one logical record.

    Splits on paragraph boundaries, then groups paragraphs that belong to the
    same record (detected by ID mentions or continuation patterns).
    """
    paragraphs = _paragraph_split(text)
    if not paragraphs:
        return []

    # Each segment is one or more consecutive paragraphs about a record
    segments: list[str] = []
    current: list[str] = []

    # Build ID pattern from sample IDs if available (handles alphanumeric IDs)
    id_patterns: list[re.Pattern] = []

    if sample_ids:
        str_ids = [str(s) for s in sample_ids if s is not None]
        # Detect alphanumeric prefix pattern (e.g. rec[A-Za-z0-9]+)
        prefixes: dict[str, int] = {}
        for sid in str_ids:
            m = re.match(r"^([a-zA-Z]+)", sid)
            if m and len(m.group(1)) >= 2 and not sid.isdigit():
                prefix = m.group(1)
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
        if prefixes:
            top_prefix = max(prefixes, key=lambda k: prefixes[k])
            if prefixes[top_prefix] >= 2:
                # Require mixed case/digits after prefix to avoid matching English words
                suffix_len = max(len(s) - len(top_prefix) for s in str_ids if s.startswith(top_prefix))
                min_suffix = max(5, suffix_len - 3)
                # Use lookahead to require at least one digit OR one uppercase after prefix
                id_patterns.append(re.compile(
                    r"\b" + re.escape(top_prefix)
                    + r"(?=[A-Za-z0-9]*[A-Z0-9])[A-Za-z0-9]{"
                    + str(min_suffix) + r",}"
                ))

    # Fallback numeric patterns
    id_patterns.extend([
        re.compile(r"\bpatient\s+(\d{3,})", re.IGNORECASE),
        re.compile(r"\bID\s+(\d{3,})", re.IGNORECASE),
        re.compile(r"\b(?:entry|record|ruling|asset|unit)\s+(?:ID\s+)?(\d{3,})", re.IGNORECASE),
        re.compile(r"\b(?:Medical Record Number|MRN)\s+(\d{3,})", re.IGNORECASE),
        re.compile(r"\bcataloged under ID\s+(\d+)", re.IGNORECASE),
        re.compile(r"\bregistered (?:as|under) ID\s+(\d+)", re.IGNORECASE),
    ])

    def _starts_new_record(para: str) -> bool:
        for pat in id_patterns:
            m = pat.search(para[:300])
            if m:
                return True
        return False

    for para in paragraphs:
        if _starts_new_record(para) and current:
            segments.append("\n\n".join(current))
            current = [para]
        else:
            current.append(para)

    if current:
        segments.append("\n\n".join(current))

    return segments


def _batch_segments(segments: list[str], max_chars: int = MAX_BATCH_CHARS, max_count: int = RECORDS_PER_BATCH) -> list[str]:
    """Group segments into batches that fit within token/count limits."""
    batches: list[str] = []
    current_batch: list[str] = []
    current_size = 0

    for seg in segments:
        seg_size = len(seg)
        if current_batch and (current_size + seg_size > max_chars or len(current_batch) >= max_count):
            batches.append("\n\n---\n\n".join(current_batch))
            current_batch = [seg]
            current_size = seg_size
        else:
            current_batch.append(seg)
            current_size += seg_size

    if current_batch:
        batches.append("\n\n---\n\n".join(current_batch))

    return batches


def _extract_batch(
    model: ModelAdapter,
    batch_text: str,
    entity_schema: dict[str, Any],
    question: str,
) -> list[dict[str, Any]]:
    """Extract records from a single batch of segments."""
    import time

    prompt = SEGMENT_EXTRACTION_PROMPT.format(
        entity_type=entity_schema["name"],
        id_field=entity_schema["id_field"],
        attributes=_format_attributes(entity_schema.get("attributes", [])),
        question=question,
        text=batch_text,
    )
    messages = [
        ModelMessage(role="system", content=EXTRACTION_SYSTEM),
        ModelMessage(role="user", content=prompt),
    ]
    for attempt in range(3):
        try:
            raw = model.complete(messages)
            data = _parse_json_response(raw)
            records = data.get("records", [])
            return [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return []


MAX_SEGMENTED_WORKERS = 4


def _normalize_field_names(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize field names across records to use consistent short names.

    When the same concept appears with different names (e.g. "Creatinine" vs "CRE"),
    collapse them to the shortest version that appears.
    """
    if not records:
        return records

    # Collect all field names and group by lowercase
    name_groups: dict[str, list[str]] = {}
    for r in records:
        for k in r:
            lower = k.lower().replace("_", "").replace("-", "").replace(" ", "")
            name_groups.setdefault(lower, []).append(k)

    # For each group, pick the shortest name (abbreviation preferred)
    canonical: dict[str, str] = {}
    for lower, names in name_groups.items():
        shortest = min(set(names), key=len)
        for n in set(names):
            canonical[n] = shortest

    # Also handle known abbreviation patterns
    abbrev_map = {
        "creatinine": "CRE", "glutamicoxaloaceticTransaminase": "GOT",
        "glutamicpyruvictransaminase": "GPT", "uricacid": "UA",
        "ureanitrogen": "UN", "lactatedehydrogenase": "LDH",
        "alkalinephosphatase": "ALP", "totalbilirubin": "T-BIL",
        "totalprotein": "TP", "albumin": "ALB",
    }
    for r in records:
        for k in list(r.keys()):
            norm = k.lower().replace("_", "").replace("-", "").replace(" ", "")
            if norm in abbrev_map:
                canonical[k] = abbrev_map[norm]

    # Apply canonicalization
    result = []
    for r in records:
        new_r: dict[str, Any] = {}
        for k, v in r.items():
            new_key = canonical.get(k, k)
            if new_key in new_r and (new_r[new_key] is None or new_r[new_key] == ""):
                new_r[new_key] = v
            elif new_key not in new_r:
                new_r[new_key] = v
        result.append(new_r)
    return result


def _extract_segmented(
    model: ModelAdapter,
    text: str,
    entity_schema: dict[str, Any],
    question: str,
) -> list[dict[str, Any]]:
    """Record-segmented extraction: split by record boundaries, batch, parallel extract."""
    segments = _segment_by_records(text, entity_schema["id_field"], entity_schema.get("sample_ids"))
    if not segments:
        return []

    batches = _batch_segments(segments)
    if not batches:
        return []

    all_records: list[dict[str, Any]] = []
    workers = min(MAX_SEGMENTED_WORKERS, len(batches))

    if workers <= 1:
        for batch in batches:
            all_records.extend(_extract_batch(model, batch, entity_schema, question))
    else:
        indexed: list[tuple[int, list[dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_extract_batch, model, batch, entity_schema, question): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    indexed.append((idx, future.result()))
                except Exception:
                    indexed.append((idx, []))
        indexed.sort(key=lambda x: x[0])
        for _, recs in indexed:
            all_records.extend(recs)

    normalized = _normalize_field_names(all_records)
    resolved = _resolve_entities(normalized, entity_schema["id_field"])
    return _repair_null_values(model, text, resolved, entity_schema, question)


# ---------------------------------------------------------------------------
# NULL value repair pass
# ---------------------------------------------------------------------------

NULL_REPAIR_PROMPT = """
A prior extraction pass found records with missing values. Search the FULL text
below and fill in the missing fields for these specific records.

RECORDS WITH MISSING DATA (fill in NULL fields):
{records_json}

SCHEMA:
- ID field: {id_field}
- Known attributes: {attributes}

Return ONLY a JSON object (no markdown fences):
{{
  "records": [
    {{"{id_field}": <id_value>, "<field>": <filled_value>, ...}},
    ...
  ]
}}

RULES:
- Only return records that you found additional data for.
- For each record, include the ID and ONLY the fields you found values for.
- The data may be scattered across different sections — search the ENTIRE text.
- If a value was corrected in the text, use the CORRECTED/FINAL value.
- If you truly cannot find a value, omit that record from the output.

TEXT:
{text}
""".strip()


def _repair_null_values(
    model: ModelAdapter,
    text: str,
    records: list[dict[str, Any]],
    entity_schema: dict[str, Any],
    question: str,
) -> list[dict[str, Any]]:
    """Second pass: find and fill NULL values in extracted records."""
    if not records:
        return records

    id_field = entity_schema["id_field"]

    # Find records with NULL in non-ID, non-status fields
    incomplete: list[dict[str, Any]] = []
    for r in records:
        null_fields = [
            k for k, v in r.items()
            if v is None and k != id_field and not k.endswith("_status")
        ]
        if null_fields and r.get(id_field) is not None:
            incomplete.append(r)

    if not incomplete or len(incomplete) > 50:
        return records

    # Build a compact representation of what's missing
    missing_info = []
    for r in incomplete:
        entry = {id_field: r[id_field]}
        for k, v in r.items():
            if v is None and k != id_field and not k.endswith("_status"):
                entry[k] = "NULL — FILL THIS"
            elif v is not None:
                entry[k] = v
        missing_info.append(entry)

    # Limit text to avoid token overflow — take only paragraphs mentioning missing IDs
    missing_ids = {str(r[id_field]) for r in incomplete}
    paragraphs = _paragraph_split(text)
    relevant_paras = [p for p in paragraphs if any(mid in p for mid in missing_ids)]
    if not relevant_paras:
        return records
    repair_text = "\n\n".join(relevant_paras)
    if len(repair_text) > MAX_BATCH_CHARS * 2:
        repair_text = repair_text[:MAX_BATCH_CHARS * 2]

    prompt = NULL_REPAIR_PROMPT.format(
        records_json=json.dumps(missing_info, ensure_ascii=False, indent=2),
        id_field=id_field,
        attributes=_format_attributes(entity_schema.get("attributes", [])),
        text=repair_text,
    )
    messages = [
        ModelMessage(role="system", content=EXTRACTION_SYSTEM),
        ModelMessage(role="user", content=prompt),
    ]

    try:
        raw = model.complete(messages)
        data = _parse_json_response(raw)
        repairs = data.get("records", [])
        if not isinstance(repairs, list):
            return records
    except Exception:
        return records

    # Apply repairs
    repair_map: dict[str, dict[str, Any]] = {}
    for r in repairs:
        if isinstance(r, dict) and r.get(id_field) is not None:
            repair_map[str(r[id_field])] = r

    if not repair_map:
        return records

    result = []
    for r in records:
        rid = str(r.get(id_field, ""))
        if rid in repair_map:
            patched = dict(r)
            for k, v in repair_map[rid].items():
                if k != id_field and v is not None and patched.get(k) is None:
                    patched[k] = v
            result.append(patched)
        else:
            result.append(r)

    return result


# ---------------------------------------------------------------------------
# Write to SQLite
# ---------------------------------------------------------------------------


def _infer_sqlite_type(val: Any) -> str:
    if isinstance(val, bool):
        return "INTEGER"
    if isinstance(val, int):
        return "INTEGER"
    if isinstance(val, float):
        return "REAL"
    return "TEXT"


def _infer_column_types_from_records(
    records: list[dict[str, Any]], columns: list[str]
) -> dict[str, str]:
    """Infer SQLite column types from actual extracted values."""
    types: dict[str, str] = {col: "INTEGER" for col in columns}

    for r in records[:100]:
        for col in columns:
            if types[col] == "TEXT":
                continue
            val = r.get(col)
            if val is None or val == "":
                continue
            if isinstance(val, bool):
                types[col] = "INTEGER"
            elif isinstance(val, int):
                pass  # stays INTEGER
            elif isinstance(val, float):
                if types[col] == "INTEGER":
                    types[col] = "REAL"
            elif isinstance(val, str):
                # Try to parse as number
                try:
                    int(val)
                    continue
                except ValueError:
                    pass
                try:
                    float(val)
                    if types[col] == "INTEGER":
                        types[col] = "REAL"
                    continue
                except ValueError:
                    pass
                types[col] = "TEXT"
            else:
                types[col] = "TEXT"
    return types


def _cast_value(val: Any, col_type: str) -> Any:
    if val is None or val == "" or val == "null":
        return None
    if col_type == "INTEGER":
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                try:
                    return int(float(val))
                except ValueError:
                    return val
    elif col_type == "REAL":
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return str(val) if val is not None else None


def _write_entity_table(
    conn: sqlite3.Connection,
    table_name: str,
    records: list[dict[str, Any]],
) -> bool:
    """Write extracted records as a SQLite table."""
    if not records:
        return False

    # Discover all columns across records
    columns: list[str] = []
    seen: set[str] = set()
    for r in records:
        for k in r:
            if k not in seen:
                columns.append(k)
                seen.add(k)

    if not columns:
        return False

    col_types = _infer_column_types_from_records(records, columns)

    cols_def = ", ".join(f'"{c}" {col_types[c]}' for c in columns)
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(f'CREATE TABLE "{table_name}" ({cols_def})')

    placeholders = ", ".join(["?"] * len(columns))
    batch: list[tuple[Any, ...]] = []
    for r in records:
        row = tuple(_cast_value(r.get(c), col_types[c]) for c in columns)
        batch.append(row)
        if len(batch) >= 1000:
            conn.executemany(f'INSERT INTO "{table_name}" VALUES ({placeholders})', batch)
            batch.clear()
    if batch:
        conn.executemany(f'INSERT INTO "{table_name}" VALUES ({placeholders})', batch)
    conn.commit()
    return True


def _clean_orphaned_status(conn: sqlite3.Connection, table_name: str) -> None:
    """Clean up hallucinated data: orphaned statuses and data-void records.

    1. Nullify _status fields where the corresponding value field is NULL.
    2. Nullify all non-ID fields in rows where most value columns are NULL
       (data void records that shouldn't have any extracted values).
    """
    try:
        cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
        columns = [r[1] for r in cursor.fetchall()]
    except Exception:
        return

    # Pass 1: Orphaned status cleanup
    status_cols = [c for c in columns if c.endswith("_status")]
    for status_col in status_cols:
        base = status_col.removesuffix("_status")
        if base in columns:
            conn.execute(
                f'UPDATE "{table_name}" SET "{status_col}" = NULL '
                f'WHERE "{base}" IS NULL AND "{status_col}" IS NOT NULL'
            )

    # Pass 2: Data void cleanup — if a row has >= 70% NULL value columns,
    # it's a void record; nullify any stray non-NULL values (likely hallucinated)
    # Detect ID columns: first column, anything ending in _id, or named "id"/"date"
    id_and_date_cols = {columns[0]} if columns else set()
    id_and_date_cols |= {c for c in columns if c.lower() in ("id", "date", "birthday")}
    id_and_date_cols |= {c for c in columns if c.lower().endswith("_id") or c.lower() == "record_id"}
    meta_cols = id_and_date_cols | {c for c in columns if c.endswith("_status") or c.endswith("_abnormal")}
    value_cols = [c for c in columns if c not in meta_cols]

    if len(value_cols) >= 3:
        # Build a CASE expression that counts NULLs among value columns
        null_count_expr = " + ".join(
            f'CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END' for c in value_cols
        )
        # Only null out rows where ALL value columns are NULL (true data voids)
        threshold = len(value_cols)

        # For data-void rows, null out status and value columns (keep ID/date)
        cols_to_null = [c for c in columns if c not in id_and_date_cols]
        set_clause = ", ".join(f'"{c}" = NULL' for c in cols_to_null)
        conn.execute(
            f'UPDATE "{table_name}" SET {set_clause} '
            f'WHERE ({null_count_expr}) >= {threshold}'
        )

    conn.commit()


def _write_edges_table(
    conn: sqlite3.Connection,
    edges: list[dict[str, Any]],
    source_entity: str,
) -> bool:
    """Append edges to the _edges table."""
    if not edges:
        return False

    conn.execute("""
        CREATE TABLE IF NOT EXISTS "_edges" (
            from_entity TEXT,
            from_id TEXT,
            relation TEXT,
            to_entity TEXT,
            to_id TEXT
        )
    """)

    batch: list[tuple[str, str, str, str, str]] = []
    for e in edges:
        from_id = e.get("from_id")
        relation = e.get("relation", "")
        to_entity = e.get("to_entity", "")
        to_id = e.get("to_id")
        if from_id is not None and to_id is not None:
            batch.append((source_entity, str(from_id), relation, to_entity, str(to_id)))

    if batch:
        conn.executemany('INSERT INTO "_edges" VALUES (?, ?, ?, ?, ?)', batch)
        conn.commit()
    return bool(batch)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def extract_graph_to_sqlite(
    model: ModelAdapter,
    doc_text: str,
    question: str,
    db_path: Path,
    *,
    doc_name: str = "document",
    known_schemas: list[dict[str, Any]] | None = None,
) -> list[str]:
    """
    Extract structured data from unstructured text using graph-based extraction
    and write directly into the SQLite database.

    If known_schemas is provided, skips the discovery phase entirely.
    Returns list of table names created.
    """
    results = _extract_text_to_memory(model, doc_text, question, known_schemas=known_schemas)
    if not results:
        return []

    tables_created: list[str] = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=DELETE")
    except Exception:
        return []

    try:
        for table_name, entity_name, records, edges in results:
            if records and _write_entity_table(conn, table_name, records):
                _clean_orphaned_status(conn, table_name)
                tables_created.append(table_name)
            _write_edges_table(conn, edges, entity_name)
        conn.close()
    except Exception:
        conn.close()

    return tables_created


def _entity_schemas_from_grounding(
    grounding_schemas: list[Any],
) -> list[dict[str, Any]] | None:
    """Convert EntitySchema objects from grounding.py into the dict format used here.

    Returns None if schemas are too sparse to be useful (triggers discovery fallback).
    """
    result: list[dict[str, Any]] = []
    for schema in grounding_schemas:
        entity_name = schema.entity_name if hasattr(schema, "entity_name") else str(schema)
        fields = schema.fields if hasattr(schema, "fields") else []
        attrs = []
        id_field = "id"
        has_explicit_id = False
        for f in fields:
            name = f.name if hasattr(f, "name") else str(f)
            ftype = f.field_type if hasattr(f, "field_type") else "TEXT"
            desc = f.description if hasattr(f, "description") else ""
            attrs.append({"name": name, "type": ftype.upper(), "description": desc})
            if name.lower() in ("id", "record_id", "entry_id") or name.lower().endswith("_id"):
                id_field = name
                has_explicit_id = True
        # Skip schemas that are too sparse — they'll produce incomplete extractions
        if not has_explicit_id or len(attrs) < 3:
            continue
        result.append({
            "name": entity_name.lower().replace(" ", "_"),
            "id_field": id_field,
            "attributes": attrs,
            "relationships": [],
            "sample_ids": [],
        })
    # If no schemas survived validation, return None to trigger discovery
    return result if result else None


def _extract_text_to_memory(
    model: ModelAdapter,
    doc_text: str,
    question: str,
    known_schemas: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]]:
    """Extract graph from text, return results in memory (no DB writes).

    If known_schemas is provided, skip discovery (saves many LLM calls).
    All LLM calls (discovery + extraction) are parallelized.
    """
    if len(doc_text.strip()) < 100:
        return []

    # Use provided schemas or discover them
    if known_schemas:
        entity_schemas = known_schemas
    else:
        discovery_chunks = _chunk_text(doc_text)
        entity_schemas = _discover_all_entities(model, discovery_chunks, question)
    if not entity_schemas:
        return []

    entity_tasks: list[tuple[str, str, dict[str, Any]]] = []
    for entity_schema in entity_schemas:
        entity_name = entity_schema["name"]
        table_name = re.sub(r"[^a-zA-Z0-9_]", "_", f"_extracted_{entity_name}")
        if known_schemas:
            entity_text = doc_text
        else:
            entity_text = _collect_all_text_for_entity(
                doc_text, entity_schema, num_entity_types=len(entity_schemas)
            )
        if len(entity_text.strip()) < 50:
            continue
        entity_tasks.append((table_name, entity_text, entity_schema))

    results: list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]] = []

    def _do_extraction(args: tuple[str, str, dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
        tbl_name, ent_text, ent_schema = args
        recs, edges = _extract_entity_records(model, ent_text, ent_schema, question)
        return tbl_name, ent_schema["name"], recs, edges

    workers = min(MAX_PARALLEL_WORKERS, len(entity_tasks))
    if workers <= 1:
        for args in entity_tasks:
            results.append(_do_extraction(args))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_do_extraction, a): i for i, a in enumerate(entity_tasks)}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass

    return results


def _match_schemas_to_doc(
    doc_path: Path, known_schemas: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """Match known schemas to a doc by filename. Returns matching schemas or all if no match."""
    doc_name = doc_path.stem.lower().replace("-", "_").replace(" ", "_")
    matched = [
        s for s in known_schemas
        if s["name"].lower() in doc_name or doc_name in s["name"].lower()
    ]
    if matched:
        return matched
    return known_schemas


def extract_multiple_docs_to_sqlite(
    model: ModelAdapter,
    doc_paths: list[Path],
    question: str,
    db_path: Path,
    *,
    known_schemas: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Extract graph data from multiple documents into the same SQLite DB.

    If known_schemas is provided, skips discovery (saves many LLM calls).
    Extraction is parallelized across docs; DB writes are sequential to avoid locks.
    """
    if not doc_paths:
        return []

    def _read_and_extract(doc_path: Path) -> list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]]:
        try:
            doc_text = doc_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        doc_schemas = _match_schemas_to_doc(doc_path, known_schemas) if known_schemas else None
        return _extract_text_to_memory(model, doc_text, question, known_schemas=doc_schemas)

    # Parallel extraction across docs (no DB access)
    all_doc_results: list[list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]]] = []

    if len(doc_paths) == 1:
        all_doc_results.append(_read_and_extract(doc_paths[0]))
    else:
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(doc_paths))) as pool:
            futures = {pool.submit(_read_and_extract, p): i for i, p in enumerate(doc_paths)}
            indexed: list[tuple[int, list]] = []
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    indexed.append((idx, future.result()))
                except Exception:
                    indexed.append((idx, []))
            indexed.sort(key=lambda x: x[0])
            all_doc_results = [r for _, r in indexed]

    # Sequential DB writes (thread-safe)
    tables_created: list[str] = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=DELETE")
    except Exception:
        return []

    try:
        for doc_results in all_doc_results:
            for table_name, entity_name, records, edges in doc_results:
                if records and _write_entity_table(conn, table_name, records):
                    _clean_orphaned_status(conn, table_name)
                    tables_created.append(table_name)
                _write_edges_table(conn, edges, entity_name)
        conn.close()
    except Exception:
        conn.close()

    return tables_created
