from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.tools.grounding import EntitySchema, render_extraction_schema

CHUNK_SIZE = 12000
MAX_EXTRACTION_CHUNKS = 16
MAX_PARALLEL_WORKERS = 8

EXTRACTION_SYSTEM_PROMPT = (
    "You are a structured data extractor. You read prose and extract records."
)

EXTRACTION_USER_PROMPT = """
Extract ALL records from the text below that match the given schema.

SCHEMA:
{schema}

The question being investigated is: {question}

RULES:
- Return ONLY a JSON array of objects. No markdown, no explanation.
- Each object must have the field names from the schema as keys.
- CRITICAL: You MUST extract EVERY numeric, categorical, or date field
  mentioned in the text for each record, even if the field is NOT in the
  schema. Use the field abbreviation or name from the text as the JSON key.
  For example, if the text mentions a measurement called "CRE" at 1.5 mg/dL,
  include "CRE": 1.5 in that record's object. Missing fields = wrong answer.
- If a value was corrected (e.g. "initially 35.0 but corrected to 28.0"), use
  the CORRECTED/FINAL value (28.0).
- If a value is missing, unavailable, or NaN, use null.
- Parse dates into YYYY-MM-DD format when possible.
- Ignore filler information unrelated to the data records (hobbies, anecdotes,
  decorative details, etc.).
- If the text contains zero matching records, return an empty array: []

TEXT:
{text}
""".strip()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    chunks: list[str] = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, chunk_size)
        if cut <= 0:
            cut = chunk_size
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


def _parse_extraction_response(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    fence = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    else:
        generic = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
        if generic:
            raw = generic.group(1).strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in text.lower().split():
        w = raw.strip(".,;:!?()\"'`*_[]{}#/'")
        if len(w) >= 3:
            tokens.add(w)
    return tokens


def _score_chunk_relevance(chunk: str, question_keywords: set[str]) -> int:
    chunk_lower = chunk.lower()
    score = 0
    for kw in question_keywords:
        count = chunk_lower.count(kw)
        if count > 0:
            score += 1 + min(count // 5, 4)
    return score


_STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "how",
    "many",
    "what",
    "which",
    "who",
    "them",
    "that",
    "this",
    "with",
    "from",
    "not",
    "yet",
    "among",
    "their",
    "have",
    "has",
    "does",
    "did",
    "was",
    "were",
    "been",
    "isn",
    "aren",
    "don",
    "doesn",
}


def _extract_question_specific_keywords(
    question: str, schema_keywords: set[str]
) -> set[str]:
    return _tokenize(question) - _STOPWORDS - schema_keywords


def _find_id_key(records: list[dict[str, Any]]) -> str | None:
    """Find the most likely ID/primary key column for merging partial records."""
    if not records:
        return None
    candidates: list[str] = []
    for key in records[0]:
        key_lower = key.lower()
        if key_lower in ("id", "id_", "record_id", "entry_id", "ruling_id"):
            return key
        if key_lower.endswith("_id") or key_lower.endswith("id"):
            candidates.append(key)
    # Check for a column with unique integer-like values
    if not candidates:
        for key in records[0]:
            vals = [r.get(key) for r in records[:20] if r.get(key) is not None]
            if vals and all(
                isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit())
                for v in vals
            ):
                unique = len(set(str(v) for v in vals))
                if unique == len(vals):
                    candidates.append(key)
    return candidates[0] if candidates else None


def _merge_partial_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge records that share the same ID but have different fields populated."""
    if not records:
        return records

    id_key = _find_id_key(records)
    if not id_key:
        return records

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for r in records:
        id_val = r.get(id_key)
        if id_val is None:
            continue
        id_str = str(id_val)
        if id_str not in merged:
            merged[id_str] = {}
            order.append(id_str)
        existing = merged[id_str]
        for k, v in r.items():
            if v is not None and v != "" and v != "null":
                if k not in existing or existing[k] is None or existing[k] == "":
                    existing[k] = v

    if not merged:
        return records

    # Only use merged if it actually combined records (reduces count)
    merged_list = [merged[id_str] for id_str in order]
    if len(merged_list) < len(records) * 0.9:
        return merged_list
    return records


def extract_records_from_document(
    model: ModelAdapter,
    doc_text: str,
    schemas: list[EntitySchema],
    *,
    chunk_size: int = CHUNK_SIZE,
    max_chunks: int = MAX_EXTRACTION_CHUNKS,
    question: str = "",
) -> list[dict[str, Any]]:
    schema_text = render_extraction_schema(schemas)
    chunks = _chunk_text(doc_text, chunk_size)

    schema_keywords = set()
    for s in schemas:
        schema_keywords.update(_tokenize(s.entity_name))
        for f in s.fields:
            schema_keywords.update(_tokenize(f.name))

    specific_keywords = _extract_question_specific_keywords(question, schema_keywords)

    scored_chunks: list[tuple[float, int, str]] = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:
            continue
        specific_score = _score_chunk_relevance(chunk, specific_keywords) * 3
        schema_score = _score_chunk_relevance(chunk, schema_keywords)
        score = specific_score + schema_score
        scored_chunks.append((score, i, chunk))

    scored_chunks.sort(key=lambda x: (-x[0], x[1]))
    selected = scored_chunks[:max_chunks]
    selected.sort(key=lambda x: x[1])

    def _extract_one(chunk: str) -> list[dict[str, Any]]:
        prompt = EXTRACTION_USER_PROMPT.format(
            schema=schema_text, question=question, text=chunk
        )
        messages = [
            ModelMessage(role="system", content=EXTRACTION_SYSTEM_PROMPT),
            ModelMessage(role="user", content=prompt),
        ]
        try:
            raw = model.complete(messages, thinking=False)
            return _parse_extraction_response(raw)
        except Exception:
            return []

    workers = min(MAX_PARALLEL_WORKERS, len(selected))
    if workers <= 1:
        all_records: list[dict[str, Any]] = []
        for _score, _idx, chunk in selected:
            all_records.extend(_extract_one(chunk))
        return _merge_partial_records(all_records)

    all_records: list[dict[str, Any]] = []
    indexed_results: list[tuple[int, list[dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_order = {
            pool.submit(_extract_one, chunk): order
            for order, (_score, _idx, chunk) in enumerate(selected)
        }
        for future in as_completed(future_to_order):
            order = future_to_order[future]
            try:
                indexed_results.append((order, future.result()))
            except Exception:
                pass

    indexed_results.sort(key=lambda x: x[0])
    for _, records in indexed_results:
        all_records.extend(records)

    return _merge_partial_records(all_records)


def format_extracted_records_as_csv(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in records:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    lines = [",".join(all_keys)]
    for r in records:
        row = [str(r.get(k, "")) if r.get(k) is not None else "" for k in all_keys]
        escaped = []
        for v in row:
            if "," in v or '"' in v or "\n" in v:
                escaped.append('"' + v.replace('"', '""') + '"')
            else:
                escaped.append(v)
        lines.append(",".join(escaped))
    return "\n".join(lines)
