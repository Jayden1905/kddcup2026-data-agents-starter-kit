"""Adaptive pattern extraction pipeline.

Architecture:
  1. PATTERN DISCOVERY (1 LLM call) — stratified sample of paragraphs →
     LLM identifies extraction patterns (regex-like templates for each field)
  2. DETERMINISTIC EXTRACTION (0 LLM calls) — apply patterns to ALL paragraphs
  3. COVERAGE CHECK — if extraction coverage < threshold, sample unmatched
     paragraphs and discover additional patterns (1 more LLM call)
  4. Repeat until convergence or max iterations

Result: 1-3 LLM calls total for any document size.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PATTERN_DISCOVERY_PROMPT = """Analyze these sample paragraphs and identify extraction PATTERNS.

QUESTION context: {question}
ENTITY TYPE: {entity}
FIELDS TO EXTRACT: {fields}

{db_context}

SAMPLE PARAGRAPHS (each between --- separators):
---
{samples}
---

Your job: find the SPECIFIC LANGUAGE PATTERNS that introduce each field value in the text.

EXAMPLE — for a document about patients with lab results:
Text: "patient 3182521, whose GOT was initially 35.0 U/L; corrected to 28.0 U/L. GPT level at 23.0 U/L."
Good patterns:
{{
  "id_pattern": "patient\\\\s+(\\\\d+)",
  "field_patterns": {{
    "GOT": {{"pattern": "GOT[^0-9]+(\\\\d+\\\\.?\\\\d*)", "type": "number", "correction_signal": "corrected(?:\\\\s+\\\\w+)*?\\\\s+(\\\\d+\\\\.?\\\\d*)", "null_signals": ["not available", "NaN"]}},
    "GPT": {{"pattern": "GPT[^0-9]+(\\\\d+\\\\.?\\\\d*)", "type": "number", "correction_signal": null, "null_signals": ["not available"]}}
  }}
}}

Return a JSON object:
{{
  "id_pattern": "regex with ONE capture group for entity ID",
  "field_patterns": {{
    "field_name": {{
      "pattern": "regex with ONE capture group — must include a KEYWORD ANCHOR specific to this field",
      "type": "text|number|date|category",
      "correction_signal": "regex to capture the CORRECTED value when text has corrections (or null)",
      "null_signals": ["phrases that mean this field is missing"]
    }}
  }}
}}

CRITICAL RULES:
1. Every pattern MUST start with a KEYWORD ANCHOR — a word or phrase that ONLY appears near that field's value. E.g. "GOT" for GOT field, "categorized as" for type/category field, "amount of" for amount field.
2. WITHOUT a keyword anchor, patterns will grab values from wrong sentences. Never use generic patterns like "(.+)" or "(\\d+)".
3. The capture group must grab ONLY the clean value — not surrounding prose.
4. For CATEGORY fields (type, classification, designation): pattern should capture the category LABEL that follows a signal phrase. E.g. "(?:categorized as|classified as|designated for|category of)\\\\s+([A-Z][a-z]+(?:\\\\s[A-Z][a-z]+)*)"
5. For NUMBER fields: anchor + number capture. E.g. "amount of\\\\s+(\\\\d+\\\\.?\\\\d*)"
6. For CORRECTIONS: the correction_signal regex captures the FINAL value after phrases like "corrected to", "amended to", "updated to", "confirmed at".
7. Only include fields that ACTUALLY appear in the sample text. If a field from the FIELDS list is not present in ANY sample, omit it.
8. Test mentally: would your pattern extract the RIGHT value from each sample?

Return ONLY the JSON object."""

PATTERN_REFINEMENT_PROMPT = """Some paragraphs did NOT match the extraction patterns.

CURRENT PATTERNS:
{current_patterns}

UNMATCHED PARAGRAPHS (patterns failed to extract _id or most fields):
---
{unmatched_samples}
---

Either these paragraphs have a DIFFERENT structure, or the patterns need adjustment.

Return a JSON object with:
{{
  "additional_patterns": [
    {{
      "id_pattern": "new regex for ID in these paragraphs",
      "field_patterns": {{...same format as before...}},
      "applies_when": "condition to detect this variant (e.g. 'paragraph contains X')"
    }}
  ],
  "pattern_fixes": {{
    "field_name": "corrected regex if original was too strict (or null if fine)"
  }}
}}

Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Pattern application engine (deterministic)
# ---------------------------------------------------------------------------

_DATE_WORDS = {
    "first": "01", "second": "02", "third": "03", "fourth": "04",
    "fifth": "05", "sixth": "06", "seventh": "07", "eighth": "08",
    "ninth": "09", "tenth": "10", "eleventh": "11", "twelfth": "12",
    "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
    "nineteenth": "19", "twentieth": "20", "twenty-first": "21",
    "twenty-second": "22", "twenty-third": "23", "twenty-fourth": "24",
    "twenty-fifth": "25", "twenty-sixth": "26", "twenty-seventh": "27",
    "twenty-eighth": "28", "twenty-ninth": "29", "thirtieth": "30",
    "thirty-first": "31",
}

_MONTH_WORDS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def _parse_prose_date(text: str) -> str | None:
    """Convert prose date like 'tenth of February, 1986' to ISO format."""
    text_lower = text.lower().strip()
    # Try: "Xth of Month, Year" or "Month Xth, Year"
    for day_word, day_num in _DATE_WORDS.items():
        if day_word in text_lower:
            for month_word, month_num in _MONTH_WORDS.items():
                if month_word in text_lower:
                    year_match = re.search(r"\b(\d{4})\b", text)
                    if year_match:
                        return f"{year_match.group(1)}-{month_num}-{day_num}"
    # Try numeric: "02/10/1986", "1986-02-10"
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso:
        return iso.group(0)
    mdy = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if mdy:
        return f"{mdy.group(3)}-{mdy.group(1).zfill(2)}-{mdy.group(2).zfill(2)}"
    return None


def _apply_pattern(
    paragraph: str,
    id_pattern: str,
    field_patterns: dict[str, dict[str, Any]],
    null_signals: list[str] | None = None,
) -> dict[str, Any] | None:
    """Apply extraction patterns to a single paragraph. Returns record or None."""
    # Extract ID
    try:
        id_match = re.search(id_pattern, paragraph)
    except re.error:
        return None
    if not id_match:
        return None

    record: dict[str, Any] = {"_id": id_match.group(1) if id_match.lastindex else id_match.group(0)}

    for field_name, spec in field_patterns.items():
        pattern = spec.get("pattern")
        if not pattern:
            continue

        field_type = spec.get("type", "text")
        correction_signal = spec.get("correction_signal")
        field_null_signals = spec.get("null_signals", []) or []

        # Check null signals first
        para_lower = paragraph.lower()
        is_null = False
        for ns in field_null_signals:
            if ns and ns.lower() in para_lower:
                # Only null if the null signal is near this field's keyword
                is_null = True
                break
        if is_null:
            record[field_name] = None
            continue

        # Try correction pattern first (if field has corrections)
        value = None
        if correction_signal:
            try:
                corr_match = re.search(correction_signal, paragraph, re.IGNORECASE)
                if corr_match:
                    value = corr_match.group(1) if corr_match.lastindex else corr_match.group(0)
            except re.error:
                pass

        # Fall back to direct pattern
        if value is None:
            try:
                match = re.search(pattern, paragraph, re.IGNORECASE)
                if match:
                    value = match.group(1) if match.lastindex else match.group(0)
            except re.error:
                continue

        if value is None:
            record[field_name] = None
            continue

        # Type conversion
        if field_type == "number":
            try:
                value = float(value)
                if value == int(value):
                    value = int(value)
            except (ValueError, TypeError):
                pass
        elif field_type == "date":
            parsed = _parse_prose_date(value)
            if parsed:
                value = parsed
        elif field_type == "category":
            value = value.strip()

        record[field_name] = value

    return record


def _apply_patterns_to_all(
    paragraphs: list[str],
    patterns: dict[str, Any],
    additional_patterns: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Apply patterns to all paragraphs. Returns (records, unmatched_indices)."""
    id_pattern = patterns.get("id_pattern", "")
    field_patterns = patterns.get("field_patterns", {})

    records: list[dict[str, Any]] = []
    unmatched: list[int] = []

    for i, para in enumerate(paragraphs):
        record = _apply_pattern(para, id_pattern, field_patterns)

        # Try additional patterns if primary failed
        if record is None and additional_patterns:
            for alt in additional_patterns:
                condition = alt.get("applies_when", "")
                if condition and condition.lower() not in para.lower():
                    continue
                record = _apply_pattern(
                    para,
                    alt.get("id_pattern", id_pattern),
                    alt.get("field_patterns", field_patterns),
                )
                if record is not None:
                    break

        if record is not None:
            records.append(record)
        else:
            unmatched.append(i)

    return records, unmatched


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _stratified_sample(paragraphs: list[str], n: int = 8) -> list[str]:
    """Pick diverse paragraphs: varying positions and lengths."""
    if len(paragraphs) <= n:
        return paragraphs

    # Sort by length to get variety
    by_length = sorted(range(len(paragraphs)), key=lambda i: len(paragraphs[i]))

    # Pick: shortest, longest, and evenly spaced by position
    indices: set[int] = set()
    indices.add(by_length[0])  # shortest
    indices.add(by_length[-1])  # longest
    indices.add(by_length[len(by_length) // 2])  # median length

    # Evenly spaced by position
    step = len(paragraphs) // (n - len(indices))
    for j in range(0, len(paragraphs), max(1, step)):
        indices.add(j)
        if len(indices) >= n:
            break

    # Fill remaining from middle if needed
    mid = len(paragraphs) // 2
    offset = 1
    while len(indices) < n and offset < len(paragraphs):
        indices.add(min(mid + offset, len(paragraphs) - 1))
        indices.add(max(mid - offset, 0))
        offset += 1

    selected = sorted(indices)[:n]
    return [paragraphs[i] for i in selected]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def adaptive_extract(
    doc_path: Path,
    db_path: Path,
    model: ModelAdapter,
    question: str,
    knowledge_text: str = "",
    log_fn: Callable[[str, str], None] | None = None,
    protected_tables: set[str] | None = None,
    coverage_threshold: float = 0.75,
    max_iterations: int = 3,
) -> int:
    """Adaptive pattern extraction: discover patterns, apply deterministically.

    Returns number of records written to SQLite.
    """
    t0 = time.time()
    try:
        text = doc_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    if len(text) < 100:
        return 0

    # Split into entity paragraphs
    from data_agent_baseline.pipeline.compiled_extractor import (
        _split_paragraphs,
        _get_existing_db_context,
        _build_knowledge_hint,
        _extract_knowledge_columns,
        _write_records,
        _resolve_fk_post_extraction,
        _run_planner,
        _get_sample_text,
    )

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        if log_fn:
            log_fn("adaptive_skip", f"{doc_path.stem}: no entity paragraphs")
        return 0

    if log_fn:
        avg_len = sum(len(p) for p in paragraphs) // len(paragraphs)
        log_fn(
            "adaptive_start",
            f"{doc_path.stem}: {len(paragraphs)} entities (avg {avg_len} chars), "
            f"coverage_threshold={coverage_threshold}",
        )

    # Context for planner
    db_context = _get_existing_db_context(db_path)
    knowledge_hint = _build_knowledge_hint(question, knowledge_text)
    plan = _run_planner(model, text, question, db_context, knowledge_hint, log_fn)
    if plan:
        entity = plan.get("entity", doc_path.stem)
        fields = plan.get("fields", ["_id"])
    else:
        entity = doc_path.stem
        fields = ["_id"]
        # Fallback: knowledge fields relevant to this entity only
        if knowledge_text:
            entity_lower = entity.lower()
            for section in re.split(r"###\s+", knowledge_text):
                if entity_lower in section[:50].lower():
                    for match in re.findall(r"\*\*(\w+)\s*\(", section):
                        if match not in fields:
                            fields.append(match)
                    break

    fields_str = ", ".join(fields)
    if log_fn:
        log_fn("adaptive_schema", f"entity={entity}, fields={fields}")

    # --- ITERATION 1: Pattern Discovery ---
    samples = _stratified_sample(paragraphs)
    sample_text = "\n---\n".join(samples)

    prompt = PATTERN_DISCOVERY_PROMPT.format(
        question=question,
        entity=entity,
        fields=fields_str,
        db_context=db_context[:800],
        samples=sample_text,
    )

    if log_fn:
        log_fn("adaptive_discovery", f"Sending {len(samples)} samples to LLM for pattern discovery")

    messages = [ModelMessage(role="user", content=prompt)]
    try:
        raw = model.complete(messages, thinking=False)
    except Exception as e:
        if log_fn:
            log_fn("adaptive_error", f"Pattern discovery LLM failed: {e}")
        return 0

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    patterns = _parse_pattern_response(raw)
    if not patterns or not patterns.get("id_pattern"):
        if log_fn:
            log_fn("adaptive_fail", f"Could not parse patterns from LLM: {raw[:300]}")
        return 0

    if log_fn:
        log_fn(
            "adaptive_patterns",
            f"id_pattern={patterns.get('id_pattern')}, "
            f"fields={list(patterns.get('field_patterns', {}).keys())}",
        )
        for fname, fspec in patterns.get("field_patterns", {}).items():
            log_fn(
                "adaptive_pattern_detail",
                f"  {fname}: pattern={fspec.get('pattern')}, "
                f"type={fspec.get('type')}, correction={fspec.get('correction_signal')}",
            )

    # --- DETERMINISTIC EXTRACTION ---
    additional_patterns: list[dict[str, Any]] = []
    records, unmatched = _apply_patterns_to_all(paragraphs, patterns, additional_patterns)

    coverage = len(records) / len(paragraphs)
    if log_fn:
        log_fn(
            "adaptive_coverage",
            f"Round 1: {len(records)}/{len(paragraphs)} matched ({coverage:.1%}), "
            f"{len(unmatched)} unmatched",
        )
        if records:
            log_fn("adaptive_sample", f"Sample record: {json.dumps(records[0], default=str)[:300]}")
            # Count non-null fields across all records
            field_fill: dict[str, int] = {}
            for r in records:
                for k, v in r.items():
                    if v is not None:
                        field_fill[k] = field_fill.get(k, 0) + 1
            log_fn("adaptive_fill_rate", f"Field fill rates: {{{', '.join(f'{k}:{v}/{len(records)}' for k, v in sorted(field_fill.items()))}}}")

    # --- REFINEMENT ITERATIONS ---
    iteration = 1
    while coverage < coverage_threshold and iteration < max_iterations and unmatched:
        iteration += 1
        # Sample from unmatched
        unmatched_paras = [paragraphs[i] for i in unmatched[:10]]
        unmatched_text = "\n---\n".join(unmatched_paras)

        refine_prompt = PATTERN_REFINEMENT_PROMPT.format(
            current_patterns=json.dumps(patterns, indent=2, default=str)[:2000],
            unmatched_samples=unmatched_text,
        )

        if log_fn:
            log_fn("adaptive_refine", f"Round {iteration}: refining with {len(unmatched_paras)} unmatched samples")

        messages = [ModelMessage(role="user", content=refine_prompt)]
        try:
            raw = model.complete(messages, thinking=False)
        except Exception:
            break

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        refinement = _parse_pattern_response(raw)

        if refinement:
            # Apply pattern fixes
            fixes = refinement.get("pattern_fixes", {})
            if fixes:
                for field_name, new_pattern in fixes.items():
                    if new_pattern and field_name in patterns.get("field_patterns", {}):
                        patterns["field_patterns"][field_name]["pattern"] = new_pattern
                if log_fn:
                    log_fn("adaptive_fixes", f"Fixed patterns for: {list(fixes.keys())}")

            # Add additional patterns
            new_alts = refinement.get("additional_patterns", [])
            if new_alts:
                additional_patterns.extend(new_alts)
                if log_fn:
                    log_fn("adaptive_alt_patterns", f"Added {len(new_alts)} alternative patterns")

            # Re-extract with updated patterns
            records, unmatched = _apply_patterns_to_all(paragraphs, patterns, additional_patterns)
            coverage = len(records) / len(paragraphs)
            if log_fn:
                log_fn(
                    "adaptive_coverage",
                    f"Round {iteration}: {len(records)}/{len(paragraphs)} matched ({coverage:.1%})",
                )

    # --- WRITE TO SQLITE ---
    if not records:
        if log_fn:
            log_fn("adaptive_empty", f"{doc_path.stem}: no records extracted")
        return 0

    # Resolve FK fields
    plan_fields = list(records[0].keys()) if records else fields
    records = _resolve_fk_post_extraction(records, db_path, plan_fields, log_fn)

    # Table name
    table_name = re.sub(r"[^a-zA-Z0-9_]", "_", doc_path.stem).lower()
    if not table_name or table_name[0].isdigit():
        table_name = "t_" + table_name
    if protected_tables and table_name in protected_tables:
        table_name = f"{table_name}_doc"

    written = _write_records(db_path, table_name, records, log_fn)

    elapsed = time.time() - t0
    if log_fn:
        log_fn(
            "adaptive_done",
            f"{doc_path.stem}: {written} records in {elapsed:.1f}s "
            f"({iteration} LLM calls, {coverage:.1%} coverage)",
        )

    return written


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pattern_response(raw: str) -> dict[str, Any]:
    """Parse JSON from LLM pattern discovery response."""
    if not raw:
        return {}
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    # Find the JSON object
    brace = raw.find("{")
    if brace < 0:
        return {}
    depth = 0
    for i in range(brace, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[brace:i + 1])
                except json.JSONDecodeError:
                    # Try fixing trailing commas
                    fixed = re.sub(r",\s*([}\]])", r"\1", raw[brace:i + 1])
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        return {}
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adaptive_extract_docs(
    doc_paths: list[Path],
    db_path: Path,
    model: ModelAdapter,
    question: str,
    knowledge_text: str = "",
    log_fn: Callable[[str, str], None] | None = None,
    structured_tables: list[str] | None = None,
) -> int:
    """Extract all docs using adaptive pattern discovery."""
    if not doc_paths:
        return 0

    protected = {t.lower() for t in structured_tables} if structured_tables else set()
    total = 0

    if log_fn:
        log_fn("adaptive_extract_start", f"{len(doc_paths)} docs: {[p.name for p in doc_paths]}")

    for doc_path in doc_paths:
        n = adaptive_extract(
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
        log_fn("adaptive_extract_done", f"Total: {total} records from {len(doc_paths)} docs")

    return total
