from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DiscoveredField:
    name: str  # e.g. "GOT", "CRE", "Birthday"
    field_type: str  # "numeric", "date", "categorical", "text"
    aliases: list[str]  # e.g. ["glutamic oxaloacetic transaminase", "GOT"]
    unit: str | None  # e.g. "U/L", "mg/dL"
    frequency: int  # how many times found in doc


@dataclass
class DocumentSchema:
    entity_name: str  # from filename (e.g. "Laboratory")
    id_field: str  # "patient_id" usually
    id_pattern: re.Pattern  # compiled regex for record IDs
    fields: list[DiscoveredField]  # all discovered fields
    composite_key_fields: list[str]  # e.g. ["patient_id", "date"] for lab
    has_multiple_records_per_id: bool  # True if same ID appears on different dates


# --- Regex patterns ---

# knowledge.md: "### EntityName" sections
_KNOWLEDGE_HEADING_RE = re.compile(r"^###\s+(\w+)\s*$", re.MULTILINE)

# knowledge.md: "- **FieldName (type):** description"
_KNOWLEDGE_FIELD_RE = re.compile(
    r"^-\s+\*\*(\w[\w\s\-]*?)\s*\((\w+)\)\s*[:.]?\*\*[:\s]*(.*)$", re.MULTILINE
)

# Text: "full name (ABBREVIATION)" e.g. "glutamic oxaloacetic transaminase (GOT)"
# Captures a run of words (first char may be upper/lower) before a parenthesized abbreviation.
# Post-processing trims leading noise words (articles, prepositions, possessives).
_ALIAS_RAW_RE = re.compile(
    r"((?:[a-zA-Z][a-z\-]*(?:\s+[a-z][a-z\-]*){0,6}))\s*\(([A-Z][A-Z0-9\-]{0,10})\)"
)

# Words that should be stripped from the start of a captured alias phrase
_NOISE_PREFIXES = {
    "a", "an", "the", "and", "or", "of", "for", "with", "in", "on", "at",
    "to", "by", "is", "was", "were", "are", "be", "s", "its", "their", "his",
    "her", "our", "your",
}

# Text: ABBREVIATION was/of/at/: VALUE UNIT
_ABBREV_VALUE_RE = re.compile(
    r"\b([A-Z][A-Z0-9\-]{1,10})\s+(?:was|of|at|is|:|=)\s+"
    r"(-?\d+\.?\d*)\s*([a-zA-Z/%]+(?:/[a-zA-Z]+)?)?",
)

# Text: VALUE UNIT patterns after known context
_VALUE_UNIT_RE = re.compile(
    r"(-?\d+\.?\d*)\s*([a-zA-Z/%]+(?:/[a-zA-Z]+)?)"
)

# Date patterns
_DATE_RE = re.compile(
    r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})"
    r"|(?:\d{1,2}[-/]\d{1,2}[-/]\d{4})"
    r"|(?:(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4})"
)

# ID-like patterns: 5+ digit numbers or recXXX
_ID_PATTERN_RE = re.compile(r"\b(\d{5,})\b")
_REC_ID_RE = re.compile(r"\b(rec[A-Za-z0-9]{3,})\b")

# Common units for sanity checking
_KNOWN_UNITS = {
    "U/L", "mg/dL", "mEq/L", "g/dL", "mg/dl", "u/l", "meq/l",
    "IU/L", "mmol/L", "umol/L", "ng/mL", "pg/mL", "mL/min",
    "mm", "cm", "kg", "g", "L", "dL", "mL", "%", "mmHg",
    "x10^3/uL", "x10^6/uL", "10^9/L", "fL", "pg",
}


def discover_schema(
    doc_text: str, doc_name: str, knowledge_text: str = ""
) -> DocumentSchema:
    """Main entry point: combine knowledge.md definitions + statistical discovery."""
    entity_name = Path(doc_name).stem  # "Laboratory.md" -> "Laboratory"

    # Get fields from knowledge.md
    knowledge_fields = _discover_from_knowledge(knowledge_text, entity_name)

    # Get fields from statistical text analysis
    text_fields = _discover_from_text(doc_text)

    # Merge: knowledge fields take priority, text can add new ones
    fields = _merge_fields(knowledge_fields, text_fields)

    # Detect ID pattern
    id_field, id_pattern = _detect_id_pattern(doc_text)

    # Detect composite keys
    has_multiple, composite_keys = _detect_composite_keys(doc_text, id_pattern)

    return DocumentSchema(
        entity_name=entity_name,
        id_field=id_field,
        id_pattern=id_pattern,
        fields=fields,
        composite_key_fields=composite_keys,
        has_multiple_records_per_id=has_multiple,
    )


def _discover_from_knowledge(
    knowledge_text: str, entity_name: str
) -> list[DiscoveredField]:
    """Parse '### EntityName' sections in knowledge.md for field definitions."""
    if not knowledge_text.strip():
        return []

    # Find the section for this entity
    headings = list(_KNOWLEDGE_HEADING_RE.finditer(knowledge_text))
    section_text = ""
    for i, match in enumerate(headings):
        if match.group(1).lower() == entity_name.lower():
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(knowledge_text)
            section_text = knowledge_text[start:end]
            break

    if not section_text:
        return []

    fields: list[DiscoveredField] = []
    for match in _KNOWLEDGE_FIELD_RE.finditer(section_text):
        raw_name = match.group(1).strip()
        raw_type = match.group(2).strip().lower()
        description = match.group(3).strip()

        # Map knowledge types to our field types
        field_type = _map_knowledge_type(raw_type)

        # Build aliases from the name
        aliases = [raw_name]

        # Try to extract unit from description
        unit = _extract_unit_from_description(description)

        fields.append(DiscoveredField(
            name=raw_name,
            field_type=field_type,
            aliases=aliases,
            unit=unit,
            frequency=0,  # unknown from knowledge alone
        ))

    return fields


def _discover_from_text(text: str) -> list[DiscoveredField]:
    """Find repeated label-value patterns in text via regex and frequency analysis."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Track: field_key -> {name, aliases, unit, type, count}
    field_stats: dict[str, dict] = {}

    # 1. Find "full name (ABBREVIATION)" patterns to build alias map
    alias_map: dict[str, str] = {}  # abbreviation -> full name
    for match in _ALIAS_RAW_RE.finditer(text):
        full_name = _strip_noise_prefix(match.group(1).strip())
        abbreviation = match.group(2).strip()
        if full_name:  # only keep if something remains after stripping
            alias_map[abbreviation] = full_name

    # 2. Find ABBREVIATION + value + unit patterns
    for para in paragraphs:
        for match in _ABBREV_VALUE_RE.finditer(para):
            abbrev = match.group(1)
            unit = match.group(3)

            # Skip very short or very common words that aren't field names
            if len(abbrev) < 2:
                continue
            if abbrev in {"The", "This", "That", "For", "And", "But", "NOT"}:
                continue

            key = abbrev.upper()
            if key not in field_stats:
                aliases = [abbrev]
                if abbrev in alias_map:
                    aliases = [alias_map[abbrev], abbrev]
                field_stats[key] = {
                    "name": abbrev,
                    "aliases": aliases,
                    "unit": _normalize_unit(unit) if unit else None,
                    "type": "numeric",
                    "count": 0,
                }
            field_stats[key]["count"] += 1

            # Update unit if we find one and didn't have one before
            if unit and not field_stats[key]["unit"]:
                field_stats[key]["unit"] = _normalize_unit(unit)

    # 3. Also scan for alias patterns not already captured
    for abbrev, full_name in alias_map.items():
        key = abbrev.upper()
        if key not in field_stats:
            # Count occurrences of the abbreviation in text
            count = len(re.findall(r"\b" + re.escape(abbrev) + r"\b", text))
            if count >= 2:
                field_stats[key] = {
                    "name": abbrev,
                    "aliases": [full_name, abbrev],
                    "unit": None,
                    "type": "numeric",
                    "count": count,
                }

    # 4. Detect date fields
    date_count = len(_DATE_RE.findall(text))
    if date_count >= 3:
        if "DATE" not in field_stats:
            field_stats["DATE"] = {
                "name": "Date",
                "aliases": ["Date", "date"],
                "unit": None,
                "type": "date",
                "count": date_count,
            }

    # 5. Filter: keep fields with frequency >= 3
    fields: list[DiscoveredField] = []
    for key, stats in field_stats.items():
        if stats["count"] >= 3:
            fields.append(DiscoveredField(
                name=stats["name"],
                field_type=stats["type"],
                aliases=stats["aliases"],
                unit=stats["unit"],
                frequency=stats["count"],
            ))

    # Sort by frequency descending
    fields.sort(key=lambda f: f.frequency, reverse=True)
    return fields


def _detect_id_pattern(text: str) -> tuple[str, re.Pattern]:
    """Find the record ID pattern (5+ digit numbers, recXXX patterns, etc.)."""
    # Try numeric IDs first (5+ digits)
    numeric_ids = _ID_PATTERN_RE.findall(text)
    rec_ids = _REC_ID_RE.findall(text)

    # Count frequencies to find the most common repeating pattern
    numeric_counter = Counter(numeric_ids)
    rec_counter = Counter(rec_ids)

    # Find IDs that repeat (appear 2+ times)
    repeating_numeric = {k: v for k, v in numeric_counter.items() if v >= 2}
    repeating_rec = {k: v for k, v in rec_counter.items() if v >= 2}

    if repeating_numeric and (
        not repeating_rec or sum(repeating_numeric.values()) >= sum(repeating_rec.values())
    ):
        # Determine digit length from most common ID
        most_common_id = max(repeating_numeric, key=repeating_numeric.get)
        id_len = len(most_common_id)
        # Build pattern that matches this length of digits
        pattern = re.compile(r"\b(\d{" + str(id_len) + r",})\b")
        return "patient_id", pattern

    if repeating_rec:
        pattern = re.compile(r"\b(rec[A-Za-z0-9]{3,})\b")
        return "record_id", pattern

    # Fallback: any 5+ digit number
    return "patient_id", re.compile(r"\b(\d{5,})\b")


def _detect_composite_keys(
    text: str, id_pattern: re.Pattern
) -> tuple[bool, list[str]]:
    """Check if same ID appears in multiple paragraphs with different dates."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Map: id -> set of dates found in its paragraphs
    id_dates: dict[str, set[str]] = {}

    for para in paragraphs:
        ids_in_para = id_pattern.findall(para)
        dates_in_para = _DATE_RE.findall(para)

        for record_id in ids_in_para:
            if record_id not in id_dates:
                id_dates[record_id] = set()
            id_dates[record_id].update(dates_in_para)

    # Check if any ID has multiple different dates
    has_multiple = any(len(dates) > 1 for dates in id_dates.values())

    if has_multiple:
        # Determine the id_field name from the pattern
        id_field = "patient_id"
        if id_pattern.pattern and "rec" in id_pattern.pattern:
            id_field = "record_id"
        return True, [id_field, "date"]

    # Even without dates, check if same ID appears in many paragraphs
    id_para_count: Counter = Counter()
    for para in paragraphs:
        ids_in_para = set(id_pattern.findall(para))
        for record_id in ids_in_para:
            id_para_count[record_id] += 1

    # If any ID appears in 3+ paragraphs, likely multiple records
    frequent_ids = {k: v for k, v in id_para_count.items() if v >= 3}
    if frequent_ids:
        id_field = "patient_id"
        if id_pattern.pattern and "rec" in id_pattern.pattern:
            id_field = "record_id"
        return True, [id_field, "date"]

    id_field = "patient_id"
    if id_pattern.pattern and "rec" in id_pattern.pattern:
        id_field = "record_id"
    return False, [id_field]


# --- Helpers ---


def _merge_fields(
    knowledge_fields: list[DiscoveredField], text_fields: list[DiscoveredField]
) -> list[DiscoveredField]:
    """Merge knowledge-defined fields with text-discovered fields.

    Knowledge fields take priority. Text fields add new discoveries.
    """
    merged: list[DiscoveredField] = []
    knowledge_names = set()

    for kf in knowledge_fields:
        knowledge_names.add(kf.name.upper())
        for alias in kf.aliases:
            knowledge_names.add(alias.upper())

        # Try to find matching text field to get frequency and unit
        matching_tf = None
        for tf in text_fields:
            if tf.name.upper() == kf.name.upper() or any(
                a.upper() == kf.name.upper() for a in tf.aliases
            ):
                matching_tf = tf
                break

        if matching_tf:
            # Enrich knowledge field with text discovery data
            combined_aliases = list(dict.fromkeys(kf.aliases + matching_tf.aliases))
            unit = kf.unit or matching_tf.unit
            merged.append(DiscoveredField(
                name=kf.name,
                field_type=kf.field_type,
                aliases=combined_aliases,
                unit=unit,
                frequency=matching_tf.frequency,
            ))
        else:
            merged.append(kf)

    # Add text-discovered fields not in knowledge
    for tf in text_fields:
        tf_names = {tf.name.upper()} | {a.upper() for a in tf.aliases}
        if not tf_names & knowledge_names:
            merged.append(tf)

    return merged


def _map_knowledge_type(raw_type: str) -> str:
    """Map knowledge.md type annotations to our field types."""
    raw = raw_type.lower()
    if raw in {"integer", "real", "float", "numeric", "number", "decimal"}:
        return "numeric"
    if raw in {"date", "datetime", "time", "timestamp"}:
        return "date"
    if raw in {"text", "string", "varchar", "char"}:
        return "text"
    if raw in {"categorical", "enum", "boolean", "bool"}:
        return "categorical"
    return "text"


def _extract_unit_from_description(description: str) -> str | None:
    """Try to extract a measurement unit from a field description."""
    # Look for common unit patterns in parentheses or after keywords
    unit_in_parens = re.search(r"\(([a-zA-Z/%]+(?:/[a-zA-Z]+)?)\)", description)
    if unit_in_parens:
        candidate = unit_in_parens.group(1)
        if candidate in _KNOWN_UNITS or "/" in candidate:
            return candidate

    # Look for "in UNIT" pattern
    in_unit = re.search(r"\bin\s+([a-zA-Z/%]+(?:/[a-zA-Z]+)?)\b", description)
    if in_unit:
        candidate = in_unit.group(1)
        if candidate in _KNOWN_UNITS or "/" in candidate:
            return candidate

    return None


def _strip_noise_prefix(phrase: str) -> str:
    """Strip leading noise words (articles, prepositions, possessives) from a phrase."""
    words = phrase.split()
    while words and words[0].lower() in _NOISE_PREFIXES:
        words.pop(0)
    return " ".join(words)


def _normalize_unit(unit: str | None) -> str | None:
    """Normalize unit strings."""
    if not unit:
        return None
    # Strip trailing punctuation
    unit = unit.rstrip(".,;:")
    # Skip if it looks like a word, not a unit
    if len(unit) > 8 and "/" not in unit and "%" not in unit:
        return None
    if unit.lower() in {"was", "the", "and", "for", "with", "from", "that", "this"}:
        return None
    return unit
