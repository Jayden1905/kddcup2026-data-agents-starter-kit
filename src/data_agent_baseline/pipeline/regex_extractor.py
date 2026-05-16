"""Pattern-based document extraction engine with calibrated LLM assistance.

Budget: up to 10 LLM calls per document (vs 30-60 in compiled_extractor).
Strategy:
  1. PLAN (1 call): Identify entity type, fields, and extraction hints
  2. CALIBRATE (2 calls): Extract a few sample paragraphs via LLM, learn
     document-specific patterns by comparing LLM output to regex output
  3. REGEX PASS (0 calls): Apply all patterns across all paragraphs
  4. GAP-FILL (up to 6 calls): Batch paragraphs where ID found but critical
     fields missing, let LLM fill gaps
  5. VALIDATE (1 call): Spot-check sample of records for accuracy

Handles multi-section documents naturally since each paragraph is processed
independently and records merge by ID.
"""

from __future__ import annotations

import re
import sqlite3
import json
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


# ---------------------------------------------------------------------------
# Robust JSON parsing (handles Qwen output quirks)
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> Any | None:
    """Parse JSON from LLM response, handling common formatting issues.

    Qwen models may:
    - Wrap JSON in <think>...</think> blocks
    - Include text before/after JSON
    - Use code fences (```json ... ```) or not
    - Produce trailing commas
    - Truncate long arrays (missing closing bracket)
    """
    # Strip thinking tokens
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try code fence extraction first
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()

    # Direct parse attempt
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array or object in the response
    # Use whichever delimiter appears FIRST (outermost structure)
    arr_pos = raw.find("[")
    obj_pos = raw.find("{")
    candidates = []
    if arr_pos >= 0:
        candidates.append((arr_pos, "[", "]"))
    if obj_pos >= 0:
        candidates.append((obj_pos, "{", "}"))
    candidates.sort(key=lambda x: x[0])

    for start, start_char, end_char in candidates:
        depth = 0
        end_pos = -1
        in_str = False
        escape_next = False
        for i in range(start, len(raw)):
            c = raw[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break

        if end_pos > start:
            try:
                return json.loads(raw[start:end_pos + 1])
            except json.JSONDecodeError:
                fixed = re.sub(r",\s*([}\]])", r"\1", raw[start:end_pos + 1])
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

    # Last resort: truncated array — try appending ]
    arr_start = raw.find("[")
    if arr_start >= 0:
        candidate = raw[arr_start:]
        # Find last complete object (ending with })
        last_brace = candidate.rfind("}")
        if last_brace > 0:
            truncated = candidate[:last_brace + 1] + "]"
            truncated = re.sub(r",\s*\]", "]", truncated)
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

    return None


# ---------------------------------------------------------------------------
# ID Pattern Discovery
# ---------------------------------------------------------------------------

# Common ID introduction phrases
_ID_INTRO_PATTERNS = [
    r"(?:registered|filed|cataloged|tracked|identified|logged|maintained|referenced)\s+"
    r"(?:under|with|by|as|at)\s+(?:the\s+)?(?:unique\s+)?"
    r"(?:identifier|ID|registration\s+number|reference\s+(?:code|ID|number)|"
    r"registry\s+(?:number|ref)|number)\s+",
    r"(?:ID|identifier|registration)\s*[:.]?\s*",
    r"(?:designated|specimen|compound|molecule|subject)\s+",
    r"(?:under|with)\s+(?:the\s+)?(?:unique\s+)?(?:identifier|ID|"
    r"registration\s+number|reference\s+(?:code|ID|number)|registry\s+number)\s+",
    r"\w+\s+ID\s*[:.]?\s*",
]


def _discover_id_pattern(text: str, sample_ids: list[str] | None = None, entity_name: str = "") -> re.Pattern | None:
    """Discover the ID pattern used in the document.

    Tries common patterns. Returns a compiled regex with group(1) = the ID value.
    """
    from collections import Counter

    # If we have sample IDs from structured data, try to find them
    if sample_ids:
        # Check if IDs are alphanumeric tokens like TR391 (alpha prefix + digits)
        # Do this FIRST because common-prefix detection on sorted IDs can be
        # too narrow (e.g., TR000-TR019 → prefix "TR0" misses TR100+)
        alpha_ids = [i for i in sample_ids if re.match(r'^[A-Za-z]+\d+$', i)]
        if alpha_ids:
            prefix = re.match(r'^([A-Za-z]+)', alpha_ids[0]).group(1)
            pat = re.compile(rf'\b({re.escape(prefix)}\d+)\b')
            matches = pat.findall(text[:5000])
            if len(matches) >= 3:
                return pat

        # Detect common prefix among sample IDs (handles Airtable rec..., etc.)
        if len(sample_ids) >= 2:
            prefix = sample_ids[0]
            for sid in sample_ids[1:]:
                while prefix and not sid.startswith(prefix):
                    prefix = prefix[:-1]
            if len(prefix) >= 2:
                # Determine typical ID length range
                lengths = [len(s) for s in sample_ids[:10]]
                min_len = min(lengths)
                max_len = max(lengths)
                if min_len == max_len:
                    # Fixed-length IDs (e.g., rec + 14 chars = 17 total)
                    suffix_len = min_len - len(prefix)
                    pat = re.compile(
                        rf'\b({re.escape(prefix)}[A-Za-z0-9]{{{suffix_len}}})\b'
                    )
                else:
                    # Variable-length
                    min_suffix = min_len - len(prefix)
                    max_suffix = max_len - len(prefix)
                    pat = re.compile(
                        rf'\b({re.escape(prefix)}[A-Za-z0-9]{{{min_suffix},{max_suffix}}})\b'
                    )
                matches = pat.findall(text[:10000])
                if len(matches) >= 3:
                    return pat

        # Check if IDs are pure numeric — find the BEST pattern (most matches)
        numeric_ids = [i for i in sample_ids if i.isdigit()]
        if numeric_ids:
            best_pat = None
            best_count = 0
            for intro in _ID_INTRO_PATTERNS:
                pat = re.compile(intro + r'(\d+)', re.IGNORECASE)
                matches = pat.findall(text[:10000])
                if len(matches) > best_count:
                    best_pat = pat
                    best_count = len(matches)
            # Also try word-prefix patterns (Patient 43003, Subject 789)
            word_num = re.findall(
                r'\b(Patient|Subject|Case|Record|Specimen|Unit|File|Entry)\s+(?:ID\s+)?(\d{3,})\b',
                text[:15000], re.IGNORECASE,
            )
            if word_num:
                prefix_counts = Counter(p.lower() for p, _ in word_num)
                best_prefix, wp_count = prefix_counts.most_common(1)[0]
                if wp_count > best_count:
                    best_pat = re.compile(
                        rf'\b(?:{re.escape(best_prefix)}|Case\s+ID|'
                        rf'identified\s+as|registered\s+(?:under|as))\s+'
                        rf'(?:ID\s+)?(\d{{3,}})\b',
                        re.IGNORECASE,
                    )
                    best_count = wp_count
            if best_pat and best_count >= 3:
                return best_pat

            # Fallback: validate sample IDs against text and discover context words.
            # Useful when IDs are short integers (1-3 digits) introduced by varied
            # domain words (e.g., "race 18", "event 27", "docket 32").
            id_set = set(numeric_ids)
            # Find all words immediately preceding numbers in the text
            context_hits = re.findall(
                r'\b([a-zA-Z]{3,})\s+(\d{1,5})\b', text[:20000]
            )
            # Count how many known IDs each context word precedes
            word_id_hits: dict[str, int] = {}
            for word, num in context_hits:
                if num in id_set:
                    word_id_hits[word.lower()] = word_id_hits.get(word.lower(), 0) + 1
            # Also check "as <number>" pattern (e.g., "docketed as 7")
            as_hits = re.findall(r'\b(\w{3,})\s+as\s+(\d{1,5})\b', text[:20000])
            for word, num in as_hits:
                if num in id_set:
                    word_id_hits[word.lower()] = word_id_hits.get(word.lower(), 0) + 1
            # Also check "number <number>" / "file <number>" patterns
            num_hits = re.findall(
                r'\b(?:file\s+)?(?:number|no\.?|#)\s+(\d{1,5})\b', text[:20000], re.IGNORECASE
            )
            file_count = sum(1 for n in num_hits if n in id_set)
            if file_count >= 3:
                word_id_hits["_file_number_"] = file_count

            if word_id_hits:
                best_word = max(word_id_hits, key=word_id_hits.get)
                hit_count = word_id_hits[best_word]
                if hit_count >= 3:
                    if best_word == "_file_number_":
                        pat = re.compile(
                            r'\b(?:file\s+)?(?:number|no\.?|#)\s+(\d{1,5})\b',
                            re.IGNORECASE,
                        )
                    else:
                        # Build pattern: "<word> <number>" OR "<word> as <number>"
                        # Also include related words that introduce IDs in this doc
                        # Scan for all context words with >= 2 hits
                        good_words = [w for w, c in word_id_hits.items()
                                      if c >= 2 and w != "_file_number_"]
                        if len(good_words) > 1:
                            alt = "|".join(re.escape(w) for w in good_words[:6])
                            pat = re.compile(
                                rf'\b(?:{alt})\s+(?:as\s+|number\s+)?(\d{{1,5}})\b',
                                re.IGNORECASE,
                            )
                        else:
                            pat = re.compile(
                                rf'\b{re.escape(best_word)}\s+(?:as\s+|number\s+)?(\d{{1,5}})\b',
                                re.IGNORECASE,
                            )
                    # Verify: pattern should find a good portion of sample IDs
                    found = set(pat.findall(text[:30000]))
                    overlap = found & id_set
                    if len(overlap) >= 3:
                        return pat

    # Word-prefix numeric IDs: "Patient 43003", "Subject 789", "Case ID 12345"
    # Check BEFORE statistical discovery since keyword detection is more reliable.
    word_num = re.findall(
        r'\b(Patient|Subject|Case|Record|Specimen|Unit|File|Entry)\s+(?:ID\s+)?(\d{3,})\b',
        text[:15000], re.IGNORECASE,
    )
    if word_num:
        prefix_counts_kw = Counter(p.lower() for p, _ in word_num)
        best_prefix_kw, count_kw = prefix_counts_kw.most_common(1)[0]
        if count_kw >= 5:
            pat = re.compile(
                rf'\b(?:{re.escape(best_prefix_kw)}|Case\s+ID|'
                rf'identified\s+as|registered\s+(?:under|as))\s+'
                rf'(?:ID\s+)?(\d{{3,}})\b',
                re.IGNORECASE,
            )
            return pat

    # Statistical word-number discovery: find words that most frequently precede
    # numbers in the document. If a small set of words accounts for many number
    # introductions, those words are likely record-ID introducers.
    # E.g., "race 18", "event 27", "docket 32" — discovers "race", "event", "docket"
    # purely from frequency, no hardcoded keywords.
    word_num_pairs = re.findall(
        r'\b([a-zA-Z]{3,15})\s+(\d{1,5})(?!:)\b', text[:30000]
    )
    if word_num_pairs:
        word_counts: dict[str, list[str]] = {}
        for word, num in word_num_pairs:
            word_counts.setdefault(word.lower(), []).append(num)
        # Rank words by how many UNIQUE numbers they precede
        word_unique = [(w, len(set(nums)), nums) for w, nums in word_counts.items()]
        word_unique.sort(key=lambda x: -x[1])
        # Take words that introduce >= 5 unique numbers
        top_words = [(w, uniq) for w, uniq, _ in word_unique if uniq >= 5]
        if top_words:
            # Use top words (up to 4) that collectively cover enough unique IDs
            best_words = [w for w, _ in top_words[:4]]
            alt = "|".join(re.escape(w) for w in best_words)
            pat = re.compile(rf'\b(?:{alt})\s+(\d{{1,5}})(?!:)\b', re.IGNORECASE)
            test_matches = pat.findall(text[:30000])
            if len(set(test_matches)) >= 5:
                return pat

    # Auto-discover: detect fixed-length alphanumeric tokens with common prefix (e.g., rec + 14 chars)
    # Look for repeated tokens with a short lowercase prefix followed by mixed alphanumeric
    fixed_tokens = re.findall(r'\b([a-z]{2,5}[A-Za-z0-9]{10,20})\b', text[:15000])
    if len(fixed_tokens) >= 5:
        prefix_counts: dict[str, list[str]] = {}
        for tok in fixed_tokens:
            # Extract prefix (lowercase letters at start)
            m = re.match(r'^([a-z]+)', tok)
            if m:
                p = m.group(1)
                prefix_counts.setdefault(p, []).append(tok)
        for p, tokens in sorted(prefix_counts.items(), key=lambda x: -len(x[1])):
            if len(tokens) < 5:
                continue
            lengths = [len(t) for t in tokens[:20]]
            if max(lengths) - min(lengths) <= 1:
                # Fixed-length tokens with common prefix
                suffix_len = min(lengths) - len(p)
                pat = re.compile(rf'\b({re.escape(p)}[A-Za-z0-9]{{{suffix_len}}})\b')
                matches = pat.findall(text[:15000])
                if len(matches) >= 5:
                    return pat

    # Auto-discover: detect repeated ALPHA+DIGIT tokens first (most reliable)
    alpha_num = re.findall(r'\b([A-Z]{1,5})(\d{2,})\b', text[:15000])
    if alpha_num:
        prefix_counts_upper = Counter(prefix for prefix, _ in alpha_num)
        best_prefix, count = prefix_counts_upper.most_common(1)[0]
        if count >= 5:
            pat = re.compile(rf'\b({re.escape(best_prefix)}\d+)\b')
            return pat

    # Try each intro pattern for numeric IDs
    for intro in _ID_INTRO_PATTERNS:
        pat = re.compile(intro + r'(\d+)', re.IGNORECASE)
        matches = pat.findall(text[:10000])
        if len(matches) >= 5:
            return pat

    # Broader word-prefix: any word + large numbers appearing frequently
    any_word_num = re.findall(r'\b([A-Z][a-z]+)\s+(\d{4,})\b', text[:15000])
    if any_word_num:
        prefix_counts = Counter(p for p, _ in any_word_num)
        best_prefix, count = prefix_counts.most_common(1)[0]
        if count >= 5:
            pat = re.compile(rf'\b(?:{re.escape(best_prefix)})\s+(\d{{4,}})\b')
            return pat

    return None


def _discover_name_pattern(text: str, paragraphs: list[str]) -> re.Pattern | None:
    """Discover patterns for entity names (codename, designation, etc.)."""
    # Common name introduction patterns
    name_intros = [
        r"(?:known\s+as|designated|codename(?:d)?|operative|unit|asset|specimen|compound)\s+([\w\s\-']+?)(?:,|\s+(?:whose|registered|filed|cataloged|tracked|identified|is))",
        r"(?:the\s+)?(?:operative|unit|asset|entity|individual)\s+(?:designated|known\s+as|called)\s+([\w\s\-']+?)(?:,|\s+(?:whose|registered|filed|cataloged))",
    ]
    for intro in name_intros:
        pat = re.compile(intro, re.IGNORECASE)
        matches = pat.findall(text[:10000])
        if len(matches) >= 3:
            return pat
    return None


# ---------------------------------------------------------------------------
# Field Value Extraction Patterns
# ---------------------------------------------------------------------------

class FieldExtractor:
    """Extracts a specific field's value from a paragraph."""

    def __init__(self, field_name: str, field_type: str, patterns: list[re.Pattern],
                 unit: str = ""):
        self.field_name = field_name
        self.field_type = field_type  # "numeric", "text", "categorical", "integer"
        self.patterns = patterns
        self.unit = unit

    def extract(self, text: str) -> Any | None:
        """Try each pattern in order, return first match.

        For numeric fields with units, handles corrections by taking the LAST
        value when a correction phrase is detected.
        """
        # For numeric fields with a unit, check for correction patterns first
        if self.unit and self.field_type in ("numeric", "integer", "real"):
            corrected = self._extract_with_correction(text)
            if corrected is not None:
                return corrected

        for pat in self.patterns:
            m = pat.search(text)
            if m:
                raw = m.group(1).strip()
                return self._coerce(raw)
        return None

    def _extract_with_correction(self, text: str) -> Any | None:
        """Handle correction patterns: take the LAST NUM+unit in the paragraph."""
        lower = text.lower()
        base_name = self.field_name.replace("_", " ").lower()
        base_name = re.sub(r'\s*(cm|kg|mm|ml|mg|lb|oz|m|km|g)$', '', base_name)

        # Only apply correction logic if the field is mentioned AND correction words exist
        if base_name not in lower:
            return None
        correction_words = ("correct", "confirm", "amend", "revis", "updat", "verif")
        if not any(w in lower for w in correction_words):
            return None

        # Find ALL "NUM unit" occurrences in the field's context
        unit_pat = re.compile(
            rf"([-+]?\d+\.?\d*)\s*{re.escape(self.unit)}", re.IGNORECASE
        )
        # Look for the field name, then find numbers after it
        field_idx = lower.find(base_name)
        if field_idx < 0:
            return None
        search_text = text[field_idx:]
        matches = unit_pat.findall(search_text)
        if matches and len(matches) >= 2:
            # Take the LAST value (corrected)
            return self._coerce(matches[-1])
        return None

    def _coerce(self, raw: str) -> Any:
        if self.field_type in ("numeric", "integer", "real"):
            try:
                val = float(raw)
                if self.field_type == "integer":
                    return int(val)
                return val
            except ValueError:
                return raw
        return raw


# Correction pattern: detects "initially X, corrected/confirmed to Y"
_CORRECTION_RE = re.compile(
    r"(?:initially|originally|first)\s+(?:logged|recorded|reported|"
    r"estimated|listed|documented|classified|assessed|indicated|flagged)\s+"
    r"(?:as\s+|at\s+|with\s+)?['\"]?([-\w.]+)['\"]?"
    r".*?"
    r"(?:correct(?:ed|ion)|confirm(?:ed)?|amend(?:ed)?|revis(?:ed)?|"
    r"updat(?:ed)?|changed|verified|finalized|authoritatively)\s+"
    r"(?:to\s+(?:be\s+)?|at\s+|as\s+|the\s+record\s+to\s+)?['\"]?([-\w.]+)['\"]?",
    re.IGNORECASE | re.DOTALL,
)


def _build_numeric_patterns(field_name: str, unit: str = "") -> list[re.Pattern]:
    """Build regex patterns for a numeric field based on its name and unit."""
    # Humanize field name: height_cm -> "height"
    base_name = field_name.replace("_", " ").lower()
    # Remove unit suffix (cm, kg, etc.)
    base_name = re.sub(r'\s*(cm|kg|mm|ml|mg|lb|oz|m|km|g)$', '', base_name)

    words = base_name.split()
    # Build alternation from name parts
    name_alts = [re.escape(base_name)]
    if len(words) > 1:
        name_alts.append(re.escape(words[0]))

    name_pat = "|".join(name_alts)
    unit_pat = re.escape(unit) if unit else r"[a-z]*"

    patterns = []
    # Pattern 1: "field is/at/as [adj] NUM unit"
    patterns.append(re.compile(
        rf"(?:{name_pat})\s+(?:is\s+|of\s+|at\s+|as\s+|=\s*)?"
        rf"(?:recorded\s+(?:at|as)\s+|listed\s+(?:at|as)\s+|logged\s+(?:at|as)\s+|"
        rf"documented\s+(?:at|as)\s+|confirmed\s+(?:at|as|to\s+be)\s+|measured\s+at\s+|"
        rf"estimated\s+(?:to\s+be\s+|at\s+))?"
        rf"(?:(?:a|an)\s+)?(?:\w+\s+){{0,3}}?"
        rf"([-+]?\d+\.?\d*)\s*(?:{unit_pat})?",
        re.IGNORECASE,
    ))
    # Pattern 2: "NUM unit" anywhere near field name (correction final value)
    if unit:
        patterns.append(re.compile(
            rf"(?:confirmed|corrected|updated|amended|revised|verified)\s+"
            rf"(?:his|her|its|the|to)?\s*(?:\w+\s+){{0,4}}?"
            rf"(?:to\s+(?:be\s+)?|at\s+|as\s+)?"
            rf"([-+]?\d+\.?\d*)\s*{re.escape(unit)}",
            re.IGNORECASE,
        ))
        # Pattern 3: "NUM unit" standalone — last number before unit in paragraph
        # Used as fallback: find ALL "NUM unit" and take the last one (corrected value)
        patterns.append(re.compile(
            rf"([-+]?\d+\.?\d*)\s*{re.escape(unit)}",
            re.IGNORECASE,
        ))
    return patterns


def _build_integer_code_patterns(field_name: str) -> list[re.Pattern]:
    """Build patterns for integer code fields (publisher_id, alignment_id, etc.)."""
    # Remove _id suffix for label matching
    base = field_name.replace("_id", "").replace("_", " ")

    patterns = []
    # "publisher affiliation is logged with the code 13"
    patterns.append(re.compile(
        rf"(?:{re.escape(base)})\s+(?:affiliation\s+)?"
        rf"(?:is\s+)?(?:logged|recorded|classified|listed|filed|documented|confirmed)\s+"
        rf"(?:with\s+(?:the\s+)?(?:code|id|number)\s+|as\s+(?:a\s+)?(?:category\s+)?|under\s+)"
        rf"(\d+)",
        re.IGNORECASE,
    ))
    # "publisher affiliation is/was NUM" (simple copula)
    patterns.append(re.compile(
        rf"(?:{re.escape(base)})\s+(?:affiliation\s+)?(?:is|was)\s+(?:confirmed\s+(?:as\s+)?)?"
        rf"(\d+)",
        re.IGNORECASE,
    ))
    # "affiliated with publisher 13" / "jurisdiction of publisher 13"
    patterns.append(re.compile(
        rf"(?:affiliated\s+with|under\s+(?:the\s+)?(?:jurisdiction\s+of)?|classified\s+under)\s+"
        rf"(?:{re.escape(base)})\s+(\d+)",
        re.IGNORECASE,
    ))
    # "of publisher NUM" (fallback for "jurisdiction of publisher 13")
    patterns.append(re.compile(
        rf"(?:of|with)\s+(?:{re.escape(base)})\s+(\d+)",
        re.IGNORECASE,
    ))
    # Broad fallback: "publisher affiliation as/at NUM" or just "publisher ... NUM"
    patterns.append(re.compile(
        rf"(?:{re.escape(base)})\s+(?:affiliation\s+)?(?:\w+\s+){{0,3}}(\d+)\s*(?:\(|,|\.|\s|$)",
        re.IGNORECASE,
    ))
    return patterns


class BinaryCategoryExtractor(FieldExtractor):
    """Specialized extractor for binary classification fields (e.g., carcinogenic +/-).

    Detects positive/negative classification from varied prose and maps to known values.
    Handles corrections: if preliminary says negative but final says positive, returns positive.
    """

    def __init__(self, field_name: str, concept: str,
                 positive_value: str, negative_value: str):
        super().__init__(field_name, "categorical", [])
        self.concept = concept.lower()
        self.positive_value = positive_value
        self.negative_value = negative_value
        # Match concept including morphological variants (carcinogenic/carcinogenicity)
        self._concept_re = re.compile(
            rf"\b{re.escape(self.concept)}\w*\b", re.IGNORECASE
        )
        # Negative indicators: negation words NEAR the concept
        self._neg_indicators = re.compile(
            rf"(?:non-?\s*{re.escape(self.concept)}|"
            rf"not\s+(?:\w+\s+){{0,3}}?{re.escape(self.concept)}|"
            rf"negative\s+(?:\w+\s+){{0,2}}?{re.escape(self.concept)}|"
            rf"{re.escape(self.concept)}\w*\s+(?:\w+\s+){{0,2}}?negative|"
            rf"\b(?:benign|inactive|non-?toxic|harmless|non-?reactive)\b)",
            re.IGNORECASE,
        )
        # Correction to positive
        self._correction_to_pos = re.compile(
            rf"(?:correct(?:ed|ion)|revis(?:ed)|updat(?:ed)|rectif(?:ied|y)|"
            rf"reassess(?:ed|ment)|re-?evaluat(?:ed|ion)|changed|amended|confirmed)\s+"
            rf"(?:\w+\s+){{0,8}}?(?:positive|{re.escape(self.concept)})",
            re.IGNORECASE,
        )

    def extract(self, text: str) -> Any | None:
        lower = text.lower()
        has_concept = bool(self._concept_re.search(lower))
        neg_found = bool(self._neg_indicators.search(text))

        # If neither the concept nor negative antonyms are found, skip
        if not has_concept and not neg_found:
            return None

        correction_to_pos = bool(self._correction_to_pos.search(text))
        correction_words = ("correct", "revis", "updat", "rectif", "reassess",
                            "re-evaluat", "amend", "overrid", "overrul")
        has_correction = any(w in lower for w in correction_words)

        if neg_found:
            # Negation present — but check if it's being corrected/overridden
            if correction_to_pos or has_correction:
                return self.positive_value
            return self.negative_value

        # Concept mentioned without negation → positive
        return self.positive_value


def _build_categorical_patterns(
    field_name: str, known_values: list[str] | None = None
) -> list[re.Pattern]:
    """Build patterns for categorical fields."""
    patterns = []
    if known_values:
        vals_pat = "|".join(re.escape(v) for v in known_values)
        # Pattern 1: direct mention with classification verbs (strict)
        patterns.append(re.compile(
            rf"(?:classified|categorized|designated|labeled|flagged|confirmed|"
            rf"determined|found|assessed|assigned|updated|revised|corrected)\s+"
            rf"(?:\w+\s+){{0,5}}?({vals_pat})",
            re.IGNORECASE,
        ))
        # Pattern 2: presence of a known value anywhere in paragraph
        # (broadest match — used as fallback)
        patterns.append(re.compile(
            rf"\b({vals_pat})\b",
            re.IGNORECASE,
        ))
    # Generic categorical: "classified as X"
    patterns.append(re.compile(
        rf"(?:classified|categorized|designated|labeled|flagged)\s+"
        rf"(?:as\s+)?(?:possessing\s+)?(?:a\s+)?(?:positive\s+)?"
        rf"([\w\-]+)",
        re.IGNORECASE,
    ))
    return patterns


def _build_text_patterns(field_name: str) -> list[re.Pattern]:
    """Build patterns for text fields like full_name, superhero_name."""
    base = field_name.replace("_", " ").lower()
    patterns = []
    # "full name ... is/of/as NAME"
    patterns.append(re.compile(
        rf"(?:{re.escape(base)})\s+(?:is\s+|of\s+|as\s+|for\s+this\s+\w+\s+is\s+)?"
        rf"(?:confirmed\s+(?:as|to\s+be)\s+|recorded\s+as\s+|documented\s+as\s+|"
        rf"listed\s+as\s+|logged\s+as\s+|amended\s+to\s+)?"
        rf"['\"]?([A-Z][\w\s\-'.]+?)(?:['\"]?\s*(?:\.|,|$|\s+(?:This|The|Her|His|Its|An?)))",
        re.MULTILINE,
    ))
    return patterns


def _build_date_patterns(field_name: str) -> list[re.Pattern]:
    """Build patterns for date fields (birthday, first_date, etc.)."""
    patterns = []
    # "Month Day, Year" format: "November 24th, 1937", "March 1st, 1997"
    patterns.append(re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b",
        re.IGNORECASE,
    ))
    # "the Nth of Month, Year": "the eighth of March, 1994"
    patterns.append(re.compile(
        r"\b(?:the\s+)?(\w+\s+of\s+(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December),?\s+\d{4})\b",
        re.IGNORECASE,
    ))
    # ISO format: "1994-03-08"
    patterns.append(re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"))
    # "MM/DD/YYYY" or "DD/MM/YYYY"
    patterns.append(re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"))
    return patterns


# ---------------------------------------------------------------------------
# NaN / Missing Value Detection
# ---------------------------------------------------------------------------

_NAN_INDICATORS = [
    "not available", "unavailable", "nan", "not recorded",
    "not applicable", "no data", "data anomaly", "placeholder",
    "marked as none", "field.*none", "redacted", "classified",
    "listed as 0.0", "listed as none", "marked with a placeholder",
    "marked as not available",
]


def _is_field_nan(text: str, field_name: str) -> bool:
    """Check if a specific field is marked as missing/NaN in text."""
    base = field_name.replace("_", " ").lower()
    # Find the field mention
    idx = text.lower().find(base)
    if idx < 0:
        return False
    # Check the context after it (200 chars)
    after = text[idx:idx + 250].lower()
    return any(indicator in after for indicator in _NAN_INDICATORS)


# ---------------------------------------------------------------------------
# Value Normalization
# ---------------------------------------------------------------------------

_CODED_VALUE_MAP: dict[str, dict[str, str]] = {
    "sex": {"male": "M", "m": "M", "man": "M", "boy": "M",
            "female": "F", "f": "F", "woman": "F", "girl": "F"},
    "gender": {"male": "M", "m": "M", "man": "M", "boy": "M",
               "female": "F", "f": "F", "woman": "F", "girl": "F"},
    "admission": {"+": "+", "-": "-", "yes": "+", "no": "-",
                  "admitted": "+", "inpatient": "+", "outpatient": "-"},
}


def _build_normalization_map(
    fields: list[str],
    knowledge_text: str,
    db_path: Path,
    entity_name: str,
) -> dict[str, dict[str, str]]:
    """Build field→{variant: canonical} maps from knowledge + existing DB values."""
    norm_map: dict[str, dict[str, str]] = {}

    for field in fields:
        if field == "_id":
            continue
        field_lower = field.lower()

        # Check hardcoded maps
        for key, mapping in _CODED_VALUE_MAP.items():
            if key in field_lower:
                norm_map[field] = mapping
                break

        if field in norm_map:
            continue

        # Check existing DB table for canonical values
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                for (tname,) in tables:
                    cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                    col_names = [c[1] for c in cols]
                    # Match field name to a column in an existing table
                    matching_col = None
                    for cn in col_names:
                        if cn.lower() == field_lower:
                            matching_col = cn
                            break
                    if not matching_col:
                        continue
                    # Get distinct values from this column
                    rows = conn.execute(
                        f'SELECT DISTINCT "{matching_col}" FROM "{tname}" '
                        f'WHERE "{matching_col}" IS NOT NULL LIMIT 20'
                    ).fetchall()
                    canonical_vals = [str(r[0]).strip() for r in rows if r[0]]
                    if 2 <= len(canonical_vals) <= 10:
                        # Build map: lowercase→canonical
                        mapping = {}
                        for cv in canonical_vals:
                            mapping[cv.lower()] = cv
                        norm_map[field] = mapping
                        break
                conn.close()
            except Exception:
                pass

    # Parse knowledge for value indicators like "'M' for male and 'F' for female"
    val_pattern = re.compile(
        r"['\"]([^'\"]{1,10})['\"]\s+(?:for|means?|indicates?|denotes?|represents?)\s+(\w+)",
        re.IGNORECASE,
    )
    for m in val_pattern.finditer(knowledge_text):
        canonical = m.group(1)
        meaning = m.group(2).lower()
        for field in fields:
            if field == "_id":
                continue
            field_lower = field.lower()
            matched = False
            if field_lower in ("sex", "gender") and meaning in ("male", "female"):
                matched = True
            elif meaning in field_lower or field_lower in meaning:
                matched = True
            if matched:
                if field not in norm_map:
                    norm_map[field] = {}
                norm_map[field][meaning] = canonical

    return {k: v for k, v in norm_map.items() if v}


def _match_known_values(
    records: dict[str, dict[str, Any]],
    fields: list[str],
    paragraphs: list[str],
    id_pattern: re.Pattern,
    db_path: Path,
    entity_name: str,
) -> None:
    """For TEXT columns with low cardinality, match known values against paragraph text.

    After gap-fill, some records may still have NULL fields. If the column has a small set
    of known distinct values and one of those values appears literally in the record's
    paragraph, assign it deterministically (0 LLM calls).
    """
    if not db_path or not db_path.exists():
        return

    # Build paragraph index: record_id → combined paragraph text
    para_index: dict[str, str] = {}
    for para in paragraphs:
        m = id_pattern.search(para)
        if m:
            rid = m.group(1) if m.lastindex else m.group(0)
            if rid in records:
                if rid not in para_index:
                    para_index[rid] = para
                else:
                    para_index[rid] += "\n" + para

    # Get known distinct values for low-cardinality TEXT columns
    try:
        conn = sqlite3.connect(str(db_path))
    except Exception:
        return

    try:
        for field in fields:
            if field.startswith("link_to_") or field == "_id":
                continue
            # Check if any records have this field NULL
            null_rids = [rid for rid, rec in records.items() if rec.get(field) is None]
            if not null_rids:
                continue

            # Try to get known distinct values from existing table
            distinct_vals: list[str] = []
            try:
                rows = conn.execute(
                    f'SELECT DISTINCT "{field}" FROM "{entity_name}" '
                    f'WHERE "{field}" IS NOT NULL AND "{field}" != "" LIMIT 50'
                ).fetchall()
                distinct_vals = [str(r[0]) for r in rows if r[0]]
            except Exception:
                pass

            # Also gather values already extracted for this field
            extracted_vals: set[str] = set()
            for rec in records.values():
                v = rec.get(field)
                if v is not None:
                    extracted_vals.add(str(v))
            all_vals = list(set(distinct_vals) | extracted_vals)
            if not all_vals or len(all_vals) > 30:
                continue

            # Sort by length descending to prefer longer (more specific) matches
            all_vals.sort(key=len, reverse=True)

            for rid in null_rids:
                para_text = para_index.get(rid, "")
                if not para_text:
                    continue
                for val in all_vals:
                    if len(val) < 3:
                        continue
                    if val in para_text:
                        records[rid][field] = val
                        break
    finally:
        conn.close()


_NUMERIC_FIELD_PATTERNS: dict[str, list[re.Pattern]] = {
    "amount": [
        re.compile(r'(?:allocated|amount[^.]{0,20}?|budget[^.]{0,20}?|total[^.]{0,10}?)\s+(\d+(?:\.\d+)?)\b'),
        re.compile(r'\b(\d+(?:\.\d+)?)\s+(?:was\s+)?(?:allocated|budgeted)'),
    ],
    "spent": [
        re.compile(r'(\d+(?:\.\d+)?)\s+has been spent'),
        re.compile(r'spent[^.]{0,10}?(\d+(?:\.\d+)?)'),
    ],
    "remaining": [
        re.compile(r'(\d+(?:\.\d+)?)\s+remaining'),
        re.compile(r'remaining[^.]{0,10}?(\d+(?:\.\d+)?)'),
    ],
}


def _extract_numeric_from_context(
    records: dict[str, dict[str, Any]],
    fields: list[str],
    paragraphs: list[str],
    id_pattern: re.Pattern,
) -> None:
    """Extract numeric values from paragraph context using field-name-specific patterns."""
    numeric_fields = [f for f in fields if f.lower() in _NUMERIC_FIELD_PATTERNS]
    if not numeric_fields:
        return

    # Build paragraph index
    para_index: dict[str, str] = {}
    for para in paragraphs:
        m = id_pattern.search(para)
        if m:
            rid = m.group(1) if m.lastindex else m.group(0)
            if rid in records:
                if rid not in para_index:
                    para_index[rid] = para
                else:
                    para_index[rid] += "\n" + para

    for field in numeric_fields:
        patterns = _NUMERIC_FIELD_PATTERNS[field.lower()]
        null_rids = [rid for rid, rec in records.items() if rec.get(field) is None]
        for rid in null_rids:
            para_text = para_index.get(rid, "")
            if not para_text:
                continue
            for pat in patterns:
                m = pat.search(para_text)
                if m:
                    try:
                        records[rid][field] = float(m.group(1))
                    except ValueError:
                        pass
                    break


def _normalize_values(
    records: dict[str, dict[str, Any]],
    fields: list[str],
    knowledge_text: str,
    db_path: Path,
    entity_name: str,
) -> None:
    """Normalize extracted values to match canonical forms in the DB schema."""
    norm_map = _build_normalization_map(fields, knowledge_text, db_path, entity_name)
    if not norm_map:
        return

    for rid, rec in records.items():
        for field, mapping in norm_map.items():
            val = rec.get(field)
            if val is None:
                continue
            val_str = str(val).strip()
            val_lower = val_str.lower()
            if val_lower in mapping:
                rec[field] = mapping[val_lower]
            elif val_str in mapping:
                rec[field] = mapping[val_str]


# ---------------------------------------------------------------------------
# Main Extraction Engine
# ---------------------------------------------------------------------------

_MAX_LLM_CALLS = 10


def regex_extract(
    doc_path: Path,
    db_path: Path,
    model: ModelAdapter,
    question: str,
    knowledge_text: str = "",
    log_fn: Callable[[str, str], None] | None = None,
    protected_tables: set[str] | None = None,
) -> int:
    """Extract structured data from a document using calibrated regex + LLM.

    Budget: up to 10 LLM calls total.
    Returns number of records written.
    """
    try:
        text = doc_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    if len(text) < 100:
        return 0

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 50]
    if not paragraphs:
        return 0

    if log_fn:
        log_fn("regex_start", f"{doc_path.stem}: {len(text)} chars, {len(paragraphs)} paragraphs")

    llm_calls_used = 0

    # === PHASE 1: PLAN (1 call) ===
    plan = _get_plan(model, text, question, knowledge_text, db_path, log_fn, doc_path.stem)
    llm_calls_used += 1
    if not plan:
        if log_fn:
            log_fn("regex_skip", "Planner returned no plan")
        return 0

    entity_name = plan.get("entity", doc_path.stem)
    fields = plan.get("fields", [])
    if "_id" not in fields:
        fields.insert(0, "_id")

    # Add knowledge-defined fields that aren't already in the plan or the main
    # structured table. Small lookup tables don't block — they may have incomplete data.
    fields_lower = {f.lower() for f in fields}
    main_table_cols: set[str] = set()
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            biggest_table = ""
            biggest_count = 0
            for (tname,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                cnt = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
                if cnt > biggest_count:
                    biggest_count = cnt
                    biggest_table = tname
            if biggest_table:
                for col in conn.execute(f'PRAGMA table_info("{biggest_table}")').fetchall():
                    main_table_cols.add(col[1].lower().replace("-", "_"))
            conn.close()
        except Exception:
            pass
    for m in re.finditer(r"[-*]\s+\*{0,2}(\w[\w\s\-]*?)\s*\((\w+)\)\s*\*{0,2}\s*:", knowledge_text):
        kf = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        if kf and kf not in fields_lower and kf not in main_table_cols:
            fields.append(kf)
            fields_lower.add(kf)

    # Inject FK field when structured tables have IDs that appear in this document.
    # E.g., if event.event_id values appear in budget.md, add "link_to_event" field.
    if db_path.exists() and "link_to_" not in " ".join(fields_lower):
        try:
            conn = sqlite3.connect(str(db_path))
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for (tname,) in tables:
                cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                id_col = None
                for c in cols:
                    if c[1].lower() in ("id", "_id", "event_id", f"{tname.lower()}_id"):
                        id_col = c[1]
                        break
                if not id_col:
                    continue
                sample_vals = [
                    str(r[0]) for r in conn.execute(
                        f'SELECT DISTINCT "{id_col}" FROM "{tname}" '
                        f'WHERE "{id_col}" IS NOT NULL LIMIT 10'
                    ).fetchall() if r[0]
                ]
                if not sample_vals:
                    continue
                matches_in_doc = sum(1 for v in sample_vals if v in text)
                if matches_in_doc >= 2:
                    fk_field = f"link_to_{tname.lower()}"
                    if fk_field not in fields_lower:
                        fields.append(fk_field)
                        fields_lower.add(fk_field)
                    break
            conn.close()
        except Exception:
            pass

    # If planner returned an entity that matches an existing structured table,
    # override with the doc stem (the doc likely has DIFFERENT data about that entity)
    if protected_tables and entity_name.lower() in protected_tables:
        entity_name = doc_path.stem.lower()

    # Field budget: with limited LLM calls, extracting too many fields produces
    # sparse results. When >8 fields planned, trim to question-relevant ones.
    _MAX_FIELDS = 8
    if len(fields) > _MAX_FIELDS and question:
        q_lower = question.lower()
        essential = {"_id", "date", "name", "title"}
        # Expand essential with age/demographic fields when question implies age
        # AND this doc is about persons/patients (not labs or events)
        person_entities = {"patient", "person", "people", "employee", "student",
                          "driver", "member", "user", "staff", "worker"}
        is_person_doc = entity_name.lower() in person_entities
        if is_person_doc:
            age_words = {"age", "aged", "old", "young", "year", "born", "elder"}
            has_age_ref = (
                any(w in q_lower for w in age_words)
                or bool(re.search(
                    r"(?:aren't|isn't|not|under|over|above|below)\s+\d{1,3}\b"
                    r"|\b\d{1,3}\s+(?:years?\s+old|year-old)",
                    q_lower,
                ))
            )
            if has_age_ref:
                essential.update({"birthday", "dob", "birth_date", "date_of_birth"})
        relevant = [f for f in fields if f.lower() in essential]
        for f in fields:
            if f.lower() in essential:
                continue
            # Keep FK fields (id, *_id, link_to_*)
            fl = f.lower()
            if fl == "id" or fl.endswith("_id") or f.startswith("link_to_"):
                if f not in relevant:
                    relevant.append(f)
                continue
            f_words = fl.replace("_", " ").split()
            if any(w in q_lower for w in f_words if len(w) > 2):
                relevant.append(f)
        if len(relevant) >= 2:
            fields = relevant
            if "_id" not in fields:
                fields.insert(0, "_id")

    if log_fn:
        log_fn("regex_plan", f"entity={entity_name}, fields={fields}")

    # === PHASE 2: DISCOVER ID PATTERN ===
    sample_ids = _get_sample_ids_from_db(db_path, entity_name)
    id_pattern = _discover_id_pattern(text, sample_ids, entity_name=entity_name)
    if not id_pattern:
        if log_fn:
            log_fn("regex_skip", "Could not discover ID pattern")
        return 0

    if log_fn:
        test_ids = id_pattern.findall(text[:5000])[:5]
        log_fn("regex_id_pattern", f"Pattern: {id_pattern.pattern[:80]}, samples: {test_ids}")

    # === PHASE 3: CALIBRATE (2 calls) ===
    # Pick diverse sample paragraphs, extract via LLM, use results to
    # learn document-specific extraction patterns
    extractors = _build_extractors(fields, entity_name, db_path, knowledge_text)
    non_id_fields = [f for f in fields if f != "_id"]

    if non_id_fields and llm_calls_used < _MAX_LLM_CALLS:
        calibration_records, calls = _calibrate(
            model, paragraphs, id_pattern, non_id_fields, entity_name, log_fn
        )
        llm_calls_used += calls
        if calibration_records and log_fn:
            log_fn("regex_calibrate", f"Calibrated from {len(calibration_records)} LLM-extracted records")
    else:
        calibration_records = {}

    if log_fn:
        log_fn("regex_extractors", f"Built {len(extractors)} field extractors")

    # === PHASE 4: REGEX PASS (0 calls) ===
    records: dict[str, dict[str, Any]] = {}

    # Seed with calibration records first
    for rid, rec in calibration_records.items():
        records[rid] = rec

    for para in paragraphs:
        id_match = id_pattern.search(para)
        if not id_match:
            continue

        record_id = id_match.group(1) if id_match.lastindex else id_match.group(0)

        if record_id not in records:
            records[record_id] = {"_id": record_id}

        for extractor in extractors:
            fname = extractor.field_name
            if fname == "_id":
                continue

            existing = records[record_id].get(fname)

            if _is_field_nan(para, fname):
                if existing is None:
                    records[record_id][fname] = None
                continue

            value = extractor.extract(para)
            if value is not None:
                records[record_id][fname] = value

    if log_fn:
        log_fn("regex_extracted", f"{len(records)} records after regex pass")

    # === PHASE 4b: FK EXTRACTION (0 calls) ===
    # For link_to_* fields, find additional same-format IDs in each paragraph.
    # When a paragraph contains the record's own ID + another ID of the same format,
    # the second ID is likely the FK value.
    fk_fields = [f for f in non_id_fields if f.startswith("link_to_")]
    if fk_fields and id_pattern:
        # Get the set of valid FK target IDs from the referenced table
        fk_target_ids: set[str] = set()
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                for fk_field in fk_fields:
                    ref_table = fk_field.replace("link_to_", "")
                    tables = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                    for (tname,) in tables:
                        if tname.lower() == ref_table:
                            cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                            id_col = next(
                                (c[1] for c in cols if c[1].lower() in ("id", "_id", f"{ref_table}_id")),
                                cols[0][1] if cols else None,
                            )
                            if id_col:
                                rows = conn.execute(
                                    f'SELECT DISTINCT "{id_col}" FROM "{tname}" WHERE "{id_col}" IS NOT NULL'
                                ).fetchall()
                                fk_target_ids.update(str(r[0]) for r in rows if r[0])
                            break
                conn.close()
            except Exception:
                pass
        if fk_target_ids:
            for para in paragraphs:
                all_ids = id_pattern.findall(para)
                if len(all_ids) < 2:
                    continue
                record_id = all_ids[0]
                if record_id not in records:
                    continue
                for other_id in all_ids[1:]:
                    if other_id != record_id and other_id in fk_target_ids:
                        for fk_field in fk_fields:
                            if records[record_id].get(fk_field) is None:
                                records[record_id][fk_field] = other_id
                        break

    # === PHASE 5: GAP-FILL (up to 6 calls) ===
    # Find records with missing critical fields and batch them for LLM extraction
    remaining_budget = _MAX_LLM_CALLS - llm_calls_used - 1  # reserve 1 for validation
    if remaining_budget > 0 and non_id_fields:
        gap_calls = _gap_fill(
            model, paragraphs, id_pattern, records, non_id_fields,
            entity_name, remaining_budget, log_fn, db_path=db_path,
        )
        llm_calls_used += gap_calls

    if log_fn:
        log_fn("regex_after_gap_fill", f"{len(records)} records, {llm_calls_used} LLM calls used")

    if not records:
        return 0

    # === PHASE 5a: DETERMINISTIC VALUE MATCHING (0 calls) ===
    # For TEXT columns with low cardinality, match known values against paragraph text.
    # For numeric columns, extract numbers from contextual patterns.
    _match_known_values(records, non_id_fields, paragraphs, id_pattern, db_path, entity_name)
    _extract_numeric_from_context(records, non_id_fields, paragraphs, id_pattern)

    # === PHASE 5b: VALUE NORMALIZATION ===
    _normalize_values(records, fields, knowledge_text, db_path, entity_name)

    # === PHASE 6: WRITE TO SQLITE (merge-into-existing or create new) ===
    record_list = list(records.values())

    written = _merge_or_create(db_path, doc_path.stem, record_list, protected_tables, log_fn)
    if log_fn:
        log_fn("regex_written", f"Wrote {written} records ({llm_calls_used} LLM calls)")

    return written


# ---------------------------------------------------------------------------
# Calibration Phase (2 LLM calls)
# ---------------------------------------------------------------------------

def _calibrate(
    model: ModelAdapter,
    paragraphs: list[str],
    id_pattern: re.Pattern,
    fields: list[str],
    entity_name: str,
    log_fn: Callable[[str, str], None] | None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Extract a few sample paragraphs via LLM to calibrate extraction.

    Picks 3 diverse paragraphs (start, middle, end) and asks LLM to extract.
    Returns (records_dict, num_llm_calls_used).
    """
    # Pick paragraphs with IDs from different parts of document
    id_paras: list[tuple[int, str, str]] = []  # (index, id, text)
    for i, para in enumerate(paragraphs):
        m = id_pattern.search(para)
        if m:
            rid = m.group(1) if m.lastindex else m.group(0)
            id_paras.append((i, rid, para))

    if len(id_paras) < 3:
        return {}, 0

    # Sample from start, middle, end (diverse styles)
    indices = [0, len(id_paras) // 3, 2 * len(id_paras) // 3, len(id_paras) - 1]
    samples = []
    seen_ids = set()
    for idx in indices:
        if idx < len(id_paras) and id_paras[idx][1] not in seen_ids:
            samples.append(id_paras[idx])
            seen_ids.add(id_paras[idx][1])
        if len(samples) >= 4:
            break

    if not samples:
        return {}, 0

    # Build calibration prompt (batch all samples in 1 call)
    field_list = ", ".join(fields)
    para_texts = []
    for i, (_, rid, para_text) in enumerate(samples):
        para_texts.append(f"--- PARAGRAPH {i+1} (ID: {rid}) ---\n{para_text[:800]}")

    prompt = f"""Extract data from these text blocks. Each describes one {entity_name}.
Fields: {field_list}. Use null if not mentioned. If corrected, use final value.

{chr(10).join(para_texts)}

Respond with ONLY a JSON array:
[{{"_id": "ID_VALUE", {", ".join(f'"{f}": VALUE' for f in fields)}}}]"""

    messages = [ModelMessage(role="user", content=prompt)]
    calls = 0
    try:
        raw = model.complete(messages)
        calls += 1
    except Exception:
        return {}, 1

    # Parse
    records: dict[str, dict[str, Any]] = {}
    data = _parse_llm_json(raw)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "_id" in item:
                records[str(item["_id"])] = {
                    k: v for k, v in item.items() if k == "_id" or k in fields
                }

    # Second calibration call: pick a paragraph with a correction/revision
    correction_paras = []
    for _, rid, para_text in id_paras:
        if rid in records:
            continue
        lower = para_text.lower()
        if any(w in lower for w in ("correct", "revis", "updat", "initially", "originally")):
            correction_paras.append((rid, para_text))
            if len(correction_paras) >= 3:
                break

    if correction_paras and calls < 2:
        para_texts2 = []
        for i, (rid, pt) in enumerate(correction_paras):
            para_texts2.append(f"--- PARAGRAPH {i+1} (ID: {rid}) ---\n{pt[:800]}")

        prompt2 = f"""Extract data from these texts. They contain corrections — use the FINAL value only.
Fields: {field_list}. Use null if not mentioned.

{chr(10).join(para_texts2)}

Respond with ONLY a JSON array:
[{{"_id": "ID_VALUE", {", ".join(f'"{f}": VALUE' for f in fields)}}}]"""

        messages2 = [ModelMessage(role="user", content=prompt2)]
        try:
            raw2 = model.complete(messages2)
            calls += 1
            data2 = _parse_llm_json(raw2)
            if isinstance(data2, list):
                for item in data2:
                    if isinstance(item, dict) and "_id" in item:
                        records[str(item["_id"])] = {
                            k: v for k, v in item.items() if k == "_id" or k in fields
                        }
        except Exception:
            pass

    if log_fn:
        log_fn("regex_calibrate_done", f"{len(records)} records from {calls} calibration calls")

    return records, calls


# ---------------------------------------------------------------------------
# Gap-Fill Phase (up to N LLM calls)
# ---------------------------------------------------------------------------

def _gap_fill(
    model: ModelAdapter,
    paragraphs: list[str],
    id_pattern: re.Pattern,
    records: dict[str, dict[str, Any]],
    fields: list[str],
    entity_name: str,
    max_calls: int,
    log_fn: Callable[[str, str], None] | None,
    db_path: Path | None = None,
) -> int:
    """Fill gaps in records where regex missed critical fields.

    Strategy: identify fields with low coverage, then batch-extract those
    from paragraphs where the field is missing.
    Prioritizes records referenced by FK columns in existing structured tables.
    Returns number of LLM calls made.
    """
    if not records:
        return 0

    # Compute per-field coverage
    field_coverage: dict[str, float] = {}
    for f in fields:
        filled = sum(1 for r in records.values() if r.get(f) is not None)
        field_coverage[f] = filled / len(records) if records else 0

    # Target fields below 80% coverage (significant gaps)
    gap_fields = [f for f in fields if field_coverage[f] < 0.80]
    if not gap_fields:
        # All fields well-covered; check for entirely missing records
        all_para_ids: set[str] = set()
        for para in paragraphs:
            m = id_pattern.search(para)
            if m:
                rid = m.group(1) if m.lastindex else m.group(0)
                all_para_ids.add(rid)
        missing_ids = all_para_ids - set(records.keys())
        if not missing_ids:
            return 0
        gap_fields = fields  # extract everything for missing records

    if log_fn:
        cov_str = ", ".join(f"{f}={field_coverage.get(f, 0):.0%}" for f in fields)
        log_fn("regex_gap_analysis", f"Coverage: {cov_str}. Targeting: {gap_fields}")

    # Identify records with missing gap fields
    gap_ids: set[str] = set()
    for rid, rec in records.items():
        if any(rec.get(f) is None for f in gap_fields):
            gap_ids.add(rid)

    # Also target records found by regex with ID but no other fields populated
    for rid, rec in records.items():
        if all(rec.get(f) is None for f in fields):
            gap_ids.add(rid)

    if not gap_ids:
        return 0

    # Collect paragraphs for gap IDs
    gap_paras: dict[str, list[str]] = {rid: [] for rid in gap_ids}
    for para in paragraphs:
        m = id_pattern.search(para)
        if m:
            rid = m.group(1) if m.lastindex else m.group(0)
            if rid in gap_paras:
                gap_paras[rid].append(para)

    # Batch gap paragraphs — 8 entities per call (conservative for small models)
    batch_size = 8
    gap_items = [(rid, "\n".join(ps[:2])[:600]) for rid, ps in gap_paras.items() if ps]

    # Prioritize records referenced by FK columns in structured tables
    referenced_ids: set[str] = set()
    if db_path and db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            entity_lower = entity_name.lower()
            for (tname,) in tables:
                cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                for col in cols:
                    cn = col[1].lower()
                    if entity_lower in cn and ("link" in cn or "id" in cn or "key" in cn):
                        rows = conn.execute(
                            f'SELECT DISTINCT "{col[1]}" FROM "{tname}" '
                            f'WHERE "{col[1]}" IS NOT NULL'
                        ).fetchall()
                        referenced_ids.update(str(r[0]) for r in rows if r[0])
                        break
                if referenced_ids:
                    break
            conn.close()
        except Exception:
            pass
    if referenced_ids:
        gap_items.sort(key=lambda x: (0 if x[0] in referenced_ids else 1))

    calls_made = 0
    consecutive_empty = 0

    for batch_start in range(0, len(gap_items), batch_size):
        if calls_made >= max_calls:
            break
        if consecutive_empty >= 2:
            if log_fn:
                log_fn("regex_gap_stop", "Stopping gap-fill: 2 consecutive empty batches")
            break

        batch = gap_items[batch_start:batch_start + batch_size]
        field_list = ", ".join(f'"{f}"' for f in fields)

        para_texts = []
        for i, (rid, para_text) in enumerate(batch):
            para_texts.append(f"[{rid}]\n{para_text}")

        prompt = f"""Extract values from each text block below.
Each block is about one {entity_name}. Extract these fields: {field_list}

If corrected/revised, use the FINAL value. Use null if not mentioned.

{chr(10).join(para_texts)}

Respond with ONLY a JSON array:
[{{"_id": "ID", {", ".join(f'"{f}": VALUE' for f in fields)}}}]"""

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = model.complete(messages)
            calls_made += 1
        except Exception:
            calls_made += 1
            continue

        data = _parse_llm_json(raw)
        if isinstance(data, list):
            filled_count = 0
            for item in data:
                if not isinstance(item, dict) or "_id" not in item:
                    continue
                rid = str(item["_id"])
                if rid not in records:
                    continue
                for f in fields:
                    val = item.get(f)
                    if val is not None and records[rid].get(f) is None:
                        records[rid][f] = val
                        filled_count += 1
            if log_fn:
                log_fn("regex_gap_batch", f"Filled {filled_count} values from batch of {len(batch)}")
            if filled_count == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0
        else:
            consecutive_empty += 1
            if log_fn:
                log_fn("regex_gap_error", "Failed to parse gap-fill response")

    return calls_made


# ---------------------------------------------------------------------------
# Planner (1 LLM call)
# ---------------------------------------------------------------------------

def _get_plan(
    model: ModelAdapter,
    text: str,
    question: str,
    knowledge_text: str,
    db_path: Path,
    log_fn: Callable[[str, str], None] | None,
    doc_stem: str = "",
) -> dict[str, Any] | None:
    """Single LLM call to determine extraction schema."""
    # Get DB context — concise list of existing tables and their columns
    db_context = ""
    existing_cols: set[str] = set()
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            parts = []
            for (tname,) in tables:
                cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                col_names = [c[1] for c in cols]
                existing_cols.update(c.lower().replace("-", "_") for c in col_names)
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
                parts.append(f"TABLE: {tname} ({row_count} rows) — columns: {', '.join(col_names)}")
            conn.close()
            db_context = "\n".join(parts)
        except Exception:
            pass

    # Get diverse sample from text
    sample_paras = []
    n = len(text.split("\n\n"))
    step = max(1, n // 6)
    for i, para in enumerate(text.split("\n\n")):
        if i % step == 0 and len(para.strip()) > 60:
            sample_paras.append(para.strip()[:600])
        if len(sample_paras) >= 6:
            break
    sample = "\n\n---\n\n".join(sample_paras)

    knowledge_hint = ""
    if knowledge_text:
        knowledge_hint = f"RELEVANT KNOWLEDGE:\n{knowledge_text[:1500]}"

    doc_hint = f"\nDOCUMENT NAME: {doc_stem}" if doc_stem else ""
    prompt = f"""I need to know the TABLE SCHEMA for data in this document. Do NOT extract data — just tell me the column names.
{doc_hint}
QUESTION (for context only — extract ALL fields from doc, not just those in the question): {question}

EXISTING DATABASE TABLES (already loaded — do NOT duplicate these columns):
{db_context[:600]}

{knowledge_hint}

SAMPLE TEXT FROM DOCUMENT:
{sample[:3000]}

What UNIQUE information does this document contain that is NOT already in the database tables above?
Respond with exactly ONE JSON object (not an array) in this format:
{{"entity": "TABLE_NAME", "fields": ["_id", "column1", "column2"]}}

Example response: {{"entity": "patient", "fields": ["_id", "sex", "birthday", "admission", "diagnosis"]}}

Rules:
- "entity" = singular lowercase name matching document subject (e.g. "patient", "superhero")
- "_id" is always first (the entity's identifier)
- Do NOT include columns that already exist in the DATABASE TABLES above UNLESS the column is "name" or "title" (these are commonly needed for lookups even if another table has a column with the same name)
- Focus on demographic/attribute columns visible in the SAMPLE TEXT (sex, birthday, dates, categories)
- Use exact column names from KNOWLEDGE if available (e.g. "height_cm" not "height")
- Only include columns whose VALUES appear in the sample text
- Do NOT extract actual data, just list the column names"""

    messages = [ModelMessage(role="user", content=prompt)]
    try:
        raw = model.complete(messages)
    except Exception as e:
        if log_fn:
            log_fn("regex_planner_error", f"LLM call failed: {e}")
        return None

    # Parse response
    plan = _parse_llm_json(raw)
    if isinstance(plan, dict) and "fields" in plan:
        # Deduplicate: remove fields that already exist in DB (except _id)
        if existing_cols and len(plan.get("fields", [])) > 2:
            # Keep descriptor columns (name, title, etc.) even if they exist in other tables —
            # they're often the lookup value needed for FK resolution
            keep_always = {"name", "title", "description", "label"}
            unique_fields = [
                f for f in plan["fields"]
                if f == "_id" or f.lower() in keep_always
                or f.lower().replace("-", "_") not in existing_cols
            ]
            if len(unique_fields) >= 2:
                plan["fields"] = unique_fields
            else:
                # All fields are duplicates — planner likely misidentified the entity.
                # Use doc-stem as entity and infer minimal demographic fields from text.
                plan["entity"] = doc_stem.lower() if doc_stem else plan.get("entity", "entity")
                demo_fields = ["_id"]
                text_lower = text[:10000].lower()
                if "male" in text_lower or "female" in text_lower:
                    demo_fields.append("sex")
                if "born" in text_lower or "birth" in text_lower:
                    demo_fields.append("birthday")
                if "admitted" in text_lower or "admission" in text_lower:
                    demo_fields.append("admission")
                if "diagnos" in text_lower:
                    demo_fields.append("diagnosis")
                if "date" in text_lower or "initiat" in text_lower:
                    demo_fields.append("first_date")
                if len(demo_fields) >= 2:
                    plan["fields"] = demo_fields
                    if log_fn:
                        log_fn("regex_plan_override", f"Planner returned duplicate cols, using doc-inferred: {demo_fields}")
        return plan
    # Fallback: if LLM returned an array of records, infer schema from keys
    if isinstance(plan, list) and plan:
        all_keys: set[str] = set()
        entity = None
        for item in plan:
            if isinstance(item, dict):
                all_keys.update(item.keys())
                if not entity and "entity" in item:
                    entity = item["entity"]
        all_keys.discard("entity")
        if all_keys:
            fields = sorted(all_keys)
            if "_id" in fields:
                fields.remove("_id")
                fields.insert(0, "_id")
            return {"entity": entity or "entity", "fields": fields}
    if log_fn:
        log_fn("regex_planner_raw", f"Could not parse: {raw[:300]}")
    return None


# ---------------------------------------------------------------------------
# Extractor Builder
# ---------------------------------------------------------------------------

def _get_sample_ids_from_db(db_path: Path, entity_name: str) -> list[str]:
    """Get sample FK values that reference this entity from structured tables."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        entity_lower = entity_name.lower()
        for (tname,) in tables:
            cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
            for col in cols:
                col_name = col[1].lower()
                # Look for FK columns referencing this entity
                if entity_lower in col_name and ("id" in col_name or "key" in col_name or "link" in col_name):
                    # Sample from different parts of the range to avoid
                    # prefix bias (e.g., LIMIT 20 on sorted IDs gives TR000-TR019)
                    total = conn.execute(
                        f'SELECT COUNT(DISTINCT "{col[1]}") FROM "{tname}" '
                        f'WHERE "{col[1]}" IS NOT NULL'
                    ).fetchone()[0]
                    if total <= 20:
                        rows = conn.execute(
                            f'SELECT DISTINCT "{col[1]}" FROM "{tname}" '
                            f'WHERE "{col[1]}" IS NOT NULL'
                        ).fetchall()
                    else:
                        # First 10 + last 10 for prefix diversity
                        rows_first = conn.execute(
                            f'SELECT DISTINCT "{col[1]}" FROM "{tname}" '
                            f'WHERE "{col[1]}" IS NOT NULL LIMIT 10'
                        ).fetchall()
                        rows_last = conn.execute(
                            f'SELECT DISTINCT "{col[1]}" FROM "{tname}" '
                            f'WHERE "{col[1]}" IS NOT NULL '
                            f'ORDER BY "{col[1]}" DESC LIMIT 10'
                        ).fetchall()
                        rows = rows_first + rows_last
                    conn.close()
                    return [str(r[0]) for r in rows if r[0]]
            # Also check: table name contains entity and has an ID column
            if entity_lower in tname.lower():
                id_col = next((c[1] for c in cols if c[1].lower() == "id"), None)
                if id_col:
                    total = conn.execute(
                        f'SELECT COUNT(DISTINCT "{id_col}") FROM "{tname}" '
                        f'WHERE "{id_col}" IS NOT NULL'
                    ).fetchone()[0]
                    if total <= 20:
                        rows = conn.execute(
                            f'SELECT DISTINCT "{id_col}" FROM "{tname}" '
                            f'WHERE "{id_col}" IS NOT NULL'
                        ).fetchall()
                    else:
                        rows_first = conn.execute(
                            f'SELECT DISTINCT "{id_col}" FROM "{tname}" '
                            f'WHERE "{id_col}" IS NOT NULL LIMIT 10'
                        ).fetchall()
                        rows_last = conn.execute(
                            f'SELECT DISTINCT "{id_col}" FROM "{tname}" '
                            f'WHERE "{id_col}" IS NOT NULL '
                            f'ORDER BY "{id_col}" DESC LIMIT 10'
                        ).fetchall()
                        rows = rows_first + rows_last
                    conn.close()
                    return [str(r[0]) for r in rows if r[0]]
        conn.close()
    except Exception:
        pass
    return []


def _detect_binary_field(field_name: str, knowledge_text: str) -> tuple[str, str, str] | None:
    """Detect if a field is a binary classification from knowledge text.

    Returns (concept, positive_value, negative_value) or None.
    Looks for patterns like: "carcinogenic ('+') or non-carcinogenic ('-')"
    """
    field_lower = field_name.lower()
    # Pattern: "field_name ... X ('val1') or Y ('val2')"
    # or: "field_name = 'val1' for X and 'val2' for Y"
    patterns = [
        # "label": ... carcinogenic ('+') or non-carcinogenic ('-')
        re.compile(
            rf"\*\*{re.escape(field_lower)}\*\*[^.]*?"
            rf"(\w[\w\-]+)\s*\(['\"]([^'\"]+)['\"]\)\s*(?:or|and)\s*"
            rf"(?:non-?)?(\w[\w\-]+)\s*\(['\"]([^'\"]+)['\"]\)",
            re.IGNORECASE,
        ),
        # Use `label = '+'` for carcinogenic and `label = '-'` for non-carcinogenic
        re.compile(
            rf"{re.escape(field_lower)}\s*=\s*['\"]([^'\"]+)['\"]\s*"
            rf"(?:for|means?|indicates?)\s+(\w[\w\-]+)\s*"
            rf".*?{re.escape(field_lower)}\s*=\s*['\"]([^'\"]+)['\"]\s*"
            rf"(?:for|means?|indicates?)\s+(?:non-?)?(\w[\w\-]+)",
            re.IGNORECASE | re.DOTALL,
        ),
    ]
    for pat in patterns:
        m = pat.search(knowledge_text)
        if m:
            groups = m.groups()
            if len(groups) == 4:
                # First pattern: concept1 (val1) or concept2 (val2)
                concept = groups[0].lower()
                pos_val = groups[1]
                neg_val = groups[3]
                return (concept, pos_val, neg_val)

    # Fallback: look for "Use `field = 'X'` for CONCEPT" style
    filter_pat = re.compile(
        rf"{re.escape(field_lower)}\s*=\s*['\"]([^'\"]+)['\"]\D*?(\w+)",
        re.IGNORECASE,
    )
    matches = filter_pat.findall(knowledge_text)
    if len(matches) >= 2:
        val1, concept1 = matches[0]
        val2, concept2 = matches[1]
        # Positive = first mentioned (usually the "active" one)
        return (concept1.lower(), val1, val2)

    return None


def _build_extractors(
    fields: list[str],
    entity_name: str,
    db_path: Path,
    knowledge_text: str,
) -> list[FieldExtractor]:
    """Build field extractors based on field names and knowledge."""
    extractors: list[FieldExtractor] = []

    # Parse knowledge for field types
    field_types: dict[str, str] = {}
    field_units: dict[str, str] = {}
    for m in re.finditer(
        r"\*\*(\w[\w\s\-]*?)\s*\((\w+)\)\s*[:.]?\*\*", knowledge_text
    ):
        fname = m.group(1).strip().lower().replace(" ", "_")
        ftype = m.group(2).lower()
        field_types[fname] = ftype
    # Also check "(integer)" mentions
    for m in re.finditer(r"[*-]\s*(\w+)\s*\((\w+)\)", knowledge_text):
        fname = m.group(1).strip().lower()
        ftype = m.group(2).lower()
        field_types[fname] = ftype

    for field in fields:
        if field == "_id":
            continue

        field_lower = field.lower()
        ftype = field_types.get(field_lower, "")

        # Check for binary classification field first
        binary_info = _detect_binary_field(field, knowledge_text)
        if binary_info:
            concept, pos_val, neg_val = binary_info
            extractors.append(BinaryCategoryExtractor(
                field, concept, pos_val, neg_val
            ))
            continue

        # Determine field type from name heuristics
        if field_lower in ("sex", "gender"):
            # Sex/gender: extract "male"/"female" from prose
            patterns = [
                re.compile(r"\b(male|female)\b", re.IGNORECASE),
                re.compile(r"\b(man|woman)\b", re.IGNORECASE),
                re.compile(r"\bsex[:\s]+['\"]?(\w+)", re.IGNORECASE),
            ]
            extractors.append(FieldExtractor(field, "categorical", patterns))
            continue
        elif any(d in field_lower for d in ("birthday", "birth_date", "dob", "date_of_birth")):
            patterns = _build_date_patterns(field)
            extractors.append(FieldExtractor(field, "text", patterns))
            continue
        elif "date" in field_lower or "first_date" == field_lower:
            patterns = _build_date_patterns(field)
            extractors.append(FieldExtractor(field, "text", patterns))
            continue
        elif field_lower in ("admission",) and "'+'" in knowledge_text:
            patterns = [
                re.compile(r"\b(admitted|inpatient|\+)\b", re.IGNORECASE),
                re.compile(r"\b(outpatient|follow[- ]?up|-)\b", re.IGNORECASE),
            ]
            extractors.append(FieldExtractor(field, "categorical", patterns))
            continue
        elif field_lower in ("diagnosis",):
            patterns = [
                re.compile(r"diagnos(?:ed|is)\s+(?:with\s+|as\s+|of\s+)?([A-Z][\w\s\-]+?)(?:\.|,|$)", re.MULTILINE),
            ]
            extractors.append(FieldExtractor(field, "text", patterns))
            continue

        if field_lower.endswith("_id") or field_lower.endswith("_key"):
            # FK / integer code field
            patterns = _build_integer_code_patterns(field)
            extractors.append(FieldExtractor(field, "integer", patterns))
        elif ftype in ("integer", "real", "numeric", "float") or any(
            s in field_lower for s in ("_cm", "_kg", "_mm", "_ml", "_mg")
        ):
            # Numeric field
            unit = ""
            if "cm" in field_lower:
                unit = "centimeters"
            elif "kg" in field_lower:
                unit = "kilograms"
            elif "mm" in field_lower:
                unit = "millimeters"
            elif "ml" in field_lower:
                unit = "milliliters"
            patterns = _build_numeric_patterns(field, unit)
            extractors.append(FieldExtractor(field, "numeric", patterns, unit=unit))
        elif "name" in field_lower or "title" in field_lower:
            patterns = _build_text_patterns(field)
            extractors.append(FieldExtractor(field, "text", patterns))
        elif "category" in field_lower or "type" in field_lower or "label" in field_lower:
            patterns = _build_categorical_patterns(field)
            extractors.append(FieldExtractor(field, "categorical", patterns))
        elif ftype == "text":
            patterns = _build_text_patterns(field)
            extractors.append(FieldExtractor(field, "text", patterns))
        else:
            # Default: try numeric then text
            patterns = _build_numeric_patterns(field)
            extractors.append(FieldExtractor(field, "numeric", patterns))

    return extractors


# ---------------------------------------------------------------------------
# SQLite Writer (merge-into-existing or create new)
# ---------------------------------------------------------------------------

def _find_merge_target(
    db_path: Path,
    records: list[dict[str, Any]],
    protected_tables: set[str] | None,
) -> tuple[str, str] | None:
    """Find an existing table whose IDs overlap >50% with extracted records.

    Prefers small lookup tables (few columns) over large fact tables.
    Returns (table_name, id_column) or None.
    """
    if not db_path.exists() or not records:
        return None
    extracted_ids = {str(r.get("_id")) for r in records if r.get("_id") is not None}
    if not extracted_ids:
        return None
    # Determine how many non-_id fields we're bringing
    new_fields = set()
    for r in records:
        new_fields.update(k for k, v in r.items() if k != "_id" and v is not None)

    try:
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        candidates: list[tuple[str, str, float, int]] = []  # (table, id_col, overlap, n_cols)
        for (tname,) in tables:
            cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
            id_col = None
            for c in cols:
                if c[1].lower() in ("id", "_id"):
                    id_col = c[1]
                    break
            if not id_col:
                continue
            existing_ids = {
                str(r[0]) for r in
                conn.execute(f'SELECT DISTINCT "{id_col}" FROM "{tname}" WHERE "{id_col}" IS NOT NULL').fetchall()
            }
            if not existing_ids:
                continue
            overlap = len(extracted_ids & existing_ids) / len(extracted_ids)
            if overlap > 0.5:
                candidates.append((tname, id_col, overlap, len(cols)))
        conn.close()

        if not candidates:
            return None
        # Prefer tables with fewer columns (lookup tables), then highest overlap
        candidates.sort(key=lambda x: (x[3], -x[2]))
        return (candidates[0][0], candidates[0][1])
    except Exception:
        return None


def _merge_into_table(
    db_path: Path,
    table_name: str,
    id_col: str,
    records: list[dict[str, Any]],
    log_fn: Callable[[str, str], None] | None,
) -> int:
    """Merge extracted records into an existing table.

    - Adds new columns if needed
    - Inserts rows for IDs not yet in the table
    - Updates NULL cells for existing rows
    """
    conn = sqlite3.connect(str(db_path))
    try:
        existing_cols = {
            c[1].lower(): c[1]
            for c in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        }
        id_col_lower = id_col.lower()

        # Determine which columns from records to write (skip _id, use the table's id_col)
        col_counts: dict[str, int] = {}
        for r in records:
            for k, v in r.items():
                if k == "_id":
                    continue
                if v is not None:
                    col_counts[k] = col_counts.get(k, 0) + 1
        threshold = max(1, len(records) // 10)
        new_fields = [k for k, c in col_counts.items() if c >= threshold]

        # Add columns that don't exist yet
        for field in new_fields:
            if field.lower() not in existing_cols:
                conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{field}" TEXT')
                existing_cols[field.lower()] = field

        # Get existing IDs
        existing_ids = {
            str(r[0])
            for r in conn.execute(f'SELECT "{id_col}" FROM "{table_name}"').fetchall()
        }

        inserted = 0
        updated = 0
        for r in records:
            rid = str(r.get("_id", ""))
            if not rid:
                continue
            non_id_values = {k: v for k, v in r.items() if k != "_id" and v is not None and k in new_fields}
            if not non_id_values:
                continue

            if rid not in existing_ids:
                # Insert new row
                cols_to_insert = [id_col] + list(non_id_values.keys())
                vals = [rid] + list(non_id_values.values())
                placeholders = ", ".join("?" * len(cols_to_insert))
                quoted = ", ".join(f'"{c}"' for c in cols_to_insert)
                conn.execute(
                    f'INSERT INTO "{table_name}" ({quoted}) VALUES ({placeholders})', vals
                )
                existing_ids.add(rid)
                inserted += 1
            else:
                # Update NULL cells only
                updates = []
                vals = []
                for col, val in non_id_values.items():
                    actual_col = existing_cols.get(col.lower(), col)
                    updates.append(f'"{actual_col}" = CASE WHEN "{actual_col}" IS NULL THEN ? ELSE "{actual_col}" END')
                    vals.append(val)
                if updates:
                    vals.append(rid)
                    conn.execute(
                        f'UPDATE "{table_name}" SET {", ".join(updates)} WHERE "{id_col}" = ?',
                        vals,
                    )
                    updated += 1

        conn.commit()
        if log_fn:
            log_fn("regex_merged", f"Merged into '{table_name}': {inserted} inserted, {updated} updated, cols={new_fields}")
        return inserted + updated
    except Exception as e:
        if log_fn:
            log_fn("regex_error", f"Merge error: {e}")
        return 0
    finally:
        conn.close()


def _write_new_table(
    db_path: Path,
    table_name: str,
    records: list[dict[str, Any]],
    log_fn: Callable[[str, str], None] | None,
) -> int:
    """Create a new table and write records."""
    if not records:
        return 0

    col_counts: dict[str, int] = {}
    for r in records:
        for k, v in r.items():
            if v is not None:
                col_counts[k] = col_counts.get(k, 0) + 1

    threshold = max(1, len(records) // 10)
    columns = sorted(
        k for k, c in col_counts.items()
        if c >= threshold or (c >= 1 and k.startswith("link_to_"))
    )
    if not columns:
        return 0

    col_types: dict[str, str] = {}
    for col in columns:
        for r in records:
            v = r.get(col)
            if v is not None:
                col_types[col] = "REAL" if isinstance(v, (int, float)) else "TEXT"
                break
        if col not in col_types:
            col_types[col] = "TEXT"

    conn = sqlite3.connect(str(db_path))
    try:
        col_defs = ", ".join(f'"{c}" {col_types[c]}' for c in columns)
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

        placeholders = ", ".join("?" * len(columns))
        quoted = ", ".join(f'"{c}"' for c in columns)
        insert_sql = f'INSERT INTO "{table_name}" ({quoted}) VALUES ({placeholders})'

        written = 0
        for r in records:
            values = [r.get(c) for c in columns]
            non_id = [v for k, v in zip(columns, values) if k != "_id" and v is not None]
            if not non_id:
                continue
            try:
                conn.execute(insert_sql, values)
                written += 1
            except Exception:
                pass

        conn.commit()
        if log_fn:
            log_fn("regex_sqlite", f"Wrote {written}/{len(records)} records to '{table_name}', cols: {columns}")
        return written
    except Exception as e:
        if log_fn:
            log_fn("regex_error", f"SQLite error: {e}")
        return 0
    finally:
        conn.close()


def _merge_or_create(
    db_path: Path,
    doc_stem: str,
    records: list[dict[str, Any]],
    protected_tables: set[str] | None,
    log_fn: Callable[[str, str], None] | None,
) -> int:
    """Merge into an existing table if IDs overlap, otherwise create a new one."""
    if not records:
        return 0

    target = _find_merge_target(db_path, records, protected_tables)
    if target:
        table_name, id_col = target
        return _merge_into_table(db_path, table_name, id_col, records, log_fn)

    # No merge target — create new table
    table_name = re.sub(r"[^a-zA-Z0-9_]", "_", doc_stem).lower()
    if table_name and table_name[0].isdigit():
        table_name = "t_" + table_name
    if protected_tables and table_name in protected_tables:
        table_name = f"{table_name}_doc"
    return _write_new_table(db_path, table_name, records, log_fn)


# ---------------------------------------------------------------------------
# Multi-doc orchestrator (parallel to compiled_extract_docs)
# ---------------------------------------------------------------------------

def regex_extract_docs(
    doc_paths: list[Path],
    db_path: Path,
    model: ModelAdapter,
    question: str,
    knowledge_text: str = "",
    log_fn: Callable[[str, str], None] | None = None,
    structured_tables: set[str] | None = None,
) -> int:
    """Extract all docs using regex-based extraction.

    Drop-in replacement for compiled_extract_docs with same interface.
    """
    protected = {t.lower() for t in (structured_tables or set())}
    total = 0
    for doc_path in doc_paths:
        n = regex_extract(
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
            log_fn("regex_doc_done", f"{doc_path.name}: {n} records")
    return total
