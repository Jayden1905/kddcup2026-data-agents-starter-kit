"""Deterministic rule-based extraction engine.

Extracts structured data values from medical/business prose using regex patterns.
No LLM calls -- processes hundreds of paragraphs in <1 second.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from data_agent_baseline.pipeline.field_discoverer import DocumentSchema, DiscoveredField


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExtractedValue:
    field_name: str
    value: Any  # float, str, None
    confidence: Confidence
    is_correction: bool  # True if came from correction pattern
    alternatives: list[Any] = field(default_factory=list)  # other candidate values


@dataclass
class ExtractedRecord:
    record_id: str
    entity_type: str
    fields: dict[str, ExtractedValue]
    ambiguous_fields: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_ORDINAL_MAP: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "twenty-first": 21, "twenty-second": 22,
    "twenty-third": 23, "twenty-fourth": 24, "twenty-fifth": 25,
    "twenty-sixth": 26, "twenty-seventh": 27, "twenty-eighth": 28,
    "twenty-ninth": 29, "thirtieth": 30, "thirty-first": 31,
}

# Numeric ordinal suffixes: 1st, 2nd, 3rd, 4th...31st
_ORDINAL_SUFFIX_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)")

# NaN / missing indicators
_NAN_PHRASES = [
    "not available (nan)",
    "unavailable (nan)",
    "not recorded (nan)",
    "marked as not available",
    "not available",
    "unavailable",
    "not recorded",
    "data gap",
    "(nan)",
    "nan",
]

# Status indicators
_ABNORMAL_PHRASES = [
    "significant reduction",
    "impaired",
    "abnormal",
    "elevated",
    "decreased",
    "reduced clearance",
    "outside normal",
    "above normal",
    "below normal",
    "compromised",
    "deteriorat",
    "dysfunction",
    "failure",
    "impaired renal filtration",
    "reduction in renal clearance",
]

_NORMAL_PHRASES = [
    "healthy kidney function",
    "healthy liver function",
    "healthy",
    "normal",
    "within expected",
    "within physiological",
    "unremarkable",
    "clinically unremarkable",
    "stable",
    "well-maintained",
    "comfortably within",
    "adequate",
]

# Date patterns
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

_MONTH_NAMES_PAT = "|".join(_MONTH_MAP.keys())

# "February 10th, 1986" or "February 10, 1986"
_DATE_MDY_RE = re.compile(
    rf"\b({_MONTH_NAMES_PAT})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)

# "the tenth of February, 1986" or "tenth of February, 1986"
_DATE_ORDINAL_OF_RE = re.compile(
    rf"\b(?:the\s+)?(\w+(?:-\w+)?)\s+of\s+({_MONTH_NAMES_PAT}),?\s+(\d{{4}})\b",
    re.IGNORECASE,
)

# "10 February 1986"
_DATE_DMY_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({_MONTH_NAMES_PAT}),?\s+(\d{{4}})\b",
    re.IGNORECASE,
)

# "December 14th, 1993" -- also matches "On June 14th, 1993"
_DATE_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b({_MONTH_NAMES_PAT})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)

# Numeric value pattern
_NUMERIC_VAL_RE = re.compile(r"[-+]?\d+\.?\d*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_day(text: str) -> int | None:
    """Parse a day from ordinal word or numeric+suffix."""
    low = text.lower().strip()
    if low in _ORDINAL_MAP:
        return _ORDINAL_MAP[low]
    m = _ORDINAL_SUFFIX_RE.fullmatch(text.strip())
    if m:
        return int(m.group(1))
    if text.strip().isdigit():
        return int(text.strip())
    return None


def _normalize_date(month_str: str, day: int, year: int) -> str:
    month = _MONTH_MAP.get(month_str.lower())
    if month is None:
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def _extract_dates(text: str) -> list[str]:
    """Extract all dates from text, normalized to YYYY-MM-DD."""
    dates: list[str] = []

    # ISO dates
    for m in _ISO_DATE_RE.finditer(text):
        dates.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    # "February 10th, 1986" / "December 14th, 1993"
    for m in _DATE_MDY_RE.finditer(text):
        month_str = m.group(1)
        day = int(m.group(2))
        year = int(m.group(3))
        d = _normalize_date(month_str, day, year)
        if d:
            dates.append(d)

    # "the tenth of February, 1986"
    for m in _DATE_ORDINAL_OF_RE.finditer(text):
        ordinal_word = m.group(1)
        month_str = m.group(2)
        year = int(m.group(3))
        day = _parse_day(ordinal_word)
        if day:
            d = _normalize_date(month_str, day, year)
            if d:
                dates.append(d)

    # "10 February 1986"
    for m in _DATE_DMY_RE.finditer(text):
        day = int(m.group(1))
        month_str = m.group(2)
        year = int(m.group(3))
        d = _normalize_date(month_str, day, year)
        if d:
            dates.append(d)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _detect_status(text: str) -> str | None:
    """Detect normal/abnormal status from qualitative phrases."""
    lower = text.lower()
    for phrase in _ABNORMAL_PHRASES:
        if phrase in lower:
            return "abnormal"
    for phrase in _NORMAL_PHRASES:
        if phrase in lower:
            return "normal"
    return None


def _detect_field_status(text: str, label_pat: re.Pattern[str]) -> str | None:
    """Detect status specifically associated with a field label.

    Only returns abnormal/normal if the status phrase is within ±200 chars of the
    field's label, meaning it's specifically about that field.
    """
    for label_match in label_pat.finditer(text):
        start = max(0, label_match.start() - 200)
        end = min(len(text), label_match.end() + 200)
        window = text[start:end].lower()
        for phrase in _ABNORMAL_PHRASES:
            if phrase in window:
                return "abnormal"
        for phrase in _NORMAL_PHRASES:
            if phrase in window:
                return "normal"
    return None


def _build_field_label_pattern(field_name: str, aliases: list[str]) -> re.Pattern[str]:
    """Build a regex that matches any of the field's labels (name + aliases)."""
    labels = [re.escape(field_name)]
    for alias in aliases:
        escaped = re.escape(alias)
        if escaped not in labels:
            labels.append(escaped)
    # Sort by length descending so longer aliases match first
    labels.sort(key=len, reverse=True)
    pat = "|".join(labels)
    return re.compile(rf"\b({pat})\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# RuleExtractor
# ---------------------------------------------------------------------------


class RuleExtractor:
    """Deterministic rule-based extractor for structured data from prose.

    Handles any domain (lab values, budget items, superhero attributes) by using
    the schema's field list dynamically. All regex patterns are pre-compiled in
    __init__ for speed.
    """

    def __init__(self, schema: DocumentSchema, knowledge_text: str = ""):
        self._schema = schema
        self._knowledge_text = knowledge_text
        self._entity_type = schema.entity_name
        self._id_re = schema.id_pattern

        # Build per-field label patterns (pre-compiled for speed)
        self._field_patterns: list[tuple[DiscoveredField, re.Pattern[str]]] = []
        for f in schema.fields:
            aliases = f.aliases if f.aliases else []
            pat = _build_field_label_pattern(f.name, aliases)
            self._field_patterns.append((f, pat))

        # --- Pre-compiled correction patterns ---

        # Pattern A: FIELD ... initially/first/originally NUM ... corrected/confirmed NUM
        self._correction_a_re = re.compile(
            r"(?:initially|first|originally)\s+"
            r"(?:logged\s+as|thought\s+to\s+be|recorded\s+as|estimated\s+at|"
            r"suggested|transcribed\s+as|reported\s+as|read\s+as|measured\s+at|"
            r"flagged\s+at|entered\s+as|noted\s+as|listed\s+as|underestimated\s+at)\s+"
            r"([-+]?\d+\.?\d*)"
            r".*?"
            r"(?:correct(?:ed|ion)|confirm(?:ed)?|verif(?:ied|y)|amend(?:ed)?|"
            r"adjust(?:ed)?|revis(?:ed|ion)|finalized?"
            r"(?:\s+and\s+corrected)?|precisely\s+measured"
            r"(?:\s+and\s+corrected)?|definitive\s+analysis\s+confirmed)\s+"
            r"(?:to\s+(?:be\s+)?|at\s+|value\s+of\s+)?"
            r"([-+]?\d+\.?\d*)",
            re.IGNORECASE | re.DOTALL,
        )

        # Pattern B: NUM was later amended/corrected/adjusted to NUM
        self._correction_b_re = re.compile(
            r"([-+]?\d+\.?\d*)\s*[A-Za-z/]*\s*"
            r"(?:,?\s*(?:but|however|though))?\s*"
            r"(?:was\s+later|was\s+subsequently|was)?\s*"
            r"(?:amend(?:ed)?|correct(?:ed)?|adjust(?:ed)?|revis(?:ed)?|"
            r"verified|confirmed|finalized)\s+"
            r"(?:to\s+(?:be\s+)?|at\s+)"
            r"([-+]?\d+\.?\d*)",
            re.IGNORECASE | re.DOTALL,
        )

        # Pattern C: adjusted from NUM to NUM
        self._correction_c_re = re.compile(
            r"(?:adjust(?:ed)?|correct(?:ed)?|revis(?:ed)?|amend(?:ed)?|"
            r"chang(?:ed)?|updat(?:ed)?)\s+"
            r"from\s+([-+]?\d+\.?\d*)\s*[A-Za-z/]*\s+"
            r"to\s+([-+]?\d+\.?\d*)",
            re.IGNORECASE,
        )

        # Pattern D: initially suggested/thought NUM, but manual verification confirmed NUM
        self._correction_d_re = re.compile(
            r"(?:initially|first|originally)\s+(?:suggested|thought|indicated)\s+"
            r"([-+]?\d+\.?\d*)\s*[A-Za-z/]*"
            r".*?"
            r"(?:manual\s+verification|re-?check|review|reanalysis|definitive\s+analysis)"
            r"\s+(?:confirmed|showed|revealed|yielded)\s+"
            r"([-+]?\d+\.?\d*)",
            re.IGNORECASE | re.DOTALL,
        )

        # Pattern E: "originally transcribed as NUM ... definitive analysis confirmed NUM"
        self._correction_e_re = re.compile(
            r"(?:originally|initially)\s+(?:transcribed|recorded|logged|noted)\s+"
            r"(?:as\s+)?([-+]?\d+\.?\d*)\s*[A-Za-z/]*"
            r".*?"
            r"(?:definitive|final|conclusive|authoritative)\s+"
            r"(?:analysis|reading|measurement|assessment)\s+"
            r"(?:confirmed|showed|revealed|yielded|established)\s+"
            r"([-+]?\d+\.?\d*)",
            re.IGNORECASE | re.DOTALL,
        )

        # Batch NaN: "full panel (GOT, GPT, LDH, ALP, T-BIL) was unavailable (NaN)"
        # Also: "entire liver function panel... was marked as not available (NaN)"
        # Also: "Complete data voids for the renal panel (UA, UN, CRE)"
        self._batch_nan_re = re.compile(
            r"(?:"
            r"(?:full\s+panel|entire\s+[\w\s]+panel|"
            r"all\s+(?:tests|values|measurements|parameters|markers))\s*"
            r"(?:\([^)]+\))?\s*"
            r"(?:was|were)\s+"
            r"(?:unavailable|not\s+available|not\s+recorded|marked\s+as\s+not\s+available)"
            r"(?:\s*\(NaN\))?"
            r"|"
            r"(?:complete\s+)?data\s+(?:void|gap)s?\s+(?:for|in)\s+(?:the\s+)?"
            r"[\w\s]+(?:panel|markers?)\s*\([^)]+\)"
            r"|"
            r"(?:UA|UN|CRE|GOT|GPT|LDH|ALP|T[_-]BIL|TP|ALB|T[_-]CHO|TG|CPK|GLU"
            r"|WBC|RBC|HGB|HCT|PLT)\s*(?:,\s*(?:and\s+)?(?:UA|UN|CRE|GOT|GPT|LDH"
            r"|ALP|T[_-]BIL|TP|ALB|T[_-]CHO|TG|CPK|GLU|WBC|RBC|HGB|HCT|PLT))+\s*"
            r"(?:was|were)\s+(?:all\s+)?(?:NaN|unavailable|not\s+available"
            r"|marked\s+as\s+not\s+available)"
            r")",
            re.IGNORECASE,
        )

        # Batch NaN field list extraction: captures (F1, F2, F3) in parens
        self._batch_nan_fields_re = re.compile(
            r"(?:full\s+panel|entire\s+[\w\s]+panel|"
            r"all\s+(?:tests|values|measurements|parameters|markers)|"
            r"(?:complete\s+)?data\s+(?:void|gap)s?\s+(?:for|in)\s+(?:the\s+)?[\w\s]+?"
            r"(?:panel|markers?))\s*"
            r"\(([^)]+)\)",
            re.IGNORECASE,
        )

        # Patient ID patterns
        self._patient_id_re = re.compile(
            r"(?:patient|subject|ID)\s*[:#]?\s*(\d{5,})", re.IGNORECASE
        )

        # Sex pattern
        self._sex_re = re.compile(
            r"(?:Sex\s*:\s*['\"]?(M|F)['\"]?"
            r"|(?:identified|denoted|classified)\s+as\s+(?:['\"]?)?(male|female|M|F)(?:['\"]?)"
            r"|denoted\s+as\s+['\"]?(M|F)['\"]?\s+for\s+(?:male|female))",
            re.IGNORECASE,
        )

        # Diagnosis pattern
        self._diagnosis_re = re.compile(
            r"Diagnosis\s*:\s*(\S+)", re.IGNORECASE
        )

    def extract_paragraph(self, paragraph: str) -> ExtractedRecord | None:
        """Extract structured data from a single paragraph of prose.

        Returns None if no meaningful data can be extracted.
        """
        if not paragraph or len(paragraph.strip()) < 10:
            return None

        text = paragraph.strip()
        fields: dict[str, ExtractedValue] = {}
        ambiguous: list[str] = []

        # --- 1. Find record ID ---
        record_id = self._extract_record_id(text)

        # --- 2. Extract corrections first (highest priority) ---
        correction_fields = self._extract_corrections(text)
        fields.update(correction_fields)

        # --- 3. Detect batch NaN ---
        batch_nan_fields = self._extract_batch_nan(text)
        for fname, ev in batch_nan_fields.items():
            if fname not in fields:  # corrections win
                fields[fname] = ev

        # --- 4. Extract per-field NaN ---
        for discovered_field, label_pat in self._field_patterns:
            fname = discovered_field.name
            if fname in fields:
                continue
            nan_val = self._extract_field_nan(text, label_pat, fname)
            if nan_val is not None:
                fields[fname] = nan_val

        # --- 5. Extract simple numeric values ---
        for discovered_field, label_pat in self._field_patterns:
            fname = discovered_field.name
            if fname in fields:
                continue
            if discovered_field.field_type == "numeric":
                val = self._extract_numeric_value(text, label_pat, fname)
                if val is not None:
                    fields[fname] = val

        # --- 6. Extract dates ---
        dates = _extract_dates(text)
        if dates:
            # Assign to date-typed fields that aren't yet populated
            date_fields = [
                f for f, _ in self._field_patterns
                if f.field_type == "date" and f.name not in fields
            ]
            if date_fields:
                for i, df in enumerate(date_fields):
                    if i < len(dates):
                        fields[df.name] = ExtractedValue(
                            field_name=df.name,
                            value=dates[i],
                            confidence=Confidence.HIGH,
                            is_correction=False,
                        )
            elif "date" not in fields:
                # Default: store first date under "date" key
                fields["date"] = ExtractedValue(
                    field_name="date",
                    value=dates[0],
                    confidence=Confidence.HIGH,
                    is_correction=False,
                    alternatives=dates[1:] if len(dates) > 1 else [],
                )

        # --- 7. Extract patient-specific info ---
        self._extract_patient_info(text, fields)

        # --- 8. Status detection deferred to LLM (paragraph-level regex is too noisy) ---

        # --- 9. Score confidence and flag ambiguous ---
        for fname, ev in fields.items():
            if ev.alternatives:
                ambiguous.append(fname)

        if not fields and not record_id:
            return None

        return ExtractedRecord(
            record_id=record_id or "",
            entity_type=self._entity_type,
            fields=fields,
            ambiguous_fields=ambiguous,
        )

    def extract_all(self, paragraphs: list[str]) -> list[ExtractedRecord]:
        """Extract records from multiple paragraphs."""
        results: list[ExtractedRecord] = []
        for para in paragraphs:
            record = self.extract_paragraph(para)
            if record is not None:
                results.append(record)
        return results

    # ------------------------------------------------------------------
    # Internal extraction methods
    # ------------------------------------------------------------------

    def _extract_record_id(self, text: str) -> str:
        """Find the primary record/entity ID in text."""
        # Try patient-specific pattern first
        m = self._patient_id_re.search(text)
        if m:
            return m.group(1)
        # Fall back to schema's id_pattern
        m = self._id_re.search(text)
        if m:
            return m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
        return ""

    def _extract_corrections(self, text: str) -> dict[str, ExtractedValue]:
        """Extract corrected values -- these always take priority.

        Tries multiple correction patterns in order and uses the CORRECTED
        (final) value. The initial/wrong value is stored in alternatives.
        Only matches corrections that appear AFTER the field label (not before),
        and stops at the next field label to avoid cross-contamination.
        """
        results: dict[str, ExtractedValue] = {}

        # Build a set of all label positions for boundary detection
        all_label_positions: list[tuple[int, int]] = []
        for _, lp in self._field_patterns:
            for m in lp.finditer(text):
                all_label_positions.append((m.start(), m.end()))
        all_label_positions.sort()

        for discovered_field, label_pat in self._field_patterns:
            fname = discovered_field.name
            if fname in results:
                continue

            for label_match in label_pat.finditer(text):
                # Only look FORWARD from the label (not backward)
                context_start = label_match.end()
                # Find the next field label after this one to bound our search
                context_end = min(len(text), label_match.end() + 500)
                for pos_start, _ in all_label_positions:
                    if pos_start > label_match.end() + 5:
                        context_end = min(context_end, pos_start)
                        break

                context = text[context_start:context_end]

                initial_val: float | None = None
                corrected_val: float | None = None

                for pattern in (
                    self._correction_a_re,
                    self._correction_b_re,
                    self._correction_c_re,
                    self._correction_d_re,
                    self._correction_e_re,
                ):
                    m = pattern.search(context)
                    if m:
                        initial_val = float(m.group(1))
                        corrected_val = float(m.group(2))
                        break

                if corrected_val is not None:
                    results[fname] = ExtractedValue(
                        field_name=fname,
                        value=corrected_val,
                        confidence=Confidence.HIGH,
                        is_correction=True,
                        alternatives=[initial_val] if initial_val is not None else [],
                    )
                    break  # One correction per field

        return results

    def _extract_batch_nan(self, text: str) -> dict[str, ExtractedValue]:
        """Detect batch NaN patterns like 'full panel (GOT, GPT, ...) was unavailable'.

        Also handles: 'The entire liver function panel... was marked as not available (NaN)'
        """
        results: dict[str, ExtractedValue] = {}

        # Check if the text has a batch NaN indicator
        if not self._batch_nan_re.search(text):
            return results

        # Try to extract the explicit field list from parens
        field_list_match = self._batch_nan_fields_re.search(text)
        if field_list_match:
            field_list_str = field_list_match.group(1)
            # Split on commas and optional spaces
            mentioned = [s.strip() for s in field_list_str.split(",")]
            for mentioned_name in mentioned:
                matched_field = self._match_field_name(mentioned_name)
                if matched_field:
                    results[matched_field] = ExtractedValue(
                        field_name=matched_field,
                        value=None,
                        confidence=Confidence.HIGH,
                        is_correction=False,
                    )
        else:
            # No explicit list -- set all numeric fields to NaN
            for discovered_field, _ in self._field_patterns:
                if discovered_field.field_type == "numeric":
                    results[discovered_field.name] = ExtractedValue(
                        field_name=discovered_field.name,
                        value=None,
                        confidence=Confidence.MEDIUM,
                        is_correction=False,
                    )

        return results

    def _extract_field_nan(
        self, text: str, label_pat: re.Pattern[str], field_name: str
    ) -> ExtractedValue | None:
        """Check if a field is marked as NaN/missing."""
        for label_match in label_pat.finditer(text):
            # Get context after the label
            after = text[label_match.end(): label_match.end() + 150].lower()
            for phrase in _NAN_PHRASES:
                if phrase in after:
                    return ExtractedValue(
                        field_name=field_name,
                        value=None,
                        confidence=Confidence.HIGH,
                        is_correction=False,
                    )
        return None

    def _extract_numeric_value(
        self, text: str, label_pat: re.Pattern[str], field_name: str
    ) -> ExtractedValue | None:
        """Extract the nearest numeric value to a field label.

        Looks for the first number appearing after any mention of the field label,
        preferring closest proximity. Skips values that look like patient IDs (5+ digits
        with no decimal), years (4-digit numbers in date contexts), or values clearly
        belonging to another field.
        """
        best_value: float | None = None
        best_distance = float("inf")
        alternatives: list[float] = []

        for label_match in label_pat.finditer(text):
            label_end = label_match.end()
            search_region = text[label_end: label_end + 120]

            # Check if we're inside a batch NaN region (data void paragraph)
            pre_context = text[max(0, label_match.start() - 100):label_match.start()].lower()
            if "data void" in pre_context or "data gap" in pre_context:
                continue

            nums = list(_NUMERIC_VAL_RE.finditer(search_region))
            for num_match in nums:
                val_str = num_match.group(0)
                val = float(val_str)
                distance = num_match.start()

                # Skip if the number looks like a patient ID (5+ digits, no decimal)
                if "." not in val_str and len(val_str.lstrip("-+")) >= 5:
                    continue

                # Skip if it looks like a year (4-digit number near "patient" or date words)
                if len(val_str) == 4 and 1900 <= val <= 2100:
                    context_before = search_region[:num_match.start()].lower()
                    if any(w in context_before for w in ["patient", "on", "dated", ","]):
                        continue

                if distance < best_distance:
                    if best_value is not None:
                        alternatives.append(best_value)
                    best_value = val
                    best_distance = distance
                else:
                    alternatives.append(val)
                break  # Take first valid number after label

        if best_value is not None:
            confidence = Confidence.HIGH if best_distance < 40 else Confidence.MEDIUM
            return ExtractedValue(
                field_name=field_name,
                value=best_value,
                confidence=confidence,
                is_correction=False,
                alternatives=alternatives if alternatives else [],
            )
        return None

    def _extract_patient_info(self, text: str, fields: dict[str, ExtractedValue]) -> None:
        """Extract patient-specific metadata (sex, diagnosis, birthday)."""
        # Sex
        if "SEX" not in fields:
            m = self._sex_re.search(text)
            if m:
                raw = m.group(1) or m.group(2) or m.group(3)
                if raw:
                    sex = "M" if raw.lower() in ("m", "male") else "F"
                    fields["SEX"] = ExtractedValue(
                        field_name="SEX",
                        value=sex,
                        confidence=Confidence.HIGH,
                        is_correction=False,
                    )

        # Diagnosis
        if "Diagnosis" not in fields:
            m = self._diagnosis_re.search(text)
            if m:
                fields["Diagnosis"] = ExtractedValue(
                    field_name="Diagnosis",
                    value=m.group(1).strip(),
                    confidence=Confidence.HIGH,
                    is_correction=False,
                )

        # Birthday (Date of birth)
        if "Birthday" not in fields:
            dob_match = re.search(
                r"(?:Date\s+of\s+birth|DOB|born\s+on|birthday)\s+(?:is\s+)?",
                text,
                re.IGNORECASE,
            )
            if dob_match:
                after = text[dob_match.end():]
                dates = _extract_dates(after)
                if dates:
                    fields["Birthday"] = ExtractedValue(
                        field_name="Birthday",
                        value=dates[0],
                        confidence=Confidence.HIGH,
                        is_correction=False,
                    )

    def _match_field_name(self, name: str) -> str:
        """Match a mentioned field name to a known schema field.

        Performs exact match, then alias match, then partial/fuzzy match.
        Returns the canonical field name or the input as-is if no match.
        """
        name_normalized = name.lower().strip().replace("-", "_")

        # Exact match
        for discovered_field, _ in self._field_patterns:
            if discovered_field.name.lower().replace("-", "_") == name_normalized:
                return discovered_field.name

        # Alias match
        for discovered_field, _ in self._field_patterns:
            for alias in discovered_field.aliases:
                if alias.lower().replace("-", "_") == name_normalized:
                    return discovered_field.name

        # Partial match: if the mentioned name is contained in or contains a field name
        for discovered_field, _ in self._field_patterns:
            field_lower = discovered_field.name.lower().replace("-", "_")
            if name_normalized in field_lower or field_lower in name_normalized:
                return discovered_field.name
            for alias in discovered_field.aliases:
                alias_lower = alias.lower().replace("-", "_")
                if name_normalized in alias_lower or alias_lower in name_normalized:
                    return discovered_field.name

        return name  # Return as-is if no match found
