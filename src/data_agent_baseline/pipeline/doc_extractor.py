"""Deterministic document extractor — fully domain-agnostic.

Extracts structured records from verbose prose documents WITHOUT LLM calls.
Strategy:
  1. Auto-detect ID pattern statistically (most frequent repeating token class)
  2. For each paragraph with an ID, extract ALL attributes blindly (numbers, categories, dates, links)
  3. Merge multi-section data by record ID
  4. Cross-reference with DB (if exists) or across docs to find FK links
  5. LLM only writes SQL against the resulting table
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractedTable:
    name: str
    id_field: str
    records: list[dict[str, Any]]
    fk_links: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ID pattern candidates — ordered by specificity
# ---------------------------------------------------------------------------

ID_CANDIDATES = [
    ("rec_id", re.compile(r"\b(rec[A-Za-z0-9]{8,})\b")),
    ("mol_id", re.compile(r"\b(TR\d{2,})\b")),
    ("numeric_id", re.compile(r"(?:patient|Patient|Case ID|Subject|file number|Medical Record Number|Record Number)\s+(\d{4,})")),
    ("keyword_numeric", re.compile(
        r"(?:registration\s+(?:number|is)|identifier|reference\s+(?:code|ID|number)|"
        r"registry\s+(?:number|Ref)|(?:Competition|Registry|Tracking)\s+(?:ID|Ref|Code))"
        r"\s*[:\s]?\s*(\d+)"
    )),
    ("generic_numeric", re.compile(r"(?<!\d)(\d{5,})(?!\d)")),
]

# Words starting with "rec" that are NOT record IDs
REC_FALSE_POSITIVES = {
    "reconciliation", "reconstruction", "reconstructive", "recalibration",
    "recalculation", "recognition", "recommendation", "recommended",
    "recrystallization", "rectification", "reconnaissance", "record",
    "records", "recorded", "recording", "recovery", "recovered",
    "recreation", "recreational", "recruitment", "recurrence", "recurring",
    "recently", "reception", "received", "receiving", "receptor",
    "recognizable", "recalibrated", "reclassification", "reclassifying",
    "reclassified", "reconciled", "reconfigured", "reconfiguration",
    "reconsidered", "reconstituted", "reconvened", "recounted",
    "rectangular", "rectified", "recuperated", "recuperating",
}


def _is_rec_false_positive(token: str) -> bool:
    """Check if a rec-prefixed token is an English word, not a record ID."""
    if token.lower() in REC_FALSE_POSITIVES:
        return True
    # Real rec IDs have mixed case alphanumeric like recXZUYlYNiRmeoxX
    # English words are all lowercase
    if token[3:].isalpha() and token[3:].islower():
        return True
    return False

# ---------------------------------------------------------------------------
# Generic attribute extractors
# ---------------------------------------------------------------------------

# Numbers with context — we extract ALL numbers and label by position
AMOUNT_RE = re.compile(
    r"(?:amount|funded|allocated|budget|budgeted|allocation)\s+(?:of|is|was|at|with)?\s*"
    r"(\d[\d,]*\.?\d*)",
    re.IGNORECASE,
)
REVISED_AMOUNT_RE = re.compile(
    r"(?:revised|corrected|adjusted|reconciled|amended|confirmed)\s+"
    r"(?:upward\s+)?(?:[\w\s]{0,30}?)(?:to\s+)?(?:an?\s+)?(?:amount\s+of\s+)?"
    r"(\d[\d,]+\.?\d*)(?:\.|,|\s)",
    re.IGNORECASE,
)
SPENT_RE = re.compile(
    r"(?:(\d[\d,]*\.?\d*)\s+(?:has|have)\s+been\s+(?:spent|processed|invoiced)|"
    r"expenditure[s]?\s+(?:of|totaling|total(?:ed|ing)?)\s+(\d[\d,]*\.?\d*)|"
    r"(?:spent|expended)\s*[,:]?\s*(\d[\d,]*\.?\d*))",
    re.IGNORECASE,
)
REMAINING_RE = re.compile(
    r"remaining\s+(?:balance|sum|contingency)?\s*(?:of|is|at|showing)?\s*(?:as\s+)?"
    r"(-?\d[\d,]*\.?\d*)",
    re.IGNORECASE,
)

# Category: capitalized phrase after classification verbs
CATEGORY_VERBS = re.compile(
    r"(?:classified|categorized|designated|reclassified|listed|corrected|amended|"
    r"re-categorized|updated|revised|assigned|allocated|earmarked)\s+"
    r"(?:as|for|under|to|within\s+\w+\s+\w+\s+as)\s+"
    r"(?:the\s+)?(?:broader\s+|more\s+\w+\s+)?(?:category\s+of\s+)?(?:the\s+)?"
    r"([A-Z][A-Za-z\s\-']+?)(?:\.|,|\s+for\s|\s+based|\s+upon|\s+after|\s+due|"
    r"\s+This|\s+The|\s+It|\s+A |\s+His|\s+Her|\s+category)",
    re.MULTILINE,
)
CATEGORY_OF_RE = re.compile(
    r"(?:classification|categorization|designation|category)\s+"
    r"(?:of|as|is)\s+([A-Z][A-Za-z\s\-']+?)(?:\.|,|\s+This|\s+The|\s+It)",
)
CATEGORY_PROVIDING_RE = re.compile(
    r"(?:providing|provision of|cover)\s+([A-Z][A-Za-z\s\-]+?)"
    r"(?:\.|,|\s+expenditure|\s+This|\s+The|\s+for\s)",
)
CATEGORY_FINAL_RE = re.compile(
    r"(?:final|proper|correct|official)\s+(?:categorization|classification|"
    r"designation|category)\s+(?:as|of|is|was)\s+"
    r"([A-Z][A-Za-z\s\-']+?)(?:\.|,|\s+This|\s+The)",
)

# Status
STATUS_RE = re.compile(
    r"(?:event_)?status\s+(?:is|was|of|now|has been)\s*"
    r"(?:formally\s+)?(?:logged|updated|recorded|listed|documented|confirmed|designated|advanced)?\s*"
    r"(?:as\s+|to\s+(?:the\s+)?|to\s+its\s+current\s+state\s+of\s+)?"
    r"([A-Z][a-z]+)",
    re.IGNORECASE,
)

# Event links (secondary rec IDs in the same paragraph)
EVENT_LINK_RE = re.compile(
    r"(?:event\s+(?:record|identifier|link|tracking|reference|file|dossier|portfolio|"
    r"marker|ID)|archived|linked|filed|indexed|consolidated|documented|tracked|"
    r"referenced|cross-referenced|accessible|stored)\s+"
    r"(?:under|via|through|within|in)?\s*(?:the\s+)?(?:event\s+)?"
    r"(?:record|identifier|link|tracking|reference|file|marker|ID|code)?\s*"
    r"(rec[A-Za-z0-9]{8,}|\d{5,})",
    re.IGNORECASE,
)

# Names (person names — various patterns)
NAME_RE = re.compile(
    r"(?:full\s+name|civilian\s+identity|legal\s+name|"
    r"verified\s+identity|full\s+identity|correct\s+designation)\s+"
    r"(?:is|of|for|associated|recorded|confirmed|documented)\s*"
    r"(?:this\s+\w+\s+\w+\s+is\s+)?(?:as\s+|to\s+be\s+)?(?:is\s+)?"
    r"([A-Z][A-Za-z\s\-']+?)(?:\.|,|\s+This|\s+The|\s+His|\s+Her|\s+Mr|\s+Ms)",
)
# "corresponds to / identified as / reveals ... as / confirmed as NAME NAME"
PERSON_NAME_RE = re.compile(
    r"(?:corresponds\s+to|identified\s+as|confirmed\s+as|verified\s+as|"
    r"reveals\s+\w+\s+(?:as|to\s+be)|"
    r"(?:stakeholder|individual|asset)\s+is|"
    r"rec[A-Za-z0-9]{8,}\s+is|"
    r"establishing\s+.*?identity\s+as)"
    r"\s+([A-Z][a-z]+\s+[A-Z][A-Za-z'\-]+)",
)

# Bold text (event names in markdown)
BOLD_RE = re.compile(r"\*\*([^*]{2,60})\*\*")

# Entity/program name: capitalized phrase immediately before a parenthetical ID
# e.g. "Finance (Registry ID: recXXX)" or "Public Health (recXXX)"
ENTITY_NAME_BEFORE_ID_RE = re.compile(
    r"(?:for|of|as|is|to|titled?|now|program|unit|discipline|track)\s+(?:now\s+)?"
    r"([A-Z][A-Za-z][A-Za-z\s,\-&']*?[A-Za-z])"
    r"\s*\((?:Registry\s+ID|identifier|tracked\s+under\s+identifier|ID)[:\s]*"
    r"rec[A-Za-z0-9]{8,}",
)
# "TITLE (recXXX)" — direct parenthetical without keyword prefix
ENTITY_NAME_DIRECT_PAREN_RE = re.compile(
    r"(?:as|now|is|of|for|to|title[,:])\s+(?:now\s+)?"
    r"([A-Z][A-Za-z][A-Za-z\s,\-&']*?[A-Za-z])"
    r"\s*\(rec[A-Za-z0-9]{8,}\)",
)
# "TITLE, registered under identifier recXXX"
ENTITY_NAME_REGISTERED_RE = re.compile(
    r"(?:for|of|as|is|titled?|track)\s+"
    r"([A-Z][A-Za-z][A-Za-z\s,\-&']*?[A-Za-z])"
    r"[,;]\s*(?:registered|tracked|cataloged|logged|filed|listed)\s+"
    r"(?:under|with)\s+(?:the\s+)?(?:record\s+)?identifier\s+"
    r"rec[A-Za-z0-9]{8,}",
)
# "corrected/updated/amended to TITLE (recXXX)"
ENTITY_NAME_CORRECTED_RE = re.compile(
    r"(?:corrected|updated|amended|finalized|rectified|designated)\s+"
    r"(?:to|as)\s+(?:the\s+)?(?:more\s+\w+[,]?\s+)?(?:\w+\s+title[,:]?\s+)?"
    r"([A-Z][A-Za-z][A-Za-z\s,\-&']*?[A-Za-z])"
    r"\s*\(rec[A-Za-z0-9]{8,}",
)

# Dates
DATE_WRITTEN_RE = re.compile(
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
)
DATE_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Carcinogenic/label classification
POSITIVE_LABEL_RE = re.compile(
    r"(?:positive\s+carcinogenic|carcinogenic\s+(?:profile|classification|potential|"
    r"characteristics|nature|properties|finding|status)|"
    r"confirmed\s+(?:to be\s+)?carcinogenic|"
    r"classification\s+(?:was|of)\s+(?:revised|corrected|updated)\s+to\s+(?:a\s+)?(?:definitive\s+)?positive|"
    r"exhibits?\s+positive\s+carcinogenic|"
    r"possess(?:ing|es)?\s+(?:a\s+)?positive\s+carcinogenic|"
    r"determined\s+to\s+(?:be|possess)\s+(?:positive\s+)?carcinogenic|"
    r"flagged\s+for\s+(?:its?\s+)?positive\s+carcinogenic)",
    re.IGNORECASE,
)

# Numeric values with correction patterns (for lab values, league IDs, etc.)
CORRECTED_VALUE_RE = re.compile(
    r"(?:corrected|confirmed|amended|rectified|adjusted|revised|finalized)\s+"
    r"(?:to|at|as|the\s+(?:correct|precise|final)\s+(?:value|figure|level|date)\s+"
    r"(?:of|to|as|is)\s*)?\s*"
    r"(\d[\d,]*\.?\d*)",
    re.IGNORECASE,
)

# Gender
GENDER_RE = re.compile(
    r"(?:(?:is|identified\s+as)\s+(?:a\s+)?(male|female)|"
    r"(?:gender|sex)\s+(?:is|marker)\s+(?:is\s+)?(?:currently\s+)?(male|female))",
    re.IGNORECASE,
)

# Height/Weight with units
HEIGHT_RE = re.compile(
    r"height\s+(?:is|of|to be|at)?\s*"
    r"(?:recorded|documented|listed|logged|confirmed|measured|updated)?\s*"
    r"(?:at|as|to be|to)?\s*(\d+\.?\d*)\s*(?:cm|centimeters?)",
    re.IGNORECASE,
)
WEIGHT_RE = re.compile(
    r"weight\s+(?:is|of|at)?\s*"
    r"(?:a\s+(?:substantial|significant|recorded)\s+)?(?:recorded|documented|listed|logged)?\s*"
    r"(?:at|as|to be)?\s*(\d+\.?\d*)\s*(?:kg|kilograms?)",
    re.IGNORECASE,
)

# Generic coded attributes: "X is/classified as/logged with CODE"
CODED_ATTR_PATTERNS = [
    ("publisher_id", re.compile(
        r"(?:publisher\s+(?:affiliation|code)\s+(?:is|was)?\s*"
        r"(?:logged|recorded|listed|classified)?\s*(?:with\s+)?(?:the\s+)?(?:code\s+)?(?:as\s+)?|"
        r"publisher\s+affiliation\s+is\s+(?:logged|recorded|listed|classified)\s+(?:as\s+)?(?:with\s+)?(?:the\s+code\s+)?|"
        r"affiliated\s+with\s+publisher\s+|"
        r"(?:registered|documented|classified)\s+(?:with|under\s+[\w\s]+of)\s+publisher\s+|"
        r"(?:lists?\s+\w+\s+)?publisher\s+affiliation\s+as\s+)"
        r"(\d+)",
        re.IGNORECASE,
    )),
    ("alignment", re.compile(
        r"(?:moral\s+)?alignment\s+(?:is|was)?\s*"
        r"(?:definitively\s+)?(?:classified|listed|logged|recorded|updated)?\s*"
        r"(?:as\s+)?(?:a\s+)?(?:category\s+)?(\d+)",
        re.IGNORECASE,
    )),
    ("gender_id", re.compile(
        r"gender\s+(?:is|was)?\s*(?:classified|listed|logged|recorded)?\s*"
        r"(?:as\s+)?(?:a\s+)?(?:category\s+)?(\d+)",
        re.IGNORECASE,
    )),
    ("race_id", re.compile(
        r"(?:race|species)\s+(?:is|was)?\s*(?:classified|listed|logged|recorded)?\s*"
        r"(?:as\s+)?(\d+\.?\d*)",
        re.IGNORECASE,
    )),
    ("skin_color_id", re.compile(
        r"skin\s+(?:is|color)?\s*(?:a\s+)?(?:category\s+)?(\d+)",
        re.IGNORECASE,
    )),
    ("hair_color_id", re.compile(
        r"hair\s+(?:is|color)?\s*(?:classified|listed|logged)?\s*(?:as\s+)?(?:a\s+)?(\d+)",
        re.IGNORECASE,
    )),
    ("eye_color_id", re.compile(
        r"eye\s+(?:color)?\s*(?:is|are)?\s*(?:classified|listed|logged)?\s*(?:as\s+)?(\d+)",
        re.IGNORECASE,
    )),
    ("country_id", re.compile(
        r"(?:country|national\s+(?:federation|jurisdiction))\s+(?:code|identifier|ID)\s+"
        r"(?:is\s+)?(\d+)",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def detect_id_pattern(text: str, db_path: Path | None = None) -> tuple[str, re.Pattern]:
    """Auto-detect which ID pattern the document uses, statistically."""
    scores: list[tuple[str, re.Pattern, int]] = []

    for name, pattern in ID_CANDIDATES:
        matches = pattern.findall(text)
        # Filter false positives for rec_id
        if name == "rec_id":
            matches = [m for m in matches if not _is_rec_false_positive(m)]
        unique = len(set(matches))
        if unique >= 3:
            scores.append((name, pattern, unique))

    # If DB exists, check which pattern's matches appear in DB columns
    if db_path and db_path.exists() and scores:
        best = _cross_reference_db(text, scores, db_path)
        if best:
            return best

    # Otherwise pick the pattern with most unique matches
    if scores:
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[0][0], scores[0][1]

    # Fallback: rec_id pattern
    return "rec_id", ID_CANDIDATES[0][1]


def _cross_reference_db(
    text: str,
    scores: list[tuple[str, re.Pattern, int]],
    db_path: Path,
) -> tuple[str, re.Pattern] | None:
    """Check which ID pattern's values actually appear in the DB."""
    try:
        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        for name, pattern, count in sorted(scores, key=lambda x: x[2], reverse=True):
            matches = pattern.findall(text)
            if name == "rec_id":
                matches = [m for m in matches if not _is_rec_false_positive(m)]
            sample = list(set(matches))[:10]
            if not sample:
                continue

            for table in tables:
                cols = [r[1] for r in conn.execute(
                    f'PRAGMA table_info("{table}")'
                ).fetchall()]
                for col in cols:
                    placeholders = ",".join("?" * len(sample))
                    try:
                        hit = conn.execute(
                            f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IN ({placeholders})',
                            sample,
                        ).fetchone()[0]
                        if hit >= 2:
                            conn.close()
                            return name, pattern
                    except Exception:
                        continue

        conn.close()
    except Exception:
        pass
    return None


def extract_from_document(
    doc_text: str,
    db_path: Path | None = None,
    doc_name: str = "doc",
) -> ExtractedTable | None:
    """Extract structured records from a document deterministically."""
    id_type, id_pattern = detect_id_pattern(doc_text, db_path)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc_text) if p.strip()]

    records_by_id: dict[str, dict[str, Any]] = {}
    id_order: list[str] = []

    for para in paragraphs:
        ids_in_para = id_pattern.findall(para)
        if id_type == "rec_id":
            ids_in_para = [i for i in ids_in_para if not _is_rec_false_positive(i)]
        if not ids_in_para:
            continue

        primary_id = ids_in_para[0]
        if primary_id not in records_by_id:
            records_by_id[primary_id] = {"_id": primary_id}
            id_order.append(primary_id)

        record = records_by_id[primary_id]
        _extract_attributes(para, record, ids_in_para, id_type)

    if not records_by_id:
        return None

    # Determine table name
    table_name = re.sub(r"[^a-zA-Z0-9_]", "_", doc_name.replace(".md", ""))
    if not table_name:
        table_name = "extracted"

    # Discover FK links
    fk_links: dict[str, str] = {}
    if db_path and db_path.exists():
        fk_links = _find_fk_links(records_by_id, id_order, db_path)

    records_list = [records_by_id[rid] for rid in id_order]

    return ExtractedTable(
        name=table_name,
        id_field="_id",
        records=records_list,
        fk_links=fk_links,
    )


def _extract_attributes(
    para: str, record: dict[str, Any], ids_in_para: list[str], id_type: str
) -> None:
    """Extract all detectable attributes from a paragraph into the record."""
    primary_id = ids_in_para[0]

    # --- Entity name (program name, org name, etc.) ---
    if "name" not in record:
        name = _extract_entity_name(para)
        if name:
            record["name"] = name

    # --- Category ---
    if "category" not in record:
        cat = _extract_category(para)
        if cat:
            record["category"] = cat

    # --- Amount (budget allocation) ---
    if "amount" not in record:
        # Prefer revised/corrected amount
        m = REVISED_AMOUNT_RE.search(para)
        if m and "amount" in para.lower():
            try:
                record["amount"] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass
        if "amount" not in record:
            m = AMOUNT_RE.search(para)
            if m:
                try:
                    record["amount"] = float(m.group(1).replace(",", ""))
                except ValueError:
                    pass

    # --- Spent ---
    if "spent" not in record:
        m = SPENT_RE.search(para)
        if m:
            val = m.group(1) or m.group(2) or m.group(3)
            if val:
                try:
                    record["spent"] = float(val.replace(",", ""))
                except ValueError:
                    pass

    # --- Remaining ---
    if "remaining" not in record:
        m = REMAINING_RE.search(para)
        if m:
            try:
                record["remaining"] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

    # --- Status ---
    if "status" not in record:
        m = STATUS_RE.search(para)
        if m:
            status = m.group(1).strip()
            if status in ("Closed", "Open", "Planning"):
                record["status"] = status

    # --- Event link / secondary ID ---
    if "link" not in record:
        m = EVENT_LINK_RE.search(para)
        if m:
            link_val = m.group(1)
            if link_val != primary_id:
                record["link"] = link_val
        # Also check for secondary IDs in paragraph
        if "link" not in record and len(ids_in_para) > 1:
            for sec_id in ids_in_para[1:]:
                if sec_id != primary_id:
                    record["link"] = sec_id
                    break

    # --- Person name (split into first/last) ---
    if "first_name" not in record:
        m = NAME_RE.search(para)
        if not m:
            m = PERSON_NAME_RE.search(para)
        if m:
            name = m.group(1).strip()
            if name and name not in ("-", "None") and len(name) < 80:
                parts = name.split(None, 1)
                record["first_name"] = parts[0]
                record["last_name"] = parts[1] if len(parts) > 1 else ""

    # --- Bold event name ---
    if "event_name" not in record:
        m = BOLD_RE.search(para)
        if m:
            record["event_name"] = m.group(1).strip()

    # --- Carcinogenic label ---
    if "label" not in record:
        if POSITIVE_LABEL_RE.search(para):
            record["label"] = "+"

    # --- Gender ---
    if "gender" not in record:
        m = GENDER_RE.search(para)
        if m:
            val = (m.group(1) or m.group(2)).strip().upper()[0]
            record["gender"] = "M" if val == "M" else "F"

    # --- Dates ---
    if "date_1" not in record:
        dates = DATE_ISO_RE.findall(para)
        if not dates:
            dates = DATE_WRITTEN_RE.findall(para)
        for i, d in enumerate(dates[:2]):
            key = f"date_{i+1}" if i > 0 else "date_1"
            if key not in record:
                record[key] = d

    # --- Height/Weight ---
    if "height" not in record:
        m = HEIGHT_RE.search(para)
        if m:
            try:
                record["height"] = float(m.group(1))
            except ValueError:
                pass

    if "weight" not in record:
        m = WEIGHT_RE.search(para)
        if m:
            try:
                record["weight"] = float(m.group(1))
            except ValueError:
                pass

    # --- Coded numeric attributes (publisher_id, alignment, etc.) ---
    for attr_name, attr_re in CODED_ATTR_PATTERNS:
        if attr_name not in record:
            m = attr_re.search(para)
            if m:
                try:
                    record[attr_name] = float(m.group(1))
                except ValueError:
                    pass

    # --- Generic numeric values (lab results etc.) ---
    _extract_lab_values(para, record)


def _extract_entity_name(para: str) -> str | None:
    """Extract the entity/program name associated with a record ID in a paragraph."""
    skip = {"The", "This", "It", "A", "An", "We", "Its", "Our", "That", "These"}
    # Try patterns in priority order (corrected > registered > direct > generic)
    for regex in (
        ENTITY_NAME_CORRECTED_RE,
        ENTITY_NAME_REGISTERED_RE,
        ENTITY_NAME_DIRECT_PAREN_RE,
        ENTITY_NAME_BEFORE_ID_RE,
    ):
        m = regex.search(para)
        if m:
            name = m.group(1).strip().rstrip(" ,.-")
            if name and name not in skip and 2 < len(name) < 80:
                name = re.sub(r"\s+(the|a|an|is|was|of|for|to|in|and|or)\s*$", "", name, flags=re.I)
                if len(name) > 2:
                    return name
    return None


def _extract_category(para: str) -> str | None:
    """Try all category patterns, return first valid match."""
    status_words = {"Closed", "Open", "Planning", "None", "The", "This", "It", "A"}

    for regex in (CATEGORY_FINAL_RE, CATEGORY_VERBS, CATEGORY_OF_RE, CATEGORY_PROVIDING_RE):
        m = regex.search(para)
        if m:
            cat = m.group(1).strip().rstrip(".")
            if 2 < len(cat) < 50 and cat not in status_words:
                return cat
    return None


# Lab value patterns: "METRIC was/at/of VALUE U/L" or "METRIC level of VALUE"
LAB_METRICS = [
    "GOT", "GPT", "LDH", "ALP", "T-BIL", "CRE", "creatinine",
    "TP", "ALB", "UA", "UN", "GLU", "WBC", "RBC", "HGB", "HCT",
    "PLT", "PT", "APTT", "FIB", "cholesterol", "triglycerides",
]

LAB_RE_TEMPLATE = (
    r"(?:{metric})\s+(?:was|at|of|level\s+(?:was|of|at)|value\s+(?:of|at|was))?\s*"
    r"(?:confirmed\s+(?:as|at)\s+)?(\d+\.?\d*)"
)


def _extract_lab_values(para: str, record: dict[str, Any]) -> None:
    """Extract lab-style numeric metrics from paragraph."""
    for metric in LAB_METRICS:
        key = metric.lower().replace("-", "_")
        if key in record:
            continue
        # Check for corrected values first
        corrected_pat = re.compile(
            r"(?:" + re.escape(metric) + r").*?"
            r"(?:corrected|confirmed|amended|adjusted|revised|finalized)\s+"
            r"(?:to|at|as|the\s+\w+\s+\w+\s+(?:of|to|as)\s*)?\s*"
            r"(\d+\.?\d*)",
            re.IGNORECASE,
        )
        m = corrected_pat.search(para)
        if m:
            try:
                record[key] = float(m.group(1))
            except ValueError:
                pass
            continue

        # Regular extraction
        pat = re.compile(
            LAB_RE_TEMPLATE.format(metric=re.escape(metric)),
            re.IGNORECASE,
        )
        m = pat.search(para)
        if m:
            try:
                record[key] = float(m.group(1))
            except ValueError:
                pass


def _find_fk_links(
    records: dict[str, dict[str, Any]],
    id_order: list[str],
    db_path: Path,
) -> dict[str, str]:
    """Find FK relationships between extracted IDs and existing DB columns."""
    fk_links: dict[str, str] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        sample_ids = id_order[:10]

        # Check if primary IDs match any DB column
        for table in tables:
            cols = [r[1] for r in conn.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()]
            for col in cols:
                placeholders = ",".join("?" * len(sample_ids))
                try:
                    hit = conn.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IN ({placeholders})',
                        sample_ids,
                    ).fetchone()[0]
                    if hit >= 2:
                        fk_links["_id"] = f"{table}.{col}"
                        break
                except Exception:
                    continue
            if "_id" in fk_links:
                break

        # Check if "link" values match any DB column
        link_vals = [
            records[rid].get("link")
            for rid in id_order
            if records[rid].get("link")
        ][:10]
        if link_vals:
            for table in tables:
                cols = [r[1] for r in conn.execute(
                    f'PRAGMA table_info("{table}")'
                ).fetchall()]
                for col in cols:
                    placeholders = ",".join("?" * len(link_vals))
                    try:
                        hit = conn.execute(
                            f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IN ({placeholders})',
                            link_vals,
                        ).fetchone()[0]
                        if hit >= 2:
                            fk_links["link"] = f"{table}.{col}"
                            break
                    except Exception:
                        continue
                if "link" in fk_links:
                    break

        conn.close()
    except Exception:
        pass
    return fk_links


def write_extracted_table(db_path: Path, table: ExtractedTable) -> None:
    """Write extracted records into the SQLite database."""
    if not table.records:
        return

    # Gather all column names and infer types
    all_cols: dict[str, str] = {}
    for r in table.records:
        for k, v in r.items():
            if k not in all_cols:
                if isinstance(v, float):
                    all_cols[k] = "REAL"
                elif isinstance(v, int):
                    all_cols[k] = "INTEGER"
                else:
                    all_cols[k] = "TEXT"

    if not all_cols:
        return

    conn = sqlite3.connect(str(db_path))

    col_defs = ", ".join(f'"{c}" {t}' for c, t in all_cols.items())
    conn.execute(f'DROP TABLE IF EXISTS "{table.name}"')
    conn.execute(f'CREATE TABLE "{table.name}" ({col_defs})')

    col_names = list(all_cols.keys())
    placeholders = ", ".join("?" * len(col_names))
    quoted_cols = ", ".join(f'"{c}"' for c in col_names)
    insert_sql = f'INSERT INTO "{table.name}" ({quoted_cols}) VALUES ({placeholders})'

    for r in table.records:
        values = [r.get(c) for c in col_names]
        try:
            conn.execute(insert_sql, values)
        except Exception:
            continue

    conn.commit()
    conn.close()


def extract_all_docs(
    doc_paths: list[Path],
    db_path: Path | None = None,
) -> list[ExtractedTable]:
    """Extract tables from all document files."""
    tables: list[ExtractedTable] = []
    for doc_path in doc_paths:
        text = doc_path.read_text(errors="replace")
        if len(text) < 100:
            continue
        result = extract_from_document(
            text, db_path=db_path, doc_name=doc_path.stem
        )
        if result and result.records:
            tables.append(result)
    return tables
