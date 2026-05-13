"""Deterministic Knowledge Graph builder from SQLite schema.

Introspects a consolidated SQLite database and produces a KG metadata
structure that captures tables, columns, types, primary keys, foreign keys
(both explicit and inferred), and sample values.

No LLM calls — purely code-based.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    sql_type: str
    is_pk: bool = False
    is_nullable: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass(slots=True)
class TableSchema:
    name: str
    columns: list[Column] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    row_count: int = 0
    sample_values: dict[str, list[Any]] = field(default_factory=dict)
    col_stats: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeGraph:
    tables: list[TableSchema] = field(default_factory=list)
    inferred_fks: list[tuple[str, ForeignKey]] = field(default_factory=list)

    def get_table(self, name: str) -> TableSchema | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None

    def all_foreign_keys(self) -> list[tuple[str, ForeignKey]]:
        """Return all FKs as (source_table, FK) pairs."""
        result = []
        for t in self.tables:
            for fk in t.foreign_keys:
                result.append((t.name, fk))
        for src, fk in self.inferred_fks:
            result.append((src, fk))
        return result


def build_kg_from_sqlite(db_path: Path) -> KnowledgeGraph:
    """Introspect SQLite DB and build a KnowledgeGraph metadata structure."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    tables = _discover_tables(conn)
    kg = KnowledgeGraph(tables=tables)

    # Infer implicit FKs: name patterns first, then validate via value overlap
    kg.inferred_fks = _infer_foreign_keys(conn, tables)

    conn.close()
    return kg


def _discover_tables(conn: sqlite3.Connection) -> list[TableSchema]:
    """Extract schema for all tables in the DB."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    table_names = [row[0] for row in cursor.fetchall()]

    tables: list[TableSchema] = []
    for name in sorted(table_names):
        schema = _introspect_table(conn, name)
        if schema:
            tables.append(schema)
    return tables


def _introspect_table(conn: sqlite3.Connection, table_name: str) -> TableSchema | None:
    """Get full schema info for a single table."""
    try:
        cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
        col_info = cursor.fetchall()
    except Exception:
        return None

    if not col_info:
        return None

    columns: list[Column] = []
    primary_keys: list[str] = []
    for row in col_info:
        col = Column(
            name=row[1],
            sql_type=row[2] or "TEXT",
            is_pk=bool(row[5]),
            is_nullable=not bool(row[3]),
        )
        columns.append(col)
        if col.is_pk:
            primary_keys.append(col.name)

    # Get explicit foreign keys
    foreign_keys: list[ForeignKey] = []
    try:
        fk_cursor = conn.execute(f"PRAGMA foreign_key_list('{table_name}')")
        for fk_row in fk_cursor.fetchall():
            foreign_keys.append(ForeignKey(
                column=fk_row[3],
                ref_table=fk_row[2],
                ref_column=fk_row[4],
            ))
    except Exception:
        pass

    # Row count
    try:
        count_row = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
        row_count = count_row[0] if count_row else 0
    except Exception:
        row_count = 0

    # Sample values: for text columns get distinct values (more useful for filters)
    sample_values: dict[str, list[Any]] = {}
    col_names = [c.name for c in columns]
    for i, col in enumerate(columns):
        try:
            if col.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", ""):
                # Get distinct values for text columns (up to 8)
                cursor = conn.execute(
                    f'SELECT DISTINCT "{col.name}" FROM "{table_name}" '
                    f'WHERE "{col.name}" IS NOT NULL LIMIT 8'
                )
                vals = [row[0] for row in cursor.fetchall()]
            else:
                cursor = conn.execute(
                    f'SELECT "{col.name}" FROM "{table_name}" '
                    f'WHERE "{col.name}" IS NOT NULL LIMIT 5'
                )
                vals = [row[0] for row in cursor.fetchall()]
            if vals:
                sample_values[col.name] = vals
        except Exception:
            pass

    # Column statistics: cardinality, min/max/avg for numeric columns
    col_stats: dict[str, dict[str, Any]] = {}
    for col in columns:
        stats: dict[str, Any] = {}
        try:
            distinct_count = conn.execute(
                f'SELECT COUNT(DISTINCT "{col.name}") FROM "{table_name}" '
                f'WHERE "{col.name}" IS NOT NULL'
            ).fetchone()[0]
            stats["distinct"] = distinct_count
            # Numeric stats
            if col.sql_type.upper() in ("INTEGER", "INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE"):
                row = conn.execute(
                    f'SELECT MIN("{col.name}"), MAX("{col.name}"), AVG("{col.name}") '
                    f'FROM "{table_name}" WHERE "{col.name}" IS NOT NULL'
                ).fetchone()
                if row and row[0] is not None:
                    stats["min"] = row[0]
                    stats["max"] = row[1]
                    stats["avg"] = round(row[2], 2) if row[2] is not None else None
        except Exception:
            pass
        if stats:
            col_stats[col.name] = stats

    return TableSchema(
        name=table_name,
        columns=columns,
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        row_count=row_count,
        sample_values=sample_values,
        col_stats=col_stats,
    )


def _resolve_table_name(ref_name: str, table_names_lower: dict[str, str]) -> str | None:
    """Try to match a reference name to an actual table (handles plural/singular)."""
    if ref_name in table_names_lower:
        return ref_name
    # Plural forms
    if ref_name + "s" in table_names_lower:
        return ref_name + "s"
    if ref_name + "es" in table_names_lower:
        return ref_name + "es"
    if ref_name + "ies" in table_names_lower:
        return ref_name + "ies"
    # Singular forms
    if ref_name.endswith("s") and ref_name[:-1] in table_names_lower:
        return ref_name[:-1]
    if ref_name.endswith("es") and ref_name[:-2] in table_names_lower:
        return ref_name[:-2]
    if ref_name.endswith("ies") and ref_name[:-3] + "y" in table_names_lower:
        return ref_name[:-3] + "y"
    # Underscore/space variants: "order_item" ↔ "order_items", "orderitem"
    if "_" in ref_name:
        joined = ref_name.replace("_", "")
        if joined in table_names_lower:
            return joined
    else:
        for tname in table_names_lower:
            if tname.replace("_", "") == ref_name:
                return tname
    return None


def _find_id_column(
    ref_table_name: str, src_col_name: str, tables: list[TableSchema]
) -> str:
    """Find the best ID column in the referenced table.

    Priority: "id" > same column name as source > "_id" > first column.
    """
    ref_table = next((t for t in tables if t.name == ref_table_name), None)
    if not ref_table:
        return "id"
    col_names = {c.name.lower(): c.name for c in ref_table.columns}
    if "id" in col_names:
        return col_names["id"]
    if src_col_name.lower() in col_names:
        return col_names[src_col_name.lower()]
    if "_id" in col_names:
        return col_names["_id"]
    # Try PK columns
    for c in ref_table.columns:
        if c.is_pk:
            return c.name
    return ref_table.columns[0].name if ref_table.columns else "id"


def _extract_ref_name(col_name: str) -> str | None:
    """Extract referenced entity name from various FK naming conventions.

    Handles: customer_id, customerId, CustomerID, customer_key, customer_no,
    customer_code, fk_customer, ref_customer, id_customer
    """
    col_lower = col_name.lower()

    # <table>_id / <table>_key / <table>_no / <table>_code / <table>_num
    m = re.match(r"^(.+?)_(?:id|key|no|num|code)$", col_lower)
    if m:
        return m.group(1)

    # fk_<table> / ref_<table> / id_<table>
    m = re.match(r"^(?:fk|ref|id)_(.+)$", col_lower)
    if m:
        return m.group(1)

    # camelCase: customerId, CustomerID, eventID
    m = re.match(r"^(.+?)(?:Id|ID|Key|Code|No|Num)$", col_name)
    if m and len(m.group(1)) > 1:
        return m.group(1).lower()

    return None


def _infer_foreign_keys(
    conn: sqlite3.Connection, tables: list[TableSchema]
) -> list[tuple[str, ForeignKey]]:
    """Infer FK relationships using name patterns + value overlap validation.

    Returns (source_table, ForeignKey) tuples.
    Two-pass approach:
    1. Generate candidates from column name patterns (cheap)
    2. Validate each candidate by checking value overlap in actual data
    Also discovers relationships purely from value overlap for columns with
    matching names across tables (handles CSV/JSON tables with no declared PKs).
    """
    explicit_fk_cols: dict[str, set[str]] = {}
    for t in tables:
        explicit_fk_cols[t.name] = {fk.column.lower() for fk in t.foreign_keys}

    # Build index of unique-valued columns (likely PKs or join keys)
    unique_cols: dict[str, list[tuple[str, str]]] = {}  # col_lower -> [(table, col)]
    for t in tables:
        for col in t.columns:
            col_lower = col.name.lower()
            unique_cols.setdefault(col_lower, []).append((t.name, col.name))

    # Generate candidates from naming patterns
    # Each candidate: (src_table, src_col, ref_table, ref_col, name_match)
    candidates: list[tuple[str, str, str, str, bool]] = []

    table_names_lower = {t.name.lower(): t.name for t in tables}

    for table in tables:
        for col in table.columns:
            col_lower = col.name.lower()

            if col_lower in explicit_fk_cols.get(table.name, set()):
                continue

            # Pattern 1: naming convention → referenced table
            # Handles: customer_id, customerId, CustomerID, customer_key,
            #          fk_customer, ref_customer, id_customer, customer_code
            ref_name = _extract_ref_name(col.name)
            if ref_name:
                ref_match = _resolve_table_name(ref_name, table_names_lower)
                if ref_match and ref_match != table.name.lower():
                    ref_col = _find_id_column(
                        table_names_lower[ref_match], col.name, tables
                    )
                    candidates.append((
                        table.name, col.name,
                        table_names_lower[ref_match], ref_col, True,
                    ))

            # Pattern 2: "link_to_<table>" → table's ID column
            m3 = re.match(r"^link_to_(.+)$", col_lower)
            if m3:
                link_ref = m3.group(1)
                ref_match = _resolve_table_name(link_ref, table_names_lower)
                if ref_match and ref_match != table.name.lower():
                    ref_col = _find_id_column(
                        table_names_lower[ref_match], col.name, tables
                    )
                    candidates.append((
                        table.name, col.name,
                        table_names_lower[ref_match], ref_col, True,
                    ))

            # Pattern 3: same column name exists in another table (shared key)
            # Only for ID-like columns to avoid false positives (e.g. "Diagnosis")
            if col_lower in unique_cols and _is_joinable_column(col_lower):
                for ref_table_name, ref_col_name in unique_cols[col_lower]:
                    if ref_table_name == table.name:
                        continue
                    candidates.append((
                        table.name, col.name,
                        ref_table_name, ref_col_name,
                        _is_specific_id(col_lower),
                    ))

            # Pattern 4: _id ↔ ID equivalence (doc extraction uses _id, CSVs use ID)
            if col_lower == "_id":
                for ref_table_name, ref_col_name in unique_cols.get("id", []):
                    if ref_table_name == table.name:
                        continue
                    candidates.append((
                        table.name, col.name,
                        ref_table_name, ref_col_name, False,
                    ))
            elif col_lower == "id":
                for ref_table_name, ref_col_name in unique_cols.get("_id", []):
                    if ref_table_name == table.name:
                        continue
                    candidates.append((
                        table.name, col.name,
                        ref_table_name, ref_col_name, False,
                    ))

    # Validate candidates via value overlap + uniqueness analysis
    inferred: list[tuple[str, ForeignKey]] = []
    seen: set[tuple[str, str, str, str]] = set()
    # Track bidirectional pairs to avoid A->B and B->A for same column
    linked_pairs: set[tuple[str, str, str]] = set()

    # Pre-compute uniqueness ratios for smarter direction detection
    uniqueness_cache: dict[tuple[str, str], float] = {}

    def _get_uniqueness(tbl: str, col: str) -> float:
        key = (tbl, col)
        if key not in uniqueness_cache:
            try:
                row_count = conn.execute(
                    f'SELECT COUNT(*) FROM "{tbl}"'
                ).fetchone()[0]
                if row_count == 0:
                    uniqueness_cache[key] = 0.0
                else:
                    distinct = conn.execute(
                        f'SELECT COUNT(DISTINCT "{col}") FROM "{tbl}"'
                    ).fetchone()[0]
                    uniqueness_cache[key] = distinct / row_count
            except Exception:
                uniqueness_cache[key] = 0.0
        return uniqueness_cache[key]

    for src_table, src_col, ref_table, ref_col, name_match in candidates:
        key = (src_table, src_col, ref_table, ref_col)
        if key in seen:
            continue
        seen.add(key)

        if src_table == ref_table:
            continue

        # Normalize direction: FK should point from many-side → one-side
        # (source has duplicates, reference is unique)
        pair = tuple(sorted([src_table, ref_table]))
        col_pair = (pair[0], pair[1], src_col.lower())
        if col_pair in linked_pairs:
            continue

        if _check_value_overlap(
            conn, src_table, src_col, ref_table, ref_col, name_match=name_match
        ):
            # Determine correct direction using uniqueness
            src_uniq = _get_uniqueness(src_table, src_col)
            ref_uniq = _get_uniqueness(ref_table, ref_col)

            # If ref is more unique (PK-like), direction is correct: src → ref
            # If src is more unique, flip direction: ref → src
            if ref_uniq >= src_uniq:
                inferred.append((src_table, ForeignKey(
                    column=src_col,
                    ref_table=ref_table,
                    ref_column=ref_col,
                )))
            else:
                inferred.append((ref_table, ForeignKey(
                    column=ref_col,
                    ref_table=src_table,
                    ref_column=src_col,
                )))
            linked_pairs.add(col_pair)

    return inferred


def _is_joinable_column(col_name: str) -> bool:
    """Check if a column name looks like a join key (for Pattern 3 — shared names)."""
    col_lower = col_name.lower()
    return (
        col_lower == "id"
        or col_lower.endswith("_id")
        or col_lower.endswith("id") and len(col_lower) > 2
        or col_lower in ("key", "code", "number", "no", "num")
    )


def _is_specific_id(col_name: str) -> bool:
    """Check if a column name is a specific (non-generic) ID.

    Specific IDs like "CustomerID", "molecule_id" get name_match=True (lenient).
    Generic "id"/"Id" gets name_match=False (requires real overlap).
    """
    col_lower = col_name.lower()
    return col_lower not in ("id",) and (
        col_lower.endswith("_id") or col_lower.endswith("id") and len(col_lower) > 2
    )


def _check_value_overlap(
    conn: sqlite3.Connection,
    src_table: str, src_col: str,
    ref_table: str, ref_col: str,
    name_match: bool = False,
    sample_size: int = 50,
    min_overlap: float = 0.3,
) -> bool:
    """Check if values in src_table.src_col overlap with ref_table.ref_col.

    Returns True if:
    - name_match and ≥1 match exists (strong naming signal), OR
    - ≥min_overlap of sampled src values exist in ref column
    Rejects bare auto-increment ID overlap (both start from 1 sequentially).
    """
    try:
        # Verify ref_col exists in ref_table
        ref_cols = conn.execute(f'PRAGMA table_info("{ref_table}")').fetchall()
        ref_col_names = [r[1].lower() for r in ref_cols]
        if ref_col.lower() not in ref_col_names:
            for actual_name in [r[1] for r in ref_cols]:
                if actual_name.lower() == "id":
                    ref_col = actual_name
                    break
            else:
                return False

        # For bare "id"/"Id" columns: check if both are independent PKs
        is_one_to_many = False
        if src_col.lower() == "id" and ref_col.lower() == "id":
            if _both_are_pks(conn, src_table, src_col, ref_table, ref_col):
                return False
            # One side is PK, other has duplicates → classic FK pattern, be lenient
            is_one_to_many = True

        # Sample distinct non-null values from source
        src_vals = conn.execute(
            f'SELECT DISTINCT "{src_col}" FROM "{src_table}" '
            f'WHERE "{src_col}" IS NOT NULL LIMIT {sample_size}'
        ).fetchall()

        if not src_vals:
            return False

        # Check how many exist in ref table
        matches = 0
        for (val,) in src_vals:
            hit = conn.execute(
                f'SELECT 1 FROM "{ref_table}" WHERE "{ref_col}" = ? LIMIT 1',
                (val,),
            ).fetchone()
            if hit:
                matches += 1

        overlap = matches / len(src_vals)

        if name_match or is_one_to_many:
            return matches > 0

        return overlap >= min_overlap

    except Exception:
        return False


def _both_are_pks(
    conn: sqlite3.Connection,
    table_a: str, col_a: str,
    table_b: str, col_b: str,
) -> bool:
    """Check if both id columns are primary keys of their respective tables.

    If both columns have unique values (distinct count = row count), they're
    both PKs and not FK references to each other.
    True FK pattern: src has duplicates (many-to-one) pointing to unique ref.
    """
    try:
        count_a = conn.execute(f'SELECT COUNT(*) FROM "{table_a}"').fetchone()[0]
        distinct_a = conn.execute(
            f'SELECT COUNT(DISTINCT "{col_a}") FROM "{table_a}"'
        ).fetchone()[0]

        count_b = conn.execute(f'SELECT COUNT(*) FROM "{table_b}"').fetchone()[0]
        distinct_b = conn.execute(
            f'SELECT COUNT(DISTINCT "{col_b}") FROM "{table_b}"'
        ).fetchone()[0]

        # Both columns are unique (PK-like) → independent tables, not FK
        a_is_unique = distinct_a >= count_a * 0.95
        b_is_unique = distinct_b >= count_b * 0.95

        if a_is_unique and b_is_unique:
            return True

        return False
    except Exception:
        return False


def format_kg_for_llm(kg: KnowledgeGraph, max_sample_values: int = 8) -> str:
    """Format the KG metadata as a compact text string for LLM context.

    This is the grounding text that every LLM call receives.
    """
    lines: list[str] = []
    lines.append("=== DATABASE SCHEMA ===")
    lines.append("")

    for table in kg.tables:
        pk_str = ", ".join(table.primary_keys) if table.primary_keys else "(none)"
        lines.append(f"TABLE: {table.name} ({table.row_count} rows, PK: {pk_str})")

        for col in table.columns:
            nullable = "" if col.is_nullable else " NOT NULL"
            pk_mark = " [PK]" if col.is_pk else ""
            sample = ""
            if col.name in table.sample_values:
                vals = table.sample_values[col.name][:max_sample_values]
                sample = f"  e.g. {vals}"
            # Add compact stats
            stats_str = ""
            if col.name in table.col_stats:
                st = table.col_stats[col.name]
                stat_parts = []
                if "distinct" in st:
                    stat_parts.append(f"{st['distinct']} unique")
                if "min" in st and "max" in st:
                    stat_parts.append(f"range [{st['min']}..{st['max']}]")
                if stat_parts:
                    stats_str = f"  ({', '.join(stat_parts)})"
            desc_str = f"  -- {col.description}" if col.description else ""
            lines.append(f"  - {col.name} ({col.sql_type}{nullable}){pk_mark}{desc_str}{stats_str}{sample}")

        # Explicit FKs
        for fk in table.foreign_keys:
            lines.append(f"  FK: {fk.column} → {fk.ref_table}.{fk.ref_column}")

        lines.append("")

    # Inferred relationships
    if kg.inferred_fks:
        lines.append("=== INFERRED RELATIONSHIPS ===")
        for src_table, fk in kg.inferred_fks:
            lines.append(f"  {src_table}.{fk.column} → {fk.ref_table}.{fk.ref_column}")
        lines.append("")

    # Relationship summary for quick reference
    all_fks = _collect_all_relationships(kg)
    if all_fks:
        lines.append("=== JOIN PATHS ===")
        for src_table, fk in all_fks:
            lines.append(
                f"  JOIN {fk.ref_table} ON {src_table}.{fk.column} = {fk.ref_table}.{fk.ref_column}"
            )
        lines.append("")

    return "\n".join(lines)


def _collect_all_relationships(kg: KnowledgeGraph) -> list[tuple[str, ForeignKey]]:
    """Collect all FK relationships (explicit + inferred) with source table."""
    result: list[tuple[str, ForeignKey]] = []

    for table in kg.tables:
        for fk in table.foreign_keys:
            result.append((table.name, fk))

    for src_table, fk in kg.inferred_fks:
        result.append((src_table, fk))

    return result


_ENRICH_PROMPT = """\
You are a data dictionary expert. Given a database schema with sample values and \
optional domain knowledge, produce a short (≤12 words) semantic description for each column.

Focus on disambiguating columns that could be confused with each other. \
For columns whose meaning is obvious AND unique across all tables, return empty string "". \
ALWAYS describe columns that appear in MULTIPLE tables — explain how each table's version differs. \
Each description must be ≤8 words.

SCHEMA:
{schema}

DOMAIN KNOWLEDGE (if available):
{knowledge_text}

Return ONLY a JSON object mapping "table.column" to its description string. \
Omit columns that need no description (obvious from name).
Example: {{"frpm.District Type": "district organizational category", \
"frpm.Charter Funding Type": "charter school funding method"}}
"""


def enrich_kg_with_descriptions(
    kg: KnowledgeGraph,
    model: ModelAdapter,
    knowledge_text: str = "",
    log_fn: Callable[..., None] | None = None,
) -> KnowledgeGraph:
    """Use LLM to generate semantic descriptions for KG columns."""
    schema_lines: list[str] = []
    for table in kg.tables:
        schema_lines.append(f"TABLE: {table.name}")
        for col in table.columns:
            sample = ""
            if col.name in table.sample_values:
                vals = table.sample_values[col.name][:6]
                sample = f"  e.g. {vals}"
            schema_lines.append(f"  - {col.name} ({col.sql_type}){sample}")
        schema_lines.append("")

    schema_text = "\n".join(schema_lines)
    # Cap schema to avoid blowing context
    if len(schema_text) > 6000:
        schema_text = schema_text[:6000]

    prompt = _ENRICH_PROMPT.format(
        schema=schema_text,
        knowledge_text=knowledge_text[:3000] if knowledge_text else "(none)",
    )
    messages = [ModelMessage(role="user", content=prompt)]
    raw = model.complete(messages)

    # Parse JSON from response
    descriptions: dict[str, str] = {}
    try:
        # Try direct parse
        descriptions = json.loads(raw)
    except json.JSONDecodeError:
        # Extract JSON from markdown fences
        m = re.search(r"```(?:json)?\s*\n(.+?)```", raw, re.DOTALL)
        if m:
            try:
                descriptions = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    if not descriptions:
        if log_fn:
            log_fn("kg_enrich_fail", "Failed to parse column descriptions from LLM")
        return kg

    # Apply descriptions to columns
    new_tables: list[TableSchema] = []
    applied = 0
    for table in kg.tables:
        new_cols: list[Column] = []
        for col in table.columns:
            key = f"{table.name}.{col.name}"
            desc = descriptions.get(key, "")
            if desc:
                new_cols.append(replace(col, description=desc))
                applied += 1
            else:
                new_cols.append(col)
        new_tables.append(TableSchema(
            name=table.name,
            columns=new_cols,
            primary_keys=table.primary_keys,
            foreign_keys=table.foreign_keys,
            row_count=table.row_count,
            sample_values=table.sample_values,
            col_stats=table.col_stats,
        ))

    if log_fn:
        log_fn("kg_enriched", f"{applied} column descriptions added")

    return KnowledgeGraph(tables=new_tables, inferred_fks=kg.inferred_fks)
