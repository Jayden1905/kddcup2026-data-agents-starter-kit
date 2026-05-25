"""KG Tools: tool functions for the KGAgent loop.

Each tool takes structured inputs and returns a formatted string observation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.pipeline.kg_builder import KnowledgeGraph

STEM_RATIO = 0.6


def _term_matches(col_lower: str, q_terms: set[str]) -> bool:
    """Check if a column name is relevant to question terms.

    Matches on: exact, substring either direction, or shared stem (common prefix
    >= 60% of the shorter word). Handles morphological variants like
    "diagnosed" ↔ "diagnosis" without false-matching unrelated words.
    """
    if col_lower in q_terms:
        return True
    for t in q_terms:
        if t in col_lower or col_lower in t:
            return True
        shorter = min(len(t), len(col_lower))
        if shorter < 3:
            continue
        common = 0
        for a, b in zip(t, col_lower):
            if a != b:
                break
            common += 1
        if common / shorter >= STEM_RATIO:
            return True
    return False


def tool_overview(kg: KnowledgeGraph, question: str) -> str:
    """Tables with roles, row counts, grain, PKs, joins. Highlights question-relevant columns."""
    q_terms = {w.lower() for w in question.split() if len(w) > 2}
    lines = []
    for t in kg.tables:
        role_tag = f" [{t.role}]" if t.role else ""
        grain = f", grain=[{', '.join(t.grain_columns)}]" if t.grain_columns else ""
        pk = f", PK=[{', '.join(t.primary_keys)}]" if t.primary_keys else ""
        col_parts = []
        for c in t.columns:
            c_lower = c.name.lower()
            if _term_matches(c_lower, q_terms):
                col_parts.append(f"**{c.name}**")
            else:
                col_parts.append(c.name)
        cols = ", ".join(col_parts)
        lines.append(f"{t.name}{role_tag} ({t.row_count} rows{pk}{grain}): {cols}")

    fks = kg.all_foreign_keys()
    if fks:
        lines.append("\nJOINS:")
        for src, fk in fks:
            lines.append(f"  {src}.{fk.column} -> {fk.ref_table}.{fk.ref_column}")

    return "\n".join(lines)


def tool_schema(kg: KnowledgeGraph, table: str, question: str, knowledge_text: str) -> str:
    """Full table detail: columns, types, stats, samples, semantic roles, collisions."""
    text, _ = _tool_schema_inner(kg, table, question, knowledge_text)
    return text


def detect_ambiguities(
    action: str,
    kg: KnowledgeGraph,
    question: str,
    db_path: Path,
    *,
    table: str = "",
    tables: list[str] | None = None,
    sql: str = "",
) -> list[dict[str, Any]]:
    """Generic ambiguity detection after any tool call.

    Returns list of ambiguity dicts: {"type": str, "description": str, "evidence": str}
    Each ambiguity must be resolved before the agent continues.
    """
    ambiguities: list[dict[str, Any]] = []

    if action == "schema":
        ambiguities.extend(_detect_column_collisions(kg, table, question, db_path))

    if action == "topology":
        ambiguities.extend(_detect_join_ambiguity(kg, tables or [], db_path))

    if action == "run_sql" and sql:
        ambiguities.extend(_detect_sql_ambiguity(kg, sql, question, db_path))

    return ambiguities


def format_resolution_prompt(ambiguities: list[dict[str, Any]], question: str) -> str:
    """Format a generic resolution prompt from detected ambiguities."""
    parts = ["AMBIGUITY DETECTED — you must resolve before continuing.\n"]
    for i, amb in enumerate(ambiguities, 1):
        parts.append(f"--- Ambiguity {i}: {amb['type']} ---")
        parts.append(amb["description"])
        if amb.get("evidence"):
            parts.append(amb["evidence"])
        parts.append("")

    parts.append(f"QUESTION: {question}\n")
    parts.append(
        "Resolve based on the QUESTION wording (it overrides domain knowledge examples). "
        "For each ambiguity, state which option to use. "
        'Respond with a JSON object: {"resolved": {"<item>": "<choice>", ...}}'
    )
    return "\n".join(parts)


def _detect_column_collisions(
    kg: KnowledgeGraph, table: str, question: str, db_path: Path
) -> list[dict[str, Any]]:
    """Detect question-relevant column collisions for a table."""
    ts = kg.get_table(table)
    if not ts:
        return []

    q_terms = {w.lower() for w in question.split() if len(w) > 2}
    col_names = {c.name for c in ts.columns}
    relevant = set()
    for col in ts.columns:
        col_lower = col.name.lower()
        if _term_matches(col_lower, q_terms):
            relevant.add(col.name)

    ambiguities: list[dict[str, Any]] = []
    seen: set[str] = set()
    ts = kg.get_table(table)
    for other_t in kg.tables:
        if other_t.name == table:
            continue
        for c in other_t.columns:
            if c.name in col_names and c.name in relevant and c.name not in seen:
                seen.add(c.name)
                evidence = _sample_column_comparison(db_path, c.name, table, other_t.name)
                # Add semantic context: table roles and whether column is a
                # primary attribute of the entity or a measurement in a fact table
                role_a = ts.role if ts else "unknown"
                role_b = other_t.role
                context = (
                    f'  Context: "{table}" is a {role_a} table; '
                    f'"{other_t.name}" is a {role_b} table.\n'
                    f'  Hint: If the question asks about a property OF an entity '
                    f'(e.g. "his number", "their name"), prefer the entity/dimension table.'
                )
                ambiguities.append(
                    {
                        "type": "column_collision",
                        "description": (
                            f'Column "{c.name}" exists in both "{table}" and "{other_t.name}". '
                            f"Which table's column is relevant to the question?"
                        ),
                        "evidence": f"{evidence}\n{context}" if evidence else context,
                    }
                )
    return ambiguities


def _detect_join_ambiguity(
    kg: KnowledgeGraph, tables: list[str], db_path: Path
) -> list[dict[str, Any]]:
    """Detect when multiple join paths exist between requested tables."""
    if len(tables) < 2:
        return []

    fks = kg.all_foreign_keys()
    ambiguities: list[dict[str, Any]] = []

    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            t_a, t_b = tables[i], tables[j]
            paths = []
            for src, fk in fks:
                if src == t_a and fk.ref_table == t_b:
                    paths.append(f"{src}.{fk.column} → {fk.ref_table}.{fk.ref_column}")
                elif src == t_b and fk.ref_table == t_a:
                    paths.append(f"{src}.{fk.column} → {fk.ref_table}.{fk.ref_column}")
            # Also check indirect paths (A→X→B)
            for src, fk in fks:
                if src == t_a:
                    mid = fk.ref_table
                    for src2, fk2 in fks:
                        if src2 == mid and fk2.ref_table == t_b:
                            paths.append(
                                f"{src}.{fk.column} → {mid}.{fk.ref_column} → "
                                f"{fk2.ref_table}.{fk2.ref_column}"
                            )

            if len(paths) > 1:
                ambiguities.append(
                    {
                        "type": "join_ambiguity",
                        "description": (
                            f"Multiple join paths between {t_a} and {t_b}:\n"
                            + "\n".join(f"  {p}" for p in paths)
                        ),
                        "evidence": "",
                    }
                )
    return ambiguities


def _detect_sql_ambiguity(
    kg: KnowledgeGraph, sql: str, question: str, db_path: Path
) -> list[dict[str, Any]]:
    """Detect when SQL references columns that exist in multiple tables.

    Only fires when the SQL actually involves multiple tables (JOIN or subquery
    referencing another table) — single-table queries are unambiguous.
    """
    q_terms = {w.lower() for w in question.split() if len(w) > 2}

    all_table_names = {t.name for t in kg.tables}
    tables_in_sql = {t for t in all_table_names if f'"{t}"' in sql}
    if len(tables_in_sql) < 2:
        return []

    all_cols: dict[str, list[str]] = {}
    for t in kg.tables:
        for c in t.columns:
            all_cols.setdefault(c.name, []).append(t.name)

    ambiguities: list[dict[str, Any]] = []
    for col_name, owning_tables in all_cols.items():
        if len(owning_tables) < 2:
            continue
        tables_present = [t for t in owning_tables if t in tables_in_sql]
        if len(tables_present) < 2:
            continue
        col_lower = col_name.lower()
        if not _term_matches(col_lower, q_terms):
            continue
        if f'"{col_name}"' in sql or col_name in sql:
            qualified = any(f'"{t}"."{col_name}"' in sql for t in tables_present)
            if not qualified:
                evidence = _sample_column_comparison(db_path, col_name, *tables_present[:2])
                ambiguities.append(
                    {
                        "type": "unqualified_column",
                        "description": (
                            f'Column "{col_name}" used without table qualifier but exists in: '
                            f"{', '.join(tables_present)}. Which table should it come from?"
                        ),
                        "evidence": evidence,
                    }
                )
    return ambiguities


def _sample_column_comparison(db_path: Path, col_name: str, table_a: str, table_b: str) -> str:
    """Compare a column's values across two tables. Returns formatted comparison."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        cur_a = conn.execute(
            f'SELECT DISTINCT "{col_name}" FROM "{table_a}" '
            f'WHERE "{col_name}" IS NOT NULL AND "{col_name}" != \'\' LIMIT 8'
        )
        vals_a = [str(r[0]) for r in cur_a.fetchall()]
        cur_b = conn.execute(
            f'SELECT DISTINCT "{col_name}" FROM "{table_b}" '
            f'WHERE "{col_name}" IS NOT NULL AND "{col_name}" != \'\' LIMIT 8'
        )
        vals_b = [str(r[0]) for r in cur_b.fetchall()]

        cur_a2 = conn.execute(
            f'SELECT COUNT(*), SUM(CASE WHEN "{col_name}" IS NULL OR "{col_name}" = \'\' '
            f'THEN 1 ELSE 0 END) FROM "{table_a}"'
        )
        total_a, empty_a = cur_a2.fetchone()
        cur_b2 = conn.execute(
            f'SELECT COUNT(*), SUM(CASE WHEN "{col_name}" IS NULL OR "{col_name}" = \'\' '
            f'THEN 1 ELSE 0 END) FROM "{table_b}"'
        )
        total_b, empty_b = cur_b2.fetchone()
        conn.close()

        empty_pct_a = (empty_a or 0) / max(total_a, 1) * 100
        empty_pct_b = (empty_b or 0) / max(total_b, 1) * 100

        lines = [
            f'  {table_a}."{col_name}": {vals_a} ({empty_pct_a:.0f}% empty)',
            f'  {table_b}."{col_name}": {vals_b} ({empty_pct_b:.0f}% empty)',
        ]
        return "\n".join(lines)
    except Exception:
        return ""


def _tool_schema_inner(
    kg: KnowledgeGraph, table: str, question: str, knowledge_text: str
) -> tuple[str, list[tuple[str, str, str]]]:
    """Returns (formatted_schema, collisions_list)."""
    ts = kg.get_table(table)
    if not ts:
        available = [t.name for t in kg.tables]
        return f"Table '{table}' not found. Available: {available}", []

    lines = [f"TABLE: {table} (role={ts.role}, {ts.row_count} rows)"]

    q_terms = {w.lower() for w in question.split() if len(w) > 2}
    relevant = []
    for col in ts.columns:
        parts = [f"  {col.name} ({col.sql_type})"]

        if col.is_pk:
            parts.append("[PK]")
        elif col.name in ts.measure_columns:
            parts.append("[MEASURE]")
        elif col.name in ts.grain_columns:
            parts.append("[GRAIN]")
        elif col.name in ts.temporal_columns:
            parts.append("[TEMPORAL]")
        else:
            parts.append("[ATTR]")

        if col.name in ts.col_stats:
            st = ts.col_stats[col.name]
            stat_parts = []
            if "distinct" in st:
                stat_parts.append(f"{st['distinct']} unique")
            if "min" in st and "max" in st:
                stat_parts.append(f"[{st['min']}..{st['max']}]")
            if stat_parts:
                parts.append(f"({', '.join(stat_parts)})")

        if col.name in ts.sample_values:
            vals = ts.sample_values[col.name][:5]
            parts.append(f"e.g. {vals}")

        if col.description:
            parts.append(f"-- {col.description}")

        for fk in ts.foreign_keys:
            if fk.column == col.name:
                parts.append(f"→ {fk.ref_table}.{fk.ref_column}")

        col_lower = col.name.lower()
        if _term_matches(col_lower, q_terms):
            relevant.append(col.name)
            parts.append("⮕")

        lines.append(" ".join(parts))

    if relevant:
        lines.append(f"\n⮕ RELEVANT TO QUESTION: {', '.join(relevant)}")

    # Detect collisions
    col_names = {c.name for c in ts.columns}
    collisions: list[tuple[str, str, str]] = []
    collision_lines = []
    for other_t in kg.tables:
        if other_t.name == table:
            continue
        for c in other_t.columns:
            if c.name in col_names and c.name in relevant:
                collisions.append((c.name, table, other_t.name))
                collision_lines.append(
                    f'  ⚠ "{c.name}" also in {other_t.name} '
                    f"(role={other_t.role}, {other_t.row_count} rows)"
                )
    if collision_lines:
        lines.append("\nCOLUMN COLLISIONS (same name in multiple tables):")
        lines.extend(collision_lines)

    # Relevant domain knowledge
    if knowledge_text:
        table_lower = table.lower()
        k_lines = []
        for line in knowledge_text.split("\n"):
            if table_lower in line.lower():
                k_lines.append(f"  {line.strip()}")
        if k_lines:
            lines.append("\nDOMAIN KNOWLEDGE:")
            lines.extend(k_lines[:10])

    return "\n".join(lines), collisions


def tool_topology(kg: KnowledgeGraph, tables: list[str], db_path: Path) -> str:
    """Join paths, relationship cardinality, dimensional model."""
    if not tables:
        tables = [t.name for t in kg.tables]

    lines = []

    lines.append("TABLE ROLES:")
    for t in kg.tables:
        if t.name in tables:
            lines.append(f"  {t.name}: {t.role} ({t.row_count} rows)")

    fks = kg.all_foreign_keys()
    relevant_fks = []
    table_set = {t.lower() for t in tables}
    for src, fk in fks:
        if src.lower() in table_set or fk.ref_table.lower() in table_set:
            relevant_fks.append((src, fk))

    if relevant_fks:
        lines.append("\nRELATIONSHIPS:")
        for src, fk in relevant_fks:
            src_table = kg.get_table(src)
            ref_table = kg.get_table(fk.ref_table)
            cardinality = _infer_cardinality(
                src, fk.column, fk.ref_table, fk.ref_column, src_table, ref_table, db_path
            )
            lines.append(f"  {src}.{fk.column} → {fk.ref_table}.{fk.ref_column} [{cardinality}]")
            lines.append(f'    JOIN: "{src}"."{fk.column}" = "{fk.ref_table}"."{fk.ref_column}"')

    if kg.dim_model:
        lines.append("\nDIMENSIONAL MODEL:")
        if kg.dim_model.get("fact"):
            lines.append(f"  Fact: {kg.dim_model['fact']}")
        for dim in kg.dim_model.get("dimensions", []):
            lines.append(f"  Dim: {dim.get('table', '')} via {dim.get('join_col', '')}")

    return "\n".join(lines)


def tool_ontology(kg: KnowledgeGraph, column: str) -> str:
    """Get semantic metadata for a column: type, role, stats, valid values, relationships.

    column format: "table.column" or just "column" (searches all tables).
    """
    parts = column.split(".", 1)
    matches: list[tuple[str, str]] = []  # (table_name, col_name)

    if len(parts) == 2:
        matches.append((parts[0], parts[1]))
    else:
        # Search all tables
        for t in kg.tables:
            for c in t.columns:
                if c.name.lower() == column.lower():
                    matches.append((t.name, c.name))

    if not matches:
        return f"Column '{column}' not found in any table."

    lines = []
    for table_name, col_name in matches:
        ts = kg.get_table(table_name)
        if not ts:
            continue
        col = next((c for c in ts.columns if c.name == col_name), None)
        if not col:
            continue

        lines.append(f'COLUMN: "{table_name}"."{col_name}"')
        lines.append(f"  Type: {col.sql_type}")
        lines.append(f"  Role: {'PK' if col.is_pk else 'nullable' if col.is_nullable else 'NOT NULL'}")

        # Stats
        stats = ts.col_stats.get(col_name, {})
        if stats:
            if "n_unique" in stats:
                lines.append(f"  Distinct values: {stats['n_unique']}")
            if "min" in stats:
                lines.append(f"  Range: [{stats.get('min')}, {stats.get('max')}]")
            if "pct_null" in stats:
                lines.append(f"  NULL %: {stats['pct_null']:.1%}")

        # Sample values
        samples = ts.sample_values.get(col_name, [])
        if samples:
            lines.append(f"  Sample values: {samples[:10]}")

        # Structural role
        if col_name in ts.grain_columns:
            lines.append("  Semantic: GRAIN (defines row uniqueness)")
        elif col_name in ts.measure_columns:
            agg = ts.measure_agg_level.get(col_name, "raw")
            lines.append(f"  Semantic: MEASURE ({agg})")
        elif col_name in ts.temporal_columns:
            lines.append("  Semantic: TEMPORAL")

        # FK relationships
        for fk in ts.foreign_keys:
            if fk.column == col_name:
                lines.append(f"  FK → \"{fk.ref_table}\".\"{fk.ref_column}\"")

        # Value nodes from property graph
        if kg.graph:
            col_id = f"{table_name}.{col_name}"
            val_nodes = kg.graph.get_column_values(col_id)
            if val_nodes:
                top_vals = sorted(val_nodes, key=lambda v: -v.count)[:10]
                val_strs = [f"'{v.value}' ({v.count})" for v in top_vals]
                lines.append(f"  Top values: {', '.join(val_strs)}")

        lines.append("")

    return "\n".join(lines) if lines else f"No metadata found for '{column}'."


def tool_find_value(kg: KnowledgeGraph, value: str, db_path: Path) -> str:
    """Search the KG for which tables/columns contain a given value.

    Also shows related tables reachable via JOINs from matching columns.
    Falls back to a live DB query if the KG value index has no hit.
    """
    results: list[str] = []

    # 1. Search the property graph value index
    if kg.graph:
        matches = kg.graph.find_value(value)
        for table, column, count in matches:
            results.append(f'  "{table}"."{column}" contains \'{value}\' ({count} rows)')
            # Show neighbors (related columns via FK)
            col_id = f"{table}.{column}"
            for neighbor_id, weight, edge_type in kg.graph.neighbors(col_id):
                parts = neighbor_id.split(".", 1)
                if len(parts) == 2:
                    results.append(
                        f"    → {edge_type} to \"{parts[0]}\".\"{parts[1]}\" "
                        f"(weight={weight:.2f})"
                    )

    # 2. Fallback: query the DB directly if no KG hit
    if not results:
        try:
            conn = sqlite3.connect(str(db_path), timeout=10)
            for t in kg.tables:
                for c in t.columns:
                    if c.sql_type.upper() not in ("TEXT", "VARCHAR", "NVARCHAR"):
                        continue
                    cur = conn.execute(
                        f'SELECT COUNT(*) FROM "{t.name}" '
                        f'WHERE "{c.name}" = ? COLLATE NOCASE',
                        (value,),
                    )
                    cnt = cur.fetchone()[0]
                    if cnt > 0:
                        results.append(
                            f'  "{t.name}"."{c.name}" = \'{value}\' ({cnt} rows)'
                        )
            conn.close()
        except Exception:
            pass

    if not results:
        return f"Value '{value}' not found in any table."
    return f"VALUE SEARCH: '{value}'\n" + "\n".join(results)


def tool_knowledge(knowledge_text: str, query: str) -> str:
    """Search domain knowledge by relevance to query terms."""
    if not knowledge_text:
        return "No domain knowledge available for this dataset."

    query_terms = [t.lower() for t in query.split() if len(t) > 2]
    paragraphs = knowledge_text.split("\n\n")
    scored: list[tuple[int, str]] = []

    for para in paragraphs:
        para_lower = para.lower()
        score = sum(1 for term in query_terms if term in para_lower)
        if score > 0:
            scored.append((score, para.strip()))

    scored.sort(key=lambda x: -x[0])
    if not scored:
        return f"No knowledge found matching '{query}'. Full knowledge:\n{knowledge_text[:2000]}"

    results = [para for _, para in scored[:5]]
    return "\n\n".join(results)


def tool_run_sql(db_path: Path, sql: str) -> tuple[str, dict[str, Any] | None]:
    """Execute SQL and return (formatted_observation, raw_result).

    raw_result is {"columns": [...], "rows": [...]} or None on error.
    """
    result = _execute_sql(db_path, sql)
    if result is None:
        return "ERROR: SQL execution failed.", None
    if isinstance(result, str):
        return f"ERROR: {result}", None
    if not result.get("rows"):
        return "Result: 0 rows returned.", result

    cols = result["columns"]
    rows = result["rows"]
    total = len(rows)
    show = rows[:15]
    lines = [f"columns: {cols}", f"rows ({total} total, showing first {len(show)}):"]
    for row in show:
        lines.append(f"  {row}")
    return "\n".join(lines), result


def _execute_sql(db_path: Path, sql: str) -> dict[str, Any] | str | None:
    """Execute read-only SQL. Returns result dict or error string."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = [list(r) for r in cursor.fetchall()]
        conn.close()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        return str(e)


def _infer_cardinality(
    src_table: str,
    src_col: str,
    ref_table: str,
    ref_col: str,
    src_ts: Any,
    ref_ts: Any,
    db_path: Path,
) -> str:
    """Infer relationship cardinality from data."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        cur = conn.execute(f'SELECT COUNT(DISTINCT "{src_col}") FROM "{src_table}"')
        src_distinct = cur.fetchone()[0]
        cur = conn.execute(f'SELECT COUNT(DISTINCT "{ref_col}") FROM "{ref_table}"')
        ref_distinct = cur.fetchone()[0]
        conn.close()

        src_rows = src_ts.row_count if src_ts else 0
        ref_rows = ref_ts.row_count if ref_ts else 0

        if src_rows > src_distinct and ref_distinct == ref_rows:
            return f"many-to-one ({src_table} N → 1 {ref_table})"
        elif src_distinct == src_rows and ref_distinct == ref_rows:
            return f"one-to-one ({src_table} 1 → 1 {ref_table})"
        elif src_rows > src_distinct and ref_rows > ref_distinct:
            return "many-to-many (via junction)"
        else:
            return f"one-to-many ({src_table} 1 → N {ref_table})"
    except Exception:
        return "unknown"
