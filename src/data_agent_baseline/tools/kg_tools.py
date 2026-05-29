"""KG Tools: tool functions for the DataAgent loop.

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

    # Surface relevant aliases from the question
    if kg.alias_registry:
        alias_hits = []
        for canonical, info in kg.alias_registry.items():
            aliases = info.get("aliases", [])
            all_terms = [canonical.lower()] + aliases
            if any(t in q_terms or any(t in a for a in all_terms) for t in q_terms):
                alias_hits.append(
                    f"  '{canonical}' ({info['table']}.{info['column']})"
                    f" aliases: {aliases[:3]}"
                )
        if alias_hits:
            lines.append("\nALIAS HINTS (use resolve tool for full mapping):")
            lines.extend(alias_hits[:5])

    # Surface relevant topology expansions
    if kg.business_topology:
        topo_hits = []
        for h in kg.business_topology:
            for parent_val in h.get("membership", {}):
                if any(t in parent_val.lower() for t in q_terms if len(t) > 2):
                    children = h["membership"][parent_val]
                    topo_hits.append(
                        f"  '{parent_val}' → {len(children)} "
                        f"{h['child_column']}s in {h['child_table']}"
                    )
        if topo_hits:
            lines.append("\nTOPOLOGY (use resolve tool to expand):")
            lines.extend(topo_hits[:5])

    # Show available documents with preview
    if kg.doc_names:
        lines.append(
            f"\nDOCUMENTS (execute_python to parse/extract): "
            f"{', '.join(kg.doc_names)}"
        )
        if kg.tables:
            lines.append(
                "  NOTE: Documents use IDs from the SQL tables above to link "
                "records.\n"
                "  WORKFLOW: run_sql to get IDs → execute_python to search "
                "docs for those IDs."
            )
        lines.append(
            "  TIP: docs variable has full text. Explore with:\n"
            "    print(docs[:3000])  — see structure\n"
            "    print([l for l in docs.split('\\n') if 'ID_HERE' in l][:10])  "
            "— find linked records"
        )

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
        msg = f"Table '{table}' not found. Available: {available}"
        # If a doc with similar name exists, hint to use execute_python
        if kg.doc_names:
            table_lower = table.lower()
            for doc_name in kg.doc_names:
                if table_lower in doc_name.lower() or doc_name.lower() in table_lower:
                    msg += (
                        f"\n\nNOTE: Document '{doc_name}' exists and may contain "
                        f"this data as prose. Use execute_python to parse it:\n"
                        f"  lines = [l for l in docs.split('\\n') if l.strip()]\n"
                        f"  print(lines[:30])  # see structure first"
                    )
                    break
        return msg, []

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
            like_pattern = f"%{value}%"
            for t in kg.tables:
                for c in t.columns:
                    if c.sql_type.upper() not in ("TEXT", "VARCHAR", "NVARCHAR"):
                        continue
                    cur = conn.execute(
                        f'SELECT COUNT(*) FROM "{t.name}" '
                        f'WHERE "{c.name}" LIKE ? COLLATE NOCASE',
                        (like_pattern,),
                    )
                    cnt = cur.fetchone()[0]
                    if cnt > 0:
                        results.append(
                            f'  "{t.name}"."{c.name}" contains \'{value}\' ({cnt} rows)'
                        )
            conn.close()
        except Exception:
            pass

    if not results:
        msg = f"Value '{value}' not found in any table."
        if kg.doc_names:
            msg += (
                f"\nDocuments available: {kg.doc_names}. "
                f"The value may exist in document text. "
                f"Use execute_python to search:\n"
                f"  print([l for l in docs.split('\\n') if '{value}' in l][:10])"
            )
        return msg
    return f"VALUE SEARCH: '{value}'\n" + "\n".join(results)


def tool_resolve(kg: KnowledgeGraph, term: str, db_path: Path) -> str:
    """Resolve a user term to canonical DB value(s) using alias registry and topology.

    Handles:
    1. Alias resolution: "injection molding machine 1" → "IM-M1"
    2. Topology expansion: "Line 1" → [machine IDs belonging to Line 1]
    """
    term_lower = term.lower().strip()
    results: list[str] = []

    # 1. Check alias registry
    if kg.alias_registry:
        for canonical, info in kg.alias_registry.items():
            aliases = info.get("aliases", [])
            # Match: term matches an alias, or canonical matches term
            if (term_lower in aliases
                    or term_lower == canonical.lower()
                    or any(term_lower in a for a in aliases)
                    or any(a in term_lower for a in aliases if len(a) > 3)):
                results.append(
                    f"ALIAS MATCH: '{term}' → canonical value '{canonical}' "
                    f"in {info['table']}.{info['column']}"
                )
                results.append(f"  Known aliases: {aliases}")
                results.append(
                    f"  USE: WHERE \"{info['column']}\" = '{canonical}'"
                )

    # 2. Check business topology (expand parent → children)
    if kg.business_topology:
        for hierarchy in kg.business_topology:
            membership = hierarchy.get("membership", {})
            parent_col = hierarchy["parent_column"]
            child_table = hierarchy["child_table"]
            child_col = hierarchy["child_column"]
            description = hierarchy.get("description", "")

            for parent_val, children in membership.items():
                if (term_lower == parent_val.lower()
                        or term_lower in parent_val.lower()
                        or parent_val.lower() in term_lower):
                    results.append(
                        f"TOPOLOGY EXPANSION: '{term}' → '{parent_val}' "
                        f"({description})"
                    )
                    results.append(
                        f"  Contains {len(children)} children in "
                        f"{child_table}.{child_col}: {children[:15]}"
                    )
                    if len(children) <= 20:
                        child_list = ", ".join(f"'{c}'" for c in children)
                        results.append(
                            f"  USE: WHERE \"{child_col}\" IN ({child_list})"
                        )
                    else:
                        results.append(
                            f"  USE: JOIN with parent table on {parent_col}"
                        )

    # 3. Fallback: fuzzy search in sample values
    if not results:
        for t in kg.tables:
            for col_name, samples in t.sample_values.items():
                for val in samples:
                    val_str = str(val).lower()
                    if (term_lower in val_str or val_str in term_lower
                            or _fuzzy_match(term_lower, val_str)):
                        results.append(
                            f"FUZZY MATCH: '{term}' ≈ '{val}' in "
                            f"{t.name}.{col_name}"
                        )
                        results.append(
                            f"  USE: WHERE \"{col_name}\" = '{val}'"
                        )

    if not results:
        return (
            f"No resolution found for '{term}'. Try:\n"
            f"  - find_value to search DB content directly\n"
            f"  - schema on the relevant table to see actual values"
        )

    return f"RESOLVE: '{term}'\n" + "\n".join(results)


def _fuzzy_match(a: str, b: str) -> bool:
    """Simple fuzzy matching: shared token overlap > 50%."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return False
    overlap = tokens_a & tokens_b
    shorter = min(len(tokens_a), len(tokens_b))
    return len(overlap) / shorter > 0.5 if shorter > 0 else False


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
    output = "\n\n".join(results)

    # If many paragraphs match, hint that execute_python may be better
    if len(scored) > 20:
        output += (
            f"\n\n[NOTE: {len(scored)} paragraphs match this query. "
            f"If you need to filter/aggregate across many records in the "
            f"documents, use execute_python with regex to parse the full "
            f"text — it's available as `docs` variable.]"
        )

    return output


def tool_distribution(db_path: Path, table: str, column: str) -> str:
    """Analyze data distribution for a numeric column.

    Returns: min, max, avg, median, stddev, percentiles (10th, 25th, 75th, 90th),
    and suggested normal/abnormal thresholds based on the actual data.
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.execute("PRAGMA busy_timeout = 10000")

        # Basic stats
        cursor = conn.execute(
            f'SELECT COUNT("{column}"), MIN("{column}"), MAX("{column}"), '
            f'AVG("{column}") FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL AND "{column}" != \'\''
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            conn.close()
            return f"No non-null values found for {table}.{column}"

        count, val_min, val_max, val_avg = row

        # Check if numeric
        try:
            float(val_min)
            float(val_max)
        except (ValueError, TypeError):
            conn.close()
            return (
                f"{table}.{column}: not numeric (min='{val_min}', max='{val_max}'). "
                f"Use run_sql with SELECT DISTINCT to see categorical values."
            )

        # Get all values sorted for percentile calculation
        cursor = conn.execute(
            f'SELECT CAST("{column}" AS REAL) FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL AND "{column}" != \'\' '
            f'ORDER BY CAST("{column}" AS REAL)'
        )
        values = [r[0] for r in cursor.fetchall() if r[0] is not None]
        conn.close()

        if not values:
            return f"No numeric values for {table}.{column}"

        n = len(values)

        def percentile(pct: float) -> float:
            idx = int(n * pct / 100)
            idx = max(0, min(idx, n - 1))
            return values[idx]

        p10 = percentile(10)
        p25 = percentile(25)
        p50 = percentile(50)
        p75 = percentile(75)
        p90 = percentile(90)
        iqr = p75 - p25

        lines = [
            f"DISTRIBUTION: {table}.{column} ({n} non-null values)",
            f"  Range: [{val_min}, {val_max}]",
            f"  Mean: {val_avg:.2f}, Median: {p50:.2f}",
            f"  Percentiles: P10={p10:.2f}, P25={p25:.2f}, "
            f"P75={p75:.2f}, P90={p90:.2f}",
            f"  IQR: {iqr:.2f} (P75 - P25)",
            "",
            "STATISTICAL PERCENTILES (NOT medical reference ranges):",
            f"  P10-P90 range: [{p10:.2f}, {p90:.2f}]",
            f"  P25-P75 range: [{p25:.2f}, {p75:.2f}]",
            "",
            "NOTE: For medical/clinical data, use standard reference ranges "
            "(e.g., WBC 3.5-10.5, PLT 100-400, FG 150-400 mg/dL) instead of "
            "these percentiles. Percentiles only describe this dataset's "
            "distribution, NOT what is medically normal/abnormal.",
        ]

        # Count how many fall outside P10-P90
        n_abnormal = sum(1 for v in values if v < p10 or v > p90)
        lines.append(
            f"  Values outside P10-P90: {n_abnormal}/{n} ({n_abnormal*100/n:.1f}%)"
        )

        # Histogram (5 buckets)
        bucket_size = (float(val_max) - float(val_min)) / 5
        if bucket_size > 0:
            lines.append("")
            lines.append("HISTOGRAM (5 buckets):")
            for i in range(5):
                lo = float(val_min) + i * bucket_size
                hi = lo + bucket_size
                cnt = sum(1 for v in values if lo <= v < hi) if i < 4 else \
                    sum(1 for v in values if lo <= v <= float(val_max))
                bar = "#" * min(30, int(cnt * 30 / n)) if n > 0 else ""
                lines.append(f"  [{lo:.1f}-{hi:.1f}]: {cnt} {bar}")

        return "\n".join(lines)

    except Exception as e:
        return f"ERROR analyzing {table}.{column}: {str(e)[:100]}"


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


def tool_execute_python(code: str, knowledge_text: str, db_path: Path) -> str:
    """Execute Python code with access to doc text and DB path.

    The code runs in a restricted namespace with:
      - `docs`: the full knowledge/doc text (string)
      - `db_path`: path to the SQLite database (string)
      - `re`, `json`, `sqlite3`, `collections` pre-imported
      - `print()` output is captured and returned

    Returns stdout output (max 3000 chars) or error message.
    """
    import collections
    import io
    import json as json_mod
    import re as re_mod
    import contextlib

    namespace = {
        "docs": knowledge_text,
        "db_path": str(db_path),
        "re": re_mod,
        "json": json_mod,
        "sqlite3": sqlite3,
        "collections": collections,
        "Path": Path,
    }

    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(code, "<agent_python>", "exec"), namespace)  # noqa: S102
    except Exception as e:
        error_output = stdout_capture.getvalue()
        error_msg = f"ERROR: {type(e).__name__}: {e}"
        if error_output:
            return f"{error_output}\n{error_msg}"
        return error_msg

    output = stdout_capture.getvalue()
    if not output:
        if "result" in namespace:
            output = str(namespace["result"])
    if len(output) > 3000:
        output = output[:3000] + "\n... (truncated)"
    return output or "(no output)"


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
