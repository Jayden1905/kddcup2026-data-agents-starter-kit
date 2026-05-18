"""Grounding format mixin for QuestionDrivenAgent."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.pipeline.kg_builder import KnowledgeGraph
from data_agent_baseline.pipeline.kg_path_planner import QueryNode, QueryPath, QueryPlan


def _decide_query_complexity(
    comp_type: str, path: QueryPath, kg: KnowledgeGraph | None,
) -> str:
    """Layer 2 (Dim Model) as ARCHITECT: decide query shape.

    Returns: "single" | "cte"
    CTE when:
      - 3+ tables AND computation is derived (ratio, percentage, growth)
      - Filter on dimension but output from fact (resolve filter first)
      - Pre-aggregated measure needs window function
    """
    if not kg or not kg.dim_model:
        return "single"

    tables_count = len(path.tables_in_path) if path.tables_in_path else 1
    is_derived = comp_type in ("ratio", "percentage")

    # 3+ tables with derived computation → CTE
    if tables_count >= 3 and is_derived:
        return "cte"

    # Filter on dimension, output from fact → CTE to resolve dimension first
    if path.filter_nodes and path.output_nodes and kg.dim_model.get("fact_table"):
        fact_table = kg.dim_model["fact_table"]
        dim_tables = {d["table"] for d in kg.dim_model.get("dimensions", [])}
        output_on_fact = any(n.table == fact_table for n in path.output_nodes)
        filter_on_dim = any(n.table in dim_tables for n in path.filter_nodes)
        if output_on_fact and filter_on_dim and tables_count >= 3:
            return "cte"

    # Pre-aggregated measure needing derived computation
    if path.output_nodes and comp_type in ("ratio", "percentage", "avg"):
        for n in path.output_nodes:
            ts = kg.get_table(n.table)
            if ts and ts.measure_agg_level.get(n.column) == "pre_aggregated":
                if comp_type != "avg":
                    return "cte"

    return "single"


def _build_cte_plan(
    comp_type: str, path: QueryPath, kg: KnowledgeGraph,
) -> str | None:
    """When query_shape is 'cte', produce a CTE skeleton for the LLM.

    Uses dim_model to decompose: resolve dimension filters first, then aggregate fact.
    """
    if not kg.dim_model or not path.output_nodes:
        return None

    fact_table = kg.dim_model.get("fact_table", "")
    dimensions = kg.dim_model.get("dimensions", [])
    dim_tables = {d["table"]: d for d in dimensions}

    # Find filters on dimension tables
    dim_filter_nodes = [n for n in (path.filter_nodes or []) if n.table in dim_tables]
    fact_filter_nodes = [n for n in (path.filter_nodes or []) if n.table == fact_table]

    if not dim_filter_nodes:
        return None

    cte_parts: list[str] = []

    # CTE 1: Resolve dimension filters to IDs
    for dim_table, nodes in _group_by_table(dim_filter_nodes).items():
        dim_info = dim_tables.get(dim_table)
        if not dim_info:
            continue
        where_parts = []
        for n in nodes:
            if n.operator == "IS NOT NULL":
                where_parts.append(f'"{n.column}" IS NOT NULL')
            else:
                where_parts.append(f'"{n.column}" {n.operator} \'{n.value}\'')
        where_sql = " AND ".join(where_parts)
        ref_col = dim_info["ref_col"]
        cte_parts.append(
            f'  filtered_{dim_table} AS (\n'
            f'    SELECT "{ref_col}" FROM "{dim_table}" WHERE {where_sql}\n'
            f'  )'
        )

    # Main query: aggregate fact using resolved IDs
    out_node = path.output_nodes[0]
    out_col = out_node.column
    out_table = out_node.table

    # Build main WHERE using fact filters + dimension CTE references
    main_wheres: list[str] = []
    for n in fact_filter_nodes:
        if n.operator == "IS NOT NULL":
            main_wheres.append(f'"{n.column}" IS NOT NULL')
        else:
            main_wheres.append(f'"{n.column}" {n.operator} \'{n.value}\'')

    for dim_table in _group_by_table(dim_filter_nodes):
        dim_info = dim_tables.get(dim_table)
        if dim_info:
            join_col = dim_info["join_col"]
            main_wheres.append(
                f'"{join_col}" IN (SELECT "{dim_info["ref_col"]}" FROM filtered_{dim_table})'
            )

    where_clause = f" WHERE {' AND '.join(main_wheres)}" if main_wheres else ""

    # Build SELECT expression based on comp_type
    if comp_type == "avg":
        select_expr = f'AVG("{out_col}")'
    elif comp_type == "sum":
        select_expr = f'SUM("{out_col}")'
    elif comp_type == "count":
        select_expr = f'COUNT(DISTINCT "{out_col}")'
    elif comp_type in ("ratio", "percentage"):
        select_expr = f'-- ratio/percentage of "{out_col}"'
    else:
        select_expr = f'"{out_col}"'

    main_sql = f'  SELECT {select_expr} FROM "{out_table}"{where_clause}'

    cte_plan = "WITH\n" + ",\n".join(cte_parts) + "\n" + main_sql
    return cte_plan


def _group_by_table(nodes: list[QueryNode]) -> dict[str, list[QueryNode]]:
    groups: dict[str, list[QueryNode]] = {}
    for n in nodes:
        groups.setdefault(n.table, []).append(n)
    return groups


def _build_data_notes(path: QueryPath, kg: KnowledgeGraph | None) -> list[str]:
    """Layer 3 (Statistics) as ADVISOR: emit data quality hints.

    Advisory only — never overrides SQL FORMULA or other structural decisions.
    """
    if not kg:
        return []
    notes: list[str] = []

    for n in (path.output_nodes or []):
        ts = kg.get_table(n.table)
        if not ts or not hasattr(ts, 'col_stats'):
            continue
        stats = ts.col_stats.get(n.column, {})
        if not stats:
            continue

        # Range note for numeric output columns
        if "min" in stats and "max" in stats:
            notes.append(f'"{n.column}" range: [{stats["min"]}, {stats["max"]}]')

        # High NULL warning
        null_ratio = stats.get("null_ratio", 0)
        if null_ratio and null_ratio > 0.3:
            notes.append(f'"{n.column}" is {null_ratio*100:.0f}% NULL — filtered rows may be fewer than expected')

        # Pre-aggregated warning
        if ts.measure_agg_level.get(n.column) == "pre_aggregated":
            notes.append(f'"{n.column}" is PRE-AGGREGATED (each row already contains a sum/avg for that period)')

    for n in (path.filter_nodes or []):
        if n.column.startswith("_expr:"):
            continue
        ts = kg.get_table(n.table)
        if not ts or not hasattr(ts, 'col_stats'):
            continue
        stats = ts.col_stats.get(n.column, {})
        # Cardinality note: if cardinality_ratio ≈ 1.0, DISTINCT is safe to skip
        cr = stats.get("cardinality_ratio", 0)
        if cr and cr > 0.95:
            notes.append(f'"{n.table}"."{n.column}" has near-unique values — DISTINCT is redundant here')

    return notes


def _route_sql_pattern(
    comp_type: str, out_node: QueryNode | None, path: QueryPath,
    kg: KnowledgeGraph | None,
) -> str | None:
    """Layer 4: Route computation_type + schema topology to an explicit SQL pattern.

    Absorbs Layer 1 (Normalization) constraints internally.
    Returns a short SQL skeleton string or None if no special pattern applies.
    """
    if not out_node or not kg:
        return None
    out_schema = kg.get_table(out_node.table)
    if not out_schema:
        return None

    out_role = out_schema.role
    out_col = out_node.column
    is_measure = out_col in out_schema.measure_columns
    is_pre_agg = out_schema.measure_agg_level.get(out_col) == "pre_aggregated"
    has_join = bool(path.edges)

    # Find entity table (the other side of the join)
    entity_table = ""
    entity_pk = ""
    if has_join:
        for edge in path.edges:
            if edge.src_table == out_node.table:
                entity_table = edge.dst_table
            elif edge.dst_table == out_node.table:
                entity_table = edge.src_table
        if entity_table:
            ets = kg.get_table(entity_table)
            if ets and ets.columns:
                entity_pk = ets.columns[0].name

    if comp_type == "avg":
        if out_role == "bridge" or (not is_measure and has_join and entity_table):
            if entity_pk:
                return (
                    f'CAST(COUNT("{out_node.table}"."{out_col}") AS REAL) / '
                    f'COUNT(DISTINCT "{entity_table}"."{entity_pk}")'
                )
        if is_pre_agg:
            return f'AVG("{out_node.table}"."{out_col}")'
        if is_measure:
            return f'AVG("{out_node.table}"."{out_col}")'

    elif comp_type == "sum":
        # Layer 1 constraint: pre-aggregated measures should not be blindly SUMmed
        if is_pre_agg and has_join:
            return (
                f'-- WARNING: "{out_col}" is pre-aggregated per period. '
                f'SUM will double-count if JOINing to detail rows.\n'
                f'SUM("{out_node.table}"."{out_col}")'
            )

    elif comp_type == "count" and has_join:
        if out_role == "dimension" and entity_table:
            return f'COUNT(DISTINCT "{out_node.table}"."{out_col}")'
        if entity_pk and out_node.table != entity_table:
            return f'COUNT("{out_node.table}"."{out_col}")'

    return None


class GroundingFormatMixin:
    """Format KG plan as grounding context for SQL LLM."""

    def _format_kg_plan_as_grounding(
        self,
        path: QueryPath,
        plan: QueryPlan | None,
        goal: dict[str, Any],
        sql: str,
        kg: KnowledgeGraph | None = None,
        db_path: Path | None = None,
        user_intent: str = "",
    ) -> str:
        """Format the KG path plan as grounding context for the SQL LLM."""
        parts: list[str] = []

        parts.append(f"GOAL: {goal.get('what_user_wants', '')}")
        comp_type = plan.computation_type if plan else goal.get("computation_type", "simple_lookup")
        if comp_type == "grouped_list":
            parts.append(
                "RESULT TYPE: grouped_list — answer is a list/breakdown across categories (multiple rows)"
            )
        elif comp_type == "count_distinct":
            parts.append(
                "RESULT TYPE: count_distinct — use SELECT DISTINCT to return unique values only"
            )
        else:
            parts.append(f"RESULT TYPE: {comp_type}")

        # Layer 2 (ARCHITECT): decide query shape — single SQL or CTE
        query_shape = _decide_query_complexity(comp_type, path, kg)

        # For "derived" computation type, give the model freedom — no pattern router
        derived_logic = goal.get("derived_logic", "")
        if comp_type == "derived" and derived_logic:
            parts.append(f"COMPUTATION: {derived_logic}")

        # Layer 4 (GENERATOR): produce SQL expression or CTE plan
        # Skip for derived — let the model figure it out from evidence
        if path.output_nodes and kg and comp_type != "derived":
            if query_shape == "cte":
                _cte_plan = _build_cte_plan(comp_type, path, kg)
                if _cte_plan:
                    parts.append(f"QUERY STRUCTURE (CTE decomposition — resolve dimensions first, then aggregate):\n{_cte_plan}")
                else:
                    query_shape = "single"

            if query_shape == "single":
                _pattern = _route_sql_pattern(comp_type, path.output_nodes[0], path, kg)
                if _pattern:
                    parts.append(f"SQL EXPRESSION (derived from schema topology):\n  {_pattern}")
                elif comp_type not in ("simple_lookup", "grouped_list") and path.edges:
                    _out = path.output_nodes[0]
                    _out_ts = kg.get_table(_out.table)
                    _reasoning_lines = []
                    if _out_ts:
                        _reasoning_lines.append(f'Output table "{_out.table}" is a {_out_ts.role or "table"} ({_out_ts.row_count} rows)')
                        if _out_ts.measure_columns:
                            _reasoning_lines.append(f'  Measures: {_out_ts.measure_columns}')
                        if _out_ts.grain_columns:
                            _reasoning_lines.append(f'  Grain (one row per): {_out_ts.grain_columns}')
                        agg_hint = _out_ts.measure_agg_level.get(_out.column, "")
                        if agg_hint == "pre_aggregated":
                            _reasoning_lines.append(f'  "{_out.column}" is pre-aggregated (row already holds a period total)')
                    for edge in path.edges:
                        _other = edge.dst_table if edge.src_table == _out.table else edge.src_table
                        _other_ts = kg.get_table(_other)
                        if _other_ts:
                            _reasoning_lines.append(f'Joined to "{_other}" ({_other_ts.role or "table"}, {_other_ts.row_count} rows)')
                    if _reasoning_lines:
                        parts.append("SCHEMA CONTEXT:\n" + "\n".join(f"  {l}" for l in _reasoning_lines))

        # Tables: full schema for active tables (have output/filter nodes), compact for pass-through
        if kg and path.tables_in_path:
            active_tables = set()
            for n in (path.output_nodes or []):
                active_tables.add(n.table)
            for n in (path.filter_nodes or []):
                if not n.column.startswith("_expr:"):
                    active_tables.add(n.table)

            table_lines: list[str] = []
            for tname in path.tables_in_path:
                table_schema = kg.get_table(tname)
                if not table_schema:
                    continue
                role = table_schema.role or "table"
                row_count = table_schema.row_count or "?"
                grain = f" | grain: {table_schema.grain_columns}" if table_schema.grain_columns else ""
                if tname in active_tables:
                    cols = ", ".join(
                        f'"{c.name}" ({c.sql_type})'
                        for c in table_schema.columns
                    )
                    table_lines.append(f'  "{tname}" ({role}, {row_count} rows{grain}): [{cols}]')
                else:
                    # Pass-through table (only used for JOIN) — just show role and key columns
                    key_cols = [c.name for c in table_schema.columns if c.is_pk or c.name.lower().endswith("id")][:3]
                    table_lines.append(f'  "{tname}" ({role}, {row_count} rows) — join-through, keys: {key_cols}')
            if table_lines:
                parts.append("TABLES:\n" + "\n".join(table_lines))

        # Output columns
        if path.output_nodes:
            out_lines = []
            for n in path.output_nodes:
                extra = ""
                if n.agg_func:
                    extra = f" → {n.agg_func}"
                if kg:
                    ts = kg.get_table(n.table)
                    if ts and ts.measure_agg_level.get(n.column) == "pre_aggregated":
                        extra += " [pre-aggregated per row]"
                out_lines.append(f'  "{n.table}"."{n.column}"{extra}')
            parts.append("OUTPUT COLUMNS:\n" + "\n".join(out_lines))

        # Ontology: only surface for columns where it adds non-obvious info
        if kg and kg.ontology:
            # Collect filter values already resolved (no need to repeat vocab for these)
            resolved_filter_vals = set()
            for n in (path.filter_nodes or []):
                if not n.column.startswith("_expr:"):
                    resolved_filter_vals.add((f"{n.table}.{n.column}", str(n.value)))

            relevant_cols = set()
            for n in (path.output_nodes or []):
                relevant_cols.add(f"{n.table}.{n.column}")
            for n in (path.filter_nodes or []):
                if not n.column.startswith("_expr:"):
                    relevant_cols.add(f"{n.table}.{n.column}")

            ont_lines: list[str] = []
            for col_ref in sorted(relevant_cols):
                entry = kg.ontology.get(col_ref)
                if not entry:
                    continue
                parts_desc: list[str] = []
                # Skip trivial semantic types
                st = entry.get("semantic_type", "")
                if st and st not in ("identifier", "free_text"):
                    parts_desc.append(st)
                if entry.get("unit"):
                    parts_desc.append(f"unit={entry['unit']}")
                if entry.get("value_vocab"):
                    vocab = entry["value_vocab"]
                    # Skip vocab if filter already has correct value from it
                    already_used = any(
                        col_ref == cr and val in vocab
                        for cr, val in resolved_filter_vals
                    )
                    if not already_used:
                        if len(vocab) <= 8:
                            vocab_str = ", ".join(f"{k}={v}" for k, v in vocab.items())
                            parts_desc.append(f"values: {{{vocab_str}}}")
                        else:
                            vocab_str = ", ".join(f"{k}={v}" for k, v in list(vocab.items())[:6])
                            parts_desc.append(f"values: {{{vocab_str}, ...}}")
                if entry.get("derived_from"):
                    parts_desc.append(f"derived: {entry['derived_from']}")
                if entry.get("hierarchy"):
                    h = entry["hierarchy"]
                    parts_desc.append(f"hierarchy: {h.get('level', '')}")
                if parts_desc:
                    ont_lines.append(f'  {col_ref}: {" | ".join(parts_desc)}')
            if ont_lines:
                parts.append("SEMANTICS:\n" + "\n".join(ont_lines))

        # --- Ratio of independent counts detection ---
        # When computation_type is "ratio" and filters target different tables that have
        # NO direct FK relationship, the counts are independent — emit scalar subquery formula.
        # If tables ARE connected by FK, it's likely a subset/total pattern (CASE WHEN with JOIN).
        is_independent_ratio = False
        if comp_type in ("ratio", "percentage") and path.filter_nodes and path.edges:
            filter_tables = {n.table for n in path.filter_nodes if not n.column.startswith("_expr:")}
            if len(filter_tables) >= 2:
                # Check if filter tables are connected by a direct FK edge
                fk_connected = False
                if kg and kg.graph:
                    for edge in kg.graph.fk_edges:
                        src_col = kg.graph.columns.get(edge.src)
                        dst_col = kg.graph.columns.get(edge.dst)
                        if src_col and dst_col:
                            if {src_col.table_id, dst_col.table_id} <= filter_tables:
                                fk_connected = True
                                break
                if not fk_connected:
                    # Truly independent populations — no FK relationship
                    is_independent_ratio = True
                if is_independent_ratio:
                    # Group filters by table
                    table_filters: dict[str, list[QueryNode]] = {}
                    for node in path.filter_nodes:
                        if not node.column.startswith("_expr:"):
                            table_filters.setdefault(node.table, []).append(node)
                    # Build scalar subquery formula
                    subqueries: list[str] = []
                    for tbl, nodes in table_filters.items():
                        # Find the table's PK or Id column
                        pk_col = "Id"
                        if kg:
                            table_schema = kg.get_table(tbl)
                            if table_schema and table_schema.columns:
                                pk_col = table_schema.columns[0].name
                        where_parts = []
                        for n in nodes:
                            where_parts.append(f'"{n.column}" {n.operator} \'{n.value}\'')
                        where_clause = " AND ".join(where_parts)
                        subqueries.append(f'(SELECT COUNT(DISTINCT "{pk_col}") FROM "{tbl}" WHERE {where_clause})')

                    if len(subqueries) >= 2:
                        if comp_type == "percentage":
                            formula = f"CAST({subqueries[0]} AS REAL) * 100.0 / {subqueries[1]}"
                        else:
                            formula = f"CAST({subqueries[0]} AS REAL) / {subqueries[1]}"
                        parts.append(
                            f"INDEPENDENT RATIO: The two populations have NO foreign key connecting them — "
                            f"they cannot be JOINed. Each count is a separate scalar subquery.\n"
                            f"  SELECT {formula}"
                        )

        # --- Subquery filter makes JOIN redundant detection ---
        # When ALL output columns are from one table and a filter uses IN (SELECT ... FROM other_table),
        # the JOIN to that other table is redundant and harmful for AVG/SUM (duplicates rows).
        suppress_join = False
        if (
            comp_type in ("avg", "sum", "count")
            and path.edges
            and path.output_nodes
            and path.filter_nodes
            and not is_independent_ratio
        ):
            output_tables = {n.table for n in path.output_nodes}
            if len(output_tables) == 1:
                entity_table = next(iter(output_tables))
                # Check if any filter is a subquery that references other tables
                for fnode in path.filter_nodes:
                    val_upper = str(fnode.value).upper()
                    if fnode.operator.upper() == "IN" and "SELECT" in val_upper:
                        # Subquery references another table — the JOIN is redundant
                        subq_tables = set(re.findall(r'\bFROM\s+(\w+)', val_upper, re.IGNORECASE))
                        joined_tables = set()
                        for e in path.edges:
                            joined_tables.add(e.src_table)
                            joined_tables.add(e.dst_table)
                        joined_tables.discard(entity_table)
                        # If subquery already references the joined table, suppress the JOIN
                        if subq_tables & {t.upper() for t in joined_tables}:
                            suppress_join = True
                            break

        # Join structure
        if path.edges and not is_independent_ratio and not suppress_join:
            jp_lines = []
            for e in path.edges:
                overlap = f" ({e.weight:.0%} overlap)" if e.weight < 1.0 else ""
                jp_lines.append(f'  "{e.src_table}"."{e.src_column}" = "{e.dst_table}"."{e.dst_column}"{overlap}')
            low_overlap_edges = [e for e in path.edges if e.weight < 0.3]
            if low_overlap_edges:
                jp_lines.append(
                    f"  Note: {low_overlap_edges[0].weight:.0%} overlap means many rows won't match. "
                    f"A subquery (WHERE ... IN (SELECT ...)) may return more complete results."
                )
            parts.append("JOINS:\n" + "\n".join(jp_lines))
        elif suppress_join:
            parts.append(
                f"STRUCTURE: All output columns are from \"{entity_table}\". "
                f"The subquery filter already covers the other table — no JOIN needed."
            )

        # Filter values (for independent ratios, already embedded in the formula)
        # For percentage/ratio: detect single-table or multi-table (FK-connected) patterns
        filter_tables_set = {n.table for n in path.filter_nodes if not n.column.startswith("_expr:")} if path.filter_nodes else set()
        is_single_table_ratio = (
            comp_type in ("percentage", "ratio") and path.filter_nodes and not is_independent_ratio
            and len(filter_tables_set) == 1
            and path.output_nodes
            and path.filter_nodes[0].table == path.output_nodes[0].table
        )
        # Multi-table percentage: FK-connected tables, one filter is population (WHERE),
        # the other is subset (CASE WHEN). Fires when independent ratio was suppressed.
        # Suppress when any output column is a numeric measurement — value comparison, not row-counting.
        _output_is_numeric_measure = False
        if path.output_nodes and db_path:
            try:
                _conn2 = sqlite3.connect(str(db_path), timeout=5)
                for _onode in path.output_nodes:
                    _ocol = _onode.column.lower()
                    # ID/PK columns are for counting, not value comparison
                    if _ocol == "id" or _ocol == "_id" or _ocol.endswith("_id") or _ocol.endswith("id"):
                        continue
                    _col_info2 = _conn2.execute(f'PRAGMA table_info("{_onode.table}")').fetchall()
                    for _ci in _col_info2:
                        if _ci[1].lower() == _ocol:
                            if _ci[5]:  # pk flag
                                break
                            _ct = (_ci[2] or "").upper()
                            if _ct in ("REAL", "INTEGER", "NUMERIC", "FLOAT", "DOUBLE", "INT"):
                                _output_is_numeric_measure = True
                            break
                    if _output_is_numeric_measure:
                        break
                _conn2.close()
            except Exception:
                pass
        is_multi_table_pct = (
            comp_type in ("percentage", "ratio") and path.filter_nodes and not is_independent_ratio
            and not is_single_table_ratio
            and len(filter_tables_set) >= 2
            and path.edges
            and not _output_is_numeric_measure
        )

        if path.filter_nodes and not is_independent_ratio:
            if is_single_table_ratio:
                tbl = path.output_nodes[0].table
                pk_col = "id"
                if kg:
                    ts = kg.get_table(tbl)
                    if ts and ts.columns:
                        pk_col = ts.columns[0].name
                # Detect TEXT columns needing CAST for range comparisons
                _st_text_cols: set[str] = set()
                if db_path:
                    try:
                        _conn_st = sqlite3.connect(str(db_path), timeout=5)
                        for _ci_st in _conn_st.execute(f'PRAGMA table_info("{tbl}")').fetchall():
                            if (_ci_st[2] or "").upper() == "TEXT":
                                _st_text_cols.add(_ci_st[1])
                        _conn_st.close()
                    except Exception:
                        pass
                conds = []
                for node in path.filter_nodes:
                    if node.column.startswith("_expr:"):
                        continue
                    col_ref = f'"{node.column}"'
                    is_num = re.match(r'^-?\d+\.?\d*$', str(node.value))
                    if node.operator in (">=", "<=", ">", "<") and node.column in _st_text_cols and is_num:
                        conds.append(f'CAST("{node.column}" AS REAL) {node.operator} {node.value}')
                    elif node.operator.upper() == "LIKE":
                        conds.append(f'{col_ref} LIKE \'{node.value}\'')
                    else:
                        conds.append(f'{col_ref} {node.operator} \'{node.value}\'')
                cond_sql = " AND ".join(conds)
                if comp_type == "percentage":
                    parts.append(
                        f'SAME-TABLE RATIO: Population = all {tbl} rows. '
                        f'Subset = rows where {cond_sql}. '
                        f'The subset IS PART OF the population (same table), so use CASE WHEN:\n'
                        f'  SELECT CAST(COUNT(CASE WHEN {cond_sql} THEN 1 END) AS REAL) * 100 / COUNT("{pk_col}") FROM "{tbl}"'
                    )
                else:
                    parts.append(
                        f'SAME-TABLE RATIO: Population = all {tbl} rows. '
                        f'Subset = rows where {cond_sql}. '
                        f'The subset IS PART OF the population (same table), so use CASE WHEN:\n'
                        f'  SELECT CAST(COUNT(CASE WHEN {cond_sql} THEN 1 END) AS REAL) / COUNT("{pk_col}") FROM "{tbl}"'
                    )
            elif is_multi_table_pct:
                # Determine population (WHERE) vs subset (CASE WHEN) using user_intent.
                # "In X, what % is Y?" → X is population, Y is subset.
                entity_tbl = path.output_nodes[0].table if path.output_nodes else None
                pop_filters: list[QueryNode] = []
                subset_filters: list[QueryNode] = []

                # Extract population description from user_intent
                _pop_desc = ""
                if user_intent:
                    _pop_match = re.search(r'Population \(WHERE\):\s*(.+)', user_intent)
                    if _pop_match:
                        _pop_desc = _pop_match.group(1).strip().lower()

                for node in path.filter_nodes:
                    if node.column.startswith("_expr:"):
                        pop_filters.append(node)
                        continue
                    # Match column/value to population description flexibly
                    col_words = node.column.lower().replace("_", " ").split()
                    val_str = str(node.value).lower().rstrip("0").rstrip(".")
                    in_pop = False
                    if _pop_desc:
                        in_pop = (
                            any(w in _pop_desc for w in col_words if len(w) > 2)
                            or val_str in _pop_desc
                        )
                    if in_pop:
                        pop_filters.append(node)
                    elif _pop_desc:
                        subset_filters.append(node)
                    elif node.table == entity_tbl:
                        pop_filters.append(node)
                    else:
                        subset_filters.append(node)
                # If no subset filters identified, treat the smaller group as subset
                if not subset_filters:
                    subset_filters = pop_filters
                    pop_filters = []
                # Build the CASE WHEN pattern with JOIN
                # Base table = population table (WHERE filters), not necessarily entity table
                pop_tables = {n.table for n in pop_filters if not n.column.startswith("_expr:")}
                base_table = next(iter(pop_tables), None) or entity_tbl or path.tables_in_path[0]
                join_parts_list = []
                for e in path.edges:
                    if e.dst_table != base_table:
                        join_parts_list.append(
                            f'JOIN "{e.dst_table}" ON "{e.src_table}"."{e.src_column}" = "{e.dst_table}"."{e.dst_column}"'
                        )
                    elif e.src_table != base_table:
                        join_parts_list.append(
                            f'JOIN "{e.src_table}" ON "{e.src_table}"."{e.src_column}" = "{e.dst_table}"."{e.dst_column}"'
                        )
                join_sql = " ".join(join_parts_list)

                # Determine TEXT columns that hold numeric values (need CAST for range ops)
                _text_cols: set[str] = set()
                if db_path:
                    try:
                        _conn_tc = sqlite3.connect(str(db_path), timeout=5)
                        for _tbl_name in {n.table for n in pop_filters + subset_filters}:
                            for _ci_tc in _conn_tc.execute(f'PRAGMA table_info("{_tbl_name}")').fetchall():
                                if (_ci_tc[2] or "").upper() == "TEXT":
                                    _text_cols.add(f"{_tbl_name}.{_ci_tc[1]}")
                        _conn_tc.close()
                    except Exception:
                        pass

                def _filter_expr(n: QueryNode) -> str:
                    col_ref = f'"{n.table}"."{n.column}"'
                    # CAST TEXT columns for numeric range comparisons
                    range_ops = {">=", "<=", ">", "<"}
                    is_numeric_val = re.match(r'^-?\d+\.?\d*$', str(n.value))
                    if n.operator in range_ops and f"{n.table}.{n.column}" in _text_cols and is_numeric_val:
                        col_ref = f'CAST("{n.table}"."{n.column}" AS REAL)'
                        return f'{col_ref} {n.operator} {n.value}'
                    if n.operator.upper() == "LIKE":
                        return f'{col_ref} LIKE \'{n.value}\''
                    return f'{col_ref} {n.operator} \'{n.value}\''

                # Population WHERE clause
                pop_parts = []
                for n in pop_filters:
                    if n.column.startswith("_expr:"):
                        pop_parts.append(f'{n.column[len("_expr:"):]} {n.operator} {n.value}')
                    else:
                        pop_parts.append(_filter_expr(n))
                # Subset CASE WHEN clause
                subset_parts = [_filter_expr(n) for n in subset_filters]
                subset_cond = " AND ".join(subset_parts)
                pop_where = f" WHERE {' AND '.join(pop_parts)}" if pop_parts else ""
                # Find PK of base table (population table)
                pk_col = "Id"
                if kg and base_table:
                    ts = kg.get_table(base_table)
                    if ts and ts.columns:
                        pk_col = ts.columns[0].name
                pop_desc = f" matching [{' AND '.join(pop_parts)}]" if pop_parts else ""
                if comp_type == "percentage":
                    parts.append(
                        f'CROSS-TABLE RATIO: Population = "{base_table}" rows{pop_desc}. '
                        f'Subset = those where {subset_cond} (via JOIN). '
                        f'Subset is WITHIN population, so use CASE WHEN for subset + WHERE for population:\n'
                        f'  SELECT CAST(COUNT(CASE WHEN {subset_cond} THEN 1 END) AS REAL) * 100 / COUNT("{base_table}"."{pk_col}") '
                        f'FROM "{base_table}" {join_sql}{pop_where}'
                    )
                else:
                    parts.append(
                        f'CROSS-TABLE RATIO: Population = "{base_table}" rows{pop_desc}. '
                        f'Subset = those where {subset_cond} (via JOIN). '
                        f'Subset is WITHIN population, so use CASE WHEN for subset + WHERE for population:\n'
                        f'  SELECT CAST(COUNT(CASE WHEN {subset_cond} THEN 1 END) AS REAL) / COUNT("{base_table}"."{pk_col}") '
                        f'FROM "{base_table}" {join_sql}{pop_where}'
                    )
            else:
                fv_lines: list[str] = []
                for node in path.filter_nodes:
                    if node.column.startswith("_expr:"):
                        expr_sql = node.column[len("_expr:"):]
                        fv_lines.append(f'  COMPUTED: WHERE {expr_sql} {node.operator} {node.value}')
                    elif node.operator.upper() == "IS NOT NULL":
                        fv_lines.append(f'  "{node.table}"."{node.column}": IS NOT NULL')
                    elif node.operator.upper() == "LIKE":
                        like_val = node.value
                        if "_" in like_val:
                            like_val = like_val.replace("_", r"\_")
                            fv_lines.append(f'  "{node.table}"."{node.column}" LIKE \'{like_val}\' ESCAPE \'\\\' (case-insensitive)')
                        else:
                            fv_lines.append(f'  "{node.table}"."{node.column}" LIKE \'{like_val}\' (case-insensitive)')
                    else:
                        _val_display = node.value
                        _is_text_val = node.operator == "=" and not re.match(r'^-?\d+\.?\d*$', str(node.value))
                        if _is_text_val:
                            _val_display = f"'{node.value}' COLLATE NOCASE"
                        fv_lines.append(f'  "{node.table}"."{node.column}": {node.operator} {_val_display}')
                        # When comparing a year value against a TEXT date column,
                        # hint the LLM to use proper date/year extraction
                        val_str = str(node.value)
                        if (node.operator in ("<", ">", "<=", ">=")
                                and re.match(r'^(19|20)\d{2}$', val_str)
                                and db_path and db_path.exists()):
                            try:
                                _dc = sqlite3.connect(str(db_path), timeout=5)
                                _sample = _dc.execute(
                                    f'SELECT "{node.column}" FROM "{node.table}" '
                                    f'WHERE "{node.column}" IS NOT NULL LIMIT 5'
                                ).fetchall()
                                _dc.close()
                                _has_dates = any(
                                    re.match(r'^\d{4}-\d{2}-\d{2}', str(s[0]))
                                    for s in _sample if s[0]
                                )
                                if _has_dates:
                                    fv_lines.append(
                                        f'  Note: "{node.column}" stores full dates (YYYY-MM-DD text). '
                                        f'Year comparison: "{node.column}" {node.operator} \'{val_str}-01-01\''
                                    )
                            except Exception:
                                pass
                        # Subquery scope propagation: if value is a subquery with WHERE,
                        # those conditions must ALSO apply in the outer query
                        if val_str.lstrip().upper().startswith("(SELECT") or val_str.lstrip().upper().startswith("SELECT"):
                            where_match = re.search(r'\bWHERE\s+(.+?)(?:\)$|\bGROUP\b|\bORDER\b|\bLIMIT\b)', val_str, re.IGNORECASE)
                            if where_match:
                                inner_where = where_match.group(1).strip().rstrip(")")
                                fv_lines.append(
                                    f'  Scope: subquery filters [{inner_where}] — outer query covers the same scope.'
                                )
                parts.append("CONDITIONS:\n" + "\n".join(fv_lines))

        # --- Self-join / OR-logic / AND-logic detection ---
        # When multiple filters target the same column with "=" and different values,
        # check question language: "X or Y" → OR (IN), "X and Y" / "both" → self-join
        # Skip for ratio/percentage — those use CASE WHEN pattern, not self-join
        if path.filter_nodes and comp_type not in ("ratio", "percentage"):
            col_values: dict[str, list[str]] = {}
            for node in path.filter_nodes:
                if node.operator == "=" and not node.column.startswith("_expr:"):
                    key = f"{node.table}.{node.column}"
                    col_values.setdefault(key, []).append(str(node.value))
            q_lower = (goal.get("what_user_wants") or question).lower()
            for col_key, values in col_values.items():
                if len(values) > 1:
                    tbl, col = col_key.split(".", 1)
                    # Check if the question uses "or" between the concepts represented by these values.
                    # Look for: "value1 or value2", or words containing the values near "or"
                    is_or = False
                    # Direct check: values (or words containing them) separated by "or"
                    for i in range(len(values)):
                        for j in range(i+1, len(values)):
                            vi, vj = values[i].lower(), values[j].lower()
                            # For short values (≤2 chars), require longer word containment
                            # to avoid matching random short substrings
                            if len(vi) >= 3 and len(vj) >= 3:
                                pat_i = rf'\b\w*{re.escape(vi)}\w*\b'
                                pat_j = rf'\b\w*{re.escape(vj)}\w*\b'
                            else:
                                pat_i = rf'\b\w*{re.escape(vi)}\w{{2,}}\b'
                                pat_j = rf'\b\w*{re.escape(vj)}\w{{2,}}\b'
                            pattern = rf'{pat_i}.*?\bor\b.*?{pat_j}'
                            pattern_rev = rf'{pat_j}.*?\bor\b.*?{pat_i}'
                            if re.search(pattern, q_lower) or re.search(pattern_rev, q_lower):
                                is_or = True
                                break
                        if is_or:
                            break
                    # Fallback: question has "or" and no "and"/"both" — likely OR semantics
                    if not is_or and " or " in q_lower and " and " not in q_lower and "both" not in q_lower:
                        is_or = True
                    if is_or:
                        in_list = ", ".join(f"'{v}'" for v in values)
                        parts.append(
                            f'OR FILTER: Multiple values for "{tbl}"."{col}" connected by OR. '
                            f'Use: WHERE "{col}" IN ({in_list})'
                        )
                    else:
                        parts.append(
                            f'SELF-JOIN REQUIRED: Multiple values for "{tbl}"."{col}" ({values}). '
                            f'These must BOTH hold on related rows (not the same row). '
                            f'Use a self-join or subquery pattern: find entities where one related row has value {values[0]} '
                            f'AND another related row has value {values[1]}.'
                        )

        # --- Aggregation duplication warning ---
        # When SUM/AVG targets a NUMERIC column on the "one" side of a one-to-many join,
        # the value gets multiplied. Only warn for numeric columns (not GROUP BY text columns).
        if comp_type in ("sum", "avg") and path.edges and path.output_nodes and kg and kg.graph and not suppress_join:
            child_to_parent: dict[str, str] = {}
            for edge in kg.graph.fk_edges:
                src_col = kg.graph.columns.get(edge.src)
                dst_col = kg.graph.columns.get(edge.dst)
                if src_col and dst_col:
                    child_to_parent[src_col.table_id] = dst_col.table_id
            tables_in_path = set(path.tables_in_path) if path.tables_in_path else set()
            for node in path.output_nodes:
                # Only warn for numeric columns (candidates for SUM/AVG), not text (GROUP BY)
                node_col = kg.graph.columns.get(f"{node.table}.{node.column}")
                if not node_col or node_col.sql_type.upper() not in ("REAL", "FLOAT", "NUMERIC", "INTEGER", "INT"):
                    continue
                child_tables_in_path = [
                    child for child, parent in child_to_parent.items()
                    if parent == node.table and child in tables_in_path
                ]
                if child_tables_in_path:
                    child_table = child_tables_in_path[0]
                    child_schema = kg.get_table(child_table)
                    if child_schema:
                        numeric_cols = [
                            c.name for c in child_schema.columns
                            if c.sql_type.upper() in ("REAL", "INTEGER", "NUMERIC", "FLOAT", "INT")
                            and not c.name.lower().endswith("id")
                            and "link" not in c.name.lower()
                        ]
                        if numeric_cols:
                            parts.append(
                                f'FAN-OUT: "{node.table}" is on the ONE side of a one-to-many join to "{child_table}". '
                                f'Each "{node.table}" row repeats per child row. '
                                f'Aggregating "{node.column}" directly would multiply the value. '
                                f'"{child_table}"."{numeric_cols[0]}" has the per-row detail values.'
                            )

        # --- COUNT DISTINCT hint for count with JOINs ---
        if comp_type == "count" and path.edges and path.output_nodes and not suppress_join:
            # Determine what to count: prefer the entity of interest's FK/PK
            count_table = path.output_nodes[0].table
            count_col = None

            # If there's an entity-of-interest that differs from the output table,
            # count through its FK column in the bridge/output table
            entity_of_interest = goal.get("_entity_of_interest", "")
            if entity_of_interest and entity_of_interest.lower() != count_table.lower():
                # Find FK column in the output table pointing to entity of interest
                if kg:
                    for edge in (kg.graph.fk_edges if kg.graph else []):
                        src_tbl = edge.src.split(".")[0] if "." in edge.src else ""
                        dst_tbl = edge.dst.split(".")[0] if "." in edge.dst else ""
                        src_col = edge.src.split(".")[1] if "." in edge.src else ""
                        if src_tbl.lower() == count_table.lower() and dst_tbl.lower() == entity_of_interest.lower():
                            count_col = src_col
                            break

            # Structural fallback: on a bridge table with multiple FK edges,
            # if the output column is an FK whose target table is equality-filtered
            # to a single entity, counting that column gives trivially 1.
            # Count the other FK instead.
            if not count_col and kg and kg.graph and kg.graph.fk_edges:
                output_col = path.output_nodes[0].column
                # Collect FK columns in the count_table
                fk_cols_in_table: list[tuple[str, str]] = []  # (src_col, dst_table)
                for edge in kg.graph.fk_edges:
                    src_tbl = edge.src.split(".")[0] if "." in edge.src else ""
                    src_col = edge.src.split(".")[1] if "." in edge.src else ""
                    dst_tbl = edge.dst.split(".")[0] if "." in edge.dst else ""
                    if src_tbl.lower() == count_table.lower():
                        fk_cols_in_table.append((src_col, dst_tbl))

                if len(fk_cols_in_table) >= 2:
                    # Check if the output column's FK target is equality-constrained
                    output_fk_target = None
                    for col, tgt in fk_cols_in_table:
                        if col.lower() == output_col.lower():
                            output_fk_target = tgt
                            break

                    if output_fk_target:
                        # Is the target table equality-filtered?
                        target_eq_filtered = any(
                            fnode.table.lower() == output_fk_target.lower()
                            and fnode.operator in ("=", "==")
                            for fnode in (path.filter_nodes or [])
                        )
                        if target_eq_filtered:
                            # Count the other FK instead
                            for col, tgt in fk_cols_in_table:
                                if col.lower() != output_col.lower():
                                    count_col = col
                                    break

            if not count_col:
                if kg:
                    table_schema = kg.get_table(count_table)
                    if table_schema and table_schema.columns:
                        count_col = table_schema.columns[0].name
                if not count_col:
                    count_col = path.output_nodes[0].column

            parts.append(
                f'COUNT TARGET: "{count_table}"."{count_col}". '
                f'The JOIN may produce duplicate rows (one-to-many fan-out).'
            )

        # (Cross-row hint moved to rules engine — injected via engine_out.sql_directives)

        # --- DISTINCT hint for lookup queries joining to detail tables ---
        if (comp_type == "simple_lookup" and path.edges and path.output_nodes
                and path.filter_nodes and kg):
            output_tables = {n.table for n in path.output_nodes}
            filter_tables = {n.table for n in path.filter_nodes}
            detail_tables = filter_tables - output_tables
            if detail_tables:
                # Confirm the filter table is on the "many" side (fact/bridge/higher row count)
                needs_distinct = False
                for dt in detail_tables:
                    dt_schema = kg.get_table(dt)
                    if dt_schema and dt_schema.role in ("fact", "bridge", ""):
                        needs_distinct = True
                        break
                    # Fallback: if filter table has more rows, it's likely many-side
                    for ot in output_tables:
                        ot_schema = kg.get_table(ot)
                        if dt_schema and ot_schema and dt_schema.row_count > ot_schema.row_count:
                            needs_distinct = True
                            break
                if needs_distinct:
                    parts.append(
                        'DUPLICATES: The JOIN crosses to a detail table (many-side). '
                        'Each entity may appear multiple times in the result.'
                    )

        # (Per-entity AVG pattern moved to Layer 4 pattern router — emitted near top of grounding)

        # --- Temporal scope propagation ---
        # If date filters exist on one table but joined tables also have date columns,
        # the output table's date column likely needs the same temporal constraint.
        if path.filter_nodes and path.edges and kg:
            date_keywords = ("date", "year", "month", "time", "period")
            filter_tables_with_dates: dict[str, list[QueryNode]] = {}
            for node in path.filter_nodes:
                if any(dk in node.column.lower() for dk in date_keywords):
                    filter_tables_with_dates.setdefault(node.table, []).append(node)
            if filter_tables_with_dates:
                output_tables = {n.table for n in path.output_nodes} if path.output_nodes else set()
                for out_table in output_tables:
                    if out_table in filter_tables_with_dates:
                        continue
                    table_schema = kg.get_table(out_table)
                    if not table_schema:
                        continue
                    date_cols = [
                        c.name for c in table_schema.columns
                        if any(dk in c.name.lower() for dk in date_keywords)
                    ]
                    if date_cols:
                        # Sample the target date column to reconcile format
                        target_col = date_cols[0]
                        target_sample = ""
                        if db_path:
                            try:
                                _conn = sqlite3.connect(str(db_path))
                                _row = _conn.execute(
                                    f'SELECT "{target_col}" FROM "{out_table}" WHERE "{target_col}" IS NOT NULL LIMIT 1'
                                ).fetchone()
                                if _row:
                                    target_sample = str(_row[0])
                                _conn.close()
                            except Exception:
                                pass
                        # Skip if sample value doesn't look like a date/timestamp
                        # (e.g., "+5.478" is a race duration, not a temporal value)
                        if target_sample and not re.search(r'\d{4}', target_sample):
                            continue
                        for filter_table, date_filters in filter_tables_with_dates.items():
                            for df in date_filters:
                                # Extract year/month digits from filter value
                                digits = re.findall(r'\d+', str(df.value))
                                year = digits[0] if digits else ""
                                month = digits[1] if len(digits) > 1 else ""
                                # Build condition matching target column's actual format
                                if target_sample and year:
                                    if target_sample.isdigit() and len(target_sample) == 6:
                                        # YYYYMM integer format
                                        if month:
                                            cond = f'"{out_table}"."{target_col}" = {year}{month.zfill(2)}'
                                        else:
                                            cond = f'CAST("{out_table}"."{target_col}" AS TEXT) LIKE \'{year}%\''
                                    elif "-" in target_sample:
                                        # YYYY-MM or YYYY-MM-DD format
                                        if month:
                                            cond = f'"{out_table}"."{target_col}" LIKE \'{year}-{month.zfill(2)}%\''
                                        else:
                                            cond = f'"{out_table}"."{target_col}" LIKE \'{year}%\''
                                    else:
                                        cond = f'"{out_table}"."{target_col}" LIKE \'{df.value}\''
                                else:
                                    cond = f'"{out_table}"."{target_col}" LIKE \'{df.value}\''
                                parts.append(
                                    f'TEMPORAL: "{out_table}"."{target_col}" covers all time periods. '
                                    f'Without filtering it, the JOIN includes rows outside the target period. '
                                    f'Matching condition: {cond}'
                                )

        # ORDER BY (from picked dict) — used for superlatives (lowest/highest/best)
        order_by = goal.get("order_by")
        if order_by and isinstance(order_by, dict):
            col_ref = order_by.get("column", "")
            direction = order_by.get("direction", "ASC")
            parts.append(f'ORDER: "{col_ref}" {direction} — if ties matter, a subquery (SELECT MIN/MAX) handles them')

        # --- Data distribution for numeric filter columns ---
        # Show actual min/max so the LLM can judge whether thresholds are meaningful
        if db_path and path.filter_nodes:
            dist_lines: list[str] = []
            seen_cols: set[str] = set()
            try:
                conn = sqlite3.connect(str(db_path), timeout=5)
                for f in path.filter_nodes:
                    key = f"{f.table}.{f.column}"
                    if key in seen_cols or f.operator in ("=", "IS NOT NULL", "LIKE"):
                        continue
                    seen_cols.add(key)
                    try:
                        # Skip date columns — CAST to REAL gives garbage for date text
                        _sample_vals = conn.execute(
                            f'SELECT "{f.column}" FROM "{f.table}" '
                            f'WHERE "{f.column}" IS NOT NULL LIMIT 5'
                        ).fetchall()
                        _is_date_col = any(
                            re.match(r'^\d{4}-\d{2}-\d{2}', str(s[0]))
                            for s in _sample_vals if s[0]
                        )
                        if _is_date_col:
                            _min_date = conn.execute(
                                f'SELECT MIN("{f.column}") FROM "{f.table}" '
                                f'WHERE "{f.column}" IS NOT NULL'
                            ).fetchone()
                            _max_date = conn.execute(
                                f'SELECT MAX("{f.column}") FROM "{f.table}" '
                                f'WHERE "{f.column}" IS NOT NULL'
                            ).fetchone()
                            cnt = conn.execute(
                                f'SELECT COUNT(*) FROM "{f.table}" '
                                f'WHERE "{f.column}" IS NOT NULL'
                            ).fetchone()[0]
                            dist_lines.append(
                                f'  "{f.table}"."{f.column}": DATE column (YYYY-MM-DD), '
                                f'range=[{_min_date[0]}..{_max_date[0]}], {cnt} records'
                            )
                            continue
                        row = conn.execute(
                            f'SELECT MIN(CAST("{f.column}" AS REAL)), '
                            f'MAX(CAST("{f.column}" AS REAL)), '
                            f'COUNT(*) '
                            f'FROM "{f.table}" WHERE "{f.column}" IS NOT NULL'
                        ).fetchone()
                        if row and row[2] > 0:
                            if row[0] == 0.0 and row[1] == 0.0:
                                samples = conn.execute(
                                    f'SELECT DISTINCT "{f.column}" FROM "{f.table}" '
                                    f'WHERE "{f.column}" IS NOT NULL LIMIT 3'
                                ).fetchall()
                                sample_str = ", ".join(repr(s[0]) for s in samples)
                                dist_lines.append(
                                    f'  "{f.table}"."{f.column}": TEXT format, '
                                    f'samples: [{sample_str}], {row[2]} records'
                                )
                            else:
                                dist_lines.append(
                                    f'  "{f.table}"."{f.column}": min={row[0]}, max={row[1]}, '
                                    f'{row[2]} records'
                                )
                    except Exception:
                        pass
                conn.close()
            except Exception:
                pass
            if dist_lines:
                parts.append("DATA RANGES:\n" + "\n".join(dist_lines))

        # Previous attempt (retry)
        if sql:
            parts.append(f"PREVIOUS ATTEMPT (returned wrong result or error):\n  {sql}")

        # Layer 3 (ADVISOR): Data notes — advisory hints, never override structural decisions
        data_notes = _build_data_notes(path, kg)
        if data_notes:
            parts.append("DATA NOTES (advisory — use to sanity-check your result):\n" + "\n".join(f"  - {n}" for n in data_notes[:5]))

        return "EVIDENCE:\n" + "\n".join(parts)
