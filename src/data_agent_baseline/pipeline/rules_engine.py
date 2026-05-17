"""
Evidence-based rules engine for SQL query construction.

Operates on the KnowledgeGraph + DB probes to make deterministic decisions
that a small LLM cannot reliably make (threshold selection, query structure).

Each rule:
1. Checks if it applies (trigger condition based on graph/data properties)
2. Probes the DB for evidence
3. Transforms filter_nodes and/or emits SQL structural directives

Rules are ordered by priority. Later rules can override earlier ones.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.pipeline.kg_builder import KnowledgeGraph, TableSchema
from data_agent_baseline.pipeline.kg_path_planner import QueryNode, QueryPath


@dataclass(slots=True)
class DecompositionStep:
    """One step of a multi-step SQL execution plan."""
    description: str  # what this step retrieves
    sql_template: str  # SQL with {prev_result} placeholders for prior step outputs
    output_var: str  # variable name for result (e.g., "value_a", "value_b")


@dataclass(slots=True)
class RuleResult:
    """Output of a single rule evaluation."""
    filter_nodes: list[QueryNode] | None = None  # replacement filter nodes (None = no change)
    sql_directives: list[str] = field(default_factory=list)  # lines injected into grounding
    log_entries: list[tuple[str, str]] = field(default_factory=list)  # (tag, message) pairs
    decomposition: list[DecompositionStep] = field(default_factory=list)  # multi-step plan


@dataclass(slots=True)
class EngineContext:
    """All evidence available to rules."""
    question: str
    question_lower: str
    user_intent: str
    comp_type: str
    filter_nodes: list[QueryNode]
    output_nodes: list[QueryNode]
    kg: KnowledgeGraph
    db_path: Path | None
    knowledge_text: str
    anchor_text: str
    model_call: Callable | None = None
    domain_locked_columns: set[str] = None  # "table.column" entries that adaptive loop must not change

    # Computed lazily
    _rows_per_entity: dict[str, float] | None = None
    _col_coverage: dict[str, float] | None = None

    @property
    def entity_table(self) -> str:
        if self.output_nodes:
            return self.output_nodes[0].table
        return ""

    def rows_per_entity(self, table: str, id_col: str) -> float:
        """How many rows per distinct ID — indicates 1:many (detail table)."""
        if self._rows_per_entity is None:
            self._rows_per_entity = {}
        key = f"{table}.{id_col}"
        if key not in self._rows_per_entity:
            if not self.db_path:
                return 1.0
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=5)
                row = conn.execute(
                    f'SELECT COUNT(*) * 1.0 / MAX(COUNT(DISTINCT "{id_col}"), 1) '
                    f'FROM "{table}" WHERE "{id_col}" IS NOT NULL'
                ).fetchone()
                conn.close()
                self._rows_per_entity[key] = row[0] if row else 1.0
            except Exception:
                self._rows_per_entity[key] = 1.0
        return self._rows_per_entity[key]

    def col_null_ratio(self, table: str, column: str) -> float:
        """Fraction of rows where column IS NULL. Uses Layer 3 pre-computed stats first."""
        # Layer 3: use pre-computed stats if available
        ts = self.kg.get_table(table)
        if ts and hasattr(ts, 'col_stats') and column in ts.col_stats:
            cached = ts.col_stats[column].get("null_ratio")
            if cached is not None:
                return cached
        if not self.db_path:
            return 0.0
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            row = conn.execute(
                f'SELECT 1.0 - (COUNT("{column}") * 1.0 / MAX(COUNT(*), 1)) '
                f'FROM "{table}"'
            ).fetchone()
            conn.close()
            return row[0] if row else 0.0
        except Exception:
            return 0.0

    def col_stats(self, table: str, column: str) -> dict[str, Any]:
        """Get min/max/avg/distinct for a column from KG or live query."""
        ts = self.kg.get_table(table)
        if ts and column in ts.col_stats:
            return ts.col_stats[column]
        # Live probe
        if not self.db_path:
            return {}
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            row = conn.execute(
                f'SELECT MIN(CAST("{column}" AS REAL)), MAX(CAST("{column}" AS REAL)), '
                f'AVG(CAST("{column}" AS REAL)), COUNT(DISTINCT "{column}"), COUNT(*) '
                f'FROM "{table}" WHERE "{column}" IS NOT NULL'
            ).fetchone()
            conn.close()
            if row and row[4] > 0:
                return {"min": row[0], "max": row[1], "avg": row[2], "distinct": row[3], "count": row[4]}
        except Exception:
            pass
        return {}

    def count_matching(self, table: str, column: str, operator: str, value: Any) -> int:
        """Count rows matching a condition. Returns -1 if unevaluable."""
        if not self.db_path:
            return -1
        try:
            val_str = str(value).strip()
            if val_str.lstrip("(").lower().startswith("select"):
                return -1
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            if operator.upper() == "IN":
                vals = re.findall(r"'([^']*)'", val_str)
                if not vals:
                    return -1
                placeholders = ",".join("?" * len(vals))
                row = conn.execute(
                    f'SELECT COUNT(*) FROM "{table}" '
                    f'WHERE "{column}" IN ({placeholders})',
                    vals,
                ).fetchone()
            else:
                row = conn.execute(
                    f'SELECT COUNT(*) FROM "{table}" '
                    f'WHERE "{column}" IS NOT NULL AND "{column}" {operator} ?',
                    (value,),
                ).fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return -1

    def co_occurrence(self, table: str, id_col: str, col_a: str, col_b: str) -> tuple[int, int]:
        """How many entities have both col_a and col_b non-null?
        Returns (entities_with_both_on_same_row, entities_with_both_across_rows)."""
        if not self.db_path:
            return (0, 0)
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            # Same row: both non-null on one record
            same = conn.execute(
                f'SELECT COUNT(DISTINCT "{id_col}") FROM "{table}" '
                f'WHERE "{col_a}" IS NOT NULL AND "{col_b}" IS NOT NULL'
            ).fetchone()
            # Across rows: entity has at least one row with A and at least one with B
            across = conn.execute(
                f'SELECT COUNT(*) FROM '
                f'(SELECT "{id_col}" FROM "{table}" WHERE "{col_a}" IS NOT NULL '
                f' INTERSECT '
                f' SELECT "{id_col}" FROM "{table}" WHERE "{col_b}" IS NOT NULL)'
            ).fetchone()
            conn.close()
            return (same[0] if same else 0, across[0] if across else 0)
        except Exception:
            return (0, 0)

    def cardinality_ratio(self, table: str, column: str) -> float:
        """distinct_values / total_non_null_rows. Low = categorical, high = continuous."""
        stats = self.col_stats(table, column)
        if not stats:
            return 0.5
        distinct = stats.get("distinct", 1)
        count = stats.get("count", 1)
        return distinct / max(count, 1)

    def join_fan_out(self, src_table: str, dst_table: str) -> float:
        """How much does joining src to dst multiply rows? > 1 means fan-out."""
        if not self.db_path or not self.kg.graph:
            return 1.0
        edges = self.kg.graph.get_fk_between(src_table, dst_table)
        if not edges:
            return 1.0
        edge = edges[0]
        src_col_node = self.kg.graph.columns.get(edge.src)
        dst_col_node = self.kg.graph.columns.get(edge.dst)
        if not src_col_node or not dst_col_node:
            return 1.0
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            src_count = conn.execute(f'SELECT COUNT(*) FROM "{src_table}"').fetchone()[0]
            dst_count = conn.execute(f'SELECT COUNT(*) FROM "{dst_table}"').fetchone()[0]
            conn.close()
            return max(src_count, 1) / max(dst_count, 1)
        except Exception:
            return 1.0

    def is_sparse_column(self, table: str, column: str) -> bool:
        """Column has >70% nulls — existence is more meaningful than value."""
        return self.col_null_ratio(table, column) > 0.7

    def value_spread(self, table: str, column: str) -> str:
        """Characterize value distribution: 'tight_cluster', 'wide_spread', or 'unknown'."""
        stats = self.col_stats(table, column)
        if not stats or "min" not in stats or "max" not in stats:
            return "unknown"
        data_min, data_max = stats["min"], stats["max"]
        if data_max == data_min:
            return "tight_cluster"
        # Coefficient of variation proxy: (max-min) / avg
        avg = stats.get("avg")
        if avg and avg != 0:
            spread = (data_max - data_min) / abs(avg)
            if spread < 1.0:
                return "tight_cluster"
            elif spread > 5.0:
                return "wide_spread"
        return "moderate"

    def count_distinct_matching(self, table: str, id_col: str, filter_col: str, operator: str, value: Any) -> int:
        """Count distinct entities matching a condition."""
        if not self.db_path:
            return -1
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            row = conn.execute(
                f'SELECT COUNT(DISTINCT "{id_col}") FROM "{table}" '
                f'WHERE "{filter_col}" IS NOT NULL AND "{filter_col}" {operator} ?',
                (value,),
            ).fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return -1

    def total_non_null(self, table: str, column: str) -> int:
        """Count total non-null rows for a column."""
        if not self.db_path:
            return -1
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            row = conn.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NOT NULL'
            ).fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return -1


# ---------------------------------------------------------------------------
# Individual Rules
# ---------------------------------------------------------------------------


def rule_impossible_threshold(ctx: EngineContext) -> RuleResult | None:
    """Detect filter conditions that match 0 rows and replace with IS NOT NULL.

    Trigger: 2+ numeric conditions on same column where each matches 0 rows.
    Evidence: DB probe of each condition independently.
    Action: Replace all conditions on that column with IS NOT NULL.
    """
    if not ctx.db_path:
        return None

    col_groups: dict[str, list[int]] = {}
    for i, n in enumerate(ctx.filter_nodes):
        if n.operator in ("=", "LIKE", "IN", "IS NOT NULL"):
            continue
        col_groups.setdefault(f"{n.table}.{n.column}", []).append(i)

    changes: list[tuple[str, list[int]]] = []
    for key, indices in col_groups.items():
        if len(indices) < 2:
            continue
        all_zero = True
        for idx in indices:
            f = ctx.filter_nodes[idx]
            count = ctx.count_matching(f.table, f.column, f.operator, f.value)
            if count != 0:
                all_zero = False
                break
        if all_zero:
            changes.append((key, indices))

    if not changes:
        return None

    result = RuleResult(log_entries=[])
    new_nodes = list(ctx.filter_nodes)
    to_remove: list[int] = []
    for key, indices in changes:
        first = new_nodes[indices[0]]
        new_nodes[indices[0]] = QueryNode(
            table=first.table, column=first.column, role="filter",
            operator="IS NOT NULL", value="",
        )
        to_remove.extend(indices[1:])
        result.log_entries.append(("rule_impossible_threshold", f"{key}: conditions matched 0 rows → IS NOT NULL"))
    for idx in sorted(to_remove, reverse=True):
        new_nodes.pop(idx)
    result.filter_nodes = new_nodes
    return result


def rule_coverage_simplification(ctx: EngineContext) -> RuleResult | None:
    """Detect filter conditions that match ALL rows and simplify to IS NOT NULL.

    Trigger: numeric condition matches every non-null row (the threshold is below min or above max).
    Evidence: count with threshold == count without threshold.
    Action: Replace with IS NOT NULL (the condition is vacuous but existence matters).
    """
    if not ctx.db_path:
        return None

    changes: list[tuple[int, str]] = []
    for i, n in enumerate(ctx.filter_nodes):
        if n.operator in ("=", "LIKE", "IN", "IS NOT NULL"):
            continue
        total = ctx.total_non_null(n.table, n.column)
        if total <= 0:
            continue
        matching = ctx.count_matching(n.table, n.column, n.operator, n.value)
        if matching == total:
            changes.append((i, f"{n.table}.{n.column}"))

    if not changes:
        return None

    result = RuleResult(log_entries=[])
    new_nodes = list(ctx.filter_nodes)
    for idx, key in changes:
        f = new_nodes[idx]
        new_nodes[idx] = QueryNode(
            table=f.table, column=f.column, role="filter",
            operator="IS NOT NULL", value="",
        )
        result.log_entries.append(("rule_coverage_simplify", f"{key} {f.operator} {f.value} matches all rows → IS NOT NULL"))
    result.filter_nodes = new_nodes
    return result


def rule_cross_row_structure(ctx: EngineContext) -> RuleResult | None:
    """Determine if query needs cross-row subqueries.

    Trigger: comp_type is count/sum/avg AND entity table is a detail table
             (rows_per_entity > 1) AND 2+ filter conditions on different columns.
    Evidence: rows_per_entity from DB probe.
    Action: Emit SQL directive with subquery pattern.
    """
    if ctx.comp_type not in ("count", "sum", "avg"):
        return None
    if not ctx.output_nodes:
        return None

    entity_table = ctx.entity_table
    detail_filters = [n for n in ctx.filter_nodes if n.table == entity_table]
    if len(detail_filters) < 2:
        return None

    distinct_cols = {n.column for n in detail_filters}
    if len(distinct_cols) < 2:
        return None

    # Check if entity table is actually a detail table (multiple rows per ID)
    pk_col = None
    ts = ctx.kg.get_table(entity_table)
    if ts and ts.columns:
        pk_col = ts.columns[0].name
    if not pk_col:
        pk_col = ctx.output_nodes[0].column

    rpe = ctx.rows_per_entity(entity_table, pk_col)
    if rpe <= 1.5:
        return None  # Not a detail table — same-row filtering is correct

    # Verify cross-row is actually needed: check co-occurrence
    # If entities with both columns on same row == entities across rows, no need for subqueries
    cols_list = list(distinct_cols)
    if len(cols_list) >= 2:
        same, across = ctx.co_occurrence(entity_table, pk_col, cols_list[0], cols_list[1])
        if same == across and same > 0:
            return None  # All co-occurrences are on same row — simple WHERE is correct

    # Build subquery pattern grouped by column
    col_groups: dict[str, list[QueryNode]] = {}
    for f in detail_filters:
        col_groups.setdefault(f.column, []).append(f)

    subq_parts: list[str] = []
    for col, filters in col_groups.items():
        conditions: list[str] = []
        for f in filters:
            if f.operator == "IS NOT NULL":
                conditions.append(f'"{f.column}" IS NOT NULL')
            else:
                conditions.append(f'"{f.column}" {f.operator} \'{f.value}\'')
        conditions = list(dict.fromkeys(conditions))
        where_clause = " AND ".join(conditions)
        subq_parts.append(
            f'"{entity_table}"."{pk_col}" IN '
            f'(SELECT "{pk_col}" FROM "{entity_table}" WHERE {where_clause})'
        )

    directive = (
        "CROSS-ROW HINT (MANDATORY — you MUST use this subquery pattern, do NOT put these "
        "conditions in the outer WHERE directly):\n  WHERE " + "\n    AND ".join(subq_parts)
    )

    return RuleResult(
        sql_directives=[directive],
        log_entries=[("rule_cross_row", f"rows_per_entity={rpe:.1f}, {len(subq_parts)} subqueries")]
    )


def rule_normal_abnormal_resolution(ctx: EngineContext) -> RuleResult | None:
    """Resolve normal/abnormal semantics using full range resolution cascade.

    Trigger: question contains "normal" or "abnormal" + numeric filter columns.
    Evidence: knowledge text → hardcoded → data distribution → LLM inference.
    Action: Fix filter operators/values to match correct semantics.
    """
    q = ctx.question_lower
    has_normal = "normal" in q
    has_abnormal = "abnormal" in q
    if not has_normal and not has_abnormal:
        return None

    if not ctx.db_path:
        return None

    # Identify filter nodes that are numeric thresholds on relevant columns
    numeric_filters: dict[str, list[int]] = {}
    for i, n in enumerate(ctx.filter_nodes):
        if n.operator in ("=", "LIKE", "IN", "IS NOT NULL"):
            continue
        numeric_filters.setdefault(f"{n.table}.{n.column}", []).append(i)

    if not numeric_filters:
        return None

    # Determine which columns relate to "normal" vs "abnormal" in the question
    population_text = ""
    metric_text = ""
    for line in ctx.user_intent.split("\n"):
        if "Population (WHERE):" in line:
            population_text = line.split("Population (WHERE):")[1].strip().lower()
        elif "Metric (SELECT):" in line:
            metric_text = line.split("Metric (SELECT):")[1].strip().lower()

    log_entries: list[tuple[str, str]] = []
    replacements: dict[int, QueryNode | None] = {}
    additions: list[QueryNode] = []

    for key, indices in numeric_filters.items():
        table, column = key.split(".", 1)
        col_lower = column.lower()

        # Determine if this column is in the "normal" or "abnormal" part of the question
        is_normal_target = "normal" in population_text and _col_mentioned_in(col_lower, population_text)
        is_abnormal_target = "abnormal" in metric_text and _col_mentioned_in(col_lower, metric_text)

        if not is_normal_target and not is_abnormal_target:
            is_normal_target = _col_near_keyword(col_lower, "normal", ctx.question_lower)
            is_abnormal_target = _col_near_keyword(col_lower, "abnormal", ctx.question_lower)

        if not is_normal_target and not is_abnormal_target:
            continue

        # Get data distribution
        stats = ctx.col_stats(table, column)
        if not stats:
            continue

        data_min = stats.get("min")
        data_max = stats.get("max")
        if data_min is None or data_max is None:
            continue

        # Full cascade: knowledge → hardcoded → distribution → LLM
        normal_range = _resolve_normal_range(ctx, table, column, ctx.model_call)

        if is_abnormal_target:
            if normal_range:
                norm_low, norm_high = normal_range
                if data_max < norm_low or data_min > norm_high:
                    replacements[indices[0]] = QueryNode(
                        table=table, column=column, role="filter",
                        operator="IS NOT NULL", value="",
                    )
                    for idx in indices[1:]:
                        replacements[idx] = None
                    log_entries.append(("rule_abnormal", f"{key}: data [{data_min},{data_max}] entirely outside normal [{norm_low},{norm_high}] → IS NOT NULL"))
                else:
                    below_count = ctx.count_matching(table, column, "<", norm_low)
                    above_count = ctx.count_matching(table, column, ">", norm_high)
                    if below_count > 0 and above_count > 0:
                        replacements[indices[0]] = QueryNode(
                            table=table, column=column, role="filter",
                            operator="<", value=str(norm_low),
                        )
                        if len(indices) > 1:
                            replacements[indices[1]] = QueryNode(
                                table=table, column=column, role="filter",
                                operator=">", value=str(norm_high),
                            )
                        for idx in indices[2:]:
                            replacements[idx] = None
                        log_entries.append(("rule_abnormal", f"{key}: abnormal = < {norm_low} OR > {norm_high}"))
                    elif below_count > 0:
                        replacements[indices[0]] = QueryNode(
                            table=table, column=column, role="filter",
                            operator="<", value=str(norm_low),
                        )
                        for idx in indices[1:]:
                            replacements[idx] = None
                        log_entries.append(("rule_abnormal", f"{key}: abnormal = < {norm_low}"))
                    elif above_count > 0:
                        replacements[indices[0]] = QueryNode(
                            table=table, column=column, role="filter",
                            operator=">", value=str(norm_high),
                        )
                        for idx in indices[1:]:
                            replacements[idx] = None
                        log_entries.append(("rule_abnormal", f"{key}: abnormal = > {norm_high}"))

        elif is_normal_target:
            if normal_range:
                norm_low, norm_high = normal_range
                replacements[indices[0]] = QueryNode(
                    table=table, column=column, role="filter",
                    operator=">=", value=str(norm_low),
                )
                if len(indices) > 1:
                    replacements[indices[1]] = QueryNode(
                        table=table, column=column, role="filter",
                        operator="<=", value=str(norm_high),
                    )
                else:
                    additions.append(QueryNode(
                        table=table, column=column, role="filter",
                        operator="<=", value=str(norm_high),
                    ))
                for idx in indices[2:]:
                    replacements[idx] = None
                log_entries.append(("rule_normal", f"{key}: normal = [{norm_low}, {norm_high}]"))

    if not replacements and not additions:
        return None

    # Apply changes: replace in-place, then remove marked nodes
    new_nodes = list(ctx.filter_nodes)
    for idx, replacement in replacements.items():
        if replacement is not None:
            new_nodes[idx] = replacement
    # Remove None-marked nodes (reverse order to preserve indices)
    for idx in sorted((i for i, r in replacements.items() if r is None), reverse=True):
        new_nodes.pop(idx)
    new_nodes.extend(additions)

    return RuleResult(filter_nodes=new_nodes, log_entries=log_entries)


def rule_existence_pattern(ctx: EngineContext) -> RuleResult | None:
    """Detect 'have a [value]' patterns where the column is sparse.

    Trigger: numeric filter on a column with >70% nulls AND question implies
             "have" / "with" existence semantics.
    Evidence: null ratio from DB, question language.
    Action: If the threshold filters out most of what exists, relax to IS NOT NULL.
    """
    if not ctx.db_path or not ctx.output_nodes:
        return None

    existence_keywords = ("have", "has", "with", "who have", "that have", "who has")
    has_existence = any(kw in ctx.question_lower for kw in existence_keywords)
    if not has_existence:
        return None

    entity_table = ctx.entity_table
    new_nodes = list(ctx.filter_nodes)
    log_entries: list[tuple[str, str]] = []
    changed = False

    for i, n in enumerate(new_nodes):
        if n.table != entity_table or n.operator in ("=", "LIKE", "IN", "IS NOT NULL"):
            continue
        if not ctx.is_sparse_column(n.table, n.column):
            continue
        # Column is sparse — check if the filter is too strict
        # (matches very few of the already-rare non-null values)
        total_non_null = ctx.total_non_null(n.table, n.column)
        if total_non_null <= 0:
            continue
        matching = ctx.count_matching(n.table, n.column, n.operator, n.value)
        # If filter passes <10% of already-sparse data, it's likely wrong
        if matching >= 0 and matching < total_non_null * 0.1:
            new_nodes[i] = QueryNode(
                table=n.table, column=n.column, role="filter",
                operator="IS NOT NULL", value="",
            )
            changed = True
            log_entries.append(("rule_existence",
                               f"{n.table}.{n.column}: sparse ({ctx.col_null_ratio(n.table, n.column):.0%} null), "
                               f"filter matches {matching}/{total_non_null} → IS NOT NULL"))

    if not changed:
        return None
    return RuleResult(filter_nodes=new_nodes, log_entries=log_entries)


def rule_plausibility_check(ctx: EngineContext) -> RuleResult | None:
    """Data-driven validation: detect filters that produce implausible result counts.

    This is the ADAPTIVE rule — it works in any domain by asking:
    "Does the full filter combination produce a plausible number of results?"

    Trigger: comp_type=count AND result count is 0 with current filters.
    Evidence: Execute the full filter chain against DB and check result count.
    Action: Identify which filter causes the 0 and relax it.

    The logic: if applying all filters gives 0 results but dropping one filter
    gives N > 0 results, that filter is likely wrong. Replace it with IS NOT NULL
    (existence check) which is the safest relaxation.
    """
    if ctx.comp_type != "count" or not ctx.db_path:
        return None
    if not ctx.output_nodes:
        return None

    entity_table = ctx.entity_table
    detail_filters = [n for n in ctx.filter_nodes if n.table == entity_table
                      and n.operator not in ("=", "LIKE", "IS NOT NULL")]

    if len(detail_filters) < 2:
        return None

    # Build full WHERE clause and check if it produces 0
    pk_col = None
    ts = ctx.kg.get_table(entity_table)
    if ts and ts.columns:
        pk_col = ts.columns[0].name
    if not pk_col:
        pk_col = ctx.output_nodes[0].column

    try:
        conn = sqlite3.connect(str(ctx.db_path), timeout=5)

        def _build_where(filters: list[QueryNode]) -> str:
            parts = []
            for f in filters:
                if f.operator == "IS NOT NULL":
                    parts.append(f'"{f.column}" IS NOT NULL')
                else:
                    parts.append(f'"{f.column}" {f.operator} ?')
            return " AND ".join(parts)

        def _count(filters: list[QueryNode]) -> int:
            where = _build_where(filters)
            params = tuple(f.value for f in filters if f.operator != "IS NOT NULL")
            row = conn.execute(
                f'SELECT COUNT(DISTINCT "{pk_col}") FROM "{entity_table}" WHERE {where}',
                params,
            ).fetchone()
            return row[0] if row else 0

        full_count = _count(detail_filters)
        if full_count > 0:
            conn.close()
            return None  # Filters produce results — no fix needed

        # Try dropping each filter group (by column) to find the culprit
        col_groups: dict[str, list[QueryNode]] = {}
        for f in detail_filters:
            col_groups.setdefault(f.column, []).append(f)

        culprit_col: str | None = None
        best_count = 0
        for col, group in col_groups.items():
            remaining = [f for f in detail_filters if f.column != col]
            if not remaining:
                continue
            count_without = _count(remaining)
            if count_without > best_count:
                best_count = count_without
                culprit_col = col

        conn.close()

        if not culprit_col or best_count == 0:
            return None

        # Replace the culprit column's filters with IS NOT NULL
        new_nodes = list(ctx.filter_nodes)
        to_remove: list[int] = []
        replaced = False
        for i, n in enumerate(new_nodes):
            if n.table == entity_table and n.column == culprit_col and n.operator not in ("=", "LIKE", "IS NOT NULL"):
                if not replaced:
                    new_nodes[i] = QueryNode(
                        table=n.table, column=n.column, role="filter",
                        operator="IS NOT NULL", value="",
                    )
                    replaced = True
                else:
                    to_remove.append(i)
        for idx in sorted(to_remove, reverse=True):
            new_nodes.pop(idx)

        return RuleResult(
            filter_nodes=new_nodes,
            log_entries=[("rule_plausibility",
                         f"Full filter → 0 results. Dropping {culprit_col} → {best_count}. Relaxed to IS NOT NULL.")]
        )

    except Exception:
        return None


def rule_ratio_pattern(ctx: EngineContext) -> RuleResult | None:
    """Detect percentage/ratio questions and emit template.

    Trigger: comp_type contains 'ratio' or 'percentage', or question has '%'/'percent'/'ratio'.
    Evidence: population and metric from intent.
    Action: Emit CASE WHEN SQL template directive.
    """
    q = ctx.question_lower
    is_ratio = ctx.comp_type in ("ratio", "percentage")
    if not is_ratio:
        is_ratio = any(w in q for w in ("percentage", "percent", "%", "proportion", "ratio", "fraction"))
    if not is_ratio:
        return None

    # Parse population vs subset from intent
    population_text = ""
    metric_text = ""
    for line in ctx.user_intent.split("\n"):
        if "Population (WHERE):" in line:
            population_text = line.split("Population (WHERE):")[1].strip()
        elif "Metric (SELECT):" in line:
            metric_text = line.split("Metric (SELECT):")[1].strip()

    if not population_text or not metric_text:
        return None

    # Distinguish value ratio (A/B) from percentage (subset/population * 100).
    # Structural signal: if any output column is numeric (measurement like milliseconds,
    # amount, score) → comparing values, not counting rows. Skip the COUNT template.
    is_value_ratio = False
    if ctx.output_nodes and ctx.db_path:
        try:
            _conn = sqlite3.connect(str(ctx.db_path), timeout=5)
            for _node in ctx.output_nodes:
                col_lower = _node.column.lower()
                # ID/PK columns are used for counting, not value comparison
                is_id_col = (
                    col_lower == "id" or col_lower == "_id"
                    or col_lower.endswith("_id") or col_lower.endswith("id")
                )
                if is_id_col:
                    continue
                col_info = _conn.execute(f'PRAGMA table_info("{_node.table}")').fetchall()
                for ci in col_info:
                    if ci[1].lower() == col_lower:
                        # ci[5] is the pk flag — skip primary keys
                        if ci[5]:
                            break
                        col_type = (ci[2] or "").upper()
                        if col_type in ("REAL", "INTEGER", "INT", "NUMERIC", "FLOAT", "DOUBLE"):
                            is_value_ratio = True
                        break
                if is_value_ratio:
                    break
            _conn.close()
        except Exception:
            pass
        # Also check: multiple equality filter values on same column (group comparison).
        # Exclude range operators (>=, <=, >, <) which indicate a BETWEEN range, not comparison.
        if not is_value_ratio and ctx.filter_nodes:
            range_ops = {">=", "<=", ">", "<", "BETWEEN"}
            filter_cols: dict[str, int] = {}
            for fn in ctx.filter_nodes:
                if fn.operator in range_ops:
                    continue
                key = f"{fn.table}.{fn.column}"
                filter_cols[key] = filter_cols.get(key, 0) + 1
            if any(v > 1 for v in filter_cols.values()):
                is_value_ratio = True

    if is_value_ratio:
        asks_pct = any(w in q for w in ("percentage", "percent", "%"))
        if asks_pct:
            directive = (
                f"VALUE COMPARISON: This compares two numeric values — do NOT use COUNT.\n"
                f"  Base context: {population_text}\n"
                f"  Comparison: {metric_text}\n"
                f"  NULL SAFETY: When finding extremes (first/last), put the IS NOT NULL filter for the output column INSIDE the MIN/MAX subquery, not outside."
            )
            # Emit decomposition: get each value in a separate simple query, then compute
            # Find the numeric output column for decomposition
            out_col = None
            out_tbl = None
            if ctx.output_nodes and ctx.db_path:
                try:
                    _conn2 = sqlite3.connect(str(ctx.db_path), timeout=5)
                    for _nd in ctx.output_nodes:
                        _ci2 = _conn2.execute(f'PRAGMA table_info("{_nd.table}")').fetchall()
                        for _r in _ci2:
                            if _r[1].lower() == _nd.column.lower():
                                if (_r[2] or "").upper() in ("REAL", "INTEGER", "INT", "NUMERIC", "FLOAT", "DOUBLE"):
                                    out_col = _nd.column
                                    out_tbl = _nd.table
                                break
                        if out_col:
                            break
                    _conn2.close()
                except Exception:
                    pass
            if not out_col and ctx.output_nodes:
                out_col = ctx.output_nodes[0].column
                out_tbl = ctx.output_nodes[0].table
            if out_col and out_tbl:
                # Detect ordering column from filter_nodes or question context
                order_col = ""
                if ctx.filter_nodes:
                    pos_cols = {"position", "rank", "positionorder", "place", "standing"}
                    for fn in ctx.filter_nodes:
                        if fn.column.lower() in pos_cols:
                            order_col = fn.column
                            break
                if not order_col:
                    # Infer from question keywords
                    if any(w in ctx.question_lower for w in ("champion", "winner", "first", "last", "finish")):
                        if ctx.db_path:
                            try:
                                _conn3 = sqlite3.connect(str(ctx.db_path), timeout=5)
                                for _ci3 in _conn3.execute(f'PRAGMA table_info("{out_tbl}")').fetchall():
                                    if _ci3[1].lower() in ("position", "positionorder"):
                                        order_col = _ci3[1]
                                        break
                                _conn3.close()
                            except Exception:
                                pass

                first_hint = f' — use WHERE "{order_col}" = 1' if order_col else ""
                last_hint = f' — use WHERE "{order_col}" = (SELECT MAX("{order_col}") FROM ... WHERE output IS NOT NULL)' if order_col else ""
                steps = [
                    DecompositionStep(
                        description=f'SELECT "{out_col}" FROM "{out_tbl}" for the FIRST entity (champion/winner){first_hint} within: {population_text}',
                        sql_template="",
                        output_var="value_a",
                    ),
                    DecompositionStep(
                        description=f'SELECT "{out_col}" FROM "{out_tbl}" for the LAST entity (last finisher){last_hint} within: {population_text}',
                        sql_template="",
                        output_var="value_b",
                    ),
                    DecompositionStep(
                        description=f"Compute: {metric_text}",
                        sql_template="__compute__",
                        output_var="result",
                    ),
                ]
                return RuleResult(
                    sql_directives=[directive],
                    log_entries=[("rule_ratio", "Detected value comparison — emitting decomposition")],
                    decomposition=steps,
                )
        else:
            directive = (
                f"RATIO PATTERN: This is a value ratio (A / B) question — do NOT multiply by 100.\n"
                f"  Population (denominator): {population_text}\n"
                f"  Subset (numerator): {metric_text}\n"
                f"  Use: SELECT CAST(SUM(CASE WHEN [numerator_condition] THEN value END) AS REAL) / SUM(CASE WHEN [denominator_condition] THEN value END)"
            )
    else:
        directive = (
            f"RATIO PATTERN: This is a percentage/ratio question.\n"
            f"  Population (denominator): {population_text}\n"
            f"  Subset (numerator): {metric_text}\n"
            f"  Use: SELECT CAST(COUNT(CASE WHEN [subset_condition] THEN 1 END) AS REAL) * 100 / COUNT(*)"
        )
    return RuleResult(
        sql_directives=[directive],
        log_entries=[("rule_ratio", "Detected ratio pattern")]
    )


def rule_temporal_format(ctx: EngineContext) -> RuleResult | None:
    """Fix date format mismatches between filter value and actual DB format.

    Trigger: filter node with a date-like value on a TEXT column.
    Evidence: sample actual values from DB.
    Action: Adjust operator to LIKE with correct format pattern.
    """
    if not ctx.db_path:
        return None

    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    changes: list[tuple[int, QueryNode]] = []

    for i, n in enumerate(ctx.filter_nodes):
        val = str(n.value)
        if not date_pattern.match(val):
            continue
        if n.operator != "=":
            continue

        # Check what the actual format looks like in DB
        try:
            conn = sqlite3.connect(str(ctx.db_path), timeout=5)
            sample = conn.execute(
                f'SELECT "{n.column}" FROM "{n.table}" '
                f'WHERE "{n.column}" IS NOT NULL LIMIT 1'
            ).fetchone()
            conn.close()
            if sample and sample[0]:
                db_val = str(sample[0])
                # If DB has longer format (e.g. "2008-09-24 00:00:00"), use LIKE
                if len(db_val) > 10 and val in db_val:
                    changes.append((i, QueryNode(
                        table=n.table, column=n.column, role="filter",
                        operator="LIKE", value=f"%{val}%",
                    )))
        except Exception:
            pass

    if not changes:
        return None

    new_nodes = list(ctx.filter_nodes)
    for idx, node in changes:
        new_nodes[idx] = node
    return RuleResult(
        filter_nodes=new_nodes,
        log_entries=[("rule_temporal_format", f"Adjusted {len(changes)} date filters to LIKE")]
    )


def rule_count_distinct(ctx: EngineContext) -> RuleResult | None:
    """Emit COUNT(DISTINCT ...) hint for count queries with joins.

    Trigger: comp_type=count AND path has edges (joins).
    Evidence: presence of FK relationships between tables.
    Action: Emit COUNT DISTINCT directive.
    """
    if ctx.comp_type != "count":
        return None
    if not ctx.output_nodes:
        return None

    # Check if there are filters on different tables (implying a join)
    tables_in_filters = {n.table for n in ctx.filter_nodes}
    tables_in_filters.add(ctx.entity_table)
    if len(tables_in_filters) < 2:
        return None

    entity_table = ctx.entity_table
    pk_col = None
    ts = ctx.kg.get_table(entity_table)
    if ts and ts.columns:
        pk_col = ts.columns[0].name
    if not pk_col:
        pk_col = ctx.output_nodes[0].column

    directive = (
        f'COUNT HINT: Use COUNT(DISTINCT "{entity_table}"."{pk_col}") to avoid '
        f'counting duplicate rows created by the JOIN.'
    )
    return RuleResult(
        sql_directives=[directive],
        log_entries=[("rule_count_distinct", f"entity={entity_table}, pk={pk_col}")]
    )


# ---------------------------------------------------------------------------
# Knowledge Parsing Helpers + Distribution-Based Range Resolution
# ---------------------------------------------------------------------------

# Acceleration layer: known medical ranges (used as fast-path before distribution analysis)
_KNOWN_NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "wbc": (3.5, 9.0),
    "rbc": (4.0, 5.5),
    "hgb": (12.0, 17.5),
    "hct": (36.0, 50.0),
    "plt": (150.0, 400.0),
    "fg": (150.0, 400.0),
    "got": (0.0, 40.0),
    "gpt": (0.0, 40.0),
    "ldh": (0.0, 500.0),
    "alp": (44.0, 147.0),
    "tp": (6.0, 8.3),
    "alb": (3.5, 5.5),
    "ua": (2.4, 8.0),
    "un": (7.0, 20.0),
    "cre": (0.6, 1.2),
    "t-bil": (0.1, 1.2),
    "t-cho": (0.0, 250.0),
    "tg": (0.0, 150.0),
    "glu": (70.0, 100.0),
    "crp": (0.0, 0.5),
    "igg": (700.0, 1600.0),
    "iga": (70.0, 400.0),
    "igm": (40.0, 230.0),
    "c3": (90.0, 180.0),
    "c4": (10.0, 40.0),
    "pt": (11.0, 13.5),
    "aptt": (25.0, 35.0),
}


def _parse_ranges_from_knowledge(text: str, out: dict[str, tuple[float, float]]) -> None:
    """Extract normal ranges from knowledge/anchor text."""
    if not text:
        return

    for line in text.split("\n"):
        line_lower = line.lower()
        field_match = re.match(r'^-\s+(\w[\w\-]*)\s*(?:\([^)]*\))?\s*:', line)
        if not field_match:
            continue
        field_name = field_match.group(1).lower()

        # "normal range X-Y" or "X to Y" or "between X and Y"
        range_match = re.search(
            r'normal\s+(?:range\s+)?(?:is\s+)?(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)',
            line_lower
        )
        if range_match:
            out[field_name] = (float(range_match.group(1)), float(range_match.group(2)))
            continue

        # "between X and Y" for normal
        between_match = re.search(
            r'(?:normal|healthy|reference)\s+.*?(?:between|from)\s+(\d+\.?\d*)\s+(?:and|to)\s+(\d+\.?\d*)',
            line_lower
        )
        if between_match:
            out[field_name] = (float(between_match.group(1)), float(between_match.group(2)))
            continue

        # ">= X and <= Y" style
        bound_match = re.search(r'>=?\s*(\d+\.?\d*).*?<=?\s*(\d+\.?\d*)', line_lower)
        if bound_match and "normal" in line_lower:
            out[field_name] = (float(bound_match.group(1)), float(bound_match.group(2)))
            continue

        # "above X considered abnormal" → normal upper = X
        above_match = re.search(
            r'(?:above|over|greater than|>)\s*(\d+\.?\d*)\s*(?:is\s+|are\s+)?(?:considered\s+)?(?:abnormal|beyond|outside)',
            line_lower
        )
        if above_match:
            upper = float(above_match.group(1))
            if field_name not in out:
                out[field_name] = (0.0, upper)
            else:
                out[field_name] = (out[field_name][0], upper)


def _derive_range_from_distribution(
    ctx: EngineContext, table: str, column: str
) -> tuple[float, float] | None:
    """Derive "normal" range from actual data distribution using statistical signals.

    Strategy: Find where the bulk of data concentrates (the "main cluster").
    Uses percentile-based approach: P5-P95 as a proxy for "where most values live".
    Falls back to IQR-based fences if distribution is skewed.

    Returns (low, high) or None if can't determine.
    """
    if not ctx.db_path:
        return None

    try:
        conn = sqlite3.connect(str(ctx.db_path), timeout=5)

        # Get total non-null count
        total_row = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NOT NULL'
        ).fetchone()
        total = total_row[0] if total_row else 0
        if total < 10:
            conn.close()
            return None

        # Get percentiles via sorted offset (P5, P25, P50, P75, P95)
        p5_offset = max(int(total * 0.05) - 1, 0)
        p25_offset = max(int(total * 0.25) - 1, 0)
        p50_offset = max(int(total * 0.50) - 1, 0)
        p75_offset = max(int(total * 0.75) - 1, 0)
        p95_offset = max(int(total * 0.95) - 1, 0)

        def _get_pctl(offset: int) -> float | None:
            row = conn.execute(
                f'SELECT CAST("{column}" AS REAL) FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL '
                f'ORDER BY CAST("{column}" AS REAL) LIMIT 1 OFFSET {offset}'
            ).fetchone()
            return row[0] if row else None

        p5 = _get_pctl(p5_offset)
        p25 = _get_pctl(p25_offset)
        p50 = _get_pctl(p50_offset)
        p75 = _get_pctl(p75_offset)
        p95 = _get_pctl(p95_offset)
        conn.close()

        if None in (p5, p25, p50, p75, p95):
            return None

        # Strategy 1: IQR-based fences (standard outlier detection)
        iqr = p75 - p25
        if iqr > 0:
            lower_fence = p25 - 1.5 * iqr
            upper_fence = p75 + 1.5 * iqr
            # Validate: fences should exclude some data (otherwise range is too wide)
            # and include most data (otherwise too narrow)
            # Good fence: includes ~85-98% of data
            return (round(lower_fence, 4), round(upper_fence, 4))

        # Strategy 2: If IQR = 0 (many identical values), use P5-P95 directly
        if p5 != p95:
            return (round(p5, 4), round(p95, 4))

        return None
    except Exception:
        return None


def _validate_range_against_data(
    ctx: EngineContext, table: str, column: str, low: float, high: float
) -> dict[str, Any]:
    """Validate a proposed normal range against actual data distribution.

    Returns a dict with:
    - inside_count: rows within [low, high]
    - outside_count: rows outside [low, high]
    - inside_pct: percentage within range
    - below_count: rows below low
    - above_count: rows above high
    - is_useful: whether this range actually discriminates (splits data meaningfully)
    """
    if not ctx.db_path:
        return {"is_useful": False}

    try:
        conn = sqlite3.connect(str(ctx.db_path), timeout=5)
        total_row = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NOT NULL'
        ).fetchone()
        total = total_row[0] if total_row else 0
        if total == 0:
            conn.close()
            return {"is_useful": False}

        inside_row = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL AND CAST("{column}" AS REAL) >= ? AND CAST("{column}" AS REAL) <= ?',
            (low, high),
        ).fetchone()
        inside = inside_row[0] if inside_row else 0

        below_row = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL AND CAST("{column}" AS REAL) < ?',
            (low,),
        ).fetchone()
        below = below_row[0] if below_row else 0

        above_row = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL AND CAST("{column}" AS REAL) > ?',
            (high,),
        ).fetchone()
        above = above_row[0] if above_row else 0

        conn.close()

        outside = below + above
        inside_pct = (inside / total * 100) if total > 0 else 0

        # A useful range splits data: not everything is inside, not everything is outside
        # Typical "normal" range: 60-95% inside
        is_useful = 5 < inside_pct < 99 and outside > 0

        return {
            "inside_count": inside,
            "outside_count": outside,
            "inside_pct": round(inside_pct, 1),
            "below_count": below,
            "above_count": above,
            "total": total,
            "is_useful": is_useful,
        }
    except Exception:
        return {"is_useful": False}


def _llm_infer_normal_range(
    ctx: EngineContext,
    table: str,
    column: str,
    stats: dict[str, Any],
    distribution_range: tuple[float, float] | None,
    model_call: Callable,
) -> tuple[float, float] | None:
    """Ask LLM to determine normal range given data distribution context.

    Only called when:
    1. No knowledge text defines the range
    2. No hardcoded range exists
    3. Distribution analysis provides a candidate but needs semantic validation
    """
    import json

    dist_desc = ""
    if distribution_range:
        validation = _validate_range_against_data(ctx, table, column, *distribution_range)
        dist_desc = (
            f"DISTRIBUTION-DERIVED RANGE: [{distribution_range[0]}, {distribution_range[1]}]\n"
            f"  Inside: {validation.get('inside_count', '?')}/{validation.get('total', '?')} "
            f"({validation.get('inside_pct', '?')}%)\n"
            f"  Below: {validation.get('below_count', '?')}, Above: {validation.get('above_count', '?')}\n"
        )

    prompt = f"""Given a data column, determine what constitutes "normal" values.

COLUMN: {table}.{column}
DATA STATISTICS:
  Min: {stats.get('min')}, Max: {stats.get('max')}, Avg: {stats.get('avg')}
  Distinct values: {stats.get('distinct')}, Total records: {stats.get('count')}

{dist_desc}
QUESTION CONTEXT: {ctx.question}
KNOWLEDGE AVAILABLE: {ctx.knowledge_text[:500] if ctx.knowledge_text else 'None'}

TASK: Determine the normal/healthy/expected range for this column.
- If you recognize this as a known measurement (lab value, metric, score), use domain knowledge.
- If unknown, use the distribution: "normal" = where 60-90% of values concentrate (the main cluster).
- The range must split the data meaningfully: some values inside, some outside.

Respond ONLY with JSON:
{{"low": <number>, "high": <number>, "confidence": "high|medium|low", "reasoning": "brief"}}

If you cannot determine a meaningful range, respond: {{"low": null, "high": null, "confidence": "none", "reasoning": "why"}}
"""

    try:
        from data_agent_baseline.agents.model import ModelMessage
        messages = [ModelMessage(role="user", content=prompt)]
        raw = model_call(messages)
        json_match = re.search(r'\{[^{}]*\}', raw)
        if not json_match:
            return None
        result = json.loads(json_match.group())
        low = result.get("low")
        high = result.get("high")
        confidence = result.get("confidence", "none")
        if low is None or high is None or confidence == "none":
            return None
        low, high = float(low), float(high)
        if low >= high:
            return None
        # Validate: the LLM-proposed range must actually split the data
        validation = _validate_range_against_data(ctx, table, column, low, high)
        if not validation.get("is_useful"):
            return None
        return (low, high)
    except Exception:
        return None


def _resolve_normal_range(
    ctx: EngineContext,
    table: str,
    column: str,
    model_call: Callable | None = None,
) -> tuple[float, float] | None:
    """Full cascade to determine "normal" range for a column.

    Priority:
    1. Knowledge text (explicit domain documentation)
    2. Hardcoded accelerator (_KNOWN_NORMAL_RANGES)
    3. Data distribution analysis (statistical, domain-agnostic)
    4. LLM inference (given distribution + context)

    Each tier validates against actual data before accepting.
    """
    col_lower = column.lower()

    # Tier 1: Knowledge text (highest priority — explicit domain docs)
    knowledge_ranges: dict[str, tuple[float, float]] = {}
    _parse_ranges_from_knowledge(ctx.anchor_text, knowledge_ranges)
    _parse_ranges_from_knowledge(ctx.knowledge_text, knowledge_ranges)
    if col_lower in knowledge_ranges:
        r = knowledge_ranges[col_lower]
        validation = _validate_range_against_data(ctx, table, column, r[0], r[1])
        if validation.get("is_useful") or validation.get("outside_count", 0) > 0:
            return r

    # Tier 2: Hardcoded accelerator (fast-path for known domains)
    if col_lower in _KNOWN_NORMAL_RANGES:
        r = _KNOWN_NORMAL_RANGES[col_lower]
        validation = _validate_range_against_data(ctx, table, column, r[0], r[1])
        if validation.get("is_useful") or validation.get("outside_count", 0) > 0:
            return r

    # Tier 3: Distribution analysis (domain-agnostic)
    dist_range = _derive_range_from_distribution(ctx, table, column)
    if dist_range:
        validation = _validate_range_against_data(ctx, table, column, *dist_range)
        if validation.get("is_useful"):
            # Distribution range is useful on its own — no need for LLM
            return dist_range

    # Tier 4: LLM inference (given distribution context)
    if model_call:
        stats = ctx.col_stats(table, column)
        if stats:
            llm_range = _llm_infer_normal_range(
                ctx, table, column, stats, dist_range, model_call
            )
            if llm_range:
                return llm_range

    # Fallback: distribution range even if not perfectly "useful"
    # (some data is better than no data)
    if dist_range:
        return dist_range

    return None


def _col_mentioned_in(col_lower: str, text: str) -> bool:
    """Check if a column name is semantically referenced in text."""
    if col_lower in text:
        return True
    # Common abbreviation expansions
    expansions = {
        "wbc": ["white blood cell", "white blood"],
        "rbc": ["red blood cell", "red blood"],
        "hgb": ["hemoglobin"],
        "hct": ["hematocrit"],
        "plt": ["platelet"],
        "fg": ["fibrinogen"],
        "got": ["glutamic oxaloacetic"],
        "gpt": ["glutamic pyruvic"],
        "ldh": ["lactate dehydrogenase"],
        "alp": ["alkaline phosphatase"],
        "tp": ["total protein"],
        "alb": ["albumin"],
        "ua": ["uric acid"],
        "un": ["urea nitrogen"],
        "cre": ["creatinine"],
        "t-bil": ["bilirubin"],
        "t-cho": ["cholesterol"],
        "tg": ["triglyceride"],
        "glu": ["glucose"],
        "crp": ["c-reactive protein"],
        "pt": ["prothrombin time"],
        "aptt": ["activated partial thromboplastin"],
    }
    for expansion in expansions.get(col_lower, []):
        if expansion in text:
            return True
    return False


def _col_near_keyword(col_lower: str, keyword: str, question: str) -> bool:
    """Check if column name appears near a keyword in the question text."""
    # Find all positions of keyword
    kw_positions = [m.start() for m in re.finditer(keyword, question)]
    if not kw_positions:
        return False

    # Check if column (or its expansion) appears within 50 chars of keyword
    expansions = {
        "wbc": ["white blood cell", "white blood"],
        "rbc": ["red blood cell", "red blood"],
        "fg": ["fibrinogen"],
        "plt": ["platelet"],
        "hgb": ["hemoglobin"],
    }
    targets = [col_lower] + expansions.get(col_lower, [])

    for target in targets:
        positions = [m.start() for m in re.finditer(re.escape(target), question)]
        for tp in positions:
            for kp in kw_positions:
                if abs(tp - kp) < 60:
                    return True
    return False


# ---------------------------------------------------------------------------
# Engine: runs all rules in order, merges results
# ---------------------------------------------------------------------------


# Ordered list of rules (priority: earlier runs first, later can override)
RULES: list[Callable[[EngineContext], RuleResult | None]] = [
    rule_normal_abnormal_resolution,
    rule_impossible_threshold,
    rule_coverage_simplification,
    rule_plausibility_check,
    rule_temporal_format,
    rule_cross_row_structure,
    rule_ratio_pattern,
    rule_existence_pattern,
]


@dataclass(slots=True)
class EngineOutput:
    """Final merged output from all rules."""
    filter_nodes: list[QueryNode]
    sql_directives: list[str]
    log_entries: list[tuple[str, str]]
    decomposition: list[DecompositionStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adaptive Layer: Hypothesis-driven reasoning with multi-signal plausibility
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Hypothesis:
    """A candidate interpretation of the query filters."""
    label: str
    filter_nodes: list[QueryNode]
    sql_directives: list[str] = field(default_factory=list)
    score: float = 0.0
    result_count: int = -1
    reasoning: str = ""


@dataclass(slots=True)
class PlausibilityReport:
    """Assessment of whether current filters produce a plausible result."""
    is_plausible: bool
    result_count: int = -1
    entity_count: int = -1
    anomaly: str = ""  # What specifically is wrong


def _probe_result_count(ctx: EngineContext, filters: list[QueryNode] | None = None) -> int:
    """Execute the filter chain and count distinct entities, including join-table filters."""
    if not ctx.db_path:
        return -1
    entity_table = ctx.entity_table
    if not entity_table:
        return -1

    use_filters = filters if filters is not None else ctx.filter_nodes
    detail_filters = [n for n in use_filters if n.table == entity_table]
    if not detail_filters:
        return -1

    ts = ctx.kg.get_table(entity_table)
    pk_col = ts.columns[0].name if ts and ts.columns else "ID"

    # Separate filters on joined tables
    join_filters: dict[str, list[QueryNode]] = {}
    for n in use_filters:
        if n.table != entity_table:
            join_filters.setdefault(n.table, []).append(n)

    try:
        conn = sqlite3.connect(str(ctx.db_path), timeout=5)
        parts = []
        params: list[Any] = []
        joins: list[str] = []

        # Build JOIN clauses for filters on other tables
        for other_table, other_nodes in join_filters.items():
            fk_edges = ctx.kg.graph.get_fk_between(entity_table, other_table) if ctx.kg.graph else []
            if not fk_edges:
                continue
            edge = fk_edges[0]
            if edge.src.split(".")[0] == entity_table:
                src_col = edge.src.split(".")[1]
                dst_col = edge.dst.split(".")[1]
            else:
                src_col = edge.dst.split(".")[1]
                dst_col = edge.src.split(".")[1]
            joins.append(
                f'JOIN "{other_table}" ON "{entity_table}"."{src_col}" = "{other_table}"."{dst_col}"'
            )
            for f in other_nodes:
                if f.column.startswith("_expr:"):
                    expr_sql = f.column[len("_expr:"):]
                    parts.append(f'{expr_sql} {f.operator} ?')
                    params.append(f.value)
                    continue
                val_str = str(f.value).strip()
                if val_str.lstrip("(").lower().startswith("select"):
                    continue
                if f.operator == "IS NOT NULL":
                    parts.append(f'"{other_table}"."{f.column}" IS NOT NULL')
                elif f.operator.upper() == "IN":
                    vals = re.findall(r"'([^']*)'", val_str)
                    if vals:
                        placeholders = ",".join("?" * len(vals))
                        parts.append(f'"{other_table}"."{f.column}" IN ({placeholders})')
                        params.extend(vals)
                elif f.operator in ("=", "LIKE"):
                    parts.append(f'"{other_table}"."{f.column}" {f.operator} ?')
                    params.append(f.value)
                else:
                    parts.append(f'"{other_table}"."{f.column}" {f.operator} ?')
                    params.append(f.value)

        # Entity table filters
        for f in detail_filters:
            if f.column.startswith("_expr:"):
                expr_sql = f.column[len("_expr:"):]
                parts.append(f'{expr_sql} {f.operator} ?')
                params.append(f.value)
                continue
            val_str = str(f.value).strip()
            if val_str.lstrip("(").lower().startswith("select"):
                return -1
            if f.operator == "IS NOT NULL":
                parts.append(f'"{entity_table}"."{f.column}" IS NOT NULL')
            elif f.operator.upper() == "IN":
                vals = re.findall(r"'([^']*)'", val_str)
                if vals:
                    placeholders = ",".join("?" * len(vals))
                    parts.append(f'"{entity_table}"."{f.column}" IN ({placeholders})')
                    params.extend(vals)
            elif f.operator in ("=", "LIKE"):
                parts.append(f'"{entity_table}"."{f.column}" {f.operator} ?')
                params.append(f.value)
            else:
                parts.append(f'"{entity_table}"."{f.column}" {f.operator} ?')
                params.append(f.value)

        where = " AND ".join(parts)
        join_sql = " ".join(joins)
        row = conn.execute(
            f'SELECT COUNT(DISTINCT "{entity_table}"."{pk_col}") FROM "{entity_table}" {join_sql} WHERE {where}',
            tuple(params),
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return -1


def _assess_plausibility(ctx: EngineContext) -> PlausibilityReport:
    """Multi-signal plausibility check — goes beyond just zero-result detection."""
    result_count = _probe_result_count(ctx)

    # Signal 1: Zero results
    if result_count == 0:
        return PlausibilityReport(
            is_plausible=False, result_count=0,
            anomaly="Filter combination produces 0 results — at least one condition is wrong."
        )

    # Signal 2: Singular question but multiple results
    answer_shape = ""
    for line in ctx.user_intent.split("\n"):
        if "Answer shape:" in line:
            answer_shape = line.split("Answer shape:")[1].strip().lower()
    if answer_shape == "single_value" and ctx.comp_type == "simple_lookup" and result_count > 5:
        return PlausibilityReport(
            is_plausible=False, result_count=result_count,
            anomaly=f"Question expects a single answer but filters match {result_count} rows — filters are too broad or missing a constraint."
        )

    # Signal 3: Count that seems implausibly high (>80% of all entities)
    if ctx.comp_type == "count":
        entity_table = ctx.entity_table
        ts = ctx.kg.get_table(entity_table)
        if ts and ts.row_count:
            pk_col = ts.columns[0].name if ts.columns else "ID"
            try:
                conn = sqlite3.connect(str(ctx.db_path), timeout=5)
                total_entities = conn.execute(
                    f'SELECT COUNT(DISTINCT "{pk_col}") FROM "{entity_table}"'
                ).fetchone()[0]
                conn.close()
                if total_entities > 0 and result_count > total_entities * 0.8:
                    # Check if this is suspicious — multiple restrictive conditions should narrow
                    restrictive = [n for n in ctx.filter_nodes
                                   if n.table == entity_table and n.operator not in ("IS NOT NULL",)]
                    if len(restrictive) >= 2:
                        return PlausibilityReport(
                            is_plausible=False, result_count=result_count,
                            entity_count=total_entities,
                            anomaly=f"Count={result_count}/{total_entities} entities despite {len(restrictive)} restrictive filters — conditions may be too loose or semantically inverted."
                        )
            except Exception:
                pass

    # Signal 4 removed: result_count < 0 only occurs when -1 (unevaluable),
    # which is not meaningful for ratio anomaly detection.

    return PlausibilityReport(is_plausible=True, result_count=result_count)


def _gather_evidence(ctx: EngineContext) -> dict[str, Any]:
    """Collect all data signals for hypothesis generation."""
    evidence: dict[str, Any] = {}
    entity_table = ctx.entity_table

    # Per-filter evidence with rich signals
    filter_evidence: list[dict[str, Any]] = []
    for n in ctx.filter_nodes:
        fe: dict[str, Any] = {
            "table": n.table, "column": n.column,
            "operator": n.operator, "value": str(n.value),
        }
        if n.operator not in ("=", "LIKE", "IS NOT NULL"):
            stats = ctx.col_stats(n.table, n.column)
            if stats:
                fe["data_min"] = stats.get("min")
                fe["data_max"] = stats.get("max")
                fe["data_avg"] = stats.get("avg")
                fe["distinct_values"] = stats.get("distinct")
                fe["total_non_null"] = stats.get("count")
            matching = ctx.count_matching(n.table, n.column, n.operator, n.value)
            fe["rows_matching"] = matching
            fe["null_ratio"] = round(ctx.col_null_ratio(n.table, n.column), 2)
            fe["is_sparse"] = ctx.is_sparse_column(n.table, n.column)
            fe["spread"] = ctx.value_spread(n.table, n.column)
            fe["cardinality_ratio"] = round(ctx.cardinality_ratio(n.table, n.column), 3)
        elif n.operator == "IS NOT NULL":
            fe["null_ratio"] = round(ctx.col_null_ratio(n.table, n.column), 2)
            fe["total_non_null"] = ctx.total_non_null(n.table, n.column)
        filter_evidence.append(fe)
    evidence["filters"] = filter_evidence

    # Entity table structure
    if entity_table:
        ts = ctx.kg.get_table(entity_table)
        pk_col = ts.columns[0].name if ts and ts.columns else "ID"
        evidence["entity_table"] = entity_table
        evidence["pk_col"] = pk_col
        evidence["rows_per_entity"] = round(ctx.rows_per_entity(entity_table, pk_col), 1)
        evidence["total_rows"] = ts.row_count if ts else 0
        # Total distinct entities
        if ctx.db_path:
            try:
                conn = sqlite3.connect(str(ctx.db_path), timeout=5)
                total = conn.execute(
                    f'SELECT COUNT(DISTINCT "{pk_col}") FROM "{entity_table}"'
                ).fetchone()
                evidence["total_entities"] = total[0] if total else 0
                conn.close()
            except Exception:
                evidence["total_entities"] = 0

    # Co-occurrence between filter columns (for cross-row detection)
    if entity_table:
        filter_cols = list({n.column for n in ctx.filter_nodes if n.table == entity_table})
        if len(filter_cols) >= 2:
            ts = ctx.kg.get_table(entity_table)
            pk_col = ts.columns[0].name if ts and ts.columns else "ID"
            same, across = ctx.co_occurrence(entity_table, pk_col, filter_cols[0], filter_cols[1])
            evidence["co_occurrence"] = {
                "cols": filter_cols[:2],
                "same_row": same,
                "across_rows": across,
                "needs_cross_row": same < across,
            }

    # Drop-one analysis: which filter is the bottleneck?
    if entity_table:
        drop_one: dict[str, int] = {}
        detail_filters = [n for n in ctx.filter_nodes if n.table == entity_table]
        if len(detail_filters) >= 2:
            col_groups: dict[str, list[QueryNode]] = {}
            for f in detail_filters:
                col_groups.setdefault(f.column, []).append(f)
            for drop_col in col_groups:
                remaining = [f for f in detail_filters if f.column != drop_col]
                if remaining:
                    count = _probe_result_count(ctx, remaining + [
                        n for n in ctx.filter_nodes if n.table != entity_table
                    ])
                    drop_one[drop_col] = count
        evidence["drop_one_analysis"] = drop_one

    return evidence


def _generate_hypotheses(
    ctx: EngineContext,
    evidence: dict[str, Any],
    anomaly: str,
) -> list[Hypothesis]:
    """Generate candidate interpretations based on data signals.

    Domain-agnostic templates covering common analytics patterns:
    structural, threshold, semantic, temporal, statistical, relational.
    """
    hypotheses: list[Hypothesis] = []
    entity_table = ctx.entity_table

    # H0: Current filters are correct (baseline)
    hypotheses.append(Hypothesis(
        label="current",
        filter_nodes=list(ctx.filter_nodes),
        reasoning="Keep filters as-is",
    ))

    # ---------------------------------------------------------------
    # STRUCTURAL TEMPLATES (how filters combine)
    # ---------------------------------------------------------------

    # T1: Relax bottleneck — drop-one analysis identifies the culprit
    drop_one = evidence.get("drop_one_analysis", {})
    if drop_one:
        worst_col = max(drop_one, key=drop_one.get)
        if drop_one[worst_col] > 0:
            relaxed_nodes = []
            replaced = False
            for n in ctx.filter_nodes:
                if n.table == entity_table and n.column == worst_col and n.operator not in ("=", "LIKE", "IS NOT NULL"):
                    if not replaced:
                        relaxed_nodes.append(QueryNode(
                            table=n.table, column=n.column, role="filter",
                            operator="IS NOT NULL", value="",
                        ))
                        replaced = True
                else:
                    relaxed_nodes.append(n)
            hypotheses.append(Hypothesis(
                label=f"relax_{worst_col}",
                filter_nodes=relaxed_nodes,
                reasoning=f"Relax {worst_col} to IS NOT NULL (drop-one shows {drop_one[worst_col]} results without it)",
            ))

    # T2: OR instead of AND — two conditions on same column that individually
    # match rows but together match 0 (e.g. col < X AND col > Y where X < Y)
    col_groups: dict[str, list[tuple[int, QueryNode]]] = {}
    for i, n in enumerate(ctx.filter_nodes):
        if n.table == entity_table and n.operator not in ("LIKE", "IS NOT NULL"):
            col_groups.setdefault(n.column, []).append((i, n))
    for col, group in col_groups.items():
        if len(group) < 2:
            continue
        # Detect impossible equality AND: col = 'X' AND col = 'Y' (different values → must be OR)
        eq_nodes = [(i, n) for i, n in group if n.operator == "="]
        if len(eq_nodes) >= 2:
            values = [str(n.value) for _, n in eq_nodes]
            if len(set(values)) > 1:
                in_list = ", ".join(f"'{v}'" for v in values)
                keep_indices = {i for i, _ in eq_nodes}
                or_nodes = [n for j, n in enumerate(ctx.filter_nodes) if j not in keep_indices]
                or_nodes.append(QueryNode(
                    table=entity_table, column=col, role="filter",
                    operator="IN", value=f"({in_list})",
                ))
                hypotheses.append(Hypothesis(
                    label=f"equality_or_{col}",
                    filter_nodes=or_nodes,
                    sql_directives=[
                        f'OR FILTER: "{col}" has multiple values connected by OR. '
                        f'Use: WHERE "{col}" IN ({in_list})'
                    ],
                    reasoning=f"col = '{values[0]}' AND col = '{values[1]}' is impossible — use IN ({in_list})",
                ))
        if len(group) != 2:
            continue
        (i0, n0), (i1, n1) = group
        # Detect impossible AND: < X AND > Y where X <= Y
        if n0.operator in ("<", "<=") and n1.operator in (">", ">="):
            try:
                v0, v1 = float(n0.value), float(n1.value)
                if v0 <= v1:
                    # These can't both be true simultaneously — likely meant OR
                    hypotheses.append(Hypothesis(
                        label=f"or_instead_of_and_{col}",
                        filter_nodes=list(ctx.filter_nodes),
                        sql_directives=[
                            f'FILTER LOGIC HINT: Conditions on "{col}" should use OR (not AND): '
                            f'"{col}" {n0.operator} {n0.value} OR "{col}" {n1.operator} {n1.value}'
                        ],
                        reasoning=f"{col} {n0.operator} {v0} AND {col} {n1.operator} {v1} is impossible — likely OR (outside range)",
                    ))
            except (ValueError, TypeError):
                pass
        # Detect redundant AND: >= X AND >= Y (should be just the tighter one)
        elif n0.operator == n1.operator:
            try:
                v0, v1 = float(n0.value), float(n1.value)
                tighter_idx = i0 if (n0.operator in (">=", ">") and v0 >= v1) or (n0.operator in ("<=", "<") and v0 <= v1) else i1
                remove_idx = i1 if tighter_idx == i0 else i0
                deduped = [n for j, n in enumerate(ctx.filter_nodes) if j != remove_idx]
                hypotheses.append(Hypothesis(
                    label=f"dedup_{col}",
                    filter_nodes=deduped,
                    reasoning=f"Redundant duplicate direction on {col} — keep only the tighter bound",
                ))
            except (ValueError, TypeError):
                pass

    # T3: Widen narrow equality — exact match on continuous column misses rows
    for i, n in enumerate(ctx.filter_nodes):
        if n.table != entity_table or n.operator != "=":
            continue
        fe = next((f for f in evidence.get("filters", []) if f["column"] == n.column), None)
        if not fe or "data_min" not in fe:
            continue
        # If equality produces 0 or very few matches on a wide-spread column
        matching = ctx.count_matching(n.table, n.column, "=", n.value)
        if matching <= 1 and (fe.get("distinct_values") or 0) > 20:
            try:
                val = float(n.value)
                data_range = fe["data_max"] - fe["data_min"]
                tolerance = data_range * 0.05  # 5% band
                widened = list(ctx.filter_nodes)
                widened[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                       operator=">=", value=str(val - tolerance))
                widened.insert(i + 1, QueryNode(table=n.table, column=n.column, role="filter",
                                                 operator="<=", value=str(val + tolerance)))
                hypotheses.append(Hypothesis(
                    label=f"widen_{n.column}",
                    filter_nodes=widened,
                    reasoning=f"Exact match on continuous {n.column}={val} too narrow — widen to ±5% band",
                ))
            except (ValueError, TypeError):
                pass

    # ---------------------------------------------------------------
    # THRESHOLD TEMPLATES (value boundary decisions)
    # ---------------------------------------------------------------

    # T4: Invert threshold semantics (normal↔abnormal, high↔low)
    for i, n in enumerate(ctx.filter_nodes):
        if n.table != entity_table or n.operator in ("=", "LIKE", "IS NOT NULL"):
            continue
        col_lower = n.column.lower()
        # Use full resolution cascade (no LLM in hypothesis gen — keep fast)
        normal_range = _resolve_normal_range(ctx, n.table, n.column, model_call=None)
        if not normal_range:
            continue

        norm_low, norm_high = normal_range
        try:
            val = float(n.value)
        except (ValueError, TypeError):
            continue

        if n.operator == ">=" and abs(val - norm_low) < norm_low * 0.5:
            inverted = list(ctx.filter_nodes)
            inverted[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                    operator="<", value=str(norm_low))
            for j, m in enumerate(inverted):
                if m.table == entity_table and m.column == n.column and m.operator == "<=":
                    inverted[j] = QueryNode(table=m.table, column=m.column, role="filter",
                                            operator=">", value=str(norm_high))
                    break
            hypotheses.append(Hypothesis(
                label=f"invert_{col_lower}_to_abnormal",
                filter_nodes=inverted,
                reasoning=f"Invert {col_lower}: normal [{norm_low},{norm_high}] → abnormal (< {norm_low} OR > {norm_high})",
            ))
        elif n.operator in ("<", ">"):
            inverted = list(ctx.filter_nodes)
            inverted[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                    operator=">=", value=str(norm_low))
            for j, m in enumerate(inverted):
                if j != i and m.table == entity_table and m.column == n.column and m.operator in ("<", ">"):
                    inverted[j] = QueryNode(table=m.table, column=m.column, role="filter",
                                            operator="<=", value=str(norm_high))
                    break
            else:
                inverted.append(QueryNode(table=entity_table, column=n.column, role="filter",
                                          operator="<=", value=str(norm_high)))
            hypotheses.append(Hypothesis(
                label=f"invert_{col_lower}_to_normal",
                filter_nodes=inverted,
                reasoning=f"Invert {col_lower}: abnormal → normal [{norm_low},{norm_high}]",
            ))

    # T5: Data-boundary threshold — use actual data percentiles as boundary
    # (domain-agnostic: "high" = top quartile, "low" = bottom quartile)
    q = ctx.question_lower
    has_high_low = any(w in q for w in ("high", "above average", "elevated", "exceeds", "greater than normal"))
    has_low = any(w in q for w in ("low", "below average", "decreased", "deficient", "less than normal"))
    if has_high_low or has_low:
        for i, n in enumerate(ctx.filter_nodes):
            if n.table != entity_table or n.operator in ("=", "LIKE", "IS NOT NULL"):
                continue
            col_lower = n.column.lower()
            if _resolve_normal_range(ctx, n.table, n.column, model_call=None):
                continue  # Already handled by T4
            fe = next((f for f in evidence.get("filters", []) if f["column"] == n.column), None)
            if not fe or "data_avg" not in fe:
                continue
            avg = fe["data_avg"]
            if avg is None:
                continue
            # "High" = above average, "Low" = below average
            if has_high_low and _col_mentioned_in(col_lower, q):
                boundary_nodes = list(ctx.filter_nodes)
                boundary_nodes[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                               operator=">", value=str(round(avg, 2)))
                hypotheses.append(Hypothesis(
                    label=f"above_avg_{n.column}",
                    filter_nodes=boundary_nodes,
                    reasoning=f"{n.column} 'high' = above data average ({avg:.2f})",
                ))
            elif has_low and _col_mentioned_in(col_lower, q):
                boundary_nodes = list(ctx.filter_nodes)
                boundary_nodes[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                               operator="<", value=str(round(avg, 2)))
                hypotheses.append(Hypothesis(
                    label=f"below_avg_{n.column}",
                    filter_nodes=boundary_nodes,
                    reasoning=f"{n.column} 'low' = below data average ({avg:.2f})",
                ))

    # T6: Flip comparison direction — operator is reversed (< should be >, etc.)
    for i, n in enumerate(ctx.filter_nodes):
        if n.table != entity_table or n.operator in ("=", "LIKE", "IS NOT NULL"):
            continue
        matching = ctx.count_matching(n.table, n.column, n.operator, n.value)
        if matching == 0:
            flipped_op = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}.get(n.operator)
            if flipped_op:
                flipped_count = ctx.count_matching(n.table, n.column, flipped_op, n.value)
                if flipped_count > 0:
                    flipped = list(ctx.filter_nodes)
                    flipped[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                           operator=flipped_op, value=n.value)
                    hypotheses.append(Hypothesis(
                        label=f"flip_{n.column}",
                        filter_nodes=flipped,
                        reasoning=f"{n.column} {n.operator} {n.value} → 0 rows, but {flipped_op} → {flipped_count} rows",
                    ))

    # ---------------------------------------------------------------
    # SEMANTIC TEMPLATES (meaning interpretation)
    # ---------------------------------------------------------------

    # T7: Existence semantics for sparse columns
    for fe in evidence.get("filters", []):
        if fe.get("is_sparse") and fe["operator"] not in ("IS NOT NULL",):
            col = fe["column"]
            sparse_nodes = []
            replaced = False
            for n in ctx.filter_nodes:
                if n.table == entity_table and n.column == col and not replaced:
                    sparse_nodes.append(QueryNode(
                        table=n.table, column=n.column, role="filter",
                        operator="IS NOT NULL", value="",
                    ))
                    replaced = True
                elif n.table == entity_table and n.column == col:
                    continue
                else:
                    sparse_nodes.append(n)
            if replaced:
                hypotheses.append(Hypothesis(
                    label=f"existence_{col}",
                    filter_nodes=sparse_nodes,
                    reasoning=f"{col} is sparse ({fe['null_ratio']*100:.0f}% null) — use existence check",
                ))

    # T8: Non-zero semantics — "have X" where X is often 0 means > 0
    has_have = any(w in q for w in ("have", "has", "with", "who have", "that have"))
    if has_have:
        for i, n in enumerate(ctx.filter_nodes):
            if n.table != entity_table or n.operator in ("=", "LIKE", "IS NOT NULL"):
                continue
            fe = next((f for f in evidence.get("filters", []) if f["column"] == n.column), None)
            if not fe or "data_min" not in fe:
                continue
            # If column has min=0 and many 0s, "have" might mean > 0
            if fe["data_min"] == 0:
                zero_count = ctx.count_matching(n.table, n.column, "=", 0)
                total = fe.get("total_non_null") or 0
                if total > 0 and zero_count > total * 0.3:
                    nonzero_nodes = list(ctx.filter_nodes)
                    nonzero_nodes[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                                  operator=">", value="0")
                    hypotheses.append(Hypothesis(
                        label=f"nonzero_{n.column}",
                        filter_nodes=nonzero_nodes,
                        reasoning=f"'have {n.column}' means > 0 (30%+ of values are zero)",
                    ))

    # T9: Boolean/flag interpretation — column is categorical with 2 values
    for i, n in enumerate(ctx.filter_nodes):
        if n.table != entity_table or n.operator not in (">", ">=", "<", "<="):
            continue
        fe = next((f for f in evidence.get("filters", []) if f["column"] == n.column), None)
        if not fe:
            continue
        distinct = fe.get("distinct_values") or 0
        if distinct == 2:
            # Binary column treated as numeric — should probably use = with one of the values
            stats = ctx.col_stats(n.table, n.column)
            if stats and stats.get("min") is not None:
                # Try both values
                for val in [stats["min"], stats["max"]]:
                    eq_nodes = list(ctx.filter_nodes)
                    eq_nodes[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                            operator="=", value=str(int(val) if val == int(val) else val))
                    count = _probe_result_count(ctx, eq_nodes)
                    if count > 0:
                        hypotheses.append(Hypothesis(
                            label=f"flag_{n.column}_{int(val)}",
                            filter_nodes=eq_nodes,
                            reasoning=f"{n.column} is binary (2 values) — treat as flag = {int(val)}",
                        ))
                        break  # Only add the first viable flag interpretation

    # ---------------------------------------------------------------
    # STATISTICAL TEMPLATES (distribution-aware)
    # ---------------------------------------------------------------

    # T10: Outlier boundary — use IQR-style detection
    # "Abnormal" in unknown domains = outside 1.5*IQR from median
    if "abnormal" in q or "outlier" in q or "unusual" in q or "atypical" in q:
        for i, n in enumerate(ctx.filter_nodes):
            if n.table != entity_table or n.operator in ("=", "LIKE", "IS NOT NULL"):
                continue
            col_lower = n.column.lower()
            if _resolve_normal_range(ctx, n.table, n.column, model_call=None):
                continue  # Has a known/derived range — handled by T4
            fe = next((f for f in evidence.get("filters", []) if f["column"] == n.column), None)
            if not fe or fe.get("data_min") is None or fe.get("data_max") is None or fe.get("data_avg") is None:
                continue
            # Approximate IQR: use data range / 4 as quartile proxy
            data_min, data_max, data_avg = fe["data_min"], fe["data_max"], fe["data_avg"]
            iqr_approx = (data_max - data_min) / 4.0
            q1 = data_avg - iqr_approx
            q3 = data_avg + iqr_approx
            lower_fence = q1 - 1.5 * iqr_approx
            upper_fence = q3 + 1.5 * iqr_approx
            # Check if any data falls outside fences
            below = ctx.count_matching(n.table, n.column, "<", lower_fence)
            above = ctx.count_matching(n.table, n.column, ">", upper_fence)
            if below > 0 or above > 0:
                outlier_nodes = list(ctx.filter_nodes)
                if below > 0 and above > 0:
                    outlier_nodes[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                                  operator="<", value=str(round(lower_fence, 2)))
                    outlier_nodes.append(QueryNode(table=n.table, column=n.column, role="filter",
                                                    operator=">", value=str(round(upper_fence, 2))))
                    hypotheses.append(Hypothesis(
                        label=f"outlier_{n.column}",
                        filter_nodes=outlier_nodes,
                        sql_directives=[f'FILTER LOGIC HINT: "{n.column}" < {lower_fence:.2f} OR "{n.column}" > {upper_fence:.2f}'],
                        reasoning=f"Outlier detection via IQR: {n.column} outside [{lower_fence:.1f}, {upper_fence:.1f}]",
                    ))
                elif above > 0:
                    outlier_nodes[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                                  operator=">", value=str(round(upper_fence, 2)))
                    hypotheses.append(Hypothesis(
                        label=f"outlier_high_{n.column}",
                        filter_nodes=outlier_nodes,
                        reasoning=f"Outlier high: {n.column} > {upper_fence:.1f} (IQR fence)",
                    ))

    # T11: Percentile-based threshold — use actual data quantile
    # For "top X%", "bottom X%", "above Nth percentile"
    pct_match = re.search(r'(?:top|bottom|above|below)\s*(\d+)\s*%', q)
    if pct_match:
        pct = int(pct_match.group(1))
        is_top = "top" in q or "above" in q
        for i, n in enumerate(ctx.filter_nodes):
            if n.table != entity_table or n.operator in ("=", "LIKE", "IS NOT NULL"):
                continue
            fe = next((f for f in evidence.get("filters", []) if f["column"] == n.column), None)
            if not fe or "data_min" not in fe:
                continue
            # Approximate percentile from min/max
            data_min, data_max = fe["data_min"], fe["data_max"]
            if is_top:
                threshold = data_min + (data_max - data_min) * (1 - pct / 100.0)
                pct_nodes = list(ctx.filter_nodes)
                pct_nodes[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                          operator=">=", value=str(round(threshold, 2)))
            else:
                threshold = data_min + (data_max - data_min) * (pct / 100.0)
                pct_nodes = list(ctx.filter_nodes)
                pct_nodes[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                          operator="<=", value=str(round(threshold, 2)))
            hypotheses.append(Hypothesis(
                label=f"pct_{pct}_{n.column}",
                filter_nodes=pct_nodes,
                reasoning=f"{'Top' if is_top else 'Bottom'} {pct}% of {n.column} ≈ threshold {threshold:.2f}",
            ))

    # ---------------------------------------------------------------
    # RELATIONAL TEMPLATES (how tables connect)
    # ---------------------------------------------------------------

    # T12: Cross-table filter migration — filter might belong on a different table
    # If a filter on the entity table produces 0 but similar column exists in joined table
    for i, n in enumerate(ctx.filter_nodes):
        if n.table != entity_table or n.operator in ("IS NOT NULL",):
            continue
        matching = ctx.count_matching(n.table, n.column, n.operator, n.value)
        if matching > 0:
            continue
        # Look for same-named column in other tables
        for ts in ctx.kg.tables:
            if ts.name == entity_table:
                continue
            for col in ts.columns:
                if col.name.lower() == n.column.lower():
                    other_match = ctx.count_matching(ts.name, col.name, n.operator, n.value)
                    if other_match > 0:
                        migrated = list(ctx.filter_nodes)
                        migrated[i] = QueryNode(table=ts.name, column=col.name, role="filter",
                                                operator=n.operator, value=n.value)
                        hypotheses.append(Hypothesis(
                            label=f"migrate_{n.column}_to_{ts.name}",
                            filter_nodes=migrated,
                            reasoning=f"{n.column} filter matches 0 in {entity_table} but {other_match} in {ts.name}",
                        ))
                        break

    # T13: Join direction swap — filter on parent vs child entity
    tables_in_filters = {n.table for n in ctx.filter_nodes}
    if len(tables_in_filters) >= 2:
        for other_table in tables_in_filters:
            if other_table == entity_table:
                continue
            fan_out = ctx.join_fan_out(entity_table, other_table)
            if fan_out > 3.0:
                hypotheses.append(Hypothesis(
                    label=f"fan_out_warning_{other_table}",
                    filter_nodes=list(ctx.filter_nodes),
                    sql_directives=[
                        f"JOIN WARNING: Joining {entity_table}→{other_table} has fan-out {fan_out:.1f}x. "
                        f"Use EXISTS subquery instead of JOIN to avoid inflated counts."
                    ],
                    reasoning=f"Join to {other_table} fans out {fan_out:.1f}x — use EXISTS instead",
                ))

    # ---------------------------------------------------------------
    # TEMPORAL TEMPLATES (time-aware patterns)
    # ---------------------------------------------------------------

    # T14: Date range instead of exact date
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    for i, n in enumerate(ctx.filter_nodes):
        if n.operator != "LIKE" or not date_pattern.match(str(n.value).strip('%')):
            if n.operator != "=" or not date_pattern.match(str(n.value)):
                continue
        # Check if this is a "during" or "in" question that might need a range
        if any(w in q for w in ("during", "in the month", "in the year", "between")):
            val = str(n.value).strip('%')
            # Year-level: expand to full year range
            if re.match(r'^\d{4}$', val):
                range_nodes = list(ctx.filter_nodes)
                range_nodes[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                           operator="LIKE", value=f"{val}%")
                hypotheses.append(Hypothesis(
                    label=f"year_range_{n.column}",
                    filter_nodes=range_nodes,
                    reasoning=f"Expand '{val}' to full year range (LIKE '{val}%')",
                ))
            # Month-level: expand to full month
            elif re.match(r'^\d{4}-\d{2}$', val):
                range_nodes = list(ctx.filter_nodes)
                range_nodes[i] = QueryNode(table=n.table, column=n.column, role="filter",
                                           operator="LIKE", value=f"{val}%")
                hypotheses.append(Hypothesis(
                    label=f"month_range_{n.column}",
                    filter_nodes=range_nodes,
                    reasoning=f"Expand '{val}' to full month range (LIKE '{val}%')",
                ))

    # T15: Most recent / latest record per entity
    if any(w in q for w in ("latest", "most recent", "last", "current", "newest")):
        ts = ctx.kg.get_table(entity_table)
        if ts:
            date_cols = [c.name for c in ts.columns if any(
                d in c.name.lower() for d in ("date", "time", "timestamp", "created", "updated")
            )]
            if date_cols:
                hypotheses.append(Hypothesis(
                    label="latest_record",
                    filter_nodes=list(ctx.filter_nodes),
                    sql_directives=[
                        f"TEMPORAL HINT: Question asks for latest/most recent. "
                        f"Use: WHERE {date_cols[0]} = (SELECT MAX({date_cols[0]}) FROM {entity_table} WHERE ...)"
                    ],
                    reasoning=f"'Latest' semantics — filter to most recent {date_cols[0]} per entity",
                ))

    # ---------------------------------------------------------------
    # AGGREGATION TEMPLATES (how to combine values)
    # ---------------------------------------------------------------

    # T16: Per-entity aggregation — question asks "how many X have Y" where Y
    # needs to be computed across rows, not checked on a single row
    rpe = evidence.get("rows_per_entity", 1.0)
    if rpe > 1.5 and ctx.comp_type == "count":
        for i, n in enumerate(ctx.filter_nodes):
            if n.table != entity_table or n.operator in ("=", "LIKE", "IS NOT NULL"):
                continue
            # Check if question implies aggregation ("total", "average", "sum", "overall")
            if any(w in q for w in ("total", "average", "overall", "combined", "cumulative", "aggregate")):
                hypotheses.append(Hypothesis(
                    label=f"agg_{n.column}",
                    filter_nodes=list(ctx.filter_nodes),
                    sql_directives=[
                        f"AGGREGATION HINT: Rows per entity = {rpe:.1f}. "
                        f"The condition on {n.column} {n.operator} {n.value} might need to be checked "
                        f"against the per-entity aggregate (SUM/AVG), not individual rows. "
                        f"Use: HAVING {n.operator.replace('<','').replace('>','')}({n.column}) {n.operator} {n.value}"
                    ],
                    reasoning=f"Multi-row entity — {n.column} condition may need per-entity aggregation",
                ))
                break  # One aggregation hint is enough

    # T17: Distinct counting when question says "different" or "unique"
    if any(w in q for w in ("different", "unique", "distinct", "separate")):
        for i, n in enumerate(ctx.output_nodes):
            hypotheses.append(Hypothesis(
                label="distinct_output",
                filter_nodes=list(ctx.filter_nodes),
                sql_directives=[
                    f'COUNT DISTINCT HINT: Question asks for distinct/unique values. '
                    f'Use COUNT(DISTINCT "{n.column}") or SELECT DISTINCT.'
                ],
                reasoning="Question language implies distinct counting",
            ))
            break

    return hypotheses


def _score_hypothesis(ctx: EngineContext, h: Hypothesis) -> None:
    """Score a hypothesis by probing DB and checking plausibility signals."""
    h.result_count = _probe_result_count(ctx, h.filter_nodes)
    score = 0.0

    # Zero results = very bad
    if h.result_count == 0:
        h.score = -100.0
        return

    # Non-zero is good
    score += 10.0

    # Bonus: result is a small, specific number (good for count queries)
    if ctx.comp_type == "count" and 0 < h.result_count <= 50:
        score += 5.0

    # Penalty: too many results for a singular question
    answer_shape = ""
    for line in ctx.user_intent.split("\n"):
        if "Answer shape:" in line:
            answer_shape = line.split("Answer shape:")[1].strip().lower()
    if answer_shape == "single_value" and ctx.comp_type == "simple_lookup" and h.result_count > 5:
        score -= 20.0

    # Bonus: uses known/derived range boundaries (more semantically grounded)
    for n in h.filter_nodes:
        if n.operator in ("IS NOT NULL",):
            continue
        resolved = _resolve_normal_range(ctx, n.table, n.column, model_call=None)
        if resolved:
            try:
                val = float(n.value)
                if val == resolved[0] or val == resolved[1]:
                    score += 3.0  # Uses exact known/derived boundary
            except (ValueError, TypeError):
                pass

    # Penalty: IS NOT NULL is weaker semantics (less discriminating)
    null_filters = sum(1 for n in h.filter_nodes if n.operator == "IS NOT NULL")
    score -= null_filters * 1.0

    h.score = score


def _llm_select_hypothesis(
    ctx: EngineContext,
    hypotheses: list[Hypothesis],
    evidence: dict[str, Any],
    anomaly: str,
    model_call: Callable,
    iteration: int,
) -> Hypothesis | None:
    """LLM picks the best hypothesis given evidence and scored candidates."""
    import json

    # Format hypotheses for LLM
    h_descriptions: list[str] = []
    for i, h in enumerate(hypotheses):
        filters_desc = ", ".join(
            f"{n.column} {n.operator} {n.value}" for n in h.filter_nodes
            if n.table == ctx.entity_table
        )
        h_descriptions.append(
            f"  [{i}] {h.label} (result_count={h.result_count}, score={h.score:.1f})\n"
            f"      Filters: {filters_desc}\n"
            f"      Reasoning: {h.reasoning}"
        )

    # Format drop-one analysis
    drop_one_desc = ""
    drop_one = evidence.get("drop_one_analysis", {})
    if drop_one:
        drop_one_desc = "DROP-ONE ANALYSIS (count when each filter column is removed):\n"
        for col, count in drop_one.items():
            drop_one_desc += f"  Without {col}: {count} entities\n"

    # Format co-occurrence
    co_occ_desc = ""
    co_occ = evidence.get("co_occurrence", {})
    if co_occ:
        co_occ_desc = (
            f"CO-OCCURRENCE: {co_occ['cols'][0]} and {co_occ['cols'][1]} — "
            f"same_row={co_occ['same_row']}, across_rows={co_occ['across_rows']}"
        )

    prompt = f"""You are a query analyst. A data query has implausible results. Choose the best fix.

QUESTION: {ctx.question}
INTENT: {ctx.user_intent}
COMPUTATION TYPE: {ctx.comp_type}
ANOMALY DETECTED: {anomaly}

ENTITY TABLE: {evidence.get('entity_table', '?')}
  Total entities: {evidence.get('total_entities', '?')}
  Rows per entity: {evidence.get('rows_per_entity', '?')}

{drop_one_desc}
{co_occ_desc}

CANDIDATE HYPOTHESES (scored by plausibility):
{chr(10).join(h_descriptions)}

DECISION FRAMEWORK:
1. A result_count of 0 with restrictive filters usually means wrong threshold or inverted semantics.
2. Higher scores = more plausible. But score alone isn't enough — semantic correctness matters.
3. "Normal X" means WITHIN healthy range. "Abnormal X" means OUTSIDE healthy range.
4. If drop-one shows removing column C gives N results, but C's threshold gives 0 → C's threshold is wrong.
5. Sparse columns with existence language ("have", "with") should use IS NOT NULL.
6. If rows_per_entity > 1 and co-occurrence shows same_row < across_rows → cross-row subqueries needed.

Pick the hypothesis index that BEST matches the question semantics AND produces plausible results.
If NONE of the hypotheses are good, you may propose a new one.

Respond ONLY with JSON:
{{
  "chosen_index": <int or null if proposing new>,
  "reasoning": "why this is the correct interpretation",
  "new_hypothesis": null or {{
    "corrections": [{{"index": <filter_index>, "operator": "...", "value": "..."}}],
    "add_filters": [{{"table": "T", "column": "C", "operator": "...", "value": "..."}}],
    "remove_indices": [<filter_indices_to_remove>]
  }}
}}
"""

    try:
        from data_agent_baseline.agents.model import ModelMessage
        messages = [ModelMessage(role="user", content=prompt)]
        raw = model_call(messages)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            return None
        decision = json.loads(json_match.group())
    except Exception:
        return None

    chosen_idx = decision.get("chosen_index")
    reasoning = decision.get("reasoning", "")

    # If LLM chose an existing hypothesis
    if chosen_idx is not None and 0 <= chosen_idx < len(hypotheses):
        chosen = hypotheses[chosen_idx]
        chosen.reasoning = reasoning or chosen.reasoning
        return chosen

    # If LLM proposed a new hypothesis
    new_h = decision.get("new_hypothesis")
    if new_h:
        new_nodes = list(ctx.filter_nodes)
        # Apply corrections
        for corr in new_h.get("corrections", []):
            idx = corr.get("index")
            if idx is not None and 0 <= idx < len(new_nodes):
                n = new_nodes[idx]
                new_nodes[idx] = QueryNode(
                    table=n.table, column=n.column, role="filter",
                    operator=corr.get("operator", n.operator),
                    value=corr.get("value", ""),
                )
        # Remove indices
        for idx in sorted(new_h.get("remove_indices", []), reverse=True):
            if 0 <= idx < len(new_nodes):
                new_nodes.pop(idx)
        # Add filters
        for add in new_h.get("add_filters", []):
            new_nodes.append(QueryNode(
                table=add["table"], column=add["column"], role="filter",
                operator=add["operator"], value=add.get("value", ""),
            ))
        return Hypothesis(
            label="llm_proposed",
            filter_nodes=new_nodes,
            reasoning=reasoning,
        )

    return None


def _llm_generate_hypotheses(
    ctx: EngineContext,
    evidence: dict[str, Any],
    anomaly: str,
    model_call: Callable,
    failed_labels: list[str],
) -> list[Hypothesis]:
    """LLM generates novel hypotheses when deterministic ones all fail.

    This is the true adaptive component — the LLM can propose structural
    patterns it has never been explicitly taught: negation, temporal
    comparison, set operations, novel threshold logic, etc.
    """
    import json

    entity_table = ctx.entity_table

    # Build a rich context of what the data looks like
    filter_desc: list[str] = []
    for fe in evidence.get("filters", []):
        desc = f"  {fe['table']}.{fe['column']} {fe['operator']} {fe['value']}"
        if "data_min" in fe:
            desc += f"\n    Data: [{fe['data_min']}, {fe['data_max']}], avg={fe['data_avg']}, distinct={fe['distinct_values']}"
            desc += f"\n    Matching: {fe['rows_matching']}/{fe.get('total_non_null', '?')} rows"
            desc += f"\n    Null ratio: {(fe.get('null_ratio') or 0)*100:.0f}%, Sparse: {fe.get('is_sparse', False)}"
            desc += f"\n    Spread: {fe.get('spread', '?')}, Cardinality: {fe.get('cardinality_ratio', '?')}"
        elif fe["operator"] == "IS NOT NULL":
            desc += f"\n    Non-null rows: {fe.get('total_non_null', '?')}, Null ratio: {(fe.get('null_ratio') or 0)*100:.0f}%"
        filter_desc.append(desc)

    drop_one = evidence.get("drop_one_analysis", {})
    drop_one_desc = ""
    if drop_one:
        drop_one_desc = "DROP-ONE (entities when each column removed):\n"
        for col, count in drop_one.items():
            drop_one_desc += f"  Without {col}: {count}\n"

    co_occ = evidence.get("co_occurrence", {})
    co_occ_desc = ""
    if co_occ:
        co_occ_desc = (
            f"CO-OCCURRENCE between {co_occ['cols'][0]} & {co_occ['cols'][1]}:\n"
            f"  Same row: {co_occ['same_row']} entities, Across rows: {co_occ['across_rows']} entities\n"
        )

    # Schema context: show all columns in the entity table
    schema_desc = ""
    ts = ctx.kg.get_table(entity_table)
    if ts:
        cols = [f"{c.name} ({c.sql_type})" for c in ts.columns[:30]]
        schema_desc = f"AVAILABLE COLUMNS in {entity_table}: {', '.join(cols)}\n"

    prompt = f"""You are an expert data analyst. The current query filters produce implausible results. All standard fix strategies have failed. You must propose novel interpretations.

QUESTION: {ctx.question}
INTENT: {ctx.user_intent}
COMPUTATION TYPE: {ctx.comp_type}
ANOMALY: {anomaly}

CURRENT FILTERS (with evidence):
{chr(10).join(filter_desc)}

{schema_desc}
ENTITY TABLE: {entity_table}
  Total entities: {evidence.get('total_entities', '?')}
  Rows per entity: {evidence.get('rows_per_entity', '?')}
  Total rows: {evidence.get('total_rows', '?')}

{drop_one_desc}
{co_occ_desc}
FAILED STRATEGIES (already tried, didn't produce plausible results): {', '.join(failed_labels)}

STRUCTURAL PATTERNS TO CONSIDER (propose whichever fits):
1. Threshold correction — wrong comparison value or direction
2. Semantic inversion — "normal" vs "abnormal" swapped, "above" vs "below" confused
3. Existence semantics — "have X" means X IS NOT NULL, not X > threshold
4. Negation — "never had" / "without" / "excluding" → NOT EXISTS or NOT IN
5. Temporal filter — "before/after/during" → date comparison
6. Set difference — "had A but not B" → EXCEPT or NOT IN subquery
7. Relative comparison — "more than average" → subquery for avg
8. Group-based — "most/least/top" → GROUP BY + ORDER + LIMIT
9. Range boundary — threshold should be an exact boundary from the data
10. Column swap — wrong column selected, should use a different one
11. Combined conditions should be OR not AND (or vice versa)

You MUST propose 2-3 different hypotheses. For EACH hypothesis, specify exactly which filter indices to change and how.

Respond with JSON:
{{
  "hypotheses": [
    {{
      "label": "short_name",
      "reasoning": "why this interpretation fits the question and data",
      "corrections": [
        {{"index": 0, "action": "replace", "operator": ">=", "value": "150"}},
        {{"index": 1, "action": "remove"}}
      ],
      "add_filters": [
        {{"table": "{entity_table}", "column": "COL", "operator": "...", "value": "..."}}
      ],
      "sql_hint": "optional structural hint for SQL generation (e.g. 'use OR between conditions on this column')"
    }}
  ]
}}
"""

    try:
        from data_agent_baseline.agents.model import ModelMessage
        messages = [ModelMessage(role="user", content=prompt)]
        raw = model_call(messages)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            return []
        result = json.loads(json_match.group())
    except Exception:
        return []

    hypotheses: list[Hypothesis] = []
    for h_data in result.get("hypotheses", []):
        new_nodes = list(ctx.filter_nodes)
        to_remove: list[int] = []

        for corr in h_data.get("corrections", []):
            idx = corr.get("index")
            if idx is None or idx >= len(new_nodes):
                continue
            action = corr.get("action", "replace")
            if action == "remove":
                to_remove.append(idx)
            elif action == "replace":
                n = new_nodes[idx]
                new_nodes[idx] = QueryNode(
                    table=n.table, column=n.column, role="filter",
                    operator=corr.get("operator", n.operator),
                    value=corr.get("value", ""),
                )

        for idx in sorted(to_remove, reverse=True):
            if 0 <= idx < len(new_nodes):
                new_nodes.pop(idx)

        for add in h_data.get("add_filters", []):
            table = add.get("table", entity_table)
            new_nodes.append(QueryNode(
                table=table, column=add["column"], role="filter",
                operator=add["operator"], value=add.get("value", ""),
            ))

        directives: list[str] = []
        sql_hint = h_data.get("sql_hint", "")
        if sql_hint:
            directives.append(f"LLM STRUCTURAL HINT: {sql_hint}")

        hypotheses.append(Hypothesis(
            label=h_data.get("label", f"llm_novel_{len(hypotheses)}"),
            filter_nodes=new_nodes,
            sql_directives=directives,
            reasoning=h_data.get("reasoning", ""),
        ))

    return hypotheses


def _apply_domain_column_fixes(ctx: EngineContext) -> list[QueryNode]:
    """Re-apply domain column fixes to current filter nodes.

    Uses anchor_text definitions and synonym groups to ensure the correct
    domain-defined column is used (e.g. rank vs position).
    """
    if not ctx.anchor_text or not ctx.filter_nodes:
        return ctx.filter_nodes

    q_lower = ctx.question_lower
    used_cols = {n.column.lower() for n in ctx.filter_nodes}

    domain_cols: dict[str, tuple[str, str]] = {}
    seen: set[str] = set()
    for m in re.finditer(r'^- (\w+):\s+(.+)', ctx.anchor_text, re.MULTILINE):
        defined_col = m.group(1).lower()
        if defined_col in seen:
            continue
        seen.add(defined_col)
        q_words = re.findall(r'\b[a-z]+', q_lower)
        if not any(w.startswith(defined_col) for w in q_words) or defined_col in used_cols:
            continue
        for t in ctx.kg.tables:
            for c in t.columns:
                if c.name.lower() == defined_col:
                    domain_cols[defined_col] = (t.name, m.group(2))
                    break

    if not domain_cols:
        return ctx.filter_nodes

    synonym_groups = [
        {"position", "rank", "positionorder"},
        {"round", "number"},
    ]
    fixed = []
    for node in ctx.filter_nodes:
        swapped = False
        for defined_col, (col_table, _) in domain_cols.items():
            for group in synonym_groups:
                if node.column.lower() in group and defined_col in group:
                    domain_count = ctx.count_matching(col_table, defined_col, node.operator, node.value)
                    if domain_count > 0:
                        fixed.append(QueryNode(
                            table=col_table, column=defined_col, role=node.role,
                            operator=node.operator, value=node.value,
                        ))
                        swapped = True
                    break
            if swapped:
                break
        if not swapped:
            fixed.append(node)

    # Second pass: correct filter values using domain-defined mappings
    # e.g. "Use bond_type = '#' for triple bonds" or "'M' for male"
    value_map: dict[str, list[tuple[str, str]]] = {}
    for m in re.finditer(r"['\"`]([^'\"` ]+)['\"`]\s+for\s+(\w+)", ctx.anchor_text):
        domain_val, keyword = m.group(1), m.group(2).lower()
        value_map.setdefault(keyword, []).append((domain_val, keyword))
    for m in re.finditer(r"(\w[\w_]+)\s*=\s*['\"`]([^'\"` ]+)['\"`]\s+for\s+(\w+)", ctx.anchor_text):
        col_name, domain_val, keyword = m.group(1).lower(), m.group(2), m.group(3).lower()
        value_map.setdefault(keyword, []).append((domain_val, col_name))

    if value_map:
        q_words_val = set(re.findall(r'\b[a-z]+', ctx.question_lower))
        corrected = []
        for node in fixed:
            replaced = False
            node_val_lower = str(node.value).lower()
            # Check if the current value is a natural-language word that has a domain mapping
            for keyword, mappings in value_map.items():
                if keyword in q_words_val and node_val_lower in (keyword, keyword + "s"):
                    for domain_val, mapped_col in mappings:
                        if mapped_col == node.column.lower() or mapped_col == keyword:
                            count = ctx.count_matching(node.table, node.column, node.operator, domain_val)
                            if count > 0:
                                corrected.append(QueryNode(
                                    table=node.table, column=node.column, role=node.role,
                                    operator=node.operator, value=domain_val,
                                ))
                                replaced = True
                                break
                    if replaced:
                        break
            if not replaced:
                corrected.append(node)
        fixed = corrected

    return fixed


def _enforce_domain_locks(
    ctx: EngineContext,
    new_filters: list[QueryNode],
    all_logs: list[tuple[str, str]],
) -> list[QueryNode]:
    """Preserve domain-locked filters: if a hypothesis swaps a locked column to a synonym, swap it back."""
    if not ctx.domain_locked_columns:
        return new_filters

    locked_keys = ctx.domain_locked_columns
    # Find the original domain-locked filter nodes from ctx (before hypothesis applied)
    original_locked = {
        f"{n.table}.{n.column}": n for n in ctx.filter_nodes
        if f"{n.table}.{n.column}" in locked_keys
    }
    if not original_locked:
        return new_filters

    synonym_groups = [
        {"position", "rank", "positionorder"},
        {"round", "number"},
    ]

    result = []
    for node in new_filters:
        node_key = f"{node.table}.{node.column}"
        if node_key in locked_keys:
            result.append(node)
            continue
        # Check if this node's column is a synonym of a locked column on the same table
        replaced = False
        for locked_key, locked_node in original_locked.items():
            if locked_node.table != node.table:
                continue
            for group in synonym_groups:
                if node.column.lower() in group and locked_node.column.lower() in group:
                    result.append(locked_node)
                    all_logs.append(("domain_lock",
                        f"Preserved {locked_key} (hypothesis tried {node.table}.{node.column})"))
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            result.append(node)
    return result


def _adaptive_loop(
    ctx: EngineContext,
    model_call: Callable | None,
    all_logs: list[tuple[str, str]],
    all_directives: list[str],
) -> None:
    """Hypothesis-driven adaptive loop: assess → generate → score → select → validate.

    Two-tier hypothesis generation:
      Tier 1: Deterministic hypotheses (template-based, fast, no LLM)
      Tier 2: LLM-generated hypotheses (creative, handles novel patterns)
    Tier 2 only fires when Tier 1 produces no viable alternatives.
    """
    for iteration in range(3):
        # Step 1: Assess plausibility
        report = _assess_plausibility(ctx)
        if report.is_plausible:
            all_logs.append(("adaptive_ok", f"iter={iteration}: plausible (count={report.result_count})"))
            break

        all_logs.append(("adaptive_anomaly", f"iter={iteration+1}: {report.anomaly}"))

        # Step 2: Gather evidence
        evidence = _gather_evidence(ctx)

        # Step 3: Generate deterministic hypotheses (Tier 1)
        hypotheses = _generate_hypotheses(ctx, evidence, report.anomaly)

        # Step 4: Score each hypothesis by probing DB
        for h in hypotheses:
            _score_hypothesis(ctx, h)

        # Filter out impossible hypotheses
        viable = [h for h in hypotheses if h.result_count > 0 or h.label == "current"]
        non_current_viable = [h for h in viable if h.label != "current" and h.score > 0]

        # Step 4b: If deterministic hypotheses all fail → LLM generates novel ones (Tier 2)
        if not non_current_viable and model_call:
            failed_labels = [h.label for h in hypotheses if h.label != "current"]
            all_logs.append(("adaptive_tier2", "Deterministic hypotheses failed, invoking LLM generation"))
            llm_hypotheses = _llm_generate_hypotheses(
                ctx, evidence, report.anomaly, model_call, failed_labels,
            )
            for h in llm_hypotheses:
                _score_hypothesis(ctx, h)
            hypotheses.extend(llm_hypotheses)
            viable = [h for h in hypotheses if h.result_count > 0 or h.label == "current"]
            non_current_viable = [h for h in viable if h.label != "current" and h.score > 0]

        all_logs.append(("adaptive_hypotheses",
                        f"Generated {len(hypotheses)}, {len(viable)} viable: " +
                        ", ".join(f"{h.label}({h.result_count})" for h in viable)))

        # Step 5: If only one viable non-current hypothesis, pick it without LLM
        if len(non_current_viable) == 1 and non_current_viable[0].score > 5:
            chosen = non_current_viable[0]
            all_logs.append(("adaptive_auto", f"Single clear winner: {chosen.label} → {chosen.reasoning}"))
            enforced = _enforce_domain_locks(ctx, chosen.filter_nodes, all_logs)
            ctx.filter_nodes = enforced
            ctx.filter_nodes = _apply_domain_column_fixes(ctx)
            if chosen.sql_directives:
                all_directives.extend(chosen.sql_directives)
            continue

        # Step 6: Multiple viable options — LLM selects
        if not model_call:
            if non_current_viable:
                best = max(non_current_viable, key=lambda h: h.score)
                enforced = _enforce_domain_locks(ctx, best.filter_nodes, all_logs)
                ctx.filter_nodes = enforced
                ctx.filter_nodes = _apply_domain_column_fixes(ctx)
                if best.sql_directives:
                    all_directives.extend(best.sql_directives)
                all_logs.append(("adaptive_pick", f"Best score: {best.label} (score={best.score:.1f})"))
            break

        chosen = _llm_select_hypothesis(ctx, viable, evidence, report.anomaly, model_call, iteration + 1)
        if chosen and chosen.label != "current":
            enforced = _enforce_domain_locks(ctx, chosen.filter_nodes, all_logs)
            ctx.filter_nodes = enforced
            ctx.filter_nodes = _apply_domain_column_fixes(ctx)
            if chosen.sql_directives:
                all_directives.extend(chosen.sql_directives)
            all_logs.append(("adaptive_llm_pick", f"iter={iteration+1}: chose {chosen.label} — {chosen.reasoning}"))
        else:
            all_logs.append(("adaptive_llm_no_change", f"iter={iteration+1}: LLM kept current filters"))
            break

        # Re-run cross-row detection after filter changes
        cross_result = rule_cross_row_structure(ctx)
        if cross_result:
            all_directives[:] = [d for d in all_directives if "CROSS-ROW" not in d]
            all_directives.extend(cross_result.sql_directives)
            all_logs.extend(cross_result.log_entries)


def run_rules_engine(
    question: str,
    user_intent: str,
    comp_type: str,
    filter_nodes: list[QueryNode],
    output_nodes: list[QueryNode],
    kg: KnowledgeGraph,
    db_path: Path | None,
    knowledge_text: str = "",
    anchor_text: str = "",
    model_call: Callable | None = None,
    domain_locked_columns: set[str] | None = None,
) -> EngineOutput:
    """Run deterministic rules, then hypothesis-driven adaptive loop.

    Phase 1: Deterministic rules (fast, no LLM) — handles known patterns.
    Phase 2: Adaptive loop — detects implausible results via multi-signal
             assessment, generates alternative hypotheses, scores them against
             the DB, and uses LLM to select the best interpretation.
    """
    ctx = EngineContext(
        question=question,
        question_lower=question.lower(),
        user_intent=user_intent,
        comp_type=comp_type,
        filter_nodes=list(filter_nodes),
        output_nodes=list(output_nodes),
        kg=kg,
        db_path=db_path,
        knowledge_text=knowledge_text,
        anchor_text=anchor_text,
        model_call=model_call,
        domain_locked_columns=domain_locked_columns or set(),
    )

    all_directives: list[str] = []
    all_logs: list[tuple[str, str]] = []
    all_decomposition: list[DecompositionStep] = []

    # Phase 0: Skip subquery filter values — they encode scope internally
    # and should be passed through to the SQL LLM as-is

    # Phase 1: Deterministic rules (fast, no LLM)
    for rule_fn in RULES:
        result = rule_fn(ctx)
        if result is None:
            continue
        if result.filter_nodes is not None:
            ctx.filter_nodes = result.filter_nodes
        all_directives.extend(result.sql_directives)
        all_logs.extend(result.log_entries)
        all_decomposition.extend(result.decomposition)

    # Phase 2: Adaptive loop (hypothesis-driven)
    if db_path:
        _adaptive_loop(ctx, model_call, all_logs, all_directives)

    # Ensure cross-row runs at least once after all modifications
    if not any("CROSS-ROW" in d for d in all_directives):
        cross_result = rule_cross_row_structure(ctx)
        if cross_result:
            all_directives.extend(cross_result.sql_directives)
            all_logs.extend(cross_result.log_entries)

    return EngineOutput(
        filter_nodes=ctx.filter_nodes,
        sql_directives=all_directives,
        log_entries=all_logs,
        decomposition=all_decomposition,
    )
