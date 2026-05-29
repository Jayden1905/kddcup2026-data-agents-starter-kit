"""ApproachGraph: queryable memory of attempted strategies.

Prevents the agent from repeating the same logical approach and lets it
query past failures by structural similarity. The fingerprint is modeled
after the user intent structure: tables, joins, select columns, filter
columns, filter values, aggregations, grain, temporal, and ordering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class ApproachFingerprint:
    """Full structural decomposition of an SQL approach."""

    tables: list[str]
    joins: list[str]
    select_cols: list[str]
    filter_cols: list[str]
    filter_values: list[str]
    aggs: list[str]
    group_by: list[str]
    order_by: list[str]
    has_limit: bool
    has_distinct: bool
    has_subquery: bool
    grain: str  # "scalar", "single_row", "multi_row", "grouped"
    temporal: str  # "none", column name if date-filtered

    @property
    def key(self) -> str:
        """Structural identity — two SQL with the same key are the same approach."""
        return (
            f"tables={self.tables}"
            f"|joins={self.joins}"
            f"|select={self.select_cols}"
            f"|filters={self.filter_cols}"
            f"|aggs={self.aggs}"
            f"|group={self.group_by}"
            f"|grain={self.grain}"
            f"|temporal={self.temporal}"
        )


@dataclass(slots=True)
class ApproachNode:
    """A single attempted approach with full context."""

    fingerprint: ApproachFingerprint
    sql: str
    result: str
    reason: str
    turn: int
    parent_key: str | None = None

    @property
    def key(self) -> str:
        return self.fingerprint.key


class ApproachGraph:
    """Queryable graph of attempted SQL approaches.

    Stores every failed approach with a full structural fingerprint derived
    from SQL parsing. The fingerprint captures the logical strategy (which
    tables, how joined, what filtered, what aggregated, what grain) so that
    trivial variations (different literal values, whitespace, aliases) are
    recognized as the same approach.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, ApproachNode] = {}
        self._attempt_counts: dict[str, int] = {}

    def fingerprint(self, sql: str) -> ApproachFingerprint:
        """Extract full structural fingerprint from SQL."""
        sql_upper = sql.upper()

        tables = sorted(set(self._extract_tables(sql)))
        joins = sorted(set(self._extract_joins(sql)))
        select_cols = sorted(set(self._extract_select_cols(sql)))
        filter_cols = sorted(set(self._extract_filter_cols(sql)))
        filter_values = sorted(set(self._extract_filter_values(sql)))
        aggs = sorted(set(self._extract_aggs(sql)))
        group_by = sorted(set(self._extract_group_by(sql)))
        order_by = sorted(set(self._extract_order_by(sql)))
        has_limit = "LIMIT" in sql_upper
        has_distinct = "DISTINCT" in sql_upper
        has_subquery = sql_upper.count("SELECT") > 1

        # Infer grain
        if aggs and not group_by:
            grain = "scalar"
        elif aggs and group_by:
            grain = "grouped"
        elif has_limit and "1" in sql_upper.split("LIMIT")[-1].strip()[:3]:
            grain = "single_row"
        else:
            grain = "multi_row"

        # Infer temporal
        temporal = "none"
        temporal_keywords = ("date", "year", "month", "day", "time", "period", "quarter")
        for col in filter_cols:
            if any(tk in col.lower() for tk in temporal_keywords):
                temporal = col.lower()
                break

        return ApproachFingerprint(
            tables=tables,
            joins=joins,
            select_cols=select_cols,
            filter_cols=filter_cols,
            filter_values=filter_values,
            aggs=aggs,
            group_by=group_by,
            order_by=order_by,
            has_limit=has_limit,
            has_distinct=has_distinct,
            has_subquery=has_subquery,
            grain=grain,
            temporal=temporal,
        )

    def is_duplicate(self, sql: str) -> ApproachNode | None:
        """Check if this SQL matches a previously attempted approach.

        Returns the matching node if duplicate, None otherwise.
        """
        fp = self.fingerprint(sql)
        return self.nodes.get(fp.key)

    def record(
        self,
        sql: str,
        result: str,
        reason: str,
        turn: int,
        parent_sql: str | None = None,
    ) -> ApproachNode:
        """Record a failed approach. Returns the node."""
        fp = self.fingerprint(sql)
        parent_key = self.fingerprint(parent_sql).key if parent_sql else None

        node = ApproachNode(
            fingerprint=fp,
            sql=sql,
            result=result,
            reason=reason,
            turn=turn,
            parent_key=parent_key,
        )
        self.nodes[fp.key] = node
        self._attempt_counts[fp.key] = self._attempt_counts.get(fp.key, 0) + 1
        return node

    def recall(
        self,
        tables: list[str] | None = None,
        columns: list[str] | None = None,
        agg: str | None = None,
        grain: str | None = None,
        temporal: str | None = None,
        joins: list[str] | None = None,
    ) -> list[ApproachNode]:
        """Query past approaches by any combination of structural dimensions."""
        results: list[tuple[int, ApproachNode]] = []
        for node in self.nodes.values():
            score = 0
            fp = node.fingerprint
            if tables:
                overlap = set(t.lower() for t in tables) & set(
                    t.lower() for t in fp.tables
                )
                if overlap:
                    score += len(overlap)
            if columns:
                all_cols = set(c.lower() for c in fp.filter_cols + fp.select_cols)
                overlap = set(c.lower() for c in columns) & all_cols
                if overlap:
                    score += len(overlap)
            if agg:
                if agg.upper() in [a.upper() for a in fp.aggs]:
                    score += 1
            if grain:
                if grain.lower() == fp.grain.lower():
                    score += 1
            if temporal:
                if temporal.lower() in fp.temporal.lower():
                    score += 1
            if joins:
                join_set = set(j.lower() for j in joins)
                overlap = join_set & set(j.lower() for j in fp.joins)
                if overlap:
                    score += len(overlap)
            if score > 0:
                results.append((score, node))
        results.sort(key=lambda x: (-x[0], x[1].turn))
        return [n for _, n in results]

    def similar_to(self, sql: str) -> list[ApproachNode]:
        """Find approaches structurally similar to the given SQL."""
        target = self.fingerprint(sql)
        target_tables = set(t.lower() for t in target.tables)
        target_filters = set(c.lower() for c in target.filter_cols)
        target_aggs = set(a.upper() for a in target.aggs)
        target_joins = set(j.lower() for j in target.joins)
        target_selects = set(c.lower() for c in target.select_cols)

        results: list[tuple[float, ApproachNode]] = []
        for node in self.nodes.values():
            fp = node.fingerprint
            node_tables = set(t.lower() for t in fp.tables)
            node_filters = set(c.lower() for c in fp.filter_cols)
            node_aggs = set(a.upper() for a in fp.aggs)
            node_joins = set(j.lower() for j in fp.joins)
            node_selects = set(c.lower() for c in fp.select_cols)

            # Weighted Jaccard across all dimensions
            similarity = 0.0

            t_union = len(target_tables | node_tables)
            if t_union:
                similarity += 2.0 * len(target_tables & node_tables) / t_union

            f_union = len(target_filters | node_filters)
            if f_union:
                similarity += 1.5 * len(target_filters & node_filters) / f_union

            a_union = len(target_aggs | node_aggs)
            if a_union:
                similarity += 1.0 * len(target_aggs & node_aggs) / a_union

            j_union = len(target_joins | node_joins)
            if j_union:
                similarity += 1.5 * len(target_joins & node_joins) / j_union

            s_union = len(target_selects | node_selects)
            if s_union:
                similarity += 1.0 * len(target_selects & node_selects) / s_union

            if target.grain == fp.grain:
                similarity += 0.5

            if target.temporal == fp.temporal:
                similarity += 0.5

            if similarity > 0.5:
                results.append((similarity, node))

        results.sort(key=lambda x: -x[0])
        return [n for _, n in results]

    def dead_ends(self) -> list[ApproachNode]:
        """Return approaches that have been tried 2+ times — exhausted paths."""
        return [
            self.nodes[k]
            for k, count in self._attempt_counts.items()
            if count >= 2 and k in self.nodes
        ]

    def render_for_prompt(self, relevant: list[ApproachNode] | None = None) -> str:
        """Render all failed approaches for injection into the agent prompt.

        Never truncates — the agent sees the complete history of what was tried.
        """
        nodes = relevant if relevant is not None else list(self.nodes.values())
        if not nodes:
            return ""

        lines = ["[Failed Approaches — do NOT repeat these strategies]"]
        for node in nodes:
            fp = node.fingerprint
            lines.append(f"  Turn {node.turn}:")
            lines.append(f"    SQL: {node.sql}")
            lines.append(f"    Tables: {fp.tables}")
            lines.append(f"    Joins: {fp.joins}")
            lines.append(f"    Select: {fp.select_cols}")
            lines.append(f"    Filters: {fp.filter_cols} = {fp.filter_values}")
            lines.append(f"    Aggregations: {fp.aggs}")
            lines.append(f"    Group By: {fp.group_by}")
            lines.append(f"    Order By: {fp.order_by}")
            lines.append(f"    Grain: {fp.grain}")
            lines.append(f"    Temporal: {fp.temporal}")
            lines.append(f"    Distinct: {fp.has_distinct} | Subquery: {fp.has_subquery}")
            lines.append(f"    Result: {node.result}")
            lines.append(f"    Why Failed: {node.reason}")
            lines.append("")

        dead = self.dead_ends()
        if dead:
            lines.append("[Dead Ends — these structural patterns are exhausted]")
            for d in dead:
                fp = d.fingerprint
                count = self._attempt_counts.get(fp.key, 0)
                lines.append(
                    f"  ⊘ tables={fp.tables} joins={fp.joins} "
                    f"filters={fp.filter_cols} aggs={fp.aggs} "
                    f"grain={fp.grain} (tried {count}x)"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SQL parsing — full decomposition
    # ------------------------------------------------------------------

    _TABLE_RE = re.compile(
        r'\b(?:FROM|JOIN)\s+["\']?(\w+)["\']?', re.IGNORECASE
    )

    _JOIN_RE = re.compile(
        r'\b(\w+)\s*\.\s*(\w+)\s*=\s*(\w+)\s*\.\s*(\w+)', re.IGNORECASE
    )

    _SELECT_COL_RE = re.compile(
        r'\bSELECT\s+(?:DISTINCT\s+)?(.+?)\s+FROM\b',
        re.IGNORECASE | re.DOTALL,
    )

    _WHERE_RE = re.compile(
        r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)',
        re.IGNORECASE | re.DOTALL,
    )

    _FILTER_COL_RE = re.compile(
        r'["\']?(\w+)["\']?\s*(?:=|!=|<>|<=|>=|<|>|LIKE|IN|IS|BETWEEN|NOT)',
        re.IGNORECASE,
    )

    _FILTER_VALUE_RE = re.compile(
        r"(?:=|!=|<>|<=|>=|<|>|LIKE|IN\s*\(|IS|BETWEEN)\s*'([^']*)'",
        re.IGNORECASE,
    )

    _AGG_RE = re.compile(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', re.IGNORECASE)

    _GROUP_BY_RE = re.compile(
        r'\bGROUP\s+BY\b\s+(.+?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|$)',
        re.IGNORECASE | re.DOTALL,
    )

    _ORDER_BY_RE = re.compile(
        r'\bORDER\s+BY\b\s+(.+?)(?:\bLIMIT\b|$)',
        re.IGNORECASE | re.DOTALL,
    )

    def _extract_tables(self, sql: str) -> list[str]:
        return [m.group(1) for m in self._TABLE_RE.finditer(sql)]

    def _extract_joins(self, sql: str) -> list[str]:
        joins = []
        for m in self._JOIN_RE.finditer(sql):
            t1, c1, t2, c2 = m.group(1), m.group(2), m.group(3), m.group(4)
            pair = sorted([f"{t1}.{c1}", f"{t2}.{c2}"])
            joins.append(f"{pair[0]}={pair[1]}")
        return joins

    def _extract_select_cols(self, sql: str) -> list[str]:
        m = self._SELECT_COL_RE.search(sql)
        if not m:
            return []
        select_clause = m.group(1)
        # Split by comma at depth 0 (respect parentheses)
        cols = []
        current = ""
        depth = 0
        for ch in select_clause:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                cols.append(self._normalize_col(current.strip()))
                current = ""
            else:
                current += ch
        if current.strip():
            cols.append(self._normalize_col(current.strip()))
        return [c for c in cols if c and c != "*"]

    def _extract_filter_cols(self, sql: str) -> list[str]:
        m = self._WHERE_RE.search(sql)
        if not m:
            return []
        where_clause = m.group(1)
        return [
            fm.group(1)
            for fm in self._FILTER_COL_RE.finditer(where_clause)
            if fm.group(1).upper() not in ("AND", "OR", "NOT", "NULL", "LIKE", "IN")
        ]

    def _extract_filter_values(self, sql: str) -> list[str]:
        m = self._WHERE_RE.search(sql)
        if not m:
            return []
        where_clause = m.group(1)
        return [vm.group(1) for vm in self._FILTER_VALUE_RE.finditer(where_clause)]

    def _extract_aggs(self, sql: str) -> list[str]:
        return [m.group(1).upper() for m in self._AGG_RE.finditer(sql)]

    def _extract_group_by(self, sql: str) -> list[str]:
        m = self._GROUP_BY_RE.search(sql)
        if not m:
            return []
        clause = m.group(1).strip()
        return [c.strip().strip('"').strip("'") for c in clause.split(",")]

    def _extract_order_by(self, sql: str) -> list[str]:
        m = self._ORDER_BY_RE.search(sql)
        if not m:
            return []
        clause = m.group(1).strip()
        parts = []
        for part in clause.split(","):
            col = part.strip().split()[0].strip('"').strip("'")
            direction = "DESC" if "DESC" in part.upper() else "ASC"
            parts.append(f"{col} {direction}")
        return parts

    def _normalize_col(self, expr: str) -> str:
        """Normalize a SELECT expression to its structural identity.

        Strips aliases, quotes, table prefixes. Preserves aggregation wrappers.
        """
        # Remove AS alias
        as_match = re.search(r'\bAS\b\s+\S+', expr, re.IGNORECASE)
        if as_match:
            expr = expr[:as_match.start()].strip()
        # Remove table prefix (T. or "T".)
        expr = re.sub(r'["\']?\w+["\']?\s*\.\s*', '', expr)
        # Remove quotes
        expr = expr.strip('"').strip("'")
        return expr
