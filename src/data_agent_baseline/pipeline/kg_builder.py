"""Property Graph Knowledge Graph builder from SQLite schema.

Builds a full property graph (Neo4j-style) from a SQLite database:
- Node types: TableNode, ColumnNode, ValueNode
- Edge types: HAS_COLUMN, FOREIGN_KEY, SEMANTIC_SIMILAR, CONTAINS_VALUE
- Weighted edges with overlap ratios, selectivity scores
- Pattern matching and multi-hop traversal support

Backward-compatible: KnowledgeGraph facade provides the old interface
(tables, inferred_fks, get_table, all_foreign_keys) while exposing the
new PropertyGraph via kg.graph.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


# ---------------------------------------------------------------------------
# Legacy dataclasses (kept for backward compatibility with existing code)
# ---------------------------------------------------------------------------


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
    # Structural role annotation (set by profile_schema)
    role: str = ""  # "fact" | "dimension" | "bridge" | "snapshot"
    grain_columns: list[str] = field(default_factory=list)  # columns defining row uniqueness
    measure_columns: list[str] = field(default_factory=list)  # numeric columns suitable for SUM/AVG
    temporal_columns: list[str] = field(default_factory=list)  # date/time/period columns
    measure_agg_level: dict[str, str] = field(default_factory=dict)  # col → "raw" | "pre_aggregated"


# ---------------------------------------------------------------------------
# Property Graph Node Types
# ---------------------------------------------------------------------------

VALUE_NODE_CARDINALITY_THRESHOLD = 500
VALUE_NODE_MAX_PER_COLUMN = 200
SEMANTIC_SIMILARITY_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class TableNode:
    id: str
    name: str
    row_count: int


@dataclass(frozen=True, slots=True)
class ColumnNode:
    id: str  # "table.column"
    table_id: str
    name: str
    sql_type: str
    is_pk: bool = False
    is_nullable: bool = True
    description: str = ""
    distinct_count: int = 0
    null_ratio: float = 0.0
    min_val: Any = None
    max_val: Any = None
    avg_val: float | None = None


@dataclass(frozen=True, slots=True)
class ValueNode:
    id: str  # "table.column::value"
    value: str
    column_id: str
    count: int = 0
    frequency: float = 0.0


# ---------------------------------------------------------------------------
# Property Graph Edge Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FKEdge:
    src: str  # ColumnNode.id (many-side)
    dst: str  # ColumnNode.id (one-side)
    overlap_ratio: float = 1.0
    direction: str = "inferred"  # "declared" | "inferred"
    validated: bool = True


@dataclass(frozen=True, slots=True)
class SemanticEdge:
    src: str  # ColumnNode.id
    dst: str  # ColumnNode.id
    similarity_score: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Property Graph Container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PropertyGraph:
    # Node stores (dict by ID for O(1) lookup)
    tables: dict[str, TableNode] = field(default_factory=dict)
    columns: dict[str, ColumnNode] = field(default_factory=dict)
    values: dict[str, ValueNode] = field(default_factory=dict)

    # Adjacency: table → columns
    has_column: dict[str, list[str]] = field(default_factory=dict)
    column_of: dict[str, str] = field(default_factory=dict)

    # FK edges (column-to-column)
    fk_edges: list[FKEdge] = field(default_factory=list)
    fk_from: dict[str, list[FKEdge]] = field(default_factory=dict)
    fk_to: dict[str, list[FKEdge]] = field(default_factory=dict)

    # Semantic similarity edges
    semantic_edges: list[SemanticEdge] = field(default_factory=list)
    sem_adj: dict[str, list[SemanticEdge]] = field(default_factory=dict)

    # Value edges
    contains_value: dict[str, list[str]] = field(default_factory=dict)
    value_in: dict[str, str] = field(default_factory=dict)

    # Value lookup index (normalized_value → [ValueNode.id])
    value_index: dict[str, list[str]] = field(default_factory=dict)

    # FK display map: fk_col_id → label_col_id (human-readable column for FK resolution)
    fk_display_map: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Graph Query Methods
    # ------------------------------------------------------------------

    def find_value(self, value: str) -> list[tuple[str, str, int]]:
        """Find columns containing a value. Returns [(table, column, count)]."""
        results: list[tuple[str, str, int]] = []
        normalized = value.strip().lower()

        # Exact match
        for vid in self.value_index.get(normalized, []):
            vnode = self.values.get(vid)
            if vnode:
                col_node = self.columns.get(vnode.column_id)
                if col_node:
                    results.append((col_node.table_id, col_node.name, vnode.count))

        # Substring match if no exact match
        if not results:
            for norm_val, vids in self.value_index.items():
                if normalized in norm_val or norm_val in normalized:
                    for vid in vids:
                        vnode = self.values.get(vid)
                        if vnode:
                            col_node = self.columns.get(vnode.column_id)
                            if col_node:
                                results.append((col_node.table_id, col_node.name, vnode.count))

        return results

    def get_fk_between(self, table_a: str, table_b: str) -> list[FKEdge]:
        """Get all FK edges connecting two tables (either direction)."""
        edges: list[FKEdge] = []
        for edge in self.fk_edges:
            src_table = self.column_of.get(edge.src, "")
            dst_table = self.column_of.get(edge.dst, "")
            if (src_table == table_a and dst_table == table_b) or \
               (src_table == table_b and dst_table == table_a):
                edges.append(edge)
        return sorted(edges, key=lambda e: -e.overlap_ratio)

    def get_table_columns(self, table_name: str) -> list[ColumnNode]:
        """Get all columns for a table."""
        col_ids = self.has_column.get(table_name, [])
        return [self.columns[cid] for cid in col_ids if cid in self.columns]

    def get_column_values(self, column_id: str) -> list[ValueNode]:
        """Get all materialized values for a column."""
        val_ids = self.contains_value.get(column_id, [])
        return [self.values[vid] for vid in val_ids if vid in self.values]

    def neighbors(self, column_id: str) -> list[tuple[str, float, str]]:
        """Get all connected columns via FK or semantic edges.
        Returns [(column_id, weight, edge_type)]."""
        results: list[tuple[str, float, str]] = []
        for edge in self.fk_from.get(column_id, []):
            results.append((edge.dst, edge.overlap_ratio, "fk"))
        for edge in self.fk_to.get(column_id, []):
            results.append((edge.src, edge.overlap_ratio, "fk"))
        for edge in self.sem_adj.get(column_id, []):
            other = edge.dst if edge.src == column_id else edge.src
            results.append((other, edge.similarity_score, "semantic"))
        return results


# ---------------------------------------------------------------------------
# KnowledgeGraph: backward-compatible facade over PropertyGraph
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class KnowledgeGraph:
    """Backward-compatible facade over PropertyGraph.

    Provides the old interface (tables, inferred_fks, get_table, all_foreign_keys)
    while exposing the rich property graph via .graph attribute.
    """
    _tables: list[TableSchema] = field(default_factory=list)
    _inferred_fks: list[tuple[str, ForeignKey]] = field(default_factory=list)
    graph: PropertyGraph = field(default_factory=PropertyGraph)
    # Dimensional model: fact_table, dimensions [{table, join_col, label_col}]
    dim_model: dict[str, Any] = field(default_factory=dict)
    # Semantic concept map: abstract_concept → "table.column"
    concept_map: dict[str, str] = field(default_factory=dict)
    # Ontology: "table.column" → {semantic_type, value_vocab, unit, derived_from, hierarchy}
    ontology: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Alias registry: canonical_value → {aliases: [...], table, column}
    alias_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Business topology: parent_entity → [child_entities] with table/column context
    business_topology: list[dict[str, Any]] = field(default_factory=list)
    # Document names loaded into knowledge (searchable via knowledge tool)
    doc_names: list[str] = field(default_factory=list)

    @property
    def tables(self) -> list[TableSchema]:
        return self._tables

    @tables.setter
    def tables(self, value: list[TableSchema]) -> None:
        self._tables = value

    @property
    def inferred_fks(self) -> list[tuple[str, ForeignKey]]:
        return self._inferred_fks

    @inferred_fks.setter
    def inferred_fks(self, value: list[tuple[str, ForeignKey]]) -> None:
        self._inferred_fks = value

    def get_table(self, name: str) -> TableSchema | None:
        for t in self._tables:
            if t.name == name:
                return t
        return None

    def all_foreign_keys(self) -> list[tuple[str, ForeignKey]]:
        """Return all FKs as (source_table, FK) pairs."""
        result = []
        for t in self._tables:
            for fk in t.foreign_keys:
                result.append((t.name, fk))
        for src, fk in self._inferred_fks:
            result.append((src, fk))
        return result


# ---------------------------------------------------------------------------
# KGQueryService: on-demand querying (replaces full dump approach)
# ---------------------------------------------------------------------------


class KGQueryService:
    """On-demand KG querying: produces minimal, targeted schema strings.

    Replaces format_kg_for_llm() dump with focused queries that return
    only what a specific prompt needs.
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self._full_cache: str | None = None

    def full_dump(self) -> str:
        """Same as format_kg_for_llm. Use only as fallback."""
        if self._full_cache is None:
            self._full_cache = format_kg_for_llm(self.kg)
        return self._full_cache

    def table_overview(self) -> str:
        """Lightweight: table names, roles, row counts, column names only."""
        lines = []
        for t in self.kg.tables:
            role_tag = f" [{t.role}]" if t.role else ""
            col_names = ", ".join(c.name for c in t.columns)
            lines.append(f"{t.name}{role_tag} ({t.row_count} rows): {col_names}")
        fks = self.kg.all_foreign_keys()
        if fks:
            lines.append("JOINS:")
            for src, fk in fks[:12]:
                lines.append(f"  {src}.{fk.column} -> {fk.ref_table}.{fk.ref_column}")
            # Show transitive paths (A->B->C) for bridge tables
            fk_graph: dict[str, list[tuple[str, str, str]]] = {}
            for src, fk in fks:
                fk_graph.setdefault(src, []).append((fk.column, fk.ref_table, fk.ref_column))
            paths: list[str] = []
            for start, edges in fk_graph.items():
                if len(edges) >= 2:
                    for col_a, ref_a, ref_col_a in edges:
                        for col_b, ref_b, ref_col_b in edges:
                            if ref_a != ref_b:
                                paths.append(f"  {ref_a} -> {start}({col_a},{col_b}) -> {ref_b}")
            if paths:
                lines.append("BRIDGE PATHS (multi-hop):")
                for p in sorted(set(paths))[:8]:
                    lines.append(p)
        return "\n".join(lines)

    def schema_for_tables(
        self, table_names: list[str], include_samples: bool = True, max_sample: int = 5
    ) -> str:
        """Full detail for specific tables only."""
        lines: list[str] = []
        tnames_lower = {n.lower() for n in table_names}
        for tname in table_names:
            ts = self.kg.get_table(tname)
            if not ts:
                continue
            pk_str = ", ".join(ts.primary_keys) if ts.primary_keys else "(none)"
            lines.append(f"TABLE: {tname} ({ts.row_count} rows, PK: {pk_str})")
            for col in ts.columns:
                nullable = "" if col.is_nullable else " NOT NULL"
                pk_mark = " [PK]" if col.is_pk else ""
                stats_str = ""
                if col.name in ts.col_stats:
                    st = ts.col_stats[col.name]
                    parts = []
                    if "distinct" in st:
                        parts.append(f"{st['distinct']} unique")
                    if "min" in st and "max" in st:
                        parts.append(f"range [{st['min']}..{st['max']}]")
                    if parts:
                        stats_str = f"  ({', '.join(parts)})"
                sample = ""
                if include_samples and col.name in ts.sample_values:
                    vals = ts.sample_values[col.name][:max_sample]
                    sample = f"  e.g. {vals}"
                desc_str = f"  -- {col.description}" if col.description else ""
                lines.append(
                    f"  - {col.name} ({col.sql_type}{nullable}){pk_mark}{desc_str}{stats_str}{sample}"
                )
            for fk in ts.foreign_keys:
                lines.append(f"  FK: {fk.column} -> {fk.ref_table}.{fk.ref_column}")
            lines.append("")
        # Inferred FKs between the selected tables
        for src, fk in self.kg.inferred_fks:
            if src.lower() in tnames_lower and fk.ref_table.lower() in tnames_lower:
                lines.append(f"  INFERRED FK: {src}.{fk.column} -> {fk.ref_table}.{fk.ref_column}")
        return "\n".join(lines)

    def join_paths_between(self, table_names: list[str]) -> str:
        """FK paths connecting the given tables."""
        tnames_lower = {n.lower() for n in table_names}
        lines: list[str] = []
        all_fks = self.kg.all_foreign_keys()
        for src, fk in all_fks:
            if src.lower() in tnames_lower or fk.ref_table.lower() in tnames_lower:
                lines.append(
                    f"JOIN {fk.ref_table} ON {src}.{fk.column} = {fk.ref_table}.{fk.ref_column}"
                )
        return "\n".join(lines)

    def focused_context(self, question: str, selected_tables: list[str] | None = None) -> str:
        """Produces a focused schema string for a specific question.

        If selected_tables is given, provides full detail for those tables
        plus a lightweight overview of others. Otherwise falls back to table_overview.
        """
        if not selected_tables:
            return self.table_overview()
        parts = [self.schema_for_tables(selected_tables)]
        # Add join paths
        joins = self.join_paths_between(selected_tables)
        if joins:
            parts.append(f"\n=== JOIN PATHS ===\n{joins}")
        # Add brief overview of other tables
        other_tables = [t.name for t in self.kg.tables if t.name not in selected_tables]
        if other_tables:
            other_lines = []
            for tname in other_tables:
                ts = self.kg.get_table(tname)
                if ts:
                    col_names = ", ".join(c.name for c in ts.columns[:8])
                    if len(ts.columns) > 8:
                        col_names += f" ... (+{len(ts.columns)-8})"
                    other_lines.append(f"  {tname} ({ts.row_count} rows): {col_names}")
            if other_lines:
                parts.append("\n=== OTHER TABLES ===\n" + "\n".join(other_lines))
        return "\n".join(parts)

    @staticmethod
    def tables_from_sql(sql: str) -> list[str]:
        """Extract table names from SQL FROM/JOIN clauses."""
        import re
        tables: list[str] = []
        for match in re.finditer(r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE):
            tname = match.group(1)
            if tname.upper() not in ("SELECT", "WHERE", "ON", "AND", "OR", "NULL"):
                tables.append(tname)
        return list(dict.fromkeys(tables))


# ---------------------------------------------------------------------------
# Construction: build_kg_from_sqlite
# ---------------------------------------------------------------------------


def build_kg_from_sqlite(db_path: Path) -> KnowledgeGraph:
    """Introspect SQLite DB and build a full property graph KnowledgeGraph."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Phase 1: Introspect tables (produces legacy TableSchema objects)
    tables = _discover_tables(conn)

    # Phase 2: Build property graph
    graph = PropertyGraph()

    # 2a: TableNodes + ColumnNodes
    for table in tables:
        tnode = TableNode(id=table.name, name=table.name, row_count=table.row_count)
        graph.tables[tnode.id] = tnode
        graph.has_column[tnode.id] = []

        for col in table.columns:
            stats = table.col_stats.get(col.name, {})
            null_count = 0
            if table.row_count > 0:
                distinct = stats.get("distinct", 0)
                null_ratio = 0.0
            else:
                null_ratio = 0.0

            col_node = ColumnNode(
                id=f"{table.name}.{col.name}",
                table_id=table.name,
                name=col.name,
                sql_type=col.sql_type,
                is_pk=col.is_pk,
                is_nullable=col.is_nullable,
                description=col.description,
                distinct_count=stats.get("distinct", 0),
                null_ratio=null_ratio,
                min_val=stats.get("min"),
                max_val=stats.get("max"),
                avg_val=stats.get("avg"),
            )
            graph.columns[col_node.id] = col_node
            graph.has_column[tnode.id].append(col_node.id)
            graph.column_of[col_node.id] = tnode.id

    # Phase 3: ValueNodes (categorical columns with low cardinality)
    _build_value_nodes(conn, tables, graph)

    # Phase 4: FK edges (inferred + declared)
    inferred_fks = _infer_foreign_keys_with_overlap(conn, tables, graph)

    # Add declared FK edges
    for table in tables:
        for fk in table.foreign_keys:
            src_id = f"{table.name}.{fk.column}"
            dst_id = f"{fk.ref_table}.{fk.ref_column}"
            if src_id in graph.columns and dst_id in graph.columns:
                edge = FKEdge(
                    src=src_id, dst=dst_id,
                    overlap_ratio=1.0, direction="declared", validated=True,
                )
                graph.fk_edges.append(edge)
                graph.fk_from.setdefault(src_id, []).append(edge)
                graph.fk_to.setdefault(dst_id, []).append(edge)

    # Phase 5: Semantic similarity edges
    _build_semantic_edges(graph)

    conn.close()

    # Build KnowledgeGraph facade
    kg = KnowledgeGraph(
        _tables=tables,
        _inferred_fks=inferred_fks,
        graph=graph,
    )
    return kg


def update_kg_with_new_tables(
    kg: KnowledgeGraph, db_path: Path, existing_table_names: list[str]
) -> KnowledgeGraph:
    """Incrementally update a KG with newly added tables (e.g. from doc extraction).

    Only processes tables not already in the graph. Adds nodes, value indexes,
    FK edges, and semantic edges for the new tables.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    all_tables = _discover_tables(conn)
    new_tables = [t for t in all_tables if t.name not in existing_table_names]

    if not new_tables:
        conn.close()
        return kg

    graph = kg.graph

    # Add TableNodes + ColumnNodes for new tables
    for table in new_tables:
        tnode = TableNode(id=table.name, name=table.name, row_count=table.row_count)
        graph.tables[tnode.id] = tnode
        graph.has_column[tnode.id] = []

        for col in table.columns:
            stats = table.col_stats.get(col.name, {})
            null_ratio = 0.0

            col_node = ColumnNode(
                id=f"{table.name}.{col.name}",
                table_id=table.name,
                name=col.name,
                sql_type=col.sql_type,
                is_pk=col.is_pk,
                is_nullable=col.is_nullable,
                description=col.description,
                distinct_count=stats.get("distinct", 0),
                null_ratio=null_ratio,
                min_val=stats.get("min"),
                max_val=stats.get("max"),
                avg_val=stats.get("avg"),
            )
            graph.columns[col_node.id] = col_node
            graph.has_column[tnode.id].append(col_node.id)
            graph.column_of[col_node.id] = tnode.id

    # Build value nodes for new tables
    _build_value_nodes(conn, new_tables, graph)

    # Infer FK edges between new tables and all tables
    new_fks = _infer_foreign_keys_with_overlap(conn, all_tables, graph)

    # Build semantic edges involving new tables
    _build_semantic_edges(graph)

    conn.close()

    # Update the KnowledgeGraph facade
    kg._tables.extend(new_tables)
    kg._inferred_fks.extend(new_fks)

    return kg


def _build_value_nodes(
    conn: sqlite3.Connection,
    tables: list[TableSchema],
    graph: PropertyGraph,
) -> None:
    """Materialize ValueNodes for categorical columns (low cardinality text columns)."""
    for table in tables:
        for col in table.columns:
            if col.sql_type.upper() not in ("TEXT", "VARCHAR", "CHAR", ""):
                continue
            stats = table.col_stats.get(col.name, {})
            distinct = stats.get("distinct", 0)
            if distinct == 0 or distinct > VALUE_NODE_CARDINALITY_THRESHOLD:
                continue

            col_id = f"{table.name}.{col.name}"
            try:
                rows = conn.execute(
                    f'SELECT "{col.name}", COUNT(*) as cnt FROM "{table.name}" '
                    f'WHERE "{col.name}" IS NOT NULL AND "{col.name}" != \'\' '
                    f'GROUP BY "{col.name}" ORDER BY cnt DESC LIMIT {VALUE_NODE_MAX_PER_COLUMN}'
                ).fetchall()

                val_ids: list[str] = []
                for row in rows:
                    value = str(row[0])
                    count = row[1]
                    freq = count / table.row_count if table.row_count > 0 else 0.0
                    vid = f"{col_id}::{value}"
                    vnode = ValueNode(
                        id=vid, value=value, column_id=col_id,
                        count=count, frequency=freq,
                    )
                    graph.values[vid] = vnode
                    val_ids.append(vid)
                    # Index by normalized value
                    normalized = value.strip().lower()
                    graph.value_index.setdefault(normalized, []).append(vid)
                    graph.value_in[vid] = col_id

                if val_ids:
                    graph.contains_value[col_id] = val_ids
            except Exception:
                pass


def _build_semantic_edges(graph: PropertyGraph) -> None:
    """Build SEMANTIC_SIMILAR edges between columns with similar names across tables."""
    col_ids = list(graph.columns.keys())

    for i in range(len(col_ids)):
        col_a = graph.columns[col_ids[i]]
        for j in range(i + 1, len(col_ids)):
            col_b = graph.columns[col_ids[j]]
            # Only cross-table
            if col_a.table_id == col_b.table_id:
                continue
            # Skip if already FK-linked
            already_fk = any(
                (e.src == col_ids[i] and e.dst == col_ids[j]) or
                (e.src == col_ids[j] and e.dst == col_ids[i])
                for e in graph.fk_edges
            )
            if already_fk:
                continue

            score = _compute_column_similarity(col_a, col_b)
            if score >= SEMANTIC_SIMILARITY_THRESHOLD:
                reason = _similarity_reason(col_a, col_b)
                edge = SemanticEdge(
                    src=col_ids[i], dst=col_ids[j],
                    similarity_score=score, reason=reason,
                )
                graph.semantic_edges.append(edge)
                graph.sem_adj.setdefault(col_ids[i], []).append(edge)
                graph.sem_adj.setdefault(col_ids[j], []).append(edge)


def _compute_column_similarity(a: ColumnNode, b: ColumnNode) -> float:
    """Compute similarity between two columns based on name and type.

    Only produces high scores for genuinely similar columns:
    - Same name (Diagnosis ↔ Diagnosis): 1.0
    - Strong word overlap (School Name ↔ SchoolName): 0.6+
    - Requires at least 50% token overlap to score above threshold
    """
    words_a = set(_tokenize_name(a.name))
    words_b = set(_tokenize_name(b.name))

    if not words_a or not words_b:
        return 0.0

    # Exact name match (case-insensitive)
    if a.name.lower() == b.name.lower():
        return 1.0

    # Jaccard on name tokens — requires real overlap
    intersection = words_a & words_b
    union = words_a | words_b
    jaccard = len(intersection) / len(union) if union else 0.0

    # Stem overlap: only for genuinely related words (5+ chars, one contains the other)
    stem_bonus = 0.0
    if jaccard == 0:
        stem_matches = 0
        for wa in words_a:
            for wb in words_b:
                if len(wa) >= 5 and len(wb) >= 5:
                    shorter = min(wa, wb, key=len)
                    longer = max(wa, wb, key=len)
                    if longer.startswith(shorter):
                        stem_matches += 1
                        break
        stem_bonus = min(stem_matches * 0.3, 0.4)

    # Type compatibility (only relevant if names already match)
    type_bonus = 0.1 if jaccard > 0 and a.sql_type.upper() == b.sql_type.upper() else 0.0

    total = jaccard + stem_bonus + type_bonus

    # Hard threshold: need at least 50% overlap or strong stem match
    if jaccard < 0.5 and stem_bonus == 0 and total < SEMANTIC_SIMILARITY_THRESHOLD:
        return 0.0

    return min(total, 1.0)


def _similarity_reason(a: ColumnNode, b: ColumnNode) -> str:
    """Generate a human-readable reason for similarity."""
    words_a = set(_tokenize_name(a.name))
    words_b = set(_tokenize_name(b.name))
    shared = words_a & words_b
    if shared:
        return f"shared words: {', '.join(sorted(shared))}"
    return f"similar names: {a.name} ~ {b.name}"


def _tokenize_name(name: str) -> list[str]:
    """Tokenize a column name into lowercase words."""
    # Split on camelCase, underscores, spaces
    parts = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    tokens = re.split(r'[_\s]+', parts.lower())
    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# FK inference (with overlap ratio tracking)
# ---------------------------------------------------------------------------


def _infer_foreign_keys_with_overlap(
    conn: sqlite3.Connection,
    tables: list[TableSchema],
    graph: PropertyGraph,
) -> list[tuple[str, ForeignKey]]:
    """Infer FK relationships and add FKEdge to graph with overlap ratios."""
    explicit_fk_cols: dict[str, set[str]] = {}
    for t in tables:
        explicit_fk_cols[t.name] = {fk.column.lower() for fk in t.foreign_keys}

    unique_cols: dict[str, list[tuple[str, str]]] = {}
    for t in tables:
        for col in t.columns:
            col_lower = col.name.lower()
            unique_cols.setdefault(col_lower, []).append((t.name, col.name))

    candidates: list[tuple[str, str, str, str, bool]] = []
    table_names_lower = {t.name.lower(): t.name for t in tables}

    for table in tables:
        for col in table.columns:
            col_lower = col.name.lower()
            if col_lower in explicit_fk_cols.get(table.name, set()):
                continue

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

            if col_lower in unique_cols and _is_joinable_column(col_lower):
                for ref_table_name, ref_col_name in unique_cols[col_lower]:
                    if ref_table_name == table.name:
                        continue
                    candidates.append((
                        table.name, col.name,
                        ref_table_name, ref_col_name,
                        _is_specific_id(col_lower),
                    ))

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

    # Cross-table overlap for uncovered pairs
    candidate_pairs: set[tuple[str, str]] = set()
    for src_t, _, ref_t, _, _ in candidates:
        candidate_pairs.add((min(src_t, ref_t), max(src_t, ref_t)))

    named_fk_cols: set[tuple[str, str]] = set()
    for table in tables:
        for col in table.columns:
            ref = _extract_ref_name(col.name)
            if ref and _resolve_table_name(ref, table_names_lower):
                named_fk_cols.add((table.name, col.name))

    for i, t1 in enumerate(tables):
        for t2 in tables[i + 1:]:
            pair_key = (min(t1.name, t2.name), max(t1.name, t2.name))
            if pair_key in candidate_pairs:
                continue
            t1_keys: list[Column] = []
            t2_keys: list[Column] = []
            for t, keys in [(t1, t1_keys), (t2, t2_keys)]:
                for c in t.columns:
                    if (t.name, c.name) in named_fk_cols:
                        continue
                    if c.is_pk:
                        keys.append(c)
                    elif t.row_count > 0:
                        stats = t.col_stats.get(c.name, {})
                        distinct = stats.get("distinct", 0)
                        if distinct > 0 and distinct / t.row_count >= 0.9:
                            keys.append(c)
            for c1 in t1_keys:
                for c2 in t2_keys:
                    candidates.append((t1.name, c1.name, t2.name, c2.name, False))
                    candidates.append((t2.name, c2.name, t1.name, c1.name, False))

    # Validate candidates
    inferred: list[tuple[str, ForeignKey]] = []
    seen: set[tuple[str, str, str, str]] = set()
    linked_pairs: set[tuple[str, str, str]] = set()
    uniqueness_cache: dict[tuple[str, str], float] = {}

    def _get_uniqueness(tbl: str, col: str) -> float:
        key = (tbl, col)
        if key not in uniqueness_cache:
            try:
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
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

        pair = tuple(sorted([src_table, ref_table]))
        col_pair = (pair[0], pair[1], src_col.lower())
        if col_pair in linked_pairs:
            continue

        overlap = _check_value_overlap(
            conn, src_table, src_col, ref_table, ref_col, name_match=name_match
        )
        if overlap is None:
            continue

        # Determine direction
        src_uniq = _get_uniqueness(src_table, src_col)
        ref_uniq = _get_uniqueness(ref_table, ref_col)

        if ref_uniq >= src_uniq:
            fk = ForeignKey(column=src_col, ref_table=ref_table, ref_column=ref_col)
            inferred.append((src_table, fk))
            src_id = f"{src_table}.{src_col}"
            dst_id = f"{ref_table}.{ref_col}"
        else:
            fk = ForeignKey(column=ref_col, ref_table=src_table, ref_column=src_col)
            inferred.append((ref_table, fk))
            src_id = f"{ref_table}.{ref_col}"
            dst_id = f"{src_table}.{src_col}"

        # Add FKEdge to graph
        if src_id in graph.columns and dst_id in graph.columns:
            edge = FKEdge(
                src=src_id, dst=dst_id,
                overlap_ratio=overlap, direction="inferred", validated=True,
            )
            graph.fk_edges.append(edge)
            graph.fk_from.setdefault(src_id, []).append(edge)
            graph.fk_to.setdefault(dst_id, []).append(edge)

        linked_pairs.add(col_pair)

    return inferred


# ---------------------------------------------------------------------------
# Helper functions (introspection, naming patterns, value overlap)
# ---------------------------------------------------------------------------


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

    try:
        count_row = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
        row_count = count_row[0] if count_row else 0
    except Exception:
        row_count = 0

    sample_values: dict[str, list[Any]] = {}
    for col in columns:
        try:
            if col.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", ""):
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

    col_stats: dict[str, dict[str, Any]] = {}
    for col in columns:
        stats: dict[str, Any] = {}
        try:
            row_info = conn.execute(
                f'SELECT COUNT(*), COUNT("{col.name}"), COUNT(DISTINCT "{col.name}") '
                f'FROM "{table_name}"'
            ).fetchone()
            total, non_null, distinct_count = row_info[0], row_info[1], row_info[2]
            stats["distinct"] = distinct_count
            stats["null_ratio"] = round(1.0 - non_null / total, 3) if total else 0.0
            stats["cardinality_ratio"] = round(distinct_count / total, 4) if total else 0.0

            is_numeric = col.sql_type.upper() in ("INTEGER", "INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE", "NUM")
            if is_numeric:
                row = conn.execute(
                    f'SELECT MIN("{col.name}"), MAX("{col.name}"), AVG("{col.name}") '
                    f'FROM "{table_name}" WHERE "{col.name}" IS NOT NULL'
                ).fetchone()
                if row and row[0] is not None:
                    stats["min"] = row[0]
                    stats["max"] = row[1]
                    stats["avg"] = round(row[2], 2) if row[2] is not None else None
                    # Percentiles (p25, p75) via offset on sorted values
                    if non_null >= 4:
                        p25_off = max(0, non_null // 4 - 1)
                        p75_off = max(0, (non_null * 3) // 4 - 1)
                        p25_row = conn.execute(
                            f'SELECT CAST("{col.name}" AS REAL) FROM "{table_name}" '
                            f'WHERE "{col.name}" IS NOT NULL ORDER BY CAST("{col.name}" AS REAL) '
                            f'LIMIT 1 OFFSET {p25_off}'
                        ).fetchone()
                        p75_row = conn.execute(
                            f'SELECT CAST("{col.name}" AS REAL) FROM "{table_name}" '
                            f'WHERE "{col.name}" IS NOT NULL ORDER BY CAST("{col.name}" AS REAL) '
                            f'LIMIT 1 OFFSET {p75_off}'
                        ).fetchone()
                        if p25_row:
                            stats["p25"] = round(p25_row[0], 2)
                        if p75_row:
                            stats["p75"] = round(p75_row[0], 2)
            else:
                # Text columns: top-1 value and frequency
                if distinct_count and distinct_count <= 200 and non_null:
                    top_row = conn.execute(
                        f'SELECT "{col.name}", COUNT(*) as cnt FROM "{table_name}" '
                        f'WHERE "{col.name}" IS NOT NULL GROUP BY "{col.name}" '
                        f'ORDER BY cnt DESC LIMIT 1'
                    ).fetchone()
                    if top_row:
                        stats["top1_value"] = top_row[0]
                        stats["top1_freq"] = round(top_row[1] / non_null, 3)
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
    if ref_name + "s" in table_names_lower:
        return ref_name + "s"
    if ref_name + "es" in table_names_lower:
        return ref_name + "es"
    if ref_name + "ies" in table_names_lower:
        return ref_name + "ies"
    if ref_name.endswith("s") and ref_name[:-1] in table_names_lower:
        return ref_name[:-1]
    if ref_name.endswith("es") and ref_name[:-2] in table_names_lower:
        return ref_name[:-2]
    if ref_name.endswith("ies") and ref_name[:-3] + "y" in table_names_lower:
        return ref_name[:-3] + "y"
    if "_" in ref_name:
        joined = ref_name.replace("_", "")
        if joined in table_names_lower:
            return joined
        # Try suffix match: "eye_colour" → check if "colour" is a table
        parts = ref_name.split("_")
        for i in range(1, len(parts)):
            suffix = "_".join(parts[i:])
            if suffix in table_names_lower:
                return suffix
            if suffix + "s" in table_names_lower:
                return suffix + "s"
    else:
        for tname in table_names_lower:
            if tname.replace("_", "") == ref_name:
                return tname
    return None


def _find_id_column(
    ref_table_name: str, src_col_name: str, tables: list[TableSchema]
) -> str:
    """Find the best ID column in the referenced table."""
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
    for c in ref_table.columns:
        if c.is_pk:
            return c.name
    return ref_table.columns[0].name if ref_table.columns else "id"


def _extract_ref_name(col_name: str) -> str | None:
    """Extract referenced entity name from FK naming conventions."""
    col_lower = col_name.lower()
    m = re.match(r"^(.+?)_(?:id|key|no|num|code)$", col_lower)
    if m:
        return m.group(1)
    m = re.match(r"^(?:fk|ref|id)_(.+)$", col_lower)
    if m:
        return m.group(1)
    m = re.match(r"^(.+?)(?:Id|ID|Key|Code|No|Num)$", col_name)
    if m and len(m.group(1)) > 1:
        return m.group(1).lower()
    return None


def _is_joinable_column(col_name: str) -> bool:
    """Check if a column name looks like a join key."""
    col_lower = col_name.lower()
    return (
        col_lower == "id"
        or col_lower.endswith("_id")
        or col_lower.endswith("id") and len(col_lower) > 2
        or col_lower.endswith("code") or col_lower.endswith("_code")
        or col_lower.endswith("_key") or col_lower.endswith("_no")
        or col_lower in ("key", "code", "number", "no", "num")
    )


def _is_specific_id(col_name: str) -> bool:
    """Check if a column name is a specific (non-generic) ID."""
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
) -> float | None:
    """Check value overlap. Returns overlap ratio (0.0-1.0) or None if no overlap."""
    try:
        ref_cols = conn.execute(f'PRAGMA table_info("{ref_table}")').fetchall()
        ref_col_names = [r[1].lower() for r in ref_cols]
        if ref_col.lower() not in ref_col_names:
            for actual_name in [r[1] for r in ref_cols]:
                if actual_name.lower() == "id":
                    ref_col = actual_name
                    break
            else:
                return None

        is_one_to_many = False
        if src_col.lower() == "id" and ref_col.lower() == "id":
            if _both_are_pks(conn, src_table, src_col, ref_table, ref_col):
                return None
            is_one_to_many = True

        src_vals = conn.execute(
            f'SELECT DISTINCT "{src_col}" FROM "{src_table}" '
            f'WHERE "{src_col}" IS NOT NULL LIMIT {sample_size}'
        ).fetchall()

        if not src_vals:
            return None

        if not name_match and not is_one_to_many:
            try:
                src_row_count = conn.execute(
                    f'SELECT COUNT(*) FROM "{src_table}"'
                ).fetchone()[0]
                src_distinct = conn.execute(
                    f'SELECT COUNT(DISTINCT "{src_col}") FROM "{src_table}" '
                    f'WHERE "{src_col}" IS NOT NULL'
                ).fetchone()[0]
                if src_row_count > 0 and src_distinct / src_row_count < 0.5:
                    return None
            except Exception:
                pass

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
            return overlap if matches > 0 else None

        return overlap if overlap >= min_overlap else None

    except Exception:
        return None


def _both_are_pks(
    conn: sqlite3.Connection,
    table_a: str, col_a: str,
    table_b: str, col_b: str,
) -> bool:
    """Check if both id columns are primary keys of their respective tables."""
    try:
        count_a = conn.execute(f'SELECT COUNT(*) FROM "{table_a}"').fetchone()[0]
        distinct_a = conn.execute(
            f'SELECT COUNT(DISTINCT "{col_a}") FROM "{table_a}"'
        ).fetchone()[0]
        count_b = conn.execute(f'SELECT COUNT(*) FROM "{table_b}"').fetchone()[0]
        distinct_b = conn.execute(
            f'SELECT COUNT(DISTINCT "{col_b}") FROM "{table_b}"'
        ).fetchone()[0]
        a_is_unique = distinct_a >= count_a * 0.95
        b_is_unique = distinct_b >= count_b * 0.95
        return a_is_unique and b_is_unique
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LLM-based enrichment + formatting (public API, backward compatible)
# ---------------------------------------------------------------------------


def format_kg_for_llm(kg: KnowledgeGraph, max_sample_values: int = 8) -> str:
    """Format the KG metadata as a compact text string for LLM context."""
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

        for fk in table.foreign_keys:
            lines.append(f"  FK: {fk.column} → {fk.ref_table}.{fk.ref_column}")

        lines.append("")

    if kg.inferred_fks:
        lines.append("=== INFERRED RELATIONSHIPS ===")
        for src_table, fk in kg.inferred_fks:
            # Include overlap ratio from graph if available
            overlap_str = ""
            if kg.graph:
                src_id = f"{src_table}.{fk.column}"
                dst_id = f"{fk.ref_table}.{fk.ref_column}"
                for edge in kg.graph.fk_edges:
                    if edge.src == src_id and edge.dst == dst_id:
                        overlap_str = f" (overlap: {edge.overlap_ratio:.0%})"
                        break
            lines.append(f"  {src_table}.{fk.column} → {fk.ref_table}.{fk.ref_column}{overlap_str}")
        lines.append("")

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
    if len(schema_text) > 6000:
        schema_text = schema_text[:6000]

    prompt = _ENRICH_PROMPT.format(
        schema=schema_text,
        knowledge_text=knowledge_text[:3000] if knowledge_text else "(none)",
    )
    messages = [ModelMessage(role="user", content=prompt)]
    raw = model.complete(messages, thinking=False)

    descriptions: dict[str, str] = {}
    try:
        descriptions = json.loads(raw)
    except json.JSONDecodeError:
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

    # Update graph ColumnNodes with descriptions
    new_graph = kg.graph
    if new_graph and descriptions:
        new_columns = dict(new_graph.columns)
        for col_key, desc in descriptions.items():
            if desc and col_key in new_columns:
                old_node = new_columns[col_key]
                new_columns[col_key] = ColumnNode(
                    id=old_node.id,
                    table_id=old_node.table_id,
                    name=old_node.name,
                    sql_type=old_node.sql_type,
                    is_pk=old_node.is_pk,
                    is_nullable=old_node.is_nullable,
                    description=desc,
                    distinct_count=old_node.distinct_count,
                    null_ratio=old_node.null_ratio,
                    min_val=old_node.min_val,
                    max_val=old_node.max_val,
                    avg_val=old_node.avg_val,
                )
        new_graph.columns = new_columns

    return KnowledgeGraph(_tables=new_tables, _inferred_fks=kg.inferred_fks, graph=new_graph)


# ---------------------------------------------------------------------------
# Ontology builder: sequential small LLM calls
# ---------------------------------------------------------------------------


def _is_obvious_column(col: Column, table: "TableSchema") -> str | None:
    """Return a deterministic semantic_type if the column is trivially classifiable, else None."""
    cl = col.name.lower()
    ct = col.sql_type.upper()

    if col.is_pk or cl == "id" or cl == "_id" or cl.endswith("_id") or cl.endswith("id"):
        return "identifier"
    if cl.startswith("link_to") or cl.startswith("fk_"):
        return "identifier"
    if any(kw in cl for kw in ("date", "time", "timestamp", "created_at", "updated_at")):
        return "timestamp"
    # Year columns (e.g., "year", "birth_year")
    if cl == "year" or cl.endswith("_year"):
        return "timestamp"

    stats = table.col_stats.get(col.name, {})
    cr = stats.get("cardinality_ratio", 0)
    if ct in ("TEXT", "VARCHAR", "CHAR") and cr and cr > 0.5 and table.row_count > 20:
        return "free_text"

    return None


def _parse_json_response(raw: str) -> dict | None:
    """Extract JSON object from LLM response."""
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _step_classify(
    ambiguous_cols: list[tuple["TableSchema", Column]],
    model: "ModelAdapter",
    log_fn: Any = None,
) -> dict[str, str]:
    """Step 1: Classify ambiguous columns by semantic_type. Fast — tiny output."""
    if log_fn:
        log_fn("kg_onto", f"classifying {len(ambiguous_cols)} columns...")
    col_lines = []
    for t, c in ambiguous_cols:
        samples = t.sample_values.get(c.name, [])
        sample_str = f" e.g. {samples[:4]}" if samples else ""
        col_lines.append(f"  {t.name}.{c.name} ({c.sql_type}){sample_str}")

    prompt = f"""Classify each column's semantic type.

COLUMNS:
{chr(10).join(col_lines)}

Return JSON: {{"table.column": "type"}}
Types: flag, category, currency, rate, count, score, duration, percentage, identifier, timestamp, free_text
"""
    from data_agent_baseline.agents.model import ModelMessage
    messages = [ModelMessage(role="user", content=prompt)]
    raw = model.complete(messages, thinking=False)
    if not raw:
        return {}
    result = _parse_json_response(raw)
    if isinstance(result, dict):
        valid = {k: v for k, v in result.items() if isinstance(v, str)}
        if log_fn:
            log_fn("kg_onto_classify", f"done — {len(valid)} types assigned")
        return valid
    return {}


_VOCAB_BATCH_SIZE = 7


def _step_decode_vocab(
    category_cols: list[tuple["TableSchema", Column]],
    model: "ModelAdapter",
    log_fn: Any = None,
) -> dict[str, dict[str, str]]:
    """Step 2: Decode value vocabularies for flag/category columns only.

    Batches into chunks to avoid LLM timeout on large column sets.
    """
    if not category_cols:
        return {}
    if log_fn:
        log_fn("kg_onto", f"decoding vocab for {len(category_cols)} flag/category columns...")

    from data_agent_baseline.agents.model import ModelMessage

    all_valid: dict[str, dict[str, str]] = {}

    for batch_start in range(0, len(category_cols), _VOCAB_BATCH_SIZE):
        batch = category_cols[batch_start:batch_start + _VOCAB_BATCH_SIZE]

        col_lines = []
        for t, c in batch:
            samples = t.sample_values.get(c.name, [])
            sample_str = f" values: {samples[:8]}" if samples else ""
            col_lines.append(f"  {t.name}.{c.name}{sample_str}")

        prompt = f"""For each column, explain what the column REPRESENTS and decode its stored values.

COLUMNS:
{chr(10).join(col_lines)}

Return JSON: {{"table.column": {{"_purpose": "what this column represents (1-5 words)", "stored_value": "human meaning", ...}}}}
Rules:
- _purpose: describes what the column categorizes (e.g., "transaction direction", "payment method", "account status")
- Each stored_value gets a short human-readable meaning
- Skip columns where values are already readable English words
"""
        messages = [ModelMessage(role="user", content=prompt)]
        raw = model.complete(messages, thinking=False)
        if not raw:
            continue
        result = _parse_json_response(raw)
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, dict):
                    all_valid[k] = {sk: sv for sk, sv in v.items() if isinstance(sv, str)}

    if log_fn:
        log_fn("kg_onto_vocab", f"done — {len(all_valid)} columns decoded")
    return all_valid


def _step_concepts(
    kg: KnowledgeGraph,
    types_map: dict[str, str],
    model: "ModelAdapter",
    log_fn: Any = None,
) -> dict[str, str]:
    """Step 3: Map abstract user terms to columns, informed by type classification."""
    if log_fn:
        log_fn("kg_onto", "mapping concept words...")
    col_lines = []
    for t in kg.tables:
        for c in t.columns:
            col_ref = f"{t.name}.{c.name}"
            stype = types_map.get(col_ref, "")
            type_hint = f" [{stype}]" if stype else ""
            col_lines.append(f"  {col_ref}{type_hint}")

    prompt = f"""Map abstract user concepts to the best column. A "concept" is a word a user might say that doesn't literally match a column name.

COLUMNS:
{chr(10).join(col_lines)}

Return JSON: {{"concept_word": "table.column"}}
Rules:
- Only non-obvious mappings (skip "name"→name, "date"→date)
- Max 10 entries
"""
    from data_agent_baseline.agents.model import ModelMessage
    messages = [ModelMessage(role="user", content=prompt)]
    raw = model.complete(messages, thinking=False)
    if not raw:
        return {}
    result = _parse_json_response(raw)
    if isinstance(result, dict):
        all_cols = {f"{t.name}.{c.name}" for t in kg.tables for c in t.columns}
        valid = {}
        for concept, col_ref in result.items():
            if isinstance(concept, str) and isinstance(col_ref, str) and col_ref in all_cols:
                valid[concept.lower()] = col_ref
        if log_fn:
            log_fn("kg_onto_concepts", f"{len(valid)} concepts mapped")
        return valid
    return {}


def build_ontology(
    kg: KnowledgeGraph,
    model: "ModelAdapter",
    db_path: Path | None = None,
    log_fn: Any = None,
) -> dict[str, dict[str, Any]]:
    """Build semantic ontology via sequential small LLM calls.

    Steps:
      1. Deterministic pre-classification (instant)
      2. LLM classify remaining columns by type (~2-5s)
      3. LLM decode vocab for flag/category columns only (~2-5s)
      4. LLM map concept words (~2-5s)

    Sets kg.concept_map as a side effect.
    Returns dict of "table.column" → {semantic_type, value_vocab, unit, ...}.
    """
    if not kg or not kg.tables:
        return {}

    # Step 0: Deterministic pre-classification
    ontology: dict[str, dict[str, Any]] = {}
    ambiguous_cols: list[tuple["TableSchema", Column]] = []

    for t in kg.tables:
        for c in t.columns:
            obvious_type = _is_obvious_column(c, t)
            if obvious_type:
                ontology[f"{t.name}.{c.name}"] = {"semantic_type": obvious_type}
            else:
                ambiguous_cols.append((t, c))

    if not ambiguous_cols:
        if log_fn:
            log_fn("kg_ontology", f"{len(ontology)} columns (all deterministic), 0 concepts")
        return ontology

    # Step 1: Classify ambiguous columns
    types_map = _step_classify(ambiguous_cols, model, log_fn)
    for col_ref, stype in types_map.items():
        ontology.setdefault(col_ref, {})["semantic_type"] = stype

    # Step 2: Decode vocab for flag/category columns only
    category_cols = [
        (t, c) for t, c in ambiguous_cols
        if types_map.get(f"{t.name}.{c.name}") in ("flag", "category")
    ]
    if category_cols:
        vocab_map = _step_decode_vocab(category_cols, model, log_fn)
        for col_ref, vocab in vocab_map.items():
            entry = ontology.setdefault(col_ref, {})
            # Extract _purpose if present, store separately
            purpose = vocab.pop("_purpose", None)
            if purpose:
                entry["purpose"] = purpose
            entry["value_vocab"] = vocab

    # Step 3: Concept map
    # Merge deterministic types into types_map for context
    full_types = {k: v.get("semantic_type", "") for k, v in ontology.items()}
    kg.concept_map = _step_concepts(kg, full_types, model, log_fn)

    if log_fn:
        det_count = sum(1 for t in kg.tables for c in t.columns if _is_obvious_column(c, t))
        log_fn("kg_ontology", f"{len(ontology)} columns ({det_count} deterministic + {len(ontology)-det_count} LLM), {len(kg.concept_map)} concepts")
    return ontology


def build_concept_map(
    kg: KnowledgeGraph,
    model: "ModelAdapter",
    knowledge_text: str = "",
    log_fn: Any = None,
) -> dict[str, str]:
    """Deprecated — concept_map is now built inside build_ontology. Returns existing map."""
    return kg.concept_map


_RELATIONSHIP_DISCOVERY_PROMPT = """\
You are a database schema expert. Analyze this schema and identify ALL table relationships.

SCHEMA:
{schema}

Return ONLY a JSON object with two keys:

{{
  "tables": {{
    "table_name": {{
      "pk": "column_name",
      "label_column": "column_name or null",
      "is_bridge": true/false
    }}
  }},
  "relationships": [
    {{
      "from": "table.column",
      "to": "table.column",
      "cardinality": "many_to_one|one_to_one|many_to_many",
      "from_display": "table.column or null"
    }}
  ]
}}

RULES:
- "pk": the primary key column of each table (unique row identifier). Every table has one.
- "label_column": the human-readable name/title column users want to see (e.g. "event_name", "colour"). null if table has no natural label.
- "is_bridge": true if the table is a junction/bridge connecting two other tables (e.g. "hero_power" connecting "superhero" and "superpower"). Bridge tables have 2+ FK columns and no useful domain data of their own.
- "from": the FK column (many-side) that references another table's PK.
- "to": the PK column (one-side) being referenced.
- "cardinality": "many_to_one" (most FKs), "one_to_one" (both are unique), "many_to_many" (through a bridge table — list both FKs of the bridge).
- "from_display": for FKs, the label column in the referenced (to) table — what users want to see instead of the ID. E.g. for "budget.link_to_event" → from_display: "event.event_name".
- Look for patterns: "link_to_X", "X_id", "Xid" columns that reference table X's PK.
- Sample values confirm relationships — matching values between tables = FK relationship.
- Do NOT skip any relationships. List ALL of them even if obvious from naming."""


def discover_joins_with_llm(
    kg: KnowledgeGraph,
    model: ModelAdapter,
    log_fn: Callable[..., None] | None = None,
) -> KnowledgeGraph:
    """Use LLM to discover all table relationships: PKs, FKs, cardinality, bridge tables, display columns."""
    if log_fn:
        log_fn("kg_joins", f"discovering relationships across {len(kg.tables)} tables...")
    schema_lines: list[str] = []
    for table in kg.tables:
        schema_lines.append(f"TABLE: {table.name} ({table.row_count} rows)")
        for col in table.columns:
            sample = ""
            if col.name in table.sample_values:
                vals = table.sample_values[col.name][:5]
                sample = f"  e.g. {vals}"
            pk_marker = " [PK]" if col.is_pk else ""
            schema_lines.append(f"  - {col.name} ({col.sql_type}){pk_marker}{sample}")
        schema_lines.append("")

    schema_text = "\n".join(schema_lines)
    if len(schema_text) > 6000:
        schema_text = schema_text[:6000]

    prompt = _RELATIONSHIP_DISCOVERY_PROMPT.format(schema=schema_text)
    messages = [ModelMessage(role="user", content=prompt)]
    raw = model.complete(messages, thinking=False)

    result: dict = {}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*\n(.+?)```", raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    if not result or not isinstance(result, dict):
        if log_fn:
            log_fn("kg_llm_fk", "Failed to parse LLM relationship discovery response")
        return kg

    graph = kg.graph
    if not graph:
        return kg

    # --- Apply table-level info (PK, label, bridge) ---
    tables_info = result.get("tables", {})
    for table_name, info in tables_info.items():
        if not isinstance(info, dict):
            continue
        label_col = info.get("label_column")
        if label_col:
            label_id = _fuzzy_find_col(f"{table_name}.{label_col}", graph)
            if label_id:
                # Store as self-display (table's own label column)
                graph.fk_display_map[f"_label_.{table_name}"] = label_id

    # --- Apply relationships ---
    relationships = result.get("relationships", [])
    if not isinstance(relationships, list):
        relationships = []

    added = 0
    existing_edges = {(e.src, e.dst) for e in graph.fk_edges}
    fk_display_map: dict[str, str] = {}

    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        from_ref = rel.get("from", "")
        to_ref = rel.get("to", "")
        if "." not in from_ref or "." not in to_ref:
            continue

        # Validate both columns exist in graph
        from_id = _fuzzy_find_col(from_ref, graph)
        to_id = _fuzzy_find_col(to_ref, graph)
        if not from_id or not to_id:
            continue
        if from_id == to_id:
            continue
        if (from_id, to_id) in existing_edges or (to_id, from_id) in existing_edges:
            # Still capture display column for existing edges
            from_display = rel.get("from_display")
            if from_display:
                display_id = _fuzzy_find_col(from_display, graph)
                if display_id:
                    fk_display_map[from_id] = display_id
            continue

        edge = FKEdge(
            src=from_id, dst=to_id,
            overlap_ratio=0.9,
            direction="llm_discovered", validated=False,
        )
        graph.fk_edges.append(edge)
        graph.fk_from.setdefault(from_id, []).append(edge)
        graph.fk_to.setdefault(to_id, []).append(edge)
        existing_edges.add((from_id, to_id))
        added += 1

        # Capture display column
        from_display = rel.get("from_display")
        if from_display:
            display_id = _fuzzy_find_col(from_display, graph)
            if display_id:
                fk_display_map[from_id] = display_id

    graph.fk_display_map.update(fk_display_map)

    # --- Deterministic label column fallback ---
    # If LLM didn't provide label columns, auto-detect obvious ones
    _NAME_PATTERNS = re.compile(
        r"(name|title|label|description|display_name|full_name|event_name|"
        r"product_name|category_name|city|country|region)$", re.IGNORECASE
    )
    for table in kg.tables:
        label_key = f"_label_.{table.name}"
        if label_key in graph.fk_display_map:
            continue
        for col in table.columns:
            if col.is_pk:
                continue
            if col.sql_type.upper() not in ("TEXT", "VARCHAR", "NVARCHAR", "CHAR"):
                continue
            if _NAME_PATTERNS.search(col.name):
                graph.fk_display_map[label_key] = f"{table.name}.{col.name}"
                break

    total_fks = len(graph.fk_edges)
    label_count = sum(1 for k in graph.fk_display_map if k.startswith("_label_."))
    if log_fn:
        log_fn("kg_llm_fk", f"{added} new FKs ({total_fks} total), {label_count} label columns, {len(fk_display_map)} display mappings")

    return kg


def classify_columns_with_llm(
    kg: KnowledgeGraph,
    model: ModelAdapter,
    log_fn: Callable[..., None] | None = None,
) -> KnowledgeGraph:
    """No-op — relationship discovery now handles FK + display column mapping."""
    if log_fn:
        log_fn("kg_classify", "merged into discover_joins_with_llm")
    return kg


def _fuzzy_find_col(ref: str, graph: PropertyGraph) -> str | None:
    """Case-insensitive column lookup in the graph."""
    if ref in graph.columns:
        return ref
    ref_lower = ref.lower()
    for col_id in graph.columns:
        if col_id.lower() == ref_lower:
            return col_id
    return None


# ---------------------------------------------------------------------------
# Schema Profiler — structural role annotation
# ---------------------------------------------------------------------------

def profile_schema(kg: KnowledgeGraph) -> None:
    """Annotate tables with structural roles derived from topology and statistics.

    Roles:
      - dimension: low cardinality, referenced by FKs, no outgoing FKs (or few)
      - fact: has measures + multiple FK references to dimensions
      - bridge: connects two entities (2+ FK columns, few/no measures)
      - snapshot: entity × time grain (has entity FK + temporal column + measures)

    This runs once after KG build. Downstream logic reads table.role,
    table.grain_columns, table.measure_columns instead of re-deriving heuristics.
    """
    if not kg or not kg.tables:
        return

    # Pre-compute FK topology per table
    fk_out: dict[str, list[str]] = {}  # table → [target_tables]
    fk_in: dict[str, int] = {}  # table → count of tables referencing it
    for src_table, fk in kg.all_foreign_keys():
        fk_out.setdefault(src_table, []).append(fk.ref_table)
        fk_in[fk.ref_table] = fk_in.get(fk.ref_table, 0) + 1

    for table in kg.tables:
        n = table.name
        cols = table.columns
        num_fk_out = len(fk_out.get(n, []))
        num_fk_in = fk_in.get(n, 0)

        # Classify columns
        id_cols: list[str] = []
        measures: list[str] = []
        temporal: list[str] = []
        for c in cols:
            cl = c.name.lower()
            t = c.sql_type.upper()
            is_numeric = t in ("INT", "INTEGER", "REAL", "FLOAT", "NUMERIC", "NUM", "DOUBLE")
            is_id = cl.endswith("id") or cl.startswith("link") or c.is_pk
            is_temporal = any(kw in cl for kw in ("date", "year", "month", "time", "period"))

            if is_temporal:
                temporal.append(c.name)
            elif is_id:
                id_cols.append(c.name)
            elif is_numeric:
                measures.append(c.name)

        table.measure_columns = measures
        table.temporal_columns = temporal

        # Determine role
        if num_fk_out >= 2 and not measures:
            # 2+ outgoing FKs, no measures → bridge/junction table
            table.role = "bridge"
            table.grain_columns = id_cols[:2] if len(id_cols) >= 2 else id_cols
        elif num_fk_out >= 1 and temporal and measures:
            # FK + temporal + measures → snapshot (entity × time)
            table.role = "snapshot"
            table.grain_columns = [id_cols[0]] if id_cols else []
            if temporal:
                table.grain_columns.append(temporal[0])
        elif num_fk_out >= 2 and measures:
            # Multiple FKs + measures → fact/transaction table
            table.role = "fact"
            table.grain_columns = id_cols
        elif num_fk_out <= 1 and num_fk_in >= 1 and table.row_count < 500:
            # Referenced by others, small, few outgoing → dimension
            table.role = "dimension"
            table.grain_columns = [cols[0].name] if cols else []
        elif num_fk_out >= 1 and measures:
            # Has FK + measures but only 1 FK → still fact-like
            table.role = "fact"
            table.grain_columns = id_cols
        else:
            # Default: if it has measures treat as fact, otherwise dimension
            if measures:
                table.role = "fact"
            else:
                table.role = "dimension"
            table.grain_columns = id_cols or ([cols[0].name] if cols else [])

        # Layer 1: Normalization — classify measures as raw vs pre-aggregated
        for m in measures:
            if table.role == "snapshot":
                table.measure_agg_level[m] = "pre_aggregated"
            else:
                table.measure_agg_level[m] = "raw"

    # Layer 2: Dimensional model map
    _build_dimensional_map(kg)


def _build_dimensional_map(kg: KnowledgeGraph) -> None:
    """Build a star/snowflake model identifying the central fact and its dimensions."""
    # Find the fact table (highest row count among fact/snapshot tables)
    fact_table = None
    for t in sorted(kg.tables, key=lambda x: -x.row_count):
        if t.role in ("fact", "snapshot", "bridge"):
            fact_table = t
            break
    if not fact_table:
        return

    # Build dimensions: tables referenced by fact table's FKs
    dimensions: list[dict[str, str]] = []
    all_fks = kg.all_foreign_keys()
    for src_table, fk in all_fks:
        if src_table != fact_table.name:
            continue
        dim_schema = kg.get_table(fk.ref_table)
        if not dim_schema:
            continue
        # Find best label column: first TEXT non-PK non-ID column, or PK
        label_col = fk.ref_column
        for c in dim_schema.columns:
            cl = c.name.lower()
            if c.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR") and not cl.endswith("id"):
                label_col = c.name
                break
        dimensions.append({
            "table": fk.ref_table,
            "join_col": fk.column,
            "ref_col": fk.ref_column,
            "label_col": label_col,
            "role": dim_schema.role,
        })

    kg.dim_model = {
        "fact_table": fact_table.name,
        "fact_role": fact_table.role,
        "fact_measures": fact_table.measure_columns,
        "fact_grain": fact_table.grain_columns,
        "dimensions": dimensions,
    }


# ---------------------------------------------------------------------------
# Alias Registry: map user terms to canonical DB values
# ---------------------------------------------------------------------------

_ALIAS_PROMPT = """\
You are a data dictionary expert. Given these distinct values from a database column, \
generate common aliases/synonyms a user might type when referring to each value.

TABLE: {table}
COLUMN: {column}
CONTEXT: {context}

VALUES:
{values}

For each value, produce aliases that users commonly use: abbreviations, full names, \
informal variants, codes, and natural language references.

Return ONLY a JSON object (no markdown fences):
{{"aliases": {{
  "<canonical_value>": ["<alias1>", "<alias2>", ...],
  ...
}}}}

Rules:
- Only include values that have non-obvious aliases (skip if the value IS the obvious term)
- Include: abbreviations ↔ full names, codes ↔ descriptions, informal names
- Max 5 aliases per value
- Lowercase all aliases"""


def build_alias_registry(
    kg: KnowledgeGraph,
    model: "ModelAdapter",
    db_path: Path,
    log_fn: Any = None,
) -> None:
    """Build alias registry for low-cardinality identifier/category columns.

    Identifies columns where users might use alternative names (machine IDs,
    product codes, status values, etc.) and generates aliases via LLM.
    """
    if not kg or not kg.tables:
        return

    # Find candidate columns: text columns with 2-50 distinct values
    # that serve as identifiers or categories (not free text)
    candidates: list[tuple[TableSchema, Column, list[str]]] = []
    conn = sqlite3.connect(str(db_path))

    for table in kg.tables:
        for col in table.columns:
            stats = table.col_stats.get(col.name, {})
            distinct = stats.get("distinct", 0)
            if not (2 <= distinct <= 50):
                continue
            if col.sql_type.upper() not in ("TEXT", "VARCHAR", "CHAR", ""):
                continue
            # Skip columns that are clearly free-text (high cardinality ratio)
            if stats.get("cardinality_ratio", 0) > 0.8 and table.row_count > 20:
                continue

            try:
                rows = conn.execute(
                    f'SELECT DISTINCT "{col.name}" FROM "{table.name}" '
                    f'WHERE "{col.name}" IS NOT NULL AND "{col.name}" != "" '
                    f'ORDER BY "{col.name}" LIMIT 50'
                ).fetchall()
                values = [r[0] for r in rows if r[0]]
                if len(values) >= 2:
                    candidates.append((table, col, values))
            except Exception:
                pass

    conn.close()

    if not candidates:
        if log_fn:
            log_fn("alias_registry", "0 candidate columns")
        return

    # Batch candidates into a single LLM call (group by semantic relevance)
    # For efficiency, process max 5 columns per call
    all_aliases: dict[str, dict[str, Any]] = {}
    batch_size = 5

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        values_section = ""
        batch_context = []
        for table, col, values in batch:
            values_section += f"\n[{table.name}.{col.name}]: {', '.join(str(v) for v in values[:20])}\n"
            batch_context.append(f"{table.name}.{col.name} ({table.role})")

        prompt = _ALIAS_PROMPT.format(
            table=", ".join(t.name for t, _, _ in batch),
            column=", ".join(c.name for _, c, _ in batch),
            context="; ".join(batch_context),
            values=values_section,
        )

        try:
            response = model.complete([
                ModelMessage(role="user", content=prompt)
            ])
            parsed = _parse_json_response(response)
            if parsed and "aliases" in parsed:
                alias_map = parsed["aliases"]
                # Map back to table.column context
                for canonical, aliases in alias_map.items():
                    if not isinstance(aliases, list):
                        continue
                    # Find which table.column this canonical value belongs to
                    for table, col, values in batch:
                        if canonical in values or str(canonical) in [str(v) for v in values]:
                            all_aliases[canonical] = {
                                "aliases": [a.lower() for a in aliases[:5]],
                                "table": table.name,
                                "column": col.name,
                            }
                            break
        except Exception:
            pass

    kg.alias_registry = all_aliases
    if log_fn:
        log_fn("alias_registry", f"{len(all_aliases)} values with aliases from {len(candidates)} columns")


# ---------------------------------------------------------------------------
# Business Topology: parent-child hierarchies from FK + grouping patterns
# ---------------------------------------------------------------------------

_TOPOLOGY_PROMPT = """\
You are a business domain expert. Given these tables and their relationships, \
identify parent-child hierarchies (containment relationships) where a higher-level \
entity groups or contains lower-level entities.

SCHEMA:
{schema}

RELATIONSHIPS (FK):
{relationships}

SAMPLE DATA:
{samples}

Identify hierarchies like:
- Line → Machine (a production line contains multiple machines)
- Department → Employee
- Region → Store → Product
- Category → Subcategory → Item

Return ONLY a JSON object (no markdown fences):
{{"hierarchies": [
  {{
    "parent_table": "table_name",
    "parent_column": "column_name",
    "child_table": "table_name",
    "child_column": "column_name",
    "relationship": "contains/groups/manages",
    "description": "Line contains machines"
  }}
]}}

Rules:
- Only include REAL containment hierarchies (not just any FK relationship)
- The parent must group/contain the children (1:N containment, not just a reference)
- If no hierarchies exist, return {{"hierarchies": []}}"""


def build_business_topology(
    kg: KnowledgeGraph,
    model: "ModelAdapter",
    db_path: Path,
    log_fn: Any = None,
) -> None:
    """Detect parent-child business hierarchies and expand groupings.

    After identifying hierarchies (e.g., Line→Machine), pre-computes the
    membership mapping so the agent can expand "Line 1" to its machine IDs.
    """
    if not kg or not kg.tables or len(kg.tables) < 2:
        return

    # Build schema description for LLM
    schema_lines = []
    for t in kg.tables:
        cols = ", ".join(f"{c.name}({c.sql_type})" for c in t.columns[:10])
        schema_lines.append(f"{t.name} [{t.role}, {t.row_count} rows]: {cols}")

    relationships = []
    for src, fk in kg.all_foreign_keys():
        relationships.append(f"{src}.{fk.column} → {fk.ref_table}.{fk.ref_column}")

    # Get sample data for dimension/bridge tables (where groupings live)
    conn = sqlite3.connect(str(db_path))
    sample_lines = []
    for t in kg.tables:
        if t.role in ("dimension", "bridge") and t.row_count <= 200:
            try:
                cols_to_show = [c.name for c in t.columns[:5]]
                col_str = ", ".join(f'"{c}"' for c in cols_to_show)
                rows = conn.execute(
                    f'SELECT {col_str} FROM "{t.name}" LIMIT 5'
                ).fetchall()
                if rows:
                    sample_lines.append(f"[{t.name}] columns: {cols_to_show}")
                    for r in rows[:3]:
                        sample_lines.append(f"  {list(r)}")
            except Exception:
                pass
    conn.close()

    if not relationships:
        if log_fn:
            log_fn("business_topology", "no FKs, skipping")
        return

    prompt = _TOPOLOGY_PROMPT.format(
        schema="\n".join(schema_lines),
        relationships="\n".join(relationships),
        samples="\n".join(sample_lines) if sample_lines else "(no dimension samples)",
    )

    try:
        response = model.complete([
            ModelMessage(role="user", content=prompt)
        ])
        parsed = _parse_json_response(response)
        if not parsed or "hierarchies" not in parsed:
            if log_fn:
                log_fn("business_topology", "no hierarchies detected")
            return

        hierarchies = parsed["hierarchies"]
        if not isinstance(hierarchies, list):
            return

        # For each hierarchy, pre-compute the membership mapping
        conn = sqlite3.connect(str(db_path))
        enriched: list[dict[str, Any]] = []

        for h in hierarchies:
            if not isinstance(h, dict):
                continue
            parent_table = h.get("parent_table", "")
            parent_col = h.get("parent_column", "")
            child_table = h.get("child_table", "")
            child_col = h.get("child_column", "")

            if not all([parent_table, parent_col, child_table, child_col]):
                continue

            # Validate tables exist
            if not kg.get_table(parent_table) or not kg.get_table(child_table):
                continue

            # Build membership: parent_value → [child_values]
            membership: dict[str, list[str]] = {}
            try:
                if parent_table == child_table:
                    # Self-referencing hierarchy (e.g., category → subcategory in same table)
                    rows = conn.execute(
                        f'SELECT DISTINCT "{parent_col}", "{child_col}" '
                        f'FROM "{parent_table}" '
                        f'WHERE "{parent_col}" IS NOT NULL AND "{child_col}" IS NOT NULL'
                    ).fetchall()
                    for parent_val, child_val in rows:
                        membership.setdefault(str(parent_val), []).append(str(child_val))
                else:
                    # Cross-table: join parent to child via FK
                    # Find the FK linking them
                    join_col = None
                    for src, fk in kg.all_foreign_keys():
                        if src == child_table and fk.ref_table == parent_table:
                            join_col = (fk.column, fk.ref_column)
                            break
                        if src == parent_table and fk.ref_table == child_table:
                            join_col = (fk.ref_column, fk.column)
                            break
                    if join_col:
                        child_fk, parent_pk = join_col
                        rows = conn.execute(
                            f'SELECT DISTINCT p."{parent_col}", c."{child_col}" '
                            f'FROM "{parent_table}" p '
                            f'JOIN "{child_table}" c ON c."{child_fk}" = p."{parent_pk}" '
                            f'WHERE p."{parent_col}" IS NOT NULL '
                            f'AND c."{child_col}" IS NOT NULL '
                            f'LIMIT 500'
                        ).fetchall()
                        for parent_val, child_val in rows:
                            membership.setdefault(str(parent_val), []).append(str(child_val))
            except Exception:
                pass

            if membership:
                enriched.append({
                    "parent_table": parent_table,
                    "parent_column": parent_col,
                    "child_table": child_table,
                    "child_column": child_col,
                    "relationship": h.get("relationship", "contains"),
                    "description": h.get("description", ""),
                    "membership": membership,
                })

        conn.close()
        kg.business_topology = enriched

        if log_fn:
            total_mappings = sum(len(e["membership"]) for e in enriched)
            log_fn(
                "business_topology",
                f"{len(enriched)} hierarchies, {total_mappings} parent→child mappings"
            )

    except Exception as e:
        if log_fn:
            log_fn("business_topology_error", str(e))
