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


# ---------------------------------------------------------------------------
# Property Graph Node Types
# ---------------------------------------------------------------------------

VALUE_NODE_CARDINALITY_THRESHOLD = 50
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
            distinct_count = conn.execute(
                f'SELECT COUNT(DISTINCT "{col.name}") FROM "{table_name}" '
                f'WHERE "{col.name}" IS NOT NULL'
            ).fetchone()[0]
            stats["distinct"] = distinct_count
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
# LLM-powered FK/Join Discovery
# ---------------------------------------------------------------------------

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

    if log_fn:
        log_fn("kg_llm_fk", f"{added} FKs added, {len(fk_display_map)} display columns mapped")

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
