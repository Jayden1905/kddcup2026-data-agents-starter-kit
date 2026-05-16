"""KG Path Planner: graph-based query planning from PropertyGraph.

Leverages the full property graph (column-level nodes, typed edges,
value nodes) for:
- Phrase-to-column mapping via name scoring + ValueNode index
- Value-to-column lookup via graph traversal (CONTAINS_VALUE edges)
- Weighted path finding using FK overlap ratios
- Multi-hop traversal through column-level adjacency

No LLM calls — purely deterministic graph algorithms.
"""

from __future__ import annotations

import heapq
import re
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from data_agent_baseline.pipeline.kg_builder import (
    Column,
    ForeignKey,
    KnowledgeGraph,
    PropertyGraph,
    TableSchema,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryNode:
    """A resolved reference to a specific table.column."""
    table: str
    column: str
    role: str  # "output" | "filter" | "aggregate" | "order"
    operator: str = "="
    value: Any = None
    agg_func: str = ""


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A validated FK edge between two tables."""
    src_table: str
    src_column: str
    dst_table: str
    dst_column: str
    weight: float = 1.0  # overlap ratio (higher = better)


@dataclass(frozen=True, slots=True)
class QueryPath:
    """A connected path through the KG satisfying all query nodes."""
    edges: tuple[GraphEdge, ...]
    output_nodes: tuple[QueryNode, ...]
    filter_nodes: tuple[QueryNode, ...]
    tables_in_path: tuple[str, ...]


@dataclass(slots=True)
class QueryPlan:
    """Complete plan with alternatives for backtracking."""
    primary_path: QueryPath
    alternative_paths: list[QueryPath] = field(default_factory=list)
    computation_type: str = ""
    failed_paths: list[tuple[QueryPath, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Node Mapping: phrase → table.column (deterministic scoring)
# ---------------------------------------------------------------------------


def map_phrase_to_columns(
    phrase: str,
    kg: KnowledgeGraph,
    selected_tables: list[str] | None = None,
    role: str = "output",
) -> list[tuple[str, str, float]]:
    """Score all KG columns against a phrase. Returns [(table, column, score)] ranked.

    Uses property graph ValueNode index for value-based scoring boost.
    """
    phrase_lower = phrase.lower().strip()
    phrase_words = set(re.findall(r'[a-z]{2,}', phrase_lower))
    phrase_no_spaces = phrase_lower.replace(" ", "").replace("_", "")

    candidates: list[tuple[str, str, float]] = []

    # Phase 1: Score via traditional column name/description matching
    for table in kg.tables:
        if selected_tables and table.name not in selected_tables:
            continue
        for col in table.columns:
            score = _score_column(phrase_lower, phrase_words, phrase_no_spaces, col, table, role)
            if score > 0:
                candidates.append((table.name, col.name, score))

    # Phase 2: Boost from property graph ValueNode index
    # Only apply if phrase doesn't already match a column name exactly
    # (avoids "Thrombosis" the column being outscored by "thrombosis" the value in Symptoms)
    has_exact_name_match = any(s >= 10.0 for _, _, s in candidates)
    if kg.graph and role == "filter" and not has_exact_name_match:
        value_hits = kg.graph.find_value(phrase_lower)
        for table_name, col_name, count in value_hits:
            if selected_tables and table_name not in selected_tables:
                continue
            existing = next(
                (i for i, (t, c, _) in enumerate(candidates) if t == table_name and c == col_name),
                None,
            )
            if existing is not None:
                t, c, s = candidates[existing]
                candidates[existing] = (t, c, s + 8.0)
            else:
                candidates.append((table_name, col_name, 8.0))

    # Phase 3: Boost from semantic edges (if a column matches well, boost its neighbors)
    if kg.graph:
        top_candidates = sorted(candidates, key=lambda x: -x[2])[:3]
        for table_name, col_name, score in top_candidates:
            if score < 5.0:
                continue
            col_id = f"{table_name}.{col_name}"
            for edge in kg.graph.sem_adj.get(col_id, []):
                neighbor_id = edge.dst if edge.src == col_id else edge.src
                neighbor_col = kg.graph.columns.get(neighbor_id)
                if not neighbor_col:
                    continue
                if selected_tables and neighbor_col.table_id not in selected_tables:
                    continue
                # Add with reduced score
                boosted_score = score * edge.similarity_score * 0.5
                existing = next(
                    (i for i, (t, c, _) in enumerate(candidates)
                     if t == neighbor_col.table_id and c == neighbor_col.name),
                    None,
                )
                if existing is not None:
                    t, c, s = candidates[existing]
                    if boosted_score > s:
                        candidates[existing] = (t, c, boosted_score)
                elif boosted_score > 2.0:
                    candidates.append((neighbor_col.table_id, neighbor_col.name, boosted_score))

    candidates.sort(key=lambda x: -x[2])
    return candidates[:5]


def _stem_overlap(words_a: set[str], words_b: set[str]) -> set[str]:
    """Find overlapping words including stem matches (names↔name, schools↔school)."""
    matched: set[str] = words_a & words_b
    for wa in words_a - matched:
        for wb in words_b - matched:
            if len(wa) >= 3 and len(wb) >= 3:
                if wa.startswith(wb) or wb.startswith(wa):
                    matched.add(wa)
                    break
    return matched


def _score_column(
    phrase_lower: str,
    phrase_words: set[str],
    phrase_no_spaces: str,
    col: Column,
    table: TableSchema,
    role: str,
) -> float:
    """Score a single column against a phrase."""
    score = 0.0
    col_lower = col.name.lower()
    col_words = set(re.findall(r'[a-z]{2,}', col_lower))
    col_no_spaces = col_lower.replace(" ", "").replace("_", "").replace("(", "").replace(")", "").replace("-", "")

    # --- Name match (0-10) ---
    if phrase_lower in col_lower or phrase_no_spaces in col_no_spaces:
        score += 10.0
    elif col_lower in phrase_lower or col_no_spaces in phrase_no_spaces:
        score += 8.0
    else:
        overlap = _stem_overlap(phrase_words, col_words)
        if overlap:
            coverage = len(overlap) / max(len(phrase_words), 1)
            score += coverage * 6.0

    # Multi-word bonus
    if len(col_words) > 1 and len(phrase_words) > 1:
        multi_overlap = _stem_overlap(phrase_words, col_words)
        if len(multi_overlap) >= 2:
            score += 3.0

    # --- Description match (0-5) ---
    if col.description:
        desc_lower = col.description.lower()
        desc_words = set(re.findall(r'[a-z]{3,}', desc_lower))
        desc_overlap = _stem_overlap(phrase_words, desc_words)
        if desc_overlap:
            score += min(len(desc_overlap) * 1.5, 5.0)

    # --- Sample value match (0-8) for filter role ---
    if role == "filter" and col.name in table.sample_values:
        samples = table.sample_values[col.name]
        for sv in samples:
            sv_str = str(sv).lower()
            if len(sv_str) < 3:
                continue
            if phrase_lower == sv_str:
                score += 8.0
                break
            elif len(sv_str) >= 3 and (phrase_lower in sv_str or sv_str in phrase_lower):
                score += 5.0
                break

    # --- Type compatibility (0-3) ---
    if role == "filter":
        col_type_upper = col.sql_type.upper()
        is_numeric = col_type_upper in ("INTEGER", "INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE")
        if re.match(r'^[<>!=]*\s*\d+\.?\d*$', phrase_lower.strip()):
            if is_numeric:
                score += 3.0
        else:
            if col_type_upper in ("TEXT", "VARCHAR", "CHAR", ""):
                score += 2.0

    # --- Cardinality signal (0-2) ---
    if role == "filter":
        stats = table.col_stats.get(col.name, {})
        distinct = stats.get("distinct", 0)
        if distinct > 10:
            score += 1.0
        if distinct > 100:
            score += 1.0

    if score < 1.0:
        score = 0.0

    return score


def map_value_to_column(
    value: str,
    kg: KnowledgeGraph,
    selected_tables: list[str] | None = None,
    db_path: Path | None = None,
) -> list[tuple[str, str, str, int]]:
    """Find which column(s) contain a specific value. Returns [(table, column, method, count)].

    Uses property graph ValueNode index first (O(1)), falls back to DB probe.
    """
    results: list[tuple[str, str, str, int]] = []

    # Phase 1: Property graph ValueNode lookup (instant)
    if kg.graph:
        normalized = value.strip().lower()
        seen_cols: set[tuple[str, str]] = set()

        # Exact match via index
        for vid in kg.graph.value_index.get(normalized, []):
            vnode = kg.graph.values.get(vid)
            if vnode:
                col_node = kg.graph.columns.get(vnode.column_id)
                if col_node:
                    if selected_tables and col_node.table_id not in selected_tables:
                        continue
                    key = (col_node.table_id, col_node.name)
                    if key not in seen_cols:
                        seen_cols.add(key)
                        results.append((col_node.table_id, col_node.name, "exact", vnode.count))

        # Substring match if no exact
        if not results:
            for norm_val, vids in kg.graph.value_index.items():
                if normalized in norm_val or norm_val in normalized:
                    for vid in vids:
                        vnode = kg.graph.values.get(vid)
                        if vnode:
                            col_node = kg.graph.columns.get(vnode.column_id)
                            if col_node:
                                if selected_tables and col_node.table_id not in selected_tables:
                                    continue
                                key = (col_node.table_id, col_node.name)
                                if key not in seen_cols:
                                    seen_cols.add(key)
                                    results.append((col_node.table_id, col_node.name, "like", vnode.count))

        if results:
            return sorted(results, key=lambda x: -x[3])

    # Phase 2: Fall back to sample_values scan
    for table in kg.tables:
        if selected_tables and table.name not in selected_tables:
            continue
        for col in table.columns:
            if col.name not in table.sample_values:
                continue
            samples = table.sample_values[col.name]
            for sv in samples:
                if str(sv).lower() == value.lower():
                    results.append((table.name, col.name, "exact", -1))
                    break
                elif value.lower() in str(sv).lower():
                    results.append((table.name, col.name, "like", -1))
                    break

    # Phase 3: DB probe for actual counts
    if db_path and db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            verified: list[tuple[str, str, str, int]] = []
            for tname, cname, method, _ in results:
                try:
                    if method == "exact":
                        cnt = conn.execute(
                            f'SELECT COUNT(*) FROM "{tname}" WHERE "{cname}" = ? COLLATE NOCASE',
                            (value,)
                        ).fetchone()[0]
                    else:
                        cnt = conn.execute(
                            f'SELECT COUNT(*) FROM "{tname}" WHERE "{cname}" LIKE ? COLLATE NOCASE',
                            (f'%{value}%',)
                        ).fetchone()[0]
                    if cnt > 0:
                        verified.append((tname, cname, method, cnt))
                except Exception:
                    pass
            conn.close()
            if verified:
                return sorted(verified, key=lambda x: -x[3])
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Graph Traversal: weighted path finding
# ---------------------------------------------------------------------------


def build_adjacency(kg: KnowledgeGraph) -> dict[str, list[GraphEdge]]:
    """Build bidirectional adjacency list from KG FK + semantic edges."""
    adj: dict[str, list[GraphEdge]] = {}
    seen_pairs: set[tuple[str, str]] = set()

    # Use property graph FK edges if available (have overlap ratios)
    if kg.graph and kg.graph.fk_edges:
        for fk_edge in kg.graph.fk_edges:
            src_col = kg.graph.columns.get(fk_edge.src)
            dst_col = kg.graph.columns.get(fk_edge.dst)
            if not src_col or not dst_col:
                continue
            pair = (min(src_col.table_id, dst_col.table_id), max(src_col.table_id, dst_col.table_id))
            seen_pairs.add(pair)
            # Forward
            edge_fwd = GraphEdge(
                src_col.table_id, src_col.name,
                dst_col.table_id, dst_col.name,
                weight=fk_edge.overlap_ratio,
            )
            adj.setdefault(src_col.table_id, []).append(edge_fwd)
            # Reverse
            edge_rev = GraphEdge(
                dst_col.table_id, dst_col.name,
                src_col.table_id, src_col.name,
                weight=fk_edge.overlap_ratio,
            )
            adj.setdefault(dst_col.table_id, []).append(edge_rev)
    else:
        # Fallback: use legacy FK list
        for src_table, fk in kg.all_foreign_keys():
            pair = (min(src_table, fk.ref_table), max(src_table, fk.ref_table))
            seen_pairs.add(pair)
            edge_fwd = GraphEdge(src_table, fk.column, fk.ref_table, fk.ref_column)
            adj.setdefault(src_table, []).append(edge_fwd)
            edge_rev = GraphEdge(fk.ref_table, fk.ref_column, src_table, fk.column)
            adj.setdefault(fk.ref_table, []).append(edge_rev)

    # Add semantic edges for table pairs not already connected by FK
    # Only add edges where a column name references the other table (bridge table pattern)
    # e.g. hero_power.hero_id ↔ superhero.id — "hero_id" references "superhero"
    if kg.graph and kg.graph.semantic_edges:
        for sem_edge in kg.graph.semantic_edges:
            src_col = kg.graph.columns.get(sem_edge.src)
            dst_col = kg.graph.columns.get(sem_edge.dst)
            if not src_col or not dst_col:
                continue
            if src_col.table_id == dst_col.table_id:
                continue
            pair = (min(src_col.table_id, dst_col.table_id), max(src_col.table_id, dst_col.table_id))
            if pair in seen_pairs:
                continue
            # Only allow if one column name references the other table
            # e.g. "hero_id" in table "hero_power" referencing table "superhero" (contains "hero")
            src_name_lower = src_col.name.lower()
            dst_name_lower = dst_col.name.lower()
            dst_table_lower = dst_col.table_id.lower()
            src_table_lower = src_col.table_id.lower()

            # Extract prefix from column name (e.g. "hero_id" → "hero", "driverId" → "driver")
            src_prefix = re.sub(r'[_]?(id|key|code|no|num)$', '', src_name_lower, flags=re.IGNORECASE).replace("_", "")
            dst_prefix = re.sub(r'[_]?(id|key|code|no|num)$', '', dst_name_lower, flags=re.IGNORECASE).replace("_", "")

            # Check if column prefix is found in the other table's name
            src_refs_dst = (src_prefix and len(src_prefix) >= 3 and
                          (src_prefix in dst_table_lower or dst_table_lower in src_prefix))
            dst_refs_src = (dst_prefix and len(dst_prefix) >= 3 and
                          (dst_prefix in src_table_lower or src_table_lower in dst_prefix))
            if not (src_refs_dst or dst_refs_src):
                continue

            weight = sem_edge.similarity_score * 0.4
            edge_fwd = GraphEdge(
                src_col.table_id, src_col.name,
                dst_col.table_id, dst_col.name,
                weight=weight,
            )
            adj.setdefault(src_col.table_id, []).append(edge_fwd)
            edge_rev = GraphEdge(
                dst_col.table_id, dst_col.name,
                src_col.table_id, src_col.name,
                weight=weight,
            )
            adj.setdefault(dst_col.table_id, []).append(edge_rev)
            seen_pairs.add(pair)

    return adj


def find_shortest_path(
    adj: dict[str, list[GraphEdge]],
    start: str,
    end: str,
    max_depth: int = 4,
) -> list[GraphEdge] | None:
    """Dijkstra-like shortest path preferring high-weight (high overlap) edges."""
    if start == end:
        return []

    # Priority queue: (cost, counter, table_name, path)
    counter = 0
    heap: list[tuple[float, int, str, list[GraphEdge]]] = [(0.0, counter, start, [])]
    visited: set[str] = set()

    while heap:
        cost, _, current, path = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)

        if current == end:
            return path

        if len(path) >= max_depth:
            continue

        for edge in adj.get(current, []):
            if edge.dst_table in visited:
                continue
            edge_cost = 1.0 - edge.weight
            # Tiebreaker: prefer edges following FK naming convention (<table>_id)
            # Check both directions since edges are bidirectional
            src_col_lower = edge.src_column.lower()
            dst_col_lower = edge.dst_column.lower()
            src_tbl_lower = edge.src_table.lower()
            dst_tbl_lower = edge.dst_table.lower()
            if (src_col_lower in (f"{dst_tbl_lower}_id", f"{dst_tbl_lower}id")
                    or dst_col_lower in (f"{src_tbl_lower}_id", f"{src_tbl_lower}id")):
                edge_cost -= 0.01
            counter += 1
            heapq.heappush(heap, (cost + edge_cost, counter, edge.dst_table, path + [edge]))

    return None


def find_all_paths(
    adj: dict[str, list[GraphEdge]],
    start: str,
    end: str,
    max_depth: int = 3,
) -> list[list[GraphEdge]]:
    """Find all paths (up to max_depth) between two tables for alternatives."""
    if start == end:
        return [[]]

    all_paths: list[list[GraphEdge]] = []
    stack: list[tuple[str, list[GraphEdge], set[str]]] = [(start, [], {start})]

    while stack:
        current, path, visited = stack.pop()
        if len(path) >= max_depth:
            continue
        for edge in adj.get(current, []):
            if edge.dst_table in visited:
                continue
            new_path = path + [edge]
            if edge.dst_table == end:
                all_paths.append(new_path)
            else:
                stack.append((edge.dst_table, new_path, visited | {edge.dst_table}))

    # Sort by path quality: fewest hops first, then by weight (higher = better)
    def path_score(p: list[GraphEdge]) -> tuple[int, float]:
        avg_weight = sum(e.weight for e in p) / len(p) if p else 1.0
        return (len(p), -avg_weight)

    return sorted(all_paths, key=path_score)


def find_steiner_path(
    adj: dict[str, list[GraphEdge]],
    required_tables: list[str],
) -> list[GraphEdge]:
    """Find minimum edges connecting all required tables (approximate Steiner tree)."""
    if len(required_tables) <= 1:
        return []

    connected: set[str] = {required_tables[0]}
    all_edges: list[GraphEdge] = []

    for target in required_tables[1:]:
        if target in connected:
            continue
        best_path: list[GraphEdge] | None = None
        for src in connected:
            path = find_shortest_path(adj, src, target)
            if path is not None:
                if best_path is None or len(path) < len(best_path):
                    best_path = path

        if best_path:
            all_edges.extend(best_path)
            for edge in best_path:
                connected.add(edge.dst_table)
                connected.add(edge.src_table)

    # Deduplicate edges
    seen: set[tuple[str, str, str, str]] = set()
    unique_edges: list[GraphEdge] = []
    for e in all_edges:
        key = (e.src_table, e.src_column, e.dst_table, e.dst_column)
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    return unique_edges


def score_path(edges: list[GraphEdge]) -> float:
    """Score a path by average FK overlap ratio (higher = more trustworthy)."""
    if not edges:
        return 1.0
    return sum(e.weight for e in edges) / len(edges)


# ---------------------------------------------------------------------------
# Query Path Construction
# ---------------------------------------------------------------------------


def build_query_path(
    output_nodes: list[QueryNode],
    filter_nodes: list[QueryNode],
    kg: KnowledgeGraph,
    order_nodes: list[QueryNode] | None = None,
) -> QueryPath | None:
    """Build a QueryPath connecting all output, filter, and order nodes through KG edges."""
    required_tables: list[str] = []
    seen_tables: set[str] = set()
    for node in output_nodes + filter_nodes + (order_nodes or []):
        if node.table not in seen_tables:
            required_tables.append(node.table)
            seen_tables.add(node.table)

    if not required_tables:
        return None

    adj = build_adjacency(kg)
    edges = find_steiner_path(adj, required_tables)

    tables_in_path: list[str] = list(required_tables)
    for edge in edges:
        if edge.src_table not in tables_in_path:
            tables_in_path.append(edge.src_table)
        if edge.dst_table not in tables_in_path:
            tables_in_path.append(edge.dst_table)

    return QueryPath(
        edges=tuple(edges),
        output_nodes=tuple(output_nodes),
        filter_nodes=tuple(filter_nodes),
        tables_in_path=tuple(tables_in_path),
    )


def build_alternative_paths(
    output_nodes: list[QueryNode],
    filter_nodes: list[QueryNode],
    kg: KnowledgeGraph,
    exclude_edges: set[tuple[str, str, str, str]] | None = None,
) -> list[QueryPath]:
    """Generate alternative paths for backtracking."""
    required_tables: list[str] = []
    seen: set[str] = set()
    for node in output_nodes + filter_nodes:
        if node.table not in seen:
            required_tables.append(node.table)
            seen.add(node.table)

    if len(required_tables) < 2:
        return []

    adj = build_adjacency(kg)
    alternatives: list[QueryPath] = []

    for i in range(len(required_tables)):
        for j in range(i + 1, len(required_tables)):
            alt_paths = find_all_paths(adj, required_tables[i], required_tables[j])
            for path_edges in alt_paths[1:3]:
                if exclude_edges:
                    edge_keys = {(e.src_table, e.src_column, e.dst_table, e.dst_column) for e in path_edges}
                    if edge_keys & exclude_edges:
                        continue
                tables = list(required_tables)
                for e in path_edges:
                    if e.src_table not in tables:
                        tables.append(e.src_table)
                    if e.dst_table not in tables:
                        tables.append(e.dst_table)
                alternatives.append(QueryPath(
                    edges=tuple(path_edges),
                    output_nodes=tuple(output_nodes),
                    filter_nodes=tuple(filter_nodes),
                    tables_in_path=tuple(tables),
                ))

    # Sort alternatives by path quality
    alternatives.sort(key=lambda p: -score_path(list(p.edges)))
    return alternatives
