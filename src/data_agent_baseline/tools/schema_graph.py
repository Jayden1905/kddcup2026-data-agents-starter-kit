from __future__ import annotations

import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    name: str
    col_type: str
    notnull: bool
    pk: bool


@dataclass(frozen=True, slots=True)
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass(slots=True)
class TableSchema:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)


@dataclass(slots=True)
class SchemaGraph:
    tables: dict[str, TableSchema] = field(default_factory=dict)
    adjacency: dict[str, set[str]] = field(default_factory=dict)

    def add_table(self, schema: TableSchema) -> None:
        self.tables[schema.name] = schema
        self.adjacency.setdefault(schema.name, set())

    def add_edge(self, table_a: str, table_b: str) -> None:
        self.adjacency.setdefault(table_a, set()).add(table_b)
        self.adjacency.setdefault(table_b, set()).add(table_a)


def build_schema_graph(db_path: Path) -> SchemaGraph:
    graph = SchemaGraph()
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

        for (table_name,) in tables:
            cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            columns = [
                ColumnInfo(
                    name=row[1],
                    col_type=row[2] or "TEXT",
                    notnull=bool(row[3]),
                    pk=bool(row[5]),
                )
                for row in cols
            ]
            fks = conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
            foreign_keys = [
                ForeignKey(
                    from_table=table_name,
                    from_column=row[3],
                    to_table=row[2],
                    to_column=row[4],
                )
                for row in fks
            ]
            schema = TableSchema(name=table_name, columns=columns, foreign_keys=foreign_keys)
            graph.add_table(schema)

            for fk in foreign_keys:
                if fk.to_table not in graph.tables:
                    ref_cols = conn.execute(f"PRAGMA table_info({fk.to_table})").fetchall()
                    if ref_cols:
                        ref_columns = [
                            ColumnInfo(
                                name=r[1],
                                col_type=r[2] or "TEXT",
                                notnull=bool(r[3]),
                                pk=bool(r[5]),
                            )
                            for r in ref_cols
                        ]
                        graph.add_table(TableSchema(name=fk.to_table, columns=ref_columns))
                graph.add_edge(table_name, fk.to_table)

    return graph


def build_schema_graph_multi(db_paths: list[Path]) -> SchemaGraph:
    merged = SchemaGraph()
    for db_path in db_paths:
        sub = build_schema_graph(db_path)
        for name, schema in sub.tables.items():
            if name not in merged.tables:
                merged.add_table(schema)
            else:
                for fk in schema.foreign_keys:
                    if fk not in merged.tables[name].foreign_keys:
                        merged.tables[name].foreign_keys.append(fk)
        for node, neighbors in sub.adjacency.items():
            for neighbor in neighbors:
                merged.add_edge(node, neighbor)
    return merged


# ---------------------------------------------------------------------------
# KMB Steiner Tree Approximation
# ---------------------------------------------------------------------------


def _bfs_shortest(adjacency: dict[str, set[str]], source: str) -> dict[str, tuple[int, str | None]]:
    dist: dict[str, tuple[int, str | None]] = {source: (0, None)}
    queue: deque[str] = deque([source])
    while queue:
        node = queue.popleft()
        d = dist[node][0]
        for neighbor in adjacency.get(node, ()):
            if neighbor not in dist:
                dist[neighbor] = (d + 1, node)
                queue.append(neighbor)
    return dist


def _reconstruct_path(dist_map: dict[str, tuple[int, str | None]], target: str) -> list[str]:
    path = []
    current: str | None = target
    while current is not None:
        path.append(current)
        current = dist_map[current][1]
    path.reverse()
    return path


def steiner_tree_tables(graph: SchemaGraph, terminal_tables: list[str]) -> list[str]:
    terminals = [t for t in terminal_tables if t in graph.adjacency]
    if len(terminals) <= 1:
        return terminals

    all_dists: dict[str, dict[str, tuple[int, str | None]]] = {}
    for t in terminals:
        all_dists[t] = _bfs_shortest(graph.adjacency, t)

    edges: list[tuple[int, str, str]] = []
    for i, t1 in enumerate(terminals):
        for t2 in terminals[i + 1 :]:
            if t2 in all_dists[t1]:
                edges.append((all_dists[t1][t2][0], t1, t2))
    edges.sort()

    parent: dict[str, str] = {t: t for t in terminals}
    rank: dict[str, int] = {t: 0 for t in terminals}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    mst_edges: list[tuple[str, str]] = []
    for cost, u, v in edges:
        if union(u, v):
            mst_edges.append((u, v))
            if len(mst_edges) == len(terminals) - 1:
                break

    result_tables: set[str] = set()
    for u, v in mst_edges:
        path = _reconstruct_path(all_dists[u], v)
        result_tables.update(path)

    return sorted(result_tables)


# ---------------------------------------------------------------------------
# Schema slice rendering
# ---------------------------------------------------------------------------


def render_schema_slice(graph: SchemaGraph, tables: list[str]) -> str:
    lines: list[str] = []
    table_set = set(tables)
    for tbl_name in sorted(table_set):
        schema = graph.tables.get(tbl_name)
        if schema is None:
            continue
        col_descs = []
        for col in schema.columns:
            parts = [col.name, col.col_type]
            if col.pk:
                parts.append("PK")
            if col.notnull:
                parts.append("NOT NULL")
            col_descs.append(" ".join(parts))
        lines.append(f"TABLE {tbl_name} ({', '.join(col_descs)})")
        relevant_fks = [fk for fk in schema.foreign_keys if fk.to_table in table_set]
        for fk in relevant_fks:
            lines.append(f"  FK {fk.from_column} -> {fk.to_table}.{fk.to_column}")
    return "\n".join(lines)


def slice_schema_for_task(
    db_paths: list[Path], requested_tables: list[str] | None = None
) -> tuple[SchemaGraph, list[str], str]:
    graph = build_schema_graph_multi(db_paths)
    if requested_tables:
        valid = [t for t in requested_tables if t in graph.tables]
        tables = steiner_tree_tables(graph, valid) if len(valid) > 1 else valid
    else:
        tables = sorted(graph.tables.keys())
    rendered = render_schema_slice(graph, tables)
    return graph, tables, rendered
