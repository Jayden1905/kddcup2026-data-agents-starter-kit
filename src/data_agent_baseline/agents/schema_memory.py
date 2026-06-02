"""SchemaMemoryGraph: persistent, mutable knowledge graph about database schemas.

Accumulates discoveries across tasks. Each task has a unique schema (different
tables/columns), so facts are keyed by structural elements (table name, column
name, join path). The graph is queryable — never dumped into context.

First run on a new schema: no prior knowledge, agent explores from scratch.
Subsequent runs on the same schema: agent recalls known facts instantly.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FactEntry:
    """A single discovered fact about a schema element."""

    text: str
    confidence: int = 1
    source_task: str = ""
    timestamp: float = field(default_factory=time.time)
    superseded: bool = False


@dataclass
class SchemaNode:
    """A node in the schema memory graph (table, column, join, or value)."""

    node_id: str
    node_type: str  # "table", "column", "join", "value"
    facts: list[FactEntry] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)  # related node_ids

    def add_fact(self, text: str, source_task: str = "") -> None:
        """Add or reinforce a fact. Handles conflicts via supersession.

        - Exact duplicate: bump confidence
        - Conflicting fact (same topic prefix, different conclusion): supersede
          the older one, keep the newer as authoritative
        - New fact: append
        """
        for f in self.facts:
            if f.superseded:
                continue
            if f.text == text:
                f.confidence += 1
                f.timestamp = time.time()
                return
            # Detect conflict: same structural prefix but different conclusion
            if self._is_conflicting(f.text, text):
                f.superseded = True
                break
        self.facts.append(
            FactEntry(text=text, source_task=source_task, timestamp=time.time())
        )

    @staticmethod
    def _is_conflicting(existing: str, new: str) -> bool:
        """Two facts conflict if they describe the same specific subject differently.

        Conflict means: same column/entity referenced, but different value or
        conclusion. E.g. "Thrombosis: severe = 3" vs "Thrombosis: severe = 2".
        Different columns never conflict — they're additive.
        """
        if existing == new:
            return False

        # Extract the specific subject (column name or entity referenced)
        def _subject(s: str) -> str:
            lower = s.lower()
            # For "domain mapping: WHERE X = Y" → subject is X
            if "where" in lower:
                after_where = lower.split("where", 1)[-1].strip()
                # First word after WHERE is the column
                parts = after_where.split()
                if parts:
                    return parts[0].strip('"').strip("'")
            # For "value gotcha: ..." → use the full text as subject (unique)
            if "gotcha" in lower:
                return lower
            # For "distribution: ..." → unique per call
            if "distribution" in lower:
                return lower
            # Generic: use first 60 chars
            return lower[:60]

        return _subject(existing) == _subject(new)

    def add_edge(self, target_id: str) -> None:
        if target_id not in self.edges:
            self.edges.append(target_id)

    def render(self) -> str:
        """Render active (non-superseded) facts for this node."""
        active = [f for f in self.facts if not f.superseded]
        if not active and not self.edges:
            return ""
        lines = [f"[{self.node_type}] {self.node_id}"]
        for f in sorted(active, key=lambda x: -x.confidence):
            conf = f"(x{f.confidence})" if f.confidence > 1 else ""
            lines.append(f"  {conf} {f.text}")
        if self.edges:
            lines.append(f"  → connected to: {self.edges}")
        return "\n".join(lines)


class SchemaMemoryGraph:
    """Persistent, mutable knowledge graph about database schemas.

    Keyed by schema fingerprint (hash of table+column names).
    Stored as JSON on disk. Loaded at task start, written at task end.
    Queryable by table, column, join path, or value.
    """

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.nodes: dict[str, SchemaNode] = {}
        self._storage_dir = storage_dir
        self._fingerprint: str = ""
        self._dirty = False

    @staticmethod
    def compute_fingerprint(tables: list[dict[str, Any]]) -> str:
        """Compute a stable fingerprint from schema structure.

        Args:
            tables: list of {"name": str, "columns": list[str]}
        """
        parts = []
        for t in sorted(tables, key=lambda x: x["name"]):
            cols = sorted(t["columns"])
            parts.append(f"{t['name']}:{','.join(cols)}")
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def load(self, fingerprint: str) -> bool:
        """Load existing graph for this fingerprint. Returns True if found."""
        self._fingerprint = fingerprint
        if not self._storage_dir:
            return False
        path = self._storage_dir / f"{fingerprint}.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            for node_id, node_data in data.get("nodes", {}).items():
                facts = [
                    FactEntry(
                        text=f["text"],
                        confidence=f.get("confidence", 1),
                        source_task=f.get("source_task", ""),
                        timestamp=f.get("timestamp", 0),
                        superseded=f.get("superseded", False),
                    )
                    for f in node_data.get("facts", [])
                ]
                self.nodes[node_id] = SchemaNode(
                    node_id=node_id,
                    node_type=node_data.get("node_type", "unknown"),
                    facts=facts,
                    edges=node_data.get("edges", []),
                )
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def save(self) -> None:
        """Persist the graph to disk."""
        if not self._storage_dir or not self._fingerprint:
            return
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_dir / f"{self._fingerprint}.json"
        data = {"fingerprint": self._fingerprint, "nodes": {}}
        for node_id, node in self.nodes.items():
            data["nodes"][node_id] = {
                "node_type": node.node_type,
                "facts": [
                    {
                        "text": f.text,
                        "confidence": f.confidence,
                        "source_task": f.source_task,
                        "timestamp": f.timestamp,
                        "superseded": f.superseded,
                    }
                    for f in node.facts
                ],
                "edges": node.edges,
            }
        path.write_text(json.dumps(data, indent=2))
        self._dirty = False

    # ------------------------------------------------------------------
    # Mutation: record discoveries
    # ------------------------------------------------------------------

    def record_table(self, table: str, fact: str, task_id: str = "") -> None:
        """Record a fact about a table."""
        node_id = f"table:{table.lower()}"
        if node_id not in self.nodes:
            self.nodes[node_id] = SchemaNode(node_id=node_id, node_type="table")
        self.nodes[node_id].add_fact(fact, task_id)
        self._dirty = True

    def record_column(
        self, table: str, column: str, fact: str, task_id: str = ""
    ) -> None:
        """Record a fact about a column."""
        node_id = f"col:{table.lower()}.{column.lower()}"
        if node_id not in self.nodes:
            self.nodes[node_id] = SchemaNode(node_id=node_id, node_type="column")
        self.nodes[node_id].add_fact(fact, task_id)
        # Connect to table node
        table_id = f"table:{table.lower()}"
        if table_id not in self.nodes:
            self.nodes[table_id] = SchemaNode(node_id=table_id, node_type="table")
        self.nodes[node_id].add_edge(table_id)
        self.nodes[table_id].add_edge(node_id)
        self._dirty = True

    def record_join(
        self, from_table: str, from_col: str,
        to_table: str, to_col: str, fact: str, task_id: str = "",
    ) -> None:
        """Record a fact about a join path."""
        node_id = f"join:{from_table.lower()}.{from_col.lower()}->{to_table.lower()}.{to_col.lower()}"
        if node_id not in self.nodes:
            self.nodes[node_id] = SchemaNode(node_id=node_id, node_type="join")
        self.nodes[node_id].add_fact(fact, task_id)
        # Connect to both table nodes
        for t in (from_table, to_table):
            tid = f"table:{t.lower()}"
            if tid not in self.nodes:
                self.nodes[tid] = SchemaNode(node_id=tid, node_type="table")
            self.nodes[node_id].add_edge(tid)
            self.nodes[tid].add_edge(node_id)
        self._dirty = True

    def record_value(
        self, table: str, column: str, value: str, fact: str, task_id: str = ""
    ) -> None:
        """Record a fact about a specific value in a column."""
        node_id = f"val:{table.lower()}.{column.lower()}::{value.lower()}"
        if node_id not in self.nodes:
            self.nodes[node_id] = SchemaNode(node_id=node_id, node_type="value")
        self.nodes[node_id].add_fact(fact, task_id)
        # Connect to column node
        col_id = f"col:{table.lower()}.{column.lower()}"
        if col_id not in self.nodes:
            self.nodes[col_id] = SchemaNode(
                node_id=col_id, node_type="column"
            )
        self.nodes[node_id].add_edge(col_id)
        self.nodes[col_id].add_edge(node_id)
        self._dirty = True

    # ------------------------------------------------------------------
    # Query: recall knowledge
    # ------------------------------------------------------------------

    def recall(
        self,
        table: str | None = None,
        column: str | None = None,
        join: str | None = None,
        value: str | None = None,
    ) -> str:
        """Query the schema memory graph.

        Returns rendered facts matching the query. All params optional.
        If no params given, returns summary of all known facts.
        """
        matching: list[SchemaNode] = []

        if table and not column:
            # All facts about a table + its columns
            table_id = f"table:{table.lower()}"
            if table_id in self.nodes:
                matching.append(self.nodes[table_id])
            # Also include column nodes for this table
            prefix = f"col:{table.lower()}."
            for nid, node in self.nodes.items():
                if nid.startswith(prefix):
                    matching.append(node)

        elif table and column:
            # Specific column
            col_id = f"col:{table.lower()}.{column.lower()}"
            if col_id in self.nodes:
                matching.append(self.nodes[col_id])
            # Include value nodes for this column
            prefix = f"val:{table.lower()}.{column.lower()}::"
            for nid, node in self.nodes.items():
                if nid.startswith(prefix):
                    matching.append(node)

        elif join:
            # Join path — search by substring
            for nid, node in self.nodes.items():
                if node.node_type == "join" and join.lower() in nid.lower():
                    matching.append(node)

        elif value:
            # Search all value nodes
            for nid, node in self.nodes.items():
                if node.node_type == "value" and value.lower() in nid.lower():
                    matching.append(node)

        else:
            # Full summary
            matching = list(self.nodes.values())

        if not matching:
            return ""

        return "\n\n".join(node.render() for node in matching)

    # Fact categories: actionable vs structural noise
    _ACTIONABLE_PREFIXES = (
        "domain mapping:",
        "resolved:",
        "known values:",
        "value format:",
        "value gotcha:",
        "contains values",
        "FK:",
    )

    def _is_actionable(self, fact_text: str) -> bool:
        """A fact is actionable if it tells the agent something it can use
        directly in SQL generation — not just structural metadata."""
        return any(fact_text.startswith(p) for p in self._ACTIONABLE_PREFIXES)

    def summarize_for_prompt(self) -> str:
        """Produce a concise, actionable summary for injection into the prompt.

        Only includes facts that directly help SQL generation:
        domain mappings, column resolutions, known values, value gotchas,
        and join paths. Skips structural noise (column types, pk flags).
        """
        lines: list[str] = []

        # Collect actionable facts grouped by category
        domain_mappings: list[str] = []
        resolutions: list[str] = []
        value_facts: list[str] = []
        join_paths: list[str] = []

        for node in self.nodes.values():
            for fact in node.facts:
                if fact.superseded:
                    continue
                text = fact.text
                if text.startswith("domain mapping:"):
                    domain_mappings.append(text.replace("domain mapping: ", ""))
                elif text.startswith("resolved:"):
                    resolutions.append(text.replace("resolved: ", ""))
                elif text.startswith("known values:"):
                    col = node.node_id.replace("col:", "")
                    value_facts.append(f"{col}: {text.replace('known values: ', '')}")
                elif text.startswith("value format:") or text.startswith("value gotcha:"):
                    col = node.node_id.replace("col:", "")
                    value_facts.append(f"{col}: {text}")
                elif text.startswith("FK:"):
                    join_paths.append(text.replace("FK: ", ""))

        if domain_mappings:
            lines.append("Domain mappings (use these exact values in WHERE):")
            for dm in domain_mappings:
                lines.append(f"  {dm}")

        if resolutions:
            lines.append("Column ownership (which table to SELECT from):")
            for r in resolutions:
                lines.append(f"  {r}")

        if join_paths:
            lines.append("Join paths:")
            for jp in join_paths:
                lines.append(f"  {jp}")

        if value_facts:
            lines.append("Value formats and vocabularies:")
            for vf in value_facts:
                lines.append(f"  {vf}")

        return "\n".join(lines)

    def has_knowledge(self) -> bool:
        """Check if any actionable facts exist in this graph."""
        for node in self.nodes.values():
            for fact in node.facts:
                if not fact.superseded and self._is_actionable(fact.text):
                    return True
        return False

    def summary(self) -> str:
        """Brief summary of what's known."""
        n_tables = sum(1 for n in self.nodes.values() if n.node_type == "table")
        n_cols = sum(1 for n in self.nodes.values() if n.node_type == "column")
        n_joins = sum(1 for n in self.nodes.values() if n.node_type == "join")
        n_values = sum(1 for n in self.nodes.values() if n.node_type == "value")
        total_facts = sum(len(n.facts) for n in self.nodes.values())
        return (
            f"Schema memory: {n_tables} tables, {n_cols} columns, "
            f"{n_joins} joins, {n_values} values, {total_facts} total facts"
        )
