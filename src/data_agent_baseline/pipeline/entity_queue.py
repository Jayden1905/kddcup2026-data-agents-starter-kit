"""Thread-safe entity queue with deduplication.

Extraction agents push entities/relationships; graph builder pulls them.
Deduplication happens at push time — same (entity_type, entity_id) merges fields.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Relationship:
    source_type: str
    source_id: str
    rel_type: str
    target_type: str
    target_id: str


class EntityQueue:
    """Thread-safe dedup buffer for extracted entities and relationships."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entities: dict[tuple[str, str], dict[str, Any]] = {}
        self._relationships: set[Relationship] = set()
        self._has_new_data = threading.Event()

    def push_entity(self, entity_type: str, entity_id: str, fields: dict[str, Any]) -> None:
        """Push entity, merging with existing if duplicate (non-null wins)."""
        with self._lock:
            key = (entity_type, entity_id)
            if key in self._entities:
                existing = self._entities[key]
                for k, v in fields.items():
                    if v is not None and v != "":
                        existing[k] = v
            else:
                self._entities[key] = {k: v for k, v in fields.items() if v is not None and v != ""}
            self._has_new_data.set()

    def push_relationship(self, rel: Relationship) -> None:
        with self._lock:
            self._relationships.add(rel)
            self._has_new_data.set()

    def drain_entities(self) -> dict[str, list[dict[str, Any]]]:
        """Pull all entities grouped by type. Clears internal buffer."""
        with self._lock:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for (etype, eid), fields in self._entities.items():
                grouped.setdefault(etype, []).append({"_id": eid, **fields})
            self._entities.clear()
            self._has_new_data.clear()
            return grouped

    def drain_relationships(self) -> list[Relationship]:
        with self._lock:
            rels = list(self._relationships)
            self._relationships.clear()
            return rels

    def wait_for_data(self, timeout: float = 0.5) -> bool:
        """Block until new data arrives or timeout. Returns True if data available."""
        return self._has_new_data.wait(timeout)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._entities) == 0 and len(self._relationships) == 0

    @property
    def entity_count(self) -> int:
        with self._lock:
            return len(self._entities)

    @property
    def relationship_count(self) -> int:
        with self._lock:
            return len(self._relationships)
