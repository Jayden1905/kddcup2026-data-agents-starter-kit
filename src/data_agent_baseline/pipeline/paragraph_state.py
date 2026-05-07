"""Paragraph state machine for parallel document extraction.

Uses structure-aware parsing to skip noise sections, then tracks assignment atomically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock

from data_agent_baseline.pipeline.md_parser import build_extraction_units


class UnitState(Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"


@dataclass
class ParagraphUnit:
    unit_id: int
    doc_path: Path
    text: str
    token_estimate: int
    state: UnitState = UnitState.PENDING
    claimed_by: str | None = None


MAX_TOKENS_PER_UNIT = 3000


class ParagraphStateMachine:
    """Thread-safe state machine tracking paragraph extraction progress."""

    def __init__(self, doc_paths: list[Path], max_tokens_per_unit: int = MAX_TOKENS_PER_UNIT):
        self._lock = Lock()
        self._units: list[ParagraphUnit] = []
        self._next_scan_idx: int = 0
        self._build_units(doc_paths, max_tokens_per_unit)

    def _build_units(self, doc_paths: list[Path], max_tokens: int) -> None:
        unit_id = 0
        for doc_path in doc_paths:
            try:
                extraction_units = build_extraction_units(doc_path, max_tokens_per_unit=max_tokens)
            except (OSError, UnicodeDecodeError):
                continue

            for eu in extraction_units:
                self._units.append(ParagraphUnit(
                    unit_id=unit_id,
                    doc_path=doc_path,
                    text=eu.text,
                    token_estimate=eu.token_estimate,
                ))
                unit_id += 1

    def claim_next(self, agent_id: str) -> ParagraphUnit | None:
        """Atomically claim the next pending unit. Returns None if all claimed/done."""
        with self._lock:
            for i in range(self._next_scan_idx, len(self._units)):
                unit = self._units[i]
                if unit.state == UnitState.PENDING:
                    unit.state = UnitState.CLAIMED
                    unit.claimed_by = agent_id
                    self._next_scan_idx = i + 1
                    return unit
            return None

    def mark_done(self, unit_id: int) -> None:
        with self._lock:
            self._units[unit_id].state = UnitState.DONE

    @property
    def all_done(self) -> bool:
        with self._lock:
            return all(u.state == UnitState.DONE for u in self._units)

    @property
    def total_units(self) -> int:
        return len(self._units)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for u in self._units if u.state == UnitState.PENDING)
