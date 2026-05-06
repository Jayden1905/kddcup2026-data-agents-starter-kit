from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AnomalyFlag:
    step_id: str
    rule: str
    detail: str


def detect_anomalies(evidence_entries: dict[str, dict[str, Any]]) -> list[AnomalyFlag]:
    flags: list[AnomalyFlag] = []
    for step_id, entry in evidence_entries.items():
        if not entry.get("ok", True):
            flags.append(AnomalyFlag(step_id, "error_result", entry.get("error", "tool error")))
            continue

        content = entry.get("content")
        if content is None:
            flags.append(AnomalyFlag(step_id, "null_content", "tool returned None"))
            continue

        if isinstance(content, dict):
            rows = content.get("rows")
            if isinstance(rows, list) and len(rows) == 0:
                flags.append(AnomalyFlag(step_id, "empty_result", "query returned 0 rows"))

            if isinstance(rows, list) and rows:
                all_null = all(
                    all(v is None or str(v).strip().upper() in ("", "NONE", "NULL") for v in row)
                    for row in rows
                )
                if all_null:
                    flags.append(AnomalyFlag(step_id, "all_null_rows", "every value is null/empty"))

        if isinstance(content, str):
            stripped = content.strip()
            if not stripped:
                flags.append(AnomalyFlag(step_id, "empty_string", "tool returned empty string"))
            elif stripped.lower().startswith("error"):
                flags.append(AnomalyFlag(step_id, "error_in_output", stripped[:200]))

    return flags


def format_anomaly_flags(flags: list[AnomalyFlag]) -> str:
    if not flags:
        return ""
    lines = ["ANOMALIES DETECTED IN EVIDENCE:"]
    for f in flags:
        lines.append(f"  - [{f.step_id}] {f.rule}: {f.detail}")
    return "\n".join(lines)
