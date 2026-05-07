"""Merges extracted records by ID and applies LLM resolutions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.pipeline.field_discoverer import DocumentSchema
from data_agent_baseline.pipeline.rule_extractor import (
    Confidence,
    ExtractedRecord,
    ExtractedValue,
)


@dataclass
class MergedRecord:
    record_id: str  # composite key (e.g. "3182521_1986-02-10")
    entity_type: str
    fields: dict[str, Any] = field(default_factory=dict)  # final values only (no wrappers)


@dataclass
class MergeResult:
    records: list[MergedRecord] = field(default_factory=list)
    entity_type: str = ""


def _build_composite_key(record: ExtractedRecord, schema: DocumentSchema) -> str:
    """Build composite key: if schema.has_multiple_records_per_id, key = id_date."""
    if schema.has_multiple_records_per_id:
        date_val = None
        for fname, val in record.fields.items():
            if "date" in fname.lower():
                if isinstance(val, ExtractedValue):
                    date_val = val.value
                else:
                    date_val = val
                break
        if date_val:
            return f"{record.record_id}_{date_val}"
    return record.record_id


def _pick_best_value(values: list[ExtractedValue]) -> Any:
    """Pick the best value from a list of candidates.

    Priority: HIGH confidence > correction > latest in list.
    """
    # Prefer HIGH confidence
    for v in values:
        if v.confidence == Confidence.HIGH:
            return v.value

    # Prefer corrections
    for v in values:
        if v.is_correction:
            return v.value

    # Fall back to last (latest) value
    if values:
        return values[-1].value

    return None


def merge_records(
    extracted: list[ExtractedRecord],
    resolutions: dict[tuple[str, str], Any],
    schema: DocumentSchema,
) -> MergeResult:
    """Merge records by composite key.

    Strategy:
    1. Group by record_id
    2. For duplicate fields across paragraphs: prefer HIGH confidence, then correction,
       then latest
    3. Apply LLM resolutions
    4. Build composite key: if schema.has_multiple_records_per_id, key = id_date
       otherwise key = record_id
    5. Include the natural ID (patient_id) as a separate field for FK joins
    """
    # Group records by composite key
    grouped: dict[str, list[ExtractedRecord]] = defaultdict(list)
    for record in extracted:
        key = _build_composite_key(record, schema)
        grouped[key].append(record)

    merged_records: list[MergedRecord] = []

    for composite_key, group in grouped.items():
        merged = MergedRecord(
            record_id=composite_key,
            entity_type=schema.entity_name,
        )

        # Collect all values per field
        field_values: dict[str, list[ExtractedValue]] = defaultdict(list)
        for record in group:
            for fname, val in record.fields.items():
                if isinstance(val, ExtractedValue):
                    field_values[fname].append(val)

        # Pick best value for each field
        for fname, values in field_values.items():
            # Check if LLM resolution exists for this record+field
            resolution_key = (composite_key, fname)
            if resolution_key in resolutions:
                merged.fields[fname] = resolutions[resolution_key]
            else:
                merged.fields[fname] = _pick_best_value(values)

        # Include the natural ID as a separate field for FK joins
        if group:
            natural_id = group[0].record_id
            if schema.id_field and schema.id_field not in merged.fields:
                merged.fields[schema.id_field] = natural_id

        merged_records.append(merged)

    return MergeResult(records=merged_records, entity_type=schema.entity_name)
