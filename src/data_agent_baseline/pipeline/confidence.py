"""Confidence scoring and ambiguity detection for extracted records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_baseline.pipeline.rule_extractor import (
    Confidence,
    ExtractedRecord,
    ExtractedValue,
)


@dataclass(frozen=True, slots=True)
class AmbiguityFlag:
    record_id: str
    field_name: str
    reason: str  # "multiple_values", "unclear_association", "no_value_found"
    candidates: list[Any]
    paragraph_text: str


def score_and_flag(records: list[ExtractedRecord]) -> list[AmbiguityFlag]:
    """Analyze records, return ambiguity flags for items needing LLM resolution.

    Only flag if:
    - A field has alternatives AND is_correction=False (correction already resolved it)
    - A field has LOW confidence
    """
    flags: list[AmbiguityFlag] = []

    for record in records:
        for field_name, value in record.fields.items():
            if not isinstance(value, ExtractedValue):
                continue

            # Skip fields already resolved by correction
            if value.is_correction:
                continue

            # Flag low confidence values
            if value.confidence == Confidence.LOW:
                flags.append(
                    AmbiguityFlag(
                        record_id=record.record_id,
                        field_name=field_name,
                        reason="unclear_association",
                        candidates=[value.value],
                        paragraph_text="",
                    )
                )
            # Flag fields with multiple candidate values
            elif value.alternatives:
                flags.append(
                    AmbiguityFlag(
                        record_id=record.record_id,
                        field_name=field_name,
                        reason="multiple_values",
                        candidates=[value.value, *value.alternatives],
                        paragraph_text="",
                    )
                )
            # Flag missing values
            elif value.value is None:
                flags.append(
                    AmbiguityFlag(
                        record_id=record.record_id,
                        field_name=field_name,
                        reason="no_value_found",
                        candidates=[],
                        paragraph_text="",
                    )
                )

    return flags
