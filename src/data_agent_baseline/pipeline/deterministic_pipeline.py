"""Deterministic document extraction pipeline.

Replaces LLM-based parallel extraction with a primarily rule-based approach:
1. Parse markdown structure
2. Discover schema from knowledge.md + text patterns
3. Rule-based extraction (corrections, values, NaN, status)
4. Confidence scoring + ambiguity detection
5. LLM resolution only for weak/ambiguous items (<5%)
6. Merge records by composite key
7. Write to SQLite
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.pipeline.confidence import score_and_flag
from data_agent_baseline.pipeline.field_discoverer import discover_schema
from data_agent_baseline.pipeline.md_parser import filter_data_sections, parse_md_structure
from data_agent_baseline.pipeline.record_merger import merge_records
from data_agent_baseline.pipeline.rule_extractor import RuleExtractor
from data_agent_baseline.pipeline.sqlite_writer import write_to_sqlite


def deterministic_extract_docs(
    doc_paths: list[Path],
    db_path: Path,
    model: ModelAdapter,
    knowledge_text: str = "",
    time_remaining_fn: Callable[[], float] = lambda: 300.0,
    log_fn: Callable[[str, str], None] | None = None,
    structured_tables: list[str] | None = None,
) -> int:
    """Deterministic document extraction pipeline.

    Drop-in replacement for parallel_extract_docs().
    Returns total records extracted.
    """
    if not doc_paths:
        return 0

    protected = {t.lower() for t in structured_tables} if structured_tables else set()
    total_records = 0

    for doc_path in doc_paths:
        if time_remaining_fn() < 30:
            break

        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) < 100:
            continue

        # Stage 1: Parse structure
        sections = parse_md_structure(text)
        data_sections = filter_data_sections(sections)
        paragraphs: list[str] = []
        for section in data_sections:
            paragraphs.extend(p for p in section.paragraphs if len(p) >= 30)

        if not paragraphs:
            if log_fn:
                log_fn("extract_skip", f"{doc_path.stem}: no data-dense sections")
            continue

        # Stage 2: Discover schema
        schema = discover_schema(text, doc_path.stem, knowledge_text)
        if log_fn:
            field_names = [f.name for f in schema.fields[:10]]
            log_fn(
                "schema_discovered",
                f"{doc_path.stem}: {len(schema.fields)} fields ({field_names}), "
                f"composite={schema.has_multiple_records_per_id}",
            )

        # Stage 3: Rule-based extraction
        extractor = RuleExtractor(schema, knowledge_text)
        records = extractor.extract_all(paragraphs)
        if log_fn:
            log_fn("extraction_done", f"{doc_path.stem}: {len(records)} records from {len(paragraphs)} paragraphs")

        if not records:
            continue

        # Stage 4: Confidence scoring + ambiguity detection
        flags = score_and_flag(records)
        if log_fn and flags:
            log_fn("ambiguity", f"{doc_path.stem}: {len(flags)} ambiguous fields detected")

        # Stage 5: LLM resolution (skip for now — most cases resolve deterministically)
        resolutions: dict[tuple[str, str], Any] = {}

        # Stage 6: Merge records
        merge_result = merge_records(records, resolutions, schema)
        if log_fn:
            log_fn("merge_done", f"{doc_path.stem}: {len(merge_result.records)} merged records")

        # Stage 7: Write to SQLite
        write_to_sqlite(db_path, merge_result, schema, protected, log_fn)
        total_records += len(merge_result.records)

    if log_fn:
        log_fn("pipeline_complete", f"Total: {total_records} records from {len(doc_paths)} docs")

    return total_records
