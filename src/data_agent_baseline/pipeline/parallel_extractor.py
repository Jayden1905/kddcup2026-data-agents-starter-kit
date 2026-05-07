"""Parallel document extraction with multiple agents per document.

Architecture:
- 4 agents per doc claim paragraphs from a shared state machine
- Each agent extracts entities/relationships via LLM and pushes to a dedup queue
- A graph builder runs concurrently, consuming from the queue and writing to SQLite
- Schema hints (from existing structured data) guide extraction when available
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.pipeline.entity_queue import EntityQueue, Relationship
from data_agent_baseline.pipeline.graph_builder_parallel import GraphBuilder
from data_agent_baseline.pipeline.paragraph_state import ParagraphStateMachine, ParagraphUnit


SCHEMA_GUIDED_PROMPT = """Extract ALL {entity_type} records from this text. Capture every piece of data.
{knowledge_section}
TEXT:
{paragraph_text}

Return ONLY a JSON array:
[{{"id": "unique_row_key", "fields": {{"field1": "value1", "field2": "value2"}}}}]

RULES:
- Extract EVERY {entity_type} mentioned. Do not skip any.
- Include ALL data for each record — every number, date, status, category, description, flag, label.
- Use the CORRECTED/FINAL value when the text mentions corrections or revisions.
- "id" must be UNIQUE per row. If an entity (e.g. patient) has multiple records (e.g. lab tests on different dates), use a composite key like "patientID_date" (e.g. "3182521_1986-02-10").
- Always include the entity's natural ID as a field (e.g. "patient_id": "3182521") so tables can be joined.
- Return ONLY the JSON array, nothing else.
"""

DISCOVERY_PROMPT = """Extract ALL entities from this text. Capture every piece of data.
{knowledge_section}
TEXT:
{paragraph_text}

Return ONLY a JSON object:
{{"entities": [{{"type": "EntityType", "id": "unique_row_key", "fields": {{"field1": "value1"}}}}]}}

RULES:
- Identify entity types from the text (e.g. Patient, Laboratory, Examination, Product, etc.)
- Include ALL data for each entity — every number, date, status, category, description, flag, label.
- Use the CORRECTED/FINAL value when the text mentions corrections or revisions.
- "id" must be UNIQUE per row. If an entity has multiple records (e.g. same patient, different dates), use a composite key like "entityID_date" (e.g. "3182521_1986-02-10").
- Always include the entity's natural ID as a field (e.g. "patient_id": "3182521") so tables can be joined.
- Do NOT skip any entity — extract every one mentioned.
- Return ONLY the JSON, nothing else.
"""


class ExtractionAgent:
    """A single extraction agent that claims and processes paragraphs."""

    def __init__(
        self,
        agent_id: str,
        model: ModelAdapter,
        state_machine: ParagraphStateMachine,
        queue: EntityQueue,
        schema_hints: dict[str, list[str]] | None,
        knowledge_text: str,
        time_remaining_fn: Callable[[], float],
        log_fn: Callable[[str, str], None] | None = None,
        structured_tables: set[str] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.model = model
        self.state_machine = state_machine
        self.queue = queue
        self.schema_hints = schema_hints
        self.knowledge_text = knowledge_text
        self.time_remaining_fn = time_remaining_fn
        self.log_fn = log_fn
        self._structured_tables = structured_tables or set()

    def run(self) -> int:
        """Claim-extract-push loop. Returns number of paragraphs processed."""
        processed = 0
        if self.log_fn:
            self.log_fn("agent_start", f"{self.agent_id} started")
        while True:
            if self.time_remaining_fn() < 30:
                if self.log_fn:
                    self.log_fn("agent_timeout", f"{self.agent_id} stopping (low time)")
                break
            unit = self.state_machine.claim_next(self.agent_id)
            if unit is None:
                break
            if self.log_fn:
                self.log_fn("agent_claim", f"{self.agent_id} claimed unit {unit.unit_id} ({unit.token_estimate} tokens)")
            self._process_unit(unit)
            self.state_machine.mark_done(unit.unit_id)
            processed += 1
        if self.log_fn:
            self.log_fn("agent_done", f"{self.agent_id} finished ({processed} units)")
        return processed

    def _process_unit(self, unit: ParagraphUnit) -> None:
        # Determine entity type from doc filename
        entity_type = self._get_entity_type_for_doc(unit.doc_path)
        prompt = self._build_prompt(unit.text, entity_type)
        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self.model.complete(messages)
        except RuntimeError:
            return

        parsed = self._parse_json(raw)
        entity_count = 0

        if entity_type and isinstance(parsed, list):
            # Schema-guided: response is a JSON array of entities for one type
            for entity in parsed:
                if not isinstance(entity, dict):
                    continue
                eid = str(entity.get("id", ""))
                fields = entity.get("fields", {})
                if eid and isinstance(fields, dict):
                    self.queue.push_entity(entity_type, eid, fields)
                    entity_count += 1
        elif isinstance(parsed, dict):
            # Discovery mode: response has "entities" key
            entities = parsed.get("entities", [])
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                etype = entity.get("type", "")
                eid = str(entity.get("id", ""))
                fields = entity.get("fields", {})
                if etype and eid and isinstance(fields, dict):
                    self.queue.push_entity(etype, eid, fields)
                    entity_count += 1

        if self.log_fn:
            self.log_fn(
                "extract_agent",
                f"{self.agent_id}: unit {unit.unit_id} ({unit.doc_path.stem}) → {entity_count} entities",
            )

    def _get_entity_type_for_doc(self, doc_path: Path) -> str | None:
        """Use doc filename as entity type (e.g. Patient.md → Patient).

        If a structured table with the same name exists, suffix with _doc
        to avoid collision (e.g. Patient_doc).
        """
        stem = doc_path.stem
        if stem and stem[0].isupper():
            if stem.lower() in self._structured_tables:
                return f"{stem}_doc"
            return stem
        return None

    def _build_prompt(self, text: str, entity_type: str | None = None) -> str:
        knowledge_section = ""
        if self.knowledge_text:
            knowledge_section = f"\nDOMAIN KNOWLEDGE:\n{self.knowledge_text[:1500]}\n"

        if entity_type:
            return SCHEMA_GUIDED_PROMPT.format(
                entity_type=entity_type,
                knowledge_section=knowledge_section,
                paragraph_text=text,
            )
        return DISCOVERY_PROMPT.format(
            knowledge_section=knowledge_section,
            paragraph_text=text,
        )

    def _parse_json(self, raw: str) -> Any:
        if not raw:
            return {}
        raw = raw.strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1).strip()
        # Find whichever delimiter comes first
        brace_idx = raw.find("{")
        bracket_idx = raw.find("[")
        if bracket_idx >= 0 and (brace_idx < 0 or bracket_idx < brace_idx):
            pairs = [("[", "]"), ("{", "}")]
        else:
            pairs = [("{", "}"), ("[", "]")]
        for start, end in pairs:
            idx = raw.find(start)
            if idx >= 0:
                depth = 0
                for i in range(idx, len(raw)):
                    if raw[i] == start:
                        depth += 1
                    elif raw[i] == end:
                        depth -= 1
                        if depth == 0:
                            candidate = raw[idx:i + 1]
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                                try:
                                    return json.loads(fixed)
                                except json.JSONDecodeError:
                                    break
                break
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}


def _derive_schema_hints(
    db_path: Path,
    structured_tables: list[str] | None = None,
    knowledge_text: str = "",
) -> dict[str, list[str]] | None:
    """Derive schema hints from structured tables OR knowledge.md entity definitions.

    Priority:
    1. Structured tables from DB (CSV/JSON loaded in Step 2) — exact columns
    2. Knowledge.md entity definitions (### EntityName + **Field (type)**) — for doc-only tasks
    """
    hints: dict[str, list[str]] = {}

    # From structured DB tables
    if structured_tables and db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            for table_name in structured_tables:
                cursor = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 0')
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                if columns:
                    hints[table_name] = columns
            conn.close()
        except Exception:
            pass

    # From knowledge.md — parse "### EntityName" sections with "- **FieldName (type)**"
    if not hints and knowledge_text:
        current_entity = ""
        current_fields: list[str] = []
        for line in knowledge_text.split("\n"):
            heading = re.match(r"^###\s+(\w+)\s*$", line)
            if heading:
                if current_entity and current_fields:
                    hints[current_entity] = current_fields
                current_entity = heading.group(1)
                current_fields = []
            elif current_entity:
                field_match = re.match(r"\s*-\s*\*\*(\w+)\s*\(", line)
                if field_match:
                    current_fields.append(field_match.group(1))
        if current_entity and current_fields:
            hints[current_entity] = current_fields

    return hints if hints else None


def parallel_extract_docs(
    doc_paths: list[Path],
    db_path: Path,
    model: ModelAdapter,
    knowledge_text: str = "",
    time_remaining_fn: Callable[[], float] = lambda: 300.0,
    agents_per_doc: int = 4,
    log_fn: Callable[[str, str], None] | None = None,
    structured_tables: list[str] | None = None,
) -> int:
    """Run parallel extraction on all documents, writing results to db_path.

    Args:
        structured_tables: Table names from structured data (CSV/JSON) loaded in Step 2.
            Pass empty list for doc-only tasks to force discovery mode.

    Returns total entities extracted.
    """
    if not doc_paths:
        return 0

    # 1. Schema hints from structured tables OR knowledge.md entity definitions
    schema_hints = _derive_schema_hints(db_path, structured_tables=structured_tables, knowledge_text=knowledge_text)

    # 2. Build state machine across all docs
    state_machine = ParagraphStateMachine(doc_paths)
    if state_machine.total_units == 0:
        return 0

    # 3. Create entity queue
    queue = EntityQueue()

    # 4. Start graph builder thread (runs concurrently with extraction)
    protected = set(structured_tables) if structured_tables else set()
    builder = GraphBuilder(db_path, queue, log_fn=log_fn, protected_tables=protected)
    builder_thread = threading.Thread(target=builder.run, daemon=True)
    builder_thread.start()

    # 5. Determine number of agents (4 per doc, capped by total units)
    total_agents = min(agents_per_doc * len(doc_paths), state_machine.total_units)

    if log_fn:
        log_fn(
            "parallel_extract_start",
            f"{len(doc_paths)} docs, {state_machine.total_units} units, "
            f"{total_agents} agents, schema_hints={list(schema_hints.keys()) if schema_hints else 'discovery'}",
        )

    # 6. Create extraction agents — each gets its own model adapter for independent connections
    from data_agent_baseline.agents.model import OpenAIModelAdapter

    def _make_agent_model() -> ModelAdapter:
        return OpenAIModelAdapter(
            model=model.model,
            api_base=model.api_base,
            api_key=model.api_key,
            temperature=model.temperature,
        )

    structured_set = {t.lower() for t in structured_tables} if structured_tables else set()
    agents = [
        ExtractionAgent(
            agent_id=f"ext_{i}",
            model=_make_agent_model(),
            state_machine=state_machine,
            queue=queue,
            schema_hints=schema_hints,
            knowledge_text=knowledge_text,
            time_remaining_fn=time_remaining_fn,
            log_fn=log_fn,
            structured_tables=structured_set,
        )
        for i in range(total_agents)
    ]

    with ThreadPoolExecutor(max_workers=total_agents) as executor:
        futures = [executor.submit(agent.run) for agent in agents]
        results = [f.result() for f in futures]

    # 7. Stop graph builder and wait for final drain
    builder.stop()
    builder_thread.join(timeout=10.0)

    total_processed = sum(results)
    if log_fn:
        log_fn(
            "parallel_extract_done",
            f"Processed {total_processed}/{state_machine.total_units} units, "
            f"entities in DB",
        )

    return total_processed
