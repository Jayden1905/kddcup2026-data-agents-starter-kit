"""Question-driven agent: 5 LLM calls max, deterministic KG grounding.

Pipeline:
  1. [Code] Scan context, consolidate structured data, build KG
  2. [LLM Call 1] Question → schema plan (what do I need to answer this?)
  3. [Code] Segment docs based on schema plan
  4. [LLM Calls 2-4] Extract targeted data from doc segments
  5. [Code] Merge extracted data into DB, update KG
  6. [LLM Call 5] KG + data + question → answer
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.pipeline.context_scanner import TaskContext, scan_context
from data_agent_baseline.pipeline.kg_builder import (
    KnowledgeGraph,
    build_kg_from_sqlite,
    format_kg_for_llm,
)
from data_agent_baseline.tools.knowledge_graph import consolidate_to_sqlite

logger = logging.getLogger(__name__)

CONSOLIDATED_DB_NAME = "_consolidated.db"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SCHEMA_DESIGN_PROMPT = """You are a data analyst. Given a question and database schema, determine what information is needed to answer it.

CURRENT DATE: {current_date}

DATABASE SCHEMA:
{kg_context}

DOCUMENTS AVAILABLE:
{doc_summary}

DOMAIN KNOWLEDGE:
{knowledge_text}

QUESTION: {question}

Analyze what data is needed. Return ONLY a JSON object:
{{
  "answer_column": "the column name for the answer",
  "sql_attempt": "SQL query against existing tables that might answer the question (or null if docs needed)",
  "needs_doc_extraction": true/false,
  "extract_schema": {{
    "table_name": "name for the extracted table",
    "fields": [
      {{"name": "field_name", "type": "TEXT|INTEGER|REAL", "description": "what to look for"}}
    ],
    "id_field": "primary key field name",
    "relationship_to_existing": "e.g. extracted_table.event_id = event.event_id (or null)"
  }}
}}

RULES:
- If the question can be answered from existing tables alone, set needs_doc_extraction=false and provide sql_attempt.
- If documents contain needed data, set needs_doc_extraction=true and describe what to extract.
- The extract_schema should only include fields NEEDED for this specific question.
- Keep field count minimal — only what's required to answer.
- For text filters in SQL: use LIKE '%keyword%' COLLATE NOCASE instead of exact match.
  The question may phrase values differently from the data (e.g. "Brazilian Portuguese" vs "Portuguese (Brazil)").
- Use CAST(x AS REAL) for division to avoid integer truncation.
- "How many times was X more than Y" = ratio X/Y (decimal), not a count.
""".strip()


EXTRACTION_PROMPT = """Extract records from this text following the schema below.

SCHEMA:
- Table: {table_name}
- ID field: {id_field}
- Fields: {fields_description}

QUESTION (context for what matters): {question}

Return ONLY a JSON array of records:
[
  {{"{id_field}": "value", "field1": "value", ...}},
  ...
]

RULES:
- Extract ALL records that have the required fields.
- If a value was corrected in the text (e.g. "adjusted from X to Y"), use the final value Y.
- If a value is missing, use null.
- Do not include filler/irrelevant text in values.

TEXT:
{text}
""".strip()


ANSWER_PROMPT = """Answer this question using the data below.

CURRENT DATE: {current_date}

DATABASE SCHEMA:
{kg_context}

QUESTION: {question}

DATA (query results):
{data_text}

DOMAIN KNOWLEDGE:
{knowledge_text}

Return ONLY a JSON object:
{{
  "sql": "the SQL query you would use (for reference)",
  "columns": ["col1"],
  "rows": [["value1"], ["value2"]]
}}

RULES:
- Answer must come from the data provided.
- Do not round numbers unless the question asks for it.
- If the answer is a count, the column name should describe what's counted.
- If data is empty or insufficient, still provide your best answer from what's available.
- "How many times was X more than Y" means RATIO = X / Y (a decimal), not a count.
""".strip()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class QuestionDrivenAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        config: dict[str, Any] | None = None,
        log_callback: Any = None,
        **kwargs: Any,
    ):
        self.model = model
        self.config = config or {}
        self.log_callback = log_callback
        self.steps: list[dict[str, Any]] = []

    def run(self, task: PublicTask) -> AgentRunResult:
        """Execute the question-driven pipeline."""
        context_dir = task.context_dir
        question = task.question

        # Clean up stale DB
        stale_db = context_dir / CONSOLIDATED_DB_NAME
        if stale_db.exists():
            try:
                stale_db.unlink()
            except OSError:
                pass

        try:
            # Step 0: Scan context (deterministic)
            ctx = scan_context(context_dir)
            self._log("scan", f"Scanned context: {ctx.task_type}, "
                      f"{len(ctx.structured_sources)} structured, "
                      f"{len(ctx.doc_sources)} docs")

            # Step 1: Consolidate structured data (deterministic)
            db_path = consolidate_to_sqlite(context_dir)
            if not db_path or not db_path.exists():
                db_path = context_dir / CONSOLIDATED_DB_NAME
                sqlite3.connect(str(db_path)).close()

            # Step 2: Build KG (deterministic)
            kg = build_kg_from_sqlite(db_path)
            kg_context = format_kg_for_llm(kg)
            self._log("kg_built", f"KG: {len(kg.tables)} tables, "
                      f"{len(kg.inferred_fks)} inferred FKs")

            # Step 3: LLM Call 1 — Schema design
            schema_plan = self._call_schema_design(question, kg_context, ctx)
            self._log("schema_design", json.dumps(schema_plan, default=str)[:500])

            # Try SQL directly if no doc extraction needed
            if not schema_plan.get("needs_doc_extraction"):
                sql = schema_plan.get("sql_attempt")
                if sql:
                    result = self._try_sql(db_path, sql)
                    if result and result["rows"]:
                        answer = self._call_answer(
                            question, kg_context, result, ctx.knowledge_text
                        )
                        return self._build_result(answer, task)

            # Step 4: Extract from docs if needed
            if schema_plan.get("needs_doc_extraction") and ctx.doc_sources:
                extract_schema = schema_plan.get("extract_schema", {})
                if extract_schema:
                    records = self._extract_from_docs(
                        ctx.doc_sources, extract_schema, question
                    )
                    if records:
                        self._write_extracted_to_db(db_path, extract_schema, records)
                        # Rebuild KG with new table
                        kg = build_kg_from_sqlite(db_path)
                        kg_context = format_kg_for_llm(kg)
                        self._log("extraction_done",
                                  f"Extracted {len(records)} records into "
                                  f"{extract_schema.get('table_name')}")

            # Step 5: LLM Call 5 — Answer with full data
            # Try the SQL from schema plan first, then let LLM figure it out
            data_result = None
            sql = schema_plan.get("sql_attempt")
            if sql:
                data_result = self._try_sql(db_path, sql)

            # If SQL failed or empty, try a broader query
            if not data_result or not data_result["rows"]:
                data_result = self._gather_relevant_data(db_path, kg, question)

            answer = self._call_answer(
                question, kg_context, data_result, ctx.knowledge_text
            )

            # Cleanup
            if db_path.exists():
                try:
                    db_path.unlink()
                except OSError:
                    pass

            return self._build_result(answer, task)

        except Exception as e:
            logger.exception("Pipeline failed")
            # Cleanup on failure
            cleanup_db = context_dir / CONSOLIDATED_DB_NAME
            if cleanup_db.exists():
                try:
                    cleanup_db.unlink()
                except OSError:
                    pass
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=[],
                failure_reason=str(e),
            )

    # ------------------------------------------------------------------
    # LLM Call 1: Schema Design
    # ------------------------------------------------------------------

    def _call_schema_design(
        self, question: str, kg_context: str, ctx: TaskContext
    ) -> dict[str, Any]:
        doc_summary = "None"
        if ctx.doc_sources:
            parts = []
            for doc in ctx.doc_sources:
                parts.append(
                    f"- {doc.path.name} ({doc.size_bytes} bytes)\n"
                    f"  Preview: {doc.preview[:200]}..."
                )
            doc_summary = "\n".join(parts)

        prompt = SCHEMA_DESIGN_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            kg_context=kg_context or "(no structured tables yet)",
            doc_summary=doc_summary,
            knowledge_text=ctx.knowledge_text[:3000] if ctx.knowledge_text else "(none)",
            question=question,
        )

        messages = [
            ModelMessage(role="user", content=prompt),
        ]
        raw = self.model.complete(messages)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # LLM Calls 2-4: Extraction
    # ------------------------------------------------------------------

    def _extract_from_docs(
        self,
        doc_sources: list,
        extract_schema: dict[str, Any],
        question: str,
    ) -> list[dict[str, Any]]:
        """Extract records from docs using the schema from Call 1."""
        all_records: list[dict[str, Any]] = []

        table_name = extract_schema.get("table_name", "extracted")
        id_field = extract_schema.get("id_field", "id")
        fields = extract_schema.get("fields", [])
        fields_desc = ", ".join(
            f"{f['name']} ({f.get('type', 'TEXT')}): {f.get('description', '')}"
            for f in fields
        )

        for doc in doc_sources:
            text = doc.path.read_text(errors="replace")
            # Split into segments of ~8K chars
            segments = self._segment_doc(text, max_chars=8000)

            for segment in segments[:3]:  # Max 3 extraction calls per doc
                prompt = EXTRACTION_PROMPT.format(
                    table_name=table_name,
                    id_field=id_field,
                    fields_description=fields_desc,
                    question=question,
                    text=segment,
                )
                messages = [
                    ModelMessage(role="user", content=prompt),
                ]
                try:
                    raw = self.model.complete(messages)
                    data = self._parse_json(raw)
                    if isinstance(data, list):
                        all_records.extend(r for r in data if isinstance(r, dict))
                    elif isinstance(data, dict) and "records" in data:
                        all_records.extend(
                            r for r in data["records"] if isinstance(r, dict)
                        )
                except Exception:
                    continue

        # Deduplicate by ID
        return self._dedup_records(all_records, id_field)

    def _segment_doc(self, text: str, max_chars: int = 8000) -> list[str]:
        """Split doc into segments at paragraph boundaries."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        segments: list[str] = []
        current: list[str] = []
        current_size = 0

        for para in paragraphs:
            if current_size + len(para) > max_chars and current:
                segments.append("\n\n".join(current))
                current = [para]
                current_size = len(para)
            else:
                current.append(para)
                current_size += len(para)

        if current:
            segments.append("\n\n".join(current))

        return segments

    # ------------------------------------------------------------------
    # LLM Call 5: Answer
    # ------------------------------------------------------------------

    def _call_answer(
        self,
        question: str,
        kg_context: str,
        data_result: dict[str, Any] | None,
        knowledge_text: str,
    ) -> dict[str, Any]:
        if data_result and data_result.get("rows"):
            data_text = self._format_data_as_table(data_result)
        else:
            data_text = "(no data found)"

        prompt = ANSWER_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            kg_context=kg_context,
            question=question,
            data_text=data_text,
            knowledge_text=knowledge_text[:2000] if knowledge_text else "(none)",
        )

        messages = [
            ModelMessage(role="user", content=prompt),
        ]
        raw = self.model.complete(messages)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_sql(self, db_path: Path, sql: str) -> dict[str, Any] | None:
        """Execute SQL and return results."""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()
            return {"columns": columns, "rows": [list(r) for r in rows]}
        except Exception as e:
            self._log("sql_error", f"SQL failed: {e}")
            return None

    def _gather_relevant_data(
        self, db_path: Path, kg: KnowledgeGraph, question: str
    ) -> dict[str, Any]:
        """Gather sample data from all tables as fallback."""
        conn = sqlite3.connect(str(db_path))
        parts: list[str] = []
        for table in kg.tables:
            try:
                cursor = conn.execute(f'SELECT * FROM "{table.name}" LIMIT 20')
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                parts.append(f"TABLE {table.name} ({table.row_count} total rows):")
                parts.append(f"  Columns: {columns}")
                for row in rows[:10]:
                    parts.append(f"  {list(row)}")
            except Exception:
                continue
        conn.close()
        return {"columns": ["raw_data"], "rows": [], "_raw": "\n".join(parts)}

    def _write_extracted_to_db(
        self,
        db_path: Path,
        extract_schema: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> None:
        """Write extracted records to the consolidated DB."""
        table_name = f"_extracted_{extract_schema.get('table_name', 'data')}"
        table_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)

        # Gather all column names from records
        all_cols: dict[str, str] = {}
        for r in records:
            for k, v in r.items():
                if k not in all_cols:
                    if isinstance(v, int):
                        all_cols[k] = "INTEGER"
                    elif isinstance(v, float):
                        all_cols[k] = "REAL"
                    else:
                        all_cols[k] = "TEXT"

        if not all_cols:
            return

        conn = sqlite3.connect(str(db_path))
        col_defs = ", ".join(f'"{c}" {t}' for c, t in all_cols.items())
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')

        col_names = list(all_cols.keys())
        placeholders = ", ".join("?" * len(col_names))
        insert_sql = f'INSERT INTO "{table_name}" ({", ".join(f"{c!r}" for c in col_names)}) VALUES ({placeholders})'

        # Fix the insert SQL to use proper quoting
        quoted_cols = ", ".join(f'"{c}"' for c in col_names)
        insert_sql = f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({placeholders})'

        for r in records:
            values = [r.get(c) for c in col_names]
            try:
                conn.execute(insert_sql, values)
            except Exception:
                continue

        conn.commit()
        conn.close()

    def _dedup_records(
        self, records: list[dict[str, Any]], id_field: str
    ) -> list[dict[str, Any]]:
        """Merge records by ID, keeping non-null values."""
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        for r in records:
            rid = str(r.get(id_field, ""))
            if not rid:
                continue
            if rid not in merged:
                merged[rid] = {}
                order.append(rid)
            for k, v in r.items():
                if v is not None:
                    merged[rid][k] = v

        return [merged[key] for key in order]

    def _format_data_as_table(self, result: dict[str, Any]) -> str:
        """Format SQL result as readable text table."""
        if result.get("_raw"):
            return result["_raw"]
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        if not columns:
            return "(empty)"
        lines = [" | ".join(str(c) for c in columns)]
        lines.append("-" * len(lines[0]))
        for row in rows[:50]:
            lines.append(" | ".join(str(v) for v in row))
        if len(rows) > 50:
            lines.append(f"... ({len(rows)} total rows)")
        return "\n".join(lines)

    def _parse_json(self, raw: str) -> Any:
        """Parse JSON from LLM response, handling markdown fences."""
        raw = raw.strip()
        # Strip markdown fences
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1).strip()
        # Try to find JSON object or array
        for start, end in [("{", "}"), ("[", "]")]:
            idx = raw.find(start)
            if idx >= 0:
                # Find matching end
                depth = 0
                for i in range(idx, len(raw)):
                    if raw[i] == start:
                        depth += 1
                    elif raw[i] == end:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(raw[idx:i + 1])
                            except json.JSONDecodeError:
                                break
                break
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}

    def _log(self, action: str, detail: str) -> None:
        """Log a pipeline step."""
        step = {"action": action, "detail": detail}
        self.steps.append(step)
        if self.log_callback:
            self.log_callback(step)

    def _build_result(self, answer: dict[str, Any], task: PublicTask) -> AgentRunResult:
        """Convert LLM answer to AgentRunResult."""
        step_records = [
            StepRecord(
                step_index=i + 1,
                thought=s.get("detail", ""),
                action=s.get("action", ""),
                action_input={},
                raw_response="",
                observation=s,
                ok=True,
            )
            for i, s in enumerate(self.steps)
        ]

        if not answer:
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=step_records,
                failure_reason="No answer produced",
            )

        columns = answer.get("columns", [])
        rows = answer.get("rows", [])

        if not columns or not rows:
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=step_records,
                failure_reason="Empty answer",
            )

        # Ensure all row values are strings for CSV output
        str_rows = [[str(v) for v in row] for row in rows]

        return AgentRunResult(
            task_id=task.task_id,
            answer=AnswerTable(columns=columns, rows=str_rows),
            steps=step_records,
            failure_reason=None,
        )
