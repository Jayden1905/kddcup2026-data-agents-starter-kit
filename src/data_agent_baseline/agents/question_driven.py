"""Question-driven agent: semantic grounding + closed-loop SQL.

Pipeline:
  1. [Code] Scan context, consolidate structured data → SQLite
  2. [Code] Deterministic doc extraction → additional tables in SQLite
  3. [Code] Build KG from full SQLite (structured + extracted)
  4. [LLM] Semantic grounding: question + schema → structured decomposition
  5. [LLM] Closed loop: SQL generation → execute → evaluate → iterate
  6. [LLM] Answer formatting
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.pipeline.context_scanner import TaskContext, scan_context
from data_agent_baseline.pipeline.doc_extractor import (
    extract_all_docs,
    write_extracted_table,
)
from data_agent_baseline.pipeline.llm_extractor import (
    discover_schema,
    extract_from_docs,
    should_use_llm_extraction,
    write_llm_extracted_table,
)
from data_agent_baseline.pipeline.kg_builder import (
    KnowledgeGraph,
    build_kg_from_sqlite,
    format_kg_for_llm,
)
from data_agent_baseline.tools.knowledge_graph import consolidate_to_sqlite

logger = logging.getLogger(__name__)

CONSOLIDATED_DB_NAME = "_consolidated.db"
TASK_TIME_BUDGET_SECONDS = 480  # bail before 600s hard timeout


# ---------------------------------------------------------------------------
# Prompt builders — dynamic, only include sections that have content
# ---------------------------------------------------------------------------

SQL_RULES = [
    "Use only tables and columns shown in the schema.",
    "SELECT only columns that answer the question. Never SELECT *.",
    "Check SAMPLE DATA for date ranges and formats before writing WHERE clauses.",
    "If a table's date range doesn't cover the period, start from a different table.",
    "JOIN through the full FK path. Never skip intermediate linking tables.",
    "\"X containing Y\" means filter X itself (WHERE X.prop = Y).",
    "Use LIKE '%keyword%' COLLATE NOCASE for text. Use CAST(x AS REAL) for division.",
    "Never return raw record IDs — JOIN to get human-readable names.",
    "Never use LIMIT unless the question asks for a specific count.",
]


def _build_sql_prompt(
    *,
    question: str,
    kg_context: str,
    sample_data: str,
    knowledge_text: str = "",
    column_hints: str = "",
    gaps: str = "",
    extra_context: str = "",
    grounding_context: str = "",
) -> str:
    parts = ["Write a SQL query to answer the question."]

    parts.append(f"\nDATABASE SCHEMA:\n{kg_context}")

    if sample_data:
        parts.append(f"\nSAMPLE DATA:\n{sample_data[:4000]}")

    if grounding_context:
        parts.append(f"\n{grounding_context}")

    if column_hints:
        parts.append(f"\n{column_hints}")

    if knowledge_text:
        parts.append(f"\nDOMAIN KNOWLEDGE:\n{knowledge_text[:2000]}")

    if gaps:
        parts.append(f"\nPREVIOUS ATTEMPT FAILED — fix these issues:\n{gaps}")

    if extra_context:
        parts.append(f"\nEXPLORATORY RESULTS:\n{extra_context}")

    parts.append(f"\nQUESTION: {question}")

    parts.append('\nReturn ONLY a JSON object:\n{"thought": "reasoning", "sql": "SELECT ..."}')

    rules = "\n".join(f"- {r}" for r in SQL_RULES)
    if grounding_context and "FORMULA" in grounding_context:
        rules += "\n- Follow the FORMULA in SEMANTIC GROUNDING exactly."
    elif knowledge_text and "formula" in knowledge_text.lower():
        rules += "\n- If DOMAIN KNOWLEDGE defines a formula, follow it exactly."
    parts.append(f"\nRULES:\n{rules}")

    return "\n".join(parts)


def _build_evaluate_prompt(
    *,
    question: str,
    sql: str,
    sql_error: str,
    data_text: str,
    kg_context: str = "",
) -> str:
    parts = ["Evaluate whether these query results answer the question."]

    parts.append(f"\nQUESTION: {question}")

    if kg_context:
        parts.append(f"\nSCHEMA:\n{kg_context}")

    parts.append(f"\nSQL: {sql or '(none)'}")
    if sql_error:
        parts.append(f"ERROR: {sql_error}")

    parts.append(f"\nRESULTS:\n{data_text}")

    parts.append("""
Return ONLY a JSON object:
{"verdict": "complete"/"incomplete", "reasoning": "why", "gaps": [], "info_queries": [], "suggested_sql": "..."}

- "complete" = data answers the question. Don't over-think.
- "incomplete" = error, empty, or wrong data.
- suggested_sql must be a DIFFERENT approach. Never repeat the same failing query.""")

    return "\n".join(parts)


SEMANTIC_GROUNDING_PROMPT = """Decompose this question into a structured plan for SQL generation.

QUESTION: {question}

DATABASE SCHEMA:
{kg_context}
{sample_section}
{knowledge_section}
Return ONLY a JSON object with these fields:
{{
  "formula": "mathematical formula or aggregation needed (e.g. SUM(amount)/COUNT(*), MAX(date) - MIN(date)). Use 'direct_lookup' if no calculation needed.",
  "computation_steps": ["step1: find X", "step2: calculate Y from X"],
  "data_requirements": ["table.column needed", "table2.column2 for filter"],
  "reasoning": "brief explanation of HOW to get the answer from the data",
  "domain_rules": ["any constraints: date formats, units, special cases"],
  "known_values": {{"column_name": ["specific values to filter on"]}},
  "join_paths": ["tableA.col -> tableB.col -> tableC.col"]
}}

RULES:
- For join_paths, trace the FULL path from source to target using FK relationships in the schema.
- For known_values, use EXACT values from DOMAIN KNOWLEDGE definitions. If knowledge says "X means value Y", use Y.
- For formula, write the actual SQL expression (SUM, COUNT, AVG, etc.) or "direct_lookup".
- Keep computation_steps as concrete SQL operations, not abstract descriptions.
- Pay attention to DOMAIN KNOWLEDGE definitions — they override common-sense interpretations.
""".strip()

GROUNDING_VALIDATE_PROMPT = """Check if this grounding plan correctly answers the question.

QUESTION: {question}

GROUNDING PLAN:
{grounding_json}
{knowledge_section}
{sample_section}
Verify:
1. Do known_values use the EXACT correct values? (Check DOMAIN KNOWLEDGE definitions carefully)
2. Are join_paths complete — no missing intermediate tables?
3. Does the formula match what the question asks (count, list, sum, average, etc.)?
4. Are there any ambiguous terms in the question that need clarification from DOMAIN KNOWLEDGE?

Return ONLY a JSON object:
{{"verdict": "sufficient"/"insufficient", "issues": ["issue1", "issue2"], "corrections": {{"field_name": "corrected_value"}}}}

- "sufficient" = plan is correct and complete.
- "insufficient" = something is wrong or missing. List issues and provide corrections.
""".strip()


def _build_semantic_prompt(
    *,
    question: str,
    kg_context: str,
    sample_data: str = "",
    knowledge_text: str = "",
) -> str:
    sample_section = f"\nSAMPLE DATA:\n{sample_data[:3000]}" if sample_data else ""
    knowledge_section = f"\nDOMAIN KNOWLEDGE:\n{knowledge_text[:1500]}" if knowledge_text else ""
    return SEMANTIC_GROUNDING_PROMPT.format(
        question=question,
        kg_context=kg_context,
        sample_section=sample_section,
        knowledge_section=knowledge_section,
    )


def _build_grounding_validate_prompt(
    *,
    question: str,
    grounding: dict[str, Any],
    knowledge_text: str = "",
    sample_data: str = "",
) -> str:
    knowledge_section = f"\nDOMAIN KNOWLEDGE:\n{knowledge_text[:1500]}" if knowledge_text else ""
    sample_section = f"\nSAMPLE DATA:\n{sample_data[:2000]}" if sample_data else ""
    return GROUNDING_VALIDATE_PROMPT.format(
        question=question,
        grounding_json=json.dumps(grounding, indent=2)[:3000],
        knowledge_section=knowledge_section,
        sample_section=sample_section,
    )


def _format_grounding_for_sql(grounding: dict[str, Any]) -> str:
    """Format semantic grounding output as structured context for SQL generation."""
    parts: list[str] = []

    formula = grounding.get("formula", "")
    if formula and formula != "direct_lookup":
        parts.append(f"FORMULA: {formula}")

    steps = grounding.get("computation_steps", [])
    if steps:
        parts.append("STEPS:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)))

    join_paths = grounding.get("join_paths", [])
    if join_paths:
        parts.append("JOIN PATHS:\n" + "\n".join(f"  {jp}" for jp in join_paths))

    known_values = grounding.get("known_values", {})
    if known_values:
        kv_lines = [f"  {k} IN ({', '.join(repr(v) for v in vs)})"
                    for k, vs in known_values.items() if vs]
        if kv_lines:
            parts.append("FILTER VALUES:\n" + "\n".join(kv_lines))

    domain_rules = grounding.get("domain_rules", [])
    if domain_rules:
        parts.append("CONSTRAINTS:\n" + "\n".join(f"  - {r}" for r in domain_rules))

    reasoning = grounding.get("reasoning", "")
    if reasoning:
        parts.append(f"APPROACH: {reasoning}")

    if not parts:
        return ""
    return "SEMANTIC GROUNDING:\n" + "\n".join(parts)


def _build_answer_prompt(
    *,
    question: str,
    data_text: str,
    knowledge_text: str = "",
) -> str:
    parts = [f"Answer this question from the data.\n\nQUESTION: {question}"]

    parts.append(f"\nDATA:\n{data_text}")

    if knowledge_text:
        parts.append(f"\nDOMAIN KNOWLEDGE:\n{knowledge_text[:2000]}")

    parts.append("""
Return ONLY a JSON object:
{"columns": ["col1"], "rows": [["value1"], ["value2"]]}

- Only include columns that answer the question.
- Return ALL rows — never truncate.""")

    return "\n".join(parts)


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
        self._start_time = time.monotonic()
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
            # Step 1: Scan context (deterministic, instant)
            ctx = scan_context(context_dir)
            self._log("scan", f"Scanned: {ctx.task_type}, "
                      f"{len(ctx.structured_sources)} structured, "
                      f"{len(ctx.doc_sources)} docs")

            # Step 2: Consolidate structured data → SQLite (deterministic)
            # Force-remove any stale DB that might block consolidation
            for stale in context_dir.glob("_consolidated*.db"):
                try:
                    stale.unlink()
                except OSError:
                    pass
            db_path = consolidate_to_sqlite(context_dir)
            if not db_path or not db_path.exists():
                db_path = context_dir / CONSOLIDATED_DB_NAME
                sqlite3.connect(str(db_path)).close()

            # Step 3: Doc extraction
            extracted_tables = []
            if ctx.doc_sources:
                doc_paths = [doc.path for doc in ctx.doc_sources]
                # First try fast regex extraction
                extracted_tables = extract_all_docs(doc_paths, db_path=db_path)
                for table in extracted_tables:
                    write_extracted_table(db_path, table)
                    self._log("doc_extracted",
                              f"Table '{table.name}': {len(table.records)} records, "
                              f"FK: {table.fk_links}")

                # If regex is insufficient, use LLM extraction
                if should_use_llm_extraction(ctx.task_type, question, extracted_tables, db_path):
                    self._log("llm_extract_start", "Regex insufficient — using LLM extraction")
                    try:
                        schema = discover_schema(self.model, doc_paths, question)
                        self._log("llm_schema", f"Schema: {schema.get('columns', [])}")
                        llm_tables = extract_from_docs(self.model, doc_paths, schema)
                        for table in llm_tables:
                            write_llm_extracted_table(db_path, table)
                            self._log("llm_extracted",
                                      f"Table '{table.name}': {len(table.records)} records")
                    except RuntimeError as e:
                        self._log("llm_extract_error", str(e))

            # Step 4: Build KG from full DB (deterministic)
            kg = build_kg_from_sqlite(db_path)
            kg_context = format_kg_for_llm(kg)
            self._log("kg_built", f"KG: {len(kg.tables)} tables, "
                      f"{len(kg.inferred_fks)} inferred FKs")

            # Get sample data for each table
            sample_data = self._get_sample_data(db_path, kg)

            # Build column hints: map question words to actual column names
            col_hints = self._build_column_hints(question, kg)

            # Step 5: Semantic grounding — decompose question before SQL planning
            grounding_context = ""
            if self._time_remaining() > 200:
                grounding_context = self._call_semantic_grounding(
                    question, kg_context, sample_data, ctx.knowledge_text
                )

            # ----------------------------------------------------------
            # Closed loop: PLAN → EXECUTE → EVALUATE (max 3 iterations)
            # ----------------------------------------------------------
            max_iterations = 4
            data_result = None
            sql = ""
            gaps_text = ""
            extra_context = ""  # info from exploratory queries

            for iteration in range(1, max_iterations + 1):
                if self._time_remaining() < 120:
                    self._log("time_budget", "Low time — skipping further iterations")
                    break

                self._log("iteration", f"--- Iteration {iteration}/{max_iterations} ---")

                # PLAN: Generate SQL (incorporate gaps + extra context if any)
                sql = self._call_sql(
                    question, kg_context, sample_data,
                    ctx.knowledge_text, gaps=gaps_text,
                    extra_context=extra_context,
                    column_hints=col_hints,
                    grounding_context=grounding_context,
                )
                self._log("sql_generated", sql)

                # EXECUTE: Run the SQL
                sql_error = ""
                data_result = self._try_sql(db_path, sql)
                if data_result is None:
                    sql_error = self.steps[-1].get("detail", "") if self.steps else ""
                    data_result = {"columns": [], "rows": []}

                # Skip evaluate on last iteration or low time — just use what we have
                if iteration == max_iterations or self._time_remaining() < 120:
                    if data_result.get("rows"):
                        break

                # EVALUATE: Check if result answers the question
                eval_result = self._call_evaluate(
                    question, sql, sql_error, data_result, kg_context
                )
                verdict = eval_result.get("verdict", "complete")
                self._log("evaluate", f"Verdict: {verdict}")

                if verdict == "complete":
                    break

                # Gaps found — collect feedback for next iteration
                gaps = eval_result.get("gaps", [])
                suggested = eval_result.get("suggested_sql")

                # Try suggested SQL — if it works, use it as the current best
                if suggested and suggested.strip().upper() != sql.strip().upper():
                    suggested_result = self._try_sql(db_path, suggested)
                    if suggested_result and suggested_result["rows"]:
                        data_result = suggested_result
                        sql = suggested
                        self._log("suggested_sql_ok", "Suggested SQL returned data")
                        # Don't break — let next iteration evaluate if result is correct

                # Run info queries to gather more context about the data
                info_queries = eval_result.get("info_queries", [])
                info_parts: list[str] = []
                for iq in info_queries[:3]:
                    iq_result = self._try_sql(db_path, iq)
                    if iq_result and iq_result.get("rows"):
                        info_parts.append(
                            f"Query: {iq}\n"
                            f"Result: {self._format_data_as_table(iq_result)}"
                        )
                        self._log("info_query", f"{iq} → {len(iq_result['rows'])} rows")

                if info_parts:
                    extra_context = "\n\n".join(info_parts)

                if not gaps and not info_parts and not suggested:
                    break

                for g in gaps:
                    self._log("gap", g)

                gaps_text = "\n".join(f"- {g}" for g in gaps)
                # Include the failed SQL so the generator knows what NOT to repeat
                gaps_text += f"\n- FAILED SQL (do not repeat): {sql[:200]}"

            # Fallback if loop exhausted without good data
            if not data_result or not data_result.get("rows"):
                if self._time_remaining() > 60:
                    data_result = self._gather_relevant_data(db_path, kg, question)
                else:
                    data_result = {"columns": [], "rows": []}

            # Final LLM call: Format answer
            if self._time_remaining() < 30:
                # Emergency: no time for LLM — use raw SQL result directly
                answer = self._raw_result_to_answer(data_result)
            else:
                answer = self._call_answer(question, data_result, ctx.knowledge_text)

            # Cleanup
            if db_path.exists():
                try:
                    db_path.unlink()
                except OSError:
                    pass

            return self._build_result(answer, task)

        except Exception as e:
            logger.exception("Pipeline failed")
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
    # LLM Call 1: SQL Generation
    # ------------------------------------------------------------------

    def _call_sql(
        self, question: str, kg_context: str, sample_data: str,
        knowledge_text: str, gaps: str = "", extra_context: str = "",
        column_hints: str = "", grounding_context: str = "",
    ) -> str:
        prompt = _build_sql_prompt(
            question=question,
            kg_context=kg_context or "(no tables)",
            sample_data=sample_data,
            knowledge_text=knowledge_text,
            column_hints=column_hints,
            gaps=gaps,
            extra_context=extra_context,
            grounding_context=grounding_context,
        )

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if isinstance(parsed, dict):
            return parsed.get("sql", "")
        return ""

    # ------------------------------------------------------------------
    # LLM Call: Evaluate
    # ------------------------------------------------------------------

    def _call_evaluate(
        self,
        question: str,
        sql: str,
        sql_error: str,
        data_result: dict[str, Any],
        kg_context: str = "",
    ) -> dict[str, Any]:
        if data_result and data_result.get("rows"):
            data_text = self._format_data_as_table(data_result)
        else:
            data_text = "(empty — no rows returned)"

        prompt = _build_evaluate_prompt(
            question=question,
            sql=sql or "(none)",
            sql_error=sql_error,
            data_text=data_text,
            kg_context=kg_context,
        )

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if isinstance(parsed, dict):
            return parsed
        return {"verdict": "complete", "reasoning": "Could not parse evaluation.", "gaps": []}

    # ------------------------------------------------------------------
    # LLM Call 2: Answer Formatting
    # ------------------------------------------------------------------

    def _call_answer(
        self,
        question: str,
        data_result: dict[str, Any] | None,
        knowledge_text: str,
    ) -> dict[str, Any]:
        if data_result and data_result.get("rows"):
            data_text = self._format_data_as_table(data_result)
        elif data_result and data_result.get("_raw"):
            data_text = data_result["_raw"]
        else:
            data_text = "(no data found)"

        prompt = _build_answer_prompt(
            question=question,
            data_text=data_text,
            knowledge_text=knowledge_text,
        )

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # LLM Call: Semantic Grounding (pre-planning decomposition with validation)
    # ------------------------------------------------------------------

    def _call_semantic_grounding(
        self,
        question: str,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
    ) -> str:
        """Decompose the question into structured grounding, validate, iterate."""
        # Step 1: Generate initial grounding
        prompt = _build_semantic_prompt(
            question=question,
            kg_context=kg_context,
            sample_data=sample_data,
            knowledge_text=knowledge_text,
        )
        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        grounding = self._parse_json(raw)

        if not isinstance(grounding, dict) or not grounding:
            self._log("semantic_grounding", "(failed to parse)")
            return ""

        self._log("semantic_grounding_v1", json.dumps(grounding, default=str)[:300])

        # Step 2: Validate loop (max 2 iterations)
        for validate_iter in range(2):
            if self._time_remaining() < 150:
                break

            val_prompt = _build_grounding_validate_prompt(
                question=question,
                grounding=grounding,
                knowledge_text=knowledge_text,
                sample_data=sample_data,
            )
            val_messages = [ModelMessage(role="user", content=val_prompt)]
            val_raw = self._model_call_with_retry(val_messages)
            val_result = self._parse_json(val_raw)

            if not isinstance(val_result, dict):
                break

            verdict = val_result.get("verdict", "sufficient")
            if verdict == "sufficient":
                self._log("grounding_validated", f"OK after {validate_iter + 1} check(s)")
                break

            # Apply corrections
            corrections = val_result.get("corrections", {})
            issues = val_result.get("issues", [])
            self._log("grounding_issues", f"{issues}")

            if not corrections:
                break

            for field, corrected in corrections.items():
                grounding[field] = corrected
            self._log("grounding_corrected", f"Applied: {list(corrections.keys())}")

        formatted = _format_grounding_for_sql(grounding)
        self._log("semantic_grounding_final", formatted[:400] if formatted else "(empty)")
        return formatted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_sample_data(self, db_path: Path, kg: KnowledgeGraph) -> str:
        """Get sample rows + date ranges + value ranges for each table."""
        parts: list[str] = []
        conn = sqlite3.connect(str(db_path))
        for table in kg.tables:
            try:
                cursor = conn.execute(f'SELECT * FROM "{table.name}" LIMIT 3')
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                parts.append(f"TABLE {table.name} ({table.row_count} rows):")
                parts.append(f"  Columns: {columns}")
                for row in rows:
                    parts.append(f"  {list(row)}")
                # Detect date/time columns and show their range
                for col in columns:
                    col_lower = col.lower()
                    if any(kw in col_lower for kw in ("date", "time", "year", "month", "period")):
                        try:
                            rng = conn.execute(
                                f'SELECT MIN("{col}"), MAX("{col}") FROM "{table.name}"'
                            ).fetchone()
                            if rng and rng[0] is not None:
                                parts.append(f"  >> {col} range: {rng[0]} to {rng[1]}")
                        except Exception:
                            pass
                # Show key column lengths when they look like IDs (for JOIN alignment)
                for col in columns:
                    col_lower = col.lower()
                    if any(kw in col_lower for kw in ("id", "code", "key", "cds")):
                        try:
                            lens = conn.execute(
                                f'SELECT MIN(LENGTH("{col}")), MAX(LENGTH("{col}")) '
                                f'FROM "{table.name}" WHERE "{col}" IS NOT NULL'
                            ).fetchone()
                            if lens and lens[0] is not None and lens[0] != lens[1]:
                                parts.append(f"  >> {col} length varies: {lens[0]}-{lens[1]} chars")
                            elif lens and lens[0] is not None:
                                parts.append(f"  >> {col} length: {lens[0]} chars")
                        except Exception:
                            pass
            except Exception:
                continue
        conn.close()
        return "\n".join(parts)

    def _try_sql(self, db_path: Path, sql: str) -> dict[str, Any] | None:
        """Execute SQL and return results."""
        if not sql:
            return None
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
        """Parse JSON from LLM response, handling markdown fences and Qwen quirks."""
        if not raw:
            return {}
        raw = raw.strip()
        # Strip thinking tags (Qwen sometimes outputs <think>...</think>)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1).strip()
        for start, end in [("{", "}"), ("[", "]")]:
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
                                # Try fixing trailing commas
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

    def _build_column_hints(self, question: str, kg: KnowledgeGraph) -> str:
        """Map words from the question to actual column names in the schema."""
        q_words = set(re.findall(r"[a-z]{3,}", question.lower()))
        hints: list[str] = []
        for table in kg.tables:
            for col in table.columns:
                col_lower = col.name.lower()
                # Check if any question word matches (or is substring of) a column name
                for word in q_words:
                    if word == col_lower or (len(word) >= 4 and word in col_lower):
                        hints.append(f"  \"{word}\" → {table.name}.{col.name}")
                        break
        if hints:
            return "COLUMN HINTS (question words matching schema columns):\n" + "\n".join(hints)
        return ""

    def _raw_result_to_answer(self, data_result: dict[str, Any]) -> dict[str, Any]:
        """Convert raw SQL result to answer format without LLM call."""
        columns = data_result.get("columns", [])
        rows = data_result.get("rows", [])
        if columns and rows:
            return {"columns": columns, "rows": [[str(v) for v in row] for row in rows]}
        return {}

    def _time_remaining(self) -> float:
        """Seconds remaining in the time budget."""
        elapsed = time.monotonic() - self._start_time
        return max(0.0, TASK_TIME_BUDGET_SECONDS - elapsed)

    def _model_call_with_retry(self, messages: list[ModelMessage]) -> str:
        """Call model, return empty string on failure or if time budget is low."""
        remaining = self._time_remaining()
        if remaining < 30:
            self._log("time_budget", f"Skipping model call — only {remaining:.0f}s left")
            return ""
        try:
            return self.model.complete(messages)
        except RuntimeError as e:
            self._log("model_error", str(e))
            return ""

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

        str_rows = [[str(v) for v in row] for row in rows]

        return AgentRunResult(
            task_id=task.task_id,
            answer=AnswerTable(columns=columns, rows=str_rows),
            steps=step_records,
            failure_reason=None,
        )
