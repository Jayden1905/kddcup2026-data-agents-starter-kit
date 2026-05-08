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
from data_agent_baseline.pipeline.kg_builder import (
    KnowledgeGraph,
    build_kg_from_sqlite,
    format_kg_for_llm,
)
from data_agent_baseline.tools.knowledge_graph import consolidate_to_sqlite

logger = logging.getLogger(__name__)

CONSOLIDATED_DB_NAME = "_consolidated.db"


# ---------------------------------------------------------------------------
# Prompt builders — dynamic, only include sections that have content
# ---------------------------------------------------------------------------

SQL_RULES = [
    "PRIORITY: Your SQL must answer the user's EXACT question — re-read it before writing.",
    "Use only tables and columns shown in the schema.",
    "SELECT only columns that answer the question. Never SELECT *.",
    "Check SAMPLE DATA for date ranges and formats before writing WHERE clauses.",
    "If a table's date range doesn't cover the period, start from a DIFFERENT table that has the needed dates and JOIN back.",
    "The same column name in different tables may have DIFFERENT formats (e.g., 'YYYYMM' vs 'YYYY-MM-DD'). Always filter the table that actually contains the target value.",
    "JOIN through the full FK path. Never skip intermediate linking tables.",
    "\"X containing Y\" means filter X itself (WHERE X.prop = Y).",
    "Use LIKE '%keyword%' COLLATE NOCASE for text. Use CAST(x AS REAL) for division.",
    "If the question asks for names/descriptions, JOIN to get human-readable values instead of raw IDs.",
    "Only SELECT columns the question explicitly asks for. Do NOT add extra columns (dates, amounts, etc.) that the question didn't mention.",
    "NEVER use LIMIT unless the question explicitly asks for a specific count (e.g., 'top 3'). For superlatives (lowest/highest/most/least/best/worst), use a subquery: WHERE col = (SELECT MIN/MAX(col) ...) — there may be ties.",
    "When using MIN/MAX on text columns (times, dates), ALWAYS exclude empty strings: add WHERE col != '' AND col IS NOT NULL in the subquery.",
    "If a column is often NULL (shown as None/null in SAMPLE DATA), comparisons like col < X return nothing for NULL rows. Consider whether the question refers to a DIFFERENT column or table.",
    "If the question asks 'which/what X' (asking for names/identifiers), use SELECT DISTINCT to avoid duplicate rows.",
    "Escape apostrophes in strings: use '' (double single-quote) inside SQL string literals.",
    "For COUNT/SUM/aggregations: the column being aggregated must semantically match what the question asks about. Re-read the question to determine the correct column.",
    "When the question says 'per unit/per item/each/per person', compute a ratio (total ÷ count) — do NOT compare a total column directly.",
    "Do NOT use GROUP BY + aggregate (SUM/AVG) unless the question explicitly asks for totals or averages. 'lowest/highest X' means MIN/MAX of individual rows, not grouped sums.",
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

    rules = "\n".join(f"- {r}" for r in SQL_RULES)
    has_gaps = bool(gaps)
    if grounding_context:
        if has_gaps:
            rules += "\n- PREVIOUS ATTEMPT FAILED feedback takes PRIORITY over SEMANTIC GROUNDING. If the feedback contradicts the grounding (formula, filter values, columns), follow the feedback."
            rules += "\n- Use SEMANTIC GROUNDING as general context, but fix what the feedback says is wrong."
        else:
            if "FORMULA" in grounding_context:
                if "WRONG TABLE" in grounding_context or "Filter on" in grounding_context:
                    rules += "\n- Use the FORMULA as a starting point but CONSTRAINTS override it — if a constraint says to filter a different table, do so."
                else:
                    rules += "\n- Follow the FORMULA in SEMANTIC GROUNDING exactly."
            if "FILTER VALUES" in grounding_context:
                rules += "\n- ⚠️ MANDATORY: Your WHERE clause MUST use EXACTLY the values from FILTER VALUES above. Do NOT substitute other values."
        if "SELECT THESE COLUMNS" in grounding_context:
            rules += "\n- SELECT the columns listed in SELECT THESE COLUMNS."
    elif knowledge_text and "formula" in knowledge_text.lower():
        rules += "\n- If DOMAIN KNOWLEDGE defines a formula, follow it exactly."
    parts.append(f"\nRULES:\n{rules}")

    # Put mandatory filter constraint LAST so it's freshest in model's context (only if no gaps)
    if grounding_context and "FILTER VALUES" in grounding_context and not has_gaps:
        import re as _re
        fv_match = _re.search(r"FILTER VALUES:\n((?:  .+\n?)+)", grounding_context)
        if fv_match:
            parts.append(f"\n⚠️ MANDATORY WHERE CLAUSE (do NOT change these values):\n{fv_match.group(1).strip()}")

    parts.append('\nReturn ONLY a JSON object:\n{"thought": "reasoning", "sql": "SELECT ..."}')

    return "\n".join(parts)


def _build_evaluate_prompt(
    *,
    question: str,
    sql: str,
    sql_error: str,
    data_text: str,
    kg_context: str = "",
    knowledge_text: str = "",
    grounding_context: str = "",
) -> str:
    parts = ["Evaluate whether these query results answer the question."]

    parts.append(f"\nQUESTION: {question}")

    if knowledge_text:
        parts.append(f"\nDOMAIN KNOWLEDGE:\n{knowledge_text[:1500]}")

    if grounding_context:
        parts.append(f"\nVALIDATED PLAN:\n{grounding_context}")

    if kg_context:
        parts.append(f"\nSCHEMA:\n{kg_context}")

    parts.append(f"\nSQL: {sql or '(none)'}")
    if sql_error:
        parts.append(f"ERROR: {sql_error}")

    parts.append(f"\nRESULTS:\n{data_text}")

    parts.append("""
Return ONLY a JSON object:
{"verdict": "complete"/"incomplete", "reasoning": "why", "gaps": [], "info_queries": [], "suggested_sql": "..."}

- Re-read the QUESTION. Does the data ACTUALLY answer what was asked?
- TRUST the VALIDATED PLAN's FILTER VALUES — those literal values are verified.
- FIRST CHECK: Does the SQL use the EXACT literal values from FILTER VALUES? If not → "incomplete".
- The SQL may use different arithmetic/expressions than the FORMULA if it better matches the QUESTION's semantics. Do NOT reject a query just because its expression differs from FORMULA — only reject if FILTER VALUES are wrong.
- "complete" = SQL uses correct filter values AND data has rows AND answers the question.
- "incomplete" = SQL uses wrong filter values, or error, empty, or wrong columns.
- Do NOT question filter values that match the VALIDATED PLAN — those are correct.
- MULTIPLE ROWS ARE VALID: If the query returns multiple rows, that means there are ties or multiple matches — this is CORRECT. Do NOT mark as incomplete just because there are multiple results.
- suggested_sql must use the VALIDATED PLAN's filter values. Never repeat the same failing query.""")

    return "\n".join(parts)


DOMAIN_ANCHOR_PROMPT = """Given this question and domain knowledge, extract ONLY the definitions and rules that are directly relevant to answering the question.

QUESTION: {question}

DOMAIN KNOWLEDGE:
{knowledge_text}

Return ONLY a JSON object:
{{"anchors": ["exact quote of each relevant definition — include the exact numeric values/mappings"], "use_case_sql": "the complete SQL from a USE CASE that answers the same or very similar question, or null if none match"}}

RULES:
- For anchors: quote the EXACT definition including numeric mappings (e.g., "'severe' corresponds to value 2").
- CRITICAL: If ANY word from the question matches a column/field name defined in DOMAIN KNOWLEDGE, you MUST include that definition. For example, if the question mentions "type" and the knowledge defines a "type" field, include it.
- For use_case_sql: if the domain knowledge has a USE CASE whose question matches or is very similar to the user's question, copy its SQL EXACTLY. This SQL is the authoritative answer pattern.
- If a definition distinguishes between similar terms (e.g., "most severe = 1" vs "severe = 2"), quote BOTH so the distinction is clear.
- Be precise and complete — these anchors will be used as immutable ground truth.
""".strip()

SEMANTIC_GROUNDING_PROMPT = """Your ONLY goal is to answer the user's EXACT question — nothing more, nothing less.

QUESTION: {question}

DATABASE SCHEMA:
{kg_context}
{sample_section}
{anchor_section}
{previous_attempt}
Decompose the question into a structured plan. Return ONLY a JSON object:
{{
  "what_user_wants": "restate EXACTLY what output the user expects — only columns the question EXPLICITLY mentions",
  "formula": "the EXACT SQL expression to compute the answer — if GROUND TRUTH defines a metric formula, translate it literally to SQL without simplifying or removing any operations",
  "computation_steps": ["step1: find X", "step2: calculate Y from X"],
  "data_requirements": ["table.column needed for output", "table2.column2 for filter"],
  "reasoning": "brief HOW to get the answer — must trace back to what_user_wants",
  "domain_rules": ["constraints from DOMAIN KNOWLEDGE that affect the query"],
  "known_values": {{"table.column": ["filter values or expressions verified against SAMPLE DATA"]}},
  "join_paths": ["tableA.col -> tableB.col -> tableC.col"]
}}

RULES:
- Start by understanding what_user_wants — every other field must serve that goal.
- GROUND TRUTH section contains immutable facts extracted from domain knowledge. You MUST follow them exactly.
- For known_values: always include the TABLE name (e.g., "yearmonth.Date" not just "Date"). Only use values that exist within that table's SAMPLE DATA range.
- CRITICAL: Check SAMPLE DATA to decide WHICH TABLE to filter. The same column name in different tables may have different formats or data coverage. Always filter the table that actually contains the data you need.
- For formula: if GROUND TRUTH defines a metric formula (e.g., "Metric = X / N"), translate it literally to SQL. Keep ALL parts — do NOT remove operations even if you think the data makes them redundant. The aggregation function must match the question intent ("average" → AVG, "total" → SUM).
- "per unit/per item/each" in the question means a RATIO (total ÷ quantity). Check SAMPLE DATA to determine which column is a total vs a quantity, then use division in the formula.
- For join_paths: trace the FULL FK path shown in DATABASE SCHEMA. Never skip intermediate tables.
- For data_requirements: list ONLY columns needed for output + filters. Do NOT include extra columns.
- Do NOT invent output columns the question didn't ask for. If the question says "list all X", SELECT only the column that identifies X — do NOT add properties (like amount, date) unless the question explicitly asks for them.
- If GROUND TRUTH includes a USE CASE SQL, follow its structure but ensure the selected/aggregated column semantically matches what the question asks about.
""".strip()

GROUNDING_VALIDATE_PROMPT = """You are a strict validator. Check if this plan EXACTLY answers the user's question.

QUESTION: {question}
{anchor_section}
{sample_section}
PLAN:
{grounding_json}

VALIDATE EACH:
1. FILTER VALUES — CRITICAL:
   - GROUND TRUTH is the FINAL AUTHORITY. If the plan's filter values contradict GROUND TRUTH, fix them.
   - If GROUND TRUTH has a USE CASE SQL, the plan's known_values MUST match that SQL's WHERE clause.
   - If a filter value comes directly from the QUESTION (e.g., a name, title, or specific term the user mentioned), keep it even if it's not in SAMPLE DATA. SAMPLE DATA is only a small subset.
   - Only reject values that contradict GROUND TRUTH definitions, NOT values that are simply absent from the sample.
2. OUTPUT GRAIN: What shape does the user expect?
   - "how many" → single COUNT value
   - "list all" → multiple rows
   - "what is the X" → single value
   - Do NOT add extra columns the question didn't ask for.
3. FORMULA: Does it compute what the USER asked for?
   - The aggregation function is determined by the QUESTION, not GROUND TRUTH: "average" → AVG(), "total" → SUM(), "percentage" → *100.
   - GROUND TRUTH defines the calculation structure (what to divide by, what columns to use), but the QUESTION determines the aggregation type.
   - Do NOT change AVG to SUM or SUM to AVG. Only check that the arithmetic and columns are correct.
4. JOIN PATHS: Complete? No missing intermediate tables?

Return ONLY a JSON object:
{{"verdict": "correct"/"needs_fix", "fixed_known_values": {{"column_name": ["corrected_values"]}}, "fixed_data_requirements": ["table.column", ...], "fixed_join_paths": ["tableA.col -> tableB.col"], "reasoning": "one sentence explaining what was wrong"}}

- "correct" = ALL filter values, formula, columns, and joins match GROUND TRUTH and the question's intent.
- "needs_fix" = ANY mismatch with GROUND TRUTH found. You MUST provide the corrected values.
- For fixed_known_values: ONLY suggest values that actually exist in SAMPLE DATA. If the filter requires a range or pattern (e.g., all dates in a month), put a representative value from SAMPLE DATA — the SQL will use LIKE or BETWEEN. Do NOT invent values that aren't in the sample.
""".strip()


def _build_semantic_prompt(
    *,
    question: str,
    kg_context: str,
    sample_data: str = "",
    anchor_text: str = "",
    previous_attempt: str = "",
) -> str:
    sample_section = f"\nSAMPLE DATA:\n{sample_data[:3000]}" if sample_data else ""
    anchor_section = f"\nGROUND TRUTH (immutable — you MUST follow these):\n{anchor_text}" if anchor_text else ""
    prev_section = f"\nPREVIOUS ATTEMPT (fix the issues below):\n{previous_attempt}" if previous_attempt else ""
    return SEMANTIC_GROUNDING_PROMPT.format(
        question=question,
        kg_context=kg_context,
        sample_section=sample_section,
        anchor_section=anchor_section,
        previous_attempt=prev_section,
    )


def _build_grounding_validate_prompt(
    *,
    question: str,
    grounding: dict[str, Any],
    anchor_text: str = "",
    sample_data: str = "",
) -> str:
    anchor_section = f"\nGROUND TRUTH (immutable — you MUST follow these):\n{anchor_text}" if anchor_text else ""
    sample_section = f"\nSAMPLE DATA:\n{sample_data[:2000]}" if sample_data else ""
    return GROUNDING_VALIDATE_PROMPT.format(
        question=question,
        grounding_json=json.dumps(grounding, indent=2)[:3000],
        anchor_section=anchor_section,
        sample_section=sample_section,
    )


def _format_grounding_for_sql(grounding: dict[str, Any]) -> str:
    """Format semantic grounding output as structured context for SQL generation."""
    parts: list[str] = []

    what_user_wants = grounding.get("what_user_wants", "")
    if what_user_wants:
        parts.append(f"USER WANTS: {what_user_wants}")

    formula = grounding.get("formula", "")
    if formula and formula != "direct_lookup":
        parts.append(f"FORMULA: {formula}")

    join_paths = grounding.get("join_paths", [])
    if join_paths:
        parts.append("JOIN PATHS:\n" + "\n".join(f"  {jp}" for jp in join_paths))

    known_values = grounding.get("known_values", {})

    # Rebuild steps to reflect validated known_values (avoid stale filter references)
    steps = grounding.get("computation_steps", [])
    if steps:
        rebuilt_steps = []
        for s in steps:
            # Replace any stale filter value references with the validated ones
            rebuilt = s
            for col, vals in known_values.items():
                if col.lower() in s.lower() and vals:
                    correct_val = ", ".join(str(v) for v in vals)
                    import re as _re
                    rebuilt = _re.sub(
                        rf"(?i){col}\s*=\s*\S+",
                        f"{col} = {correct_val}",
                        rebuilt,
                    )
            rebuilt_steps.append(rebuilt)
        parts.append("STEPS:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(rebuilt_steps)))

    known_values = grounding.get("known_values", {})
    if known_values:
        kv_lines = []
        for k, vs in known_values.items():
            if not vs:
                continue
            kv_lines.append(f"  {k}: {', '.join(str(v) for v in vs)}")
        if kv_lines:
            parts.append("FILTER VALUES:\n" + "\n".join(kv_lines))

    data_reqs = grounding.get("data_requirements", [])
    output_cols = [r for r in data_reqs if "output" in r.lower() or "select" in r.lower()]
    if not output_cols:
        output_cols = [r for r in data_reqs if "." in r]
    if output_cols:
        parts.append("SELECT THESE COLUMNS:\n" + "\n".join(f"  {c}" for c in output_cols))

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
    grounding_context: str = "",
) -> str:
    parts = [f"Format this SQL output into exactly what the user asked for.\n\nQUESTION: {question}"]

    if grounding_context:
        parts.append(f"\nPLAN:\n{grounding_context}")

    parts.append(f"\nSQL OUTPUT:\n{data_text}")

    if knowledge_text:
        parts.append(f"\nDOMAIN KNOWLEDGE:\n{knowledge_text[:1500]}")

    parts.append("""
Return ONLY a JSON object:
{"columns": ["col1", "col2"], "rows": [["value1", "value2"], ...]}

RULES:
- Return EVERY row from SQL OUTPUT — never drop or truncate rows.
- Drop columns that are NOT needed to answer the question (e.g., intermediate IDs used only for joining).
- NEVER merge multiple columns into one (e.g., don't combine first_name + last_name into full_name).
- NEVER split one column into multiple.
- NEVER rename columns — use the exact SQL column names.
- Do NOT transform values — keep them exactly as in SQL OUTPUT.
- Do NOT add rows that aren't in SQL OUTPUT.""")

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
        self._log_file = context_dir / "_agent.log"
        self._log_file.write_text(f"=== {task.task_id} ===\nQ: {question}\n\n")

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

            # Track which tables came from structured data (CSV/JSON)
            structured_tables: list[str] = []
            if db_path.exists():
                try:
                    _conn = sqlite3.connect(str(db_path))
                    structured_tables = [
                        r[0] for r in _conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall() if not r[0].startswith("_")
                    ]
                    _conn.close()
                except Exception:
                    pass

            # Step 3: Doc extraction (LLM-based, batched by section)
            if ctx.doc_sources:
                doc_paths = [doc.path for doc in ctx.doc_sources]
                from data_agent_baseline.pipeline.llm_extractor import (
                    hybrid_extract_docs,
                )
                hybrid_extract_docs(
                    doc_paths=doc_paths,
                    db_path=db_path,
                    model=self.model,
                    knowledge_text=ctx.knowledge_text,
                    log_fn=self._log,
                    structured_tables=structured_tables,
                )

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
            grounding_context = self._call_semantic_grounding(
                question, kg_context, sample_data, ctx.knowledge_text,
                db_path=db_path,
            )

            # ----------------------------------------------------------
            # Closed loop: PLAN → EXECUTE → EVALUATE (max 4 iterations)
            # ----------------------------------------------------------
            max_iterations = 4
            data_result = None
            sql = ""
            gaps_text = ""
            extra_context = ""  # info from exploratory queries
            all_gaps: set[str] = set()  # dedup gaps across iterations
            failed_sqls: list[str] = []

            for iteration in range(1, max_iterations + 1):
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

                # Detect duplicate SQL — if model generated the same query, force break
                sql_normalized = " ".join(sql.split()).strip().upper()
                if sql_normalized in {" ".join(s.split()).strip().upper() for s in failed_sqls}:
                    self._log("evaluate", "Verdict: duplicate SQL — stopping iterations")
                    break

                # Force-incomplete on empty results (skip LLM eval)
                if not data_result.get("rows"):
                    if sql_error:
                        self._log("evaluate", f"Verdict: incomplete (SQL error: {sql_error})")
                        error_hint = self._diagnose_sql_error(db_path, sql, sql_error)
                        all_gaps.add(f"SQL ERROR: {sql_error}. {error_hint}")
                    else:
                        self._log("evaluate", "Verdict: incomplete (empty result)")
                        diag = self._diagnose_empty_result(db_path, sql)
                        if diag:
                            all_gaps.add(diag)
                        else:
                            all_gaps.add(
                                "Query returned 0 rows — check table names, "
                                "join columns, and filter values against SAMPLE DATA"
                            )
                    failed_sqls.append(sql)
                    gaps_text = "\n".join(f"- {g}" for g in all_gaps)
                    gaps_text += "\n".join(
                        f"\n- FAILED SQL (do not repeat): {s}" for s in failed_sqls
                    )
                    continue

                # Skip evaluate on last iteration or low time — just use what we have
                if iteration == max_iterations:
                    break

                # EVALUATE: Check if result answers the question
                eval_result = self._call_evaluate(
                    question, sql, sql_error, data_result, kg_context,
                    knowledge_text=ctx.knowledge_text,
                    grounding_context=grounding_context,
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

                # Deduplicate gaps
                new_gaps = [g for g in gaps if g not in all_gaps]
                all_gaps.update(new_gaps)
                for g in new_gaps:
                    self._log("gap", g)

                failed_sqls.append(sql[:200])
                gaps_text = "\n".join(f"- {g}" for g in all_gaps)
                gaps_text += "\n".join(
                    f"\n- FAILED SQL (do not repeat): {s}" for s in failed_sqls
                )

            # Deduplicate rows if the question asks for unique items
            if data_result and data_result.get("rows"):
                rows = data_result["rows"]
                unique_rows = []
                seen = set()
                for row in rows:
                    key = tuple(str(v) for v in row)
                    if key not in seen:
                        seen.add(key)
                        unique_rows.append(row)
                if len(unique_rows) < len(rows):
                    self._log("dedup", f"Removed {len(rows) - len(unique_rows)} duplicate rows")
                    data_result["rows"] = unique_rows

            # Fallback if loop exhausted without good data
            if not data_result or not data_result.get("rows"):
                data_result = self._gather_relevant_data(db_path, kg, question)

            # Format answer via schema-based synthesizer
            raw_row_count = len(data_result.get("rows", [])) if data_result else 0
            self._log("pre_answer", f"cols={data_result.get('columns') if data_result else None}, rows={raw_row_count}")
            if data_result and data_result.get("rows"):
                answer = self._call_answer_with_schema(
                    question, data_result, ctx.knowledge_text,
                    grounding_context=grounding_context,
                )
                if not answer or not answer.get("rows"):
                    self._log("answer_fallback", "Synthesizer failed — using raw SQL result")
                    answer = self._raw_result_to_answer(data_result)
            else:
                answer = self._raw_result_to_answer(data_result)

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
        knowledge_text: str = "",
        grounding_context: str = "",
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
            knowledge_text=knowledge_text,
            grounding_context=grounding_context,
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
        grounding_context: str = "",
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
            grounding_context=grounding_context,
        )

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        self._log("answer_raw", raw if raw else "(empty)")
        parsed = self._parse_json(raw)
        self._log("answer_parsed", json.dumps(parsed, default=str) if parsed else "(empty)")
        return parsed

    def _call_answer_with_schema(
        self,
        question: str,
        data_result: dict[str, Any],
        knowledge_text: str,
        grounding_context: str = "",
    ) -> dict[str, Any]:
        """Two-phase answer: LLM picks columns from schema, code applies to full data."""
        columns = data_result.get("columns", [])
        rows = data_result.get("rows", [])

        if not columns or not rows:
            return {}

        # Single column — no need to ask LLM
        if len(columns) == 1:
            return self._raw_result_to_answer(data_result)

        # Extract user intent from grounding context
        user_wants = ""
        if grounding_context:
            match = re.search(r"USER WANTS:\s*(.+)", grounding_context)
            if match:
                user_wants = match.group(1).strip()

        col_list = "\n".join(f"  {i}: {c}" for i, c in enumerate(columns))
        prompt = f"""The user asked a question. The SQL returned these columns. Which columns should appear in the final output?

QUESTION: {question}
USER INTENT: {user_wants or question}

SQL RESULT COLUMNS:
{col_list}

Return ONLY: {{"keep_columns": [0, 2]}}

RULES:
- The output must contain ONLY the information the user EXPLICITLY asked for — nothing extra.
- "list all X" or "list the X" = ONLY the identifier/ID column of X. Do NOT add properties (amount, date, name, etc.) unless the question EXPLICITLY mentions them.
- "X and Y" = both X and Y columns, but ONLY those two.
- Remove columns that were only used for filtering (WHERE) or joining — they are not part of the answer.
- Remove columns whose values are constant (same for every row) — those are filter echoes.
- When in doubt, keep FEWER columns. Only include a column if the question directly asks for that information.
- NEVER merge columns. Just pick indices to keep."""
        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if not isinstance(parsed, dict) or "keep_columns" not in parsed:
            self._log("answer_schema", "(failed to parse, using raw)")
            return self._raw_result_to_answer(data_result)

        keep_indices = parsed.get("keep_columns", [])

        # Validate indices
        if not keep_indices or not all(isinstance(i, int) and 0 <= i < len(columns) for i in keep_indices):
            self._log("answer_schema", f"Invalid indices {keep_indices}, using raw")
            return self._raw_result_to_answer(data_result)

        # Use SQL column names directly
        output_names = [columns[i] for i in keep_indices]

        # Apply column selection to ALL rows (no LLM, no truncation)
        filtered_rows = [[str(row[i]) for i in keep_indices] for row in rows]

        self._log("answer_schema", f"Kept columns {keep_indices} → {output_names} ({len(filtered_rows)} rows)")
        return {"columns": output_names, "rows": filtered_rows}

    # ------------------------------------------------------------------
    # LLM Call: Semantic Grounding (pre-planning decomposition with validation)
    # ------------------------------------------------------------------

    def _extract_domain_anchors(self, question: str, knowledge_text: str) -> str:
        """Extract relevant domain definitions as immutable ground truth.

        Uses LLM to identify which definitions and use cases are relevant,
        then formats them as clear, unambiguous anchors.
        """
        if not knowledge_text:
            return ""
        prompt = DOMAIN_ANCHOR_PROMPT.format(
            question=question,
            knowledge_text=knowledge_text[:4000],
        )
        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if not isinstance(parsed, dict):
            return knowledge_text[:2000]

        parts: list[str] = []
        anchors = parsed.get("anchors", [])
        for a in anchors:
            parts.append(f"- {a}")
        use_case_sql = parsed.get("use_case_sql")
        if use_case_sql:
            parts.append(f"\nMATCHING USE CASE SQL (follow this exactly):\n  {use_case_sql}")

        if not parts:
            return knowledge_text[:2000]

        # Deterministic: find field definitions in knowledge_text that match question words
        q_words = set(re.findall(r'\b[a-z_]{3,}\b', question.lower()))
        # Match lines like "- **field_name**: description" or "- field_name: description"
        field_defs = re.findall(
            r'-\s+\*{0,2}(\w+)\*{0,2}\s*:\s*(.+)',
            knowledge_text,
        )
        anchor_lower = "\n".join(parts).lower()
        for field_name, definition in field_defs:
            if field_name.lower() in q_words and field_name.lower() not in anchor_lower:
                parts.append(f"- {field_name}: {definition.strip()}")

        anchor_text = "\n".join(parts)

        # Translate formula anchors to SQL-ready form based on question intent
        # Replace "[Total ... X] / N" with "AVG(X) / N" when question asks for average
        q_lower = question.lower()
        if "average" in q_lower or "avg" in q_lower:
            def _rewrite_formula(m: re.Match) -> str:
                # Extract the last word before / as the column name
                before_div = m.group(1).strip().rstrip("]")
                col = before_div.split()[-1]
                n = m.group(2)
                return f"AVG({col}) / {n}"
            anchor_text = re.sub(
                r'\[?Total\s+([\w\s]+?)\]?\s*/\s*(\d+)',
                _rewrite_formula,
                anchor_text,
            )

        self._log("domain_anchors", anchor_text)
        return anchor_text

    def _call_semantic_grounding(
        self,
        question: str,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
        db_path: Path | None = None,
    ) -> str:
        """Closed loop: Ground → Validate → Verify data → Re-ground (max 3 iterations)."""
        max_grounding_iters = 3
        grounding: dict[str, Any] = {}
        previous_attempt = ""

        # Extract domain anchors as immutable ground truth
        anchor_text = self._extract_domain_anchors(question, knowledge_text)

        # Track previous fixes to detect oscillation
        prev_fixed_kv: dict[str, list] = {}
        first_grounding: dict[str, Any] = {}

        for g_iter in range(1, max_grounding_iters + 1):
            self._log("grounding_iter", f"--- Grounding iteration {g_iter} ---")

            # GROUND: Decompose question (pass previous feedback on re-ground)
            prompt = _build_semantic_prompt(
                question=question,
                kg_context=kg_context,
                sample_data=sample_data,
                anchor_text=anchor_text,
                previous_attempt=previous_attempt,
            )
            messages = [ModelMessage(role="user", content=prompt)]
            raw = self._model_call_with_retry(messages)
            grounding = self._parse_json(raw)

            if not isinstance(grounding, dict) or not grounding:
                # Retry once
                self._log("semantic_grounding", "(failed to parse, retrying)")
                raw = self._model_call_with_retry(messages)
                grounding = self._parse_json(raw)

            if not isinstance(grounding, dict) or not grounding:
                self._log("semantic_grounding", "(failed to parse after retry)")
                if first_grounding:
                    grounding = first_grounding
                    break
                return ""

            if not first_grounding:
                first_grounding = json.loads(json.dumps(grounding))

            # Verify filter values against actual DB before validation
            if db_path and grounding.get("known_values"):
                grounding = self._validate_filter_values(db_path, grounding)

            self._log(f"grounding_v{g_iter}",
                      json.dumps(grounding, default=str))

            # VALIDATE: Check if grounding is correct and complete
            val_prompt = _build_grounding_validate_prompt(
                question=question,
                grounding=grounding,
                anchor_text=anchor_text,
                sample_data=sample_data,
            )
            val_messages = [ModelMessage(role="user", content=val_prompt)]
            val_raw = self._model_call_with_retry(val_messages)
            val_result = self._parse_json(val_raw)

            if not isinstance(val_result, dict):
                break

            verdict = val_result.get("verdict", "correct")

            if verdict == "correct":
                self._log("grounding_validated", "OK")
                break

            # Build feedback for next iteration
            reasoning = val_result.get("reasoning", "")
            self._log("grounding_fix", reasoning)

            fixed_kv = val_result.get("fixed_known_values", {})
            fixed_dr = val_result.get("fixed_data_requirements", [])
            fixed_jp = val_result.get("fixed_join_paths", [])

            # Detect oscillation: if validator reverses a previous fix, stop
            if fixed_kv and prev_fixed_kv:
                oscillating = False
                for col, vals in fixed_kv.items():
                    if col in prev_fixed_kv and set(str(v) for v in vals) != set(str(v) for v in prev_fixed_kv[col]):
                        oscillating = True
                        break
                if oscillating:
                    self._log("grounding_oscillation", "Validator reversed previous fix — using current grounding")
                    break

            rejected_fixes: list[str] = []

            if fixed_kv:
                # Re-verify validator's suggested values against actual DB
                if db_path:
                    verified_kv, rejections = self._verify_fixed_values(
                        db_path, fixed_kv, grounding.get("known_values", {}))
                    rejected_fixes.extend(rejections)
                    fixed_kv = verified_kv
                if fixed_kv:
                    kv = grounding.get("known_values", {})
                    kv.update(fixed_kv)
                    grounding["known_values"] = kv
                    prev_fixed_kv = fixed_kv
                    self._log("grounding_fix_kv", str(fixed_kv))
            if fixed_dr:
                grounding["data_requirements"] = fixed_dr
                self._log("grounding_fix_dr", str(fixed_dr))
            if fixed_jp:
                grounding["join_paths"] = fixed_jp
                self._log("grounding_fix_jp", str(fixed_jp))

            # If validator didn't suggest any usable fixes, no point looping
            if not fixed_kv and not fixed_dr and not fixed_jp:
                if not rejected_fixes:
                    break
                # Validator tried fixes but they were all invalid — feed back what
                # actually exists so the next grounding attempt can adapt

            # Pass feedback to next generation so it doesn't repeat the mistake
            mismatch_rules = [r for r in grounding.get("domain_rules", []) if "exists in" in r and "NOT in" in r]
            feedback_parts: list[str] = []
            feedback_parts.append(f"Your previous plan had this error: {reasoning}")
            feedback_parts.append(f"Corrected values: {json.dumps(grounding.get('known_values', {}))}")
            feedback_parts.append("You MUST use these corrected values and the correct tables in known_values.")
            if rejected_fixes:
                feedback_parts.append("\nREJECTED FIXES (these values do NOT exist in the DB — do NOT use them):")
                for rf in rejected_fixes:
                    feedback_parts.append(f"- {rf}")
                feedback_parts.append("Use LIKE patterns or range filters instead of exact match for date/time columns.")
            if mismatch_rules:
                feedback_parts.append("\nDATA VERIFIED FACTS:")
                for r in mismatch_rules:
                    feedback_parts.append(f"- {r}")
            previous_attempt = "\n".join(feedback_parts)

        formatted = _format_grounding_for_sql(grounding)
        self._log("semantic_grounding_final", formatted if formatted else "(empty)")
        return formatted

    def _validate_filter_values(
        self, db_path: Path, grounding: dict[str, Any]
    ) -> dict[str, Any]:
        """Check that filter values in grounding actually exist in the DB.

        If a value doesn't exist, try to find the closest match.
        Also detects table mismatches: value exists in table A but formula queries table B.
        """
        known_values = grounding.get("known_values", {})
        if not known_values:
            return grounding

        formula = grounding.get("formula", "")

        conn = sqlite3.connect(str(db_path))
        try:
            tables_cols = {}
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall():
                tname = row[0]
                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                tables_cols[tname] = cols

            for col_name, values in list(known_values.items()):
                if not values:
                    continue
                # Handle table.column format
                if "." in col_name:
                    hint_table, bare_col = col_name.split(".", 1)
                else:
                    hint_table, bare_col = None, col_name

                # Find ALL tables that have this column
                candidate_tables: list[tuple[str, str]] = []
                # Prioritize the hinted table
                if hint_table:
                    for tname, cols in tables_cols.items():
                        if tname.lower() == hint_table.lower():
                            col_match = next(
                                (c for c in cols if c.lower() == bare_col.lower()), None
                            )
                            if col_match:
                                candidate_tables.append((tname, col_match))
                            break
                # Also check all other tables
                for tname, cols in tables_cols.items():
                    if any(tname == ct[0] for ct in candidate_tables):
                        continue
                    col_match = next(
                        (c for c in cols if c.lower() == bare_col.lower()), None
                    )
                    if col_match:
                        candidate_tables.append((tname, col_match))

                if not candidate_tables:
                    continue

                # Check each value against all candidate tables
                for val in values:
                    found_in: list[tuple[str, int]] = []
                    not_found_in: list[str] = []
                    for tname, actual_col in candidate_tables:
                        try:
                            result = conn.execute(
                                f'SELECT COUNT(*) FROM "{tname}" '
                                f'WHERE "{actual_col}" = ?', (val,)
                            ).fetchone()
                            if result and result[0] > 0:
                                found_in.append((tname, result[0]))
                            else:
                                not_found_in.append(tname)
                        except Exception:
                            pass

                    if found_in:
                        self._log("filter_verified",
                                  f"{col_name}='{val}' exists in {found_in[0][0]} ({found_in[0][1]} rows)")
                        # Check for table mismatch: value exists in one table but
                        # formula references a different table
                        if not_found_in and formula:
                            formula_tables = set()
                            for nf_table in not_found_in:
                                if nf_table.lower() in formula.lower():
                                    formula_tables.add(nf_table)
                            if formula_tables:
                                correct_table = found_in[0][0]
                                wrong_tables = formula_tables
                                rule = (
                                    f"{bare_col}='{val}' exists in {correct_table}, "
                                    f"NOT in {', '.join(wrong_tables)}. "
                                    f"Filter on {correct_table}.{bare_col} and JOIN to get needed data."
                                )
                                domain_rules = grounding.setdefault("domain_rules", [])
                                domain_rules.append(rule)
                                self._log("filter_table_mismatch", rule)
                                # Rewrite known_values to point to the correct table
                                correct_key = f"{correct_table}.{bare_col}"
                                if col_name != correct_key:
                                    known_values[correct_key] = values
                                    del known_values[col_name]
                                    self._log("filter_rewrite",
                                              f"Moved filter from {col_name} to {correct_key}")
                        continue

                    # Value not found in any table — try fuzzy matching on first candidate
                    target_table, actual_col = candidate_tables[0]
                    try:
                        like_result = conn.execute(
                            f'SELECT DISTINCT "{actual_col}" FROM "{target_table}" '
                            f'WHERE "{actual_col}" LIKE ? LIMIT 5',
                            (f'%{val}%',)
                        ).fetchall()
                        if like_result:
                            known_values[col_name] = [r[0] for r in like_result]
                            self._log("filter_fix",
                                      f"{col_name}: '{val}' not found, "
                                      f"using {known_values[col_name]}")
                            break
                    except Exception:
                        pass
        finally:
            conn.close()

        grounding["known_values"] = known_values
        return grounding

    def _verify_fixed_values(
        self, db_path: Path, fixed_kv: dict[str, list], original_kv: dict[str, list]
    ) -> tuple[dict[str, list], list[str]]:
        """Re-verify validator-suggested filter values against the DB.

        Returns (verified_values, rejection_details).
        Rejects fixes where the suggested value doesn't exist in the target column.
        Falls back to the original value when the fix is invalid.
        """
        conn = sqlite3.connect(str(db_path))
        verified: dict[str, list] = {}
        rejections: list[str] = []
        try:
            tables_cols: dict[str, list[str]] = {}
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall():
                tname = row[0]
                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                tables_cols[tname] = cols

            for col_name, values in fixed_kv.items():
                if not values:
                    continue
                bare_col = col_name.split(".")[-1] if "." in col_name else col_name

                # Find all tables containing this column
                candidate_tables: list[tuple[str, str]] = []
                if "." in col_name:
                    hint_table = col_name.split(".", 1)[0]
                    for tname, cols in tables_cols.items():
                        if tname.lower() == hint_table.lower():
                            col_match = next(
                                (c for c in cols if c.lower() == bare_col.lower()), None
                            )
                            if col_match:
                                candidate_tables.append((tname, col_match))
                            break
                if not candidate_tables:
                    for tname, cols in tables_cols.items():
                        col_match = next(
                            (c for c in cols if c.lower() == bare_col.lower()), None
                        )
                        if col_match:
                            candidate_tables.append((tname, col_match))

                if not candidate_tables:
                    verified[col_name] = values
                    continue

                # Check if value exists in ANY table with this column
                valid_values = []
                for val in values:
                    found_anywhere = False
                    for tname, actual_col in candidate_tables:
                        try:
                            count = conn.execute(
                                f'SELECT COUNT(*) FROM "{tname}" WHERE "{actual_col}" = ?',
                                (val,)
                            ).fetchone()[0]
                            if count > 0:
                                found_anywhere = True
                                self._log("fix_verified", f"{col_name}='{val}' OK in {tname} ({count} rows)")
                                break
                        except Exception:
                            pass
                    if found_anywhere:
                        valid_values.append(val)
                    else:
                        # Gather actual sample values to show what DOES exist
                        sample_vals = []
                        for tname, actual_col in candidate_tables:
                            try:
                                rows = conn.execute(
                                    f'SELECT DISTINCT "{actual_col}" FROM "{tname}" '
                                    f'WHERE "{actual_col}" IS NOT NULL LIMIT 5'
                                ).fetchall()
                                sample_vals = [r[0] for r in rows]
                            except Exception:
                                pass
                            if sample_vals:
                                break
                        checked = [f"{t}.{c}" for t, c in candidate_tables]
                        rejection_msg = (
                            f"{col_name}='{val}' does NOT exist. "
                            f"Actual sample values in {checked[0] if checked else col_name}: {sample_vals}"
                        )
                        rejections.append(rejection_msg)
                        self._log("fix_rejected", rejection_msg)

                if valid_values:
                    verified[col_name] = valid_values
                else:
                    # Only revert if original value is verified to exist
                    bare = col_name.split(".")[-1] if "." in col_name else col_name
                    orig_val = original_kv.get(col_name) or original_kv.get(bare)
                    if orig_val:
                        orig_valid = []
                        for ov in orig_val:
                            for tname, actual_col in candidate_tables:
                                try:
                                    count = conn.execute(
                                        f'SELECT COUNT(*) FROM "{tname}" WHERE "{actual_col}" = ?',
                                        (ov,)
                                    ).fetchone()[0]
                                    if count > 0:
                                        orig_valid.append(ov)
                                        break
                                except Exception:
                                    pass
                        if orig_valid:
                            self._log("fix_reverted", f"Keeping original {col_name}={orig_valid}")
                            verified[col_name] = orig_valid
                        else:
                            self._log("fix_dropped", f"Both fix and original invalid for {col_name}")
        finally:
            conn.close()
        return verified, rejections

    def _diagnose_empty_result(self, db_path: Path, sql: str) -> str:
        """When SQL returns 0 rows, gather DB evidence about what went wrong."""
        if not sql or not db_path.exists():
            return ""

        conn = sqlite3.connect(str(db_path))
        diagnostics: list[str] = []
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            for tname in tables:
                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
                for col in cols:
                    if col.lower() in sql.lower():
                        try:
                            sample = conn.execute(
                                f'SELECT DISTINCT "{col}" FROM "{tname}" '
                                f'WHERE "{col}" IS NOT NULL AND "{col}" != \'\' LIMIT 5'
                            ).fetchall()
                            vals = [r[0] for r in sample]
                            rng = conn.execute(
                                f'SELECT MIN("{col}"), MAX("{col}") FROM "{tname}"'
                            ).fetchone()
                            diagnostics.append(
                                f"{tname}.{col} ({row_count} rows): "
                                f"sample={vals}, range=[{rng[0]}..{rng[1]}]"
                            )
                        except Exception:
                            pass
        finally:
            conn.close()

        if diagnostics:
            return "ACTUAL DATA IN DB:\n" + "\n".join(diagnostics)
        return ""

    def _diagnose_sql_error(self, db_path: Path, sql: str, error: str) -> str:
        """When SQL has a column/table error, show what actually exists."""
        if not db_path.exists():
            return ""

        conn = sqlite3.connect(str(db_path))
        hints: list[str] = []
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            if "no such column" in error:
                # Extract the bad column reference from error
                # e.g. "no such column: e.event_name" or "no such column: format"
                bad_col = error.split("no such column:")[-1].strip()
                # If it has a table alias prefix, extract just the column name
                bare_col = bad_col.split(".")[-1] if "." in bad_col else bad_col

                # Find tables referenced in the SQL and show their actual columns
                for tname in tables:
                    if tname.lower() in sql.lower():
                        cols = [c[1] for c in conn.execute(
                            f'PRAGMA table_info("{tname}")'
                        ).fetchall()]
                        hints.append(f"Table '{tname}' has columns: {cols}")

            elif "no such table" in error:
                hints.append(f"Available tables: {tables}")

        finally:
            conn.close()

        return "\n".join(hints)

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
        if not db_path.exists():
            self._log("sql_error", f"DB missing: {db_path}")
            return None
        try:
            conn = sqlite3.connect(str(db_path))
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if not tables:
                self._log("sql_error", f"DB empty (0 tables): {db_path}")
                conn.close()
                return None
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()
            return {"columns": columns, "rows": [list(r) for r in rows]}
        except Exception as e:
            self._log("sql_error", f"SQL failed (tables={tables}): {e}")
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

        def _try_parse(text: str) -> Any:
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                pass
            # Fix trailing commas
            fixed = re.sub(r",\s*([}\]])", r"\1", text)
            try:
                return json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                pass
            # Fix single quotes → double quotes (but not apostrophes in words)
            fixed2 = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '"', fixed)
            try:
                return json.loads(fixed2)
            except (json.JSONDecodeError, ValueError):
                pass
            # Fix unquoted keys: key: → "key":
            fixed3 = re.sub(r'(?m)^\s*([a-zA-Z_]\w*)\s*:', r'"\1":', fixed)
            try:
                return json.loads(fixed3)
            except (json.JSONDecodeError, ValueError):
                pass
            return None

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
                            result = _try_parse(raw[idx:i + 1])
                            if result is not None:
                                return result
                            break
                break

        result = _try_parse(raw)
        if result is not None:
            return result
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

    def _model_call_with_retry(self, messages: list[ModelMessage]) -> str:
        """Call model. Raises on API failure to stop the task."""
        return self.model.complete(messages)

    def _log(self, action: str, detail: str) -> None:
        """Log a pipeline step."""
        step = {"action": action, "detail": detail}
        self.steps.append(step)
        if self.log_callback:
            self.log_callback(step)
        elapsed = time.monotonic() - self._start_time
        print(f"[{elapsed:6.1f}s] [{action}] {detail}", flush=True)
        if hasattr(self, "_log_file") and self._log_file:
            with open(self._log_file, "a") as f:
                f.write(f"[{elapsed:6.1f}s] [{action}] {detail}\n")

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
