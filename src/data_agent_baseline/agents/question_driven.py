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
    # Scope/population rules
    "POPULATION vs METRIC: 'In X, what is Y?' or 'Among X, what is Y?' → X defines the WHERE filter (population/denominator), Y is what you compute ON that filtered set. Example: 'In employees with salary > 50000, what % are managers?' → WHERE salary > 50000, then compute COUNT(managers)/COUNT(*)*100.",
    "RATIO LANGUAGE: 'How many times X more than Y' or 'how many times was X more than Y' = X/Y (division producing a ratio). It does NOT mean X-Y (subtraction) or COUNT.",
    # Aggregation grain
    "AGGREGATION GRAIN: When computing AVG/SUM of an entity's own attribute (e.g., user age, user upvotes), query that entity table DIRECTLY with a subquery filter. Do NOT join to detail tables — joining users to posts duplicates user rows and corrupts the average. Correct: SELECT AVG(age) FROM users WHERE id IN (SELECT user_id FROM posts GROUP BY user_id HAVING COUNT(*)>N).",
    # Domain column mapping
    "DOMAIN KNOWLEDGE COLUMN MAPPING: If DOMAIN KNOWLEDGE defines what a column means (e.g., 'rank = fastest lap ranking', 'position = race finish order'), use the EXACT column that matches the question's intent. 'ranked second' with rank defined as fastest lap → WHERE rank=2, NOT WHERE position=2.",
    # Format matching
    "VALUE FORMAT: Check SAMPLE DATA for the EXACT format of values before filtering. Time might be '1:54.123' not '0:01:54'. Dates might be integers (20130601) not strings ('2013-06-01'). Names might be 'DisplayName' not 'username'. Always match the exact format shown in SAMPLE DATA.",
    "EMPTY RESULT RECOVERY: If the PREVIOUS ATTEMPT FAILED section shows actual values from the DB, use THOSE exact values in your query — do not guess differently.",
    # Output shape rules
    "OUTPUT COLUMNS: 'What is the X and the Y?' or 'the average X and the average Y' = TWO SEPARATE columns in SELECT (one for X, one for Y). Do NOT combine them with + or concatenation.",
    "SINGULAR vs PLURAL: If the question uses 'the X' (singular/definite) AND provides enough filters to uniquely identify one row, expect 1 row. But NEVER add LIMIT 1 just because the grammar sounds singular — a person can have multiple payments, an entity can appear multiple times. Only LIMIT 1 if the question explicitly says 'the most recent' or 'the first'.",
    "TIED VALUES / NO LIMIT ON SUPERLATIVES: For 'which/what has the lowest/highest/most/least', ALWAYS use WHERE col = (SELECT MIN/MAX(col) ...) without LIMIT. Multiple rows may share the same min/max — return ALL of them. NEVER use ORDER BY + LIMIT 1 for superlatives.",
    # Temporal/ordering rules
    "TEMPORAL ORDERING: 'last time' / 'most recent' / 'latest' = ORDER BY date/time DESC LIMIT 1. 'first time' / 'earliest' = ORDER BY date/time ASC LIMIT 1. Check which column represents chronological order.",
    "MONTHLY from YEARLY: If the data stores YEARLY/ANNUAL totals and the question asks for 'monthly average' or 'per month', DIVIDE by 12. If data stores MONTHLY values, just use AVG directly. Check SAMPLE DATA to determine the granularity.",
    "TIME STRING PARSING: For time columns like '1:36.483' (mm:ss.ms), convert to seconds for comparison: CAST(SUBSTR(col,1,INSTR(col,':')-1) AS REAL)*60 + CAST(SUBSTR(col,INSTR(col,':')+1) AS REAL). Always handle this when computing time differences or percentages.",
    # NULL rejection
    "NEVER RETURN NULL: If your computation might produce NULL (e.g., division by zero, no matching rows in subquery), wrap with COALESCE or add WHERE col IS NOT NULL. A NULL answer is ALWAYS wrong — the question expects a real value.",
    # Aggregation scope
    "HAVING vs WHERE: 'where the average/total X exceeds/is greater than N' across a GROUP = GROUP BY + HAVING AVG(X) > N. This is NOT a per-row WHERE filter. 'average across schools' = group schools by district, compute avg per district, filter with HAVING.",
    "PER-GROUP POSITIONAL: 'the Nth item of EACH group' = use ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY position_col) then filter WHERE rn = N. Do NOT use a global OFFSET.",
    "INTERSECTION LOGIC: 'X with Y containing Z' means: find entities where BOTH conditions hold. Use: WHERE id IN (SELECT id WHERE condition1) AND id IN (SELECT id WHERE condition2). Two separate subqueries intersected, NOT one combined WHERE clause.",
    "SUPERLATIVE WITH TIES (reinforcement): WHERE col = (SELECT MIN/MAX(col)...) is the ONLY correct pattern for superlatives. LIMIT 1 is FORBIDDEN for any question with lowest/highest/most/least/best/worst.",
    "CO-LOCATED MEASURES: If the question applies a filter (e.g., 'approved', 'active') and that filter column lives in a detail table, use the value/measure column from that SAME detail table (e.g., expense.cost where expense.approved='true'), NOT from a parent summary table (e.g., budget.spent). Detail tables with per-record filters give accurate totals.",
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
    parts = [f"QUESTION: {question}\n\nWrite a SQL query to answer the QUESTION above."]

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

    rules = "\n".join(f"- {r}" for r in SQL_RULES)
    has_gaps = bool(gaps)
    if grounding_context:
        if has_gaps:
            rules += "\n- PREVIOUS ATTEMPT FAILED feedback takes PRIORITY over GROUNDING CONTEXT. If the feedback contradicts the grounding, follow the feedback."
            rules += "\n- Use GROUNDING CONTEXT for filter values and join paths, but fix what the feedback says is wrong."
        else:
            if "FILTER VALUES" in grounding_context:
                rules += "\n- ⚠️ MANDATORY: Your WHERE clause MUST use EXACTLY the values from FILTER VALUES above. Do NOT substitute other values."
        if "DATA FORMAT WARNINGS" in grounding_context:
            rules += "\n- ⚠️ Read DATA FORMAT WARNINGS carefully. Handle time strings, relative values, and encoded formats as described."
        if "IMPORTANT" in grounding_context and "RE-READ THE QUESTION" in grounding_context:
            rules += "\n- ⚠️ Read the IMPORTANT section carefully. It flags a potential column mismatch — verify your SELECT/GROUP BY uses the column the question actually refers to."
    elif knowledge_text and "formula" in knowledge_text.lower():
        rules += "\n- If DOMAIN KNOWLEDGE defines a formula, follow it exactly."
    parts.append(f"\nRULES:\n{rules}")

    # Put mandatory filter constraint LAST so it's freshest in model's context (only if no gaps)
    if grounding_context and "FILTER VALUES" in grounding_context and not has_gaps:
        import re as _re
        fv_match = _re.search(r"FILTER VALUES:\n((?:  .+\n?)+)", grounding_context)
        if fv_match:
            parts.append(f"\n⚠️ MANDATORY WHERE CLAUSE (do NOT change these values):\n{fv_match.group(1).strip()}")

    parts.append(f"\nREMINDER — answer THIS question: {question}")
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
    parts = [f"QUESTION: {question}\n\nDoes the SQL result below answer this QUESTION?"]

    if knowledge_text:
        parts.append(f"\nDOMAIN KNOWLEDGE:\n{knowledge_text[:1500]}")

    if grounding_context:
        parts.append(f"\nGROUNDING CONTEXT:\n{grounding_context}")

    if kg_context:
        parts.append(f"\nSCHEMA:\n{kg_context}")

    parts.append(f"\nSQL: {sql or '(none)'}")
    if sql_error:
        parts.append(f"ERROR: {sql_error}")

    parts.append(f"\nRESULTS:\n{data_text}")

    parts.append(f"""
Re-read the QUESTION: {question}
Does the RESULTS data answer it? Return ONLY a JSON object:
{{"verdict": "complete"/"incomplete", "reasoning": "why", "gaps": [], "info_queries": [], "suggested_sql": "..."}}

- If the result has data and it answers the QUESTION, verdict is "complete" — even if SQL differs from the plan.
- "incomplete" only if: empty result, error, wrong columns, or data clearly does NOT answer the question.
- Multiple rows are valid (ties/multiple matches). Do NOT reject just because of multiple results.
- suggested_sql must fix the actual problem. Never repeat the same failing query.
- NULL CHECK: Any NULL value = wrong query. Mark incomplete.
- COLUMN COUNT: "X and Y" in question = 2+ columns in result.
- LOGIC: "In X, what % of Y?" → X is WHERE filter. "How many times more" = division not subtraction.
- AGGREGATION: AVG of entity attributes via JOIN to detail table = WRONG (duplicated rows).
- TEMPORAL: "last/most recent" needs ORDER BY DESC LIMIT 1.
- HAVING: "where the average exceeds N" = GROUP BY + HAVING, not WHERE.
- SCOPE: "the X" (singular definite) MAY still return multiple rows if the entity has multiple records (e.g., "the date X paid" could be multiple payments). Only mark incomplete if there are clearly UNRELATED rows (wrong entity, wrong filter).""")

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
- Include definitions that DISTINGUISH between similar columns (e.g., "rank: fastest lap ranking" vs "position: race finish order") — this prevents using the wrong column.
- For use_case_sql: if the domain knowledge has a USE CASE whose question matches or is very similar to the user's question, copy its SQL EXACTLY. This SQL is the authoritative answer pattern.
- If a definition distinguishes between similar terms (e.g., "most severe = 1" vs "severe = 2"), quote BOTH so the distinction is clear.
- Be precise and complete — these anchors will be used as domain context for query planning.
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
  "expected_output": {{"columns": "number of output columns", "rows": "single/multiple/all-matching (use 'all-matching' unless a COUNT/SUM/AVG guarantees exactly 1 row; superlatives and lookups by name may return multiple)", "description": "brief description"}},
  "formula": "the EXACT SQL expression to compute the answer — if DOMAIN KNOWLEDGE defines a metric formula, translate it literally to SQL without simplifying or removing any operations",
  "computation_steps": ["step1: find X", "step2: calculate Y from X"],
  "data_requirements": ["table.column — include ALL columns that could be relevant: any column whose name appears in the question, columns needed for joins, columns needed for filtering, and columns needed for aggregation. Be INCLUSIVE — if the question mentions a word that matches a column name, include that column."],
  "data_format_notes": ["any unusual formats from SAMPLE DATA that need handling, e.g., time strings need parsing, relative values with + prefix, integer-encoded dates"],
  "reasoning": "brief HOW to get the answer — must trace back to what_user_wants",
  "domain_rules": ["constraints from DOMAIN KNOWLEDGE that affect the query"],
  "known_values": {{"table.column": ["filter values or expressions verified against SAMPLE DATA"]}},
  "join_paths": ["tableA.col -> tableB.col -> tableC.col"]
}}

RULES:
- Start by understanding what_user_wants — every other field must serve that goal.
- DOMAIN KNOWLEDGE section contains facts from domain knowledge. FIELD DEFINITIONS and METRIC FORMULAS provide context. Always verify column choices against the actual DATABASE SCHEMA and SAMPLE DATA. If a question term matches an actual column name in the schema, strongly consider using that column directly.
- EXACT LEVEL MATCHING: When DOMAIN KNOWLEDGE defines distinct named levels (e.g., "high = 1", "medium = 2", "low = 3"), the question's EXACT wording determines WHICH SINGLE level to use. "medium priority" = only the value labeled "medium" (2), NOT "high" (1). Do NOT combine multiple levels unless the question explicitly says "X or above" or "at least X". Each named label maps to exactly one value.
- USE CASE AUTHORITY: If DOMAIN KNOWLEDGE includes a MATCHING USE CASE whose title/explanation directly addresses the same condition as the question, copy its WHERE clause EXACTLY. The use case IS the answer — do not second-guess its filter values.
- For known_values: always include the TABLE name (e.g., "orders.order_date" not just "order_date"). Only use values that exist within that table's SAMPLE DATA range.
- CRITICAL: Check SAMPLE DATA to decide WHICH TABLE to filter. The same column name in different tables may have different formats or data coverage. Always filter the table that actually contains the data you need.
- For formula: if DOMAIN KNOWLEDGE defines a metric formula (e.g., "Metric = X / N"), translate it literally to SQL. Keep ALL parts — do NOT remove operations even if you think the data makes them redundant. The aggregation function must match the question intent ("average" → AVG, "total" → SUM).
- "per unit/per item/each" in the question means a RATIO (total ÷ quantity). Check SAMPLE DATA to determine which column is a total vs a quantity, then use division in the formula.
- For join_paths: trace the FULL FK path shown in DATABASE SCHEMA. Never skip intermediate tables.
- For data_requirements: be INCLUSIVE. List every table.column that could help answer the question — if a word in the question matches a column name in the schema, INCLUDE that column. Also include columns for joins and filters. The SQL planner will decide which to actually SELECT.
- Do NOT invent output columns the question didn't ask for. If the question says "list all X", SELECT only the column that identifies X — do NOT add properties (like amount, date) unless the question explicitly asks for them.
- If DOMAIN KNOWLEDGE includes a USE CASE SQL marked as AUTHORITATIVE, follow it EXACTLY — same WHERE values, same columns, same logic. Do NOT override its filter values even if they seem counterintuitive. The use case is the definitive answer pattern.
- If DOMAIN KNOWLEDGE includes a non-authoritative USE CASE SQL, follow its structure but ensure the selected/aggregated column semantically matches what the question asks about.
- POPULATION vs METRIC (critical for percentages/counts): Parse sentence structure carefully:
  * "In X, what is the percentage/count of Y?" → X is the population (WHERE filter = denominator), Y is what you measure.
  * "Among X, how many have Y?" → X is the population, Y is the condition being counted.
  * "Of X, what percentage are Y?" → denominator = COUNT(X), numerator = COUNT(X where Y).
  * WRONG: "In employees with salary > 50000, % managers" → filtering by role='manager' and computing % with salary. CORRECT: filter by salary > 50000, compute % that are managers.
- RATIO LANGUAGE: "How many times is X more than Y?" or "How many times was X more than Y?" = X divided by Y (a ratio). NOT subtraction, NOT a count. Result is a decimal number (e.g., 2.73).
- AGGREGATION GRAIN: When computing AVG/SUM of an entity's own attributes (e.g., "average age of users who..."), aggregate FROM the entity table with a WHERE/IN filter. Do NOT join to a detail table — that duplicates entity rows per detail record and corrupts the average. Example: AVG(users.age) for users with >10 posts → SELECT AVG(age) FROM users WHERE id IN (subquery on posts).
- COLUMN SEMANTICS: If DOMAIN KNOWLEDGE defines column meanings (e.g., "rank = based on fastest lap time", "position = race finish order"), map the question's wording to the CORRECT column. "ranked second" → use the column defined as "ranking", not "position".
- OUTPUT FORMAT: "What is the X and the Y?" or "average X and average Y" = formula must produce TWO SEPARATE columns. The formula should be "SELECT col1, col2 FROM ..." not "SELECT col1 + col2". Each distinct requested value = one column.
- TEMPORAL: "last time" / "most recent" / "latest" → ORDER BY date/time DESC LIMIT 1. "posted it last time" means the most recent poster, not any poster. Include ORDER BY in formula.
- MONTHLY vs YEARLY: If data stores one row PER MONTH (e.g., monthly_stats table with 12 rows per year per entity), "average monthly X" = AVG(value). If data stores ANNUAL totals (one row per year), "average monthly" = AVG(value) / 12. Check SAMPLE DATA row count vs time range to determine granularity.
- DATA FORMAT INSPECTION: Look at SAMPLE DATA's "distinct values" annotations. If values have FORMAT tags (e.g., [FORMAT: time string mm:ss.ms]), record this in data_format_notes. Your formula must handle these formats (e.g., convert time strings to seconds before doing math).
- GRANULARITY: Check SAMPLE DATA's GRANULARITY annotations. "~12 rows/entity" with monthly dates = monthly data. "~1 row/entity" with yearly = annual data. This determines whether to divide by 12 or not.
- HAVING vs WHERE: "where the average X exceeds N" or "schools where the average exceeds N" = this is a GROUP-level filter. The formula must use GROUP BY + HAVING, NOT a per-row WHERE clause. The GROUP BY groups by the entity (school, district, etc.), and HAVING filters groups by their aggregate.
- PER-GROUP POSITIONAL: "the Nth item of EACH group" needs ROW_NUMBER() OVER (PARTITION BY group ORDER BY position). Do NOT use global LIMIT/OFFSET.
- SUPERLATIVE TIES: "which has the lowest/highest" → set expected_output.rows = "all-matching" (NOT "single"). Use WHERE col = (SELECT MIN/MAX...) to get ALL ties. NEVER use LIMIT 1 for superlatives. Multiple rows sharing the same min/max are ALL correct answers.
- CO-LOCATED MEASURES: When the question mentions a filter condition (e.g., "approved", "active", "completed") AND that filter column exists in a detail table, prefer using the MEASURE column (cost, amount, value) from that SAME detail table rather than a pre-aggregated summary column in a parent table. Detail-level measures with detail-level filters give accurate results; summary columns may double-count or miss the filter.
- DETAIL vs SUMMARY: If a detail table (more rows, individual records) and a summary table (fewer rows, aggregated totals) BOTH have a value column, use the detail table when the question asks about filtered subsets. Summary tables lose per-record granularity needed for filtering.
- SUBSET DISCRIMINATION: When multiple similar values exist in the same column (e.g., 'X' and 'X Y'), and the question uses a QUALIFIER that distinguishes them, choose ONLY the value matching the qualifier — do NOT combine them. The SHORT/BASE form typically represents the default unmodified operation, while 'X Y' specifies a VARIANT (e.g., 'VYBER' = cash withdrawal vs 'VYBER KARTOU' = card withdrawal). "cash transactions" = ONLY the base form that means cash, NOT the extended form that specifies a different method. Only use IN(...) with multiple values when the question says "all" without a distinguishing qualifier.
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
    anchor_section = f"\nDOMAIN KNOWLEDGE:\n{anchor_text}" if anchor_text else ""
    prev_section = f"\nPREVIOUS ATTEMPT (fix the issues below):\n{previous_attempt}" if previous_attempt else ""
    return SEMANTIC_GROUNDING_PROMPT.format(
        question=question,
        kg_context=kg_context,
        sample_section=sample_section,
        anchor_section=anchor_section,
        previous_attempt=prev_section,
    )



def _format_grounding_for_sql(grounding: dict[str, Any]) -> str:
    """Format grounding as factual context for SQL planner — no SQL, no steps, no approach."""
    parts: list[str] = []

    # Join paths (factual FK relationships)
    join_paths = grounding.get("join_paths", [])
    if join_paths:
        parts.append("JOIN PATHS:\n" + "\n".join(f"  {jp}" for jp in join_paths))

    # Verified filter values
    known_values = grounding.get("known_values", {})
    if known_values:
        kv_lines = []
        for k, vs in known_values.items():
            if not vs:
                continue
            kv_lines.append(f"  {k}: {', '.join(str(v) for v in vs)}")
        if kv_lines:
            parts.append("FILTER VALUES:\n" + "\n".join(kv_lines))

    # Domain constraints (non-override rules from grounding)
    domain_rules = grounding.get("domain_rules", [])
    # Separate semantic feedback overrides from regular rules
    override_rules = grounding.get("_semantic_overrides", [])
    regular_rules = [r for r in domain_rules if r not in override_rules]
    if regular_rules:
        parts.append("CONSTRAINTS:\n" + "\n".join(f"  - {r}" for r in regular_rules))

    # Data format warnings
    format_notes = grounding.get("data_format_notes", [])
    if format_notes:
        parts.append("DATA FORMAT WARNINGS:\n" + "\n".join(f"  ⚠️ {n}" for n in format_notes))

    # Semantic overrides go last and marked as mandatory
    if override_rules:
        parts.append("⚠️ IMPORTANT — RE-READ THE QUESTION and consider this:\n" + "\n".join(f"  - {r}" for r in override_rules))

    if not parts:
        return ""
    return "GROUNDING CONTEXT:\n" + "\n".join(parts)


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

    parts.append(f"""
Return ONLY a JSON object:
{{"columns": ["col1", "col2"], "rows": [["value1", "value2"], ...]}}

RULES:
- Return EVERY row from SQL OUTPUT — never drop or truncate rows.
- Drop columns NOT needed to answer the question (e.g., intermediate IDs used only for joining).
- NEVER merge multiple columns into one.
- If a name column has "Firstname Lastname" as one string and the question asks for both, split into two columns.
- NEVER rename columns — use the exact SQL column names.
- Do NOT transform values — keep them exactly as in SQL OUTPUT.
- Do NOT add rows that aren't in SQL OUTPUT.

QUESTION being answered: {question}""")

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

            # Get sample data for each table (question-aware probing)
            sample_data = self._get_sample_data(db_path, kg, question)

            # Build column hints: map question words to actual column names
            col_hints = self._build_column_hints(question, kg)

            # Step 5: Semantic grounding — decompose question before SQL planning
            grounding_context, schema_slice = self._call_semantic_grounding(
                question, kg_context, sample_data, ctx.knowledge_text,
                db_path=db_path,
            )
            # Use schema slice for SQL planner if available, otherwise full schema
            sql_schema = schema_slice if schema_slice else kg_context

            # Filter sample data to only include tables in schema slice
            sql_sample_data = sample_data
            if schema_slice:
                slice_tables = set()
                for line in schema_slice.split("\n"):
                    if line.startswith("TABLE: "):
                        tname = line.split("TABLE: ")[1].split(" ")[0].strip()
                        slice_tables.add(tname.lower())
                if slice_tables:
                    filtered_parts: list[str] = []
                    current_block: list[str] = []
                    include_block = False
                    for line in sample_data.split("\n"):
                        if line.startswith("TABLE "):
                            if current_block and include_block:
                                filtered_parts.extend(current_block)
                            current_block = [line]
                            tname = line.split("TABLE ")[1].split(" ")[0].strip()
                            include_block = tname.lower() in slice_tables
                        else:
                            current_block.append(line)
                    if current_block and include_block:
                        filtered_parts.extend(current_block)
                    if filtered_parts:
                        sql_sample_data = "\n".join(filtered_parts)

            # Step 5b: Value Discovery — probe DB for actual filter values
            value_discovery = self._discover_filter_values(
                question, db_path, kg, grounding_context, ctx.knowledge_text,
            )
            if value_discovery:
                self._log("value_discovery", value_discovery[:300])

            # Step 5c: Threshold inference — infer normal/abnormal ranges if needed
            threshold_context = self._infer_thresholds(
                question, db_path, kg, ctx.knowledge_text,
            )
            if threshold_context:
                self._log("threshold_inference", threshold_context[:200])

            # Inject discovered values into grounding context
            if value_discovery:
                grounding_context += f"\n\nDISCOVERED VALUES (actual DB values for filter terms):\n{value_discovery}"
            if threshold_context:
                grounding_context += f"\n\n{threshold_context}"

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
            seen_sql_normalized: set[str] = set()

            for iteration in range(1, max_iterations + 1):
                self._log("iteration", f"--- Iteration {iteration}/{max_iterations} ---")

                # PLAN: Generate SQL (incorporate gaps + extra context if any)
                sql = self._call_sql(
                    question, sql_schema, sql_sample_data,
                    ctx.knowledge_text, gaps=gaps_text,
                    extra_context=extra_context,
                    column_hints=col_hints,
                    grounding_context=grounding_context,
                )
                self._log("sql_generated", sql)

                # Detect duplicate SQL BEFORE execution — save time
                sql_normalized = " ".join(sql.split()).strip().upper()
                if sql_normalized in seen_sql_normalized:
                    self._log("evaluate", "Verdict: duplicate SQL — stopping iterations")
                    break
                seen_sql_normalized.add(sql_normalized)

                # EXECUTE: Run the SQL
                sql_error = ""
                data_result = self._try_sql(db_path, sql)
                if data_result is None:
                    sql_error = self.steps[-1].get("detail", "") if self.steps else ""
                    data_result = {"columns": [], "rows": []}

                # Force-incomplete on empty results (skip LLM eval)
                if not data_result.get("rows"):
                    if sql_error:
                        self._log("evaluate", f"Verdict: incomplete (SQL error: {sql_error})")
                        all_gaps.add(f"SQL ERROR: {sql_error}. Check column names and table names against the DATABASE SCHEMA.")
                    else:
                        self._log("evaluate", "Verdict: incomplete (empty result)")
                        # Extract WHERE values from the failed SQL and flag them
                        where_hint = self._diagnose_empty_from_sql(sql, sql_sample_data)
                        if where_hint:
                            all_gaps.add(where_hint)
                        else:
                            all_gaps.add(
                                "Query returned 0 rows. One or more WHERE filters don't match actual data. "
                                "Check SAMPLE DATA for correct values, formats, and column names."
                            )
                    failed_sqls.append(sql)
                    gaps_text = "\n".join(f"- {g}" for g in all_gaps)
                    gaps_text += "\n".join(
                        f"\n- FAILED SQL (do not repeat): {s}" for s in failed_sqls
                    )
                    continue

                # Skip evaluate on last iteration — just use what we have
                if iteration == max_iterations:
                    break

                # EVALUATE: lightweight LLM feedback (OK or one-sentence issue)
                feedback = self._evaluate_result_feedback(
                    question, sql, data_result, grounding_context,
                )
                if not feedback:
                    self._log("evaluate", "Verdict: complete")
                    break

                self._log("evaluate", f"Verdict: incomplete — {feedback}")
                all_gaps.add(feedback)
                failed_sqls.append(sql)
                gaps_text = "\n".join(f"- {g}" for g in all_gaps)
                gaps_text += "\n".join(
                    f"\n- FAILED SQL (do not repeat): {s}" for s in failed_sqls
                )

            # Reject results where all values are NULL/None
            if data_result and data_result.get("rows"):
                rows = data_result["rows"]
                all_null = all(
                    all(v is None or str(v).strip().lower() in ("none", "null", "") for v in row)
                    for row in rows
                )
                if all_null:
                    self._log("null_rejection", "All result values are NULL/None — treating as empty")
                    failed_sqls.append(sql)
                    all_gaps.add("Query returned only NULL values. The computation failed — check column names, JOIN conditions, and whether subqueries return data. Try a different approach.")
                    gaps_text = "\n".join(f"- {g}" for g in all_gaps)
                    data_result = {"columns": [], "rows": []}

            # Multi-hypothesis: if loop failed, try alternative interpretations
            if not data_result or not data_result.get("rows"):
                hyp_result, hyp_sql = self._try_multi_hypothesis(
                    question, db_path, sql_schema, sample_data,
                    ctx.knowledge_text, grounding_context, col_hints,
                    failed_sqls=failed_sqls,
                    diagnosis=gaps_text,
                )
                if hyp_result and hyp_result.get("rows"):
                    # Also reject NULL multi-hypothesis results
                    all_null = all(
                        all(v is None or str(v).strip().lower() in ("none", "null", "") for v in row)
                        for row in hyp_result["rows"]
                    )
                    if not all_null:
                        data_result = hyp_result
                        sql = hyp_sql
                        self._log("multi_hypothesis_ok", f"Alternative SQL returned {len(hyp_result['rows'])} rows")
                    else:
                        self._log("multi_hypothesis_null", "Multi-hypothesis also returned NULL — skipping")

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

            # Validate result shape matches question expectations
            if data_result and data_result.get("rows"):
                data_result = self._validate_result_shape(
                    question, data_result, db_path, sql_schema, sample_data,
                    ctx.knowledge_text, grounding_context, col_hints,
                )

            # Python fallback: when SQL fails entirely, let LLM write Python
            if not data_result or not data_result.get("rows"):
                py_result = self._try_python_fallback(
                    question, db_path, sql_schema, sample_data,
                    ctx.knowledge_text, grounding_context,
                    failed_sqls=failed_sqls,
                )
                if py_result and py_result.get("rows"):
                    data_result = py_result

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
    # Lightweight result feedback (replaces heavy LLM evaluator)
    # ------------------------------------------------------------------

    def _evaluate_result_feedback(
        self,
        question: str,
        sql: str,
        data_result: dict[str, Any],
        grounding_context: str = "",
    ) -> str:
        """Lightweight feedback: returns empty string if OK, one-sentence issue otherwise."""
        data_text = self._format_data_as_table(data_result)

        prompt = f"""QUESTION: {question}

SQL: {sql}

RESULTS:
{data_text}

Does this result correctly answer the QUESTION?

If YES, respond with exactly: OK
If NO, respond with ONE sentence describing what's wrong (wrong column, wrong filter, missing data, NULL values, etc.)

RULES:
- If the result has data and the columns match what the question asks → OK.
- Multiple rows are VALID — do NOT reject just because there are multiple results.
- Do NOT question filter values — those are verified.
- Only flag real problems: NULL values, clearly wrong columns, empty where data should exist, wrong aggregation type."""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        raw = raw.strip()

        raw_upper = raw.upper()
        if raw_upper == "OK" or raw_upper.startswith("OK\n") or raw_upper.startswith("OK.") or raw_upper.endswith(" OK") or raw_upper.endswith(" OK.") or "is correct" in raw.lower():
            return ""
        return raw

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

    def _extract_domain_anchors(self, question: str, knowledge_text: str, db_path: Path | None = None) -> str:
        """Extract relevant domain definitions as immutable ground truth.

        Two-phase approach:
        1. Deterministic: extract USE CASE SQLs, field definitions, and verify against schema
        2. LLM: identify which definitions are relevant and map question terms to columns
        """
        if not knowledge_text:
            return ""

        # Phase 1: Deterministic extraction of use cases and field definitions
        deterministic_parts: list[str] = []

        # Extract all USE CASE SQL blocks
        use_case_pattern = re.compile(
            r'###\s*Use Case[^:]*:\s*(.+?)\n.*?```sql\s*\n(.+?)```\s*\n.*?Explanation[:\s]*(.+?)(?:\n\n|\Z)',
            re.DOTALL | re.IGNORECASE,
        )
        use_cases = use_case_pattern.findall(knowledge_text)

        # Score each use case by relevance to question
        # Prioritize use cases whose WHERE/filter condition matches the question's filter intent
        q_lower = question.lower()
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q_lower))

        # Extract the core filter terms from the question (nouns after "with/where/for/of")
        filter_phrases = re.findall(
            r'(?:with|where|for|of)\s+([a-z\s]+?)(?:\s*,|\s*list|\s*what|\s*how|\?|$)',
            q_lower,
        )
        filter_words = set()
        for phrase in filter_phrases:
            filter_words.update(w for w in phrase.split() if len(w) >= 3)

        best_use_case = None
        best_score = 0
        for uc_title, uc_sql, uc_explanation in use_cases:
            uc_text = uc_title.lower() + " " + uc_explanation.lower()
            uc_words = set(re.findall(r'\b[a-z]{3,}\b', uc_text))
            # Base score: keyword overlap
            overlap = len(q_words & uc_words)
            # Bonus: if the use case title/explanation mentions the question's filter terms
            filter_overlap = len(filter_words & uc_words)
            score = overlap + filter_overlap * 3
            if score > best_score:
                best_score = score
                best_use_case = (uc_title.strip(), uc_sql.strip(), uc_explanation.strip())

        if best_use_case and best_score >= 5:
            uc_sql = best_use_case[1]
            uc_valid = True
            uc_error = ""
            if db_path:
                uc_error = self._validate_formula_deterministic(db_path, uc_sql)
                if uc_error:
                    uc_valid = False
            if uc_valid:
                deterministic_parts.append(
                    f"MATCHING USE CASE (score={best_score}):\n"
                    f"  Title: {best_use_case[0]}\n"
                    f"  SQL: {uc_sql}\n"
                    f"  Explanation: {best_use_case[2]}\n"
                    f"  ⚠️ THIS USE CASE closely matches your question — follow its WHERE values and logic."
                )
            else:
                deterministic_parts.append(
                    f"MATCHING USE CASE (score={best_score}):\n"
                    f"  Title: {best_use_case[0]}\n"
                    f"  SQL: {uc_sql}\n"
                    f"  Explanation: {best_use_case[2]}\n"
                    f"  ⚠️ WARNING: This SQL is INVALID ({uc_error}). Follow its WHERE filter values but FIX the column references using the actual DATABASE SCHEMA."
                )

        # Extract all field definitions with their exact values/meanings
        field_defs = re.findall(
            r'-\s+\*{0,2}(\w[\w\s]*?)\*{0,2}\s*(?:\([\w\s]+\))?\s*:\s*(.+)',
            knowledge_text,
        )
        relevant_fields: list[str] = []
        for field_name, definition in field_defs:
            field_words = set(re.findall(r'\b[a-z]{3,}\b', field_name.lower() + " " + definition.lower()))
            if q_words & field_words:
                relevant_fields.append(f"- {field_name.strip()}: {definition.strip()}")

        if relevant_fields:
            deterministic_parts.append("FIELD DEFINITIONS:\n" + "\n".join(relevant_fields))

        # Phase 2: LLM extraction for nuanced mapping
        prompt = DOMAIN_ANCHOR_PROMPT.format(
            question=question,
            knowledge_text=knowledge_text[:4000],
        )
        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        llm_parts: list[str] = []
        if isinstance(parsed, dict):
            anchors = parsed.get("anchors", [])
            for a in anchors:
                llm_parts.append(f"- {a}")

            # Column mappings intentionally excluded — they pre-bias grounding.
            # The grounding step has full schema + sample data to determine correct columns.

            use_case_sql = parsed.get("use_case_sql")
            if use_case_sql and not best_use_case:
                llm_parts.append(f"\nMATCHING USE CASE SQL (follow this exactly):\n  {use_case_sql}")

        # Combine: deterministic parts take priority (placed first = fresher in context)
        all_parts = deterministic_parts + llm_parts
        if not all_parts:
            return knowledge_text[:2000]

        anchor_text = "\n".join(all_parts)

        # Translate formula anchors to SQL-ready form based on question intent
        q_lower = question.lower()
        if "average" in q_lower or "avg" in q_lower:
            def _rewrite_formula(m: re.Match) -> str:
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
        """Ground → Deterministic validate → Semantic feedback → Re-ground (max 2 iterations)."""
        grounding: dict[str, Any] = {}
        previous_attempt = ""

        # Extract domain anchors
        anchor_text = self._extract_domain_anchors(question, knowledge_text, db_path=db_path)

        # GROUND once — only retry if formula has bad SQL (table/column errors)
        max_formula_retries = 2
        for g_iter in range(1, max_formula_retries + 1):
            self._log("grounding_iter", f"--- Grounding iteration {g_iter} ---")

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
                self._log("semantic_grounding", "(failed to parse, retrying)")
                raw = self._model_call_with_retry(messages)
                grounding = self._parse_json(raw)

            if not isinstance(grounding, dict) or not grounding:
                self._log("semantic_grounding", "(failed to parse after retry)")
                return "", ""

            # Verify filter values against actual DB
            if db_path and grounding.get("known_values"):
                grounding = self._validate_filter_values(db_path, grounding)

            self._log(f"grounding_v{g_iter}", json.dumps(grounding, default=str))

            # DETERMINISTIC VALIDATE: try formula with LIMIT 0
            formula = grounding.get("formula", "")
            if not formula or not db_path:
                self._log("grounding_validated", "OK (no formula to validate)")
                break

            error = self._validate_formula_deterministic(db_path, formula)
            if error:
                self._log("grounding_formula_error", error)
                previous_attempt = (
                    f"Your previous formula failed with SQL error: {error}\n"
                    f"Failed formula: {formula}\n"
                    f"Fix the formula to use only tables and columns that exist in the DATABASE SCHEMA."
                )
                continue

            self._log("grounding_validated", "OK (formula valid)")
            break

        # SEMANTIC FEEDBACK: run once, inject as constraint if issue found
        # Don't re-ground — the SQL planner handles the constraint better
        if grounding:
            feedback = self._semantic_feedback(question, grounding, kg_context)
            if feedback:
                self._log("grounding_feedback", feedback)
                grounding["_semantic_overrides"] = [feedback]
            else:
                # Deterministic override: if question word matches an exact column name
                # not used in formula SELECT, inject as override for the SQL planner
                override = self._check_missing_select_columns(question, grounding, kg_context)
                if override:
                    self._log("grounding_deterministic_override", override)
                    grounding["_semantic_overrides"] = [override]

        # Deterministic enrichment: add any schema columns whose name matches a question word
        if db_path:
            grounding = self._enrich_data_requirements(db_path, question, grounding)

        formatted = _format_grounding_for_sql(grounding)
        self._log("semantic_grounding_final", formatted if formatted else "(empty)")

        # Build schema slice from enriched data_requirements
        schema_slice = ""
        if db_path:
            schema_slice = self._build_schema_slice(db_path, grounding)
            if schema_slice:
                table_names = [line.split("(")[0].strip() for line in schema_slice.split("\n") if line.startswith("TABLE: ")]
                self._log("schema_slice", f"{len(schema_slice)} chars, tables: {table_names}")

        return formatted, schema_slice

    def _enrich_data_requirements(
        self, db_path: Path, question: str, grounding: dict[str, Any]
    ) -> dict[str, Any]:
        """Deterministically add columns whose names appear in the question."""
        q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))
        data_reqs = set(grounding.get("data_requirements", []))

        try:
            conn = sqlite3.connect(str(db_path))
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall():
                tname = row[0]
                if tname.startswith("_"):
                    continue
                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                for col in cols:
                    col_lower = col.lower()
                    # If column name (or parts of it) appear in the question
                    col_parts = set(re.findall(r'[a-z]{3,}', col_lower))
                    if col_parts & q_words:
                        entry = f"{tname}.{col}"
                        if not any(entry.lower() in r.lower() for r in data_reqs):
                            data_reqs.add(entry)
            conn.close()
        except Exception:
            pass

        grounding["data_requirements"] = list(data_reqs)
        return grounding

    def _build_schema_slice(self, db_path: Path, grounding: dict[str, Any]) -> str:
        """Build a focused schema string from enriched data_requirements.

        Only includes tables/columns that appear in data_requirements, plus FK info
        for those tables. The SQL planner sees this instead of the full schema.
        """
        data_reqs = grounding.get("data_requirements", [])
        if not data_reqs:
            return ""

        # Parse data_requirements into {table: set(columns)}
        table_cols: dict[str, set[str]] = {}
        for req in data_reqs:
            if "." in req:
                parts = req.split(".", 1)
                table_name = parts[0].strip()
                col_name = parts[1].strip()
                # Handle descriptions like "table.column for filtering"
                col_name = col_name.split(" ")[0] if " " in col_name else col_name
                table_cols.setdefault(table_name, set()).add(col_name)

        if not table_cols:
            return ""

        try:
            conn = sqlite3.connect(str(db_path))
            lines: list[str] = []
            lines.append("=== DATABASE SCHEMA ===")
            lines.append("")

            fk_lines: list[str] = []

            for tname, req_cols in sorted(table_cols.items()):
                # Check table exists
                exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (tname,)
                ).fetchone()
                if not exists:
                    continue

                # Get row count
                try:
                    row_count = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
                except Exception:
                    row_count = "?"

                # Get all columns with their info
                col_info = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                pk_cols = [c[1] for c in col_info if c[5]]  # c[5] = pk flag

                lines.append(f"TABLE: {tname} ({row_count} rows, PK: {', '.join(pk_cols) if pk_cols else '(none)'})")

                # Include ALL columns of the table (planner needs full table context)
                for c in col_info:
                    col_name = c[1]
                    col_type = c[2] or "TEXT"
                    nullable = "" if c[3] == 0 else " NOT NULL"
                    pk_mark = " [PK]" if c[5] else ""
                    # Get sample values
                    sample = ""
                    try:
                        vals = conn.execute(
                            f'SELECT DISTINCT "{col_name}" FROM "{tname}" WHERE "{col_name}" IS NOT NULL LIMIT 5'
                        ).fetchall()
                        if vals:
                            sample = f"  e.g. {[v[0] for v in vals]}"
                    except Exception:
                        pass
                    lines.append(f"  - {col_name} ({col_type}{nullable}){pk_mark}{sample}")

                # Get FK info for this table
                try:
                    fks = conn.execute(f'PRAGMA foreign_key_list("{tname}")').fetchall()
                    for fk in fks:
                        ref_table = fk[2]
                        from_col = fk[3]
                        to_col = fk[4]
                        lines.append(f"  FK: {from_col} → {ref_table}.{to_col}")
                        fk_lines.append(f"  JOIN {ref_table} ON {tname}.{from_col} = {ref_table}.{to_col}")
                except Exception:
                    pass

                lines.append("")

            # Add inferred FKs from grounding join_paths
            join_paths = grounding.get("join_paths", [])
            if join_paths or fk_lines:
                lines.append("=== JOIN PATHS ===")
                for jp in join_paths:
                    # Convert "tableA.col -> tableB.col" to JOIN syntax
                    if "->" in jp:
                        parts = [p.strip() for p in jp.split("->")]
                        for i in range(len(parts) - 1):
                            src = parts[i]
                            dst = parts[i + 1]
                            if "." in src and "." in dst:
                                src_t, src_c = src.split(".", 1)
                                dst_t, dst_c = dst.split(".", 1)
                                line = f"  JOIN {dst_t} ON {src_t}.{src_c} = {dst_t}.{dst_c}"
                                if line not in fk_lines:
                                    fk_lines.append(line)
                for fl in fk_lines:
                    lines.append(fl)
                lines.append("")

            conn.close()
            return "\n".join(lines)

        except Exception:
            return ""

    def _check_missing_select_columns(self, question: str, grounding: dict[str, Any], kg_context: str) -> str:
        """Deterministic: if a question word EXACTLY matches a column name not in the formula SELECT, flag it.

        Only triggers when:
        1. A question word is an exact column name in the schema
        2. That column is NOT referenced in the formula
        3. The formula SELECTs a different column from a different table for grouping/output

        Returns override string or empty.
        """
        formula = grounding.get("formula", "")
        if not formula:
            return ""

        q_lower = question.lower()
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q_lower))
        formula_lower = formula.lower()

        # Parse SELECT columns from formula
        select_match = re.match(r'select\s+(.+?)\s+from\s', formula_lower, re.DOTALL)
        if not select_match:
            return ""
        select_clause = select_match.group(1)

        # Find columns from schema whose exact name matches a question word
        # but aren't in the formula at all
        missing_exact = []
        current_table = ""
        for line in kg_context.split("\n"):
            if line.startswith("TABLE: "):
                current_table = line.split("TABLE: ")[1].split(" ")[0].strip()
            elif line.strip().startswith("- ") and current_table:
                col_name = line.strip()[2:].split(" ")[0].strip()
                # Exact match: column name IS a question word
                if col_name.lower() in q_words:
                    # Not in formula at all
                    if col_name.lower() not in formula_lower:
                        missing_exact.append((current_table, col_name))

        if not missing_exact:
            return ""

        # Check if the formula already has a GROUP BY on a different column
        # (suggesting the missing column should replace it)
        has_group_by = "group by" in formula_lower
        if not has_group_by and len(missing_exact) == 1:
            t, c = missing_exact[0]
            return f"The question mentions '{c}' which is an actual column in {t} table. Consider whether SELECT should include {t}.{c}."

        if has_group_by:
            t, c = missing_exact[0]
            return f"The question asks for '{c}' which is an actual column in the {t} table — use {t}.{c} in SELECT/GROUP BY instead of a different categorical column."

        return ""

    def _semantic_feedback(self, question: str, grounding: dict[str, Any], kg_context: str) -> str:
        """Lightweight semantic check: does the plan answer the exact question?

        Returns feedback string if there's an issue, empty string if OK.
        Only checks SELECT columns and GROUP BY — does NOT question filter values
        that are backed by domain knowledge.
        """
        formula = grounding.get("formula", "")
        what_user_wants = grounding.get("what_user_wants", "")
        domain_rules = grounding.get("domain_rules", [])

        # Deterministic check: if a question word matches a column name not in the formula,
        # flag it so the LLM can reconsider
        missing_cols_hint = ""
        q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))
        formula_lower = formula.lower()
        # Find table.column pairs from schema that match question words but aren't in formula
        missing = []
        current_table = ""
        for line in kg_context.split("\n"):
            if line.startswith("TABLE: "):
                current_table = line.split("TABLE: ")[1].split(" ")[0].strip()
            elif line.strip().startswith("- ") and current_table:
                col_name = line.strip()[2:].split(" ")[0].strip()
                col_parts = set(re.findall(r'[a-z]{3,}', col_name.lower()))
                if col_parts & q_words:
                    if col_name.lower() not in formula_lower and f"{current_table}.{col_name}".lower() not in formula_lower:
                        missing.append(f"{current_table}.{col_name}")

        if missing:
            missing_cols_hint = f"\n\nNOTE: The following columns match words in the question but are NOT used in the formula: {', '.join(missing)}. Consider whether any of these should be in the SELECT or GROUP BY instead of/in addition to the current columns."

        # Show domain rules so feedback doesn't contradict them
        domain_section = ""
        if domain_rules:
            domain_section = f"\n\nDOMAIN RULES (these are VERIFIED facts — do NOT contradict them):\n" + "\n".join(f"  - {r}" for r in domain_rules)

        prompt = f"""QUESTION: {question}

DATABASE SCHEMA:
{kg_context}

PLAN says user wants: {what_user_wants}
PLAN formula: {formula}{domain_section}{missing_cols_hint}

Check ONLY these aspects of the formula:
1. Are the SELECT columns exactly what the question asks for? (no extra, no missing)
2. Is the GROUP BY / aggregation matching what the question expects?
3. Does the question ask for individual items or a summary total?

If the plan is correct, respond with exactly: OK
If there's a problem with SELECT columns or GROUP BY, respond with ONE sentence describing what's wrong. Do NOT rewrite the SQL.

RULES:
- Do NOT question WHERE filter values — they come from verified domain knowledge.
- Do NOT suggest different filter values based on column ranges or your assumptions.
- ONLY flag issues with which columns are in SELECT or how results are grouped/aggregated.
- If a word in the question (e.g., "type", "status", "name") matches an actual column name in the schema, the question likely refers to THAT column directly.
- "total value" = a single aggregated number per group, not individual line items.
- When the question says "identify the X and their Y", X is likely a column to SELECT and Y is the aggregation."""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        raw = raw.strip()

        raw_upper = raw.upper()
        if raw_upper == "OK" or raw_upper.startswith("OK\n") or raw_upper.startswith("OK.") or raw_upper.endswith(" OK") or raw_upper.endswith(" OK.") or "is correct" in raw.lower():
            return ""
        return raw

    def _validate_formula_deterministic(self, db_path: Path, formula: str) -> str:
        """Validate formula SQL by executing with LIMIT 0. Returns error string or empty."""
        conn = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.execute("PRAGMA busy_timeout = 5000")
            test_sql = f"SELECT * FROM ({formula}) LIMIT 0"
            conn.execute(test_sql)
            conn.close()
            return ""
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return str(e)

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


    def _diagnose_empty_from_sql(self, sql: str, sample_data: str) -> str:
        """Parse the failed SQL's WHERE clause and check values against SAMPLE DATA.

        No DB access — purely string-based analysis.
        """
        if not sql or "WHERE" not in sql.upper():
            return ""

        # Extract WHERE clause
        where_idx = sql.upper().find("WHERE")
        where_clause = sql[where_idx + 5:].strip()
        for keyword in ["ORDER BY", "GROUP BY", "LIMIT", "HAVING"]:
            kw_idx = where_clause.upper().find(keyword)
            if kw_idx > 0:
                where_clause = where_clause[:kw_idx].strip()

        # Extract individual conditions
        conditions = [c.strip() for c in re.split(r'\bAND\b', where_clause, flags=re.IGNORECASE) if c.strip()]
        if not conditions:
            return ""

        # Extract string values used in filters
        issues: list[str] = []
        sample_lower = sample_data.lower()
        for cond in conditions:
            str_values = re.findall(r"'([^']*)'", cond)
            for val in str_values:
                if val and val.lower() not in sample_lower:
                    issues.append(f"Filter value '{val}' in condition '{cond}' not found in SAMPLE DATA — likely wrong value or wrong column.")
            # Check numeric comparisons against a column that might not have matching values
            num_comparisons = re.findall(r'(\w+(?:\.\w+)?)\s*[<>=!]+\s*(\d+)', cond)
            for col_ref, num_val in num_comparisons:
                col_name = col_ref.split(".")[-1].lower() if "." in col_ref else col_ref.lower()
                # Check if this column's sample values suggest the number is out of range
                # Look for the column in sample data
                if col_name in sample_lower:
                    # Found the column — check if num_val appears anywhere near sample values
                    if num_val not in sample_data:
                        issues.append(f"Numeric filter '{cond}' — value {num_val} not seen in SAMPLE DATA for column '{col_name}'. Check if this column/value is correct.")

        if issues:
            return "EMPTY RESULT — these filters likely caused it:\n" + "\n".join(f"  - {i}" for i in issues[:3])
        return "Query returned 0 rows. The combined WHERE conditions are too restrictive — try removing or relaxing one filter at a time."

    def _diagnose_empty_result(self, db_path: Path, sql: str) -> str:
        """When SQL returns 0 rows, isolate which filter causes the empty result.

        Returns an actionable diagnosis: identifies the blocker filter and shows
        actual DB values so the planner can fix it in one iteration.
        """
        if not sql or not db_path.exists():
            return ""

        conn = sqlite3.connect(str(db_path))
        diagnostics: list[str] = []
        try:
            sql_upper = sql.upper()
            if "WHERE" not in sql_upper:
                return ""

            where_idx = sql.upper().find("WHERE")
            base_sql = sql[:where_idx].strip()

            # Check if base query (no filters) returns rows
            try:
                base_count = conn.execute(
                    f"SELECT COUNT(*) FROM ({base_sql})"
                ).fetchone()[0]
                if base_count == 0:
                    diagnostics.append(
                        "JOIN itself returns 0 rows — the JOIN conditions are wrong. Check FK paths."
                    )
                    return "EMPTY RESULT DIAGNOSIS:\n" + "\n".join(diagnostics)
            except Exception:
                pass

            # Extract individual WHERE conditions
            where_clause = sql[where_idx + 5:].strip()
            for keyword in ["ORDER BY", "GROUP BY", "LIMIT", "HAVING"]:
                kw_idx = where_clause.upper().find(keyword)
                if kw_idx > 0:
                    where_clause = where_clause[:kw_idx].strip()

            conditions = [c.strip() for c in re.split(r'\bAND\b', where_clause, flags=re.IGNORECASE) if c.strip()]

            # Test each condition: remove it and see if rows appear
            blockers: list[tuple[str, int]] = []
            for i, cond in enumerate(conditions):
                remaining = [c for j, c in enumerate(conditions) if j != i]
                test_sql = f"{base_sql} WHERE {' AND '.join(remaining)}" if remaining else base_sql
                try:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM ({test_sql})"
                    ).fetchone()[0]
                    if count > 0:
                        blockers.append((cond.strip(), count))
                except Exception:
                    pass

            if not blockers:
                # All conditions together block — combined filter is too restrictive
                diagnostics.append(
                    f"All {len(conditions)} filters COMBINED produce 0 rows. "
                    f"The combination is too restrictive — remove or relax one filter."
                )
            else:
                for cond, count_without in blockers:
                    diagnostics.append(f"REMOVE THIS FILTER: '{cond}' (without it: {count_without} rows)")

                    # Show actual values for the column in this condition
                    remaining = [c for c in conditions if c.strip() != cond]
                    test_sql = f"{base_sql} WHERE {' AND '.join(remaining)}" if remaining else base_sql

                    # Find column reference in the condition
                    for part in cond.replace("(", " ").replace(")", " ").split():
                        col_ref = part.strip("\"'`=<>!,")
                        if not col_ref or col_ref.upper() in ("AND", "OR", "NOT", "IN", "LIKE", "IS", "NULL", "BETWEEN", "SELECT", "FROM"):
                            continue
                        if "." in col_ref:
                            _, col = col_ref.split(".", 1)
                            col = col.strip("\"'`")
                        else:
                            col = col_ref
                        try:
                            actual = conn.execute(
                                f'SELECT DISTINCT "{col}" FROM ({test_sql}) WHERE "{col}" IS NOT NULL AND "{col}" != \'\' LIMIT 10'
                            ).fetchall()
                            vals = [r[0] for r in actual]
                            if vals:
                                diagnostics.append(f"  Actual values for '{col}' in matching rows: {vals}")
                                # Show what the filter used vs what exists
                                str_match = re.findall(r"'([^']*)'", cond)
                                num_match = re.findall(r'[<>=!]+\s*(\d+)', cond)
                                if str_match:
                                    used_val = str_match[0]
                                    close = [v for v in vals if isinstance(v, str) and (
                                        used_val.lower() in v.lower() or v.lower() in used_val.lower()
                                    )]
                                    if close:
                                        diagnostics.append(f"  FIX: Replace '{used_val}' with '{close[0]}' (or use LIKE '%{used_val}%')")
                                    else:
                                        diagnostics.append(f"  FIX: Value '{used_val}' does NOT exist. Use one of: {vals[:5]}")
                                elif num_match:
                                    diagnostics.append(f"  FIX: Numeric filter '{cond}' excludes all. Actual range: {min(vals)} to {max(vals)}")
                                else:
                                    diagnostics.append(f"  FIX: This column doesn't have values matching your filter. Remove this condition or use a different column.")
                                break
                        except Exception:
                            continue
        finally:
            conn.close()

        if diagnostics:
            return "EMPTY RESULT DIAGNOSIS:\n" + "\n".join(diagnostics)
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
                bad_col = error.split("no such column:")[-1].strip()
                bare_col = bad_col.split(".")[-1] if "." in bad_col else bad_col
                bare_col = bare_col.strip("\"'`")

                # Find tables referenced in the SQL and show their actual columns
                # Also suggest close matches
                for tname in tables:
                    if tname.lower() in sql.lower():
                        cols = [c[1] for c in conn.execute(
                            f'PRAGMA table_info("{tname}")'
                        ).fetchall()]
                        hints.append(f"Table '{tname}' has columns: {cols}")
                        # Find close matches for the bad column
                        close = [c for c in cols if bare_col.lower() in c.lower() or c.lower() in bare_col.lower()]
                        if close:
                            hints.append(f"  Did you mean: {close}?")

            elif "no such table" in error:
                bad_table = error.split("no such table:")[-1].strip() if "no such table:" in error else ""
                hints.append(f"Available tables: {tables}")
                if bad_table:
                    close = [t for t in tables if bad_table.lower() in t.lower() or t.lower() in bad_table.lower()]
                    if close:
                        hints.append(f"  Did you mean: {close}?")

        finally:
            conn.close()

        return "\n".join(hints)

    # ------------------------------------------------------------------
    # Component: Value Discovery
    # ------------------------------------------------------------------

    def _discover_filter_values(
        self,
        question: str,
        db_path: Path,
        kg: KnowledgeGraph,
        grounding_context: str,
        knowledge_text: str,
    ) -> str:
        """Probe DB for actual values that match question filter terms."""
        if not db_path or not db_path.exists():
            return ""

        # Extract filter terms from grounding known_values and question keywords
        # Find quoted terms and meaningful nouns from the question
        quoted_terms = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
        quoted = [t[0] or t[1] for t in quoted_terms]

        # Also extract key terms from grounding
        grounding_values: list[str] = []
        if "FILTER VALUES:" in grounding_context:
            fv_section = grounding_context.split("FILTER VALUES:")[1].split("\n\n")[0]
            for line in fv_section.strip().split("\n"):
                vals = re.findall(r":\s*(.+)", line)
                if vals:
                    grounding_values.extend(v.strip() for v in vals[0].split(","))

        all_terms = quoted + grounding_values
        if not all_terms:
            return ""

        conn = sqlite3.connect(str(db_path))
        discoveries: list[str] = []
        try:
            for table in kg.tables:
                cols = [c[1] for c in conn.execute(
                    f'PRAGMA table_info("{table.name}")'
                ).fetchall()]
                for col in cols:
                    for term in all_terms:
                        term_clean = term.strip("'\"")
                        if not term_clean or len(term_clean) < 2:
                            continue
                        try:
                            # Check if the exact value exists
                            exact = conn.execute(
                                f'SELECT COUNT(*) FROM "{table.name}" WHERE "{col}" = ?',
                                (term_clean,)
                            ).fetchone()[0]
                            if exact > 0:
                                continue  # Value exists, no problem

                            # Check LIKE match
                            like_results = conn.execute(
                                f'SELECT DISTINCT "{col}" FROM "{table.name}" '
                                f'WHERE "{col}" LIKE ? COLLATE NOCASE LIMIT 5',
                                (f'%{term_clean}%',)
                            ).fetchall()
                            if like_results:
                                vals = [r[0] for r in like_results if r[0]]
                                if vals:
                                    discoveries.append(
                                        f"  '{term_clean}' not found exactly in {table.name}.{col}, "
                                        f"but LIKE matches: {vals}"
                                    )
                        except Exception:
                            continue
        finally:
            conn.close()

        if discoveries:
            return "\n".join(discoveries[:10])
        return ""

    # ------------------------------------------------------------------
    # Component: Threshold Inference
    # ------------------------------------------------------------------

    def _infer_thresholds(
        self,
        question: str,
        db_path: Path,
        kg: KnowledgeGraph,
        knowledge_text: str,
    ) -> str:
        """Infer normal/abnormal thresholds from data distribution when not in knowledge."""
        q_lower = question.lower()
        needs_threshold = any(w in q_lower for w in ("normal", "abnormal", "elevated", "low level", "high level"))
        if not needs_threshold:
            return ""

        # Check if knowledge already defines thresholds
        if knowledge_text:
            k_lower = knowledge_text.lower()
            # Find which field the question refers to
            threshold_fields: list[str] = []
            for word in re.findall(r'\b[a-z]{2,}\b', q_lower):
                if word in ("normal", "abnormal", "level", "levels", "have", "their", "them"):
                    continue
                if word in k_lower:
                    # Check if threshold is already defined
                    idx = k_lower.find(word)
                    context = knowledge_text[max(0, idx-50):idx+200]
                    if any(t in context.lower() for t in ("range", "above", "below", "between", "normal")):
                        return ""  # Already defined
                    threshold_fields.append(word)

        if not db_path or not db_path.exists():
            return ""

        conn = sqlite3.connect(str(db_path))
        inferences: list[str] = []
        try:
            for table in kg.tables:
                cols_info = conn.execute(f'PRAGMA table_info("{table.name}")').fetchall()
                for col_info in cols_info:
                    col = col_info[1]
                    col_type = col_info[2].lower()
                    col_lower = col.lower()

                    # Check if this column is referenced by the question
                    if not any(w in col_lower for w in re.findall(r'\b[a-z]{3,}\b', q_lower)):
                        continue

                    # Only for numeric columns
                    if col_type not in ("real", "integer", "numeric", "float", "double", "int"):
                        # Check if values are actually numeric
                        try:
                            test = conn.execute(
                                f'SELECT CAST("{col}" AS REAL) FROM "{table.name}" '
                                f'WHERE "{col}" IS NOT NULL LIMIT 1'
                            ).fetchone()
                            if test is None:
                                continue
                        except Exception:
                            continue

                    try:
                        stats = conn.execute(
                            f'SELECT MIN(CAST("{col}" AS REAL)), '
                            f'MAX(CAST("{col}" AS REAL)), '
                            f'AVG(CAST("{col}" AS REAL)), '
                            f'COUNT(*) '
                            f'FROM "{table.name}" WHERE "{col}" IS NOT NULL'
                        ).fetchone()
                        if stats and stats[3] > 0:
                            inferences.append(
                                f"  {table.name}.{col}: min={stats[0]}, max={stats[1]}, "
                                f"avg={stats[2]:.2f}, count={stats[3]}"
                            )
                    except Exception:
                        continue
        finally:
            conn.close()

        if inferences:
            return (
                "THRESHOLD CONTEXT (data distribution — use with DOMAIN KNOWLEDGE to determine normal ranges):\n"
                + "\n".join(inferences[:8])
            )
        return ""

    # ------------------------------------------------------------------
    # Component: Multi-Hypothesis SQL
    # ------------------------------------------------------------------

    def _try_multi_hypothesis(
        self,
        question: str,
        db_path: Path,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
        grounding_context: str,
        column_hints: str,
        failed_sqls: list[str] | None = None,
        diagnosis: str = "",
    ) -> tuple[dict[str, Any] | None, str]:
        """Generate multiple SQL interpretations and pick the one that returns data."""
        if not db_path or not db_path.exists():
            return None, ""

        failed_section = ""
        if failed_sqls:
            failed_section = "\nPREVIOUSLY FAILED SQLs (do NOT repeat these patterns):\n"
            failed_section += "\n".join(f"  - {s[:200]}" for s in failed_sqls[-3:])

        diag_section = ""
        if diagnosis:
            diag_section = f"\nDIAGNOSIS OF FAILURES:\n{diagnosis[:1000]}"

        prompt = f"""The previous SQL attempts all returned EMPTY results or failed.
The question might have ambiguous terms that map to different columns or values.

QUESTION: {question}

DATABASE SCHEMA:
{kg_context}

SAMPLE DATA:
{sample_data[:2000]}

{f"DOMAIN KNOWLEDGE: {knowledge_text[:1000]}" if knowledge_text else ""}
{failed_section}
{diag_section}

Generate 3 DIFFERENT SQL interpretations of this question. Each should try a DIFFERENT:
- Column for ambiguous terms (e.g., "number" could be car_number, grid, position, round)
- Filter value interpretation (e.g., "ranked" could mean position or rank column)
- Join path or table choice
- Value format (e.g., time as '1:54.000' vs '0:01:54', date as 20130601 vs '2013-06-01')

Return ONLY a JSON object:
{{"hypotheses": [{{"reasoning": "why this interpretation", "sql": "SELECT ..."}}, ...]}}

RULES:
- Each hypothesis MUST be materially different (different WHERE column, different JOIN, or different interpretation)
- Do NOT repeat any previously failed patterns shown above
- Use LIKE for text matching when unsure of exact format
- If DIAGNOSIS shows actual values from the DB, USE them in at least one hypothesis
- Try both strict and loose interpretations
- If the question says "X and Y" (two values), make sure at least one hypothesis returns 2 columns
- If the question uses "last/latest/most recent", use ORDER BY DESC LIMIT 1 in at least one hypothesis
- For time strings like '1:36.483', try: CAST(SUBSTR(col,1,INSTR(col,':')-1) AS REAL)*60 + CAST(SUBSTR(col,INSTR(col,':')+1) AS REAL) for conversion to seconds
- NEVER return NULL — wrap computations in COALESCE and add WHERE ... IS NOT NULL filters"""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if not isinstance(parsed, dict):
            return None, ""

        hypotheses = parsed.get("hypotheses", [])
        if not hypotheses:
            return None, ""

        # Execute each hypothesis and return the first that produces rows
        best_result = None
        best_sql = ""
        best_rows = 0

        for hyp in hypotheses[:3]:
            hyp_sql = hyp.get("sql", "")
            if not hyp_sql:
                continue

            result = self._try_sql(db_path, hyp_sql)
            if result and result.get("rows"):
                row_count = len(result["rows"])
                self._log("hypothesis_tested",
                          f"{hyp.get('reasoning', '')[:60]} → {row_count} rows")
                # Prefer the result with most rows (but not too many — likely wrong)
                if row_count > best_rows and row_count <= 100:
                    best_result = result
                    best_sql = hyp_sql
                    best_rows = row_count

        return best_result, best_sql

    # ------------------------------------------------------------------
    # Python fallback: when SQL can't solve it, write Python
    # ------------------------------------------------------------------

    def _try_python_fallback(
        self,
        question: str,
        db_path: Path,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
        grounding_context: str,
        failed_sqls: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Last resort: LLM writes Python to query DB and compute the answer."""
        if not db_path or not db_path.exists():
            return None

        from data_agent_baseline.tools.python_exec import execute_python_code

        failed_section = ""
        if failed_sqls:
            failed_section = "FAILED SQL ATTEMPTS (these all returned empty/NULL — Python must take a different approach):\n"
            failed_section += "\n".join(f"  {s[:150]}" for s in failed_sqls[-3:])

        prompt = f"""SQL failed to answer this question. Write a Python script that queries the SQLite database and computes the answer.

QUESTION: {question}

DATABASE SCHEMA:
{kg_context[:2000]}

SAMPLE DATA:
{sample_data[:1500]}

{f"DOMAIN KNOWLEDGE: {knowledge_text[:1000]}" if knowledge_text else ""}

{f"GROUNDING: {grounding_context[:1000]}" if grounding_context else ""}

{failed_section}

Write a Python script that:
1. Connects to the SQLite database at "_consolidated.db" (already in working directory)
2. Queries the data needed
3. Performs any computation (string parsing, time conversion, multi-step logic)
4. Prints the FINAL ANSWER as a single line in CSV format: col1,col2\\nval1,val2
   (first line = column names, subsequent lines = data rows)

Return ONLY a JSON object:
{{"reasoning": "step-by-step plan", "python": "import sqlite3\\n..."}}

RULES:
- The DB file is "_consolidated.db" in the current working directory
- Print ONLY the final CSV output (header + data rows). No other prints.
- Handle string time formats like "1:36.483" (minutes:seconds.ms) — convert to seconds for math
- Handle relative time formats like "+16.445" (seconds behind leader)
- Use try/except for robustness
- If a value might be NULL, filter it out or provide a default
- Keep it simple — under 30 lines"""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if not isinstance(parsed, dict) or not parsed.get("python"):
            return None

        code = parsed["python"]
        self._log("python_fallback", f"Executing Python ({len(code)} chars): {parsed.get('reasoning', '')[:100]}")

        result = execute_python_code(
            context_root=db_path.parent,
            code=code,
            timeout_seconds=30,
        )

        if not result.get("success"):
            self._log("python_error", f"Failed: {result.get('error', '')[:200]}")
            # Try once more with the error context
            retry_prompt = f"""The Python script failed with this error:
{result.get('error', '')}
{result.get('stderr', '')[:500]}

Fix the script and try again. The DB is at "_consolidated.db".
Return ONLY: {{"python": "import sqlite3\\n..."}}"""
            messages = [ModelMessage(role="user", content=retry_prompt)]
            raw = self._model_call_with_retry(messages)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("python"):
                result = execute_python_code(
                    context_root=db_path.parent,
                    code=parsed["python"],
                    timeout_seconds=30,
                )
                if not result.get("success"):
                    self._log("python_retry_error", f"Still failed: {result.get('error', '')[:200]}")
                    return None

        output = result.get("output", "").strip()
        if not output:
            self._log("python_empty", "No output produced")
            return None

        self._log("python_output", output[:300])

        # Parse CSV output into result dict
        lines = [l.strip() for l in output.split("\n") if l.strip()]
        if len(lines) < 2:
            # Single value — wrap as 1-col table
            if len(lines) == 1:
                # Could be just a value or a header
                return {"columns": ["result"], "rows": [[lines[0]]]}
            return None

        import csv as csv_mod
        try:
            reader = csv_mod.reader(lines)
            columns = next(reader)
            rows = [list(row) for row in reader]
            if rows:
                # Filter out None/NULL rows
                valid_rows = [r for r in rows if not all(
                    v.strip().lower() in ("none", "null", "") for v in r
                )]
                if valid_rows:
                    self._log("python_success", f"Got {len(valid_rows)} rows, {len(columns)} cols")
                    return {"columns": columns, "rows": valid_rows}
        except Exception as e:
            self._log("python_parse_error", str(e))

        return None

    # ------------------------------------------------------------------
    # Post-execution result shape validation
    # ------------------------------------------------------------------

    def _validate_result_shape(
        self,
        question: str,
        data_result: dict[str, Any],
        db_path: Path,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
        grounding_context: str,
        column_hints: str,
    ) -> dict[str, Any]:
        """Validate result shape matches question expectations and fix if needed."""
        rows = data_result.get("rows", [])
        cols = data_result.get("columns", [])
        if not rows:
            return data_result

        q_lower = question.lower()

        # Check: "X and Y" pattern expects 2+ columns but we got 1
        # Pattern: "what is the X and the Y" or "average X and average Y"
        and_pattern = re.search(
            r'(?:what is|identify|find)\s+(?:the\s+)?(\w+).+?\band\b\s+(?:the\s+)?(\w+)',
            q_lower,
        )
        if and_pattern and len(cols) == 1 and len(rows) == 1:
            # We have a single combined value — need to re-query as two columns
            self._log("shape_fix", f"Question asks for two values ('{and_pattern.group(1)}' and '{and_pattern.group(2)}') but got 1 column — re-querying")
            fix_prompt = f"""The SQL returned a SINGLE combined value but the question asks for TWO SEPARATE values.

QUESTION: {question}
CURRENT RESULT: {cols[0]} = {rows[0][0]}

The question asks for '{and_pattern.group(1)}' AND '{and_pattern.group(2)}' as SEPARATE values.

DATABASE SCHEMA:
{kg_context[:2000]}

Write a corrected SQL that returns TWO columns (one for each value).
Return ONLY: {{"sql": "SELECT ..."}}"""
            messages = [ModelMessage(role="user", content=fix_prompt)]
            raw = self._model_call_with_retry(messages)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("sql"):
                fix_result = self._try_sql(db_path, parsed["sql"])
                if fix_result and fix_result.get("rows") and len(fix_result.get("columns", [])) >= 2:
                    self._log("shape_fixed", f"Now has {len(fix_result['columns'])} columns")
                    return fix_result

        # Check: singular question ("what was THE score") but got many rows
        singular_patterns = [
            r"what (?:is|was|were) the .+? (?:for|of|in) the ",
            r"what is the .+? of the ",
            r"identify the .+? (?:for|of) the ",
        ]
        expects_singular = any(re.search(p, q_lower) for p in singular_patterns)
        plural_indicators = ["list", "all", "each", "every", "which", "how many",
                             "lowest", "highest", "most", "least", "best", "worst"]
        has_plural = any(p in q_lower for p in plural_indicators)

        if expects_singular and not has_plural and len(rows) > 5:
            self._log("shape_fix_singular", f"Singular question but got {len(rows)} rows — attempting fix")
            fix_prompt = f"""The SQL returned {len(rows)} rows but the question expects a SINGLE result (it uses "the" indicating one specific item).

QUESTION: {question}
CURRENT SQL RETURNED: {len(rows)} rows with columns {cols}
FIRST FEW ROWS: {rows[:3]}

The question likely needs additional filters from its context that were missed. Re-read the question and add the missing WHERE conditions to narrow to exactly 1 row.

DATABASE SCHEMA:
{kg_context[:1500]}

Return ONLY: {{"sql": "SELECT ..."}}"""
            messages = [ModelMessage(role="user", content=fix_prompt)]
            raw = self._model_call_with_retry(messages)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("sql"):
                fix_result = self._try_sql(db_path, parsed["sql"])
                if fix_result and fix_result.get("rows") and 0 < len(fix_result["rows"]) < len(rows):
                    self._log("shape_fixed_singular", f"Narrowed from {len(rows)} to {len(fix_result['rows'])} rows")
                    return fix_result

        return data_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_sample_data(self, db_path: Path, kg: KnowledgeGraph, question: str = "") -> str:
        """Get sample rows + date ranges + value ranges + question-aware probing."""
        parts: list[str] = []
        q_words = set(re.findall(r"[a-z]{3,}", question.lower())) if question else set()
        conn = sqlite3.connect(str(db_path))
        for table in kg.tables:
            try:
                cursor = conn.execute(f'SELECT * FROM "{table.name}" LIMIT 3')
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()

                # Infer table role from structure
                fk_count = len(table.foreign_keys) + sum(
                    1 for src, fk in kg.inferred_fks if src == table.name
                )
                has_numeric_measures = any(
                    c.sql_type.upper() in ("REAL", "FLOAT", "NUMERIC", "DOUBLE")
                    and not c.name.lower().endswith("id")
                    for c in table.columns
                )
                role_hint = ""
                if fk_count >= 2 and has_numeric_measures:
                    role_hint = " [FACT/DETAIL table — has measures + multiple FKs]"
                elif fk_count == 0 and table.row_count < 200:
                    role_hint = " [DIMENSION/LOOKUP table]"
                elif fk_count >= 1 and has_numeric_measures:
                    role_hint = " [TRANSACTION table — individual records with amounts]"

                parts.append(f"TABLE {table.name} ({table.row_count} rows){role_hint}:")
                parts.append(f"  Columns: {columns}")
                for row in rows:
                    parts.append(f"  {list(row)}")

                for col in columns:
                    col_lower = col.lower()
                    # Date/time column ranges
                    if any(kw in col_lower for kw in ("date", "time", "year", "month", "period")):
                        try:
                            rng = conn.execute(
                                f'SELECT MIN("{col}"), MAX("{col}") FROM "{table.name}"'
                            ).fetchone()
                            if rng and rng[0] is not None:
                                parts.append(f"  >> {col} range: {rng[0]} to {rng[1]}")
                        except Exception:
                            pass
                    # ID column lengths
                    if any(kw in col_lower for kw in ("id", "code", "key")):
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

                # Question-aware probing: show distinct values for columns matching question terms
                for col in columns:
                    col_lower = col.lower()
                    is_relevant = any(w in col_lower or col_lower in w for w in q_words if len(w) >= 3)
                    if not is_relevant:
                        continue
                    try:
                        distinct = conn.execute(
                            f'SELECT DISTINCT "{col}" FROM "{table.name}" '
                            f'WHERE "{col}" IS NOT NULL AND "{col}" != \'\' '
                            f'ORDER BY "{col}" LIMIT 8'
                        ).fetchall()
                        vals = [r[0] for r in distinct]
                        if vals:
                            # Detect format patterns
                            format_note = self._detect_value_format(vals)
                            parts.append(f"  >> {col} distinct values: {vals}{format_note}")
                    except Exception:
                        pass

                # Show row count per granularity for tables that look temporal
                if table.row_count > 10 and any(
                    any(kw in c.lower() for kw in ("date", "month", "year"))
                    for c in columns
                ):
                    # Count distinct entities to infer granularity
                    id_cols = [c for c in columns if "id" in c.lower()]
                    date_cols = [c for c in columns if any(kw in c.lower() for kw in ("date", "month", "year"))]
                    if id_cols and date_cols:
                        try:
                            n_entities = conn.execute(
                                f'SELECT COUNT(DISTINCT "{id_cols[0]}") FROM "{table.name}"'
                            ).fetchone()[0]
                            n_dates = conn.execute(
                                f'SELECT COUNT(DISTINCT "{date_cols[0]}") FROM "{table.name}"'
                            ).fetchone()[0]
                            if n_entities and n_dates:
                                rows_per_entity = table.row_count / n_entities
                                parts.append(
                                    f"  >> GRANULARITY: {n_entities} entities × {n_dates} dates, "
                                    f"~{rows_per_entity:.1f} rows/entity"
                                )
                        except Exception:
                            pass

            except Exception:
                continue
        conn.close()
        return "\n".join(parts)

    def _detect_value_format(self, vals: list[Any]) -> str:
        """Detect unusual value formats and return an annotation."""
        if not vals:
            return ""
        str_vals = [str(v) for v in vals if v is not None]
        if not str_vals:
            return ""

        # Time format: "1:36.483" or "01:54:23"
        time_pattern = re.compile(r'^\d{1,2}:\d{2}[\.:]\d{2,3}$')
        if sum(1 for v in str_vals if time_pattern.match(v)) > len(str_vals) * 0.5:
            return " [FORMAT: time string mm:ss.ms — convert to seconds for math]"

        # Relative time: "+16.445" or "+1:02.345"
        if sum(1 for v in str_vals if v.startswith("+")) > len(str_vals) * 0.3:
            return " [FORMAT: relative values with '+' prefix — these are offsets from a reference]"

        # Integer-encoded dates: 201301, 201302
        if all(re.match(r'^\d{6}$', str(v)) for v in str_vals[:5]):
            return " [FORMAT: YYYYMM integer — use range comparison, not string matching]"

        # Status codes: single characters or short codes
        if all(len(str(v)) <= 2 for v in str_vals) and len(set(str_vals)) <= 5:
            return " [FORMAT: categorical/status codes]"

        return ""

    def _try_sql(self, db_path: Path, sql: str) -> dict[str, Any] | None:
        """Execute SQL safely with timeout. Single point of DB execution."""
        if not sql:
            return None
        if not db_path.exists():
            self._log("sql_error", f"DB missing: {db_path}")
            return None
        conn = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=30)
            conn.execute("PRAGMA busy_timeout = 30000")
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
            self._log("sql_error", f"SQL failed: {e}")
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
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
        matched_words: dict[str, list[str]] = {}
        for table in kg.tables:
            for col in table.columns:
                col_lower = col.name.lower()
                for word in q_words:
                    if word == col_lower or (len(word) >= 4 and word in col_lower):
                        match_str = f"{table.name}.{col.name}"
                        matched_words.setdefault(word, []).append(match_str)
                        break

        for word, cols in matched_words.items():
            if len(cols) == 1:
                hints.append(f"  \"{word}\" → {cols[0]}")
            else:
                hints.append(f"  \"{word}\" → AMBIGUOUS: {cols} — check DOMAIN KNOWLEDGE to pick the right one")

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
