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
        if "EXPECTED OUTPUT" in grounding_context:
            rules += "\n- Your query's output shape MUST match EXPECTED OUTPUT for column count. But NEVER use LIMIT 1 just because rows=single — there may be ties or multiple valid matches. Only use LIMIT 1 when combined with ORDER BY for temporal queries (most recent/first)."
        if "DATA FORMAT WARNINGS" in grounding_context:
            rules += "\n- ⚠️ Read DATA FORMAT WARNINGS carefully. Handle time strings, relative values, and encoded formats as described."
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
        parts.append(f"\nVALIDATED PLAN:\n{grounding_context}")

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
- SCOPE: "the X" (singular definite) expecting 1 row but got many = missing filters. Mark incomplete.""")

    return "\n".join(parts)


DOMAIN_ANCHOR_PROMPT = """Given this question and domain knowledge, extract ONLY the definitions and rules that are directly relevant to answering the question.

QUESTION: {question}

DOMAIN KNOWLEDGE:
{knowledge_text}

Return ONLY a JSON object:
{{"anchors": ["exact quote of each relevant definition — include the exact numeric values/mappings"], "column_mappings": {{"question_term": "actual_column_name — meaning"}}, "use_case_sql": "the complete SQL from a USE CASE that answers the same or very similar question, or null if none match"}}

RULES:
- For anchors: quote the EXACT definition including numeric mappings (e.g., "'severe' corresponds to value 2").
- CRITICAL: If ANY word from the question matches a column/field name defined in DOMAIN KNOWLEDGE, you MUST include that definition. For example, if the question mentions "type" and the knowledge defines a "type" field, include it.
- For column_mappings: identify terms in the QUESTION that could map to specific columns. If the knowledge defines what a column represents (e.g., "rank: ranking based on fastest lap time"), and the question uses a related word ("ranked"), record the mapping. This prevents using the wrong column.
- CRITICAL column_mappings cases:
  * "ranked/ranking" → check if knowledge defines a 'rank' column vs 'position' column with different meanings
  * "normal/abnormal" → check if knowledge defines numeric thresholds for that field
  * "time/finish time" → check if knowledge distinguishes different time-related columns
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
  "expected_output": {{"columns": "number of output columns", "rows": "single/multiple/all-matching (use 'all-matching' unless a COUNT/SUM/AVG guarantees exactly 1 row; superlatives and lookups by name may return multiple)", "description": "brief description"}},
  "formula": "the EXACT SQL expression to compute the answer — if GROUND TRUTH defines a metric formula, translate it literally to SQL without simplifying or removing any operations",
  "computation_steps": ["step1: find X", "step2: calculate Y from X"],
  "data_requirements": ["table.column needed for output", "table2.column2 for filter"],
  "data_format_notes": ["any unusual formats from SAMPLE DATA that need handling, e.g., time strings need parsing, relative values with + prefix, integer-encoded dates"],
  "reasoning": "brief HOW to get the answer — must trace back to what_user_wants",
  "domain_rules": ["constraints from DOMAIN KNOWLEDGE that affect the query"],
  "known_values": {{"table.column": ["filter values or expressions verified against SAMPLE DATA"]}},
  "join_paths": ["tableA.col -> tableB.col -> tableC.col"]
}}

RULES:
- Start by understanding what_user_wants — every other field must serve that goal.
- GROUND TRUTH section contains immutable facts extracted from domain knowledge. You MUST follow them exactly.
- EXACT LEVEL MATCHING: When GROUND TRUTH defines distinct named levels (e.g., "high = 1", "medium = 2", "low = 3"), the question's EXACT wording determines WHICH SINGLE level to use. "medium priority" = only the value labeled "medium" (2), NOT "high" (1). Do NOT combine multiple levels unless the question explicitly says "X or above" or "at least X". Each named label maps to exactly one value.
- USE CASE AUTHORITY: If GROUND TRUTH includes a MATCHING USE CASE whose title/explanation directly addresses the same condition as the question, copy its WHERE clause EXACTLY. The use case IS the answer — do not second-guess its filter values.
- For known_values: always include the TABLE name (e.g., "orders.order_date" not just "order_date"). Only use values that exist within that table's SAMPLE DATA range.
- CRITICAL: Check SAMPLE DATA to decide WHICH TABLE to filter. The same column name in different tables may have different formats or data coverage. Always filter the table that actually contains the data you need.
- For formula: if GROUND TRUTH defines a metric formula (e.g., "Metric = X / N"), translate it literally to SQL. Keep ALL parts — do NOT remove operations even if you think the data makes them redundant. The aggregation function must match the question intent ("average" → AVG, "total" → SUM).
- "per unit/per item/each" in the question means a RATIO (total ÷ quantity). Check SAMPLE DATA to determine which column is a total vs a quantity, then use division in the formula.
- For join_paths: trace the FULL FK path shown in DATABASE SCHEMA. Never skip intermediate tables.
- For data_requirements: list ONLY columns needed for output + filters. Do NOT include extra columns.
- Do NOT invent output columns the question didn't ask for. If the question says "list all X", SELECT only the column that identifies X — do NOT add properties (like amount, date) unless the question explicitly asks for them.
- If GROUND TRUTH includes a USE CASE SQL marked as AUTHORITATIVE, follow it EXACTLY — same WHERE values, same columns, same logic. Do NOT override its filter values even if they seem counterintuitive. The use case is the definitive answer pattern.
- If GROUND TRUTH includes a non-authoritative USE CASE SQL, follow its structure but ensure the selected/aggregated column semantically matches what the question asks about.
- POPULATION vs METRIC (critical for percentages/counts): Parse sentence structure carefully:
  * "In X, what is the percentage/count of Y?" → X is the population (WHERE filter = denominator), Y is what you measure.
  * "Among X, how many have Y?" → X is the population, Y is the condition being counted.
  * "Of X, what percentage are Y?" → denominator = COUNT(X), numerator = COUNT(X where Y).
  * WRONG: "In employees with salary > 50000, % managers" → filtering by role='manager' and computing % with salary. CORRECT: filter by salary > 50000, compute % that are managers.
- RATIO LANGUAGE: "How many times is X more than Y?" or "How many times was X more than Y?" = X divided by Y (a ratio). NOT subtraction, NOT a count. Result is a decimal number (e.g., 2.73).
- AGGREGATION GRAIN: When computing AVG/SUM of an entity's own attributes (e.g., "average age of users who..."), aggregate FROM the entity table with a WHERE/IN filter. Do NOT join to a detail table — that duplicates entity rows per detail record and corrupts the average. Example: AVG(users.age) for users with >10 posts → SELECT AVG(age) FROM users WHERE id IN (subquery on posts).
- COLUMN SEMANTICS: If GROUND TRUTH defines column meanings (e.g., "rank = based on fastest lap time", "position = race finish order"), map the question's wording to the CORRECT column. "ranked second" → use the column defined as "ranking", not "position".
- OUTPUT FORMAT: "What is the X and the Y?" or "average X and average Y" = formula must produce TWO SEPARATE columns. The formula should be "SELECT col1, col2 FROM ..." not "SELECT col1 + col2". Each distinct requested value = one column.
- TEMPORAL: "last time" / "most recent" / "latest" → ORDER BY date/time DESC LIMIT 1. "posted it last time" means the most recent poster, not any poster. Include ORDER BY in formula.
- MONTHLY vs YEARLY: If data stores one row PER MONTH (e.g., monthly_stats table with 12 rows per year per entity), "average monthly X" = AVG(value). If data stores ANNUAL totals (one row per year), "average monthly" = AVG(value) / 12. Check SAMPLE DATA row count vs time range to determine granularity.
- DATA FORMAT INSPECTION: Look at SAMPLE DATA's "distinct values" annotations. If values have FORMAT tags (e.g., [FORMAT: time string mm:ss.ms]), record this in data_format_notes. Your formula must handle these formats (e.g., convert time strings to seconds before doing math).
- GRANULARITY: Check SAMPLE DATA's GRANULARITY annotations. "~12 rows/entity" with monthly dates = monthly data. "~1 row/entity" with yearly = annual data. This determines whether to divide by 12 or not.
- HAVING vs WHERE: "where the average X exceeds N" or "schools where the average exceeds N" = this is a GROUP-level filter. The formula must use GROUP BY + HAVING, NOT a per-row WHERE clause. The GROUP BY groups by the entity (school, district, etc.), and HAVING filters groups by their aggregate.
- PER-GROUP POSITIONAL: "the Nth item of EACH group" needs ROW_NUMBER() OVER (PARTITION BY group ORDER BY position). Do NOT use global LIMIT/OFFSET.
- SUPERLATIVE TIES: "which has the lowest/highest" → set expected_output.rows = "all-matching" (NOT "single"). Use WHERE col = (SELECT MIN/MAX...) to get ALL ties. NEVER use LIMIT 1 for superlatives. Multiple rows sharing the same min/max are ALL correct answers.
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
   - If GROUND TRUTH has an AUTHORITATIVE USE CASE SQL, the plan's known_values MUST match that SQL's WHERE clause EXACTLY. Do NOT change them. The use case defines the correct values even if they seem wrong.
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
5. POPULATION vs METRIC (critical for percentages):
   - Re-read the question: "In X, what % of Y?" means X=WHERE filter, Y=numerator condition.
   - Check: is the plan filtering by the correct population (denominator)?
   - WRONG: "In employees salary > 50000, % managers" with WHERE role='manager' and CASE on salary.
   - CORRECT: WHERE salary > 50000, then COUNT(managers)/COUNT(*).
6. RATIO vs DIFFERENCE:
   - "How many times X more than Y" = X/Y (division), NOT X-Y.
   - Check if the formula uses division when the question asks "how many times".
7. AGGREGATION GRAIN:
   - If the plan computes AVG of entity attributes via JOIN to detail table, it's WRONG — the join duplicates rows.
   - AVG(user.age) for "users with >10 posts" must query users table directly, not join through posts.
8. COLUMN SEMANTICS:
   - If GROUND TRUTH COLUMN MAPPINGS exist, verify the plan uses the correct column for each term.
   - E.g., "ranked second" must use the column defined as "ranking" not "position" if they differ.
9. OUTPUT FORMAT:
   - "What is the X and the Y?" → formula must SELECT two columns, NOT combine them.
   - "last time" / "most recent" → formula must ORDER BY date/time DESC.
   - If data is monthly rows and question asks "average monthly", formula should be AVG(col) not AVG(col)/12.
   - If data is yearly totals and question asks "average monthly", formula should divide by 12.
10. HAVING vs WHERE:
   - "where the average X exceeds N" = GROUP BY + HAVING AVG(X) > N, NOT per-row WHERE X > N.
   - Verify: does the formula use GROUP BY + HAVING for group-level aggregation filters?
11. SUPERLATIVE TIES:
   - "lowest/highest" → formula must use WHERE col = (SELECT MIN/MAX(...)) to return ALL ties.
   - If formula uses ORDER BY + LIMIT 1, it will miss ties — mark as needs_fix.

Return ONLY a JSON object:
{{"verdict": "correct"/"needs_fix", "fixed_known_values": {{"column_name": ["corrected_values"]}}, "fixed_data_requirements": ["table.column", ...], "fixed_join_paths": ["tableA.col -> tableB.col"], "fixed_formula": "corrected formula if wrong", "reasoning": "one sentence explaining what was wrong"}}

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

    # Expected output shape
    expected_output = grounding.get("expected_output", {})
    if expected_output:
        shape_desc = expected_output.get("description", "")
        n_cols = expected_output.get("columns", "")
        n_rows = expected_output.get("rows", "")
        shape_parts = []
        if n_cols:
            shape_parts.append(f"columns={n_cols}")
        if n_rows:
            shape_parts.append(f"rows={n_rows}")
        if shape_desc:
            shape_parts.append(shape_desc)
        if shape_parts:
            parts.append(f"EXPECTED OUTPUT: {', '.join(shape_parts)}")

    # Data format notes
    format_notes = grounding.get("data_format_notes", [])
    if format_notes:
        parts.append("DATA FORMAT WARNINGS:\n" + "\n".join(f"  ⚠️ {n}" for n in format_notes))

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
            grounding_context = self._call_semantic_grounding(
                question, kg_context, sample_data, ctx.knowledge_text,
                db_path=db_path,
            )

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
                    question, kg_context, sample_data,
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

                # Skip evaluate on last iteration — just use what we have
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
                    question, db_path, kg_context, sample_data,
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
                    question, data_result, db_path, kg_context, sample_data,
                    ctx.knowledge_text, grounding_context, col_hints,
                )

            # Python fallback: when SQL fails entirely, let LLM write Python
            if not data_result or not data_result.get("rows"):
                py_result = self._try_python_fallback(
                    question, db_path, kg_context, sample_data,
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
            deterministic_parts.append(
                f"MATCHING USE CASE (score={best_score}):\n"
                f"  Title: {best_use_case[0]}\n"
                f"  SQL: {best_use_case[1]}\n"
                f"  Explanation: {best_use_case[2]}\n"
                f"  ⚠️ THIS USE CASE closely matches your question — follow its WHERE values and logic."
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

            column_mappings = parsed.get("column_mappings", {})
            if column_mappings:
                llm_parts.append("\nCOLUMN MAPPINGS (use the correct column for each question term):")
                for term, mapping in column_mappings.items():
                    llm_parts.append(f"  \"{term}\" → {mapping}")

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
            fixed_formula = val_result.get("fixed_formula", "")

            # Detect oscillation: if validator reverses a previous fix, prefer
            # question-literal values verified against DB over sample-data substitutions
            if fixed_kv and prev_fixed_kv:
                oscillating = False
                for col, vals in fixed_kv.items():
                    if col in prev_fixed_kv and set(str(v) for v in vals) != set(str(v) for v in prev_fixed_kv[col]):
                        oscillating = True
                        break
                if oscillating:
                    # Resolve oscillation: extract values from question and verify in DB
                    question_values = re.findall(r"'([^']+)'|\"([^\"]+)\"", question)
                    q_vals = [v[0] or v[1] for v in question_values]
                    if q_vals and db_path:
                        conn = sqlite3.connect(str(db_path))
                        try:
                            for qv in q_vals:
                                for col_key in list(grounding.get("known_values", {}).keys()):
                                    bare_col = col_key.split(".")[-1] if "." in col_key else col_key
                                    # Find tables with this column and check if value exists
                                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                                        tname = row[0]
                                        cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                                        col_match = next((c for c in cols if c.lower() == bare_col.lower()), None)
                                        if col_match:
                                            try:
                                                cnt = conn.execute(
                                                    f'SELECT COUNT(*) FROM "{tname}" WHERE "{col_match}" = ?', (qv,)
                                                ).fetchone()[0]
                                                if cnt > 0:
                                                    grounding["known_values"][col_key] = [qv]
                                                    self._log("oscillation_resolved",
                                                              f"Question value '{qv}' exists in {tname}.{col_match} ({cnt} rows) — using it")
                                                    break
                                            except Exception:
                                                pass
                        finally:
                            conn.close()
                    self._log("grounding_oscillation", "Validator reversed previous fix — resolved with question-literal values")
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
            if fixed_formula:
                grounding["formula"] = fixed_formula
                self._log("grounding_fix_formula", fixed_formula)

            # If validator didn't suggest any usable fixes, no point looping
            if not fixed_kv and not fixed_dr and not fixed_jp and not fixed_formula:
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
        """When SQL returns 0 rows, isolate which filter causes the empty result."""
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
                        "JOIN itself returns 0 rows — check JOIN conditions and table relationships."
                    )
                    # Show available tables and their columns
                    tables = [r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()]
                    for tname in tables:
                        if tname.lower() in sql.lower() or tname.startswith("_"):
                            continue
                        cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                        diagnostics.append(f"  Available table '{tname}': columns={cols}")
                    return "EMPTY RESULT DIAGNOSIS:\n" + "\n".join(diagnostics)
            except Exception:
                pass

            # Extract individual WHERE conditions and test each removal
            where_clause = sql[where_idx + 5:].strip()
            for keyword in ["ORDER BY", "GROUP BY", "LIMIT", "HAVING"]:
                kw_idx = where_clause.upper().find(keyword)
                if kw_idx > 0:
                    where_clause = where_clause[:kw_idx].strip()

            conditions = [c.strip() for c in re.split(r'\bAND\b', where_clause, flags=re.IGNORECASE) if c.strip()]

            if len(conditions) >= 1:
                for i, cond in enumerate(conditions):
                    remaining = [c for j, c in enumerate(conditions) if j != i]
                    test_sql = f"{base_sql} WHERE {' AND '.join(remaining)}" if remaining else base_sql
                    try:
                        count = conn.execute(
                            f"SELECT COUNT(*) FROM ({test_sql})"
                        ).fetchone()[0]
                        if count > 0:
                            diagnostics.append(
                                f"BLOCKER: filter '{cond.strip()}' eliminates all rows. Without it: {count} rows."
                            )
                            # Extract the column and value from the condition
                            # Try to find table.col or col references
                            col_found = False
                            for part in cond.replace("(", " ").replace(")", " ").split():
                                col_ref = part.strip("\"'`=<>!,")
                                if not col_ref or col_ref.upper() in ("AND", "OR", "NOT", "IN", "LIKE", "IS", "NULL", "BETWEEN"):
                                    continue
                                if "." in col_ref:
                                    _, col = col_ref.split(".", 1)
                                    col = col.strip("\"'`")
                                else:
                                    col = col_ref
                                try:
                                    actual = conn.execute(
                                        f'SELECT DISTINCT "{col}" FROM ({test_sql}) WHERE "{col}" IS NOT NULL AND "{col}" != \'\' LIMIT 15'
                                    ).fetchall()
                                    vals = [r[0] for r in actual]
                                    if vals:
                                        diagnostics.append(
                                            f"  ACTUAL values for '{col}': {vals}"
                                        )
                                        # Extract what value was used in the filter
                                        str_match = re.findall(r"'([^']*)'", cond)
                                        if str_match:
                                            used_val = str_match[0]
                                            # Check for close matches (LIKE pattern)
                                            close = [v for v in vals if isinstance(v, str) and (
                                                used_val.lower() in v.lower() or v.lower() in used_val.lower()
                                            )]
                                            if close:
                                                diagnostics.append(f"  SUGGESTION: Use LIKE '%{used_val}%' or exact match '{close[0]}'")
                                            else:
                                                diagnostics.append(f"  Your filter used '{used_val}' but it does NOT exist. Use one of the actual values above.")
                                        col_found = True
                                        break
                                except Exception:
                                    continue
                            if not col_found:
                                # Show all columns of tables in the base query
                                diagnostics.append(f"  Could not identify column. Check if the column name is correct.")
                    except Exception:
                        pass

            # If single condition and no diagnosis yet, show what exists
            if not diagnostics and len(conditions) == 1:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM ({base_sql})").fetchone()[0]
                    if count > 0:
                        diagnostics.append(f"Base query has {count} rows but the single filter eliminates all.")
                        # Show column values from tables in the query
                        tables = [r[0] for r in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()]
                        for tname in tables:
                            if tname.lower() in sql.lower():
                                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                                for col in cols:
                                    if col.lower() in conditions[0].lower():
                                        sample = conn.execute(
                                            f'SELECT DISTINCT "{col}" FROM "{tname}" WHERE "{col}" IS NOT NULL LIMIT 10'
                                        ).fetchall()
                                        vals = [r[0] for r in sample]
                                        diagnostics.append(f"  {tname}.{col} actual values: {vals}")
                except Exception:
                    pass

            # Fallback: show sample values for columns that appear in WHERE clause
            if not diagnostics:
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                for tname in tables:
                    if tname.lower() not in sql.lower():
                        continue
                    cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                    for col in cols:
                        if col.lower() in where_clause.lower():
                            try:
                                sample = conn.execute(
                                    f'SELECT DISTINCT "{col}" FROM "{tname}" '
                                    f'WHERE "{col}" IS NOT NULL AND "{col}" != \'\' LIMIT 10'
                                ).fetchall()
                                vals = [r[0] for r in sample]
                                diagnostics.append(f"{tname}.{col}: actual values={vals}")
                            except Exception:
                                pass
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
                parts.append(f"TABLE {table.name} ({table.row_count} rows):")
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
