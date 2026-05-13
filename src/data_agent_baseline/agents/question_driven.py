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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.pipeline.context_scanner import TaskContext, scan_context
from data_agent_baseline.pipeline.kg_builder import (
    KnowledgeGraph,
    build_kg_from_sqlite,
    enrich_kg_with_descriptions,
    format_kg_for_llm,
)
from data_agent_baseline.tools.knowledge_graph import consolidate_to_sqlite

logger = logging.getLogger(__name__)

CONSOLIDATED_DB_NAME = "_consolidated.db"


# ---------------------------------------------------------------------------
# Prompt builders — dynamic, only include sections that have content
# ---------------------------------------------------------------------------

# Rules indexed by short label for LLM-based selection
SQL_RULES_LABELED: list[tuple[str, str]] = [
    ("exact_question", "Answer the EXACT question. SELECT only asked columns. No SELECT *."),
    ("schema_only", "Use only tables and columns shown in the schema."),
    ("date_format", "Check data for date ranges/formats before WHERE clauses. Same column name in different tables may have DIFFERENT formats."),
    ("join_fk", "JOIN through the full FK path. Never skip intermediate linking tables."),
    ("text_match", "Use LIKE '%keyword%' COLLATE NOCASE for text. Use CAST(x AS REAL) for division."),
    ("human_readable", "If question asks for names/descriptions, JOIN to get human-readable values instead of raw IDs."),
    ("select_minimal", "Only SELECT columns the question explicitly asks for. Do NOT add extra columns."),
    ("superlative", "For superlatives (lowest/highest/most/least), use WHERE col = (SELECT MIN/MAX(col) FROM ... WHERE col IS NOT NULL AND col != '') — NEVER LIMIT unless domain knowledge SQL shows LIMIT. Return ALL ties."),
    ("null_minmax", "When using MIN/MAX on text columns, exclude empty strings: WHERE col != '' AND col IS NOT NULL."),
    ("null_column", "If a column is often NULL, comparisons return nothing for NULL rows. Consider a DIFFERENT column or table."),
    ("distinct", "If question asks 'which/what X', use SELECT DISTINCT to avoid duplicates."),
    ("apostrophe", "Escape apostrophes: use '' (double single-quote) inside SQL strings."),
    ("agg_column", "For COUNT/SUM/aggregations: the aggregated column must semantically match what the question asks about."),
    ("ratio", "When question says 'per unit/per item/each', compute a ratio (total ÷ count)."),
    ("no_group_unless", "Do NOT use GROUP BY + aggregate unless question asks for totals/averages. 'lowest X' = MIN/MAX of individual rows."),
    ("population", "POPULATION vs METRIC: 'In X, what is Y?' → X is WHERE filter, Y is what you compute ON that set."),
    ("ratio_lang", "'How many times X more than Y' = X/Y (division). NOT subtraction, NOT count."),
    ("agg_grain", "AGGREGATION GRAIN: AVG/SUM of entity attribute → query entity table DIRECTLY with subquery filter. Do NOT join to detail tables (duplicates rows)."),
    ("domain_col", "DOMAIN KNOWLEDGE COLUMN MAPPING: Use the EXACT column whose DEFINITION matches the question's intent."),
    ("value_format", "VALUE FORMAT: Match the EXACT format shown in data (time strings, integer dates, display names)."),
    ("recovery", "EMPTY RESULT RECOVERY: If PREVIOUS ATTEMPT shows actual DB values, use THOSE exact values."),
    ("multi_col", "OUTPUT COLUMNS: 'What is X and Y?' = TWO SEPARATE columns. Do NOT combine with + or concat."),
    ("singular_plural", "SINGULAR: Don't add LIMIT 1 just because grammar is singular. Only LIMIT 1 for 'most recent'/'first'."),
    ("temporal", "TEMPORAL: 'last time'/'most recent' = ORDER BY DESC LIMIT 1. 'first time' = ORDER BY ASC LIMIT 1."),
    ("monthly_yearly", "MONTHLY from YEARLY: If DOMAIN KNOWLEDGE defines a formula with /12, ALWAYS include /12 in the SQL. Formula is authoritative over data granularity."),
    ("time_parse", "TIME STRING: '1:36.483' → CAST(SUBSTR)*60 + CAST(SUBSTR) for seconds conversion. ALWAYS filter WHERE col IS NOT NULL AND col != '' before converting. If domain knowledge provides an exact SQL example for this question, use ORDER BY col ASC LIMIT 1 instead of conversion."),
    ("no_null", "NEVER RETURN NULL: Wrap with COALESCE or add WHERE IS NOT NULL. NULL answer = wrong."),
    ("having", "HAVING vs WHERE: 'where the average exceeds N' = GROUP BY + HAVING, not per-row WHERE."),
    ("positional", "PER-GROUP POSITIONAL: 'Nth of each group' = ROW_NUMBER() OVER (PARTITION BY ...)."),
    ("intersection", "INTERSECTION: 'X with Y containing Z' = two subqueries intersected, NOT one WHERE."),
    ("colocated", "CO-LOCATED MEASURES: Filter column in detail table → use measure from SAME detail table, not parent summary."),
    ("same_name_col", "SAME-NAME COLUMNS: When multiple tables share a column name, read the question language to pick the right table. 'the patient's X' → Patient.X, not DetailTable.X."),
    ("col_name_match", "COLUMN NAME MATCH: If question mentions a specific term (e.g. 'up votes') and a column with that exact name exists (e.g. users.UpVotes), USE that column — not a semantically similar column from another table."),
    ("per_unit", "PER UNIT: 'paid X per unit' or 'X per item' = Price / Amount (or equivalent ratio). Do NOT use raw Price."),
]

# Full rule texts for backward compatibility
SQL_RULES = [rule for _, rule in SQL_RULES_LABELED]


def _build_sql_prompt(
    *,
    question: str,
    kg_context: str,
    sample_data: str = "",  # unused — grounding + schema_slice provide all format info
    knowledge_text: str = "",
    column_hints: str = "",
    gaps: str = "",
    extra_context: str = "",
    grounding_context: str = "",
    selected_rules: str = "",
) -> str:
    parts = [f"QUESTION: {question}\n\nWrite a SQL query to answer the QUESTION above."]

    parts.append(f"\nDATABASE SCHEMA:\n{kg_context}")

    if grounding_context:
        parts.append(f"\n{grounding_context}")
    elif knowledge_text:
        parts.append(f"\nDOMAIN KNOWLEDGE:\n{knowledge_text[:1000]}")

    if column_hints:
        parts.append(f"\n{column_hints}")

    if gaps:
        parts.append(f"\nPREVIOUS ATTEMPT FAILED — fix these issues:\n{gaps}")

    if extra_context:
        parts.append(f"\nEXPLORATORY RESULTS:\n{extra_context}")

    has_gaps = bool(gaps)

    # Use LLM-selected rules if available, otherwise fallback
    if selected_rules:
        rules = selected_rules
    else:
        rules = """- Answer the EXACT question. SELECT only asked columns. No SELECT *.
- Use FILTER VALUES exactly as given. Do NOT substitute.
- For superlatives (lowest/highest), use WHERE col = (SELECT MIN/MAX(col)...) — no LIMIT.
- JOIN through FK paths shown in schema.
- Use LIKE '%X%' COLLATE NOCASE for text. CAST(x AS REAL) for division.
- NEVER RETURN NULL — add WHERE IS NOT NULL. Escape apostrophes with ''."""

    # Always-on rules
    rules += "\n- NEVER use AS to rename columns. SELECT the original column name directly (e.g. SELECT name, NOT SELECT name AS race_name)."
    rules += "\n- TIME STRINGS: If a column contains time values like '1:36.483', ALWAYS add WHERE col IS NOT NULL AND col != '' first. Then either convert to seconds (CAST(SUBSTR(col,1,INSTR(col,':')-1) AS REAL)*60 + CAST(SUBSTR(col,INSTR(col,':')+1) AS REAL)) or simply ORDER BY col ASC/DESC (string sort works for consistent m:ss.ms format)."

    # Append context-specific rules
    if grounding_context:
        if has_gaps:
            rules += "\n- PREVIOUS ATTEMPT FAILED feedback takes PRIORITY over GROUNDING CONTEXT. Fix what the feedback says is wrong."
        else:
            if "FILTER VALUES" in grounding_context:
                rules += "\n- ⚠️ MANDATORY: Your WHERE clause MUST use the values from FILTER VALUES above."
        if "DATA FORMAT WARNINGS" in grounding_context:
            rules += "\n- ⚠️ Read DATA FORMAT WARNINGS carefully."
        if "MANDATORY CORRECTIONS" in grounding_context:
            rules += "\n- ⚠️ APPLY ALL items under MANDATORY CORRECTIONS — they override the FORMULA."
        if "CONSTRAINTS" in grounding_context and "/" in grounding_context:
            rules += "\n- ⚠️ FORMULA AUTHORITY: If CONSTRAINTS has arithmetic (/ 12, * 100), include it EXACTLY in your SQL."
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
{{"anchors": ["exact quote of each relevant definition — include the exact numeric values/mappings"]}}

RULES:
- Quote the EXACT definition including numeric mappings (e.g., "'severe' corresponds to value 2").
- If ANY word from the question matches a column/field name defined in DOMAIN KNOWLEDGE, include that definition.
- Include definitions that DISTINGUISH between similar columns (e.g., "rank: fastest lap ranking" vs "position: race finish order").
- If a definition distinguishes between similar terms (e.g., "most severe = 1" vs "severe = 2"), quote BOTH.
- Be precise and complete — these anchors will be used as domain context for query planning.
""".strip()

SEMANTIC_GROUNDING_PROMPT = """QUESTION: {question}

DATABASE SCHEMA:
{kg_context}
{sample_section}
{anchor_section}
{previous_attempt}
Decompose the question into a structured plan.

FIRST, break down the question phrase by phrase:
- For EACH noun/phrase the user asks for, identify which SPECIFIC table.column it maps to.
- "type of X" → does a column literally named "type" exist? In which table? Is it a label on the entity or a GROUP BY dimension?
- "total value" / "total cost" → SUM of which column?
- "for event X" → filter condition on which table?
- Distinguish lookup vs aggregation: "Identify the type" = SELECT type (lookup). "for each type" / "by type" = GROUP BY.

Return ONLY a JSON object:
{{
  "what_user_wants": "restate EXACTLY what output the user expects — only columns explicitly mentioned",
  "phrase_mapping": {{"quoted phrase from question": "table.column it maps to — with reasoning if ambiguous"}},
  "expected_output": {{"columns": "number", "description": "brief"}},
  "computation_type": "one of: simple_lookup | count | count_distinct | sum | avg | ratio | percentage | min_max | comparison | multi_step",
  "formula": "plain-language formula — WHAT to compute, not SQL. e.g. 'event with lowest cost', 'AVG(Consumption) / 12'",
  "computation_steps": ["step1", "step2"],
  "data_requirements": ["table.column — ALL columns relevant to question, joins, filters, aggregation"],
  "data_format_notes": ["unusual formats needing handling"],
  "reasoning": "brief HOW to get the answer",
  "domain_rules": ["constraints from DOMAIN KNOWLEDGE"],
  "known_values": {{"table.column": ["verified filter values"]}}
}}

RULES:
- what_user_wants drives everything. Do NOT invent columns the question didn't ask for.
- NO ASSUMPTIONS: Do NOT add data_format_notes or domain_rules that assume conversion/transformation unless DOMAIN KNOWLEDGE explicitly states it. If a REFERENCE SQL is provided, follow its approach exactly (e.g. if it uses ORDER BY col ASC, do NOT add "must convert to seconds").
- FORMULA AUTHORITY: If DOMAIN KNOWLEDGE defines a formula, copy it VERBATIM into "formula" field. Do NOT reason about whether any part is redundant — every operation is intentional.
- USE CASE AUTHORITY: If DOMAIN KNOWLEDGE has a matching USE CASE or REFERENCE SQL, copy its logic EXACTLY.
- EXACT LEVEL MATCHING: Named levels ("high=1", "medium=2") → use ONLY the level matching the question's exact wording. No combining unless "X or above".
- COLUMN SEMANTICS: If DOMAIN KNOWLEDGE defines column meanings, use the column whose DEFINITION matches the question intent. Definition IS authoritative.
- For known_values: include TABLE name. Only use values within SAMPLE DATA range.
- Check SAMPLE DATA to decide WHICH TABLE to filter (same column name may differ across tables).
- join_paths will be computed automatically — just list ALL tables.columns needed in data_requirements.
- For data_requirements: be INCLUSIVE — list every column that could help.
- POPULATION vs METRIC: "In X, what is Y?" → X = WHERE filter, Y = what you compute on that set.
- COMPUTATION TYPE RULES:
  - "how many times X compared to Y" / "how many times more" = ratio (X/Y division)
  - "what percentage" / "what proportion" = percentage (X * 100 / Y)
  - "how many distinct/unique" = count_distinct
  - "how many" (simple) = count
  - "average/mean" = avg
  - "total/sum" = sum
  - "highest/lowest/best/worst" = min_max
  - "how much faster/slower/more/less" = comparison (compute difference or ratio)
- RATIO: "How many times X more than Y" = X/Y (division, not subtraction).
- AGGREGATION GRAIN: AVG of entity attribute → query entity table with subquery filter. Don't join to detail tables (duplicates).
- OUTPUT: "X and Y?" = TWO columns. Each requested value = one column.
- GRAIN CONSISTENCY: All requested outputs must be at the same level of detail. If the question does NOT say "for each", "per", "by", or "breakdown", assume all outputs are at the SAME grain (no GROUP BY on any of them). Only GROUP BY when the question explicitly asks for a per-group result.
- TEMPORAL: "last/most recent" = ORDER BY DESC LIMIT 1.
- MONTHLY vs YEARLY: If DOMAIN KNOWLEDGE defines a formula with /12, ALWAYS include /12. Do NOT skip it based on data granularity.
- HAVING: "where the average exceeds N" = GROUP BY + HAVING, not per-row WHERE.
- SUPERLATIVES: "lowest/highest" → rows = "all-matching". Use WHERE col = (SELECT MIN/MAX...). NEVER LIMIT 1.
- CO-LOCATED MEASURES: Filter in detail table → use measure from SAME detail table, not parent summary.
- COLUMN NAME PRIORITY: If a column name in ANY table exactly matches a word in the question, prefer it over semantically similar columns in other tables. Literal name match > inferred meaning.
- PARTIAL MATCH: "X-related" / "from X" with proper noun → LIKE '%X%'. Exact match only for "named X" / "is X".
- SAME-NAME COLUMNS: When multiple tables have the SAME column name, read the question language to decide which table to SELECT from. "the patient's X" → Patient.X. "the exam's X" → Exam.X. The subject/possessive in the question is authoritative.
- PHRASE MAPPING: "the type of X" / "identify the type" = SELECT the literal `type` column of the entity being asked about. This is a LOOKUP, not a GROUP BY. Only use GROUP BY if the question says "for each type" / "by type" / "per type" / "breakdown".
""".strip()

def _trim_schema_by_relevance(kg_context: str, question: str, budget: int, anchor_text: str = "") -> str:
    """Trim schema using Steiner tree: keep only tables needed to connect question-relevant tables.

    Graph: tables = nodes, FK relationships = edges.
    Terminals: tables whose name or columns overlap with question keywords.
    Also uses anchor_text (domain knowledge) to bridge semantic gaps between
    question language and column names (e.g. "disease" → Diagnosis column).
    Skips trimming entirely for small schemas (≤5 tables).
    """
    if len(kg_context) <= budget:
        return kg_context

    # Split into table blocks
    blocks: dict[str, str] = {}  # table_name -> block_text
    current_name = ""
    current_lines: list[str] = []
    for line in kg_context.split("\n"):
        if line.startswith("TABLE: "):
            if current_lines and current_name:
                blocks[current_name] = "\n".join(current_lines)
            current_name = line.split("TABLE: ")[1].split(" ")[0].split("(")[0].strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines and current_name:
        blocks[current_name] = "\n".join(current_lines)

    if not blocks:
        return kg_context[:budget]

    all_tables = list(blocks.keys())

    # Small schemas: skip Steiner tree, just include all with column cap if needed
    if len(all_tables) <= 5:
        result_parts: list[str] = []
        total = 0
        for name in all_tables:
            text = blocks[name]
            if total + len(text) + 1 <= budget:
                result_parts.append(text)
                total += len(text) + 1
            else:
                lines = text.split("\n")
                trimmed = []
                col_count = 0
                for ln in lines:
                    if ln.startswith("  ") and not ln.strip().startswith("FK:") and not ln.strip().startswith("PK:"):
                        col_count += 1
                        if col_count > 12:
                            continue
                    trimmed.append(ln)
                if col_count > 12:
                    trimmed.append(f"  ... ({col_count - 12} more columns)")
                short_text = "\n".join(trimmed)
                if total + len(short_text) + 1 <= budget:
                    result_parts.append(short_text)
                    total += len(short_text) + 1
        return "\n".join(result_parts) if result_parts else kg_context[:budget]

    # Build adjacency graph from FK lines in schema
    graph: dict[str, set[str]] = {t: set() for t in all_tables}
    for name, text in blocks.items():
        for line in text.split("\n"):
            if "FK:" in line or "->" in line:
                for other in all_tables:
                    if other != name and other.lower() in line.lower():
                        graph[name].add(other)
                        graph[other].add(name)

    # Identify terminal nodes via keyword overlap with question
    # Use only nouns/meaningful words (exclude stop words)
    stop_words = {"the", "and", "for", "are", "was", "were", "that", "this", "with",
                  "from", "have", "has", "had", "been", "will", "would", "could",
                  "their", "them", "they", "what", "which", "who", "how", "many",
                  "much", "each", "every", "all", "any", "list", "find", "show",
                  "give", "name", "number", "total", "average", "count"}
    q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower())) - stop_words
    terminals: set[str] = set()

    def _stem_match(word_set_a: set[str], word_set_b: set[str]) -> bool:
        """Check if any word in A matches any word in B (exact or stem-substring for 4+ char words)."""
        for a in word_set_a:
            for b in word_set_b:
                if a == b:
                    return True
                if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
                    return True
        return False

    # Track match strength: name-match > column-match > anchor-match
    name_matched: set[str] = set()
    col_matched: set[str] = set()

    # Match 1: question words vs table names and column names
    for name, text in blocks.items():
        # Table name match (strongest signal)
        name_words = set(re.findall(r'[a-z]{3,}', name.lower()))
        if _stem_match(q_words, name_words):
            terminals.add(name)
            name_matched.add(name)
            continue
        # Column name match (weaker — FK columns like player_id match broadly)
        for line in text.split("\n"):
            if line.startswith("  ") and not line.strip().startswith("FK:") and not line.strip().startswith("PK:"):
                col_part = line.strip().split("(")[0].split(":")[0].strip().lower() if "(" in line or ":" in line else line.strip().lower()
                col_words = set(re.findall(r'[a-z]{3,}', col_part))
                if _stem_match(q_words, col_words):
                    terminals.add(name)
                    col_matched.add(name)
                    break

    # Match 2: use anchor_text to bridge semantic gaps
    # Anchor text contains natural language descriptions that may match question words
    # Map those descriptions back to table names mentioned nearby
    anchor_matched: set[str] = set()
    if anchor_text and len(terminals) < len(all_tables):
        anchor_lower = anchor_text.lower()
        for word in q_words:
            if word in anchor_lower:
                for anchor_line in anchor_text.split("\n"):
                    if word in anchor_line.lower():
                        for tbl in all_tables:
                            if tbl.lower() in anchor_line.lower() or tbl.lower() + "." in anchor_line.lower():
                                terminals.add(tbl)
                                if tbl not in name_matched and tbl not in col_matched:
                                    anchor_matched.add(tbl)

    # If no terminals found, fall back to all tables
    if not terminals:
        terminals = set(all_tables)

    # Steiner tree approximation: BFS shortest path between all terminal pairs,
    # union of paths = set of required tables
    from collections import deque

    terminal_list = list(terminals)

    def bfs_path(start: str, end: str) -> list[str]:
        if start == end:
            return [start]
        visited = {start}
        queue: deque[list[str]] = deque([[start]])
        while queue:
            path = queue.popleft()
            for neighbor in graph.get(path[-1], set()):
                if neighbor == end:
                    return path + [end]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []  # no path (disconnected)

    steiner_tables: set[str] = set(terminal_list)
    for i, t1 in enumerate(terminal_list):
        for t2 in terminal_list[i + 1:]:
            path = bfs_path(t1, t2)
            steiner_tables.update(path)

    if not steiner_tables:
        steiner_tables = set(all_tables)

    # Build result priority: name-matched > anchor-matched > col-matched > connectors > rest
    # Within each tier, smaller tables first to maximize coverage
    connectors = steiner_tables - terminals
    tier_name = sorted([t for t in all_tables if t in name_matched], key=lambda t: len(blocks[t]))
    tier_anchor = sorted([t for t in all_tables if t in anchor_matched], key=lambda t: len(blocks[t]))
    tier_col = sorted([t for t in all_tables if t in col_matched and t not in name_matched and t not in anchor_matched], key=lambda t: len(blocks[t]))
    tier_conn = sorted([t for t in all_tables if t in connectors], key=lambda t: len(blocks[t]))
    remaining_tables = [t for t in all_tables if t not in steiner_tables]
    ordered = tier_name + tier_anchor + tier_col + tier_conn

    result_parts = []
    total = 0

    for name in ordered + remaining_tables:
        text = blocks[name]
        if total + len(text) + 1 <= budget:
            result_parts.append(text)
            total += len(text) + 1
        else:
            # Try column-trimmed version (12 cols max)
            lines = text.split("\n")
            trimmed = []
            col_count = 0
            for ln in lines:
                if ln.startswith("  ") and not ln.strip().startswith("FK:") and not ln.strip().startswith("PK:"):
                    col_count += 1
                    if col_count > 12:
                        continue
                trimmed.append(ln)
            if col_count > 12:
                trimmed.append(f"  ... ({col_count - 12} more columns)")
            short_text = "\n".join(trimmed)
            if total + len(short_text) + 1 <= budget:
                result_parts.append(short_text)
                total += len(short_text) + 1

    return "\n".join(result_parts) if result_parts else kg_context[:budget]


def _build_semantic_prompt(
    *,
    question: str,
    kg_context: str,
    sample_data: str = "",
    anchor_text: str = "",
    previous_attempt: str = "",
    feedback: str = "",
) -> str:
    # Budget: keep total prompt under 8000 chars to avoid Qwen timeouts
    BUDGET = 8000
    template_len = len(SEMANTIC_GROUNDING_PROMPT) + len(question) * 2
    remaining = BUDGET - template_len - len(anchor_text[:1500]) - len(previous_attempt)

    # Schema gets priority, sample data fills remainder
    schema_budget = max(remaining - 1500, 2000)
    kg_trimmed = _trim_schema_by_relevance(kg_context, question, schema_budget, anchor_text)
    sample_budget = max(remaining - len(kg_trimmed), 500)
    sample_section = f"\nSAMPLE DATA:\n{sample_data[:sample_budget]}" if sample_data else ""
    anchor_section = f"\nDOMAIN KNOWLEDGE:\n{anchor_text[:1500]}" if anchor_text else ""
    feedback_section = f"\n⚠️ CORRECTION REQUIRED:\n{feedback}" if feedback else ""
    prev_section = f"\nPREVIOUS ATTEMPT (fix the issues below):\n{previous_attempt}{feedback_section}" if previous_attempt else ""
    return SEMANTIC_GROUNDING_PROMPT.format(
        question=question,
        kg_context=kg_trimmed,
        sample_section=sample_section,
        anchor_section=anchor_section,
        previous_attempt=prev_section,
    )



def _format_grounding_for_sql(grounding: dict[str, Any]) -> str:
    """Format grounding as factual context for SQL planner — no SQL, no steps, no approach."""
    parts: list[str] = []

    # Semantic overrides FIRST — highest priority, planner must see these before anything else
    override_rules = grounding.get("_semantic_overrides", [])
    if override_rules:
        parts.append("⚠️ MANDATORY CORRECTIONS (you MUST apply these to your SQL):\n" + "\n".join(f"  - {r}" for r in override_rules))

    # User intent and expected output shape
    user_wants = grounding.get("what_user_wants", "")
    expected_output = grounding.get("expected_output", {})
    computation_type = grounding.get("computation_type", "")
    if user_wants:
        parts.append(f"USER WANTS: {user_wants}")
    if computation_type:
        parts.append(f"COMPUTATION TYPE: {computation_type} (your SQL MUST produce this type of result)")
    if expected_output:
        col_expect = expected_output.get("columns", "")
        if col_expect:
            parts.append(f"EXPECTED COLUMNS: {col_expect}")

    # Reference formula — the planner should follow this structure
    # Skip if override contradicts formula (population rule or formula-correcting override)
    has_population_override = any("POPULATION RULE" in s for s in override_rules)
    has_formula_override = any(
        "formula" in s.lower() or "incorrectly" in s.lower() or "don't" in s.lower() or "do not" in s.lower()
        for s in override_rules
    )
    formula = grounding.get("formula", "")
    if formula and not has_population_override and not has_formula_override:
        # Strip LIMIT/ORDER BY from reference formulas — they belong to the domain example's
        # question, not the current question. The planner decides its own LIMIT.
        display_formula = re.sub(r'\s+ORDER\s+BY\s+.*$', '', formula, flags=re.IGNORECASE)
        display_formula = re.sub(r'\s+LIMIT\s+\d+\s*$', '', display_formula, flags=re.IGNORECASE)
        parts.append(f"REFERENCE FORMULA (your SQL MUST use the same columns and logic):\n  {display_formula.strip()}")

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

    # Required columns (tells planner exactly which table.column to use)
    data_reqs = grounding.get("data_requirements", [])
    if data_reqs:
        parts.append("REQUIRED COLUMNS (use these exact table.column references):\n" + "\n".join(f"  - {r}" for r in data_reqs))

    # Domain constraints (non-override rules from grounding)
    domain_rules = grounding.get("domain_rules", [])
    regular_rules = [r for r in domain_rules if r not in override_rules]
    if regular_rules:
        parts.append("CONSTRAINTS:\n" + "\n".join(f"  - {r}" for r in regular_rules))

    # Data format warnings
    format_notes = grounding.get("data_format_notes", [])
    if format_notes:
        parts.append("DATA FORMAT WARNINGS:\n" + "\n".join(f"  ⚠️ {n}" for n in format_notes))

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
def _find_join_path(db_path: Path, table_a: str, table_b: str) -> str:
    """Use BFS on FK graph to find join path between two tables. Returns e.g. 'expense.link_to_budget = budget.budget_id, budget.link_to_event = event.event_id'."""
    from collections import deque
    kg = build_kg_from_sqlite(db_path)
    # Build undirected graph: edges are (neighbor, join_clause)
    graph: dict[str, list[tuple[str, str]]] = {}
    for src_table, fk in kg.inferred_fks:
        src_l = src_table.lower()
        ref_l = fk.ref_table.lower()
        clause_fwd = f"{src_table}.{fk.column} = {fk.ref_table}.{fk.ref_column}"
        clause_rev = f"{fk.ref_table}.{fk.ref_column} = {src_table}.{fk.column}"
        graph.setdefault(src_l, []).append((ref_l, clause_fwd))
        graph.setdefault(ref_l, []).append((src_l, clause_rev))

    start, end = table_a.lower(), table_b.lower()
    if start == end:
        return ""
    visited = {start}
    queue: deque[list[tuple[str, str]]] = deque([[(start, "")]])
    while queue:
        path = queue.popleft()
        current = path[-1][0]
        for neighbor, clause in graph.get(current, []):
            if neighbor == end:
                joins = [c for _, c in path[1:]] + [clause]
                return ", ".join(joins)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [(neighbor, clause)])
    return ""





def _apply_null_guard(sql: str) -> str:
    """Add IS NOT NULL for ORDER BY ... LIMIT 1 patterns (prevents NULL from winning min/max)."""
    # Detect ORDER BY <col> ASC/DESC LIMIT 1 without existing NULL guard
    order_match = re.search(
        r'ORDER\s+BY\s+(\w+(?:\.\w+)?)\s+(ASC|DESC)\s+LIMIT\s+1',
        sql, re.IGNORECASE
    )
    if not order_match:
        return sql
    col = order_match.group(1)
    # Check if there's already a NULL guard for this column
    if re.search(rf'{re.escape(col)}\s+IS\s+NOT\s+NULL', sql, re.IGNORECASE):
        # Already has NULL guard — check if also has != ''
        if re.search(rf"{re.escape(col)}\s*!=\s*''", sql, re.IGNORECASE):
            return sql
        # Add empty string guard
        insert_pos = order_match.start()
        prefix = sql[:insert_pos].rstrip()
        suffix = sql[insert_pos:]
        return f"{prefix} AND {col} != '' {suffix}"
    # Inject WHERE/AND clause before ORDER BY
    insert_pos = order_match.start()
    prefix = sql[:insert_pos].rstrip()
    suffix = sql[insert_pos:]
    if re.search(r'\bWHERE\b', prefix, re.IGNORECASE):
        return f"{prefix} AND {col} IS NOT NULL AND {col} != '' {suffix}"
    else:
        return f"{prefix} WHERE {col} IS NOT NULL AND {col} != '' {suffix}"



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
        self._log_file: Path | None = None
        try:
            log_path = context_dir / "_agent.log"
            log_path.write_text(f"=== {task.task_id} ===\nQ: {question}\n\n")
            self._log_file = log_path
        except OSError:
            pass

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
                # Fallback: try context_dir, then /tmp
                import tempfile as _tf
                db_path = context_dir / CONSOLIDATED_DB_NAME
                try:
                    sqlite3.connect(str(db_path)).close()
                except OSError:
                    db_path = Path(_tf.gettempdir()) / f"_consolidated_{context_dir.name}.db"
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

            # Step 3: Doc extraction (batch LLM with entity-boundary awareness)
            if ctx.doc_sources:
                doc_paths = [doc.path for doc in ctx.doc_sources]
                from data_agent_baseline.pipeline.compiled_extractor import (
                    compiled_extract_docs,
                )
                n_extracted = compiled_extract_docs(
                    doc_paths=doc_paths,
                    db_path=db_path,
                    model=self.model,
                    question=question,
                    knowledge_text=ctx.knowledge_text,
                    log_fn=self._log,
                    structured_tables=structured_tables,
                )

            # Step 4: Build KG from full DB (deterministic) + enrich with descriptions
            kg = build_kg_from_sqlite(db_path)
            kg = enrich_kg_with_descriptions(
                kg, model=self.model, knowledge_text=ctx.knowledge_text,
                log_fn=self._log,
            )
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
                db_path=db_path, kg=kg,
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
            # Step 6: Deterministic SQL investigation — verify JOINs and filters
            # ----------------------------------------------------------
            grounding_context = self._investigate_joins(db_path, grounding_context)

            # ----------------------------------------------------------
            # Rule selection: LLM picks relevant rules from grounding
            # ----------------------------------------------------------
            selected_rules = self._select_sql_rules(question, grounding_context)

            # ----------------------------------------------------------
            # Multi-step SQL investigation: plan → verify steps → final
            # ----------------------------------------------------------
            data_result, sql, failed_sqls = self._run_investigation(
                question=question,
                db_path=db_path,
                sql_schema=sql_schema,
                sql_sample_data=sql_sample_data,
                knowledge_text=ctx.knowledge_text,
                grounding_context=grounding_context,
                col_hints=col_hints,
                selected_rules=selected_rules,
                sample_data=sample_data,
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

            return self._build_result(answer, task)

        except Exception as e:
            logger.exception("Pipeline failed")
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=[],
                failure_reason=str(e),
            )

    # ------------------------------------------------------------------
    # Multi-step investigation: plan → verify → finalize
    # ------------------------------------------------------------------

    def _run_investigation(
        self,
        *,
        question: str,
        db_path: Path,
        sql_schema: str,
        sql_sample_data: str,
        knowledge_text: str,
        grounding_context: str,
        col_hints: str,
        selected_rules: str,
        sample_data: str,
    ) -> tuple[dict[str, Any] | None, str, list[str]]:
        """Multi-step SQL investigation: generate plan, verify each step, build final query.

        Returns: (data_result, final_sql, failed_sqls)
        """
        failed_sqls: list[str] = []

        # Step 1: Generate multi-step plan
        plan = self._generate_sql_plan(
            question, sql_schema, grounding_context, selected_rules,
        )

        if not plan or not plan.get("steps"):
            # Fallback: single-shot SQL (old behavior)
            self._log("plan", "No multi-step plan — using single-shot SQL")
            sql = self._call_sql(
                question, sql_schema, sql_sample_data,
                knowledge_text, grounding_context=grounding_context,
                column_hints=col_hints, selected_rules=selected_rules,
            )
            if sql:
                result = self._try_sql(db_path, sql)
                if result and result.get("rows"):
                    self._log("sql_generated", sql)
                    return result, sql, failed_sqls
                failed_sqls.append(sql)
            # Fall through to multi-hypothesis
            return self._fallback_to_hypothesis(
                question, db_path, sql_schema, sample_data,
                knowledge_text, grounding_context, col_hints, failed_sqls, "",
            )

        # Step 2: Execute verification steps with retries, collect confirmed facts
        confirmed_facts: list[str] = []
        MAX_RETRIES = 2
        self._log("plan", f"{len(plan['steps'])} steps: {[s.get('purpose','') for s in plan['steps']]}")

        for i, step in enumerate(plan["steps"]):
            step_sql = step.get("sql", "")
            purpose = step.get("purpose", f"step {i+1}")
            is_final = step.get("is_final", False) or (i == len(plan["steps"]) - 1)

            if not step_sql:
                continue

            self._log("step", f"[{i+1}/{len(plan['steps'])}] {purpose}")

            # Retry loop for each step
            step_succeeded = False
            current_sql = step_sql
            step_gaps = ""

            for attempt in range(1, MAX_RETRIES + 1):
                current_sql = _apply_null_guard(current_sql)
                self._log("step_sql", f"(attempt {attempt}) {current_sql}")
                result = self._try_sql(db_path, current_sql)

                if result is None:
                    # SQL error
                    error = self.steps[-1].get("detail", "") if self.steps else "unknown error"
                    self._log("step_error", f"Step {i+1} attempt {attempt}: {error}")
                    failed_sqls.append(current_sql)
                    step_gaps = f"- SQL ERROR: {error}\n- CONFIRMED FACTS: {'; '.join(confirmed_facts)}"
                    # Diagnose for column/table hints
                    if db_path:
                        diag = self._diagnose_sql_error(db_path, current_sql, error)
                        if diag:
                            step_gaps += f"\n- {diag}"

                elif not result.get("rows"):
                    # Empty result
                    self._log("step_empty", f"Step {i+1} attempt {attempt}: 0 rows")
                    failed_sqls.append(current_sql)
                    diagnosis = ""
                    if db_path:
                        diagnosis = self._diagnose_empty_result(db_path, current_sql)
                        if diagnosis:
                            self._log("step_diagnosis", diagnosis[:200])
                    step_gaps = f"- Empty result (0 rows)\n- CONFIRMED FACTS: {'; '.join(confirmed_facts)}"
                    if diagnosis:
                        step_gaps += f"\n- {diagnosis}"

                else:
                    # Success — got data
                    cols = result.get("columns", [])
                    rows = result["rows"]

                    # Check for all-NULL
                    all_null = all(
                        all(v is None or str(v).strip().lower() in ("none", "null", "") for v in row)
                        for row in rows
                    )
                    if all_null:
                        self._log("step_null", f"Step {i+1} attempt {attempt}: all NULL")
                        failed_sqls.append(current_sql)
                        step_gaps = f"- Query returned only NULL values\n- CONFIRMED FACTS: {'; '.join(confirmed_facts)}"
                    elif is_final:
                        step_succeeded = True
                        self._log("step_final", f"cols={cols}, rows={len(rows)}")
                        return result, current_sql, failed_sqls
                    else:
                        # Verification step (non-final) — record confirmed values
                        step_succeeded = True
                        if len(rows) <= 5:
                            fact_vals = [str(rows[r][0]) for r in range(min(len(rows), 3))]
                            fact = f"{purpose}: {cols[0] if cols else '?'} = {', '.join(fact_vals)}"
                        else:
                            fact = f"{purpose}: {len(rows)} rows found"
                        confirmed_facts.append(fact)
                        self._log("step_confirmed", fact)
                        break

                # If we have retries left, generate a fix
                if attempt < MAX_RETRIES and step_gaps:
                    step_gaps += f"\n- LAST FAILED SQL (do NOT repeat): {current_sql[:150]}"
                    retry_sql = self._call_sql(
                        question, sql_schema, sql_sample_data,
                        knowledge_text, grounding_context=grounding_context,
                        column_hints=col_hints, selected_rules=selected_rules,
                        gaps=step_gaps,
                    )
                    if retry_sql and retry_sql.strip().upper() != current_sql.strip().upper():
                        current_sql = retry_sql
                    else:
                        # LLM produced same SQL — stop retrying this step
                        self._log("step_duplicate", f"Step {i+1}: retry produced same SQL, giving up")
                        break

            if not step_succeeded and is_final:
                # Final step failed all retries
                self._log("step_failed", f"Step {i+1} (final) failed after {MAX_RETRIES} attempts")
                break

        # If plan didn't produce final result, fall back
        self._log("plan_incomplete", f"Plan exhausted — confirmed: {confirmed_facts}")

        # Try single-shot SQL with confirmed facts as extra context
        if confirmed_facts:
            facts_context = "CONFIRMED FACTS (from verification queries):\n" + "\n".join(f"  - {f}" for f in confirmed_facts)
            sql = self._call_sql(
                question, sql_schema, sql_sample_data,
                knowledge_text, grounding_context=grounding_context,
                column_hints=col_hints, selected_rules=selected_rules,
                extra_context=facts_context,
            )
            if sql:
                self._log("sql_generated", sql)
                result = self._try_sql(db_path, sql)
                if result and result.get("rows"):
                    all_null = all(
                        all(v is None or str(v).strip().lower() in ("none", "null", "") for v in row)
                        for row in result["rows"]
                    )
                    if not all_null:
                        return result, sql, failed_sqls
                failed_sqls.append(sql)

        # Multi-hypothesis fallback
        return self._fallback_to_hypothesis(
            question, db_path, sql_schema, sample_data,
            knowledge_text, grounding_context, col_hints, failed_sqls,
            "\n".join(f"- {f}" for f in confirmed_facts),
        )

    def _fallback_to_hypothesis(
        self, question, db_path, sql_schema, sample_data,
        knowledge_text, grounding_context, col_hints, failed_sqls, diagnosis,
    ) -> tuple[dict[str, Any] | None, str, list[str]]:
        """Multi-hypothesis fallback when investigation fails."""
        hyp_result, hyp_sql = self._try_multi_hypothesis(
            question, db_path, sql_schema, sample_data,
            knowledge_text, grounding_context, col_hints,
            failed_sqls=failed_sqls,
            diagnosis=diagnosis,
        )
        if hyp_result and hyp_result.get("rows"):
            all_null = all(
                all(v is None or str(v).strip().lower() in ("none", "null", "") for v in row)
                for row in hyp_result["rows"]
            )
            if not all_null:
                self._log("multi_hypothesis_ok",
                          f"cols={hyp_result.get('columns')}, rows={len(hyp_result['rows'])}, "
                          f"SQL: {hyp_sql[:150]}")
                return hyp_result, hyp_sql, failed_sqls
            else:
                self._log("multi_hypothesis_null", "Multi-hypothesis also returned NULL")
        return None, "", failed_sqls

    def _generate_sql_plan(
        self, question: str, kg_context: str, grounding_context: str, selected_rules: str,
    ) -> dict[str, Any] | None:
        """LLM generates a multi-step SQL plan: verification steps + final query."""
        prompt = f"""QUESTION: {question}

DATABASE SCHEMA:
{kg_context[:3000]}

{grounding_context}

Generate a step-by-step SQL plan. Break complex queries into verification steps + final query.
Your final SQL MUST implement the REFERENCE FORMULA LITERALLY — do NOT rewrite or substitute any function or operator. Apply JOIN PATHS and FILTER VALUES from the grounding context above.

Return ONLY a JSON object:
{{"steps": [
  {{"purpose": "what this step verifies", "sql": "SELECT ...", "is_final": false}},
  {{"purpose": "final answer query", "sql": "SELECT ...", "is_final": true}}
]}}

RULES:
- Simple queries (single table, direct filter): just 1 step with is_final=true
- Complex queries (JOINs, subqueries, computed values): 1-2 verification steps + final
- Verification steps: resolve IDs, confirm filter values exist, check JOIN produces rows
- Final step: uses confirmed values from verification steps
- Each step must be valid standalone SQL
- Max 3 steps total (keep it lean)
- RATIO/AVG queries: verify the denominator count separately before computing the final ratio
- PERCENTAGE with 'In X, what is the percentage of Y?': X defines the WHERE filter (denominator population), Y is the CASE/COUNT numerator. Do NOT flip them.
- HUMAN-READABLE OUTPUT: If the question asks for names/descriptions, the final SELECT must JOIN to resolve FK IDs to display names — never return raw IDs like 'rec...' or numeric FK values
- MONTHLY vs YEARLY: If data is yearly and question asks monthly, divide by 12. Verify data granularity in a verification step.
{selected_rules}"""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if isinstance(parsed, dict) and "steps" in parsed:
            steps = parsed["steps"]
            if isinstance(steps, list) and steps:
                return parsed
        return None

    # ------------------------------------------------------------------
    # LLM Call 1: SQL Generation
    # ------------------------------------------------------------------

    def _call_sql(
        self, question: str, kg_context: str, sample_data: str,
        knowledge_text: str, gaps: str = "", extra_context: str = "",
        column_hints: str = "", grounding_context: str = "",
        selected_rules: str = "",
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
            selected_rules=selected_rules,
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
- NULL/empty values in SOME rows are VALID — the question asks to "list" items, some may legitimately have empty fields. Do NOT reject results just because some cells are NULL or empty.
- Do NOT question filter values — those are verified.
- Only flag REAL problems: clearly wrong columns, wrong aggregation type, or completely empty result where data should exist."""

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

        # Drop columns whose values look like raw FK IDs (alphanumeric hashes not asked for)
        if len(keep_indices) > 1 and rows:
            cleaned = []
            for i in keep_indices:
                col_name = columns[i].lower()
                # Check if column name suggests FK/ID and values look like hashes
                if ("link_to" in col_name or col_name.endswith("_id")) and col_name not in question.lower():
                    sample_vals = [str(row[i]) for row in rows[:5] if i < len(row)]
                    if sample_vals and all(re.match(r'^rec[A-Za-z0-9]{10,}$', v) for v in sample_vals):
                        continue
                cleaned.append(i)
            if cleaned:
                keep_indices = cleaned

        # Use SQL column names directly
        output_names = [columns[i] for i in keep_indices]

        # Apply column selection to ALL rows (no LLM, no truncation)
        filtered_rows = [
            [str(row[i]) for i in keep_indices]
            for row in rows if len(row) > max(keep_indices)
        ]
        if not filtered_rows:
            return self._raw_result_to_answer(data_result)

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

        # Extract all USE CASE SQL blocks (code-fenced format)
        use_case_pattern = re.compile(
            r'###\s*Use Case[^:]*:\s*(.+?)\n.*?```sql\s*\n(.+?)```\s*\n.*?Explanation[:\s]*(.+?)(?:\n\n|\Z)',
            re.DOTALL | re.IGNORECASE,
        )
        use_cases = use_case_pattern.findall(knowledge_text)

        # Also extract inline SQL patterns: "- SQL: `SELECT ...`" with preceding Metric/Explanation
        inline_sql_pattern = re.compile(
            r'-\s*(?:Metric|Explanation)[:\s]*(.+?)\n(?:.*?\n)*?-\s*SQL:\s*`([^`]+)`',
            re.IGNORECASE,
        )
        for match in inline_sql_pattern.finditer(knowledge_text):
            title = match.group(1).strip()
            sql = match.group(2).strip()
            use_cases.append((title, sql, title))

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

            # use_case_sql intentionally not included — it often matches
            # unrelated queries by keyword overlap and biases grounding away
            # from properly interpreting the actual question.

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
        kg: KnowledgeGraph | None = None,
    ) -> str:
        """Multi-step grounding: Table Select → Focused Ground → Validate → Feedback."""
        grounding: dict[str, Any] = {}

        # Extract domain anchors
        anchor_text = self._extract_domain_anchors(question, knowledge_text, db_path=db_path)

        # Pre-grounding: extract formula + column mappings from knowledge
        knowledge_guidance = ""
        if knowledge_text:
            knowledge_guidance = self._extract_knowledge_guidance(question, knowledge_text)
            if knowledge_guidance:
                self._log("knowledge_guidance", knowledge_guidance[:300])

        self._log("grounding_iter", "--- Grounding ---")

        # Inject knowledge guidance AFTER anchor text (anchors have precise formulas, take priority)
        effective_anchor = anchor_text
        if knowledge_guidance:
            effective_anchor = (
                f"{anchor_text}\n\n⚠️ ADDITIONAL DOMAIN KNOWLEDGE:\n"
                f"{knowledge_guidance}"
            )

        # --- Round 1: Table Selection ---
        selected_tables = self._grounding_select_tables(question, kg_context, effective_anchor, db_path)

        # --- Round 2: Focused Grounding with selected table details ---
        focused_schema = ""
        if selected_tables and db_path:
            focused_schema = self._build_focused_schema_for_grounding(db_path, selected_tables, question, kg=kg)

        # Use focused schema if available, otherwise fall back to full schema
        grounding_schema = focused_schema if focused_schema else kg_context

        # Extract matching SQL from domain knowledge — if found, inject as absolute reference
        domain_sql_ref = self._extract_matching_domain_sql(question, anchor_text)
        if domain_sql_ref:
            effective_anchor = (
                f"⚠️ REFERENCE SQL FROM DOMAIN KNOWLEDGE (use this exact pattern — do NOT add extra conversion or transformation):\n"
                f"  {domain_sql_ref}\n\n{effective_anchor}"
            )
            self._log("domain_sql_match", domain_sql_ref[:200])

        prompt = _build_semantic_prompt(
            question=question,
            kg_context=grounding_schema,
            sample_data=sample_data if not focused_schema else "",
            anchor_text=effective_anchor,
            previous_attempt="",
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

        # Run filter validation (deterministic) + column name priority (LLM) in parallel
        validation_issues = []
        if db_path and grounding:
            rules_before = len(grounding.get("domain_rules", []))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {}
                if grounding.get("known_values"):
                    futures["filter"] = pool.submit(
                        self._validate_filter_values, db_path, grounding
                    )
                futures["col_priority"] = pool.submit(
                    self._check_column_name_priority, db_path, question, grounding, knowledge_guidance
                )
                for key, fut in futures.items():
                    try:
                        result = fut.result()
                        if key == "filter" and result:
                            grounding = result
                            rules_after = len(grounding.get("domain_rules", []))
                            if rules_after > rules_before:
                                validation_issues.extend(grounding["domain_rules"][rules_before:])
                        elif key == "col_priority" and result:
                            validation_issues.append(result)
                    except Exception:
                        pass

        if validation_issues:
            self._log("grounding_v1_issues", json.dumps(validation_issues))

            # Deterministic patch: fix column references without re-grounding
            for issue in validation_issues:
                for m in re.finditer(
                    r'MISMATCH:\s*(\S+?)\.(\S+(?:\s+\S+)*?)\s+should be used instead of\s+(\S+?)\.(\S+(?:\s+\S+)*?)(?:\s+because|\s*$)',
                    issue, re.IGNORECASE,
                ):
                    correct_table, correct_col = m.group(1), m.group(2)
                    wrong_table, wrong_col = m.group(3), m.group(4)
                    self._log("grounding_patch", f"{wrong_table}.{wrong_col} → {correct_table}.{correct_col}")

                    grounding_str = json.dumps(grounding, default=str)
                    table_col_pattern = re.compile(
                        re.escape(f"{wrong_table}.{wrong_col}"), re.IGNORECASE
                    )
                    grounding_str = table_col_pattern.sub(
                        f"{correct_table}.{correct_col}", grounding_str
                    )
                    if wrong_col != correct_col:
                        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(wrong_col) + r'(?![a-zA-Z0-9_])'
                        grounding_str = re.sub(pattern, correct_col, grounding_str, flags=re.IGNORECASE)

                    fixed = self._parse_json(grounding_str)
                    if isinstance(fixed, dict) and fixed:
                        grounding = fixed
                        domain_rules = grounding.get("domain_rules", [])
                        grounding["domain_rules"] = [
                            r for r in domain_rules
                            if wrong_col not in r or correct_col in r
                        ]

            # Re-validate join paths after patching (new table may need new joins)
            if db_path and grounding:
                data_reqs = grounding.get("data_requirements", [])
                tables_needed = set()
                for req in data_reqs:
                    req_str = req if isinstance(req, str) else str(req)
                    if "." in req_str:
                        tables_needed.add(req_str.split(".")[0].lower())
                if len(tables_needed) >= 2:
                    try:
                        join_paths = []
                        tables_list = sorted(tables_needed)
                        for i, t1 in enumerate(tables_list):
                            for t2 in tables_list[i + 1:]:
                                path = _find_join_path(db_path, t1, t2)
                                if path:
                                    join_paths.append(path)
                        if join_paths:
                            grounding["join_paths"] = join_paths
                            self._log("grounding_patch_joins", str(join_paths))
                    except Exception:
                        pass

        self._log("grounding_v1", json.dumps(grounding, default=str))

        # DETERMINISTIC: compute join paths from data_requirements using BFS on FK graph
        if db_path:
            data_reqs = grounding.get("data_requirements", [])
            tables_needed = set()
            for req in data_reqs:
                if "." in req:
                    tables_needed.add(req.split(".")[0].lower())
            if len(tables_needed) >= 2:
                try:
                    join_paths = []
                    tables_list = sorted(tables_needed)
                    for i, t1 in enumerate(tables_list):
                        for t2 in tables_list[i + 1:]:
                            path = _find_join_path(db_path, t1, t2)
                            if path:
                                join_paths.append(path)
                    if join_paths:
                        grounding["join_paths"] = join_paths
                        self._log("grounding_join_paths", str(join_paths))
                except Exception:
                    pass

        # POPULATION CHECK: for "In X, what is the percentage of Y?" questions,
        # verify the formula uses X as WHERE (denominator) and Y as CASE (numerator)
        if grounding:
            q_lower = question.lower()
            has_percentage = "percentage" in q_lower or "proportion" in q_lower or "%" in q_lower
            in_match = re.match(r"^in\s+(.+?),\s+what\s+is\s+the\s+(?:percentage|proportion)", q_lower)
            if has_percentage and in_match:
                population_clause = in_match.group(1).strip()
                formula = grounding.get("formula", "")
                computation_steps = grounding.get("computation_steps", [])
                # Check if the first step filters by the population condition
                # If computation starts with filtering by the metric (Y) instead of population (X), flag it
                steps_text = " ".join(computation_steps).lower()
                if population_clause and formula:
                    # The population condition should be in the WHERE clause
                    # If we detect the formula filters by the metric entity first, override
                    override = (
                        f"POPULATION RULE: '{population_clause}' defines the WHERE filter (denominator). "
                        f"The percentage numerator is the subset matching the other condition. "
                        f"Correct pattern: SELECT COUNT(condition) / COUNT(*) FROM ... WHERE [population_filter]"
                    )
                    if "_semantic_overrides" not in grounding:
                        grounding["_semantic_overrides"] = []
                    grounding["_semantic_overrides"].append(override)
                    self._log("population_check", f"Injected population rule: '{population_clause}' = WHERE filter")

        # Run LLM checks in parallel: semantic feedback + formula validation
        feedback = ""
        corrected_formula = ""
        if grounding:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {}
                futures["feedback"] = pool.submit(
                    self._semantic_feedback, question, grounding, kg_context
                )
                if effective_anchor:
                    futures["formula"] = pool.submit(
                        self._check_formula_against_domain, question, grounding, effective_anchor
                    )
                for key, fut in futures.items():
                    try:
                        if key == "feedback":
                            feedback = fut.result()
                        elif key == "formula":
                            corrected_formula = fut.result()
                    except Exception:
                        pass

        # Save original formula for disambiguation (corrected formula may be a domain example with unrelated filter values)
        original_formula = grounding.get("formula", "") if grounding else ""

        # Apply formula correction — clear domain_rules that may contradict the corrected formula
        if corrected_formula and grounding:
            grounding["formula"] = corrected_formula
            grounding["domain_rules"] = []

        # Apply semantic feedback
        if grounding and feedback:
            # Guard: discard feedback that contradicts arithmetic in domain_rules
            domain_rules = grounding.get("domain_rules", [])
            feedback_contradicts_formula = False
            if domain_rules:
                domain_text = " ".join(domain_rules).lower()
                fb_lower = feedback.lower()
                # Check symbolic operators
                for op in ["/ 12", "/12", "* 100", "*100", "/ 4", "/4", "* 12", "*12"]:
                    if op.replace(" ", "") in domain_text.replace(" ", ""):
                        if "divid" in fb_lower or "multiply" in fb_lower or "division" in fb_lower or "without" in fb_lower or "incorrectly" in fb_lower:
                            feedback_contradicts_formula = True
                            break
                # Check word-based arithmetic ("divided by 12", "multiplied by 100")
                if not feedback_contradicts_formula:
                    arithmetic_words = re.findall(r'(?:divided|multiplied)\s+by\s+\d+', domain_text)
                    if arithmetic_words:
                        if "divid" in fb_lower or "multiply" in fb_lower or "incorrectly" in fb_lower or "don't" in fb_lower:
                            feedback_contradicts_formula = True
            if feedback_contradicts_formula:
                self._log("grounding_feedback_discarded", f"Feedback contradicts domain formula: {feedback[:100]}")
            else:
                self._log("grounding_feedback", feedback)
                grounding["_semantic_overrides"] = [feedback]
        elif grounding and not feedback:
            # Deterministic override: if question word matches an exact column name
            # not used in formula SELECT, inject as override for the SQL planner
            override = self._check_missing_select_columns(question, grounding, kg_context)
            if override:
                self._log("grounding_deterministic_override", override)
                grounding["_semantic_overrides"] = [override]

        # Deterministic column-disambiguation: if knowledge defines what a column means
        # and the question uses a word matching that column, but the formula uses a different column,
        # fix the known_values and formula to use the correct column.
        # Use original_formula to avoid false positives from domain example SQL with unrelated filter values.
        if grounding:
            disambig_grounding = grounding
            if original_formula and original_formula != grounding.get("formula", ""):
                disambig_grounding = {**grounding, "formula": original_formula}
            self._log("disambig_input", f"formula={disambig_grounding.get('formula','')[:100]} | knowledge={bool(knowledge_text)}")
            col_override, fix_info = self._check_column_disambiguation(question, disambig_grounding, kg_context, knowledge_text)
            self._log("disambig_result", f"override={col_override[:100] if col_override else 'None'} | fix_info={fix_info}")
            if col_override and fix_info:
                self._log("grounding_col_disambig", col_override)
                wrong_col = fix_info["wrong_col"]
                correct_col = fix_info["correct_col"]
                val = fix_info["val"]

                # Patch the entire grounding dict with word-boundary replacement
                grounding_str = json.dumps(grounding, default=str)
                pattern = r'(?<![a-zA-Z0-9_])' + re.escape(wrong_col) + r'(?![a-zA-Z0-9_])'
                grounding_str = re.sub(pattern, correct_col, grounding_str, flags=re.IGNORECASE)
                fixed = self._parse_json(grounding_str)
                if isinstance(fixed, dict) and fixed:
                    grounding = fixed

                # Fix known_values keys (json replace doesn't rename dict keys reliably)
                known_values = grounding.get("known_values", {})
                for key in list(known_values.keys()):
                    if key.lower().endswith(f".{wrong_col.lower()}"):
                        table = key.split(".")[0]
                        known_values[f"{table}.{correct_col}"] = known_values.pop(key)
                        break

                # Remove conflicting domain_rules that mention the wrong column
                domain_rules = grounding.get("domain_rules", [])
                grounding["domain_rules"] = [
                    r for r in domain_rules
                    if wrong_col.lower() not in r.lower()
                ]
                overrides = grounding.get("_semantic_overrides", [])
                overrides.append(col_override)
                grounding["_semantic_overrides"] = overrides

        # Fix 5: Verify all question entities appear in formula/known_values
        if grounding:
            grounding = self._verify_filter_completeness(question, grounding)

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

    def _select_sql_rules(self, question: str, grounding_context: str) -> str:
        """LLM picks which SQL rules are relevant based on question + grounding. ~2K char prompt."""
        rule_list = "\n".join(
            f"{i}: {label}" for i, (label, _) in enumerate(SQL_RULES_LABELED)
        )
        prompt = f"""Pick the SQL rules needed for this task.

QUESTION: {question}

PLAN:
{grounding_context[:1500]}

RULES (by index and label):
{rule_list}

Return ONLY a JSON: {{"indices": [0, 3, 7, ...]}}
Pick 5-10 rules most relevant to this specific question. Always include 0 (exact_question).
If the question asks for names, descriptions, or labels — include rule 5 (human_readable).
If the question asks for something with a superlative (lowest, highest, most, least, best, worst, largest, smallest, etc.) — ALWAYS include the superlative rule."""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if isinstance(parsed, dict) and "indices" in parsed:
            indices = parsed["indices"]
            if isinstance(indices, list) and indices:
                selected = []
                selected_labels = []
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(SQL_RULES_LABELED):
                        selected.append(SQL_RULES_LABELED[idx][1])
                        selected_labels.append(SQL_RULES_LABELED[idx][0])

                # Deterministic injection: always include relevant rules
                q_lower = question.lower()
                if ("percentage" in q_lower or "proportion" in q_lower or "%" in q_lower) and "population" not in selected_labels:
                    pop_rule = next((r for l, r in SQL_RULES_LABELED if l == "population"), None)
                    if pop_rule:
                        selected.append(pop_rule)
                        selected_labels.append("population")

                if ("time" in q_lower or "lap" in q_lower or "duration" in q_lower) and "time_parse" not in selected_labels:
                    tp_rule = next((r for l, r in SQL_RULES_LABELED if l == "time_parse"), None)
                    if tp_rule:
                        selected.append(tp_rule)
                        selected_labels.append("time_parse")

                if selected:
                    self._log("rules_selected", f"{len(selected)} rules: {selected_labels}")
                    return "\n".join(f"- {r}" for r in selected)

        # Fallback: compact rules
        self._log("rules_selected", "fallback (parse failed)")
        return """- Answer the EXACT question. SELECT only asked columns. No SELECT *.
- Use FILTER VALUES exactly as given. Do NOT substitute.
- For superlatives (lowest/highest), use WHERE col = (SELECT MIN/MAX(col)...) — no LIMIT.
- JOIN through FK paths shown in schema.
- Use LIKE '%X%' COLLATE NOCASE for text. CAST(x AS REAL) for division.
- NEVER RETURN NULL — add WHERE IS NOT NULL. Escape apostrophes with ''."""

    def _verify_filter_completeness(self, question: str, grounding: dict[str, Any]) -> dict[str, Any]:
        """Check that quoted strings and proper nouns from question are in formula/known_values.

        If missing, inject them as constraints so the SQL planner adds the filter.
        """
        formula = grounding.get("formula", "")
        known_values = grounding.get("known_values", {})
        kv_flat = " ".join(str(v) for vs in known_values.values() for v in vs).lower()
        formula_lower = formula.lower()

        # Extract quoted entities from question
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
        # Extract proper nouns (capitalized multi-word sequences not at start of sentence)
        proper_nouns = re.findall(r'(?:the |in |for |of )([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)+)', question)

        missing_entities = []
        for entity in quoted + proper_nouns:
            entity_lower = entity.lower()
            if entity_lower not in formula_lower and entity_lower not in kv_flat:
                # Check if any word from the entity is in formula (partial match is OK)
                words = entity_lower.split()
                if not any(w in formula_lower for w in words if len(w) > 3):
                    missing_entities.append(entity)

        if missing_entities:
            overrides = grounding.get("_semantic_overrides", [])
            for entity in missing_entities[:2]:
                overrides.append(
                    f"The question mentions '{entity}' which MUST appear as a filter condition (WHERE/HAVING). "
                    f"Find the column that contains this value and add it to the query."
                )
            grounding["_semantic_overrides"] = overrides

        return grounding

    def _extract_matching_domain_sql(self, question: str, anchor_text: str) -> str:
        """Deterministically extract SQL from domain knowledge that matches the question.
        Returns the SQL string if a close match is found, empty string otherwise."""
        if not anchor_text:
            return ""
        # Find all SQL patterns in anchor text
        sql_blocks: list[tuple[str, str]] = []  # (context, sql)
        # Pattern 1: - SQL: `SELECT ...`
        for m in re.finditer(r'-\s*SQL:\s*`([^`]+)`', anchor_text):
            # Get surrounding context (previous 200 chars)
            start = max(0, m.start() - 200)
            context = anchor_text[start:m.start()].lower()
            sql_blocks.append((context, m.group(1).strip()))
        # Pattern 2: ```sql ... ```
        for m in re.finditer(r'```sql\s*\n(.+?)```', anchor_text, re.DOTALL):
            start = max(0, m.start() - 200)
            context = anchor_text[start:m.start()].lower()
            sql_blocks.append((context, m.group(1).strip()))

        if not sql_blocks:
            return ""

        # Score each SQL block by keyword overlap with question
        q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))
        best_sql = ""
        best_score = 0
        for context, sql in sql_blocks:
            ctx_words = set(re.findall(r'\b[a-z]{3,}\b', context))
            overlap = len(q_words & ctx_words)
            # Bonus for matching specific values (numbers, proper nouns)
            q_numbers = set(re.findall(r'\b\d+\b', question))
            sql_numbers = set(re.findall(r'\b\d+\b', sql))
            number_match = len(q_numbers & sql_numbers)
            score = overlap + number_match * 3
            if score > best_score:
                best_score = score
                best_sql = sql

        # Only return if strong match (at least 5 word overlap + number match)
        if best_score >= 5:
            return best_sql
        return ""

    def _extract_knowledge_guidance(self, question: str, knowledge_text: str) -> str:
        """Use LLM to extract relevant knowledge for this question."""
        prompt = f"""QUESTION: {question}

DOMAIN KNOWLEDGE:
{knowledge_text}

Extract the parts of DOMAIN KNOWLEDGE relevant to this question — formulas, column definitions, disambiguation rules, or use cases. Return them as-is from the document. If nothing is relevant, return: NONE"""

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self._model_call_with_retry(messages)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if raw.upper().startswith("NONE") or len(raw) < 10:
                return ""
            return raw
        except Exception:
            return ""


    def _grounding_select_tables(
        self, question: str, kg_context: str, anchor_text: str, db_path: Path | None,
        feedback: str = "",
    ) -> list[str]:
        """Round 1: LLM picks which tables are relevant to the question.

        Shows only table names + column names (no types, no samples) to keep prompt small.
        Returns list of selected table names.
        """
        if not db_path or not db_path.exists():
            return []

        # Build compact table overview: table name + column names only
        conn = sqlite3.connect(str(db_path))
        table_lines: list[str] = []
        all_tables: list[str] = []
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall() if not r[0].startswith("_")]
            all_tables = tables

            if len(tables) <= 3:
                conn.close()
                self._log("grounding_tables_selected", f"≤3 tables, using all: {tables}")
                return tables

            for tname in tables:
                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
                table_lines.append(f"- {tname} ({row_count} rows): {', '.join(cols)}")
        except Exception:
            conn.close()
            return []
        conn.close()

        # Deterministic pre-pass: find tables with columns matching question words
        q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))
        candidate_tables: list[str] = []
        for line in table_lines:
            tname = line.split(" (")[0].lstrip("- ")
            line_lower = line.lower()
            if any(w in line_lower for w in q_words):
                candidate_tables.append(tname)
        self._log("grounding_candidates", f"Deterministic candidates: {candidate_tables}")

        feedback_section = ""
        if feedback:
            feedback_section = f"\n⚠️ PREVIOUS TABLE SELECTION FAILED:\n{feedback}\nYou MUST include tables that contain the needed filter data.\n"

        prompt = f"""QUESTION: {question}

TABLES IN DATABASE:
{chr(10).join(table_lines)}

{f"DOMAIN KNOWLEDGE:{chr(10)}{anchor_text[:800]}" if anchor_text else ""}
{feedback_section}
Which tables are needed to answer this question? Consider:
- Tables whose columns match terms in the question
- Tables needed for JOIN paths between relevant tables
- Tables with human-readable names/labels if the question asks for names

Return ONLY: {{"tables": ["table1", "table2", ...]}}
Select 2-5 tables. Include linking/bridge tables needed for JOINs."""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if isinstance(parsed, dict) and "tables" in parsed:
            selected = [t for t in parsed["tables"] if t in all_tables]
            # Always include candidate tables from deterministic pass
            for ct in candidate_tables:
                if ct not in selected:
                    selected.append(ct)
            if selected:
                self._log("grounding_tables_selected", f"{len(selected)} tables: {selected}")
                return selected

        # Fallback: return all tables
        return all_tables

    def _build_focused_schema_for_grounding(
        self, db_path: Path, tables: list[str], question: str,
        kg: KnowledgeGraph | None = None,
    ) -> str:
        """Build detailed schema for selected tables with sample values and format hints."""
        self._log("grounding_focused_schema", f"Building focused schema for: {tables}")
        conn = sqlite3.connect(str(db_path))
        lines: list[str] = ["=== DATABASE SCHEMA ===", ""]
        fk_lines: list[str] = []

        try:
            for tname in tables:
                exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tname,)
                ).fetchone()
                if not exists:
                    continue

                row_count = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
                col_info = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                pk_cols = [c[1] for c in col_info if c[5]]

                lines.append(f"TABLE: {tname} ({row_count} rows, PK: {', '.join(pk_cols) if pk_cols else '(none)'})")

                for c in col_info:
                    col_name = c[1]
                    col_type = c[2] or "TEXT"
                    pk_mark = " [PK]" if c[5] else ""

                    # Sample values
                    sample = ""
                    try:
                        vals = conn.execute(
                            f'SELECT DISTINCT "{col_name}" FROM "{tname}" '
                            f'WHERE "{col_name}" IS NOT NULL AND "{col_name}" != \'\' '
                            f'ORDER BY "{col_name}" LIMIT 6'
                        ).fetchall()
                        if vals:
                            sample_vals = [v[0] for v in vals]
                            sample = f"  e.g. {sample_vals}"
                            # Format hint
                            format_note = self._detect_value_format(sample_vals)
                            if format_note:
                                sample += format_note
                    except Exception:
                        pass

                    # Look up description from enriched KG
                    desc_str = ""
                    if kg:
                        kg_table = kg.get_table(tname)
                        if kg_table:
                            for kc in kg_table.columns:
                                if kc.name == col_name and kc.description:
                                    desc_str = f"  -- {kc.description}"
                                    break

                    lines.append(f"  - {col_name} ({col_type}){pk_mark}{desc_str}{sample}")

                # FK info
                try:
                    fks = conn.execute(f'PRAGMA foreign_key_list("{tname}")').fetchall()
                    for fk in fks:
                        ref_table = fk[2]
                        from_col = fk[3]
                        to_col = fk[4]
                        lines.append(f"  FK: {from_col} → {ref_table}.{to_col}")
                        fk_lines.append(f"  {tname}.{from_col} = {ref_table}.{to_col}")
                except Exception:
                    pass

                lines.append("")

            # Inferred FKs between selected tables
            kg = build_kg_from_sqlite(db_path)
            table_set = set(t.lower() for t in tables)
            for src_table, fk in kg.inferred_fks:
                if src_table.lower() in table_set and fk.ref_table.lower() in table_set:
                    fk_line = f"  {src_table}.{fk.column} = {fk.ref_table}.{fk.ref_column}"
                    if fk_line not in fk_lines:
                        fk_lines.append(fk_line)

            if fk_lines:
                lines.append("=== JOIN PATHS ===")
                for fl in fk_lines:
                    lines.append(fl)
                lines.append("")

        finally:
            conn.close()

        result = "\n".join(lines)
        self._log("grounding_focused_schema_done", f"{len(result)} chars, {len(tables)} tables, {len(fk_lines)} FKs")
        return result

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

    def _check_column_name_priority(
        self, db_path: Path, question: str, grounding: dict[str, Any], knowledge_guidance: str = ""
    ) -> str:
        """LLM check: does the grounding use the right column for the question's measure?
        Returns feedback string if there's a mismatch, empty string if OK."""
        data_reqs = grounding.get("data_requirements", [])
        formula = grounding.get("formula", "")
        if not data_reqs:
            return ""

        # Get all columns in the DB
        try:
            conn = sqlite3.connect(str(db_path))
            all_cols: list[str] = []
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall():
                tname = row[0]
                if tname.startswith("_"):
                    continue
                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                for col in cols:
                    all_cols.append(f"{tname}.{col}")
            conn.close()
        except Exception:
            return ""

        knowledge_section = ""
        if knowledge_guidance:
            knowledge_section = f"\n\nDOMAIN KNOWLEDGE (authoritative column definitions):\n{knowledge_guidance}\n"

        data_reqs_str = ", ".join(str(r) for r in data_reqs)
        what_user_wants = grounding.get("what_user_wants", "")
        phrase_mapping = grounding.get("phrase_mapping", {})
        phrase_section = ""
        if phrase_mapping:
            mapping_lines = [f"  \"{k}\" → {v}" for k, v in phrase_mapping.items()]
            phrase_section = f"\nGROUNDING phrase-to-column mapping:\n" + "\n".join(mapping_lines) + "\n"
        goal_section = ""
        if what_user_wants:
            goal_section = f"\nGROUNDING interpretation: {what_user_wants}\n"

        prompt = f"""QUESTION (original): {question}
{goal_section}{phrase_section}
GROUNDING chose these columns: {data_reqs_str}
GROUNDING formula: {formula}

ALL columns in database:
{chr(10).join(all_cols)}{knowledge_section}

Is there a column in the database that better matches a word/phrase in the question, but was NOT chosen by the grounding?

Check ALL of these:
1. Metric/measure columns: a word in the question matches a column name better than the one chosen
2. Output/attribute columns: the grounding uses a column that doesn't match the question's requested attribute
3. Entity ownership: when the question asks for a property of entity X and both TableX.col and TableY.col exist, prefer TableX
4. Domain-defined semantics: if DOMAIN KNOWLEDGE explicitly defines what a column means, and the question uses that word, the grounding MUST use the domain-defined column

Rules:
- A column whose name literally contains the question's keyword is a stronger match than one that doesn't.
- When the same column name exists in multiple tables, the column belonging to the entity the question asks about takes priority.
- Domain knowledge definitions are AUTHORITATIVE — they override assumptions based on general English.
- A word in the question that matches a column name but is clearly a filter value (not a requested output) → NOT a mismatch.
- Check EVERY output column in the grounding independently. Report ALL mismatches, not just the first one.

For example:
- Question asks for "X's attribute" and both TableX.attribute and TableY.attribute exist → prefer TableX (entity ownership)
- Column name literally contains the question's keyword but grounding uses a different column → MISMATCH
- A word in the question matches a column name but is clearly a filter value (e.g. "number 19") → NOT a mismatch

Reply ONLY:
- OK (if all grounding columns are correct)
- One or more MISMATCH lines (one per wrong column):
  MISMATCH: <table.column> should be used instead of <table.column> because <reason>"""

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self._model_call_with_retry(messages)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if not raw:
                return ""
            if raw.strip().upper().startswith("OK"):
                return ""
            # Collect all MISMATCH lines
            mismatches = []
            for line in raw.split("\n"):
                line = line.strip()
                if "MISMATCH" in line.upper():
                    mismatches.append(line)
            if mismatches:
                combined = "\n".join(mismatches)
                self._log("col_name_priority", combined)
                return combined
            return ""
        except Exception:
            return ""

    def _check_formula_against_domain(
        self, question: str, grounding: dict[str, Any], anchor_text: str
    ) -> str:
        """LLM check: does grounding formula match what domain knowledge defines?
        Returns corrected formula if mismatch, empty string if OK."""
        formula = grounding.get("formula", "")
        if not formula or not anchor_text:
            return ""

        prompt = f"""DOMAIN KNOWLEDGE defines this formula:
{anchor_text[:2000]}

GROUNDING produced this formula for the question "{question}":
  {formula}

Find the formula definition in DOMAIN KNOWLEDGE that matches the question's metric.
Then check: does the GROUNDING formula contain EVERY operator and number from the domain formula?

IMPORTANT: Do NOT think about whether operations make mathematical sense.
Just compare characters. If domain has "/ 12" and grounding does not have "/ 12" — that is MISSING.

Reply ONLY one line:
- OK
- CORRECTED: <paste the formula exactly as it appears in domain knowledge>"""

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self._model_call_with_retry(messages)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if not raw:
                return ""
            first_line = raw.split("\n")[0].strip()
            if first_line.upper().startswith("OK"):
                return ""
            if "CORRECTED" in first_line.upper():
                corrected = first_line.split(":", 1)[-1].strip()
                if corrected:
                    self._log("formula_check_corrected", f"{formula} → {corrected}")
                    return corrected
            return ""
        except Exception:
            return ""

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

                # Get FK columns for this table
                fk_cols_set: set[str] = set()
                try:
                    fks = conn.execute(f'PRAGMA foreign_key_list("{tname}")').fetchall()
                    for fk in fks:
                        fk_cols_set.add(fk[3])  # from_col
                except Exception:
                    pass

                # Include: requested columns, PKs, FKs. Skip others if table is wide.
                pk_set = set(pk_cols)
                priority_cols = req_cols | pk_set | fk_cols_set
                all_col_names = [c[1] for c in col_info]

                # If table has many columns, only show priority + a few extras
                if len(col_info) > 12:
                    show_cols = priority_cols
                else:
                    show_cols = set(all_col_names)

                for c in col_info:
                    col_name = c[1]
                    if col_name not in show_cols:
                        continue
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

                # Note omitted columns count
                omitted = len(col_info) - len([c for c in col_info if c[1] in show_cols])
                if omitted > 0:
                    lines.append(f"  ... ({omitted} more columns)")

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
        # Use prefix matching to handle stemming (e.g., "ranked" matches "rank")
        missing = []
        current_table = ""
        for line in kg_context.split("\n"):
            if line.startswith("TABLE: "):
                current_table = line.split("TABLE: ")[1].split(" ")[0].strip()
            elif line.strip().startswith("- ") and current_table:
                col_name = line.strip()[2:].split(" ")[0].strip()
                col_lower = col_name.lower()
                col_parts = set(re.findall(r'[a-z]{3,}', col_lower))
                # Check exact match OR prefix match (ranked→rank, positions→position)
                matched = col_parts & q_words
                if not matched:
                    for cp in col_parts:
                        for qw in q_words:
                            if qw.startswith(cp) or cp.startswith(qw):
                                matched = True
                                break
                        if matched:
                            break
                if matched:
                    if col_lower not in formula_lower and f"{current_table}.{col_name}".lower() not in formula_lower:
                        missing.append(f"{current_table}.{col_name}")

        if missing:
            missing_cols_hint = f"\n\nNOTE: The following columns match words in the question but are NOT used in the formula: {', '.join(missing)}. Consider whether any of these should be in the SELECT, GROUP BY, or WHERE clause instead of/in addition to the current columns. If DOMAIN RULES define what a column means (e.g., 'rank = ranking by fastest lap'), and the question uses that word ('ranked'), the formula MUST use that column — not a similar-sounding one."

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
- Do NOT question arithmetic operations (/ 12, * 100, etc.) if DOMAIN RULES define a formula. The formula IS authoritative — do not override it with your reasoning about data granularity.
- Do NOT comment on row count, LIMIT, or whether the result should be singular/multiple. Ties may exist.
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

    def _check_column_disambiguation(self, question: str, grounding: dict[str, Any], kg_context: str, knowledge_text: str = "") -> tuple[str, dict[str, str] | None]:
        """Deterministic check: knowledge defines column semantics but formula uses wrong column.

        Returns (override_text, fix_info_dict) or ("", None) if no mismatch.
        fix_info has keys: wrong_col, correct_col, val.
        """
        formula = grounding.get("formula", "")
        if not formula or not knowledge_text:
            self._log("disambig_bail", f"no formula or knowledge: formula={bool(formula)} knowledge={bool(knowledge_text)}")
            return "", None

        q_lower = question.lower()
        formula_lower = formula.lower()
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q_lower))

        # Pattern: "- **col**: definition" or "- col: definition"
        col_definitions: dict[str, str] = {}
        for m in re.finditer(
            r'[-*]\s*\*?\*?(\w+)\*?\*?\s*:\s*(.+?)(?:\n|$)',
            knowledge_text, re.IGNORECASE
        ):
            col_name = m.group(1).lower()
            definition = m.group(2).lower().strip()
            col_definitions[col_name] = definition

        if not col_definitions:
            self._log("disambig_bail", "no col_definitions found")
            return "", None

        # Get all real column names from kg_context
        schema_cols: set[str] = set()
        for line in kg_context.split("\n"):
            if line.strip().startswith("- "):
                col = line.strip()[2:].split(" ")[0].strip().lower()
                if col:
                    schema_cols.add(col)

        self._log("disambig_cols", f"defined={list(col_definitions.keys())[:10]} schema={sorted(list(schema_cols))[:15]}")

        # Find question words that match defined columns via prefix/stem
        matched_col = ""
        for col_name in col_definitions:
            if col_name not in schema_cols:
                continue
            for qw in q_words:
                if qw == col_name or qw.startswith(col_name) or col_name.startswith(qw):
                    matched_col = col_name
                    break
            if matched_col:
                break

        if not matched_col:
            self._log("disambig_bail", f"no matched_col. q_words={sorted(q_words)}")
            return "", None

        # Check if the matched column is already in the formula WHERE
        where_match = re.search(r'where\b(.+)', formula_lower, re.DOTALL)
        if not where_match:
            self._log("disambig_bail", f"no WHERE in formula: {formula_lower[:100]}")
            return "", None

        where_clause = where_match.group(1)
        if re.search(rf'\b{re.escape(matched_col)}\b', where_clause):
            self._log("disambig_bail", f"matched_col '{matched_col}' already in WHERE: {where_clause[:100]}")
            return "", None

        self._log("disambig_proceed", f"matched_col={matched_col} where={where_clause[:100]}")

        # The matched column is NOT in WHERE. Check if a DIFFERENT column with a small numeric filter
        # is being used that could be confusable (e.g., ordinal values like 1,2,3)
        other_filters = re.findall(r'(\w+)\s*=\s*(\d+)', where_clause)
        for other_col, val in other_filters:
            other_col_lower = other_col.lower()
            if other_col_lower == matched_col:
                continue
            # Only flag ordinal-like values (small numbers ≤ 100) to avoid false positives on years/IDs
            if int(val) > 100:
                continue
            # The other column must be in schema and be a plausible ordinal/ranking column
            if other_col_lower in schema_cols:
                col_def = col_definitions[matched_col][:80]
                other_def = col_definitions.get(other_col_lower, "")[:80]
                override = (
                    f"COLUMN MISMATCH: Question word matches '{matched_col}' "
                    f"(defined as: {col_def}), but formula uses '{other_col_lower}' "
                    f"(defined as: {other_def or 'no explicit definition'}). "
                    f"Use WHERE {matched_col} = {val} instead of WHERE {other_col_lower} = {val}."
                )
                fix_info = {"wrong_col": other_col_lower, "correct_col": matched_col, "val": val}
                return override, fix_info

        return "", None

    def _validate_sql_against_grounding(self, sql: str, grounding_context: str, question: str) -> str:
        """LLM check: does the SQL respect the grounding context (especially MANDATORY CORRECTIONS)?

        Returns violation description if SQL violates grounding, empty string if OK.
        """
        prompt = f"""QUESTION: {question}

GROUNDING CONTEXT:
{grounding_context}

GENERATED SQL:
{sql}

Check if the SQL uses a WRONG column for the key metric/filter that the question asks about.
Only flag if the SQL computes the answer from a DIFFERENT column than what GROUNDING specifies.
PASS if: correct metric column used, correct filter column used, joins reach the right tables.
FAIL if: wrong column for the main computation (e.g. using 'spent' when grounding says 'cost').
Do NOT flag: join style differences, extra columns, subquery vs JOIN approaches.

Reply in this EXACT format (first line must be PASS or FAIL):
PASS
or
FAIL: <one sentence: what column is wrong and what it should be>"""

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self._model_call_with_retry(messages)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if not raw:
                return ""
            first_line = raw.split("\n")[0].strip()
            if first_line.upper().startswith("PASS"):
                return ""
            if first_line.upper().startswith("FAIL"):
                return first_line[5:].strip() if len(first_line) > 5 else raw.split("\n", 1)[-1].strip()[:200]
            if "not violate" in raw.lower() or "correctly" in raw.lower() or "no violation" in raw.lower():
                return ""
            if len(raw) > 200:
                raw = raw[:200]
            return raw
        except Exception:
            return ""

    def _sanity_check_result(self, question: str, grounding_context: str, cols: list[str], rows: list[list], sql: str) -> str:
        """LLM sanity check: does the result make sense for the question?

        Catches: wrong computation type (0% when expecting non-zero), ratio=1.0 when
        expecting a real ratio, empty/None results, obviously wrong magnitudes.
        Returns issue description or empty string if OK.
        """
        if not rows:
            return ""

        result_preview = ", ".join(cols) + "\n"
        for row in rows[:5]:
            result_preview += ", ".join(str(v) for v in row) + "\n"
        if len(rows) > 5:
            result_preview += f"... ({len(rows)} total rows)\n"

        prompt = f"""QUESTION: {question}

GROUNDING:
{grounding_context[:1000]}

SQL: {sql}

RESULT:
{result_preview}

Does this result make sense as the answer? Check:
1. If question asks for a percentage/ratio — is the value plausible (not 0, not exactly 1.0 unless trivial)?
2. If question asks "how many" — is it a reasonable count (not None, not 0 when data clearly exists)?
3. If result has None/NULL values — that likely means the SQL failed to find matching data.
4. If question asks for a name/identifier — is None or empty suspicious?

DO NOT judge row count. Multiple rows are valid if the entity has multiple records. "The date X paid" can have multiple dates.

Reply PASS if the result looks reasonable, or FAIL: <one sentence why it's suspicious>"""

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self._model_call_with_retry(messages)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if not raw:
                return ""
            first_line = raw.split("\n")[0].strip()
            if first_line.upper().startswith("PASS"):
                return ""
            if first_line.upper().startswith("FAIL"):
                return first_line[5:].strip() if len(first_line) > 5 else raw.split("\n", 1)[-1].strip()[:200]
            if "reasonable" in raw.lower() or "plausible" in raw.lower() or "makes sense" in raw.lower():
                return ""
            if len(raw) > 200:
                raw = raw[:200]
            return raw
        except Exception:
            return ""

    def _validate_formula_deterministic(self, db_path: Path, formula: str) -> str:
        """Validate formula SQL by executing with LIMIT 0. Returns error string or empty."""
        # Skip validation for expression-only formulas (not full SELECT statements)
        if not formula.strip().upper().startswith("SELECT"):
            return ""
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

    def _investigate_joins(self, db_path: Path | None, grounding_context: str) -> str:
        """Deterministic: verify inferred JOIN paths actually produce rows.

        If a JOIN path uses transformations (e.g., '0' || col) and produces 0 rows,
        try a direct column-to-column join. Updates grounding_context with corrections.
        """
        if not db_path or not db_path.exists() or "JOIN PATHS:" not in grounding_context:
            return grounding_context

        # Extract join paths from grounding context
        join_lines: list[str] = []
        in_join_section = False
        for line in grounding_context.split("\n"):
            if line.strip() == "JOIN PATHS:":
                in_join_section = True
                continue
            if in_join_section:
                if line.startswith("  ") and "->" in line:
                    join_lines.append(line.strip())
                else:
                    break

        if not join_lines:
            return grounding_context

        conn = sqlite3.connect(str(db_path))
        corrections: list[str] = []
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall() if not r[0].startswith("_")]

            for join_expr in join_lines:
                # Parse: "tableA.colA -> expression" e.g. "schools.CDSCode -> '0' || satscores.cds"
                parts = join_expr.split("->")
                if len(parts) != 2:
                    continue
                left = parts[0].strip()  # e.g. "schools.CDSCode"
                right = parts[1].strip()  # e.g. "'0' || satscores.cds"

                # Only investigate if right side has transformations
                has_transform = any(op in right for op in ["||", "CAST", "SUBSTR", "REPLACE", "TRIM"])
                if not has_transform:
                    continue

                # Extract table.col from left side
                if "." not in left:
                    continue
                left_table, left_col = left.split(".", 1)

                # Extract table.col from right side (find table.col pattern in expression)
                right_table_col = re.search(r'(\w+)\.(\w+)', right)
                if not right_table_col:
                    continue
                right_table = right_table_col.group(1)
                right_col = right_table_col.group(2)

                # Check both tables exist
                if left_table not in tables or right_table not in tables:
                    continue

                # Test the transformed join
                try:
                    transformed_count = conn.execute(
                        f'SELECT COUNT(*) FROM "{left_table}" l JOIN "{right_table}" r '
                        f'ON l."{left_col}" = {right.replace(f"{right_table}.", "r.")}'
                    ).fetchone()[0]
                except Exception:
                    transformed_count = 0

                # Test direct join
                try:
                    direct_count = conn.execute(
                        f'SELECT COUNT(*) FROM "{left_table}" l JOIN "{right_table}" r '
                        f'ON l."{left_col}" = r."{right_col}"'
                    ).fetchone()[0]
                except Exception:
                    direct_count = 0

                self._log("join_probe",
                          f"{join_expr} → transformed={transformed_count}, direct={direct_count}")

                # If direct join is better (more rows), correct the grounding
                if direct_count > transformed_count and direct_count > 0:
                    old_path = join_expr
                    new_path = f"{left_table}.{left_col} -> {right_table}.{right_col}"
                    grounding_context = grounding_context.replace(
                        f"  {old_path}", f"  {new_path}"
                    )
                    corrections.append(
                        f"JOIN CORRECTED: '{old_path}' → '{new_path}' "
                        f"(direct={direct_count} rows vs transformed={transformed_count})"
                    )
                    # Also remove any CONSTRAINT about the transformation
                    for constraint_pattern in [
                        r"  - .*prefix.*match.*\n",
                        r"  - .*'0'.*added.*\n",
                        r"  ⚠️ .*prefix.*\n",
                        r"  ⚠️ .*'0'.*\n",
                    ]:
                        grounding_context = re.sub(constraint_pattern, "", grounding_context, flags=re.IGNORECASE)

        finally:
            conn.close()

        if corrections:
            for c in corrections:
                self._log("join_corrected", c)

        return grounding_context

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

                    # Value not found — reformat based on actual column data
                    target_table, actual_col = candidate_tables[0]
                    try:
                        # Get sample values from the column to understand its format
                        sample_vals = conn.execute(
                            f'SELECT DISTINCT "{actual_col}" FROM "{target_table}" '
                            f'WHERE "{actual_col}" IS NOT NULL AND "{actual_col}" != \'\' '
                            f'LIMIT 10',
                        ).fetchall()
                        sample_strs = [str(r[0]) for r in sample_vals if r[0]]

                        reformatted = self._reformat_filter_value(val, sample_strs)
                        if reformatted and reformatted != val:
                            known_values[col_name] = [reformatted]
                            # Add a domain rule about how to use this value
                            domain_rules = grounding.setdefault("domain_rules", [])
                            domain_rules.append(
                                f"Value '{val}' reformatted to '{reformatted}' to match "
                                f"actual {target_table}.{actual_col} format. Use LIKE '{reformatted}%' for matching."
                            )
                            self._log("filter_reformat",
                                      f"{col_name}: '{val}' → '{reformatted}' (matched column format)")
                            break

                        # Fallback: simple LIKE match
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
            # Generic filter coverage check: for each known_values entry, verify
            # the column actually has matchable data. If not, find alternatives.
            for col_name, values in list(known_values.items()):
                if "." not in col_name:
                    continue
                if not values or not isinstance(values, (list, tuple)):
                    continue
                table_name, col_n = col_name.split(".", 1)
                str_values = [str(v) for v in values]
                # Check if ANY filter value matches in this column (exact or LIKE)
                any_match = False
                for val in str_values:
                    try:
                        cnt = conn.execute(
                            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_n}" = ?', (val,)
                        ).fetchone()[0]
                        if cnt > 0:
                            any_match = True
                            break
                        cnt2 = conn.execute(
                            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_n}" LIKE ?',
                            (f'%{val}%',)
                        ).fetchone()[0]
                        if cnt2 > 0:
                            any_match = True
                            break
                    except Exception:
                        pass
                if any_match:
                    continue
                # None of the filter values exist in this column — find alternative tables
                # Build semantic variants of the filter values (e.g., 2013-06-01 → 201306, 2013-06%, etc.)
                search_variants = list(str_values)
                for val in str_values:
                    clean = re.sub(r'[-/]', '', val)
                    if len(clean) >= 6 and clean[:6].isdigit():
                        ym = clean[:6]  # YYYYMM
                        search_variants.append(ym)
                        search_variants.append(int(ym) if ym.isdigit() else ym)
                        search_variants.append(f"{ym[:4]}-{ym[4:6]}")  # YYYY-MM
                alt_tables = []
                for tname, cols in tables_cols.items():
                    if tname.lower() == table_name.lower():
                        continue
                    for c in cols:
                        found = False
                        for variant in search_variants:
                            try:
                                cnt = conn.execute(
                                    f'SELECT COUNT(*) FROM "{tname}" WHERE "{c}" = ?', (variant,)
                                ).fetchone()[0]
                                if cnt > 0:
                                    alt_tables.append((tname, c, cnt, variant))
                                    found = True
                                    break
                                cnt2 = conn.execute(
                                    f'SELECT COUNT(*) FROM "{tname}" WHERE CAST("{c}" AS TEXT) LIKE ?',
                                    (f'{variant}%',)
                                ).fetchone()[0]
                                if cnt2 > 0:
                                    alt_tables.append((tname, c, cnt2, variant))
                                    found = True
                                    break
                            except Exception:
                                pass
                        if found:
                            break
                    if alt_tables:
                        break
                domain_rules = grounding.setdefault("domain_rules", [])
                if alt_tables:
                    alt_desc = ", ".join(
                        f"{t}.{c} ({n} rows, use value={v})" for t, c, n, v in alt_tables[:3]
                    )
                    rule = (
                        f"WARNING: {table_name}.{col_n} has NO rows matching filter values "
                        f"{str_values}. Use instead: {alt_desc}. "
                        f"JOIN to {table_name} if you still need its columns."
                    )
                else:
                    rule = (
                        f"WARNING: {table_name}.{col_n} has NO rows matching filter values "
                        f"{str_values}. This filter will return empty results."
                    )
                domain_rules.append(rule)
                self._log("filter_no_coverage", rule)

            # Check for mostly-NULL filter columns (indicates wrong column choice)
            for col_name, values in list(known_values.items()):
                if "." not in col_name:
                    continue
                if not values or not isinstance(values, (list, tuple)):
                    continue
                table_name, col_n = col_name.split(".", 1)
                # Only check numeric comparison filters (< > <= >=)
                is_comparison = any(
                    v.strip().startswith(("<", ">")) for v in values if isinstance(v, str)
                )
                if not is_comparison:
                    continue
                try:
                    total = conn.execute(
                        f'SELECT COUNT(*) FROM "{table_name}"'
                    ).fetchone()[0]
                    non_null = conn.execute(
                        f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_n}" IS NOT NULL'
                    ).fetchone()[0]
                    if total > 0 and non_null / total < 0.1:
                        domain_rules = grounding.setdefault("domain_rules", [])
                        domain_rules.append(
                            f"WARNING: {table_name}.{col_n} is NULL for {total - non_null}/{total} rows "
                            f"({100 * (total - non_null) / total:.0f}%). This column may not be the right "
                            f"filter target. Check if another table has this attribute per-record."
                        )
                        self._log("filter_null_warning",
                                  f"{table_name}.{col_n} is {100 * (total - non_null) / total:.0f}% NULL")
                except Exception:
                    pass
        finally:
            conn.close()

        grounding["known_values"] = known_values
        return grounding

    def _reformat_filter_value(self, val: str, sample_strs: list[str]) -> str:
        """Reformat a filter value to match the column's actual data format.

        Handles time format conversions like:
          '0:01:54' (h:mm:ss) → '1:54' (m:ss) when column has 'm:ss.ms' format
          '1:54' (m:ss) → '1:54' when column has 'm:ss.ms' format (use as LIKE prefix)
          Integer dates ↔ string dates
        """
        if not val or not sample_strs:
            return ""

        # Detect column format from samples
        time_ms_pattern = re.compile(r'^\d{1,2}:\d{2}\.\d+$')  # m:ss.ms (e.g., '1:26.714')
        time_hms_pattern = re.compile(r'^\d{1,2}:\d{2}:\d{2}$')  # h:mm:ss (e.g., '0:01:54')
        time_ms_short = re.compile(r'^\d{1,2}:\d{2}$')  # m:ss (e.g., '1:54')

        col_has_ms_format = any(time_ms_pattern.match(s) for s in sample_strs[:5])

        # Case 1: value is h:mm:ss ('0:01:54') but column uses m:ss.ms ('1:54.xxx')
        if time_hms_pattern.match(val) and col_has_ms_format:
            parts = val.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            total_minutes = hours * 60 + minutes
            # Convert to m:ss format for LIKE prefix matching
            return f"{total_minutes}:{seconds:02d}"

        # Case 2: value is m:ss ('1:54') but column uses m:ss.ms — already compatible as LIKE prefix
        if time_ms_short.match(val) and col_has_ms_format:
            return val

        # Case 3: integer date (20130601) vs string date ('2013-06-01')
        if re.match(r'^\d{8}$', val):
            # Check if column has 'YYYY-MM-DD' format
            if any(re.match(r'^\d{4}-\d{2}-\d{2}', s) for s in sample_strs[:5]):
                return f"{val[:4]}-{val[4:6]}-{val[6:8]}"

        if re.match(r'^\d{4}-\d{2}-\d{2}$', val):
            # Check if column has integer date format
            if any(re.match(r'^\d{8}$', s) for s in sample_strs[:5]):
                return val.replace("-", "")

        return ""


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
                        "JOIN itself returns 0 rows — the JOIN conditions are wrong. "
                        "Try joining on a different column or without transformations (no string concatenation/padding)."
                    )
                    # Show common column names between tables in the SQL
                    tables_in_sql = [r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall() if r[0].lower() in sql.lower()]
                    if len(tables_in_sql) >= 2:
                        all_cols: dict[str, list[str]] = {}
                        for t in tables_in_sql:
                            all_cols[t] = [c[1] for c in conn.execute(f'PRAGMA table_info("{t}")').fetchall()]
                        shared = set(all_cols[tables_in_sql[0]])
                        for t in tables_in_sql[1:]:
                            shared &= set(all_cols[t])
                        if shared:
                            diagnostics.append(f"  Shared column names across joined tables: {sorted(shared)}")
                        else:
                            # Show columns with similar names
                            col_sets = [(t, set(c.lower() for c in cols)) for t, cols in all_cols.items()]
                            for i_t, (t1, cs1) in enumerate(col_sets):
                                for t2, cs2 in col_sets[i_t+1:]:
                                    for c1 in cs1:
                                        for c2 in cs2:
                                            if c1 != c2 and (c1 in c2 or c2 in c1):
                                                diagnostics.append(f"  Similar columns: {t1}.{c1} ~ {t2}.{c2} — try direct join")
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
                # No single filter removal helps — likely a JOIN mismatch.
                # Test: do the filter values exist in the individual tables (without JOIN)?
                join_suspect = False
                tables_all = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                for cond in conditions:
                    if join_suspect:
                        break
                    str_vals = re.findall(r"'([^']*)'", cond)
                    for val in str_vals:
                        if not val or join_suspect:
                            break
                        for tbl in tables_all:
                            try:
                                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()]
                                for col in cols:
                                    hit = conn.execute(
                                        f'SELECT 1 FROM "{tbl}" WHERE "{col}" = ? LIMIT 1', (val,)
                                    ).fetchone()
                                    if hit:
                                        join_suspect = True
                                        diagnostics.append(
                                            f"Value '{val}' exists in {tbl}.{col} but JOIN produces 0 rows with it — "
                                            f"the JOIN condition is WRONG. Try joining on a different column or without transformations."
                                        )
                                        break
                            except Exception:
                                pass
                            if join_suspect:
                                break

                if not diagnostics:
                    diagnostics.append(
                        f"All {len(conditions)} filters COMBINED produce 0 rows with this JOIN. "
                        f"The JOIN condition is likely wrong — try a simpler/direct join."
                    )
            else:
                for cond, count_without in blockers:
                    diagnostics.append(f"REMOVE THIS FILTER: '{cond}' (without it: {count_without} rows)")
                    diagnostics.append(f"  FIX: This filter excludes all rows. Remove it or use a different column.")
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
        """Probe DB for actual values that match question filter terms.
        Only checks each value against its designated column (from grounding)."""
        if not db_path or not db_path.exists():
            return ""

        # Parse grounding filter values WITH their column names: {table.col: [values]}
        targeted: dict[str, list[str]] = {}
        if "FILTER VALUES:" in grounding_context:
            fv_section = grounding_context.split("FILTER VALUES:")[1].split("\n\n")[0]
            for line in fv_section.strip().split("\n"):
                match = re.match(r"\s*(\S+):\s*(.+)", line)
                if match:
                    col_key = match.group(1).strip()
                    vals = [v.strip().strip("'\"") for v in match.group(2).split(",")]
                    targeted[col_key] = [v for v in vals if v and len(v) >= 2]

        if not targeted:
            return ""

        conn = sqlite3.connect(str(db_path))
        discoveries: list[str] = []
        try:
            for col_key, terms in targeted.items():
                # Parse table.col format
                if "." in col_key:
                    table_name, col_name = col_key.split(".", 1)
                else:
                    continue

                # Verify table exists
                table_exists = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                ).fetchone()[0]
                if not table_exists:
                    continue

                # Verify column exists
                col_info = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
                actual_cols = [c[1] for c in col_info]
                actual_col = next((c for c in actual_cols if c.lower() == col_name.lower()), None)
                if not actual_col:
                    continue

                for term in terms:
                    try:
                        exact = conn.execute(
                            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{actual_col}" = ?',
                            (term,)
                        ).fetchone()[0]
                        if exact > 0:
                            continue

                        # Try numeric match for ID-like columns
                        if term.isdigit():
                            exact_num = conn.execute(
                                f'SELECT COUNT(*) FROM "{table_name}" WHERE "{actual_col}" = ?',
                                (int(term),)
                            ).fetchone()[0]
                            if exact_num > 0:
                                continue

                        # LIKE match only on the designated column
                        like_results = conn.execute(
                            f'SELECT DISTINCT "{actual_col}" FROM "{table_name}" '
                            f'WHERE CAST("{actual_col}" AS TEXT) LIKE ? COLLATE NOCASE LIMIT 5',
                            (f'%{term}%',)
                        ).fetchall()
                        if like_results:
                            vals = [r[0] for r in like_results if r[0]]
                            if vals:
                                discoveries.append(
                                    f"  '{term}' not found exactly in {table_name}.{actual_col}, "
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
        join_fix_rule = ""
        if diagnosis:
            diag_section = f"\nDIAGNOSIS OF FAILURES:\n{diagnosis[:1000]}"
            if "JOIN condition is WRONG" in diagnosis or "JOIN condition is likely wrong" in diagnosis or "JOIN itself returns 0 rows" in diagnosis:
                join_fix_rule = "\n- ⚠️ JOIN FIX REQUIRED: At least one hypothesis MUST use a DIRECT column-to-column join WITHOUT string concatenation, padding, or transformations (e.g., ON a.col = b.col instead of ON a.col = '0' || b.col). If two columns share a name or similar name across tables, try joining them directly."

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
- NEVER return NULL — wrap computations in COALESCE and add WHERE ... IS NOT NULL filters
- NEVER use AS to rename columns — SELECT the original column name directly (e.g. SELECT r.name, NOT SELECT r.name AS race_name){join_fix_rule}"""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if not isinstance(parsed, dict):
            return None, ""

        hypotheses = parsed.get("hypotheses", [])
        if not hypotheses:
            return None, ""

        # Filter to valid hypotheses with SQL
        valid_hyps = [h for h in hypotheses[:3] if h.get("sql", "").strip()]
        if not valid_hyps:
            return None, ""

        # Log all hypotheses for debugging
        for i, hyp in enumerate(valid_hyps):
            self._log("hypothesis_generated",
                      f"Option {i}: {hyp.get('reasoning', '')[:100]} | SQL: {hyp['sql'][:150]}")

        # LLM picks the best hypothesis by reasoning — NO execution yet
        if len(valid_hyps) == 1:
            pick_idx = 0
        else:
            options_text = ""
            for i, hyp in enumerate(valid_hyps):
                options_text += f"\nOPTION {i}:\n  Reasoning: {hyp.get('reasoning', '')[:150]}\n  SQL: {hyp['sql'][:200]}\n"

            pick_prompt = f"""QUESTION: {question}

Multiple SQL interpretations were generated. Which one BEST answers the question?

{options_text}

Return ONLY: {{"pick": 0}}  (the index of the best option)

RULES:
- Pick the option whose SQL columns and filters most directly answer what the question asks.
- "list all X" → prefer the option that SELECTs identifiers of X.
- If the question asks for a specific value, prefer the more focused interpretation.
- If the question asks for "all" or "list", prefer the broader interpretation."""

            messages = [ModelMessage(role="user", content=pick_prompt)]
            raw = self._model_call_with_retry(messages)
            pick_parsed = self._parse_json(raw)

            pick_idx = 0
            if isinstance(pick_parsed, dict) and "pick" in pick_parsed:
                idx = pick_parsed["pick"]
                if isinstance(idx, int) and 0 <= idx < len(valid_hyps):
                    pick_idx = idx

        # Execute ONLY the chosen hypothesis
        chosen = valid_hyps[pick_idx]
        self._log("hypothesis_picked", f"Option {pick_idx}: {chosen.get('reasoning', '')[:80]}")
        self._log("hypothesis_sql", chosen["sql"])
        result = self._try_sql(db_path, chosen["sql"])

        if result and result.get("rows"):
            all_null = all(
                all(v is None or str(v).strip().lower() in ("none", "null", "") for v in row)
                for row in result["rows"]
            )
            if not all_null:
                self._log("hypothesis_result",
                          f"cols={result['columns']}, rows={len(result['rows'])}, "
                          f"sample={result['rows'][:3]}")
                return result, chosen["sql"]

        self._log("hypothesis_empty", f"Option {pick_idx} returned no data — trying others")

        # If chosen one failed, try the others in order
        for i, hyp in enumerate(valid_hyps):
            if i == pick_idx:
                continue
            self._log("hypothesis_fallback_try", f"Option {i}: {hyp['sql'][:150]}")
            result = self._try_sql(db_path, hyp["sql"])
            if result and result.get("rows"):
                all_null = all(
                    all(v is None or str(v).strip().lower() in ("none", "null", "") for v in row)
                    for row in result["rows"]
                )
                if not all_null:
                    self._log("hypothesis_fallback",
                              f"Option {i} succeeded: cols={result['columns']}, "
                              f"rows={len(result['rows'])}, sample={result['rows'][:3]}")
                    return result, hyp["sql"]

        return None, ""

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


        # Fix 4: Detect error strings or None/NULL values in result and retry
        first_row_str = " ".join(str(v) for v in rows[0]) if rows else ""
        has_none_values = any(str(v).lower() in ("none", "null", "") for v in rows[0]) if rows else False
        # Only flag None as suspicious if question asks for names/descriptions (not counts/aggregations)
        name_indicators = ["name", "who", "full name", "surname", "title", "display"]
        none_is_suspicious = has_none_values and any(w in q_lower for w in name_indicators)

        if "error" in first_row_str.lower() or first_row_str.strip() in ("0", "0.0", "0.00") or none_is_suspicious:
            # Check if result is suspicious (error text or zero when expecting real data)
            has_error = "error" in first_row_str.lower()
            if has_error or none_is_suspicious:
                issue_desc = "error/null values" if none_is_suspicious else "error string"
                self._log("shape_fix_error", f"Result contains {issue_desc}: {first_row_str[:60]}")
                if none_is_suspicious:
                    fix_prompt = f"""The SQL returned NULL/None values for columns that should have real data (names, descriptions, etc.).

QUESTION: {question}
CURRENT RESULT: columns={cols}, values={rows[0]}

The NULL values likely mean: a column name is WRONG (e.g., 'name' doesn't exist but 'first_name'+'last_name' do), or the JOIN failed.
Check the DATABASE SCHEMA carefully for the ACTUAL column names and fix the query.

DATABASE SCHEMA:
{kg_context[:2000]}

{grounding_context[:1000]}

Write a corrected SQL using the EXACT column names from the schema.
Return ONLY: {{"sql": "SELECT ..."}}"""
                else:
                    fix_prompt = f"""The SQL returned an error value instead of real data.

QUESTION: {question}
CURRENT RESULT: columns={cols}, values={rows[0]}

This is wrong. The result should be a meaningful number or value, not an error.
Possible issues: division by zero, NULL in computation, wrong column type.

DATABASE SCHEMA:
{kg_context[:2000]}

{grounding_context[:1000]}

Write a SIMPLER SQL that avoids the computation error. Use NULLIF for division, COALESCE for NULLs.
Return ONLY: {{"sql": "SELECT ..."}}"""
                messages = [ModelMessage(role="user", content=fix_prompt)]
                raw = self._model_call_with_retry(messages)
                parsed = self._parse_json(raw)
                if isinstance(parsed, dict) and parsed.get("sql"):
                    fix_result = self._try_sql(db_path, parsed["sql"])
                    if fix_result and fix_result.get("rows"):
                        fix_str = " ".join(str(v) for v in fix_result["rows"][0])
                        fix_nones = sum(1 for v in fix_result["rows"][0] if str(v).lower() in ("none", "null", ""))
                        orig_nones = sum(1 for v in rows[0] if str(v).lower() in ("none", "null", ""))
                        if "error" not in fix_str.lower() and fix_nones < orig_nones:
                            self._log("shape_fixed_error", f"Fixed: {fix_result['rows'][0]}")
                            return fix_result

        # Fix 2: Detect raw FK IDs in output and re-query with JOIN for human-readable values
        if rows and cols:
            has_raw_id = False
            raw_id_cols = []
            for i, col in enumerate(cols):
                sample_vals = [str(rows[r][i]) for r in range(min(len(rows), 3))]
                # Detect patterns: "rec..." (Airtable IDs), all-digit long strings unlikely to be answers
                for v in sample_vals:
                    if v and (v.startswith("rec") and len(v) > 10) or \
                       (col.lower().endswith("_id") and not any(w in q_lower for w in [col.lower(), "id"])):
                        has_raw_id = True
                        raw_id_cols.append((i, col, sample_vals[0]))
                        break

            if has_raw_id and len(raw_id_cols) > 0:
                id_desc = ", ".join(f"'{c}' has values like '{v}'" for _, c, v in raw_id_cols)
                self._log("shape_fix_fk", f"Raw IDs detected: {id_desc}")
                fix_prompt = f"""The SQL result contains raw foreign key IDs instead of human-readable names.

QUESTION: {question}
CURRENT RESULT: columns={cols}, sample row={rows[0]}
RAW ID COLUMNS: {id_desc}

The user expects human-readable names/descriptions, not internal IDs. Add a JOIN to resolve these IDs to their display values.

DATABASE SCHEMA:
{kg_context[:2000]}

Return ONLY: {{"sql": "SELECT ..."}}"""
                messages = [ModelMessage(role="user", content=fix_prompt)]
                raw = self._model_call_with_retry(messages)
                parsed = self._parse_json(raw)
                if isinstance(parsed, dict) and parsed.get("sql"):
                    fix_result = self._try_sql(db_path, parsed["sql"])
                    if fix_result and fix_result.get("rows"):
                        new_vals = [str(v) for v in fix_result["rows"][0]]
                        # Verify at least one raw ID was resolved
                        old_vals = [str(rows[0][idx]) for idx, _, _ in raw_id_cols]
                        if any(nv != ov for nv, ov in zip(new_vals, old_vals)):
                            self._log("shape_fixed_fk", f"Resolved IDs: {fix_result['rows'][0]}")
                            return fix_result

        # Check: "X and Y" pattern expects 2+ columns but we got 1
        and_pattern = re.search(
            r'(?:what is|identify|find)\s+(?:the\s+)?(\w+).+?\band\b\s+(?:the\s+)?(\w+)',
            q_lower,
        )
        if and_pattern and len(cols) == 1 and len(rows) == 1:
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
        """Call model with 60s timeout. Returns empty string on timeout/API error."""
        try:
            result = self.model.complete(messages)
            return result if result else ""
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "timeout" in err_msg or "connection" in err_msg or "api" in err_msg:
                self._log("llm_error", f"LLM call failed: {str(e)[:100]}")
                return ""
            raise

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
