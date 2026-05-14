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
    classify_columns_with_llm,
    discover_joins_with_llm,
    enrich_kg_with_descriptions,
    format_kg_for_llm,
)
from data_agent_baseline.pipeline.kg_path_planner import (
    QueryNode,
    QueryPath,
    QueryPlan,
    build_query_path,
    map_phrase_to_columns,
    map_value_to_column,
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
    ("singular_plural", "SINGULAR: Don't add LIMIT 1 just because grammar is singular. Only LIMIT 1 for 'most recent'/'first' OR when CONSTRAINTS explicitly say LIMIT."),
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
    ("quote_cols", "COLUMN QUOTING: Always double-quote column names that contain spaces or special characters (e.g. \"School Name\", \"District Type\")."),
]

# Full rule texts for backward compatibility
SQL_RULES = [rule for _, rule in SQL_RULES_LABELED]


def _build_sql_prompt(
    *,
    question: str,
    kg_context: str,
    column_hints: str = "",
    gaps: str = "",
    grounding_context: str = "",
) -> str:
    parts = [f"QUESTION: {question}\n\nWrite a SQL query to answer this question."]

    if grounding_context:
        parts.append(f"\n{grounding_context}")

    if gaps:
        parts.append(f"\nPREVIOUS ATTEMPT FAILED:\n{gaps}")

    rules = (
        "RULES:\n"
        "- Use ONLY columns from SCHEMA above. Quote all identifiers with double-quotes.\n"
        "- SELECT only asked columns. No SELECT *.\n"
        "- Use simple column aliases (AS) for computed columns (e.g., COUNT(*) AS count, AVG(x) AS avg_x).\n"
        "- For list/lookup queries ('what are', 'list', 'identify'), always use SELECT DISTINCT.\n"
        "- For superlatives (lowest/highest/best), use WHERE col = (SELECT MIN/MAX(col)...) to include ALL ties — no LIMIT.\n"
        "- JOIN through JOIN PATHS shown above — do not invent join conditions.\n"
        "- Use exact FILTER VALUES as-is. Only use LIKE when FILTER VALUES says 'USE → WHERE ... LIKE ...'.\n"
        "- CAST(x AS REAL) for division.\n"
        "- WHERE col IS NOT NULL to avoid nulls. Escape apostrophes with ''.\n"
        "- Never transform, concatenate, or split column values."
    )

    if gaps:
        rules += "\n- Fix what the error says."

    parts.append(f"\n{rules}")
    parts.append('\nReturn ONLY: {"thought": "...", "sql": "SELECT ..."}')

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
{knowledge_guidance_section}
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
  "formula": "computation LOGIC only — no literal filter values. e.g. 'event with lowest SUM(cost)', 'AVG(Consumption) / 12'. Filter values go in known_values, not here.",
  "computation_steps": ["step1", "step2"],
  "data_requirements": ["table.column — ALL columns relevant to question, joins, filters, aggregation"],
  "data_format_notes": ["only note actual column types like REAL/TEXT/DATE — never suggest transformations"],
  "reasoning": "brief HOW to get the answer",
  "domain_rules": ["constraints from DOMAIN KNOWLEDGE"],
  "known_values": {{"table.column": ["verified filter values"]}}
}}

RULES:
- what_user_wants drives everything. Do NOT invent columns the question didn't ask for.
- COLUMN SEMANTICS: When choosing between columns, check SAMPLE VALUES in the schema. The column whose actual data values match what the question asks for wins. Column hints are suggestions — sample values are ground truth.
- NO ASSUMPTIONS: Do NOT add data_format_notes or domain_rules that assume conversion/transformation unless DOMAIN KNOWLEDGE explicitly states it. If a REFERENCE SQL is provided, follow its approach exactly (e.g. if it uses ORDER BY col ASC, do NOT add "must convert to seconds").
- NO DATA MANIPULATION: Never transform, concatenate, split, or convert column values. Return data exactly as it exists in the database.
- FORMULA AUTHORITY: If DOMAIN KNOWLEDGE defines a formula, copy it VERBATIM into "formula" field. Do NOT reason about whether any part is redundant — every operation is intentional.
- USE CASE AUTHORITY: If DOMAIN KNOWLEDGE has a matching USE CASE or REFERENCE SQL, copy its logic EXACTLY.
- EXACT LEVEL MATCHING: Named levels ("high=1", "medium=2") → use ONLY the level matching the question's exact wording. No combining unless "X or above".
- For known_values: include TABLE name. Reason about what DB value(s) the question refers to — consider format differences and precision level. NEVER use the same literal value for multiple columns — each concept maps to exactly ONE column.
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
- HAVING: "where the average exceeds N" = GROUP BY + HAVING, not per-row WHERE. EXCEPTION: if DATA STORAGE NOTES say a column already stores a pre-row aggregate, use WHERE directly — do NOT wrap in AVG().
- SUPERLATIVES: "lowest/highest" → rows = "all-matching". Use WHERE col = (SELECT MIN/MAX...). NEVER LIMIT 1.
- SUBJECT vs CRITERION: The question's SUBJECT is what the user wants returned. The CRITERION (superlative, filter, condition) is how to find it. what_user_wants must describe the SUBJECT's identity, not the criterion value. expected_output must include the SUBJECT's identifier column.
- CO-LOCATED MEASURES: Filter in detail table → use measure from SAME detail table, not parent summary.
- COLUMN NAME PRIORITY: If a column name contains a multi-word phrase from the question, it wins over columns that only share a single word. Longest substring match wins.
- VALUE-BASED DISAMBIGUATION: When multiple columns could map to one phrase, check their SAMPLE VALUES. The column whose values answer the question wins over the column whose name merely contains a keyword.
- PARTIAL MATCH: "X-related" / "from X" with proper noun → LIKE '%X%'. Exact match only for "named X" / "is X".
- SAME-NAME COLUMNS: When multiple tables have the SAME column name, read the question language to decide which table to SELECT from. "the patient's X" → Patient.X. "the exam's X" → Exam.X. The subject/possessive in the question is authoritative.
- PHRASE MAPPING: "the type of X" / "identify the type" = SELECT the literal `type` column of the entity being asked about. This is a LOOKUP, not a GROUP BY. Only use GROUP BY if the question says "for each type" / "by type" / "per type" / "breakdown".
- FULL SENTENCE PARSING: Map columns from the COMPLETE sentence structure, not partial phrases. Parse the full question as a whole before extracting column mappings.
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
    ambiguous_columns: str = "",
    user_intent: str = "",
    knowledge_guidance: str = "",
) -> str:
    # Budget: keep total prompt under 8000 chars to avoid Qwen timeouts
    BUDGET = 8000
    template_len = len(SEMANTIC_GROUNDING_PROMPT) + len(question) * 2
    kg_guidance_section = ""
    if knowledge_guidance:
        kg_guidance_section = (
            f"\nCOLUMN HINTS (verify against SAMPLE VALUES in schema before using):\n"
            f"{knowledge_guidance[:1500]}"
        )
    remaining = BUDGET - template_len - len(anchor_text[:1500]) - len(previous_attempt) - len(ambiguous_columns) - len(user_intent) - len(kg_guidance_section)

    # Schema gets priority, sample data fills remainder
    schema_budget = max(remaining - 1500, 2000)
    kg_trimmed = _trim_schema_by_relevance(kg_context, question, schema_budget, anchor_text)
    sample_budget = max(remaining - len(kg_trimmed), 500)
    sample_section = f"\nSAMPLE DATA:\n{sample_data[:sample_budget]}" if sample_data else ""
    anchor_section = f"\nDOMAIN KNOWLEDGE:\n{anchor_text[:1500]}" if anchor_text else ""
    ambiguous_section = f"\n{ambiguous_columns}" if ambiguous_columns else ""
    # When ambiguous columns exist AND entity_of_interest is specified, reinforce output table choice
    intent_section = ""
    if user_intent:
        # Pass shape/grain/entity to grounding, but NOT metric column — grounding resolves columns
        # from the focused schema with sample values, which is more authoritative.
        intent_for_grounding = "\n".join(
            ln for ln in user_intent.split("\n")
            if not ln.startswith("Metric (SELECT):")
        )
        intent_section = f"\n⚠️ USER INTENT (your grounding MUST respect these constraints):\n{intent_for_grounding}"
        if ambiguous_columns and "prefer this table's columns for output" in user_intent:
            intent_section += "\n⚠️ For AMBIGUOUS COLUMNS above: SELECT output from the entity_of_interest table, not the filter table."
    constraints_section = f"\n{feedback}" if feedback else ""
    prev_section = f"\nPREVIOUS ATTEMPT (fix the issues below):\n{previous_attempt}" if previous_attempt else ""
    return SEMANTIC_GROUNDING_PROMPT.format(
        question=question,
        kg_context=kg_trimmed,
        sample_section=sample_section,
        anchor_section=anchor_section + ambiguous_section + intent_section + constraints_section,
        previous_attempt=prev_section,
        knowledge_guidance_section=kg_guidance_section,
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
        # Only show column expectation if it's a descriptive string, not a raw count
        # (numeric counts mislead the planner into concatenating columns to fit)
        if col_expect and not str(col_expect).strip().isdigit():
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
        display_formula = re.sub(r'\s+LIMIT\s+\d+\s*$', '', formula, flags=re.IGNORECASE)
        parts.append(f"REFERENCE FORMULA (logic only — use FILTER VALUES for actual values):\n  {display_formula.strip()}")

    # Join paths (factual FK relationships)
    join_paths = grounding.get("join_paths", [])
    if join_paths:
        parts.append("JOIN PATHS:\n" + "\n".join(f"  {jp}" for jp in join_paths))

    # Filter values
    known_values = grounding.get("known_values", {})
    filter_overrides = grounding.get("filter_overrides", {})
    if known_values:
        kv_lines = []
        for k, vs in known_values.items():
            if not vs:
                continue
            if k in filter_overrides:
                kv_lines.append(f"  {k}: USE → WHERE {filter_overrides[k]}")
            else:
                kv_lines.append(f"  {k}: {', '.join(str(v) for v in vs)}")
        if kv_lines:
            parts.append("FILTER VALUES:\n" + "\n".join(kv_lines))

    # data_requirements not shown — SQL LLM has EXPECTED COLUMNS, FORMULA, FILTER VALUES, JOIN PATHS

    # Data format warnings
    format_notes = grounding.get("data_format_notes", [])

    # Domain constraints (non-override rules from grounding)
    # Drop rules about literal/exact matching when format warnings contradict them
    # Drop rules that mention columns blocked by filter dedup
    domain_rules = grounding.get("domain_rules", [])
    regular_rules = [r for r in domain_rules if r not in override_rules]
    if format_notes and regular_rules:
        regular_rules = [r for r in regular_rules if not any(
            kw in r.lower() for kw in ("literal", "exact", "do not convert", "compare them as")
        )]
    blocked_cols = grounding.get("_blocked_filter_cols", [])
    if blocked_cols and regular_rules:
        regular_rules = [r for r in regular_rules if not any(
            col.lower() in r.lower() for col in blocked_cols
        )]
    if regular_rules:
        parts.append("CONSTRAINTS:\n" + "\n".join(f"  - {r}" for r in regular_rules))

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





def _fix_unescaped_apostrophes(sql: str) -> str:
    """Fix unescaped apostrophes inside single-quoted SQL string literals.

    'Women's Soccer' → 'Women''s Soccer'
    Handles multiple literals in one statement.
    """
    result = []
    i = 0
    while i < len(sql):
        if sql[i] == "'":
            # Find the end of this string literal
            # Walk forward collecting chars; an apostrophe followed by a letter
            # (not another apostrophe and not end-of-token) is unescaped
            result.append("'")
            i += 1
            while i < len(sql):
                if sql[i] == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    # Already escaped — keep both
                    result.append("''")
                    i += 2
                elif sql[i] == "'":
                    # Could be end of literal or unescaped apostrophe
                    # Heuristic: if next char is a word char (letter/digit) and prev char is
                    # also a word char, it's an unescaped apostrophe mid-word (e.g. Women's)
                    prev_is_word = i > 0 and (sql[i - 1].isalnum() or sql[i - 1] == ' ')
                    next_is_word = i + 1 < len(sql) and sql[i + 1].isalpha()
                    if prev_is_word and next_is_word:
                        result.append("''")
                        i += 1
                    else:
                        # End of literal
                        result.append("'")
                        i += 1
                        break
                else:
                    result.append(sql[i])
                    i += 1
        else:
            result.append(sql[i])
            i += 1
    return "".join(result)


def _sanitize_sql(sql: str, db_path: Path) -> str:
    """Fix common LLM SQL formatting issues: trailing junk, unquoted multi-word columns, unescaped apostrophes."""
    # Strip trailing braces/brackets that leak from JSON
    sql = sql.rstrip().rstrip("}").rstrip("]").rstrip()
    # Remove trailing semicolons (SQLite doesn't need them and they can cause issues with multiple statements)
    sql = sql.rstrip(";").strip()

    # Fix unescaped apostrophes inside single-quoted string literals
    # e.g. 'Women's Soccer' → 'Women''s Soccer'
    sql = _fix_unescaped_apostrophes(sql)

    # Quote unquoted multi-word column names using actual schema
    try:
        conn = sqlite3.connect(str(db_path))
        all_columns: set[str] = set()
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            for col in conn.execute(f'PRAGMA table_info("{row[0]}")').fetchall():
                col_name = col[1]
                if " " in col_name or "(" in col_name or "%" in col_name or "-" in col_name:
                    all_columns.add(col_name)
        conn.close()

        # For each multi-word column, find unquoted references and quote them
        for col in sorted(all_columns, key=len, reverse=True):
            # Match the column name not already inside quotes
            # CamelCase collapsed version (e.g. SchoolName for "School Name")
            no_space = col.replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
            if no_space in sql:
                sql = sql.replace(no_space, f'"{col}"')
            # Also fix dot-prefixed versions (e.g. frpm.SchoolName)
            for prefix in ("frpm.", "satscores.", "T1.", "T2."):
                if f"{prefix}{no_space}" in sql:
                    sql = sql.replace(f"{prefix}{no_space}", f'{prefix}"{col}"')
    except Exception:
        pass

    return sql


def _apply_null_guard(sql: str) -> str:
    """Add IS NOT NULL + != '' for columns used in ORDER BY LIMIT 1 or subquery MIN/MAX."""
    guarded = sql

    # Pattern 1: ORDER BY <col> ASC/DESC LIMIT 1
    order_match = re.search(
        r'ORDER\s+BY\s+(\w+(?:\.\w+)?)\s+(ASC|DESC)\s+LIMIT\s+1',
        guarded, re.IGNORECASE
    )
    if order_match:
        col = order_match.group(1)
        guarded = _inject_null_check(guarded, col, order_match.start())

    # Pattern 2: WHERE col = (SELECT MIN/MAX(col) ...) — guard the subquery
    # Handles both quoted ("table"."col") and unquoted (table.col) identifiers
    minmax_match = re.search(
        r'\(\s*SELECT\s+(MIN|MAX)\s*\(\s*("?\w+"?(?:\."?\w+"?)?)\s*\)\s+FROM\s+("?\w+"?)',
        guarded, re.IGNORECASE
    )
    if minmax_match:
        col = minmax_match.group(2)
        table = minmax_match.group(3)
        # Add WHERE col != '' inside the subquery if not already there
        subq_start = minmax_match.start()
        subq_end = guarded.find(")", subq_start + 1)
        if subq_end == -1:
            subq_end = len(guarded)
        # Find the closing paren of the subquery
        depth = 0
        for i in range(subq_start, len(guarded)):
            if guarded[i] == '(':
                depth += 1
            elif guarded[i] == ')':
                depth -= 1
                if depth == 0:
                    subq_end = i
                    break
        subquery = guarded[subq_start:subq_end + 1]
        bare_col = col.split(".")[-1].strip('"') if "." in col else col.strip('"')
        subq_check = subquery.lower().replace('"', '')
        has_not_null = bare_col.lower() + ' is not null' in subq_check
        has_not_empty = bare_col.lower() + " != ''" in subq_check or bare_col.lower() + " <> ''" in subq_check

        if not has_not_null and not has_not_empty:
            if re.search(r'\bWHERE\b', subquery, re.IGNORECASE):
                new_subq = subquery[:-1] + f' AND {col} IS NOT NULL AND {col} != \'\')'
            else:
                new_subq = subquery[:-1] + f' WHERE {col} IS NOT NULL AND {col} != \'\')'
            guarded = guarded[:subq_start] + new_subq + guarded[subq_end + 1:]
        elif has_not_null and not has_not_empty:
            new_subq = subquery[:-1] + f' AND {col} != \'\')'
            guarded = guarded[:subq_start] + new_subq + guarded[subq_end + 1:]
        elif not has_not_null and has_not_empty:
            new_subq = subquery[:-1] + f' AND {col} IS NOT NULL)'
            guarded = guarded[:subq_start] + new_subq + guarded[subq_end + 1:]

    return guarded


def _inject_null_check(sql: str, col: str, insert_before: int) -> str:
    """Inject IS NOT NULL AND != '' for col before the given position."""
    if re.search(rf'{re.escape(col)}\s+IS\s+NOT\s+NULL', sql, re.IGNORECASE):
        if re.search(rf"{re.escape(col)}\s*!=\s*''", sql, re.IGNORECASE):
            return sql
        prefix = sql[:insert_before].rstrip()
        suffix = sql[insert_before:]
        return f"{prefix} AND {col} != '' {suffix}"
    prefix = sql[:insert_before].rstrip()
    suffix = sql[insert_before:]
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
            kg = discover_joins_with_llm(kg, model=self.model, log_fn=self._log)
            kg = classify_columns_with_llm(kg, model=self.model, log_fn=self._log)
            kg_context = format_kg_for_llm(kg)
            g = kg.graph
            self._log("kg_built", (
                f"KG: {len(kg.tables)} tables, {len(kg.inferred_fks)} inferred FKs\n"
                f"  Graph: {len(g.columns)} columns, {len(g.values)} value nodes, "
                f"{len(g.fk_edges)} FK edges, {len(g.semantic_edges)} semantic edges\n"
                f"  Value index: {len(g.value_index)} unique values indexed"
            ))
            if g.fk_edges:
                fk_summary = "; ".join(
                    f"{e.src}→{e.dst} ({e.overlap_ratio:.0%})"
                    for e in g.fk_edges[:5]
                )
                self._log("kg_fk_edges", fk_summary)
            if g.semantic_edges:
                sem_summary = "; ".join(
                    f"{e.src}~{e.dst} ({e.similarity_score:.2f})"
                    for e in g.semantic_edges[:5]
                )
                self._log("kg_semantic_edges", sem_summary)

            # Get sample data for each table (question-aware probing)
            sample_data = self._get_sample_data(db_path, kg, question)

            # Step 5: KG path planning — graph-based reasoning to reach the goal
            grounding_context = self._kg_path_plan_grounding(
                question, kg_context, sample_data, ctx.knowledge_text,
                db_path=db_path, kg=kg,
            )

            # Step 5b: Value Discovery — probe DB for actual filter values
            value_discovery = self._discover_filter_values(
                question, db_path, kg, grounding_context, ctx.knowledge_text,
            )
            if value_discovery:
                self._log("value_discovery", value_discovery)

            # Step 5c: Threshold inference — infer normal/abnormal ranges if needed
            threshold_context = self._infer_thresholds(
                question, db_path, kg, ctx.knowledge_text,
            )
            if threshold_context:
                self._log("threshold_inference", threshold_context)

            # Inject discovered values into grounding context
            if value_discovery:
                grounding_context += f"\n\nDISCOVERED VALUES (actual DB values for filter terms):\n{value_discovery}"
            if threshold_context:
                grounding_context += f"\n\n{threshold_context}"

            # ----------------------------------------------------------
            # SQL Generation: LLM writes SQL, close-loop on failure
            # ----------------------------------------------------------
            max_sql_attempts = 3
            sql = ""
            data_result = None
            failed_sqls: list[str] = []
            gaps = ""

            for attempt in range(max_sql_attempts):
                sql = self._call_sql(
                    question,
                    grounding_context=grounding_context,
                    gaps=gaps,
                )
                if not sql:
                    break

                sql = _sanitize_sql(sql, db_path)
                sql = _apply_null_guard(sql)
                self._log("sql_generated" if attempt == 0 else f"sql_retry_{attempt}", sql)
                data_result = self._try_sql(db_path, sql)

                if data_result and data_result.get("rows"):
                    break

                # Diagnose failure for next iteration
                failed_sqls.append(sql)
                last_error = next(
                    (s.get("detail", "") for s in reversed(self.steps) if s.get("event") == "sql_error"),
                    "",
                )
                if last_error:
                    gaps = f"- SQL ERROR: {last_error}\n- Failed SQL: {sql}"
                elif data_result is not None:
                    # Executed OK but 0 rows
                    diagnosis = self._diagnose_empty_result(db_path, sql) if sql else ""
                    gaps = f"- ZERO ROWS returned.\n- Failed SQL: {sql}"
                    if diagnosis:
                        gaps += f"\n- DIAGNOSIS: {diagnosis}"
                else:
                    break

            # Last resort: if close-loop exhausted, try multi-hypothesis approach
            if not (data_result and data_result.get("rows")):
                self._log("hypothesis_trigger", "SQL close-loop exhausted — trying multi-hypothesis fallback")
                diagnosis = self._diagnose_empty_result(db_path, sql) if sql else ""
                hyp_result, hyp_sql = self._try_multi_hypothesis(
                    question=question,
                    db_path=db_path,
                    kg_context=kg_context,
                    sample_data=sample_data,
                    knowledge_text=ctx.knowledge_text,
                    failed_sqls=failed_sqls,
                    diagnosis=diagnosis,
                )
                if hyp_result and hyp_result.get("rows"):
                    data_result = hyp_result
                    self._log("hypothesis_success", f"Hypothesis produced {len(hyp_result['rows'])} rows")

            # Shape validation before formatting
            if data_result and data_result.get("rows"):
                data_result = self._validate_result_shape(
                    question, data_result, db_path, kg_context,
                    sample_data, ctx.knowledge_text,
                    grounding_context=grounding_context,
                    column_hints="",
                )

            # Format answer
            raw_row_count = len(data_result.get("rows", [])) if data_result else 0
            self._log("pre_answer", f"cols={data_result.get('columns') if data_result else None}, rows={raw_row_count}")
            if data_result and data_result.get("rows"):
                answer = self._call_answer_with_schema(
                    question, data_result, ctx.knowledge_text,
                    grounding_context=grounding_context,
                )
                if not answer or not answer.get("rows"):
                    answer = self._raw_result_to_answer(data_result)
            else:
                answer = {"columns": [], "rows": []}

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
    # LLM Call 1: SQL Generation
    # ------------------------------------------------------------------

    def _call_sql(
        self, question: str, kg_context: str = "",
        gaps: str = "", column_hints: str = "",
        grounding_context: str = "",
    ) -> str:
        prompt = _build_sql_prompt(
            question=question,
            kg_context=kg_context,
            column_hints=column_hints,
            gaps=gaps,
            grounding_context=grounding_context,
        )

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        if not raw:
            self._log("sql_call_empty", "LLM returned empty response")
            return ""
        parsed = self._parse_json(raw)

        if isinstance(parsed, dict):
            sql = parsed.get("sql") or parsed.get("query") or ""
            if not sql:
                for v in parsed.values():
                    if isinstance(v, str) and v.strip().upper().startswith("SELECT"):
                        sql = v
                        break
            if not sql:
                self._log("sql_parse_failed", f"No sql key in: {list(parsed.keys())} | raw={raw}")
            return sql
        # Try extracting SQL directly from raw text
        select_match = re.search(r'(SELECT\s.+?)(?:```|"|\Z)', raw, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip().rstrip('"').rstrip("'")
        self._log("sql_parse_failed", f"raw={raw}")
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

        # Detect superlative pattern to deterministically drop criterion column
        sup_match = re.search(
            r'\b(?:which|what|who)\b.+?\b(?:has|have|with|had)\b.+?\b(?:the\s+)?(?:lowest|highest|most|least|best|worst|fastest|slowest|largest|smallest|longest|shortest)\b\s+(\w+)',
            question.lower(),
        )
        criterion_col = sup_match.group(1) if sup_match else ""

        # If we can deterministically identify the criterion column, just drop it
        if criterion_col and len(columns) == 2:
            criterion_idx = next(
                (i for i, c in enumerate(columns) if criterion_col in c.lower()),
                None,
            )
            if criterion_idx is not None:
                keep_idx = 1 - criterion_idx
                self._log("answer_schema", f"Kept columns [{keep_idx}] → ['{columns[keep_idx]}'] ({len(rows)} rows)")
                return {
                    "columns": [columns[keep_idx]],
                    "rows": [[str(row[keep_idx])] for row in rows],
                }

        prompt = f"""The user asked a question. The SQL returned these columns. Which columns should appear in the final answer?

QUESTION: {question}
USER INTENT: {user_wants or question}

SQL RESULT COLUMNS:
{col_list}

Return ONLY: {{"keep_columns": [0, 2]}}

RULES:
- Keep columns the user explicitly asked to SEE in the answer.
- "full name" = keep BOTH first_name AND last_name (or all name-related columns).
- If the question asks for multiple attributes ("list X and Y"), keep ALL of them.
- REMOVE criterion/sorting columns used only to find the answer. "Which X has the lowest Y?" → keep X, drop Y. "What is the Y of X?" → keep Y, drop X's ID.
- REMOVE internal IDs and filter-echo columns (constant values from WHERE clause).
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
                # Extract just the WHERE clause for filter values — don't show full SQL
                where_match = re.search(r'WHERE\s+(.+)', uc_sql, re.IGNORECASE)
                where_hint = where_match.group(1).strip() if where_match else ""
                deterministic_parts.append(
                    f"MATCHING USE CASE (score={best_score}):\n"
                    f"  Title: {best_use_case[0]}\n"
                    f"  Explanation: {best_use_case[2]}\n"
                    f"  Filter condition: {where_hint}\n"
                    f"  ⚠️ WARNING: The original SQL for this use case is INVALID ({uc_error}). "
                    f"ONLY use the filter condition above. For SELECT columns, use the actual DATABASE SCHEMA and question's entity ownership."
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

    # ------------------------------------------------------------------
    # KG Path Planning Loop: graph-based reasoning to reach the goal
    # ------------------------------------------------------------------

    def _kg_path_plan_grounding(
        self,
        question: str,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
        db_path: Path | None = None,
        kg: KnowledgeGraph | None = None,
    ) -> str:
        """LLM picks nodes from property graph → validate → format as grounding for SQL LLM."""
        if not db_path or not kg:
            return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        # --- Step 1: Domain anchors + user intent ---
        anchor_text = self._extract_domain_anchors(question, knowledge_text, db_path=db_path)
        user_intent = self._detect_user_intent_only(question, kg_context=kg_context, anchor_text=anchor_text)
        if user_intent:
            self._log("user_intent", user_intent)

        # --- Step 2: LLM picks nodes from graph (1 call) ---
        picked = self._pick_graph_nodes(question, kg, anchor_text, user_intent)
        if not picked:
            self._log("kg_path", "Node picking failed, falling back")
            return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        self._log("kg_picked", json.dumps(picked, default=str))

        # --- Step 3: Validate picks against graph (deterministic) ---
        output_nodes, filter_nodes, errors = self._validate_picked_nodes(picked, kg, db_path)
        if errors:
            self._log("kg_validation_errors", "; ".join(errors))
        if not output_nodes:
            self._log("kg_path", "No valid output nodes after validation, falling back")
            return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        self._log("kg_output_nodes", ", ".join(f"{n.table}.{n.column}" for n in output_nodes))
        self._log("kg_filter_nodes", ", ".join(
            f"{n.table}.{n.column}{n.operator}{n.value}" for n in filter_nodes
        ))

        # --- Step 3a: Sanity check — does the pick match the question? ---
        sanity_issues = self._sanity_check_picks(
            question, picked, output_nodes, filter_nodes, user_intent, kg, anchor_text,
        )
        if sanity_issues:
            self._log("kg_sanity", sanity_issues)
            # Re-prompt with sanity feedback
            repicked = self._pick_graph_nodes(
                question, kg, anchor_text,
                user_intent + f"\n\nCORRECTIONS (from sanity check):\n{sanity_issues}",
            )
            if repicked and repicked.get("select_columns"):
                self._log("kg_repicked", json.dumps(repicked, default=str))
                new_output, new_filter, _ = self._validate_picked_nodes(repicked, kg, db_path)
                if new_output:
                    picked = repicked
                    output_nodes = new_output
                    filter_nodes = new_filter
                    self._log("kg_output_nodes", ", ".join(f"{n.table}.{n.column}" for n in output_nodes))
                    self._log("kg_filter_nodes", ", ".join(
                        f"{n.table}.{n.column}{n.operator}{n.value}" for n in filter_nodes
                    ))

        # --- Step 3b: Probe actual value formats for text filters ---
        filter_nodes = self._probe_filter_values(filter_nodes, db_path)

        # --- Step 3b1: Domain column fixes (deterministic swap) ---
        filter_nodes = self._apply_domain_column_fixes(question, filter_nodes, kg, anchor_text)

        # --- Step 3b2: Check filter discriminating power ---
        weak_filters = self._check_filter_selectivity(filter_nodes, db_path)
        if weak_filters:
            self._log("kg_weak_filter", weak_filters)
            repicked2 = self._pick_graph_nodes(
                question, kg, anchor_text,
                user_intent + f"\n\nCORRECTIONS:\n{weak_filters}",
            )
            if repicked2 and repicked2.get("select_columns"):
                self._log("kg_repicked", json.dumps(repicked2, default=str))
                new_output2, new_filter2, _ = self._validate_picked_nodes(repicked2, kg, db_path)
                if new_output2:
                    picked = repicked2
                    output_nodes = new_output2
                    filter_nodes = self._probe_filter_values(new_filter2, db_path)
                    filter_nodes = self._apply_domain_column_fixes(question, filter_nodes, kg, anchor_text)
                    self._log("kg_output_nodes", ", ".join(f"{n.table}.{n.column}" for n in output_nodes))
                    self._log("kg_filter_nodes", ", ".join(
                        f"{n.table}.{n.column}{n.operator}{n.value}" for n in filter_nodes
                    ))

        # --- Step 3c: Extract order_by column as a node for path planning ---
        order_nodes: list[QueryNode] = []
        order_by = picked.get("order_by")
        if order_by and isinstance(order_by, dict):
            ob_col = order_by.get("column", "")
            if "." in ob_col:
                ob_table, ob_name = ob_col.split(".", 1)
                ob_table = ob_table.strip('"').strip("'")
                ob_name = ob_name.strip('"').strip("'")
                matched = self._fuzzy_match_column(ob_table, ob_name, kg)
                if matched:
                    order_nodes.append(QueryNode(table=matched[0], column=matched[1], role="order"))
                elif kg.graph and f"{ob_table}.{ob_name}" in kg.graph.columns:
                    order_nodes.append(QueryNode(table=ob_table, column=ob_name, role="order"))

        # --- Step 4: Graph Traversal (BFS through KG edges) ---
        path = build_query_path(output_nodes, filter_nodes, kg, order_nodes=order_nodes)
        if not path:
            self._log("kg_path", "No path found, falling back")
            return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        self._log("kg_path_edges", " → ".join(
            f"{e.src_table}.{e.src_column}={e.dst_table}.{e.dst_column}" for e in path.edges
        ) if path.edges else "(single table)")

        # --- Step 5: Format as grounding context for SQL LLM ---
        grounding = self._format_kg_plan_as_grounding(path, None, picked, "", kg=kg)
        self._log("kg_grounding", grounding)
        return grounding

    def _pick_graph_nodes(
        self,
        question: str,
        kg: KnowledgeGraph,
        anchor_text: str,
        user_intent: str,
    ) -> dict[str, Any] | None:
        """LLM picks columns/filters/joins from the property graph. One call."""
        g = kg.graph

        # Build FK target map: src_col_id → (dst_table, dst_col) for annotation
        fk_targets: dict[str, tuple[str, str]] = {}
        if g:
            for edge in g.fk_edges:
                dst_col = g.columns.get(edge.dst)
                if dst_col:
                    fk_targets[edge.src] = (dst_col.table_id, dst_col.name)

        # Build graph dump for LLM: tables with columns, FK edges, categorical values
        graph_parts: list[str] = []
        for table in kg.tables:
            cols_desc: list[str] = []
            for col in table.columns:
                parts = [f'"{col.name}" ({col.sql_type})']
                if col.description:
                    parts.append(f"-- {col.description}")
                col_id = f"{table.name}.{col.name}"
                # Annotate FK reference columns
                is_fk_col = col_id in fk_targets
                if is_fk_col:
                    ref_t, ref_c = fk_targets[col_id]
                    # Show display column if known (human-readable value to SELECT instead)
                    display_col = g.fk_display_map.get(col_id, "") if g else ""
                    if display_col:
                        parts.append(f'→ references "{ref_t}"."{ref_c}" (display: "{display_col}")')
                    else:
                        parts.append(f'→ references "{ref_t}"."{ref_c}"')
                # Show sample values (skip FK columns — their IDs are opaque)
                if not is_fk_col:
                    if g and col_id in g.contains_value:
                        val_nodes = g.get_column_values(col_id)
                        vals = [v.value for v in val_nodes[:8]]
                        if vals:
                            parts.append(f"values: {vals}")
                    elif col.name in table.sample_values:
                        samples = table.sample_values[col.name][:5]
                        if samples:
                            parts.append(f"e.g. {samples}")
                cols_desc.append("  " + " ".join(parts))
            graph_parts.append(f'TABLE "{table.name}" ({table.row_count} rows):\n' + "\n".join(cols_desc))

        # FK edges
        fk_lines: list[str] = []
        if g:
            for edge in g.fk_edges:
                src_col = g.columns.get(edge.src)
                dst_col = g.columns.get(edge.dst)
                if src_col and dst_col:
                    fk_lines.append(
                        f'  "{src_col.table_id}"."{src_col.name}" → "{dst_col.table_id}"."{dst_col.name}" (overlap: {edge.overlap_ratio:.0%})'
                    )
        if fk_lines:
            graph_parts.append("JOIN RELATIONSHIPS (foreign keys):\n" + "\n".join(fk_lines))

        # Semantic edges (for bridge table discovery)
        sem_lines: list[str] = []
        if g:
            for edge in g.semantic_edges:
                src_col = g.columns.get(edge.src)
                dst_col = g.columns.get(edge.dst)
                if src_col and dst_col:
                    sem_lines.append(
                        f'  "{src_col.table_id}"."{src_col.name}" ↔ "{dst_col.table_id}"."{dst_col.name}" (similarity: {edge.similarity_score:.0%})'
                    )
        if sem_lines:
            graph_parts.append("SEMANTIC LINKS (likely joinable by matching IDs):\n" + "\n".join(sem_lines))

        graph_dump = "\n\n".join(graph_parts)

        anchor_section = f"\nDOMAIN KNOWLEDGE:\n{anchor_text[:2000]}" if anchor_text else ""
        intent_section = f"\nUSER INTENT:\n{user_intent}" if user_intent else ""

        prompt = f"""QUESTION: {question}

PROPERTY GRAPH:
{graph_dump[:6000]}
{anchor_section}{intent_section}

HOW TO READ THE GRAPH:
- Each TABLE has columns with their SQL types. Some columns show known "values:" — these are categorical values in that column.
- Columns marked "→ references Table.Column" are FOREIGN KEYS that store IDs pointing to another table. To get human-readable values (names, labels), you must SELECT from the referenced table, not the FK column itself.
- JOIN RELATIONSHIPS show which columns link tables together. SEMANTIC LINKS show columns with matching ID patterns (useful for bridge/junction tables).
- When you need to connect tables that have no direct FK, look for a BRIDGE TABLE that has FKs to both.

YOUR TASK: Pick exact columns from this graph to answer the question.

PRIORITY (resolve conflicts in this order):
1. QUESTION — the user's exact words. If a word matches a column name, use that column.
2. DOMAIN KNOWLEDGE — defines what terms mean in this database. Overrides everyday English.
3. Your own inference — only if neither 1 nor 2 applies.

Return ONLY JSON:
{{
  "what_user_wants": "one sentence restating the expected output",
  "select_columns": ["Table.Column", ...],
  "filter_conditions": [
    {{"column": "Table.Column", "operator": "= | > | < | >= | <= | LIKE | !=", "value": "..."}}
  ],
  "order_by": {{"column": "Table.Column", "direction": "ASC | DESC"}} or null,
  "computation_type": "simple_lookup | count | sum | avg | min_max | ratio | percentage"
}}

CONSTRAINTS:
- You may ONLY use Table.Column names that appear in the PROPERTY GRAPH above.
- You may ONLY use values from "values:" lists, DOMAIN KNOWLEDGE, or the QUESTION itself.
- If a column is a FK reference (marked with →), do NOT select it directly — instead JOIN to the referenced table and select the human-readable column there (e.g. the name/label column, not the ID).
- For "best/lowest/highest/fastest" questions, use order_by instead of equality filters on position/rank columns.
- For filter values, use the EXACT format shown in "values:" or DOMAIN KNOWLEDGE. If unsure of the format, use LIKE with a partial match."""

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self._model_call_with_retry(messages)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("select_columns"):
                return parsed
        except Exception:
            pass
        return None

    def _validate_picked_nodes(
        self,
        picked: dict[str, Any],
        kg: KnowledgeGraph,
        db_path: Path | None,
    ) -> tuple[list[QueryNode], list[QueryNode], list[str]]:
        """Validate LLM's picks against the actual graph. Returns (output_nodes, filter_nodes, errors)."""
        errors: list[str] = []
        output_nodes: list[QueryNode] = []
        filter_nodes: list[QueryNode] = []

        g = kg.graph

        # Validate select_columns
        for col_ref in picked.get("select_columns", []):
            if "." not in col_ref:
                errors.append(f"Invalid column ref (no dot): {col_ref}")
                continue
            table, col = col_ref.split(".", 1)
            # Strip quotes
            table = table.strip('"').strip("'")
            col = col.strip('"').strip("'")
            col_id = f"{table}.{col}"
            if g and col_id in g.columns:
                output_nodes.append(QueryNode(table=table, column=col, role="output"))
            else:
                # Try case-insensitive match
                matched = self._fuzzy_match_column(table, col, kg)
                if matched:
                    output_nodes.append(QueryNode(table=matched[0], column=matched[1], role="output"))
                else:
                    errors.append(f"Column not in graph: {col_ref}")

        # Validate filter_conditions
        for cond in picked.get("filter_conditions", []):
            if not isinstance(cond, dict):
                continue
            col_ref = cond.get("column", "")
            operator = cond.get("operator", "=")
            value = cond.get("value", "")
            if "." not in col_ref:
                errors.append(f"Invalid filter column (no dot): {col_ref}")
                continue
            table, col = col_ref.split(".", 1)
            table = table.strip('"').strip("'")
            col = col.strip('"').strip("'")
            col_id = f"{table}.{col}"
            if g and col_id in g.columns:
                filter_nodes.append(QueryNode(
                    table=table, column=col, role="filter",
                    operator=operator, value=value,
                ))
            else:
                matched = self._fuzzy_match_column(table, col, kg)
                if matched:
                    filter_nodes.append(QueryNode(
                        table=matched[0], column=matched[1], role="filter",
                        operator=operator, value=value,
                    ))
                else:
                    errors.append(f"Filter column not in graph: {col_ref}")

        return output_nodes, filter_nodes, errors

    def _fuzzy_match_column(
        self, table: str, col: str, kg: KnowledgeGraph,
    ) -> tuple[str, str] | None:
        """Case-insensitive column lookup in KG."""
        for t in kg.tables:
            if t.name.lower() == table.lower():
                for c in t.columns:
                    if c.name.lower() == col.lower():
                        return (t.name, c.name)
        return None

    def _sanity_check_picks(
        self,
        question: str,
        picked: dict[str, Any],
        output_nodes: list[QueryNode],
        filter_nodes: list[QueryNode],
        user_intent: str,
        kg: KnowledgeGraph,
        anchor_text: str = "",
    ) -> str:
        """Check if LLM picks are consistent with the question and user intent."""
        issues: list[str] = []
        q_lower = question.lower()

        # Parse structured fields from user_intent
        intent_entity = ""
        intent_metric = ""
        intent_operation = ""
        intent_population = ""
        for line in user_intent.split("\n"):
            if "Entity of interest:" in line:
                intent_entity = line.split("Entity of interest:")[1].split("(")[0].strip().lower()
            elif "Metric (SELECT):" in line:
                intent_metric = line.split("Metric (SELECT):")[1].strip().lower()
            elif "Operation:" in line:
                intent_operation = line.split("Operation:")[1].strip().lower()
            elif "Population (WHERE):" in line:
                intent_population = line.split("Population (WHERE):")[1].strip().lower()

        # --- Check 1: Named entities in question should appear in filters ---
        # Only match deliberate quoting — skip apostrophes in contractions/possessives
        quoted = re.findall(r'"([^"]+)"', question)
        named_entities = list(quoted)
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b', question)
        named_entities.extend(proper_nouns)

        if named_entities:
            filter_values = [str(n.value).lower() for n in filter_nodes]
            filter_text = " ".join(filter_values)
            output_text = " ".join(f"{n.table}.{n.column}" for n in output_nodes).lower()
            for entity in named_entities:
                entity_lower = entity.lower()
                if entity_lower not in filter_text and entity_lower not in output_text:
                    issues.append(
                        f'The question mentions "{entity}" but it is not in any filter or output. '
                        f'Add a filter condition for it.'
                    )

        # --- Check 2: Entity of interest — output tables should match ---
        # First: deterministically extract entity from question pattern "which X" / "what X"
        question_entity = ""
        entity_match = re.search(
            r'\b(?:which|what|list\s+(?:all\s+)?(?:the\s+)?)\s*(\w+)',
            q_lower,
        )
        if entity_match:
            question_entity = entity_match.group(1).rstrip("s")  # "events" → "event"

        # Use question-derived entity if it matches a table, otherwise fall back to intent
        check_entity = ""
        if question_entity:
            matching_tables = [
                t.name for t in kg.tables
                if question_entity in t.name.lower() or t.name.lower() in question_entity
            ]
            if matching_tables:
                check_entity = question_entity
        if not check_entity:
            check_entity = intent_entity

        if check_entity and output_nodes:
            output_tables = {n.table.lower() for n in output_nodes}
            entity_in_output = any(
                check_entity in t or t in check_entity
                for t in output_tables
            )
            if not entity_in_output:
                entity_tables = [
                    t.name for t in kg.tables
                    if check_entity in t.name.lower() or t.name.lower() in check_entity
                ]
                if entity_tables:
                    issues.append(
                        f'The question asks about "{check_entity}" but output columns come from '
                        f'tables {list(output_tables)}. Select the name/label column from '
                        f'"{entity_tables[0]}" table instead.'
                    )

        # --- Check 3: Metric column should be in output ---
        # Skip for aggregate/superlative operations — metric is the criterion, not the answer
        comp_type = picked.get("computation_type", "")
        is_aggregate = comp_type in ("count", "sum", "avg", "min_max", "ratio", "percentage")
        is_superlative = intent_operation in ("min_max", "count", "sum", "avg", "ratio", "percentage")
        if intent_metric and output_nodes and not is_aggregate and not is_superlative:
            metric_words = set(intent_metric.replace(",", " ").split())
            output_cols = {n.column.lower() for n in output_nodes}
            metric_found = any(
                any(mw in col or col in mw for mw in metric_words)
                for col in output_cols
            )
            if not metric_found and intent_metric not in ("all", "all columns requested"):
                issues.append(
                    f'Intent says metric is "{intent_metric}" but output columns are '
                    f'{[n.column for n in output_nodes]}. Include the metric column.'
                )

        # --- Check 4: Population filter — intent says WHERE X but no matching filter ---
        if intent_population and intent_population != "all" and filter_nodes:
            pop_words = set(re.findall(r'[a-z]+', intent_population))
            filter_cols_and_vals = " ".join(
                f"{n.column.lower()} {str(n.value).lower()}" for n in filter_nodes
            )
            # Loose check: at least some population words should appear in filters
            overlap = sum(1 for w in pop_words if w in filter_cols_and_vals and len(w) > 3)
            if overlap == 0 and len(pop_words) > 0:
                issues.append(
                    f'Intent says population filter is "{intent_population}" '
                    f'but no matching filter condition was found.'
                )

        # --- Check 5: Output should have name/label, not ID/ref ---
        asks_for_name = any(w in q_lower for w in ["which", "what is the name", "what race", "what event"])
        if asks_for_name and output_nodes:
            has_name_col = any(
                any(x in n.column.lower() for x in ["name", "title", "label", "description"])
                for n in output_nodes
            )
            has_ref_col = any(
                n.column.lower().endswith("ref") or
                (n.column.lower().endswith("id") and n.column.lower() != "id")
                for n in output_nodes
            )
            if has_ref_col and not has_name_col:
                issues.append(
                    'The question asks for a name/label but output only has ID/ref columns. '
                    'Select the human-readable name column instead.'
                )

        # --- Check 6: Superlative should use order_by, not position/rank filter ---
        superlative_words = ["best", "lowest", "highest", "fastest", "slowest", "most", "least", "top", "worst"]
        has_superlative = any(w in q_lower for w in superlative_words)
        if has_superlative and not picked.get("order_by"):
            rank_filters = [n for n in filter_nodes if n.column.lower() in ("position", "rank", "positionorder")]
            if rank_filters:
                matched_word = next(w for w in superlative_words if w in q_lower)
                issues.append(
                    f'The question uses a superlative ("{matched_word}") '
                    f'but you filtered on {rank_filters[0].table}.{rank_filters[0].column} = {rank_filters[0].value}. '
                    f'Use order_by on the actual metric column instead.'
                )

        # --- Check 7: Operation mismatch (count/sum/avg but no aggregation in picks) ---
        if intent_operation in ("count", "sum", "avg", "count_distinct"):
            comp_type = picked.get("computation_type", "")
            if comp_type == "simple_lookup" and intent_operation != "lookup":
                issues.append(
                    f'Intent says operation is "{intent_operation}" but computation_type is "simple_lookup". '
                    f'Change computation_type to "{intent_operation}".'
                )

        # Check 8 moved to _apply_domain_column_fixes (deterministic, no re-pick needed)

        return "\n".join(issues)

    def _probe_filter_values(
        self, filter_nodes: list[QueryNode], db_path: Path | None,
    ) -> list[QueryNode]:
        """Probe actual DB values for TEXT filter columns to fix format mismatches."""
        if not db_path or not filter_nodes:
            return filter_nodes

        probed: list[QueryNode] = []
        try:
            conn = sqlite3.connect(str(db_path))
            for node in filter_nodes:
                if node.operator not in ("=", "LIKE"):
                    probed.append(node)
                    continue
                # Check if exact value exists
                try:
                    row = conn.execute(
                        f'SELECT 1 FROM "{node.table}" WHERE "{node.column}" = ? LIMIT 1',
                        (node.value,),
                    ).fetchone()
                    if row:
                        probed.append(node)
                        continue
                    # Value doesn't match exactly — sample actual values to find format
                    samples = conn.execute(
                        f'SELECT DISTINCT "{node.column}" FROM "{node.table}" '
                        f'WHERE "{node.column}" IS NOT NULL LIMIT 20'
                    ).fetchall()
                    sample_vals = [str(r[0]) for r in samples]
                    # Try case-insensitive match
                    val_lower = node.value.lower()
                    matched = next((v for v in sample_vals if v.lower() == val_lower), None)
                    if matched:
                        probed.append(QueryNode(
                            table=node.table, column=node.column, role=node.role,
                            operator=node.operator, value=matched,
                        ))
                        continue
                    # Try LIKE with key digits extracted from value
                    # e.g. "0:01:54" → try LIKE "%1:54%"
                    digits_pattern = re.sub(r'^[0:]+', '', node.value).rstrip("0").rstrip(".")
                    if digits_pattern and len(digits_pattern) >= 3:
                        like_row = conn.execute(
                            f'SELECT "{node.column}" FROM "{node.table}" WHERE "{node.column}" LIKE ? LIMIT 1',
                            (f"%{digits_pattern}%",),
                        ).fetchone()
                        if like_row:
                            probed.append(QueryNode(
                                table=node.table, column=node.column, role=node.role,
                                operator="LIKE", value=f"%{digits_pattern}%",
                            ))
                            self._log("value_probe", f"{node.table}.{node.column}: '{node.value}' → LIKE '%{digits_pattern}%'")
                            continue
                    # Keep original — let close-loop handle it
                    probed.append(node)
                    self._log("value_probe", f"{node.table}.{node.column}: '{node.value}' not found in DB. Samples: {sample_vals[:5]}")
                except Exception:
                    probed.append(node)
            conn.close()
        except Exception:
            return filter_nodes
        return probed

    def _apply_domain_column_fixes(
        self,
        question: str,
        filter_nodes: list[QueryNode],
        kg: KnowledgeGraph,
        anchor_text: str,
    ) -> list[QueryNode]:
        """Deterministically swap filter columns when domain knowledge defines a better match."""
        if not anchor_text or not filter_nodes:
            return filter_nodes

        q_lower = question.lower()
        used_cols = {n.column.lower() for n in filter_nodes}

        # Build map: domain-defined column name → (table, definition)
        domain_cols: dict[str, tuple[str, str]] = {}
        seen: set[str] = set()
        for m in re.finditer(r'^- (\w+):\s+(.+)', anchor_text, re.MULTILINE):
            defined_col = m.group(1).lower()
            if defined_col in seen:
                continue
            seen.add(defined_col)
            if defined_col not in q_lower or defined_col in used_cols:
                continue
            # Find which table has this column
            for t in kg.tables:
                for c in t.columns:
                    if c.name.lower() == defined_col:
                        domain_cols[defined_col] = (t.name, m.group(2))
                        break

        if not domain_cols:
            return filter_nodes

        # Swap: if a filter uses a synonym column (position↔rank, round↔number),
        # replace it with the domain-defined column
        synonym_groups = [
            {"position", "rank", "positionorder"},
            {"round", "number"},
        ]
        fixed = []
        for node in filter_nodes:
            swapped = False
            for defined_col, (col_table, definition) in domain_cols.items():
                # Check if this filter node is in the same synonym group as the domain column
                for group in synonym_groups:
                    if node.column.lower() in group and defined_col in group:
                        # Swap to the domain-defined column
                        fixed.append(QueryNode(
                            table=col_table, column=defined_col, role=node.role,
                            operator=node.operator, value=node.value,
                        ))
                        self._log("domain_fix",
                            f'Swapped {node.table}.{node.column} → {col_table}.{defined_col} '
                            f'(domain: "{definition[:60]}")')
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                fixed.append(node)
        return fixed

    def _check_filter_selectivity(
        self, filter_nodes: list[QueryNode], db_path: Path | None,
    ) -> str:
        """Check if filters are discriminating. A filter that keeps >90% of rows is likely wrong."""
        if not db_path or not filter_nodes:
            return ""
        issues: list[str] = []
        try:
            conn = sqlite3.connect(str(db_path))
            for node in filter_nodes:
                if node.operator in ("LIKE",):
                    continue
                # Skip null-guard filters — they protect ORDER BY, not meant to discriminate
                if node.operator in ("IS NOT", "IS NOT NULL") or (
                    node.operator == "!=" and node.value in ("", "NULL", None)
                ):
                    continue
                try:
                    total = conn.execute(
                        f'SELECT COUNT(*) FROM "{node.table}"'
                    ).fetchone()[0]
                    if total == 0:
                        continue
                    if node.operator == "=":
                        matching = conn.execute(
                            f'SELECT COUNT(*) FROM "{node.table}" WHERE "{node.column}" = ?',
                            (node.value,),
                        ).fetchone()[0]
                    else:
                        matching = conn.execute(
                            f'SELECT COUNT(*) FROM "{node.table}" WHERE "{node.column}" {node.operator} ?',
                            (node.value,),
                        ).fetchone()[0]
                    ratio = matching / total
                    if ratio > 0.9 and total > 5:
                        # Find alternative numeric columns across all tables with same operator
                        alternatives: list[str] = []
                        tables = [r[0] for r in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()]
                        for tbl in tables:
                            cols = conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()
                            for c in cols:
                                col_name = c[1]
                                col_type = c[2].upper()
                                if col_name == node.column:
                                    continue
                                # Only suggest columns with compatible type
                                if node.operator in ("<", ">", "<=", ">=") and col_type not in ("INTEGER", "REAL", "NUMERIC", "FLOAT"):
                                    continue
                                try:
                                    alt_match = conn.execute(
                                        f'SELECT COUNT(*) FROM "{tbl}" WHERE "{col_name}" {node.operator} ?',
                                        (node.value,),
                                    ).fetchone()[0]
                                    alt_total = conn.execute(
                                        f'SELECT COUNT(*) FROM "{tbl}"'
                                    ).fetchone()[0]
                                    if alt_total > 0:
                                        alt_ratio = alt_match / alt_total
                                        if alt_ratio < 0.8 and alt_ratio > 0.01:
                                            samples = conn.execute(
                                                f'SELECT DISTINCT "{col_name}" FROM "{tbl}" WHERE "{col_name}" IS NOT NULL LIMIT 5'
                                            ).fetchall()
                                            sample_str = [r[0] for r in samples]
                                            alternatives.append(
                                                f'{tbl}.{col_name} (keeps {alt_ratio:.0%}, e.g. {sample_str})'
                                            )
                                except Exception:
                                    pass
                        alt_text = ""
                        if alternatives:
                            alt_text = " Better candidates: " + "; ".join(alternatives[:3])
                        issues.append(
                            f'Filter "{node.table}"."{node.column}" {node.operator} {node.value} '
                            f'keeps {matching}/{total} rows ({ratio:.0%}) — too broad, probably wrong column.'
                            f'{alt_text}'
                        )
                except Exception:
                    pass
            conn.close()
        except Exception:
            pass
        return "\n".join(issues)

    def _decompose_goal(
        self,
        question: str,
        kg: KnowledgeGraph,
        anchor_text: str,
        user_intent: str,
    ) -> dict[str, Any] | None:
        """LLM decomposes question into output terms + filter terms. One LLM call."""
        # Build compact column list from KG
        col_list_parts: list[str] = []
        for table in kg.tables:
            cols_str = ", ".join(
                f"{c.name}" + (f" -- {c.description}" if c.description else "")
                for c in table.columns
            )
            col_list_parts.append(f"{table.name}: [{cols_str}]")
        col_list = "\n".join(col_list_parts)

        anchor_section = f"\nDOMAIN KNOWLEDGE:\n{anchor_text[:1500]}" if anchor_text else ""
        intent_section = f"\nUSER INTENT:\n{user_intent}" if user_intent else ""

        prompt = f"""QUESTION: {question}

DATABASE COLUMNS:
{col_list[:3000]}
{anchor_section}{intent_section}

Decompose this question into structured parts. Return ONLY JSON:
{{
  "what_user_wants": "restate what output the user expects",
  "output_terms": ["phrase for each SELECT column — use words from the question"],
  "filter_terms": [
    {{"phrase": "the entity/value being filtered", "operator": "= | > | < | >= | <= | LIKE", "value": "the filter value"}}
  ],
  "computation_type": "simple_lookup | count | sum | avg | min_max | ratio | percentage"
}}

RULES:
- output_terms: each phrase should map to ONE column. Use the question's exact words.
- filter_terms: extract ALL conditions. "X-related" or "from X" → operator "LIKE", value "%X%".
- If domain knowledge says "X = N", use that exact value.
- "where the average exceeds N" on a column named Avg* or average* → operator ">", value N (it's pre-aggregated, use WHERE not HAVING).
- Do NOT invent filters not in the question.
- computation_type: "simple_lookup" for listing/identifying, "count" for how many, etc."""

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self._model_call_with_retry(messages)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("output_terms"):
                return parsed
        except Exception:
            pass
        return None

    def _map_output_nodes(
        self,
        output_terms: list[str],
        kg: KnowledgeGraph,
        tables: list[str],
        db_path: Path | None,
        preferred_table: str = "",
    ) -> list[QueryNode]:
        """Map output term phrases to KG columns deterministically.

        When scores tie, prefer columns from preferred_table (entity of interest).
        """
        nodes: list[QueryNode] = []
        used_columns: set[tuple[str, str]] = set()

        for term in output_terms:
            candidates = map_phrase_to_columns(term, kg, tables, role="output")
            if not candidates:
                continue
            # When multiple columns share the same name, prefer the entity of interest table
            if preferred_table and len(candidates) >= 2:
                # Find candidates with the same column name as the top match
                top_col_name = candidates[0][1].lower()
                same_name = [(t, c, s) for t, c, s in candidates if c.lower() == top_col_name]
                preferred_match = next(
                    ((t, c, s) for t, c, s in same_name if t.lower() == preferred_table.lower()),
                    None,
                )
                if preferred_match:
                    t, c, s = preferred_match
                    if (t, c) not in used_columns:
                        nodes.append(QueryNode(table=t, column=c, role="output"))
                        used_columns.add((t, c))
                        continue

            # Pick best candidate not already used
            for table, col, score in candidates:
                if (table, col) not in used_columns and score >= 3.0:
                    nodes.append(QueryNode(table=table, column=col, role="output"))
                    used_columns.add((table, col))
                    break

        return nodes

    def _map_filter_nodes(
        self,
        filter_terms: list[dict],
        kg: KnowledgeGraph,
        tables: list[str],
        db_path: Path | None,
        anchor_text: str,
    ) -> list[QueryNode]:
        """Map filter terms to KG columns, verifying values against DB."""
        nodes: list[QueryNode] = []

        for fterm in filter_terms:
            if not isinstance(fterm, dict):
                continue
            phrase = fterm.get("phrase", "")
            operator = fterm.get("operator", "=")
            value = fterm.get("value", "")

            if not phrase:
                continue

            # First try: find column by phrase
            candidates = map_phrase_to_columns(phrase, kg, tables, role="filter")

            # Second try: if value is a string, find which column actually contains it
            value_matches: list[tuple[str, str, str, int]] = []
            if value and isinstance(value, str) and not re.match(r'^[<>!=]*\s*\d+\.?\d*$', str(value).strip('%')):
                value_matches = map_value_to_column(str(value).strip('%'), kg, tables, db_path)

            # Merge: value match takes priority (ground truth from DB)
            if value_matches:
                best_table, best_col = value_matches[0][0], value_matches[0][1]
                method = value_matches[0][2]
                if method == "like" and operator == "=":
                    operator = "LIKE"
                    if not str(value).startswith('%'):
                        value = f"%{str(value).strip('%')}%"
                nodes.append(QueryNode(
                    table=best_table, column=best_col, role="filter",
                    operator=operator, value=value,
                ))
            elif candidates:
                best_table, best_col, score = candidates[0]
                if score >= 3.0:
                    nodes.append(QueryNode(
                        table=best_table, column=best_col, role="filter",
                        operator=operator, value=value,
                    ))

        return nodes

    def _format_kg_plan_as_grounding(
        self,
        path: QueryPath,
        plan: QueryPlan | None,
        goal: dict[str, Any],
        sql: str,
        kg: KnowledgeGraph | None = None,
    ) -> str:
        """Format the KG path plan as grounding context for the SQL LLM."""
        parts: list[str] = []

        parts.append(f"USER WANTS: {goal.get('what_user_wants', '')}")
        comp_type = plan.computation_type if plan else goal.get("computation_type", "simple_lookup")
        parts.append(f"COMPUTATION TYPE: {comp_type} (your SQL MUST produce this type of result)")

        # Schema for tables in path (so LLM knows exact column names)
        if kg and path.tables_in_path:
            schema_lines: list[str] = []
            for tname in path.tables_in_path:
                table_schema = kg.get_table(tname)
                if table_schema:
                    cols = ", ".join(
                        f'"{c.name}" ({c.sql_type})'
                        for c in table_schema.columns
                    )
                    schema_lines.append(f'  "{tname}": [{cols}]')
            if schema_lines:
                parts.append("SCHEMA (exact column names — use quoted identifiers):\n" + "\n".join(schema_lines))

        # Output columns (what to SELECT)
        if path.output_nodes:
            out_lines = [f'  "{n.table}"."{n.column}"' + (f" ({n.agg_func})" if n.agg_func else "")
                         for n in path.output_nodes]
            parts.append("SELECT COLUMNS (use these exact table.column references):\n" + "\n".join(out_lines))

        # Join paths
        if path.edges:
            jp_lines = [f'  "{e.src_table}"."{e.src_column}" = "{e.dst_table}"."{e.dst_column}"' for e in path.edges]
            parts.append("JOIN PATHS (use these exact join conditions):\n" + "\n".join(jp_lines))

        # Filter values
        if path.filter_nodes:
            fv_lines: list[str] = []
            for node in path.filter_nodes:
                if node.operator.upper() == "LIKE":
                    fv_lines.append(f'  "{node.table}"."{node.column}": USE → WHERE "{node.column}" LIKE \'{node.value}\' COLLATE NOCASE')
                else:
                    fv_lines.append(f'  "{node.table}"."{node.column}": {node.operator} {node.value}')
            parts.append("FILTER VALUES (use these exact conditions in WHERE):\n" + "\n".join(fv_lines))

        # ORDER BY (from picked dict) — used for superlatives (lowest/highest/best)
        order_by = goal.get("order_by")
        if order_by and isinstance(order_by, dict):
            col_ref = order_by.get("column", "")
            direction = order_by.get("direction", "ASC")
            parts.append(f'ORDER BY: "{col_ref}" {direction} (use WHERE = (SELECT MIN/MAX...) to include ties)')

        # Reference SQL if provided (from retry)
        if sql:
            parts.append(f"FAILED SQL (do NOT repeat — fix based on error):\n  {sql}")

        return "GROUNDING CONTEXT:\n" + "\n".join(parts)

    def _call_semantic_grounding(
        self,
        question: str,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
        db_path: Path | None = None,
        kg: KnowledgeGraph | None = None,
    ) -> str:
        """Validate-first-then-ground: deterministic pre-validation → one-shot constrained grounding."""

        # Extract domain anchors
        anchor_text = self._extract_domain_anchors(question, knowledge_text, db_path=db_path)

        # --- Step 1: Intent Detection (stable — never redone) ---
        user_intent = self._detect_user_intent_only(question, kg_context=kg_context, anchor_text=anchor_text)
        if user_intent:
            self._log("user_intent", user_intent)

        # Pre-grounding: extract formula + column mappings from knowledge, scoped to user intent
        knowledge_guidance = ""
        if knowledge_text:
            knowledge_guidance = self._extract_knowledge_guidance(question, knowledge_text, user_intent, kg_context=kg_context)
            if knowledge_guidance:
                self._log("knowledge_guidance", knowledge_guidance)

        self._log("grounding_iter", "--- Grounding ---")

        effective_anchor = anchor_text

        # --- Step 2: Table Selection ---
        selected_tables = self._grounding_select_tables(
            question, kg_context, effective_anchor, db_path, kg=kg, user_intent=user_intent
        )

        # Deterministic: if question words derive from column names, surface the column definition
        if knowledge_guidance and db_path and selected_tables:
            col_notes = self._match_question_words_to_columns(question, db_path, selected_tables, anchor_text)
            if col_notes:
                knowledge_guidance = f"{col_notes}\n\n{knowledge_guidance}"

        # --- Step 3: Build focused schema ---
        focused_schema = ""
        if selected_tables and db_path:
            focused_schema = self._build_focused_schema_for_grounding(db_path, selected_tables, question, kg=kg)
        grounding_schema = focused_schema if focused_schema else kg_context

        # Extract matching SQL from domain knowledge
        domain_sql_ref = self._extract_matching_domain_sql(question, anchor_text)
        if domain_sql_ref:
            effective_anchor = (
                f"⚠️ REFERENCE SQL FROM DOMAIN KNOWLEDGE (use this exact pattern — do NOT add extra conversion or transformation):\n"
                f"  {domain_sql_ref}\n\n{effective_anchor}"
            )
            self._log("domain_sql_match", domain_sql_ref)

        # --- Step 4: DETERMINISTIC PRE-VALIDATION ---
        # Build validated context BEFORE the grounding LLM call
        validated = self._build_validated_context(
            question, db_path, selected_tables, kg, user_intent, effective_anchor
        )

        # --- PRE-GROUNDING EVIDENCE ---
        ambiguous_evidence = ""
        if db_path and selected_tables and len(selected_tables) > 1:
            ambiguous_evidence = self._scan_ambiguous_columns(db_path, selected_tables, question)
            if ambiguous_evidence:
                self._log("ambiguous_cols_evidence", ambiguous_evidence)

        literal_matches = self._scan_literal_column_matches(db_path, selected_tables, question)
        if literal_matches:
            ambiguous_evidence = f"{ambiguous_evidence}\n\n{literal_matches}" if ambiguous_evidence else literal_matches

        # --- Step 5: ONE constrained grounding LLM call ---
        # Inject validated facts as hard constraints so the LLM doesn't need to guess
        validated_section = self._format_validated_context(validated)

        prompt = _build_semantic_prompt(
            question=question,
            kg_context=grounding_schema,
            sample_data=sample_data if not focused_schema else "",
            anchor_text=effective_anchor,
            previous_attempt="",
            ambiguous_columns=ambiguous_evidence,
            user_intent=user_intent,
            knowledge_guidance=knowledge_guidance,
            feedback=validated_section if validated_section else "",
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
            return ""

        self._log("grounding_v1", json.dumps(grounding, default=str))

        # --- Step 6: Merge validated facts into grounding (deterministic, no LLM) ---
        grounding = self._merge_validated_into_grounding(grounding, validated, db_path, selected_tables, kg)

        # Verify all question entities appear in formula/known_values
        if grounding:
            grounding = self._verify_filter_completeness(question, grounding)

        # Deterministic enrichment: add any schema columns whose name matches a question word
        if db_path:
            grounding = self._enrich_data_requirements(db_path, question, grounding)

        formatted = _format_grounding_for_sql(grounding)

        self._log("semantic_grounding_final", formatted if formatted else "(empty)")

        return formatted

    def _build_validated_context(
        self,
        question: str,
        db_path: Path | None,
        selected_tables: list[str],
        kg: KnowledgeGraph | None,
        user_intent: str,
        anchor_text: str,
    ) -> dict[str, Any]:
        """Deterministic pre-validation using KG as ground truth for schema facts,
        DB probes only for filter value existence. Returns validated facts."""
        validated: dict[str, Any] = {
            "join_paths": [],
            "filter_probes": {},
            "filter_overrides": {},
            "preagg_columns": [],
            "format_notes": [],
        }
        if not db_path or not db_path.exists() or not selected_tables:
            return validated

        if kg is None:
            kg = build_kg_from_sqlite(db_path)

        # --- 1. Join paths from KG (already validated by value overlap during KG build) ---
        tables_lower = {t.lower() for t in selected_tables}
        for src_table, fk in kg.all_foreign_keys():
            if src_table.lower() in tables_lower and fk.ref_table.lower() in tables_lower:
                clause = f"{src_table}.{fk.column} = {fk.ref_table}.{fk.ref_column}"
                if clause not in validated["join_paths"]:
                    validated["join_paths"].append(clause)

        if validated["join_paths"]:
            self._log("grounding_join_paths", str(validated["join_paths"]))

        # --- 2. Detect pre-aggregated columns from KG column names + stats ---
        for tname in selected_tables:
            table_schema = kg.get_table(tname)
            if not table_schema:
                continue
            for col in table_schema.columns:
                col_lower = col.name.lower()
                if (col_lower.startswith("avg") or col_lower.startswith("total")
                        or col_lower.startswith("sum") or col_lower.startswith("num")
                        or col_lower.startswith("count")):
                    stats = table_schema.col_stats.get(col.name, {})
                    distinct = stats.get("distinct", 0)
                    if table_schema.row_count > 0 and distinct > 1:
                        validated["preagg_columns"].append(f"{tname}.{col.name}")

        # --- 3. Probe filter values against DB ---
        # Extract candidates from question: quoted strings + proper nouns
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
        proper_nouns = re.findall(r'(?:the |in |for |of |from )([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)', question)
        filter_candidates = quoted + proper_nouns

        # Extract domain-anchor filter conditions (e.g., "Filter condition: Thrombosis = 2")
        anchor_filters: list[tuple[str, str]] = []
        if anchor_text:
            for m in re.finditer(r'Filter condition:\s*(\w+)\s*=\s*(\S+)', anchor_text):
                anchor_filters.append((m.group(1), m.group(2)))

        conn = sqlite3.connect(str(db_path))
        try:
            # 3a. Verify domain-anchor filters using KG to find the right table
            for col_name, val in anchor_filters:
                for tname in selected_tables:
                    table_schema = kg.get_table(tname)
                    if not table_schema:
                        continue
                    real_col = next((c.name for c in table_schema.columns if c.name.lower() == col_name.lower()), None)
                    if not real_col:
                        continue
                    try:
                        cnt = conn.execute(
                            f'SELECT COUNT(*) FROM "{tname}" WHERE "{real_col}" = ?', (val,)
                        ).fetchone()[0]
                        if cnt > 0:
                            key = f"{tname}.{real_col}"
                            validated["filter_probes"][key] = {
                                "value": val, "count": cnt, "method": "exact"
                            }
                            self._log("filter_verified", f"{key}='{val}' exists ({cnt} rows)")
                    except Exception:
                        pass
                    break

            # 3b. Probe question filter candidates against text columns (from KG)
            for candidate in filter_candidates:
                if len(candidate) < 2:
                    continue
                found_exact = False
                for tname in selected_tables:
                    if found_exact:
                        break
                    table_schema = kg.get_table(tname)
                    if not table_schema:
                        continue
                    text_cols = [c.name for c in table_schema.columns
                                 if c.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", "")]
                    for col in text_cols:
                        try:
                            cnt = conn.execute(
                                f'SELECT COUNT(*) FROM "{tname}" WHERE "{col}" = ? COLLATE NOCASE',
                                (candidate,)
                            ).fetchone()[0]
                            if cnt > 0:
                                key = f"{tname}.{col}"
                                if key not in validated["filter_probes"]:
                                    validated["filter_probes"][key] = {
                                        "value": candidate, "count": cnt, "method": "exact"
                                    }
                                found_exact = True
                                break
                        except Exception:
                            continue
                # LIKE fallback
                if not found_exact:
                    for tname in selected_tables:
                        table_schema = kg.get_table(tname)
                        if not table_schema:
                            continue
                        text_cols = [c.name for c in table_schema.columns
                                     if c.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", "")]
                        for col in text_cols:
                            try:
                                like_cnt = conn.execute(
                                    f'SELECT COUNT(*) FROM "{tname}" WHERE "{col}" LIKE ? COLLATE NOCASE',
                                    (f'%{candidate}%',)
                                ).fetchone()[0]
                                if like_cnt > 0:
                                    key = f"{tname}.{col}"
                                    if key not in validated["filter_probes"]:
                                        validated["filter_probes"][key] = {
                                            "value": candidate, "count": like_cnt, "method": "like",
                                            "override": f'"{col}" LIKE \'%{candidate}%\' COLLATE NOCASE',
                                        }
                                        validated["filter_overrides"][key] = (
                                            f'"{col}" LIKE \'%{candidate}%\' COLLATE NOCASE'
                                        )
                                        validated["format_notes"].append(
                                            f"For {tname}.{col}: use WHERE \"{col}\" LIKE '%{candidate}%' COLLATE NOCASE"
                                        )
                                    break
                            except Exception:
                                continue
        finally:
            conn.close()

        return validated

    def _format_validated_context(self, validated: dict[str, Any]) -> str:
        """Format validated facts as constraints for the grounding LLM."""
        parts: list[str] = []

        if validated["join_paths"]:
            parts.append(
                "VALIDATED JOIN PATHS (verified to produce rows — use these exactly):\n"
                + "\n".join(f"  {jp}" for jp in validated["join_paths"])
            )

        if validated["filter_probes"]:
            lines: list[str] = []
            for key, info in validated["filter_probes"].items():
                if info["method"] == "exact":
                    lines.append(f"  {key} = '{info['value']}' ({info['count']} rows)")
                else:
                    lines.append(f"  {key} LIKE '%{info['value']}%' ({info['count']} rows) — use LIKE, not =")
            if lines:
                parts.append(
                    "VALIDATED FILTER VALUES (verified against DB — use these in known_values):\n"
                    + "\n".join(lines)
                )

        if validated["preagg_columns"]:
            parts.append(
                "PRE-AGGREGATED COLUMNS (already store per-row aggregates — use WHERE, not AVG()):\n"
                + "\n".join(f"  {col}" for col in validated["preagg_columns"])
            )

        if not parts:
            return ""
        return "⚠️ VALIDATED FACTS (verified against actual database — these override assumptions):\n" + "\n".join(parts)

    def _merge_validated_into_grounding(
        self,
        grounding: dict[str, Any],
        validated: dict[str, Any],
        db_path: Path | None,
        selected_tables: list[str],
        kg: KnowledgeGraph | None,
    ) -> dict[str, Any]:
        """Deterministically merge validated DB facts into grounding output."""
        # Merge join paths (validated ones take priority)
        if validated["join_paths"]:
            grounding["join_paths"] = validated["join_paths"]

        # Merge filter overrides for LIKE patterns
        if validated["filter_overrides"]:
            existing_overrides = grounding.get("filter_overrides", {})
            known_values = grounding.get("known_values", {})
            for key, override in validated["filter_overrides"].items():
                if key in known_values and key not in existing_overrides:
                    existing_overrides[key] = override
                    self._log("filter_override_applied", f"{key}: {override}")
            grounding["filter_overrides"] = existing_overrides

        # Merge format notes
        if validated["format_notes"]:
            existing_notes = grounding.get("data_format_notes", [])
            for note in validated["format_notes"]:
                if note not in existing_notes:
                    existing_notes.append(note)
            grounding["data_format_notes"] = existing_notes

        # Validate known_values against DB — fix format mismatches
        if db_path and grounding.get("known_values"):
            grounding = self._validate_known_values_against_db(db_path, grounding, validated)

        return grounding

    def _validate_known_values_against_db(
        self,
        db_path: Path,
        grounding: dict[str, Any],
        validated: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify that grounding's known_values actually exist in DB. Fix format mismatches."""
        known_values = grounding.get("known_values", {})
        filter_overrides = grounding.get("filter_overrides", {})
        format_notes = grounding.get("data_format_notes", [])

        try:
            conn = sqlite3.connect(str(db_path))
        except Exception:
            return grounding

        try:
            for col_key, values in list(known_values.items()):
                if "." not in col_key or not values:
                    continue
                table_name, col_name = col_key.split(".", 1)
                # Skip comparison operators, SQL expressions, numerics
                if all(re.match(r'^[<>!=]', str(v).strip()) for v in values if v):
                    continue
                if any(any(kw in str(v).upper() for kw in ('COUNT', 'SUM', 'AVG', 'MIN', 'MAX')) for v in values if v):
                    continue
                if all(re.match(r'^-?\d+\.?\d*$', str(v).strip()) for v in values if v):
                    continue
                if len(values) > 5:
                    continue

                # Already validated via pre-validation
                if col_key in validated["filter_overrides"]:
                    if col_key not in filter_overrides:
                        filter_overrides[col_key] = validated["filter_overrides"][col_key]
                    continue

                corrected_values = []
                for val in values:
                    val_str = str(val)
                    try:
                        # Check LIKE wildcards first
                        if '%' in val_str:
                            like_cnt = conn.execute(
                                f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" LIKE ? COLLATE NOCASE',
                                (val_str,)
                            ).fetchone()[0]
                            if like_cnt > 0:
                                corrected_values.append(val)
                                filter_overrides[col_key] = f'"{col_name}" LIKE \'{val_str}\' COLLATE NOCASE'
                                format_notes.append(
                                    f"For {table_name}.{col_name}: use WHERE \"{col_name}\" LIKE '{val_str}' COLLATE NOCASE"
                                )
                                self._log("filter_verified", f"{col_key}='{val}' matches {like_cnt} rows via LIKE")
                                continue

                        # Exact match
                        cnt = conn.execute(
                            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" = ?',
                            (val_str,)
                        ).fetchone()[0]
                        if cnt > 0:
                            corrected_values.append(val)
                            self._log("filter_verified", f"{col_key}='{val}' exists ({cnt} rows)")
                            continue

                        # Case-insensitive check
                        ci_row = conn.execute(
                            f'SELECT "{col_name}" FROM "{table_name}" WHERE "{col_name}" = ? COLLATE NOCASE LIMIT 1',
                            (val_str,)
                        ).fetchone()
                        if ci_row:
                            corrected_values.append(str(ci_row[0]))
                            self._log("filter_verified", f"{col_key}='{val}' → case-corrected to '{ci_row[0]}'")
                            continue

                        # LIKE fallback — try prefix match
                        like_cnt = conn.execute(
                            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" LIKE ? COLLATE NOCASE',
                            (f'%{val_str}%',)
                        ).fetchone()[0]
                        if like_cnt > 0:
                            corrected_values.append(val)
                            filter_overrides[col_key] = f'"{col_name}" LIKE \'%{val_str}%\' COLLATE NOCASE'
                            format_notes.append(
                                f"For {table_name}.{col_name}: use WHERE \"{col_name}\" LIKE '%{val_str}%' COLLATE NOCASE"
                            )
                            self._log("filter_corrected", f"{col_key}: '{val}' → LIKE '%{val_str}%'")
                            continue

                        # Value not found at all — keep it but note
                        corrected_values.append(val)
                        self._log("filter_verified", f"{col_key}='{val}' NOT found in DB")
                    except Exception:
                        corrected_values.append(val)

                known_values[col_key] = corrected_values

            grounding["known_values"] = known_values
            grounding["filter_overrides"] = filter_overrides
            grounding["data_format_notes"] = format_notes
        finally:
            conn.close()

        return grounding

    def _synthesize_grounding(self, question: str, raw_grounding: str, schema: str) -> str:
        """LLM resolves contradictions in grounding context against actual schema."""
        prompt = (
            f"QUESTION: {question}\n\n"
            f"ACTUAL DATABASE SCHEMA:\n{schema[:3000]}\n\n"
            f"DRAFT GROUNDING CONTEXT (may contain contradictions):\n{raw_grounding}\n\n"
            "Your job: produce a CLEAN grounding context for a SQL writer.\n"
            "1. Remove any statements that contradict the ACTUAL SCHEMA (wrong column names, wrong join conditions).\n"
            "2. Remove contradictions between MANDATORY CORRECTIONS and CONSTRAINTS — MANDATORY CORRECTIONS always win.\n"
            "3. Keep JOIN PATHS, FILTER VALUES, and DATA FORMAT WARNINGS only if they reference real columns from the schema.\n"
            "4. Keep the same format (section headers, bullet style).\n"
            "5. Do NOT add new information. Only remove or fix contradictions.\n\n"
            "Return the cleaned grounding context as plain text (no JSON, no code fences)."
        )
        try:
            raw = self._model_call_with_retry([ModelMessage(role="user", content=prompt)])
            cleaned = raw.strip()
            if cleaned and len(cleaned) > 50:
                return cleaned
        except Exception:
            pass
        return raw_grounding

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

                # Inject multi_col when question asks for multiple attributes ("X and Y")
                # but not when "and" is used in filters ("in US and Canada")
                if "multi_col" not in selected_labels and re.search(
                    r'\b(list|what|give|show|find)\b.+ and ', q_lower
                ):
                    mc_rule = next((r for l, r in SQL_RULES_LABELED if l == "multi_col"), None)
                    if mc_rule:
                        selected.append(mc_rule)
                        selected_labels.append("multi_col")

                if "singular_plural" not in selected_labels:
                    sp_rule = next((r for l, r in SQL_RULES_LABELED if l == "singular_plural"), None)
                    if sp_rule:
                        selected.append(sp_rule)
                        selected_labels.append("singular_plural")

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
        # Extract proper nouns (multi-word capitalized sequences not at start of sentence)
        proper_nouns = re.findall(r'(?:the |in |for |of |from )([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)+)', question)

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

    def _extract_knowledge_guidance(self, question: str, knowledge_text: str, user_intent: str = "", kg_context: str = "") -> str:
        """Use LLM to extract relevant knowledge for this question."""
        intent_section = f"\nUSER INTENT:\n{user_intent}" if user_intent else ""
        # Build compact schema reference: table.column names only
        schema_section = ""
        if kg_context:
            schema_lines: list[str] = []
            current_table = ""
            for line in kg_context.split("\n"):
                if line.startswith("TABLE: "):
                    current_table = line.split("TABLE: ")[1].split(" (")[0].strip()
                    schema_lines.append(f"\n{current_table}:")
                elif line.strip().startswith("- ") and current_table:
                    # Column name is everything between "- " and " (" (type annotation)
                    col_part = line.strip()[2:]
                    paren_idx = col_part.find(" (")
                    col_name = col_part[:paren_idx].strip() if paren_idx > 0 else col_part.split("  ")[0].strip()
                    if col_name:
                        schema_lines.append(f"  {col_name}")
            if schema_lines:
                schema_section = f"\nACTUAL DATABASE COLUMNS (use ONLY these names):\n{''.join(ln + chr(10) for ln in schema_lines)}"
        prompt = f"""QUESTION: {question}
{intent_section}{schema_section}
DOMAIN KNOWLEDGE:
{knowledge_text}

Extract ONLY column definitions and disambiguation rules from DOMAIN KNOWLEDGE that are relevant to the USER INTENT.

RULES:
- Copy definitions VERBATIM from the source — do NOT rephrase or interpret them.
- Do NOT add computation logic, ranking logic, or "how to answer" conclusions. Only definitions.
- Do NOT decide which column applies to the question — just present what each column means.
- Always qualify column names with their table name (table.column format).
- Use ONLY the exact column names from ACTUAL DATABASE COLUMNS above — NEVER use human-readable descriptions as column names.
- If nothing is relevant, return: NONE"""

        # Build lookup of real column names per table for post-validation
        real_columns: dict[str, list[str]] = {}
        current_t = ""
        for line in kg_context.split("\n"):
            if line.startswith("TABLE: "):
                current_t = line.split("TABLE: ")[1].split(" (")[0].strip()
            elif line.strip().startswith("- ") and current_t:
                col_part = line.strip()[2:]
                paren_idx = col_part.find(" (")
                cname = col_part[:paren_idx].strip() if paren_idx > 0 else col_part.split("  ")[0].strip()
                if cname:
                    real_columns.setdefault(current_t, []).append(cname)

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self._model_call_with_retry(messages)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if raw.upper().startswith("NONE") or len(raw) < 10:
                return ""
            # Post-validate: snap any table.column references to real column names
            if real_columns:
                raw = self._snap_columns_to_schema(raw, real_columns)
            return raw
        except Exception:
            return ""

    def _snap_columns_to_schema(self, text: str, real_columns: dict[str, list[str]]) -> str:
        """Replace table.column references with the closest matching real column name."""
        # Find all table.column patterns
        def _best_match(table: str, col: str) -> str:
            cols = real_columns.get(table, [])
            if not cols:
                return col
            # Exact match
            for c in cols:
                if c.lower() == col.lower():
                    return c
            # Substring containment: real column contains the LLM output
            for c in cols:
                if col.lower() in c.lower():
                    return c
            # Reverse: LLM output contains a real column name
            for c in cols:
                if c.lower() in col.lower():
                    return c
            return col

        def _replace(m: re.Match) -> str:
            table, col = m.group(1), m.group(2)
            real = _best_match(table, col)
            return f"{table}.{real}"

        all_tables = "|".join(re.escape(t) for t in real_columns)
        if all_tables:
            text = re.sub(
                rf'\b({all_tables})\.([A-Za-z][A-Za-z0-9_ ]*(?:\([^)]*\))?)',
                _replace, text
            )
        return text

    def _detect_user_intent_only(self, question: str, kg_context: str = "", anchor_text: str = "") -> str:
        """Detect user intent: what to return, how to filter, what operation."""
        schema_section = f"\nDATABASE SCHEMA:\n{kg_context[:2000]}\n" if kg_context else ""
        domain_section = f"\nDOMAIN KNOWLEDGE:\n{anchor_text[:1000]}\n" if anchor_text else ""
        prompt = f"""QUESTION: {question}
{schema_section}{domain_section}
Decompose this question into its analytical intent. Return ONLY JSON:
{{
  "answer_shape": "single_value | list | grouped_table",
  "operation": "lookup | count | count_distinct | sum | avg | ratio | percentage | min_max | rank",
  "grain": "one answer for entire dataset | one row per entity | one row per group",
  "what_to_return": "the THING the user wants to see in the answer (e.g. 'race names', 'eye colour', 'total cost')",
  "who_to_filter_on": "the NAMED ENTITY or CONDITION that selects the subset (e.g. 'Alex Yoong', 'Women\\'s Soccer', 'race 19')",
  "entity_of_interest": "the database entity whose attributes the user wants RETURNED (not the filter entity)"
}}

HOW TO DECIDE:
1. Read the question and identify: WHAT does the user want to SEE in the result?
   - "Which race was X in" → user wants to see RACE NAMES
   - "What is the eye colour" → user wants to see COLOUR VALUES
   - "How many members attended" → user wants to see a COUNT
   - "List their ID, sex and disease" → user wants to see ID, SEX, DISEASE columns

2. Identify: WHO/WHAT constrains the data?
   - "Which race was ALEX YOONG in" → Alex Yoong is the filter (find rows about this person)
   - "the driver with the BEST lap time" → best is a superlative (ORDER BY, not a filter value)
   - "in RACE NUMBER 19" → race 19 is a filter

3. entity_of_interest = the entity whose ATTRIBUTES appear in the output.
   - "Which RACE was Alex Yoong in" → entity_of_interest = race (we return race attributes)
   - "What is the SURNAME of the driver" → entity_of_interest = driver (we return driver attributes)
   - "eye COLOUR of the superhero" → entity_of_interest = colour (we return from colour table)

COMMON MISTAKES TO AVOID:
- "Which race was X in" → entity is RACE not driver. The driver is the filter, not the output.
- "What is the colour of X" → if colour is stored in a lookup table, entity is the LOOKUP table.
- Do NOT confuse the filter entity with the output entity."""

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self._model_call_with_retry(messages)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict):
                lines = []
                if parsed.get("answer_shape"):
                    lines.append(f"Answer shape: {parsed['answer_shape']}")
                if parsed.get("operation"):
                    lines.append(f"Operation: {parsed['operation']}")
                if parsed.get("grain"):
                    lines.append(f"Grain: {parsed['grain']}")
                if parsed.get("who_to_filter_on") and parsed["who_to_filter_on"] not in ("all", "none", ""):
                    lines.append(f"Population (WHERE): {parsed['who_to_filter_on']}")
                if parsed.get("what_to_return"):
                    lines.append(f"Metric (SELECT): {parsed['what_to_return']}")
                if parsed.get("entity_of_interest"):
                    lines.append(f"Entity of interest: {parsed['entity_of_interest']} (prefer this table's columns for output)")
                return "\n".join(lines)
        except Exception:
            pass
        return ""

    def _scan_ambiguous_columns(
        self, db_path: Path, selected_tables: list[str], question: str,
    ) -> str:
        """Deterministic DB scan: find columns with the same name across multiple tables.
        For each, sample values from each table so the grounding LLM can make an informed choice."""
        try:
            conn = sqlite3.connect(str(db_path))
            # Collect columns per table (only selected tables)
            table_cols: dict[str, list[str]] = {}
            for tname in selected_tables:
                try:
                    cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                    table_cols[tname] = cols
                except Exception:
                    continue

            # Find columns that exist in 2+ tables
            col_to_tables: dict[str, list[str]] = {}
            for tname, cols in table_cols.items():
                for col in cols:
                    col_to_tables.setdefault(col.lower(), []).append((tname, col))

            ambiguous = {k: v for k, v in col_to_tables.items() if len(v) > 1}
            if not ambiguous:
                conn.close()
                return ""

            # For each ambiguous column, get sample values from each table
            lines: list[str] = []
            lines.append("⚠️ AMBIGUOUS COLUMNS (same name exists in multiple tables — use the question's entity ownership to decide):")
            for col_lower, table_col_pairs in ambiguous.items():
                # Skip obvious PK/FK join keys (usually ID columns used for joins)
                # Still show them but mark as join keys
                samples: list[str] = []
                for tname, col_name in table_col_pairs:
                    try:
                        rows = conn.execute(
                            f'SELECT DISTINCT "{col_name}" FROM "{tname}" '
                            f'WHERE "{col_name}" IS NOT NULL AND "{col_name}" != \'\' '
                            f'LIMIT 5'
                        ).fetchall()
                        vals = [str(r[0]) for r in rows]
                        samples.append(f"  {tname}.{col_name}: {vals}")
                    except Exception:
                        samples.append(f"  {tname}.{col_name}: (error reading)")
                if samples:
                    lines.append(f"  Column '{col_lower}':")
                    lines.extend(samples)

            conn.close()
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception:
            return ""

    def _match_question_words_to_columns(
        self, question: str, db_path: Path, selected_tables: list[str], anchor_text: str,
    ) -> str:
        """Deterministic: find question words that derive from column names and surface their domain definition."""
        try:
            conn = sqlite3.connect(str(db_path))
            q_words = set(re.findall(r'\b[a-z]+\b', question.lower()))
            col_names: list[tuple[str, str]] = []  # (table, col)
            for tname in selected_tables:
                try:
                    cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                    for c in cols:
                        col_names.append((tname, c[1]))
                except Exception:
                    continue
            conn.close()

            # Check if any question word starts with a column name (ranked→rank, etc.)
            matches: list[str] = []
            anchor_lower = anchor_text.lower()
            for tname, col in col_names:
                col_lower = col.lower()
                if len(col_lower) < 4:
                    continue
                for word in q_words:
                    if word.startswith(col_lower) and word != col_lower and len(word) - len(col_lower) <= 3:
                        # Found a derivation — check if domain knowledge defines this column
                        if col_lower in anchor_lower:
                            # Extract the definition line from anchor_text
                            for line in anchor_text.split("\n"):
                                if col_lower in line.lower() and ":" in line:
                                    matches.append(f"⚠️ Question says \"{word}\" → column `{tname}.{col}` exists. Definition: {line.strip()}")
                                    break
                            break
            return "\n".join(matches) if matches else ""
        except Exception:
            return ""

    def _scan_literal_column_matches(
        self, db_path: Path | None, selected_tables: list[str], question: str,
    ) -> str:
        """Find question words that exactly match column names in selected tables."""
        if not db_path or not selected_tables:
            return ""
        try:
            conn = sqlite3.connect(str(db_path))
            q_words = set(re.findall(r'\b[a-z_]+\b', question.lower()))
            matches: list[str] = []
            for tname in selected_tables:
                try:
                    cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                    for c in cols:
                        col_name = c[1]
                        if col_name.lower() in q_words and len(col_name) >= 3:
                            matches.append(f"  '{col_name}' → {tname}.{col_name}")
                except Exception:
                    continue
            conn.close()
            if matches:
                return "LITERAL COLUMN MATCHES (question words that are exact column names — prefer these):\n" + "\n".join(matches)
            return ""
        except Exception:
            return ""

    def _grounding_select_tables(
        self, question: str, kg_context: str, anchor_text: str, db_path: Path | None,
        feedback: str = "", kg: KnowledgeGraph | None = None, user_intent: str = "",
    ) -> list[str]:
        """LLM picks which tables are relevant based on intent + KG structure.

        Shows table names, columns with semantic descriptions, FK relationships.
        Returns list of selected table names.
        """
        if not db_path or not db_path.exists():
            return []

        conn = sqlite3.connect(str(db_path))
        table_lines: list[str] = []
        all_tables: list[str] = []
        fk_lines: list[str] = []
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall() if not r[0].startswith("_")]
            all_tables = tables

            if len(tables) <= 10:
                conn.close()
                self._log("grounding_tables_selected", f"≤10 tables, using all: {tables}")
                return tables

            for tname in tables:
                cols_info = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
                col_parts: list[str] = []
                for c in cols_info:
                    col_name = c[1]
                    col_type = c[2]
                    desc = ""
                    if kg:
                        table_schema = kg.get_table(tname)
                        if table_schema:
                            for kg_col in table_schema.columns:
                                if kg_col.name == col_name and kg_col.description:
                                    desc = f" — {kg_col.description}"
                                    break
                    col_parts.append(f"{col_name} ({col_type}){desc}")
                table_lines.append(f"- {tname} ({row_count} rows): {', '.join(col_parts)}")

            # FK relationships from KG
            if kg:
                for src_table, fk in kg.all_foreign_keys():
                    fk_lines.append(f"  {src_table}.{fk.column} → {fk.ref_table}.{fk.ref_column}")
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

        fk_section = ""
        if fk_lines:
            fk_section = f"\nRELATIONSHIPS (Foreign Keys):\n" + "\n".join(fk_lines) + "\n"

        intent_section = ""
        if user_intent:
            intent_section = f"\nUSER INTENT:\n{user_intent}\n"

        feedback_section = ""
        if feedback:
            feedback_section = f"\n⚠️ PREVIOUS TABLE SELECTION FAILED:\n{feedback}\nYou MUST include tables that contain the needed data.\n"

        prompt = f"""QUESTION: {question}

TABLES IN DATABASE:
{chr(10).join(table_lines)}
{fk_section}{intent_section}
{f"DOMAIN KNOWLEDGE:{chr(10)}{anchor_text[:800]}" if anchor_text else ""}
{feedback_section}
Which tables are needed to answer this question?

RULES:
- The entity_of_interest table MUST be included (it owns the output columns)
- Include tables needed for filter conditions (WHERE)
- Include bridge/linking tables needed for JOIN paths (use RELATIONSHIPS above)
- Include tables with human-readable labels if the question asks for names
- 2-5 tables max

Return ONLY: {{"tables": ["table1", "table2", ...]}}"""

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

            # Inferred FKs between selected tables (from KG — already validated)
            if kg is None:
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

            # Detect bidirectional link tables (stores each relationship in both directions)
            bidir_notes = self._detect_bidirectional_tables(conn, tables)

            # Detect pre-aggregated columns using KG stats
            preagg_notes: list[str] = []
            for tname in tables:
                table_schema = kg.get_table(tname) if kg else None
                if not table_schema:
                    continue
                for col in table_schema.columns:
                    col_lower = col.name.lower()
                    if (col_lower.startswith("avg") or col_lower.startswith("total")
                            or col_lower.startswith("sum") or col_lower.startswith("num")
                            or col_lower.startswith("count")):
                        stats = table_schema.col_stats.get(col.name, {})
                        distinct = stats.get("distinct", 0)
                        if table_schema.row_count > 0 and distinct > 1:
                            preagg_notes.append(
                                f"Column {tname}.{col.name} already stores a per-row "
                                f"aggregate value ({distinct} distinct values across {table_schema.row_count} rows). "
                                f"Use WHERE {col.name} > N directly — do NOT wrap in AVG()/SUM() "
                                f"unless the question explicitly asks for an average OF averages."
                            )

            if bidir_notes or preagg_notes:
                lines.append("=== DATA STORAGE NOTES ===")
                for note in bidir_notes:
                    lines.append(f"  ⚠️ {note}")
                for note in preagg_notes:
                    lines.append(f"  ⚠️ {note}")
                lines.append("")

        finally:
            conn.close()

        result = "\n".join(lines)
        self._log("grounding_focused_schema_done", f"{len(result)} chars, {len(tables)} tables, {len(fk_lines)} FKs")
        return result

    def _detect_bidirectional_tables(
        self, conn: sqlite3.Connection, tables: list[str]
    ) -> list[str]:
        """Detect structural storage patterns (bidirectional links, etc.) via data inspection."""
        notes: list[str] = []
        for tname in tables:
            try:
                col_info = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                col_names = [c[1] for c in col_info]
            except Exception:
                continue

            # Find column pairs referencing the same entity (col/col2, col_1/col_2)
            pair_candidates: list[tuple[str, str]] = []
            for i, c1 in enumerate(col_names):
                for c2 in col_names[i + 1:]:
                    base1 = re.sub(r'[_]?\d+$', '', c1)
                    base2 = re.sub(r'[_]?\d+$', '', c2)
                    if base1 == base2 and c1 != c2:
                        pair_candidates.append((c1, c2))

            for col_a, col_b in pair_candidates:
                try:
                    row = conn.execute(
                        f'SELECT "{col_a}", "{col_b}" FROM "{tname}" LIMIT 1'
                    ).fetchone()
                    if not row or row[0] == row[1]:
                        continue
                    val_a, val_b = row[0], row[1]
                    reverse = conn.execute(
                        f'SELECT 1 FROM "{tname}" WHERE "{col_a}" = ? AND "{col_b}" = ? LIMIT 1',
                        (val_b, val_a)
                    ).fetchone()
                    if reverse:
                        notes.append(
                            f"Table '{tname}' stores BIDIRECTIONAL data: for each row ({col_a}=A, {col_b}=B), "
                            f"a reverse row ({col_a}=B, {col_b}=A) also exists. "
                            f"To count relationships per entity, JOIN on ONE column only (e.g. {col_a}) — "
                            f"do NOT use OR {col_a}/{col_b}, as that double-counts."
                        )
                except Exception:
                    continue
        return notes

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
            table_cols: dict[str, list[str]] = {}
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall():
                tname = row[0]
                if tname.startswith("_"):
                    continue
                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
                table_cols[tname] = cols
                for col in cols:
                    all_cols.append(f"{tname}.{col}")
            conn.close()
        except Exception:
            return ""

        # Identify ambiguous columns: grounding uses Table1.Col but Table2.Col also exists
        ambiguous_section = ""
        ambiguous_items: list[str] = []
        for req in data_reqs:
            if not isinstance(req, str) or "." not in req:
                continue
            parts = req.split(".", 1)
            if len(parts) != 2:
                continue
            req_table, req_col = parts
            # strip annotations like "(Filter: ...)"
            req_col_clean = re.sub(r'\s*\(.*?\)\s*$', '', req_col).strip()
            # Find other tables that also have this column
            alternatives = []
            for tname, cols in table_cols.items():
                if tname.lower() == req_table.lower():
                    continue
                if any(c.lower() == req_col_clean.lower() for c in cols):
                    alternatives.append(f"{tname}.{req_col_clean}")
            if alternatives:
                ambiguous_items.append(
                    f"  - Grounding chose {req_table}.{req_col_clean}, but also exists in: {', '.join(alternatives)}"
                )
        if ambiguous_items:
            ambiguous_section = (
                "\n\nAMBIGUOUS COLUMNS (same column name in multiple tables — evaluate EACH):\n"
                + "\n".join(ambiguous_items)
                + "\n"
            )

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
{chr(10).join(all_cols)}{knowledge_section}{ambiguous_section}

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

        self._log("disambig_cols", f"defined={list(col_definitions.keys())} schema={sorted(list(schema_cols))}")

        # Find question words that match defined columns
        matched_col = ""
        for col_name in col_definitions:
            if col_name not in schema_cols:
                continue
            for qw in q_words:
                # Exact match or question word is an inflected form of column name
                # (e.g., "ranked" matches "rank") — question word must start with column name.
                # Do NOT allow column name starting with question word (e.g., "driverref" should NOT match "driver")
                if qw == col_name or qw.startswith(col_name):
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
            self._log("disambig_bail", f"no WHERE in formula: {formula_lower}")
            return "", None

        where_clause = where_match.group(1)
        if re.search(rf'\b{re.escape(matched_col)}\b', where_clause):
            self._log("disambig_bail", f"matched_col '{matched_col}' already in WHERE: {where_clause}")
            return "", None

        self._log("disambig_proceed", f"matched_col={matched_col} where={where_clause}")

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

    def _compute_join_paths(
        self, db_path: Path, grounding: dict[str, Any], selected_tables: list[str] | None,
        kg: KnowledgeGraph | None = None,
    ) -> dict[str, Any]:
        """Extract join paths from KG FK relationships, validate each against DB."""
        data_reqs = grounding.get("data_requirements", [])
        tables_needed: set[str] = set()
        for req in data_reqs:
            if isinstance(req, str) and "." in req:
                tables_needed.add(req.split(".")[0].lower())
        if len(tables_needed) < 2:
            return grounding

        # Get FK pairs from KG (already built — no rebuild)
        if kg is None:
            kg = build_kg_from_sqlite(db_path)
        tables_lower = {t.lower() for t in tables_needed}
        join_paths: list[str] = []
        try:
            conn = sqlite3.connect(str(db_path))
            for src_table, fk in kg.all_foreign_keys():
                src_l = src_table.lower()
                ref_l = fk.ref_table.lower()
                if src_l in tables_lower and ref_l in tables_lower:
                    clause = f"{src_table}.{fk.column} = {fk.ref_table}.{fk.ref_column}"
                    if clause in join_paths:
                        continue
                    # Validate: does this join actually produce rows?
                    test_sql = (
                        f'SELECT COUNT(*) FROM "{src_table}" '
                        f'JOIN "{fk.ref_table}" ON "{src_table}"."{fk.column}" = "{fk.ref_table}"."{fk.ref_column}"'
                    )
                    try:
                        cnt = conn.execute(test_sql).fetchone()[0]
                        if cnt > 0:
                            join_paths.append(clause)
                    except Exception:
                        pass
            conn.close()
        except Exception:
            pass

        if join_paths:
            grounding["join_paths"] = join_paths
            self._log("grounding_join_paths", str(join_paths))
        return grounding

    def _validate_grounding_against_db(
        self, db_path: Path, grounding: dict[str, Any], question: str, anchor_text: str,
        user_intent: str = "",
    ) -> tuple[list[str], list[str]]:
        """Pure deterministic validation of grounding against DB.
        Returns (schema_failures, grounding_failures).
        - schema_failures: column/table doesn't exist → need to redo table selection + grounding
        - grounding_failures: wrong values/joins/arithmetic → redo grounding only
        """
        schema_failures: list[str] = []
        grounding_failures: list[str] = []
        try:
            conn = sqlite3.connect(str(db_path))
        except Exception:
            return schema_failures, grounding_failures

        try:
            # --- Check 1: Filter values exist in DB — fix format if mismatch ---
            known_values = grounding.get("known_values", {})
            format_notes = grounding.get("data_format_notes", [])
            for col_key, values in list(known_values.items()):
                if "." not in col_key or not values:
                    continue
                table_name, col_name = col_key.split(".", 1)
                if all(re.match(r'^[<>!=]', str(v).strip()) for v in values if v):
                    continue
                # Skip SQL expressions (HAVING/aggregate conditions misplaced as filter values)
                sql_keywords = ('COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'HAVING')
                if any(any(kw in str(v).upper() for kw in sql_keywords) for v in values if v):
                    continue
                # Skip pure numeric values — these are range bounds (BETWEEN/>=), not text to match
                if all(re.match(r'^-?\d+\.?\d*$', str(v).strip()) for v in values if v):
                    continue
                # Skip if too many values (date ranges etc.) — not individual filter lookups
                if len(values) > 5:
                    continue
                corrected_values = []
                for val in values:
                    try:
                        # If value already contains LIKE wildcards (% or _), validate with LIKE
                        val_str = str(val)
                        if '%' in val_str or ('_' in val_str and not val_str.replace('_', '').isalnum()):
                            like_cnt = conn.execute(
                                f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" LIKE ? COLLATE NOCASE',
                                (val_str,)
                            ).fetchone()[0]
                            if like_cnt > 0:
                                corrected_values.append(val)
                                filter_overrides = grounding.setdefault("filter_overrides", {})
                                filter_overrides[col_key] = f'"{col_name}" LIKE \'{val_str}\' COLLATE NOCASE'
                                format_notes.append(
                                    f"For {table_name}.{col_name}: use WHERE \"{col_name}\" LIKE '{val_str}' COLLATE NOCASE"
                                )
                                self._log("filter_verified", f"{col_key}='{val}' matches {like_cnt} rows via LIKE")
                                continue
                        cnt = conn.execute(
                            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" = ?',
                            (val_str,)
                        ).fetchone()[0]
                        if cnt == 0:
                            # Case-insensitive check — if value exists with different case, use correct case
                            ci_row = conn.execute(
                                f'SELECT "{col_name}" FROM "{table_name}" WHERE "{col_name}" = ? COLLATE NOCASE LIMIT 1',
                                (str(val),)
                            ).fetchone()
                            if ci_row:
                                corrected_values.append(str(ci_row[0]))
                                self._log("filter_verified", f"{col_key}='{val}' → case-corrected to '{ci_row[0]}'")
                                continue
                            scope_clauses = []
                            scope_params = []
                            for other_key, other_vals in known_values.items():
                                if other_key == col_key or "." not in other_key:
                                    continue
                                ot, oc = other_key.split(".", 1)
                                if ot != table_name:
                                    continue
                                valid_vals = [v for v in other_vals if v and not re.match(r'^[<>!=]', str(v).strip())]
                                if len(valid_vals) == 1:
                                    scope_clauses.append(f'"{oc}" = ?')
                                    scope_params.append(str(valid_vals[0]))

                            where = f'"{col_name}" IS NOT NULL AND "{col_name}" != \'\''
                            if scope_clauses:
                                where += " AND " + " AND ".join(scope_clauses)
                            distinct = conn.execute(
                                f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                                f'WHERE {where} LIMIT 15',
                                scope_params
                            ).fetchall()
                            actual_vals = [str(r[0]) for r in distinct]

                            # LLM resolves correct SQL condition
                            condition = self._resolve_filter_format(
                                val, col_name, table_name, actual_vals,
                                question=question, user_intent=user_intent,
                                all_known_values=known_values,
                                anchor_text=anchor_text,
                            )
                            if condition:
                                corrected_values.append(val)
                                filter_overrides = grounding.setdefault("filter_overrides", {})
                                filter_overrides[col_key] = condition
                                format_notes.append(
                                    f"For {table_name}.{col_name}: use WHERE {condition} (NOT = '{val}')"
                                )
                                self._log("filter_corrected", f"{col_key}: '{val}' → {condition}")
                            else:
                                corrected_values.append(val)
                                format_notes.append(
                                    f"'{val}' not in {col_name}. DB samples: {actual_vals[:5]}. "
                                    f"Use LIKE pattern to match."
                                )
                                self._log("filter_verified", f"{col_key}='{val}' NOT exact match — actual: {actual_vals[:5]}")
                        else:
                            corrected_values.append(val)
                            self._log("filter_verified", f"{col_key}='{val}' exists in {table_name} ({cnt} rows)")
                    except Exception:
                        corrected_values.append(val)
                known_values[col_key] = corrected_values
            grounding["known_values"] = known_values
            grounding["data_format_notes"] = format_notes

            # --- Check 1b: Same value in multiple columns — LLM picks correct one ---
            # Proactively find other columns in the same table that also contain each filter value
            val_to_keys: dict[str, list[str]] = {}
            for col_key, values in known_values.items():
                for val in values:
                    if val and not re.match(r'^[<>!=]', str(val).strip()):
                        val_to_keys.setdefault(str(val), []).append(col_key)
            for val, keys in list(val_to_keys.items()):
                if re.match(r'^-?\d+\.?\d*$', str(val).strip()):
                    continue
                for col_key in list(keys):
                    if "." not in col_key:
                        continue
                    table_name = col_key.split(".", 1)[0]
                    try:
                        cols_info = conn.execute(
                            f"PRAGMA table_info(\"{table_name}\")"
                        ).fetchall()
                    except Exception:
                        continue
                    chosen_col = col_key.split(".", 1)[1]
                    text_cols = [
                        r[1] for r in cols_info
                        if r[2].upper() in ("TEXT", "VARCHAR", "CHAR", "")
                        and r[1] != chosen_col
                        and f"{table_name}.{r[1]}" not in keys
                    ]
                    for tc in text_cols:
                        try:
                            hit = conn.execute(
                                f'SELECT COUNT(*) FROM "{table_name}" WHERE "{tc}" LIKE ?',
                                (f"%{val}%",)
                            ).fetchone()[0]
                            if hit > 0:
                                alt_key = f"{table_name}.{tc}"
                                if alt_key not in val_to_keys[val]:
                                    val_to_keys[val].append(alt_key)
                        except Exception:
                            continue
            for val, keys in val_to_keys.items():
                if len(keys) <= 1:
                    continue
                # Group by table — only conflict if same table has same value in 2+ cols
                table_keys: dict[str, list[str]] = {}
                for k in keys:
                    t = k.split(".", 1)[0] if "." in k else ""
                    if t:
                        table_keys.setdefault(t, []).append(k)
                for t, t_keys in table_keys.items():
                    if len(t_keys) <= 1:
                        continue
                    # Show LLM matching values in each column so it can reason
                    col_context: list[str] = []
                    clean_val = val.strip("%")
                    for candidate in t_keys:
                        c = candidate.split(".", 1)[1]
                        try:
                            matches = conn.execute(
                                f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" LIKE ? LIMIT 10',
                                (f"%{clean_val}%",)
                            ).fetchall()
                            match_list = [str(r[0]) for r in matches]
                            col_context.append(f"  {candidate} (matching values): {match_list}")
                        except Exception:
                            col_context.append(f"  {candidate}: (unknown)")
                    correct_col = self._resolve_duplicate_filter_column(
                        val, t_keys, col_context, question, anchor_text,
                    )
                    if correct_col and correct_col in t_keys:
                        # Ensure correct_col is in known_values
                        if correct_col not in known_values:
                            known_values[correct_col] = [val]
                        elif val not in known_values[correct_col]:
                            known_values[correct_col].append(val)
                        # Migrate filter_overrides to correct_col
                        filter_overrides = grounding.get("filter_overrides", {})
                        for k in t_keys:
                            if k != correct_col and k in filter_overrides:
                                old_override = filter_overrides.pop(k)
                                # Rewrite the override to reference the correct column
                                old_col = k.split(".", 1)[1] if "." in k else k
                                new_col = correct_col.split(".", 1)[1] if "." in correct_col else correct_col
                                new_override = old_override.replace(f'"{old_col}"', f'"{new_col}"')
                                filter_overrides[correct_col] = new_override
                                self._log("filter_override_migrated", f"{k} → {correct_col}: {new_override}")
                        grounding["filter_overrides"] = filter_overrides
                        blocked_cols = grounding.setdefault("_blocked_filter_cols", [])
                        for k in t_keys:
                            if k != correct_col:
                                known_values[k] = [v for v in known_values.get(k, []) if str(v) != val]
                                self._log("filter_dedup", f"Removed '{val}' from {k} (LLM chose {correct_col})")
                                overrides = grounding.setdefault("_semantic_overrides", [])
                                overrides.append(
                                    f"Do NOT filter on {k} — only use {correct_col} for '{val}'"
                                )
                                # Track the column name for domain_rules filtering
                                col_name = k.split(".", 1)[1] if "." in k else k
                                if col_name not in blocked_cols:
                                    blocked_cols.append(col_name)
                        grounding["known_values"] = {k: v for k, v in known_values.items() if v}

            # --- Check 2: Tables and columns exist in DB ---
            all_tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            all_tables_lower = {t.lower(): t for t in all_tables}

            # Collect all referenced tables from data_requirements, known_values, join_paths
            referenced_tables: dict[str, str] = {}  # normalized -> original reference
            data_reqs = grounding.get("data_requirements", [])
            for req in data_reqs:
                if not isinstance(req, str) or "." not in req:
                    continue
                parts = req.split(".", 1)
                if len(parts) == 2:
                    referenced_tables.setdefault(parts[0], parts[0])
            for col_key in grounding.get("known_values", {}):
                if "." in col_key:
                    t = col_key.split(".", 1)[0]
                    referenced_tables.setdefault(t, t)
            for jp in grounding.get("join_paths", []):
                for side in jp.split("="):
                    side = side.strip()
                    if "." in side:
                        t = side.split(".")[0]
                        referenced_tables.setdefault(t, t)

            # Validate each referenced table exists
            valid_tables: set[str] = set()
            for ref_table in referenced_tables:
                if ref_table.lower() in all_tables_lower:
                    valid_tables.add(ref_table)
                else:
                    # Find closest match by case-insensitive substring or prefix
                    suggestions = []
                    ref_lower = ref_table.lower().rstrip("s")  # strip trailing 's' for plural
                    for t_lower, t_actual in all_tables_lower.items():
                        if ref_lower in t_lower or t_lower in ref_lower or t_lower.startswith(ref_lower[:4]):
                            suggestions.append(t_actual)
                    hint = f" Did you mean: {suggestions}" if suggestions else f" Available tables: {all_tables}"
                    schema_failures.append(
                        f"Table '{ref_table}' does NOT exist in database.{hint}"
                    )

            # Validate columns in valid tables
            for req in data_reqs:
                if not isinstance(req, str) or "." not in req:
                    continue
                parts = req.split(".", 1)
                if len(parts) != 2:
                    continue
                req_table = parts[0]
                if req_table not in valid_tables:
                    continue  # already flagged above
                req_col = re.sub(r'\s*\(.*?\)\s*$', '', parts[1]).strip()
                actual_table = all_tables_lower.get(req_table.lower(), req_table)
                try:
                    cols = [c[1].lower() for c in conn.execute(f'PRAGMA table_info("{actual_table}")').fetchall()]
                    if cols and req_col.lower() not in cols:
                        schema_failures.append(
                            f"Column '{req_col}' does NOT exist in table '{req_table}'. "
                            f"Available columns: {[c[1] for c in conn.execute(f'PRAGMA table_info(\"{actual_table}\")').fetchall()]}"
                        )
                except Exception:
                    pass

            # Validate join_paths columns exist
            for jp in grounding.get("join_paths", []):
                parts = jp.split("=")
                if len(parts) != 2:
                    continue
                for side in parts:
                    side = side.strip()
                    if "." not in side:
                        continue
                    t, c = side.split(".", 1)
                    if t not in valid_tables:
                        continue  # table already flagged
                    actual_table = all_tables_lower.get(t.lower(), t)
                    try:
                        cols = [col[1].lower() for col in conn.execute(f'PRAGMA table_info("{actual_table}")').fetchall()]
                        if cols and c.lower() not in cols:
                            schema_failures.append(
                                f"Join column '{c}' does NOT exist in table '{t}'. "
                                f"Available columns: {[col[1] for col in conn.execute(f'PRAGMA table_info(\"{actual_table}\")').fetchall()]}"
                            )
                    except Exception:
                        pass

            # --- Check 3: Formula arithmetic vs domain knowledge ---
            comp_type = grounding.get("computation_type", "")
            needs_arithmetic = comp_type in (
                "ratio", "percentage", "avg", "comparison", "multi_step", "sum"
            )
            if anchor_text and needs_arithmetic:
                formula = grounding.get("formula", "")
                q_words = set(re.findall(r'\b[a-z]{4,}\b', question.lower()))
                anchor_sections = anchor_text.split("\n")
                relevant_ops: list[str] = []
                for section in anchor_sections:
                    section_words = set(re.findall(r'\b[a-z]{4,}\b', section.lower()))
                    if len(q_words & section_words) >= 2:
                        relevant_ops.extend(re.findall(r'[/\*]\s*\d+', section))
                if relevant_ops:
                    formula_lower = formula.lower()
                    for op in relevant_ops:
                        normalized = op.replace(" ", "")
                        if normalized not in formula_lower.replace(" ", ""):
                            grounding_failures.append(
                                f"Domain knowledge defines '{op.strip()}' but your formula doesn't include it. "
                                f"The formula MUST include this arithmetic operation."
                            )
                            break


            # --- Check 4: Entity of interest vs output column table ---
            # If entity_of_interest names a table and the output column exists in both
            # the grounding's chosen table AND the entity table, flag as grounding failure
            if user_intent and "prefer this table's columns for output" in user_intent:
                # Extract entity name from intent
                entity_match = re.search(r"Entity of interest:\s*(\w+)", user_intent)
                if entity_match:
                    entity_name = entity_match.group(1).lower()
                    # Find the entity table (table whose name contains the entity word)
                    entity_table = None
                    for tname in [r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()]:
                        if entity_name in tname.lower():
                            entity_table = tname
                            break
                    if entity_table:
                        # Check expected_output columns
                        expected = grounding.get("expected_output", {})
                        out_col_str = expected.get("columns", "") if isinstance(expected, dict) else ""
                        if out_col_str and isinstance(out_col_str, str):
                            # Extract table.col or just col from expected_output
                            out_parts = out_col_str.split(".")
                            if len(out_parts) == 2:
                                out_table, out_col = out_parts[0], out_parts[1]
                            else:
                                out_col = out_parts[0]
                                # Infer table from formula
                                formula = grounding.get("formula", "")
                                out_table = ""
                                for req in grounding.get("data_requirements", []):
                                    if req.lower().endswith(f".{out_col.lower()}"):
                                        out_table = req.split(".")[0]
                                        break
                            # If output is from a different table than entity, and entity table has same column
                            if out_table and out_table.lower() != entity_table.lower():
                                try:
                                    entity_cols = [c[1].lower() for c in conn.execute(
                                        f'PRAGMA table_info("{entity_table}")'
                                    ).fetchall()]
                                    if out_col.lower() in entity_cols:
                                        grounding_failures.append(
                                            f"Output column '{out_col}' exists in both '{out_table}' and '{entity_table}'. "
                                            f"Entity of interest is '{entity_name}' — use {entity_table}.{out_col} for output."
                                        )
                                except Exception:
                                    pass


            # --- Check 5: Column disambiguation via lightweight LLM ---
            if anchor_text and (grounding.get("known_values") or grounding.get("data_requirements")):
                disambiguation_rules: list[str] = []
                for line in anchor_text.split("\n"):
                    line_l = line.lower()
                    if (" vs " in line_l or " vs. " in line_l
                        or ("use " in line_l and " for " in line_l)
                        or ("based on" in line_l and "`" in line)):
                        disambiguation_rules.append(line.strip())

                if disambiguation_rules:
                    used_cols: list[str] = []
                    for col_key in grounding.get("known_values", {}):
                        if "." in col_key:
                            used_cols.append(col_key)
                    for req in grounding.get("data_requirements", []):
                        if isinstance(req, str) and "." in req:
                            used_cols.append(req.split("(")[0].strip())

                    if used_cols:
                        disambig_prompt = (
                            f"QUESTION: {question}\n\n"
                            f"COLUMNS USED IN QUERY:\n" + "\n".join(f"- {c}" for c in used_cols) + "\n\n"
                            f"DOMAIN RULES (these are AUTHORITATIVE definitions — override common sense):\n"
                            + "\n".join(f"- {r}" for r in disambiguation_rules[:10]) + "\n\n"
                            f"Based ONLY on the DOMAIN RULES above, are any COLUMNS USED wrong?\n"
                            f"- Column name affinity with question wording is CORRECT. Do not override.\n"
                            f"- Only flag if the question clearly asks for a different concept that maps to another column.\n"
                            f"- When in doubt, return empty.\n\n"
                            f"Return ONLY JSON: {{\"wrong\": [{{\"used\": \"table.col\", \"should_be\": \"table.correct_col\", \"reason\": \"...\"}}]}} or {{\"wrong\": []}}"
                        )
                        try:
                            raw = self._model_call_with_retry(
                                [ModelMessage(role="user", content=disambig_prompt)]
                            )
                            parsed = self._parse_json(raw)
                            if isinstance(parsed, dict) and parsed.get("wrong"):
                                for fix in parsed["wrong"]:
                                    if isinstance(fix, dict) and fix.get("used") and fix.get("should_be"):
                                        grounding_failures.append(
                                            f"Column '{fix['used']}' is WRONG for this question. "
                                            f"Domain knowledge says use '{fix['should_be']}' instead. "
                                            f"Reason: {fix.get('reason', '')}"
                                        )
                                        self._log("disambig_fix", f"{fix['used']} → {fix['should_be']}")
                        except Exception:
                            pass

        finally:
            conn.close()

        # --- Probe: run joins + filters as COUNT(*), check for 0 rows ---
        probe_issue = self._probe_grounding_result(db_path, grounding, user_intent)
        if probe_issue:
            grounding_failures.append(probe_issue)

        return schema_failures, grounding_failures

    def _resolve_duplicate_filter_column(
        self, val: str, candidates: list[str], col_context: list[str],
        question: str, anchor_text: str,
    ) -> str:
        """LLM decides which column a duplicated filter value belongs to."""
        prompt = (
            f"QUESTION: {question}\n\n"
            f"DOMAIN KNOWLEDGE:\n{anchor_text[:1000]}\n\n"
            f"The filter value '{val}' exists in MULTIPLE columns in the same table:\n"
            + "\n".join(col_context) + "\n\n"
            f"Which ONE column should be used to filter for '{val}' based on what the QUESTION is asking?\n"
            f"Pay close attention to the SURROUNDING WORDS in the question near '{val}' — "
            f"they indicate which entity type (and therefore which column) is intended.\n\n"
            f"Return ONLY JSON: {{\"correct_column\": \"table.column\"}}"
        )
        try:
            raw = self._model_call_with_retry([ModelMessage(role="user", content=prompt)])
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("correct_column"):
                return str(parsed["correct_column"])
        except Exception:
            pass
        return ""

    def _resolve_filter_format(
        self, query_val: str, col_name: str, table_name: str, actual_vals: list[str],
        question: str = "", user_intent: str = "", all_known_values: dict | None = None,
        anchor_text: str = "",
    ) -> str:
        """LLM figures out correct SQL WHERE clause for a mismatched filter value."""
        context_parts = [f"QUESTION: {question}"] if question else []
        if user_intent:
            context_parts.append(f"USER INTENT: {user_intent}")
        if all_known_values:
            kv_str = ", ".join(f"{k}={v}" for k, v in all_known_values.items())
            context_parts.append(f"ALL FILTERS: {kv_str}")
        if anchor_text:
            context_parts.append(f"DOMAIN KNOWLEDGE:\n{anchor_text[:1000]}")
        context = "\n".join(context_parts)

        prompt = (
            f"{context}\n\n"
            f"Column: {table_name}.{col_name}\n"
            f"Question value: '{query_val}'\n"
            f"Actual DB values: {actual_vals[:15]}\n\n"
            f"'{query_val}' does not exist in the DB. The actual DB values above show the stored format.\n"
            f"Convert '{query_val}' to the DB format prefix and use LIKE to match all rows with that prefix.\n"
            f"The condition MUST match at least one actual DB value shown above.\n\n"
            f"Return ONLY JSON: {{\"sql_condition\": \"{col_name} LIKE '...%'\"}}"
        )
        try:
            raw = self._model_call_with_retry([ModelMessage(role="user", content=prompt)])
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("sql_condition"):
                return str(parsed["sql_condition"])
        except Exception:
            pass
        return ""

    def _probe_grounding_result(
        self, db_path: Path, grounding: dict[str, Any], user_intent: str,
    ) -> str:
        """Run a simple COUNT(*) probe using grounding's join paths + filters.
        Returns empty string if OK, or a description of the mismatch."""
        join_paths = grounding.get("join_paths", [])
        known_values = grounding.get("known_values", {})
        comp_type = grounding.get("computation_type", "")
        if not join_paths:
            return ""

        # Parse intent shape
        intent_shape = ""
        if user_intent:
            for line in user_intent.split("\n"):
                if "Answer shape:" in line:
                    intent_shape = line.split("Answer shape:")[1].strip()
                    break

        # Build SQL directly from join_paths (already validated "table.col = table.col" strings)
        # Extract tables from join paths
        tables: set[str] = set()
        on_clauses: list[str] = []
        for jp in join_paths:
            parts = jp.split("=")
            if len(parts) != 2:
                continue
            left = parts[0].strip()
            right = parts[1].strip()
            if "." not in left or "." not in right:
                continue
            lt = left.split(".")[0]
            rt = right.split(".")[0]
            tables.add(lt)
            tables.add(rt)
            on_clauses.append(f'"{lt}"."{left.split(".", 1)[1]}" = "{rt}"."{right.split(".", 1)[1]}"')

        if not tables:
            return ""

        # Simple cross-join with ON conditions in WHERE (avoids ordering issues)
        from_clause = ", ".join(f'"{t}"' for t in sorted(tables))
        where_parts = list(on_clauses)

        # Add known_values as filters
        filter_overrides = grounding.get("filter_overrides", {})
        for col_key, values in known_values.items():
            if "." not in col_key or not values:
                continue
            t, c = col_key.split(".", 1)
            if t not in tables:
                continue
            if col_key in filter_overrides:
                where_parts.append(filter_overrides[col_key])
            else:
                val = values[0]
                if val and not re.match(r'^[<>!=]', str(val).strip()):
                    where_parts.append(f'"{t}"."{c}" = \'{val}\'')

        where_clause = " WHERE " + " AND ".join(where_parts) if where_parts else ""
        probe_sql = f"SELECT COUNT(*) FROM {from_clause}{where_clause}"

        try:
            conn = sqlite3.connect(str(db_path))
            row_count = conn.execute(probe_sql).fetchone()[0]
            conn.close()
        except Exception:
            return ""

        if row_count == 0:
            # Isolate which filter(s) cause 0 rows and suggest alternatives
            diagnostics = [f"Probe returned 0 rows. SQL: {probe_sql}"]
            filter_overrides = grounding.get("filter_overrides", {})
            filter_parts = []
            for col_key, values in known_values.items():
                if "." not in col_key or not values:
                    continue
                t, c = col_key.split(".", 1)
                if t not in tables:
                    continue
                if col_key in filter_overrides:
                    filter_parts.append((col_key, filter_overrides[col_key]))
                else:
                    val = values[0]
                    if val and not re.match(r'^[<>!=]', str(val).strip()):
                        filter_parts.append((col_key, f'"{t}"."{c}" = \'{val}\''))

            # Test removing each filter to find blockers
            try:
                conn2 = sqlite3.connect(str(db_path))
                all_tables = [r[0] for r in conn2.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                for col_key, cond in filter_parts:
                    remaining = [c for _, c in filter_parts if c != cond] + on_clauses
                    test_where = " WHERE " + " AND ".join(remaining) if remaining else ""
                    test_sql = f"SELECT COUNT(*) FROM {from_clause}{test_where}"
                    try:
                        cnt = conn2.execute(test_sql).fetchone()[0]
                        if cnt > 0:
                            # This filter is the blocker — check if value exists in another table
                            t, c = col_key.split(".", 1)
                            val = known_values[col_key][0] if known_values.get(col_key) else ""
                            alt_tables = []
                            for other_t in all_tables:
                                if other_t == t:
                                    continue
                                other_cols = [r[1] for r in conn2.execute(f'PRAGMA table_info("{other_t}")').fetchall()]
                                for oc in other_cols:
                                    if c.lower() in oc.lower() or oc.lower() in c.lower():
                                        try:
                                            hit = conn2.execute(
                                                f'SELECT COUNT(*) FROM "{other_t}" WHERE CAST("{oc}" AS TEXT) LIKE ?',
                                                (f'%{str(val).replace("-", "").replace(" ", "")[:6]}%',)
                                            ).fetchone()[0]
                                            if hit > 0:
                                                alt_tables.append(f"{other_t}.{oc} ({hit} rows)")
                                        except Exception:
                                            pass
                            hint = ""
                            if alt_tables:
                                hint = f" Value found in OTHER tables: {alt_tables}. Use that table instead."
                            diagnostics.append(
                                f"BLOCKER FILTER: {col_key} = '{val}' — removing it gives {cnt} rows.{hint}"
                            )
                    except Exception:
                        pass
                conn2.close()
            except Exception:
                pass
            return "\n".join(diagnostics)

        if intent_shape == "single_value" and comp_type == "simple_lookup" and row_count > 50:
            return f"Probe returned {row_count} rows for single_value lookup — filters may be too broad."

        return ""

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

                # Skip comparison operators — can't verify with equality
                if all(re.match(r'^[<>!=]', str(v).strip()) for v in values if v):
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
                # Skip comparison operators — can't verify with equality/LIKE
                if all(re.match(r'^[<>!=]', v.strip()) for v in str_values):
                    continue
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
                    # Show distinct values for the column in this filter so LLM can pick alternatives
                    col_match = re.match(r"(\w+\.)?(\w+)\s*=\s*'([^']*)'", cond.strip())
                    if col_match:
                        tbl_prefix = col_match.group(1) or ""
                        col_name = col_match.group(2)
                        # Find which table has this column in the base SQL
                        for tbl_row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                            tbl = tbl_row[0]
                            if tbl_prefix and tbl_prefix.rstrip(".").lower() not in sql.lower():
                                continue
                            try:
                                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()]
                                if col_name in cols:
                                    vals = conn.execute(
                                        f'SELECT DISTINCT "{col_name}" FROM "{tbl}" WHERE "{col_name}" IS NOT NULL AND "{col_name}" != \'\' LIMIT 10'
                                    ).fetchall()
                                    if vals:
                                        distinct = [str(v[0]) for v in vals]
                                        diagnostics.append(f"  FIX: Available values in {tbl}.{col_name}: {distinct}")
                                    break
                            except Exception:
                                pass
                    else:
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
        failed_sqls: list[str] | None = None,
        diagnosis: str = "",
    ) -> tuple[dict[str, Any] | None, str]:
        """Generate multiple SQL interpretations and execute first that returns data. Kept lean to avoid timeout."""
        if not db_path or not db_path.exists():
            return None, ""

        failed_section = ""
        if failed_sqls:
            failed_section = "\nFAILED SQL (do NOT repeat):\n" + failed_sqls[-1][:300]

        diag_section = ""
        if diagnosis:
            diag_section = f"\nDIAGNOSIS:\n{diagnosis}"

        prompt = f"""Previous SQL returned 0 rows.

QUESTION: {question}

SCHEMA:
{kg_context[:3000]}

SAMPLES:
{sample_data[:500]}

{f"DOMAIN: {knowledge_text[:500]}" if knowledge_text else ""}
{failed_section}
{diag_section}

Generate 3 DIFFERENT SQL queries. Each must try a DIFFERENT column, join, filter value, or format.

Return ONLY: {{"hypotheses": [{{"sql": "SELECT ..."}}, ...]}}

RULES:
- Each must be materially different
- Use LIKE for text matching when unsure of format
- Use actual DB values from DIAGNOSIS/SAMPLES if available
- NEVER use AS to rename columns
- ALWAYS double-quote column names that contain spaces (e.g. "School Name", "District Type")
- ONLY use columns that appear in the SCHEMA above. NEVER invent or guess column names."""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        parsed = self._parse_json(raw)

        if not isinstance(parsed, dict):
            return None, ""

        hypotheses = parsed.get("hypotheses", [])
        if not hypotheses:
            return None, ""

        valid_hyps = [h for h in hypotheses[:3] if h.get("sql", "").strip()]
        if not valid_hyps:
            return None, ""

        # Build set of valid column names from all tables in the DB
        valid_columns: set[str] = set()
        try:
            conn = sqlite3.connect(str(db_path))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            for t in tables:
                for r in conn.execute(f'PRAGMA table_info("{t}")').fetchall():
                    valid_columns.add(r[1])
            conn.close()
        except Exception:
            pass

        # Execute in order, return first that produces rows
        for i, hyp in enumerate(valid_hyps):
            sql = hyp["sql"]
            # Reject SQL that references non-existent columns (SQLite treats them as string literals)
            if valid_columns:
                quoted_refs = re.findall(r'"([^"]+)"', sql)
                invalid = [r for r in quoted_refs if r not in valid_columns and r not in tables]
                if invalid:
                    self._log("hypothesis_try", f"Option {i}: REJECTED (invalid columns: {invalid})")
                    continue
            self._log("hypothesis_try", f"Option {i}: {sql}")
            result = self._try_sql(db_path, sql)
            if result and result.get("rows"):
                all_null = all(
                    all(v is None or str(v).strip().lower() in ("none", "null", "") for v in row)
                    for row in result["rows"]
                )
                if not all_null:
                    self._log("hypothesis_success",
                              f"Option {i}: cols={result['columns']}, rows={len(result['rows'])}")
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
        self._log("python_fallback", f"Executing Python ({len(code)} chars): {parsed.get('reasoning', '')}")

        result = execute_python_code(
            context_root=db_path.parent,
            code=code,
            timeout_seconds=30,
        )

        if not result.get("success"):
            self._log("python_error", f"Failed: {result.get('error', '')}")
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
                    self._log("python_retry_error", f"Still failed: {result.get('error', '')}")
                    return None

        output = result.get("output", "").strip()
        if not output:
            self._log("python_empty", "No output produced")
            return None

        self._log("python_output", output)

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
                self._log("shape_fix_error", f"Result contains {issue_desc}: {first_row_str}")
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
                # Only flag if VALUES look like opaque hashes (e.g. Airtable "rec..." IDs)
                for v in sample_vals:
                    if v and v.startswith("rec") and len(v) > 10 and v[3:].isalnum():
                        has_raw_id = True
                        raw_id_cols.append((i, col, sample_vals[0]))
                        break

            # Skip if there's already a human-readable column alongside the ID
            if has_raw_id and len(raw_id_cols) > 0 and len(cols) > len(raw_id_cols):
                non_id_cols = [c for i, c in enumerate(cols) if i not in {idx for idx, _, _ in raw_id_cols}]
                has_readable = any(
                    any(w in c.lower() for w in ("name", "title", "label", "description", "forename", "surname"))
                    for c in non_id_cols
                )
                if has_readable:
                    # Just drop the ID columns instead of re-querying
                    keep_indices = [i for i in range(len(cols)) if i not in {idx for idx, _, _ in raw_id_cols}]
                    data_result = {
                        "columns": [cols[i] for i in keep_indices],
                        "rows": [[row[i] for i in keep_indices] for row in rows],
                    }
                    self._log("shape_fix_fk", f"Dropped raw ID columns: {[c for _, c, _ in raw_id_cols]}")
                    cols = data_result["columns"]
                    rows = data_result["rows"]
                    has_raw_id = False
                    raw_id_cols = []

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

        # Check: "how many" question expects a single count but got multiple rows
        count_patterns = [r"how many (?:of them|of these|of those)?\s*(?:are|is|were|was|have|had)\b",
                          r"how many .+? (?:are|is|were|was)\b"]
        expects_count = any(re.search(p, q_lower) for p in count_patterns)
        list_indicators = ["list", "what are", "identify", "name the", "which"]
        has_list = any(p in q_lower for p in list_indicators)
        if expects_count and not has_list and len(rows) > 1:
            self._log("shape_fix_count", f"'how many' question returned {len(rows)} rows — re-querying as COUNT")
            fix_prompt = f"""The SQL returned {len(rows)} rows but the question asks "how many" — it expects a SINGLE COUNT number.

QUESTION: {question}
CURRENT RESULT: {len(rows)} rows, columns={cols}
FIRST ROWS: {rows[:3]}

Rewrite the query to return COUNT(*) — the number of items matching the criteria, not the items themselves.

DATABASE SCHEMA:
{kg_context[:2000]}

{grounding_context[:1000]}

Return ONLY: {{"sql": "SELECT COUNT(...) ..."}}"""
            messages = [ModelMessage(role="user", content=fix_prompt)]
            raw = self._model_call_with_retry(messages)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("sql"):
                fix_result = self._try_sql(db_path, parsed["sql"])
                if fix_result and fix_result.get("rows") and len(fix_result["rows"]) == 1:
                    self._log("shape_fixed_count", f"COUNT result: {fix_result['rows'][0]}")
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
        """Call model with 60s timeout. Returns empty string on failure (no retry)."""
        try:
            result = self.model.complete(messages)
            return result if result else ""
        except RuntimeError as e:
            self._log("llm_error", f"LLM call failed: {e}")
            return ""

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
