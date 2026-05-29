"""DataAgent: LLM-driven agent with queryable Knowledge Graph.

Pipeline:
  1. [Deterministic] Consolidate data → SQLite, build KG, discover joins, profile schema
  2. [LLM Agent Loop] Tools: overview, schema, topology, knowledge, run_sql, answer
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.kg_approach_graph import ApproachGraph
from data_agent_baseline.agents.kg_consult_agent import ConsultAgent
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.pipeline.context_scanner import scan_context
from data_agent_baseline.pipeline.kg_builder import (
    KGQueryService,
    KnowledgeGraph,
    build_alias_registry,
    build_business_topology,
    build_kg_from_sqlite,
    discover_joins_with_llm,
    profile_schema,
)
from data_agent_baseline.tools.kg_tools import (
    detect_ambiguities,
    format_resolution_prompt,
    tool_distribution,
    tool_execute_python,
    tool_find_value,
    tool_knowledge,
    tool_ontology,
    tool_overview,
    tool_resolve,
    tool_run_sql,
    tool_schema,
    tool_topology,
)
from data_agent_baseline.tools.knowledge_graph import consolidate_to_sqlite

logger = logging.getLogger(__name__)

AGENT_SYSTEM = """\
You are a data analyst. Answer the question by exploring a SQLite database.
Respond with exactly ONE JSON object per turn. No markdown, no extra text.
Always include a "reasoning" field explaining WHY you chose this action/filter.

== WORKFLOW ==
1. overview → understand tables and joins
2. schema/ontology/find_value → understand columns, valid values, where entities live
3. run_sql → verify filters and check data before answering
4. answer → submit verified query

== TOOLS ==

overview()
  INTENT: See all tables, their roles, row counts, columns, and joins.
  Use this FIRST to understand the database structure.
  PARAMS: none
  RETURNS: Table list with roles (fact/dim/bridge), grain, PKs, joins.
  ERRORS: none
  FORMAT: {"action": "overview", "reasoning": "why"}

schema(table)
  INTENT: Get full column details for one table — types, stats, samples,
  semantic roles, collisions with other tables.
  Use when you need to understand a specific table before writing SQL.
  PARAMS: table — exact table name from overview
  RETURNS: Column list with types, unique counts, sample values, collisions.
  ERRORS: Returns "table not found" if name is wrong — check overview.
  FORMAT: {"action": "schema", "table": "TableName", "reasoning": "why"}

topology(tables)
  INTENT: Get join paths and relationship cardinality between tables.
  Use when you need to JOIN multiple tables.
  PARAMS: tables — list of 2+ table names
  RETURNS: Join conditions, FK paths, cardinality.
  ERRORS: Returns empty if no path exists — tables may not be related.
  FORMAT: {"action": "topology", "tables": ["A", "B"], "reasoning": "why"}

ontology(column)
  INTENT: Get semantic metadata for a column — type, role, valid values, relationships.
  Use when you need to understand what a column represents or its valid values.
  PARAMS: column — "table.column" or just "column" (searches all tables)
  RETURNS: Type, role (grain/measure/temporal), stats, top values, FK links.
  ERRORS: Returns "not found" if column doesn't exist.
  FORMAT: {"action": "ontology", "column": "table.column", "reasoning": "why"}

knowledge(query)
  INTENT: Search domain knowledge for definitions, thresholds, formulas, \
ID-to-name mappings, and schema descriptions.
  Use for: column meanings, valid value ranges, metric formulas, entity lookups.
  NOT for: filtering/aggregating many records — use execute_python for that.
  PARAMS: query — search terms, IDs, or values to look up
  RETURNS: Relevant paragraphs matching the query.
  ERRORS: Returns empty if no knowledge matches — proceed without it.
  FORMAT: {"action": "knowledge", "query": "search terms", "reasoning": "why"}

distribution(table, column)
  INTENT: Get the statistical distribution of a numeric column — percentiles, \
histogram, and suggested normal/abnormal thresholds from the actual data.
  Use when the question mentions "normal", "abnormal", "high", or "low" and \
knowledge did not provide an explicit threshold.
  PARAMS: table — exact table name from overview, column — exact column name
  RETURNS: Count, min, max, mean, median, percentiles (P10/P25/P75/P90), IQR, \
suggested normal/abnormal cutoffs, histogram with 5 buckets.
  ERRORS: Returns "not numeric" if column contains text — use run_sql with \
SELECT DISTINCT instead. Returns "not found" if table/column doesn't exist.
  FORMAT: {"action": "distribution", "table": "TableName", "column": "ColName", "reasoning": "why"}

find_value(value)
  INTENT: Search the knowledge graph for which tables/columns contain a value.
  Use when you need to know WHERE a specific entity or category lives in the schema.
  PARAMS: value — the text value to search for
  RETURNS: Matching table.column pairs with row counts and related tables via JOINs.
  ERRORS: Returns "not found" if value doesn't exist anywhere.
  FORMAT: {"action": "find_value", "value": "search term", "reasoning": "why"}

resolve(term)
  INTENT: Resolve a user-facing name/term to its canonical DB value, or expand a \
group entity to its member IDs.
  Use when the question references an entity by informal name, abbreviation, or \
a higher-level group (e.g. "Line 1", "injection molding machine", "M1").
  PARAMS: term — the user's term to resolve (as it appears in the question)
  RETURNS: Canonical DB value with table.column location and SQL WHERE clause. \
For group entities, returns the list of child IDs with an IN clause.
  ERRORS: Returns "not found" if no alias or topology match — try find_value instead.
  FORMAT: {"action": "resolve", "term": "user term", "reasoning": "why"}

run_sql(sql)
  INTENT: Execute exploratory SQL to check values, verify formats, test filters.
  Use BEFORE answering to confirm your filter returns the expected data.
  BEST PRACTICES:
    1. Check DISTINCT values of filter columns first — data may use abbreviations
    2. Verify row counts make sense for the question scope
    3. If results have duplicates, you need SELECT DISTINCT in your answer
    4. If results include NULL/empty rows, add WHERE col IS NOT NULL AND col != ''
  PARAMS: sql — valid SQLite query (use LIMIT 10 for exploration)
  RETURNS: Column names + rows. If 0 rows, a FORMAT hint may appear showing actual stored values.
  ERRORS: Returns error message if SQL is invalid — fix syntax and retry.
  FORMAT: {"action": "run_sql", "sql": "SELECT ...", "reasoning": "why"}

answer(sql)
  INTENT: Submit your final answer query.
  Use ONLY when you have verified the filter works via run_sql.
  PARAMS: sql — final query (NO LIMIT unless question asks for top-N)
  RETURNS: Result is used as the answer — ensure it matches the question.
  ERRORS: Returns error if 0 rows — go back to run_sql to investigate.
  FORMAT: {"action": "answer", "sql": "SELECT ...", "reasoning": "why"}

execute_python(code)
  INTENT: Run Python code to parse/extract data from source documents or \
perform calculations not possible in SQL alone.
  Use when data lives in documents (not SQL tables) and you need to extract \
structured records, parse prose, or compute values from text.
  PREREQUISITE: You must run_sql first to get IDs before searching documents.
  PARAMS: code — Python code. Available variables:
    - `docs` (str): full text of all source documents
    - `db_path` (str): path to SQLite database
    - `re`, `json`, `sqlite3`, `collections` are pre-imported
    Print your results — stdout is captured.
  RETURNS: Stdout output (max 3000 chars).
  ERRORS: Returns error with traceback if code fails.
  FORMAT: {"action": "execute_python", "code": "...", "reasoning": "why"}

recall_approaches(tables, columns, agg, grain, temporal, joins)
  INTENT: Query your memory of past failed SQL approaches to avoid repeating them.
  Use when you're stuck or want to see what strategies have already been tried \
for specific tables, columns, join paths, or structural patterns.
  PARAMS (all optional, combine any):
    tables — filter by table names involved
    columns — filter by select or filter columns
    agg — filter by aggregation type (COUNT, SUM, AVG, MIN, MAX)
    grain — filter by result grain (scalar, single_row, multi_row, grouped)
    temporal — filter by temporal column used in filters
    joins — filter by join conditions
  RETURNS: Full history of matching failed approaches with SQL, structural \
breakdown, result, and failure reason. Never truncated.
  FORMAT: {"action": "recall_approaches", "tables": ["T"], "columns": ["C"], \
"agg": "COUNT", "grain": "scalar", "temporal": "date", "joins": ["A.id=B.id"], \
"reasoning": "why"}

recall_memory(entity)
  INTENT: Query the knowledge graph memory for facts about a specific table, \
column, or topic you have learned during exploration.
  Use to recall: column distributions, domain mappings, resolved values, \
prior query results, knowledge findings.
  PARAMS: entity (optional) — table name, "table.column", or topic like \
"_knowledge", "_resolve". If omitted, returns full memory state.
  RETURNS: All facts stored for the entity or the full graph state.
  FORMAT: {"action": "recall_memory", "entity": "tablename", "reasoning": "why"}

== SQL RULES ==
- Double-quote all identifiers: "Table"."Column"
- COLLATE NOCASE for text comparisons
- CAST numerator AS REAL for division. For ratios across tables, use \
COUNT(DISTINCT) in subqueries — NEVER cross-join two tables and count.
- For min/max/best/worst: use HAVING or WHERE col = (SELECT MIN/MAX(col)...) \
to include all ties. If domain knowledge shows ORDER BY + LIMIT 1, use that.
- Exclude empty strings/NULLs on ORDER BY, MIN, MAX columns
- SELECT DISTINCT when JOINs can duplicate rows or when exploratory results show repeated values
- SELECT only the columns the question asks about. NEVER use SELECT *. \
For "list all X" questions, return only the identifying column (ID or name). \
NEVER add extra computed columns or aggregates unless explicitly requested.
- Return ALL rows matching the filter. For top-N questions use LIMIT N.
- Answer SQL must use the SAME filter verified in run_sql. Do NOT tighten \
(e.g. LIKE to =) or add extra conditions.
- For columns with few distinct values (categorical): use exact = match, \
NOT LIKE. Each distinct value is a separate category. When the question says \
"X in Y" and values exist for both X and "X + modifier", pick the exact match.
- When the question uses a word that exactly matches a column name in any table \
(e.g. "type" matches event.type), prefer that column over synonyms \
(e.g. category). The question author chose that word deliberately.
- Temporal grain: when the question asks "average monthly/daily/yearly" and \
data is stored at that grain (one row per period), ALWAYS divide by the \
number of periods (e.g. /12 for monthly in a year, /365 for daily in a year). \
This is mandatory — do NOT skip the division.
- Pre-aggregated columns: columns whose names imply aggregation (prefixed \
with Avg, Sum, Total, Count, Num, etc.) already store computed values per row. \
Filter them with WHERE — do NOT re-aggregate with GROUP BY HAVING. \
HAVING is only needed when computing a new aggregate from raw data at query time.
- "Average number of X that Y have": use CAST(COUNT(X) AS REAL) / COUNT(DISTINCT Y). \
Do NOT use AVG(subquery) which double-counts when relationships are many-to-many. \
Use only the declared FK join path — do NOT add OR conditions on other columns \
to "catch more matches". The FK relationship defines the correct counting logic.
- Only SELECT columns that directly answer the question. Never include IDs, \
join keys, or helper columns in your final answer.
- Filter values with exact equality (=) by default. Only broaden to LIKE or IN \
with variants if exact match returns 0 rows.
- Do NOT add WHERE col IS NOT NULL unless the question explicitly excludes \
missing data. Aggregations already ignore NULLs — a NULL filter changes which \
rows participate in OTHER columns' calculations.
- When resolving a FK reference, always JOIN to the target table via the ID \
column. Never use denormalized text fields — they are often NULL or stale.

== THRESHOLD RULES ==
- When the question mentions "normal"/"abnormal"/"high"/"low" for a numeric column:
  1. FIRST check knowledge tool for an explicit threshold.
  2. If knowledge has a threshold or reference range, use it exactly.
  3. If knowledge has NO threshold, use standard medical/domain reference ranges \
from your training data (e.g., WBC normal: 3.5-10.5 x10^3/µL, \
Fibrinogen normal: 150-400 mg/dL, PLT normal: 100-400 x10^3/µL). \
The data may use different units — that's OK, apply the standard definition. \
If ALL values fall outside a standard range, they are ALL abnormal.
  4. Only fall back to distribution percentiles (P10-P90) if NO standard \
reference range exists for the metric in question.
- "How many patients" means COUNT(DISTINCT patient_id), NOT COUNT(*) of rows. \
One patient may have multiple records (lab tests, visits). Always use \
COUNT(DISTINCT) on the grain/PK column when counting entities across fact tables.

== FOREIGN KEY RESOLUTION ==
- If a value looks like an internal ID/code rather than a human-readable answer, \
use the knowledge tool to resolve it before returning it as the answer.

== DOCUMENT CROSS-REFERENCING ==
Documents reference records by internal IDs from SQL tables, not human names. \
Get IDs from SQL first, then search docs for those IDs via execute_python.

== PRIORITY ==
1. [Intent] → AUTHORITATIVE. Your final SQL MUST include ALL parts of the intent:
   - RETURN columns/formula → SELECT exactly those
   - FILTER conditions → implement each one (WHERE or HAVING as appropriate)
   - If intent specifies a column from another table → JOIN that table
   NEVER skip or simplify any part of the intent.
2. Domain knowledge SQL → AUTHORITATIVE. Use the exact values and conditions \
shown. Do NOT override with your own interpretation.
3. USER QUESTION wording → determines WHAT to select and WHERE from
4. Your own reasoning → ONLY when domain knowledge and intent are silent"""


def _normalize_sql(sql: str) -> str:
    """Normalize SQL to a canonical form for dedup comparison."""
    s = " ".join(sql.split()).lower()
    s = s.replace('"', "").replace("'", "").replace("`", "")
    for kw in (
        "select",
        "from",
        "where",
        "join",
        "on",
        "and",
        "or",
        "order by",
        "group by",
        "distinct",
        "limit",
        "inner",
        "left",
        "right",
        "as",
        "is",
        "not",
        "null",
        "in",
        "between",
        "like",
        "having",
        "count",
        "sum",
        "avg",
        "min",
        "max",
    ):
        s = s.replace(f" {kw} ", f" {kw.upper()} ")
        if s.startswith(f"{kw} "):
            s = f"{kw.upper()} " + s[len(kw) + 1 :]
    return s


def _normalize_query(query: str) -> str:
    """Normalize a knowledge query — sorted lowercase tokens."""
    tokens = sorted(set(query.lower().split()))
    return " ".join(tokens)


def _extract_sql_from_line(line: str) -> str | None:
    """Extract a SQL statement from a markdown line (backticks, bullet, bare)."""
    stripped = line.strip()
    sql = None
    bt_start = stripped.find("`")
    if bt_start != -1:
        bt_end = stripped.rfind("`")
        if bt_end > bt_start:
            inner = stripped[bt_start + 1 : bt_end].strip()
            if inner.upper().startswith("SELECT "):
                sql = inner
    if sql is None:
        clean = stripped.lstrip("-*# ").strip()
        if clean.upper().startswith("SELECT "):
            sql = clean
    return sql


def _extract_where_conditions(line: str) -> str | None:
    """Extract WHERE conditions from a SQL statement in domain knowledge.

    Returns just the WHERE clause content (value mappings), not the full SQL.
    """
    sql = _extract_sql_from_line(line)
    if not sql:
        return None
    upper = sql.upper()
    where_pos = upper.find(" WHERE ")
    if where_pos == -1:
        return None
    after_where = sql[where_pos + 7 :]
    for kw in (" ORDER BY ", " LIMIT ", " GROUP BY "):
        kw_pos = after_where.upper().find(kw)
        if kw_pos != -1:
            after_where = after_where[:kw_pos]
    return after_where.strip() if after_where.strip() else None


def _extract_order_limit(line: str) -> str | None:
    """Extract ORDER BY + LIMIT clause from domain knowledge SQL."""
    sql = _extract_sql_from_line(line)
    if not sql:
        return None
    upper = sql.upper()
    order_pos = upper.find(" ORDER BY ")
    if order_pos == -1:
        return None
    return sql[order_pos:].strip()






def _check_column_resolution(
    sql: str,
    column_resolutions: dict[str, str],
    db_path: Path,
) -> str | None:
    """Check if the answer SQL uses resolved columns from the correct table.

    If the resolution says column X should come from table T, but the SQL
    either (a) doesn't include table T at all, or (b) selects X from a different
    table, return a hint telling the agent to fix it.
    """
    if not column_resolutions:
        return None

    sql_lower = sql.lower()
    select_end = sql_lower.find("from")
    if select_end == -1:
        return None
    select_part = sql_lower[:select_end]

    for col, correct_table in column_resolutions.items():
        col_lower = col.lower()
        # Is this column in the SELECT clause?
        if col_lower not in select_part:
            continue

        # Is the correct table in the query at all?
        if correct_table not in sql_lower:
            return (
                f"COLUMN ERROR: You resolved \"{col}\" to come from "
                f"\"{correct_table}\", but your SQL doesn't use that table. "
                f"JOIN \"{correct_table}\" and SELECT \"{col}\" from it."
            )

        # Check if the column is qualified with a DIFFERENT table/alias
        # Look for pattern: other_table."col" or other_alias."col"
        # where other != correct_table

        # Find table.col or alias.col patterns in SELECT
        pattern = r'(\w+)\s*\.\s*"?' + re.escape(col_lower) + r'"?'
        matches = re.findall(pattern, select_part, re.IGNORECASE)
        for table_ref in matches:
            ref_lower = table_ref.lower().strip('"')
            if ref_lower == correct_table:
                continue
            # Check if this ref is an alias for the correct table
            # Look for: correct_table AS ref or correct_table ref
            alias_pattern = (
                correct_table + r'\s+(?:AS\s+)?' + re.escape(ref_lower)
            )
            if re.search(alias_pattern, sql_lower):
                continue
            # It's from the wrong table
            return (
                f"COLUMN ERROR: You resolved \"{col}\" to come from "
                f"\"{correct_table}\", but your SQL selects it from "
                f"\"{table_ref}\". Fix the SELECT to use "
                f"\"{correct_table}\".\"{col}\"."
            )

    return None


def _find_outer_where(sql_upper: str) -> str | None:
    """Extract the WHERE clause of the outermost query, ignoring subqueries."""
    depth = 0
    i = 0
    where_start = -1
    while i < len(sql_upper):
        if sql_upper[i] == "(":
            depth += 1
        elif sql_upper[i] == ")":
            depth -= 1
        elif depth == 0 and sql_upper[i:i + 7] == " WHERE ":
            where_start = i + 7
            break
        i += 1

    if where_start == -1:
        return None

    # Find the end: next top-level ORDER BY / LIMIT / GROUP BY / HAVING
    end = len(sql_upper)
    depth = 0
    i = where_start
    while i < len(sql_upper):
        if sql_upper[i] == "(":
            depth += 1
        elif sql_upper[i] == ")":
            depth -= 1
        elif depth == 0:
            for kw in (" ORDER BY ", " LIMIT ", " GROUP BY ", " HAVING "):
                if sql_upper[i:i + len(kw)] == kw:
                    end = i
                    break
            if end != len(sql_upper):
                break
        i += 1

    return sql_upper[where_start:end].strip() or None


def _split_outer_and(where_clause: str) -> list[str]:
    """Split a WHERE clause by AND at depth 0 only."""
    parts: list[str] = []
    depth = 0
    current = ""
    i = 0
    while i < len(where_clause):
        if where_clause[i] == "(":
            depth += 1
            current += where_clause[i]
        elif where_clause[i] == ")":
            depth -= 1
            current += where_clause[i]
        elif depth == 0 and where_clause[i:i + 5] == " AND ":
            parts.append(current.strip())
            current = ""
            i += 5
            continue
        else:
            current += where_clause[i]
        i += 1
    if current.strip():
        parts.append(current.strip())
    return [p for p in parts if p]


def _check_unverified_filter(
    sql: str, verified_sql: str | None, failed: list[str]
) -> str | None:
    """Reject answer if it introduces WHERE conditions not tested via run_sql.

    Checks both verified_sql (returned rows) and failed queries (returned 0 rows)
    since both represent conditions the agent has already tested.
    """
    if not verified_sql and not failed:
        return None

    sql_upper = sql.upper()

    # Extract outer WHERE clause only
    answer_where = _find_outer_where(sql_upper)
    if not answer_where:
        return None

    # Build pool of all tested conditions (from verified + failed queries)
    tested_pool = ""
    if verified_sql:
        vw = _find_outer_where(verified_sql.upper())
        if vw:
            tested_pool += " " + vw

    for entry in failed:
        entry_upper = entry.upper()
        # failed entries have " → reason" appended
        arrow_pos = entry_upper.find(" →")
        clean = entry_upper[:arrow_pos] if arrow_pos != -1 else entry_upper
        fw = _find_outer_where(clean)
        if fw:
            tested_pool += " " + fw

    if not tested_pool.strip():
        return None

    # Split into individual conditions by AND (respecting parens)
    answer_conds = _split_outer_and(answer_where)

    # Normalize: strip COLLATE NOCASE for comparison
    def _strip_collate(s: str) -> str:
        return s.replace(" COLLATE NOCASE", "").replace("COLLATE NOCASE", "")

    tested_normalized = _strip_collate(tested_pool)

    # Check each answer condition — is it present in any tested query?
    new_conds = []
    for cond in answer_conds:
        cond_normalized = _strip_collate(cond)
        if cond_normalized not in tested_normalized and cond not in tested_pool:
            new_conds.append(cond)

    if not new_conds:
        return None

    return (
        f"UNVERIFIED FILTER: Your answer SQL introduces conditions not tested "
        f"via run_sql: {', '.join(new_conds)}. "
        f"Test your complete WHERE clause with run_sql first, then submit the "
        f"same query as your answer."
    )


def _check_vacuous_filter(sql: str, db_path: Path) -> str | None:
    """Detect when a numeric comparison filter is vacuous (doesn't reduce rows).

    Strips LIMIT before comparing, then runs the query WITH and WITHOUT
    the numeric comparison. If row count is the same, the filter adds no
    selectivity — likely wrong column.
    """
    import sqlite3

    # Strip LIMIT/ORDER BY for comparison purposes
    check_sql = sql
    check_upper = check_sql.upper()
    limit_pos = check_upper.rfind(" LIMIT ")
    if limit_pos != -1:
        check_sql = check_sql[:limit_pos]
        check_upper = check_sql.upper()
    order_pos = check_upper.rfind(" ORDER BY ")
    if order_pos != -1:
        check_sql = check_sql[:order_pos]
        check_upper = check_sql.upper()

    if "WHERE" not in check_upper:
        return None

    where_start = check_upper.index("WHERE")
    where_part = check_sql[where_start + 5 :]  # after "WHERE "

    # Split WHERE conditions by AND
    conditions: list[str] = []
    depth = 0
    current = ""
    for ch in where_part:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        current += ch
        if depth == 0 and current.upper().rstrip().endswith(" AND"):
            conditions.append(current[:-4].strip())
            current = ""
    if current.strip():
        conditions.append(current.strip())

    # Find conditions with numeric comparisons (<, >, <=, >=)
    numeric_conds: list[tuple[int, str]] = []
    for idx, cond in enumerate(conditions):
        for comp in ("<=", ">=", "<", ">"):
            if comp in cond:
                numeric_conds.append((idx, cond))
                break

    if not numeric_conds:
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA query_only = ON")

        original_count = conn.execute(
            f"SELECT COUNT(*) FROM ({check_sql})"
        ).fetchone()[0]

        if original_count == 0:
            conn.close()
            return None

        for idx, cond in numeric_conds:
            # Skip if this is one half of a range pair on the same column
            cond_col = cond.strip().split()[0].lower()
            is_range_pair = any(
                i != idx and other.strip().split()[0].lower() == cond_col
                for i, other in numeric_conds
            )
            if is_range_pair:
                continue

            remaining = [c for i, c in enumerate(conditions) if i != idx]
            if remaining:
                new_where = "WHERE " + " AND ".join(remaining)
            else:
                new_where = ""
            without_sql = check_sql[:where_start] + new_where
            try:
                count_without = conn.execute(
                    f"SELECT COUNT(*) FROM ({without_sql})"
                ).fetchone()[0]
                if count_without == original_count and original_count > 1:
                    # Extract the comparison operator and threshold
                    comp_op = ""
                    threshold = ""
                    for comp in ("<=", ">=", "<", ">"):
                        if comp in cond:
                            comp_op = comp
                            threshold = cond.split(comp, 1)[1].strip()
                            break
                    # Test the threshold against other numeric columns
                    # by querying each table's column independently
                    effective: list[str] = []
                    if comp_op and threshold:
                        failed_col = cond.strip().split()[0]
                        failed_lower = (
                            failed_col.split(".")[-1]
                            .strip('"').strip("`").lower()
                        )
                        # Get tables used in the query (from FROM/JOIN)
                        try:
                            cursor = conn.execute(
                                "SELECT name FROM sqlite_master "
                                "WHERE type='table'"
                            )
                            all_tables = [r[0] for r in cursor.fetchall()]
                            for tbl in all_tables:
                                cursor = conn.execute(
                                    f'PRAGMA table_info("{tbl}")'
                                )
                                for col_info in cursor.fetchall():
                                    col_type = (col_info[2] or "").upper()
                                    col_n = col_info[1]
                                    if col_type not in (
                                        "INTEGER", "REAL", "NUMERIC",
                                        "INT",
                                    ):
                                        continue
                                    if col_n.lower() == failed_lower:
                                        continue
                                    if col_n.lower().endswith("id"):
                                        continue
                                    # Simple independent check: does this
                                    # column have values both above and
                                    # below the threshold?
                                    try:
                                        cnt_pass = conn.execute(
                                            f'SELECT COUNT(*) FROM "{tbl}" '
                                            f'WHERE "{col_n}" '
                                            f"{comp_op} {threshold}"
                                        ).fetchone()[0]
                                        cnt_total = conn.execute(
                                            f'SELECT COUNT(*) FROM "{tbl}"'
                                        ).fetchone()[0]
                                        if (
                                            0 < cnt_pass < cnt_total
                                            and cnt_total > 1
                                        ):
                                            effective.append(
                                                f'"{tbl}"."{col_n}"'
                                            )
                                    except sqlite3.OperationalError:
                                        continue
                        except sqlite3.OperationalError:
                            pass
                    evidence = ""
                    if effective:
                        evidence = (
                            f" Columns where {comp_op} {threshold} "
                            f"actually filters rows: {effective[:4]}"
                        )
                    conn.close()
                    return (
                        f"FILTER CHECK: \"{cond.strip()}\" has no effect "
                        f"(same {original_count} rows with or without it). "
                        f"Wrong column for this filter.{evidence}"
                    )
            except sqlite3.OperationalError:
                continue

        conn.close()
    except Exception:
        pass

    return None




def _check_empty_order(
    sql: str, result: dict[str, Any], db_path: Path
) -> str | None:
    """Reject if ORDER BY + LIMIT and the ordered column has empty values."""
    import sqlite3

    upper = sql.upper()
    if "ORDER BY" not in upper or "LIMIT" not in upper:
        return None
    # Only check ASC ordering (empty strings sort before real values)
    order_pos = upper.rfind("ORDER BY")
    after_order = upper[order_pos + 9 :].strip()
    if "DESC" in after_order.split("LIMIT")[0]:
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA query_only = ON")
        # Get first row from the original query (with LIMIT)
        cursor = conn.execute(sql)
        first_row = cursor.fetchone()
        if not first_row:
            conn.close()
            return None
        # Get first row WITHOUT the empty-string exclusion problem by
        # re-running with the LIMIT 1 — check if any value is empty
        # Actually: run the base query with LIMIT 1 and include the ORDER col
        # Simpler: check if there are empty values that would sort first
        # Extract ORDER BY column reference from original SQL
        order_section = sql[order_pos + 9 :]
        limit_in_order = order_section.upper().find("LIMIT")
        if limit_in_order != -1:
            order_section = order_section[:limit_in_order]
        col_ref = order_section.strip().split()[0].rstrip(",")
        # Remove ASC/DESC
        if col_ref.upper() in ("ASC", "DESC"):
            conn.close()
            return None
        # Query: does removing empty values change the first result?
        where_pos = upper.find("WHERE")
        if where_pos != -1:
            insert_pos = sql.upper().rfind("ORDER BY")
            fixed_sql = (
                sql[:insert_pos]
                + f'AND {col_ref} != \'\' AND {col_ref} IS NOT NULL '
                + sql[insert_pos:]
            )
        else:
            insert_pos = sql.upper().rfind("ORDER BY")
            fixed_sql = (
                sql[:insert_pos]
                + f'WHERE {col_ref} != \'\' AND {col_ref} IS NOT NULL '
                + sql[insert_pos:]
            )
        cursor2 = conn.execute(fixed_sql)
        fixed_row = cursor2.fetchone()
        conn.close()
        if fixed_row and first_row and fixed_row != first_row:
            return (
                f"ORDER BY includes empty/NULL values that sort first. "
                f"Add: {col_ref} != '' AND {col_ref} IS NOT NULL "
                f"to your WHERE clause."
            )
    except Exception:
        pass
    return None


def _check_cross_join(sql: str) -> str | None:
    """Detect cross-joins with COUNT that produce wrong ratios."""
    upper = sql.upper()
    if "COUNT(" not in upper and "SUM(" not in upper:
        return None
    # Find outer FROM: first FROM at depth 0
    depth = 0
    from_idx = -1
    for i in range(len(upper)):
        if upper[i] == "(":
            depth += 1
        elif upper[i] == ")":
            depth -= 1
        elif depth == 0 and upper[i:i + 5] == "FROM ":
            from_idx = i
            break
    if from_idx < 0:
        return None
    # Find outer WHERE at depth 0 after FROM
    where_idx = -1
    depth = 0
    for i in range(from_idx + 5, len(upper)):
        if upper[i] == "(":
            depth += 1
        elif upper[i] == ")":
            depth -= 1
        elif depth == 0 and upper[i:i + 7] == " WHERE ":
            where_idx = i
            break
    if where_idx < 0:
        return None
    from_clause = sql[from_idx + 4:where_idx].strip()
    # Skip if FROM contains a subquery or JOIN
    if "(" in from_clause or "JOIN" in from_clause.upper():
        return None
    tables = [t.strip().strip('"').split()[0].strip('"') for t in from_clause.split(",")]
    if len(tables) < 2:
        return None
    return (
        f"CROSS-JOIN DETECTED: FROM {', '.join(tables)} creates a cartesian "
        f"product making COUNT/SUM incorrect. For ratios, use separate "
        f"subqueries: SELECT (SELECT COUNT(DISTINCT ...) FROM A WHERE ...) * 1.0 "
        f"/ (SELECT COUNT(DISTINCT ...) FROM B WHERE ...)"
    )






def _check_fk_resolution(
    result: dict, columns: list[str], kg: "KnowledgeGraph", db_path: Path,
    knowledge_text: str,
) -> str | None:
    """Reject answer if it returns raw FK values that exist in knowledge docs.

    Detects when an answer column contains reference IDs (e.g. Airtable recXXX)
    that point to entities described in doc files rather than SQL tables.
    Uses two signals: column naming (link_to_X) and value presence in knowledge.
    """
    import sqlite3

    if not result.get("rows") or not columns or not knowledge_text:
        return None

    # Get all SQL tables
    try:
        conn = sqlite3.connect(str(db_path))
        sql_tables = {
            r[0].lower()
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
    except Exception:
        return None

    # Collect FK columns that reference non-SQL tables (by KG metadata or naming)
    fk_cols: set[str] = set()
    for _, fk in kg.all_foreign_keys():
        if fk.ref_table.lower() not in sql_tables:
            fk_cols.add(fk.column.lower())

    # Also detect "link_to_X" naming convention
    for t in kg.tables:
        for c in t.columns:
            if c.name.lower().startswith("link_to_"):
                ref = c.name[8:].lower()
                if ref not in sql_tables:
                    fk_cols.add(c.name.lower())

    # Check each answer column: if it's a known FK col OR its values appear
    # verbatim in knowledge_text (indicating they're doc-resolvable IDs)
    rows = result["rows"]
    for col_idx, col_name in enumerate(columns):
        # Skip numeric-looking columns
        sample_val = str(rows[0][col_idx]) if rows[0][col_idx] else ""
        if not sample_val or len(sample_val) < 6:
            continue
        if sample_val.replace(".", "").replace("-", "").isdigit():
            continue
        if " " in sample_val:
            continue

        # Signal 1: column is a known FK to a doc-only table
        is_fk = col_name.lower() in fk_cols

        # Signal 2: the value appears in knowledge docs (it's a resolvable ref)
        in_docs = sample_val in knowledge_text

        if is_fk and in_docs:
            return (
                f"FK RESOLUTION NEEDED: Column \"{col_name}\" contains reference "
                f"IDs (e.g. \"{sample_val}\") that map to names in the source "
                f"documents. Use the knowledge tool to search for "
                f"\"{sample_val}\" and find the human-readable name, then return "
                f"that name instead of the raw ID."
            )

    return None


def _check_string_literal_columns(result: dict, columns: list[str], db_path: Path) -> str | None:
    """Detect when SQLite's double-quote fallback returns a column name as a string.

    SQLite treats "nonexistent_col" as the string literal 'nonexistent_col' instead
    of raising an error. Detect this by checking if any result value equals the
    column alias stripped of quotes.
    """
    rows = result.get("rows", [])
    if not rows:
        return None

    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [r[0] for r in cursor.fetchall()]
        all_columns: set[str] = set()
        for tbl in all_tables:
            for col_info in conn.execute(f'PRAGMA table_info("{tbl}")').fetchall():
                all_columns.add(col_info[1].lower())
        conn.close()
    except Exception:
        return None

    for col_idx, col_name in enumerate(columns):
        # Strip wrapping quotes from the column alias
        clean_name = col_name.strip('"').strip("'").strip("`")
        # Check if this "column" doesn't actually exist in any table
        if clean_name.lower() in all_columns:
            continue
        # Check if the returned value IS the column name (string literal fallback)
        first_val = str(rows[0][col_idx]) if rows[0][col_idx] is not None else ""
        if first_val == clean_name:
            return (
                f"INVALID COLUMN: \"{clean_name}\" does not exist in any table. "
                f"SQLite returned it as a string literal, not actual data. "
                f"Check the schema to find the correct column name."
            )

    return None


def _check_null_ratio(result: dict, cols: list[str]) -> str | None:
    """Warn if answer has columns with very high NULL/empty ratio."""
    rows = result.get("rows", [])
    if not rows or len(rows) < 3:
        return None
    warnings = []
    for i, col in enumerate(cols):
        empty_count = sum(
            1 for r in rows if r[i] is None or r[i] == "" or r[i] == 0
        )
        ratio = empty_count / len(rows)
        if ratio > 0.8 and len(rows) > 5:
            warnings.append(
                f"Column '{col}' is {ratio:.0%} empty/NULL — you may be "
                f"selecting from wrong table or missing a JOIN condition."
            )
    return " ".join(warnings) if warnings else None


def _zero_row_hint(sql: str, db_path: Path, kg=None) -> str:
    """When SQL returns 0 rows and has a text = filter, show actual values
    from the same query context and detect prefix relationships."""
    upper = sql.upper()
    if "WHERE" not in upper:
        return ""

    # Extract text literals from = 'value' patterns
    filters: list[tuple[str, str]] = []
    where_start = upper.index("WHERE")
    where_part = sql[where_start:]
    i = 0
    while i < len(where_part):
        if where_part[i] == "'" and i > 2:
            look_back = where_part[:i].rstrip()
            if look_back.endswith("="):
                end = where_part.find("'", i + 1)
                if end != -1:
                    value = where_part[i + 1 : end]
                    col_part = look_back[:-1].rstrip()
                    col_name = col_part.split()[-1].strip('"').strip("'").strip("`")
                    filters.append((col_name, value))
                    i = end + 1
                    continue
        i += 1

    if not filters:
        return ""

    hints: list[str] = []
    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]

        for col_name, value in filters:
            for tbl in tables:
                try:
                    # First check if exact value exists in the column
                    cursor = conn.execute(
                        f'SELECT 1 FROM "{tbl}" WHERE "{col_name}" = ? LIMIT 1',
                        (value,),
                    )
                    if cursor.fetchone():
                        # Exact value exists — 0 rows caused by another filter
                        continue

                    # Build candidate prefixes from the value
                    # "0:01:54" → ["0:01:54", "01:54", "1:54", "54"]
                    candidates: list[str] = [value]
                    for sep in ":.-/":
                        if sep in value:
                            parts = value.split(sep)
                            for start in range(1, len(parts)):
                                sub = sep.join(parts[start:])
                                candidates.append(sub)
                                stripped = sep.join(
                                    p.lstrip("0") or "0" for p in parts[start:]
                                )
                                if stripped != sub:
                                    candidates.append(stripped)

                    # Try each candidate as a LIKE prefix
                    for prefix in candidates:
                        if len(prefix) < 2:
                            continue
                        cursor = conn.execute(
                            f'SELECT DISTINCT "{col_name}" FROM "{tbl}" '
                            f'WHERE "{col_name}" LIKE ? LIMIT 10',
                            (f"{prefix}%",),
                        )
                        matches = [str(r[0]) for r in cursor.fetchall() if r[0]]
                        if matches:
                            # Found matches — explain the relationship
                            extra = [
                                m[len(prefix):] for m in matches
                                if len(m) > len(prefix)
                            ]
                            if extra:
                                # Values have MORE precision than prefix
                                # Find exact case-insensitive match if one exists
                                exact_ci = [
                                    m for m in matches
                                    if m.lower() == value.lower()
                                ]
                                if exact_ci:
                                    hints.append(
                                        f"0 ROWS: '{value}' is a case mismatch. "
                                        f"The data stores it as '{exact_ci[0]}'. "
                                        f"Use = '{exact_ci[0]}' (exact match)."
                                    )
                                else:
                                    hints.append(
                                        f"0 ROWS: '{value}' is not stored exactly. "
                                        f"Actual values in \"{col_name}\": "
                                        f"{matches[:5]}. "
                                        f"Pick the one that best matches the "
                                        f"question — use exact = not LIKE."
                                    )
                            elif prefix != value:
                                hints.append(
                                    f"0 ROWS: '{value}' not found, but "
                                    f"'{prefix}' matches: {matches[:5]}. "
                                    f"Try filtering with '{prefix}' or "
                                    f"LIKE '{prefix}%'."
                                )
                            break
                    else:
                        # No prefix match — show samples from context
                        cursor = conn.execute(
                            f'SELECT DISTINCT "{col_name}" FROM "{tbl}" '
                            f'WHERE "{col_name}" IS NOT NULL '
                            f'AND "{col_name}" != \'\' LIMIT 8'
                        )
                        samples = [str(r[0]) for r in cursor.fetchall()]
                        if samples and value not in samples:
                            hints.append(
                                f"FORMAT MISMATCH: '{value}' not found in "
                                f"\"{col_name}\". Actual values: {samples}."
                            )
                    if hints:
                        break
                except sqlite3.OperationalError:
                    continue
        conn.close()
    except Exception:
        pass

    # Build join suggestions from KG
    join_hint = ""
    if kg and filters:
        try:
            all_fks = kg.all_foreign_keys()
            # Find which table was queried
            from_idx = sql.upper().find("FROM")
            if from_idx >= 0:
                from_part = sql[from_idx + 4:sql.upper().find("WHERE")].strip()
                queried_table = from_part.strip().strip('"').split()[0].strip('"')
                # Find tables joined TO/FROM this table
                related = set()
                for src, fk in all_fks:
                    if src == queried_table:
                        related.add(fk.ref_table)
                    elif fk.ref_table == queried_table:
                        related.add(src)
                # Find text columns in related tables
                for rel_name in related:
                    rel_table = kg.get_table(rel_name)
                    if rel_table:
                        text_cols = [
                            c.name for c in rel_table.columns
                            if c.sql_type in ("TEXT", "VARCHAR", "NVARCHAR")
                            and "name" in c.name.lower()
                        ]
                        if text_cols:
                            join_hint += (
                                f'\nTry checking "{rel_name}" table '
                                f"(columns: {text_cols}) via JOIN — "
                                f"the value may be stored there instead."
                            )
        except Exception:
            pass

    if not hints:
        return (
            "0 rows returned. The exact value may not exist. "
            "Check actual stored values with SELECT DISTINCT on the "
            "filtered column."
            + join_hint
        )
    return "\n".join(hints) + join_hint


def _ambiguity_key(amb: dict[str, Any]) -> str:
    """Normalize ambiguity key so table order doesn't matter.

    For column_collision: extract column name + sorted table names.
    Fallback: sorted words of the description.
    """
    desc = amb.get("description", "")
    amb_type = amb.get("type", "")
    if amb_type == "column_collision":
        words = desc.split('"')
        col = words[1] if len(words) > 1 else ""
        tables_in_desc = sorted(w for w in words[1::2] if w != col)
        return f"column_collision:{col}:{tables_in_desc}"
    return f"{amb_type}:{' '.join(sorted(desc.lower().split()))}"


def _record_vacuous(memory: "AgentMemory", msg: str) -> None:
    """Extract the column from a vacuous filter message and record it."""
    # msg starts with: FILTER CHECK: "T1.number < 20" has no effect...
    if '"' in msg:
        cond = msg.split('"')[1]
        col_ref = cond.split()[0] if cond else ""
        col_name = col_ref.split(".")[-1].strip('"').strip("`")
        if col_name:
            memory.add_fact(f"_.{col_name}", f"VACUOUS: {cond} has no effect")
    # Extract effective columns if mentioned
    if "actually filters rows:" in msg:
        after = msg.split("actually filters rows:")[1].strip()
        memory.add_fact("_filter_evidence", after)


def _check_duplicate(action: str, parsed: dict[str, Any], history: set[str]) -> str | None:
    """Returns a warning if this action (or a semantically equivalent one) was already done."""
    if action == "overview":
        key = "overview"
    elif action == "schema":
        key = f"schema:{parsed.get('table', '').lower()}"
    elif action == "ontology":
        key = f"ontology:{parsed.get('column', '').lower()}"
    elif action == "topology":
        tables = sorted(t.lower() for t in parsed.get("tables", []))
        key = f"topology:{tables}"
    elif action == "knowledge":
        key = f"knowledge:{_normalize_query(parsed.get('query', ''))}"
    elif action == "run_sql":
        key = f"sql:{_normalize_sql(parsed.get('sql', ''))}"
    else:
        return None

    if key in history:
        return "You already did this (or equivalent). Try a different approach."
    history.add(key)
    return None


class AgentMemory:
    """Dynamic graph memory for a weak (3B) model.

    Organized around data entities (tables/columns), not temporal events.
    Facts accumulate per-entity and never evict each other.
    Renders a focused subgraph into the system prompt each turn.
    """

    def __init__(self) -> None:
        # Entity graph: keyed by entity name (table or table.column)
        self.nodes: dict[str, list[str]] = {}  # entity → facts
        # Edges: (source, target, relation)
        self.edges: list[tuple[str, str, str]] = []
        # Domain knowledge conditions (authoritative value mappings)
        self.domain_sql: list[str] = []  # max 5
        # Best working query so far
        self.verified_sql: str | None = None
        self.verified_rows: int = 0
        self.verified_cols: list[str] = []
        # Column resolutions for enforcement
        self.column_resolutions: dict[str, str] = {}  # col → table
        # Failed SQL (prevents loops)
        self.failed: list[str] = []  # max 3

    def add_fact(self, entity: str, fact: str) -> None:
        """Add a fact to an entity node. No duplicates per entity."""
        if entity not in self.nodes:
            self.nodes[entity] = []
        if fact not in self.nodes[entity]:
            self.nodes[entity].append(fact)

    def add_edge(self, source: str, target: str, relation: str) -> None:
        edge = (source, target, relation)
        if edge not in self.edges:
            self.edges.append(edge)

    def add_domain_sql(self, conditions: str) -> None:
        if conditions not in self.domain_sql:
            self.domain_sql.append(conditions)
            if len(self.domain_sql) > 5:
                self.domain_sql.pop(0)

    def add_failed(self, sql: str, reason: str) -> None:
        entry = f"{sql} → {reason}"
        self.failed.append(entry)
        if len(self.failed) > 3:
            self.failed.pop(0)

    def set_verified(self, sql: str, rows: int, cols: list[str]) -> None:
        self.verified_sql = sql
        self.verified_rows = rows
        self.verified_cols = cols

    def _pending_conditions(self, intent: str, sql: str) -> str:
        """Find filter conditions in intent not yet present in verified SQL."""
        # Extract FILTER portion from intent
        parts = intent.split("|")
        filter_part = ""
        for p in parts:
            if "FILTER" in p.upper():
                filter_part = p.split(":", 1)[-1].strip()
                break
        if not filter_part:
            return ""
        # Split into individual conditions
        conditions = [c.strip() for c in filter_part.split(",")]
        sql_upper = sql.upper()
        pending = []
        for cond in conditions:
            # Check if key terms from this condition appear in the SQL
            keywords = [
                w for w in cond.split()
                if len(w) > 2 and w.upper() not in (
                    "AND", "THE", "FOR", "FROM", "WHERE"
                )
            ]
            found = any(kw.upper() in sql_upper for kw in keywords if len(kw) > 3)
            if not found:
                pending.append(cond)
        return "; ".join(pending) if pending else ""

    def add_resolution(self, text: str) -> None:
        """Parse resolution JSON and store as edges + column_resolutions."""
        try:
            parsed = json.loads(text)
            mapping = parsed.get("resolved", parsed)
            if isinstance(mapping, dict):
                for col, choice in mapping.items():
                    tbl = choice.split(".")[0] if "." in choice else choice
                    self.column_resolutions[col.lower()] = tbl.lower()
                    self.add_fact(
                        f"{tbl}.{col}",
                        "RESOLVED: use this column (not from other tables)",
                    )
                    self.add_edge(col, tbl, "resolved_to")
        except (json.JSONDecodeError, AttributeError):
            pass

    def render(self) -> str:
        """Render the graph memory as structured text for the system prompt."""
        lines: list[str] = []

        # Intent always at the top — guides all decisions
        intent_facts = self.nodes.get("_intent", [])
        if intent_facts:
            lines.append(f"[Intent] {intent_facts[-1]}")

        # Formula from chain-of-thought — the exact math the SQL must implement
        formula_facts = self.nodes.get("_formula", [])
        if formula_facts:
            lines.append("[FORMULA — your SQL MUST implement this exactly]")
            lines.append(f"  {formula_facts[-1]}")

        # Show pending conditions from intent not yet in verified SQL
        if intent_facts and self.verified_sql:
            pending = self._pending_conditions(intent_facts[-1], self.verified_sql)
            if pending:
                lines.append(f"[PENDING — not yet applied] {pending}")

        if self.domain_sql:
            lines.append("[Domain Mappings — use these exact values]")
            for s in self.domain_sql:
                lines.append(f"  >> WHERE {s}")

        if self.verified_sql:
            lines.append(
                f"[Verified] ({self.verified_rows} rows, "
                f"cols={self.verified_cols}): {self.verified_sql}"
            )

        # Render entity graph grouped by table (skip special _ nodes)
        if self.nodes:
            tables: dict[str, list[tuple[str, list[str]]]] = {}
            for entity, facts in self.nodes.items():
                if entity.startswith("_"):
                    continue
                parts = entity.split(".", 1)
                tbl = parts[0]
                if tbl not in tables:
                    tables[tbl] = []
                tables[tbl].append((entity, facts))

            lines.append("[Graph]")
            for tbl, entities in tables.items():
                # Collect edges from this table
                tbl_edges = [
                    e for e in self.edges
                    if e[0] == tbl or e[0].startswith(f"{tbl}.")
                ]
                edge_strs = [f"{e[0]} —{e[2]}→ {e[1]}" for e in tbl_edges]
                lines.append(f"  {tbl}:")
                for entity, facts in entities:
                    col_part = entity.split(".", 1)[1] if "." in entity else ""
                    if col_part:
                        lines.append(f"    .{col_part}:")
                        for f in facts[-3:]:  # last 3 facts per column
                            lines.append(f"      {f}")
                    else:
                        for f in facts[-3:]:
                            lines.append(f"    {f}")
                if edge_strs:
                    for e in edge_strs[:3]:
                        lines.append(f"    → {e}")

        if self.failed:
            lines.append("[Failed — do NOT repeat]")
            for f in self.failed:
                lines.append(f"  ✗ {f}")

        return "\n".join(lines)


class DataAgent:
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
        self._start_time: float = 0.0
        self._question: str = ""

    def run(self, task: PublicTask) -> AgentRunResult:
        self._start_time = time.monotonic()
        self.steps = []
        self._log_file: Path | None = None

        try:
            log_path = task.context_dir / "_agent.log"
            log_path.write_text(f"=== {task.task_id} ===\nQ: {task.question}\n\n")
            self._log_file = log_path
        except OSError:
            pass

        self._question = task.question

        try:
            # Phase 1: Context setup
            db_path, kg, kg_query, knowledge_text, doc_text = self._setup_context(task)
            if not db_path:
                return self._fail(task, "Failed to consolidate data to SQLite.")

            # Phase 2: Agent loop
            result = self._agent_loop(
                task.question, db_path, kg, kg_query, knowledge_text, doc_text
            )
            if not result or not result.get("rows"):
                return self._fail(task, "Agent loop failed or returned no data.")

            # Format answer
            answer = self._format_answer(result)
            return self._build_result(task, answer)

        except Exception as e:
            self._log("fatal_error", str(e))
            return self._fail(task, f"Unhandled error: {e}")

    # ------------------------------------------------------------------
    # Phase 1: Context Setup
    # ------------------------------------------------------------------

    def _setup_context(
        self, task: PublicTask
    ) -> tuple[Path | None, KnowledgeGraph | None, KGQueryService | None, str, str]:
        ctx = scan_context(task.context_dir)
        self._log(
            "scan",
            f"{ctx.task_type}, {len(ctx.structured_sources)} structured, "
            f"{len(ctx.doc_sources)} docs",
        )

        db_path = consolidate_to_sqlite(task.context_dir)
        if not db_path or not db_path.exists():
            return None, None, None, "", ""

        # Build KG from structured tables
        kg = build_kg_from_sqlite(db_path)
        self._log("kg_built", f"{len(kg.tables)} tables, {len(kg.all_foreign_keys())} FKs")

        # Load raw doc text separately for execute_python
        doc_text_full = ""
        if ctx.doc_sources:
            doc_names = []
            for doc_src in ctx.doc_sources:
                try:
                    doc_text = doc_src.path.read_text(errors="replace")
                    doc_text_full += f"\n\n## Document: {doc_src.path.stem}\n{doc_text}"
                    doc_names.append(doc_src.path.stem)
                    self._log("doc_loaded", f"{doc_src.path.stem}: {len(doc_text)} chars")
                except OSError:
                    pass
            kg.doc_names = doc_names

        try:
            kg = discover_joins_with_llm(kg, model=self.model, log_fn=self._log)
        except Exception as e:
            self._log("join_discovery_error", str(e))

        profile_schema(kg)

        # Pre-compute alias registry and business topology
        try:
            build_alias_registry(kg, self.model, db_path, log_fn=self._log)
        except Exception as e:
            self._log("alias_registry_error", str(e))
        try:
            build_business_topology(kg, self.model, db_path, log_fn=self._log)
        except Exception as e:
            self._log("business_topology_error", str(e))

        kg_query = KGQueryService(kg)
        # knowledge_text = domain formulas/definitions only (from knowledge.md)
        # doc_text_full = raw doc content for execute_python
        # Combine both for tool_knowledge (it searches all)
        combined_knowledge = ctx.knowledge_text + doc_text_full
        return db_path, kg, kg_query, combined_knowledge, doc_text_full

    @staticmethod
    def _get_existing_tables(db_path: Path) -> list[str]:
        import sqlite3 as _sql
        try:
            conn = _sql.connect(str(db_path))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            conn.close()
            return tables
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Phase 2: Agent Loop
    # ------------------------------------------------------------------

    def _agent_loop(
        self,
        question: str,
        db_path: Path,
        kg: KnowledgeGraph,
        kg_query: KGQueryService,
        knowledge_text: str,
        doc_text: str = "",
    ) -> dict[str, Any] | None:
        max_turns = 200
        max_time = 300
        max_history = 6  # keep last N assistant+user message pairs
        final_result = None
        resolved_ambiguities: set[str] = set()
        action_history: set[str] = set()
        consecutive_blocks = 0
        evidence: list[str] = []
        memory = AgentMemory()
        approaches = ApproachGraph()
        consult_agent = ConsultAgent(self.model, question)
        # Conversation buffer (assistant/user pairs, pruned to max_history)
        conv_buffer: list[ModelMessage] = []
        reflect_count = 0  # limit reflection rejections to 1
        zero_answer_count = 0  # track 0-row answer attempts
        opaque_id_count = 0  # limit opaque ID rejections to 1

        # Intent extraction: understand what the question structurally requires
        # Provide table/column context so the model picks real column names
        schema_hint = self._schema_hint(kg, question)
        intent = self._extract_intent(question, schema_hint)
        if intent:
            self._log("intent", intent)
            memory.add_fact("_intent", intent)
            self._intent_text = intent
        else:
            self._intent_text = ""

        # Chain-of-thought: decompose question into exact math formula
        # This runs once before the loop and guides all SQL generation
        all_fks = kg.all_foreign_keys()
        join_hint = "\n".join(
            f"  {src_table}.{fk.column} -> {fk.ref_table}.{fk.ref_column}"
            for src_table, fk in all_fks
        ) if all_fks else ""
        formula = self._decompose_formula(
            question, intent or "", schema_hint, join_hint
        )
        if formula:
            self._log("cot_formula", formula)
            memory.add_fact("_formula", formula)
            consult_agent.set_formula(formula)

        for turn in range(max_turns):
            if self._elapsed() > max_time:
                self._log("agent_timeout", f"at turn {turn}, {self._elapsed():.0f}s elapsed")
                break

            # Proactive memory agent consultation every 10 turns
            if turn > 0 and turn % 10 == 0 and approaches.nodes:
                guidance = consult_agent.consult(
                    approaches,
                    schema_context=self._schema_hint(kg, question),
                    trigger="proactive",
                )
                if guidance:
                    self._log("consult_agent_proactive", guidance)

            # Build messages fresh each turn: system + graph memory + question + conv
            messages = self._build_messages(
                question, memory, conv_buffer, max_history, approaches,
                consult_agent=consult_agent,
            )

            raw = self._call_llm(messages)
            if not raw:
                self._log("agent_empty", f"turn {turn}")
                break

            if len(raw) > 2000:
                self._log("long_response", f"{len(raw)} chars")

            parsed = self._parse_json(raw)
            if not parsed:
                self._log("parse_failed", f"no JSON found, len={len(raw)}")
                sql = self._extract_sql(raw)
                if sql:
                    self._log("fallback_sql", sql)
                    obs, result = tool_run_sql(db_path, sql)
                    if result and result.get("rows"):
                        self._log("fallback_result", f"{len(result['rows'])} rows")
                        evidence.append(f"SQL: {sql}\nResult:\n{obs}")
                    else:
                        self._log("fallback_failed", obs)
                    conv_buffer.append(ModelMessage(role="assistant", content=raw))
                    conv_buffer.append(ModelMessage(role="user", content=obs))
                    continue
                # No JSON, no SQL — strong directive
                nudge = "STOP. Respond with exactly ONE JSON object.\n"
                if memory.verified_sql:
                    escaped = memory.verified_sql.replace('"', '\\"')
                    nudge += (
                        f"You already verified a working query. Submit it now:\n"
                        f'{{"action": "answer", "sql": "{escaped}"}}'
                    )
                else:
                    nudge += (
                        'Either explore: {"action": "run_sql", "sql": "..."} '
                        'or submit: {"action": "answer", "sql": "..."}'
                    )
                conv_buffer.append(ModelMessage(role="assistant", content=raw))
                conv_buffer.append(ModelMessage(role="user", content=nudge))
                continue

            action = parsed.get("action", "")
            reasoning = parsed.get("reasoning", "")
            self._log("agent_action", f"turn {turn}: {action}")
            if reasoning:
                self._log("reasoning", reasoning)
            self._log("llm_response", raw)

            # --- Dedup check: block before execution ---
            dup_warning = _check_duplicate(action, parsed, action_history)
            if dup_warning:
                consecutive_blocks += 1
                self._log("duplicate_blocked", f"{action} (streak: {consecutive_blocks})")
                if consecutive_blocks >= 3:
                    self._log("loop_break", "3 consecutive blocks, forcing answer")
                    # Reset conv buffer with directive only
                    conv_buffer.clear()
                    conv_buffer.append(
                        ModelMessage(
                            role="user",
                            content=(
                                "You are stuck repeating yourself. "
                                "Submit your answer NOW based on what you know."
                            ),
                        )
                    )
                    consecutive_blocks = 0
                    action_history.clear()
                    continue
                observation = (
                    f"BLOCKED: {dup_warning} "
                    f"Move forward — submit your answer or try a DIFFERENT query."
                )
                conv_buffer.append(ModelMessage(role="assistant", content=raw))
                conv_buffer.append(ModelMessage(role="user", content=observation))
                continue
            consecutive_blocks = 0

            # --- Execute the requested tool ---
            table = parsed.get("table", "")
            tables = parsed.get("tables", [])
            sql = parsed.get("sql", "")

            if action == "overview":
                observation = tool_overview(kg, question)
                # Extract table names and add to graph
                for t in kg.tables:
                    memory.add_fact(t.name, f"{t.role}, {t.row_count} rows")
                for src, fk_list in [(t.name, t.foreign_keys) for t in kg.tables]:
                    for fk in fk_list:
                        memory.add_edge(
                            f"{src}.{fk.column}",
                            f"{fk.ref_table}.{fk.ref_column}",
                            "joins",
                        )
            elif action == "ontology":
                col_arg = parsed.get("column", table)
                self._log("tool_input", f"ontology({col_arg})")
                observation = tool_ontology(kg, col_arg)
                memory.add_fact(col_arg, "ontology inspected")
            elif action == "schema":
                self._log("tool_input", f"schema({table})")
                observation = tool_schema(kg, table, question, knowledge_text)
                memory.add_fact(table, "schema inspected")
                # Add column facts from schema
                ts = kg.get_table(table)
                if ts:
                    for col in ts.columns:
                        if col.sql_type.upper() in (
                            "INTEGER", "REAL", "INT", "NUMERIC",
                        ):
                            stats = ts.col_stats.get(col.name, {})
                            n_uniq = stats.get("n_unique", "?")
                            memory.add_fact(
                                f"{table}.{col.name}",
                                f"{col.sql_type}, {n_uniq} unique",
                            )
                # Extract WHERE conditions and ORDER BY from domain knowledge SQL
                in_dk = False
                for line in observation.split("\n"):
                    if "DOMAIN KNOWLEDGE" in line:
                        in_dk = True
                        continue
                    if not in_dk:
                        continue
                    conditions = _extract_where_conditions(line)
                    if conditions:
                        memory.add_domain_sql(conditions)
                    order_limit = _extract_order_limit(line)
                    if order_limit:
                        memory.add_domain_sql(order_limit)
            elif action == "topology":
                self._log("tool_input", f"topology({tables})")
                observation = tool_topology(kg, tables, db_path)
                for t in (tables or []):
                    memory.add_fact(t, "topology checked")
            elif action == "knowledge":
                query = parsed.get("query", question)
                self._log("tool_input", f"knowledge({query})")
                observation = tool_knowledge(knowledge_text, query)
                if observation and len(observation) > 10:
                    first_line = observation.split("\n")[0]
                    memory.add_fact("_knowledge", first_line)
            elif action == "distribution":
                dist_table = parsed.get("table", table)
                dist_col = parsed.get("column", "")
                self._log("tool_input", f"distribution({dist_table}, {dist_col})")
                observation = tool_distribution(db_path, dist_table, dist_col)
                memory.add_fact(
                    f"{dist_table}.{dist_col}", "distribution analyzed"
                )
            elif action == "find_value":
                value = parsed.get("value", "")
                self._log("tool_input", f"find_value({value})")
                observation = tool_find_value(kg, value, db_path)
                memory.add_fact("_find_value", f"{value}: {observation}")
            elif action == "resolve":
                term = parsed.get("term", "")
                self._log("tool_input", f"resolve({term})")
                observation = tool_resolve(kg, term, db_path)
                memory.add_fact("_resolve", f"{term}: {observation}")
            elif action == "run_sql":
                if not sql:
                    self._log("tool_error", "run_sql called with empty sql")
                    break
                # Hard gate: block duplicate approaches
                dup_node = approaches.is_duplicate(sql)
                if dup_node:
                    self._log("approach_blocked", f"duplicate of turn {dup_node.turn}")
                    fp = dup_node.fingerprint
                    similar = approaches.similar_to(sql)
                    observation = (
                        f"BLOCKED: You already tried this exact structural approach "
                        f"on turn {dup_node.turn}.\n"
                        f"  Tables: {fp.tables}\n"
                        f"  Joins: {fp.joins}\n"
                        f"  Filters: {fp.filter_cols} = {fp.filter_values}\n"
                        f"  Aggregations: {fp.aggs}\n"
                        f"  Grain: {fp.grain}\n"
                        f"  Result was: {dup_node.result} — {dup_node.reason}\n\n"
                        f"You MUST change your strategy. Options:\n"
                        f"  - Use different tables or join paths\n"
                        f"  - Use different filter columns\n"
                        f"  - Use a different aggregation or grain\n"
                        f"  - Add/remove subqueries\n"
                        f"  - Change GROUP BY structure"
                    )
                    if similar:
                        observation += "\n\nAll related failures:"
                        for s in similar:
                            observation += (
                                f"\n  ✗ Turn {s.turn}: {s.sql}\n"
                                f"    → {s.result}: {s.reason}"
                            )
                    # Consult memory agent for strategic guidance
                    guidance = consult_agent.consult(
                        approaches,
                        schema_context=self._schema_hint(kg, question),
                        trigger="blocked",
                    )
                    if guidance:
                        self._log("consult_agent_guidance", guidance)
                        observation += f"\n\n[Strategy Advisor]: {guidance}"
                    conv_buffer.append(ModelMessage(role="assistant", content=raw))
                    conv_buffer.append(ModelMessage(role="user", content=observation))
                    continue
                self._log("tool_input", f"run_sql: {sql}")
                observation, run_result = tool_run_sql(db_path, sql)
                if run_result is None:
                    node = approaches.record(sql, "error", observation, turn)
                    consult_agent.observe(node)
                row_count = (
                    len(run_result["rows"])
                    if run_result and run_result.get("rows")
                    else 0
                )
                cols = run_result.get("columns", []) if run_result else []
                # Zero-row hint: show actual values so agent can fix format
                # Also trigger for COUNT(*)/SUM() returning 0 (1 row, value=0)
                is_aggregate_zero = (
                    row_count == 1
                    and run_result
                    and run_result["rows"][0] == [0]
                    and any(
                        kw in sql.upper()
                        for kw in ("COUNT(", "SUM(")
                    )
                )
                if row_count == 0 or is_aggregate_zero:
                    hint = _zero_row_hint(sql, db_path, kg)
                    if hint:
                        observation += f"\n\n{hint}"
                    node = approaches.record(
                        sql, "0 rows", "no matching data", turn
                    )
                    consult_agent.observe(node)
                    memory.add_failed(sql, "0 rows")
                # Check for vacuous filter (wrong column)
                if row_count > 0:
                    vacuous_msg = _check_vacuous_filter(sql, db_path)
                    if vacuous_msg:
                        observation += f"\n\n{vacuous_msg}"
                        self._log("vacuous_filter", vacuous_msg)
                        _record_vacuous(memory, vacuous_msg)
                        node = approaches.record(
                            sql, f"{row_count} rows", "vacuous filter", turn
                        )
                        consult_agent.observe(node)
                # Warn about duplicates during exploration
                if row_count > 1 and run_result:
                    tuples = [tuple(r) for r in run_result["rows"]]
                    n_unique = len(set(tuples))
                    if n_unique < row_count:
                        observation += (
                            f"\n\nWARNING: {row_count - n_unique} duplicate rows "
                            f"detected. Use SELECT DISTINCT in your final answer."
                        )
                # Update verified SQL with latest successful query
                if row_count > 0:
                    memory.set_verified(sql, row_count, cols)
                    memory.add_fact("_query", f"{row_count} rows: {sql}")
                # Hint when SQL results contain IDs found in doc text
                if row_count > 0 and doc_text and run_result:
                    doc_refs = self._find_doc_references(
                        run_result, doc_text
                    )
                    if doc_refs:
                        observation += f"\n\n{doc_refs}"
            elif action == "execute_python":
                code = parsed.get("code", "")
                if not code:
                    self._log("tool_error", "execute_python called with empty code")
                    observation = "ERROR: no code provided"
                elif not memory.verified_sql and kg.tables and doc_text:
                    observation = (
                        "BLOCKED: Use run_sql on the SQL tables first to get "
                        "relevant IDs, then use execute_python to search docs "
                        "for those IDs."
                    )
                else:
                    self._log("tool_input", f"execute_python: {code[:200]}")
                    # Pass combined text so Python has access to everything
                    full_text = knowledge_text + doc_text
                    observation = tool_execute_python(code, full_text, db_path)
                    memory.add_fact("_python", observation[:100])
            elif action == "answer":
                if not sql:
                    self._log("tool_error", "answer called with empty sql")
                    break
                self._log("tool_input", f"answer: {sql}")
                obs, result = tool_run_sql(db_path, sql)
                if result and result.get("rows"):
                    row_count = len(result["rows"])
                    self._log(
                        "answer_result",
                        f"{row_count} rows, cols={result['columns']}",
                    )
                    # Check for duplicate rows
                    rows_as_tuples = [tuple(r) for r in result["rows"]]
                    unique_count = len(set(rows_as_tuples))
                    if unique_count < row_count and "DISTINCT" not in sql.upper():
                        dup_count = row_count - unique_count
                        self._log("answer_duplicates", f"{dup_count} duplicates")
                        observation = (
                            f"DUPLICATES: {row_count} rows but only {unique_count} "
                            f"unique. Add SELECT DISTINCT to remove duplicates."
                        )
                        conv_buffer.append(
                            ModelMessage(role="assistant", content=raw)
                        )
                        conv_buffer.append(
                            ModelMessage(role="user", content=observation)
                        )
                        continue
                    if row_count > 500:
                        self._log("answer_rejected", f"{row_count} rows exceeds limit")
                        observation = (
                            f"REJECTED: Your answer returned {row_count} rows. "
                            f"This is too many — most questions expect at most a few "
                            f"hundred rows. Your JOIN is likely wrong (cartesian product?) "
                            f"or you're missing a filter. Check your JOIN path and WHERE "
                            f"conditions with run_sql first, then submit a corrected answer."
                        )
                    else:
                        # Detect SQLite string-literal fallback
                        literal_msg = _check_string_literal_columns(
                            result, result["columns"], db_path
                        )
                        if literal_msg:
                            self._log("string_literal_col", literal_msg)
                            observation = literal_msg
                        # Check column resolution violations
                        elif (col_error := _check_column_resolution(
                            sql, memory.column_resolutions, db_path
                        )):
                            self._log("resolution_violation", col_error)
                            observation = col_error
                        # Reject if answer introduces unverified filters
                        elif (unverified_msg := _check_unverified_filter(
                            sql, memory.verified_sql, memory.failed
                        )):
                            self._log("unverified_filter", unverified_msg)
                            observation = unverified_msg
                        else:
                            # Reject if a numeric filter is vacuous
                            vacuous_msg = _check_vacuous_filter(sql, db_path)
                            if vacuous_msg:
                                self._log("vacuous_filter_answer", vacuous_msg)
                                observation = (
                                    f"{vacuous_msg} "
                                    f"Do NOT submit with this filter. "
                                    f"Find the correct column first using run_sql."
                                )
                            else:
                                # Check cross-join with COUNT
                                cross_msg = _check_cross_join(sql)
                                if cross_msg:
                                    self._log("cross_join", cross_msg)
                                    observation = cross_msg
                                # Check for empty-string ordering issue
                                elif (empty_msg := _check_empty_order(
                                    sql, result, db_path
                                )):
                                    self._log("empty_order", empty_msg)
                                    observation = empty_msg
                                else:
                                    # Log NULL ratio warning (non-blocking)
                                    null_msg = _check_null_ratio(
                                        result, result["columns"]
                                    )
                                    if null_msg:
                                        self._log("null_ratio_warn", null_msg)
                                    # FK resolution check (max 1 rejection)
                                    fk_msg = None
                                    if opaque_id_count < 1:
                                        fk_msg = _check_fk_resolution(
                                            result, result["columns"],
                                            kg, db_path, knowledge_text,
                                        )
                                    if fk_msg:
                                        opaque_id_count += 1
                                        self._log("fk_resolution_reject", fk_msg)
                                        observation = fk_msg
                                    # Self-reflection (max 1 rejection)
                                    elif reflect_count < 1:
                                        reject = self._reflect(
                                            question, sql, result, memory,
                                        )
                                        if reject:
                                            reflect_count += 1
                                            self._log("reflect_reject", reject)
                                            observation = reject
                                        else:
                                            final_result = result
                                            break
                                    else:
                                        final_result = result
                                        break
                else:
                    zero_answer_count += 1
                    node = approaches.record(
                        sql, "0 rows (answer)", "no data", turn
                    )
                    consult_agent.observe(node)
                    self._log("answer_failed", f"{obs} (attempt {zero_answer_count})")
                    if zero_answer_count >= 2:
                        self._log("answer_accept_fallback", "accepting after 2 zero-row attempts")
                        observation = (
                            "No data matches the exact filter. The data may use "
                            "a different time range or format. Remove the "
                            "problematic filter and answer with available data."
                        )
                        # Disable reflection so it doesn't fight the fallback
                        reflect_count = 1
                    else:
                        observation = f"{obs}\nUse run_sql to investigate, then try again."
            elif action == "recall_approaches":
                recall_tables = parsed.get("tables", [])
                recall_cols = parsed.get("columns", [])
                recall_agg = parsed.get("agg", "")
                recall_grain = parsed.get("grain", "")
                recall_temporal = parsed.get("temporal", "")
                recall_joins = parsed.get("joins", [])
                self._log(
                    "tool_input",
                    f"recall_approaches(tables={recall_tables}, "
                    f"cols={recall_cols}, agg={recall_agg}, "
                    f"grain={recall_grain}, temporal={recall_temporal}, "
                    f"joins={recall_joins})",
                )
                results = approaches.recall(
                    tables=recall_tables or None,
                    columns=recall_cols or None,
                    agg=recall_agg or None,
                    grain=recall_grain or None,
                    temporal=recall_temporal or None,
                    joins=recall_joins or None,
                )
                if results:
                    observation = approaches.render_for_prompt(results)
                else:
                    observation = "No matching past approaches found."
            elif action == "recall_memory":
                entity = parsed.get("entity", "")
                self._log("tool_input", f"recall_memory({entity})")
                if entity:
                    facts = memory.nodes.get(entity, [])
                    if facts:
                        observation = f"[{entity}]\n" + "\n".join(
                            f"  {f}" for f in facts
                        )
                    else:
                        observation = f"No facts stored for '{entity}'."
                else:
                    observation = memory.render()
            else:
                observation = (
                    f"Unknown action '{action}'. Available: overview, schema, "
                    f"ontology, topology, knowledge, distribution, find_value, "
                    f"resolve, run_sql, execute_python, recall_approaches, "
                    f"recall_memory, answer"
                )

            self._log("tool_output", observation)

            # --- Collect evidence for loop-break recovery ---
            if action in ("run_sql", "answer"):
                evidence.append(f"SQL: {sql}\nResult:\n{observation}")
            elif action == "knowledge":
                evidence.append(f"Knowledge:\n{observation}")
            elif action == "schema":
                evidence.append(f"Schema({table}):\n{observation}")
            elif action == "ontology":
                evidence.append(f"Ontology({parsed.get('column', '')}):\n{observation}")

            # --- Refine formula after significant discoveries ---
            if action in ("overview", "schema", "topology") and consult_agent.get_formula():
                refined = consult_agent.refine_formula(
                    f"[{action}] {observation[:500]}"
                )
                if refined:
                    self._log("formula_refined", refined)
                    memory.add_fact("_formula", refined)

            # --- Generic ambiguity detection (runs after every tool) ---
            self._log("ambiguity_check", f"checking after {action}")
            ambiguities = detect_ambiguities(
                action, kg, question, db_path, table=table, tables=tables, sql=sql
            )
            new_ambiguities = [
                a
                for a in ambiguities
                if _ambiguity_key(a) not in resolved_ambiguities
            ]
            if new_ambiguities:
                for amb in new_ambiguities:
                    self._log(
                        "ambiguity_found",
                        f"[{amb['type']}] {amb['description']}",
                    )
                    if amb.get("evidence"):
                        self._log("ambiguity_evidence", amb["evidence"])
                conv_buffer.append(ModelMessage(role="assistant", content=raw))
                conv_buffer.append(ModelMessage(role="user", content=observation))
                resolve_prompt = format_resolution_prompt(new_ambiguities, question)
                self._log("resolve_prompt", resolve_prompt)
                conv_buffer.append(ModelMessage(role="user", content=resolve_prompt))
                # Build full messages for resolution call
                resolve_msgs = self._build_messages(
                    question, memory, conv_buffer, max_history, approaches,
                    consult_agent=consult_agent,
                )
                resolve_raw = self._call_llm(resolve_msgs)
                if resolve_raw:
                    self._log("ambiguity_resolved", resolve_raw)
                    conv_buffer.append(
                        ModelMessage(role="assistant", content=resolve_raw)
                    )
                    memory.add_resolution(resolve_raw)
                else:
                    self._log("resolve_failed", "LLM returned empty response")
                for amb in new_ambiguities:
                    resolved_ambiguities.add(_ambiguity_key(amb))
                conv_buffer.append(
                    ModelMessage(
                        role="user",
                        content="Resolved. Continue exploring to answer the question.",
                    )
                )
                continue
            else:
                self._log("ambiguity_check", "none found")

            # Append to conversation buffer (graph memory is in system prompt)
            conv_buffer.append(ModelMessage(role="assistant", content=raw))
            conv_buffer.append(ModelMessage(role="user", content=observation))

        return final_result

    # ------------------------------------------------------------------
    # Intent Extraction
    # ------------------------------------------------------------------

    def _schema_hint(self, kg: KnowledgeGraph, question: str = "") -> str:
        """Build a compact schema overview for intent extraction.

        Marks columns whose name exactly matches a word in the question.
        """
        q_words = set(question.lower().split()) if question else set()
        lines = []
        for t in kg.tables:
            col_strs = []
            for c in t.columns:
                if c.name.lower() in q_words:
                    col_strs.append(f"**{c.name}** ← matches question word")
                else:
                    col_strs.append(c.name)
            lines.append(f"  {t.name}: {', '.join(col_strs)}")
        return "\n".join(lines)

    def _extract_intent(self, question: str, schema_hint: str = "") -> str | None:
        """Extract structured intent from the question before exploration.

        Returns a short string describing: what to SELECT, answer grain,
        and aggregation level.
        """
        intent_prompt = (
            f"QUESTION: {question}\n\n"
            f"AVAILABLE COLUMNS:\n{schema_hint}\n\n"
            f"What does this question want as output? Parse carefully:\n"
            f"- RETURN: the column/value being asked about "
            f"(after 'what is', 'identify the', 'give the', 'list the')\n"
            f"- FILTER: conditions that narrow rows. Distinguish:\n"
            f"  * Per-row (WHERE): column = value, or column > N\n"
            f"  * Per-group (HAVING): ONLY for aggregating raw data at query time\n"
            f"  * If a column name implies it is already aggregated (e.g. starts "
            f"with Avg, Sum, Total, Count, Num), treat it as a per-row filter "
            f"(WHERE), NOT HAVING.\n"
            f"- GRAIN: single value | single row | multiple rows\n"
            f"- TEMPORAL: if the question asks for a rate per time period "
            f"(daily/monthly/yearly), note it — you may need to divide\n\n"
            f"Format (one line each):\n"
            f"RETURN: <exact column(s) to output>\n"
            f"FILTER: <conditions>\n"
            f"GRAIN: <single value | single row | multiple rows>\n"
            f"TEMPORAL: <none | divide by N for period conversion>\n\n"
            f"RULES:\n"
            f"- 'total X' or 'total value' → GRAIN is single value (one sum)\n"
            f"- If a word in the question exactly matches a column name in the "
            f"schema, use THAT column (e.g. 'type' → event.type, not category)\n"
            f"- 'average monthly/daily' + data at that grain → divide by periods "
            f"(AVG / 12 for monthly in a year). ALWAYS include division.\n"
            f"- 'per unit' → Price/Amount\n"
            f"- 'X-related Y': filter on the column matching entity Y, not a "
            f"broader geographic column. The qualifier Y tells you WHERE to look.\n\n"
            f"Examples:\n"
            f"Q: Identify the gender of the superhero with Phoenix Force\n"
            f"RETURN: gender\nFILTER: ability = Phoenix Force\n"
            f"GRAIN: single row\nTEMPORAL: none\n\n"
            f"Q: What is the average monthly consumption for 2013?\n"
            f"RETURN: AVG(Consumption) / 12\nFILTER: year = 2013\n"
            f"GRAIN: single value\n"
            f"TEMPORAL: monthly data → divide by 12\n\n"
            f"Q: What is the type and total amount of sales for order #5?\n"
            f"RETURN: type, SUM(amount)\nFILTER: order_id = 5\n"
            f"GRAIN: single row ('total' = one aggregated sum)\nTEMPORAL: none\n\n"
            f"Q: People who paid more than 29 per unit of product 5\n"
            f"RETURN: Consumption\nFILTER: ProductID=5, Price/Amount > 29\n"
            f"GRAIN: multiple rows\n"
            f"TEMPORAL: none (but 'per unit' means Price/Amount)\n\n"
            f"Q: List departments where the total salary of employees exceeds 100000\n"
            f"RETURN: department_name\n"
            f"FILTER: GROUP BY department HAVING SUM(salary) > 100000\n"
            f"GRAIN: multiple rows\nTEMPORAL: none\n\n"
            f"Now parse the question. Be concise."
        )
        msgs = [
            ModelMessage(role="system", content="Parse question intent."),
            ModelMessage(role="user", content=intent_prompt),
        ]
        raw = self._call_llm(msgs)
        if not raw:
            return None
        # Keep it compact for graph memory
        lines = [
            ln.strip() for ln in raw.strip().split("\n")
            if ln.strip() and ":" in ln
        ]
        return " | ".join(lines[:4]) if lines else None

    # ------------------------------------------------------------------
    # Self-Reflection
    # ------------------------------------------------------------------

    def _decompose_formula(
        self, question: str, intent: str, schema_hint: str,
        join_hint: str = "",
    ) -> str | None:
        """Separate LLM call to decompose the question into an exact math formula.

        Runs once before the agent loop. The formula is stored in memory and
        guides all SQL generation throughout the session.
        """
        if not intent:
            return None

        fk_section = ""
        if join_hint:
            fk_section = (
                f"\nDECLARED FOREIGN KEYS (use ONLY these join paths):\n"
                f"{join_hint}\n"
                f"RULE: Use ONLY the FK paths above. Do NOT invent joins on "
                f"other columns (e.g. no OR on atom_id2 if FK is on atom_id).\n"
            )

        cot_prompt = (
            f"QUESTION: {question}\n"
            f"INTENT: {intent}\n"
            f"SCHEMA:\n{schema_hint}\n"
            f"{fk_section}\n"
            f"Decompose this question into an exact mathematical formula.\n"
            f"1. What computation does the question ask for? Write it as:\n"
            f"   FORMULA: <exact math, e.g. AVG(Consumption) / 12>\n"
            f"2. What does each part map to?\n"
            f"   NUMERATOR: <what to count/sum/avg>\n"
            f"   DENOMINATOR: <what to divide by, or NONE>\n"
            f"3. What join path connects the tables?\n"
            f"   JOIN: <use ONLY declared FKs above>\n"
            f"4. Critical constraint:\n"
            f"   CONSTRAINT: <any rule the SQL must follow>\n\n"
            f"Be precise. The agent's SQL MUST implement this formula exactly."
        )

        messages = [
            ModelMessage(
                role="system",
                content="You decompose questions into exact math formulas.",
            ),
            ModelMessage(role="user", content=cot_prompt),
        ]

        self._log("cot_call", "decomposing question into math formula")
        result = self._call_llm(messages)
        if not result:
            return None

        self._log("cot_result", result)
        return result.strip()

    # ------------------------------------------------------------------

    def _reflect(
        self,
        question: str,
        sql: str,
        result: dict[str, Any],
        memory: AgentMemory,
    ) -> str | None:
        """Ask the LLM to verify the answer matches user intent.

        Returns a rejection message if the answer is wrong, None if OK.
        Only runs once per answer attempt to limit LLM calls.
        """
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        if not rows:
            return None

        # Skip reflection for literal SELECTs (knowledge-resolved values)
        if "FROM" not in sql.upper():
            return None

        # Format a preview of the result (first 5 rows)
        preview_rows = rows[:5]
        preview = ", ".join(columns) + "\n"
        for row in preview_rows:
            preview += ", ".join(str(v) for v in row) + "\n"
        if len(rows) > 5:
            preview += f"... ({len(rows)} rows total)\n"

        # Get intent from memory for comparison
        intent_facts = memory.nodes.get("_intent", [])
        intent_line = intent_facts[-1] if intent_facts else "unknown"

        reflect_prompt = (
            f"QUESTION: {question}\n"
            f"INTENT (may be wrong): {intent_line}\n\n"
            f"ANSWER SQL: {sql}\n\n"
            f"RESULT ({len(rows)} rows, columns={columns}):\n{preview}\n"
            f"Check ONLY these things (re-read the QUESTION first):\n"
            f"1. FORMULA: Does the SQL match the INTENT formula?\n"
            f"   - If the INTENT says '/12' or 'divide by N', the SQL MUST "
            f"include that division. Do NOT remove it.\n"
            f"   - If the INTENT says 'Price/Amount', the SQL MUST divide.\n"
            f"   - NEVER reject a division that the INTENT explicitly requires.\n"
            f"2. COLUMNS: Does the SQL SELECT what the question asks for?\n"
            f"3. GRAIN: Single value vs multiple rows?\n\n"
            f"RULES:\n"
            f"- If the SQL matches the INTENT formula, respond OK.\n"
            f"- Do NOT reject based on schema guesses. If the SQL ran "
            f"successfully, assume columns exist.\n"
            f"- Do NOT reject based on filter columns unless the question "
            f"clearly contradicts the filter logic.\n"
            f"- ONLY reject if you are confident the SQL misses what the "
            f"question literally asks.\n\n"
            f"If correct → respond: OK\n"
            f"If wrong → respond in this EXACT format:\n"
            f"PROBLEM: <what is wrong>\n"
            f"FIX: <specific SQL change>"
        )

        msgs = [
            ModelMessage(role="system", content="You are verifying a SQL answer."),
            ModelMessage(role="user", content=reflect_prompt),
        ]
        # Include graph memory for context
        graph_state = memory.render()
        if graph_state:
            msgs[0] = ModelMessage(
                role="system",
                content=f"You are verifying a SQL answer.\n\n"
                f"--- GRAPH MEMORY ---\n{graph_state}",
            )

        raw = self._call_llm(msgs)
        if not raw:
            return None

        self._log("reflect_response", raw)

        # Parse response
        stripped = raw.strip()
        if stripped.upper().startswith("OK"):
            return None

        # Extract structured PROBLEM/FIX
        problem = ""
        fix = ""
        for line in stripped.split("\n"):
            up = line.strip().upper()
            if up.startswith("PROBLEM:"):
                problem = line.strip()[8:].strip()
            elif up.startswith("FIX:"):
                fix = line.strip()[4:].strip()

        if fix:
            msg = f"SELF-CHECK FAILED.\nPROBLEM: {problem}\nFIX: {fix}\n"
            msg += "Apply the FIX above to your SQL and resubmit with run_sql."
            return msg

        # Legacy format fallback
        if "WRONG:" in stripped.upper():
            idx = stripped.upper().index("WRONG:")
            reason = stripped[idx + 6:].strip()
            return f"SELF-CHECK FAILED: {reason}\nFix your query and resubmit."

        # Ambiguous response — accept
        return None

    # ------------------------------------------------------------------
    # Message Construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        question: str,
        memory: AgentMemory,
        conv_buffer: list[ModelMessage],
        max_history: int,
        approaches: ApproachGraph | None = None,
        consult_agent: ConsultAgent | None = None,
    ) -> list[ModelMessage]:
        """Build message list: system + minimal context, question, pruned conversation.

        Only intent + formula in system prompt. Everything else via tools
        (recall_memory, recall_approaches).
        """
        # Minimal context: only intent and formula
        intent_facts = memory.nodes.get("_intent", [])
        formula_facts = memory.nodes.get("_formula", [])
        verified = memory.verified_sql

        context_lines: list[str] = []
        if intent_facts:
            context_lines.append(f"[Intent] {intent_facts[-1]}")
        if formula_facts:
            context_lines.append(f"[Formula] {formula_facts[-1]}")
        if verified:
            context_lines.append(
                f"[Verified SQL] ({memory.verified_rows} rows): {verified}"
            )

        system_content = AGENT_SYSTEM
        if context_lines:
            system_content += "\n\n--- CONTEXT ---\n" + "\n".join(context_lines)

        msgs: list[ModelMessage] = [
            ModelMessage(role="system", content=system_content),
            ModelMessage(
                role="user",
                content=f"QUESTION: {question}\n\nRespond with ONE JSON object.",
            ),
        ]

        # Prune conversation to last max_history messages
        pruned = conv_buffer[-max_history * 2 :] if conv_buffer else []
        msgs.extend(pruned)
        return msgs

    # ------------------------------------------------------------------
    # Answer Formatting
    # ------------------------------------------------------------------

    def _format_answer(self, result: dict[str, Any]) -> AnswerTable | None:
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        if not columns or not rows:
            return None
        str_rows = [[str(v) if v is not None else "" for v in row] for row in rows]
        return AnswerTable(columns=columns, rows=str_rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[ModelMessage]) -> str:
        msg_summary = f"{len(messages)} messages, last_role={messages[-1].role}"
        self._log("llm_call", msg_summary)
        try:
            result = self.model.complete(messages)
            self._log("llm_result", f"{len(result)} chars" if result else "empty")
            return result if result else ""
        except RuntimeError as e:
            self._log("llm_error", str(e))
            return ""

    def _extract_sql(self, raw: str | None) -> str | None:
        if not raw:
            return None
        parsed = self._parse_json(raw)
        if parsed and "sql" in parsed:
            return parsed["sql"]
        upper = raw.upper()
        start = -1
        for prefix in ("SELECT ", "SELECT\n", "SELECT\t"):
            idx = upper.find(prefix)
            if idx != -1 and (start == -1 or idx < start):
                start = idx
        if start == -1:
            return None
        rest = raw[start:]
        semi = rest.find(";")
        if semi > 0:
            return rest[:semi].strip()
        return rest.strip()

    def _parse_json(self, raw: str) -> dict[str, Any] | None:
        if not raw:
            return None
        fence_start = raw.find("```")
        if fence_start != -1:
            content_start = raw.find("\n", fence_start)
            if content_start != -1:
                fence_end = raw.find("```", content_start + 1)
                if fence_end != -1:
                    content = raw[content_start + 1 : fence_end].strip()
                    if content.startswith("{"):
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            pass

        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        end = start
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        candidate = raw[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = self._remove_trailing_commas(candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    def _remove_trailing_commas(self, text: str) -> str:
        chars = list(text)
        i = 0
        while i < len(chars):
            if chars[i] == ",":
                j = i + 1
                while j < len(chars) and chars[j] in (" ", "\n", "\r", "\t"):
                    j += 1
                if j < len(chars) and chars[j] in ("}", "]"):
                    chars[i] = " "
            i += 1
        return "".join(chars)


    def _find_doc_references(
        self, result: dict[str, Any], doc_text: str
    ) -> str:
        """Detect when SQL results contain IDs/values that appear in doc text.

        Returns a hint string if cross-referenceable values are found.
        """
        rows = result.get("rows", [])
        if not rows or not doc_text:
            return ""

        # Collect string values from the result that look like reference IDs
        ref_values: list[str] = []
        for row in rows[:10]:
            for val in row:
                s = str(val) if val is not None else ""
                if len(s) < 5 or len(s) > 50:
                    continue
                if s.replace(".", "").replace("-", "").isdigit():
                    continue
                if " " in s and len(s.split()) > 3:
                    continue
                if s in doc_text:
                    ref_values.append(s)

        if not ref_values:
            return ""

        unique_refs = list(dict.fromkeys(ref_values))[:3]
        return (
            f"DOC CROSS-REFERENCE: Values from SQL results appear in the "
            f"source documents: {unique_refs}. "
            f"Use execute_python to look up these IDs in the doc text:\n"
            f"  matches = [l for l in docs.split('\\n') "
            f"if '{unique_refs[0]}' in l]\n"
            f"  print(matches[:10])"
        )

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time

    def _log(self, key: str, value: str) -> None:
        elapsed = self._elapsed()
        entry = {"action": key, "detail": value}
        self.steps.append(entry)
        if self.log_callback:
            self.log_callback(entry)
        line = f"[{elapsed:>6.1f}s] [{key}] {value}"
        print(line, flush=True)
        if self._log_file:
            try:
                with open(self._log_file, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def _fail(self, task: PublicTask, reason: str) -> AgentRunResult:
        self._log("failure", reason)
        return AgentRunResult(
            task_id=task.task_id,
            answer=None,
            steps=[],
            failure_reason=reason,
        )

    def _build_result(self, task: PublicTask, answer: AnswerTable | None) -> AgentRunResult:
        if not answer:
            return self._fail(task, "Failed to format answer.")
        return AgentRunResult(
            task_id=task.task_id,
            answer=answer,
            steps=[],
            failure_reason=None,
        )
