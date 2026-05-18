"""Prompt constants and builders for QuestionDrivenAgent."""

from __future__ import annotations

import re
from typing import Any


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
    ("no_null", "NEVER RETURN NULL: For non-aggregate queries add WHERE col IS NOT NULL. For aggregate functions (AVG, SUM, COUNT, MIN, MAX) do NOT add IS NOT NULL — they already skip NULLs and adding it restricts other columns incorrectly."),
    ("having", "HAVING vs WHERE: 'where the average exceeds N' = GROUP BY + HAVING, not per-row WHERE."),
    ("positional", "PER-GROUP POSITIONAL: 'Nth of each group' = ROW_NUMBER() OVER (PARTITION BY ... ORDER BY CAST(SUBSTR(id_col, INSTR(id_col,'_')+1) AS INTEGER)) when the ordering column has format PREFIX_N (text sort gives wrong order: _10 before _2). Use numeric extraction for the ORDER BY inside ROW_NUMBER."),
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
    parts = [f"QUESTION: {question}\n\nWrite a SQLite query to answer this question."]

    if grounding_context:
        parts.append(f"\n{grounding_context}")

    if gaps:
        parts.append(f"\nPREVIOUS ATTEMPT FAILED:\n{gaps}\nFix the error.")

    parts.append('\nEscape apostrophes with \'\'. Use double-quotes for identifiers. CAST(x AS REAL) for division.')
    parts.append('\nReturn ONLY: {"sql": "SELECT ..."}')

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
- PER-UNIT EXPRESSIONS: "X per unit/per item/per piece" means X divided by quantity column (e.g., Price/Amount). Put the computed expression in filter_conditions as a formula: {{"column": "table.Price/table.Amount", "operator": ">", "value": "29"}}.
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
    def _esc(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")

    return SEMANTIC_GROUNDING_PROMPT.format(
        question=_esc(question),
        kg_context=_esc(kg_trimmed),
        sample_section=_esc(sample_section),
        anchor_section=_esc(anchor_section + ambiguous_section + intent_section + constraints_section),
        previous_attempt=_esc(prev_section),
        knowledge_guidance_section=_esc(kg_guidance_section),
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
    # For ratio computations with filters on different tables, counts are independent —
    # a JOIN would collapse them. Suppress join paths and hint scalar subqueries.
    join_paths = grounding.get("join_paths", [])
    comp_type = grounding.get("computation_type", "")
    known_values = grounding.get("known_values", {})
    _filter_tables = {k.split(".")[0] for k in known_values if "." in k}
    _is_independent_ratio = (
        comp_type == "ratio" and len(_filter_tables) >= 2 and formula
        and ("count" in formula.lower() or "sum" in formula.lower())
    )
    if _is_independent_ratio:
        parts.append(
            "INDEPENDENT COUNTS: The numerator and denominator filter DIFFERENT tables. "
            "Do NOT JOIN them. Use two scalar subqueries: "
            "SELECT (SELECT COUNT(...) FROM tableA WHERE ...) / (SELECT COUNT(...) FROM tableB WHERE ...)"
        )
    elif join_paths:
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



