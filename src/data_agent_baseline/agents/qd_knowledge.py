"""Domain knowledge extraction mixin for QuestionDrivenAgent."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelMessage
from data_agent_baseline.pipeline.kg_builder import KnowledgeGraph
from data_agent_baseline.pipeline.kg_path_planner import QueryNode

from data_agent_baseline.agents.qd_prompts import DOMAIN_ANCHOR_PROMPT


class KnowledgeMixin:
    """Domain knowledge extraction methods."""

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

        # Extract any inline SQL in backticks, paired with nearby context lines
        lines = knowledge_text.split("\n")
        for i, line in enumerate(lines):
            sql_match = re.search(r'`(SELECT\s[^`]+)`', line, re.IGNORECASE)
            if not sql_match:
                continue
            sql = sql_match.group(1).strip()
            # Use preceding non-empty lines as context/title
            context_lines = []
            for j in range(max(0, i - 3), i):
                stripped = re.sub(r'[*#\-`]', '', lines[j]).strip()
                if stripped:
                    context_lines.append(stripped)
            title = context_lines[-1] if context_lines else f"use_case_line_{i}"
            use_cases.append((title, sql, " ".join(context_lines)))

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
        raw = self._model_call_with_retry(messages, thinking=False)
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

    def _extract_domain_formula(self, question: str, anchor_text: str) -> str:
        """Extract domain formula definitions relevant to the question from anchor text.

        Matches bullet-point blocks by word overlap with the question.
        Returns the raw text for the LLM to interpret (handles LaTeX, plain text, etc).
        """
        if not anchor_text:
            return ""
        q_words_raw = set(re.findall(r'\b[a-z]{4,}\b', question.lower()))
        q_words_stemmed = {w.rstrip('s') if len(w) > 4 else w for w in q_words_raw}

        best_formula = ""
        best_score = 0

        # Match any bullet line: "- content" (up to next bullet or blank line)
        for m in re.finditer(
            r'^-\s+(.+?)(?=\n-\s|\n\n|\Z)',
            anchor_text, re.MULTILINE | re.DOTALL,
        ):
            full_line = m.group(1).strip()
            # Must contain something that looks like a formula/calculation
            has_formula = any(ind in full_line for ind in (
                "SUM(", "AVG(", "COUNT(", "DIVIDE(", "MULTIPLY(",
                "/", "×", "÷", "frac", "=",
            ))
            if not has_formula:
                continue

            # Score by word overlap between the label portion and the question
            # Label = text before first formula-like character
            colon_pos = full_line.find(":")
            label = full_line[:colon_pos] if colon_pos > 0 else full_line[:60]

            label_words_raw = set(re.findall(r'\b[a-z]{4,}\b', label.lower()))
            label_words_stemmed = {w.rstrip('s') if len(w) > 4 else w for w in label_words_raw}
            overlap = len(q_words_stemmed & label_words_stemmed)
            if overlap > best_score:
                best_score = overlap
                best_formula = full_line

        if best_score >= 2:
            return best_formula
        return ""

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

    def _formula_to_sql_hint(
        self, formula: str, output_nodes: list[QueryNode], picked: dict,
    ) -> str:
        """Convert a domain formula into an explicit SQL expression hint.

        Extracts post-aggregation arithmetic operations from any formula format
        (LaTeX, plain text, function syntax) and builds a concrete SQL expression.
        """
        # --- Extract post-aggregation arithmetic operations ---
        operations: list[tuple[str, str]] = []  # (operator, operand)

        # LaTeX \frac: find the last brace-group (denominator) — handles nested braces
        frac_pos = formula.find("\\frac")
        if frac_pos == -1:
            frac_pos = formula.find("frac")
        if frac_pos >= 0:
            # Walk past the numerator brace group, then extract denominator
            brace_groups = self._extract_brace_groups(formula[frac_pos:])
            if len(brace_groups) >= 2:
                denom = brace_groups[-1].strip()
                if re.match(r'^[\d.]+$', denom):
                    operations.append(("/", denom))

        # DIVIDE(x, N) or MULTIPLY(x, N) function syntax
        if not operations:
            for m in re.finditer(r'DIVIDE\([^,]+,\s*([\d.]+)\)', formula, re.IGNORECASE):
                operations.append(("/", m.group(1)))
            for m in re.finditer(r'MULTIPLY\([^,]+,\s*([\d.]+)\)', formula, re.IGNORECASE):
                operations.append(("*", m.group(1)))

        # Plain text: "/ N" or "* N" (numeric constant after operator)
        if not operations:
            for m in re.finditer(r'([/*×÷])\s*([\d.]+)', formula):
                op = "/" if m.group(1) in ("/", "÷") else "*"
                operations.append((op, m.group(2)))

        if not operations:
            return ""

        # --- Find the numeric output column ---
        # Prefer the column whose name appears in the formula text
        computation_type = (picked.get("computation_type") or "").lower()
        formula_lower = formula.lower()
        numeric_col = None
        for node in output_nodes:
            if node.role != "output":
                continue
            col_ref = f'"{node.table}"."{node.column}"'
            if node.column.lower() in formula_lower:
                numeric_col = col_ref
                break
            if not numeric_col:
                numeric_col = col_ref

        if not numeric_col:
            return ""

        # --- Build SQL expression: aggregation + operations ---
        agg_map = {"avg": "AVG", "sum": "SUM", "count": "COUNT", "min": "MIN", "max": "MAX"}
        agg = agg_map.get(computation_type, "AVG")
        expr = f"{agg}({numeric_col})"
        for op, operand in operations:
            expr = f"{expr} {op} {operand}"

        return f"Your SELECT must use: {expr}"

    def _extract_brace_groups(text: str) -> list[str]:
        """Extract top-level brace-delimited groups from text. Handles nesting."""
        groups: list[str] = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i + 1
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    groups.append(text[start:i])
                    start = -1
        return groups

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

