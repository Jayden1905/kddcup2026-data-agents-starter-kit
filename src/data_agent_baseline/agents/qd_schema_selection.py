"""Schema selection and column resolution mixin for QuestionDrivenAgent."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelMessage
from data_agent_baseline.pipeline.kg_builder import KnowledgeGraph, build_kg_from_sqlite
from data_agent_baseline.pipeline.kg_path_planner import QueryNode


class SchemaSelectionMixin:
    """Schema selection and column resolution methods."""

    def _detect_user_intent_only(self, question: str, kg_context: str = "", anchor_text: str = "") -> str:
        """Detect user intent: what to return, how to filter, what operation."""
        schema_section = f"\nDATABASE SCHEMA:\n{kg_context[:2000]}\n" if kg_context else ""
        domain_section = f"\nDOMAIN KNOWLEDGE:\n{anchor_text[:1000]}\n" if anchor_text else ""
        prompt = f"""QUESTION: {question}
{schema_section}{domain_section}
You are decomposing a data question into structured analytical signals. Think step by step about what the question is REALLY asking, then output JSON.

STEP 0 — GROUND TERMS
Before interpreting the question, check if any words match column names or terms defined in DOMAIN KNOWLEDGE. If a match exists, use the database-defined meaning — not the everyday English meaning.

STEP 1 — UNDERSTAND THE ANSWER
Imagine the final answer printed on paper. What does it look like?
- A single number or value? → answer_shape = "single_value"
- A list of items (names, IDs, dates)? → answer_shape = "list"
- A table with categories and their values? → answer_shape = "grouped_table"

STEP 2 — IDENTIFY THE COMPUTATION
Ask: "Can I answer this by simply reading a stored value, or do I need to CALCULATE something?"

| The answer is...                                              | operation      |
|---------------------------------------------------------------|----------------|
| A value already stored in a column (name, date, status)       | lookup         |
| The result of adding up values across rows                    | sum            |
| The result of averaging values across rows                    | avg            |
| A count of how many rows match a condition                    | count          |
| A list of unique/distinct items (enumerate categories)        | count_distinct |
| A fraction: subset-count divided by total-count × 100         | percentage     |
| A fraction: one measurement divided by another measurement    | ratio          |
| The item that scores highest/lowest on some measure           | rank           |
| The extreme value itself (not which item has it)              | min_max        |

Key distinction: if the question asks "WHICH item has the highest X" → rank (we return the item).
If the question asks "WHAT IS the highest X" → min_max (we return the value itself).

STEP 3 — SEPARATE OUTPUT FROM FILTER
Every question has two parts:
- OUTPUT: what the user wants to SEE in the answer (the columns they'd read)
- FILTER: what constrains WHICH rows to look at (the conditions that narrow the data)

The trick: the entity being filtered is NOT the entity being returned.
- "What courses does Professor Smith teach?" → OUTPUT = course names, FILTER = Professor Smith
- "Which department has the most employees?" → OUTPUT = department name, FILTER = none (all departments), but rank by employee count

entity_of_interest = the table/entity whose attributes appear in the OUTPUT.

STEP 4 — DETECT STRUCTURAL PATTERNS

A. GROUP CONDITION — does the question require counting/summing within groups FIRST, then filtering on that aggregate?
   Pattern: "entities that have [aggregate condition] on their related items"
   - "departments with more than 5 employees" → group_condition = "more than 5 employees"
   - "products that sold over 100 units" → group_condition = "sold over 100 units"
   - "authors who published at least 3 books" → group_condition = "published at least 3 books"
   This is fundamentally different from a simple row filter. It requires GROUP BY + HAVING.
   If the threshold applies to individual rows (not a count/sum across rows) → NOT a group condition.

B. TEMPORAL CONSTRAINT — does the question restrict to a time period?
   - Any year, month, date range, season, or relative time reference ("last year", "in 2019", "before Q3")
   - null if no time constraint is mentioned

C. POSITIONAL SELECTOR — does the question ask for the Nth item in some ordering?
   - "the 3rd most expensive", "runner-up", "champion", "last place", "second oldest"
   - This implies ORDER BY + positional selection
   - null if no position is implied

D. SORT DIRECTION — if a superlative or comparison is implied:
   - "highest", "most", "best", "latest", "newest", "largest" → DESC
   - "lowest", "least", "worst", "earliest", "oldest", "smallest" → ASC
   - null if no ordering is needed

STEP 5 — PERCENTAGE / RATIO DECOMPOSITION
For percentage questions, identify TWO populations:
- POPULATION (denominator): the broader group — goes in who_to_filter_on
- SUBSET (numerator): the condition that selects items WITHIN the population — goes in what_to_return

Example: "Among tall employees, what percentage are managers?"
→ who_to_filter_on = "tall employees" (population = denominator)
→ what_to_return = "managers" (subset = numerator)

For ratio questions, identify which TWO measurements are being divided.

STEP 6 — GRAIN
- "one answer for entire dataset" — the question expects a single result (a count, a name, a total)
- "one row per entity" — the answer lists multiple items (all names, all IDs matching a condition)
- "one row per group" — the answer is grouped (per category, per year, per department)

Now return ONLY this JSON (no other text):
{{
  "answer_shape": "single_value | list | grouped_table",
  "operation": "lookup | count | count_distinct | sum | avg | ratio | percentage | min_max | rank",
  "grain": "one answer for entire dataset | one row per entity | one row per group",
  "what_to_return": "describe what appears in the output (the thing the user reads)",
  "who_to_filter_on": "the named entity, condition, or population that narrows the data",
  "entity_of_interest": "the database entity whose attributes are RETURNED (not the filter entity)",
  "group_condition": "aggregate condition on groups (e.g. 'more than 5 orders') or null",
  "temporal_filter": "time constraint mentioned in the question, or null",
  "ordinal": "positional selector (1st, 2nd, last, champion, runner-up) or null",
  "sort_direction": "ASC or DESC if ordering is implied, or null"
}}"""

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
                if parsed.get("group_condition"):
                    lines.append(f"Group condition (HAVING): {parsed['group_condition']}")
                if parsed.get("temporal_filter"):
                    lines.append(f"Temporal filter: {parsed['temporal_filter']}")
                if parsed.get("ordinal"):
                    lines.append(f"Ordinal: {parsed['ordinal']}")
                if parsed.get("sort_direction"):
                    lines.append(f"Sort: {parsed['sort_direction']}")
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

