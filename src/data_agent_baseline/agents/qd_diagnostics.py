"""Diagnostics mixin for QuestionDrivenAgent."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any  # noqa: F401

from data_agent_baseline.pipeline.kg_builder import KnowledgeGraph


class DiagnosticsMixin:
    """Empty result diagnosis and value discovery methods."""

    def _try_remove_blocker_filter(self, db_path: Path, sql: str) -> str | None:
        """Remove a single redundant WHERE filter that blocks results.

        Only removes a filter when: (1) the full SQL returns 0 rows,
        (2) removing exactly one filter produces exactly 1 row,
        meaning the remaining filters already uniquely identify the entity.
        """
        if not sql or not db_path or not db_path.exists():
            return None
        sql_upper = sql.upper()
        if "WHERE" not in sql_upper:
            return None

        where_idx = sql.upper().find("WHERE")
        base_sql = sql[:where_idx].strip()

        # Find the end of WHERE clause
        where_rest = sql[where_idx + 5:]
        where_clause = where_rest
        suffix = ""
        for keyword in ("ORDER BY", "GROUP BY", "LIMIT", "HAVING"):
            kw_idx = where_clause.upper().find(keyword)
            if kw_idx > 0:
                suffix = where_clause[kw_idx:]
                where_clause = where_clause[:kw_idx]
                break

        conditions = [c.strip() for c in re.split(r'\bAND\b', where_clause, flags=re.IGNORECASE) if c.strip()]
        if len(conditions) < 2:
            return None

        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            best_removal: tuple[int, str] | None = None
            for i, cond in enumerate(conditions):
                remaining = [c for j, c in enumerate(conditions) if j != i]
                test_where = " AND ".join(remaining)
                test_sql = f"{base_sql} WHERE {test_where} {suffix}".strip()
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM ({test_sql})").fetchone()[0]
                    if count == 1:
                        best_removal = (count, test_sql)
                        break
                except Exception:
                    continue
            conn.close()
            if best_removal:
                return best_removal[1]
        except Exception:
            pass
        return None

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
                # No single filter removal helps. Two possibilities:
                # A) JOIN mismatch — filter values exist but JOIN prevents matches
                # B) Pairwise conflict — two filters individually pass but combined produce 0

                # Test pairwise: remove 2 filters at a time to find conflicting pairs
                if len(conditions) >= 3:
                    for i in range(len(conditions)):
                        for j in range(i + 1, len(conditions)):
                            remaining = [c for k, c in enumerate(conditions) if k != i and k != j]
                            test_sql = f"{base_sql} WHERE {' AND '.join(remaining)}" if remaining else base_sql
                            try:
                                count = conn.execute(f"SELECT COUNT(*) FROM ({test_sql})").fetchone()[0]
                                if count > 0:
                                    diagnostics.append(
                                        f"CONFLICTING PAIR: '{conditions[i]}' AND '{conditions[j]}' "
                                        f"together exclude all rows (without both: {count} rows). "
                                        f"These conditions contradict each other — fix or remove one."
                                    )
                            except Exception:
                                pass

                # Test JOIN validity: do filter values exist in individual tables?
                if not diagnostics:
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
                                                f"the JOIN condition is WRONG. Try joining on a different column."
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
                        diagnostics.append("  FIX: This filter excludes all rows. Remove it or use a different column.")
        finally:
            conn.close()

        if diagnostics:
            return "EMPTY RESULT DIAGNOSIS:\n" + "\n".join(diagnostics)
        return ""

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
                # Match quoted "table"."col": or unquoted table.col:
                match = re.match(r'\s*"?(\w+)"?\."?(\w+)"?:\s*(.+)', line)
                if match:
                    col_key = f"{match.group(1)}.{match.group(2)}"
                    raw_val = match.group(3)
                    # Extract values from various formats
                    # "USE → WHERE ..." format: extract the filter value
                    like_m = re.search(r"LIKE\s+'([^']+)'", raw_val)
                    eq_m = re.search(r"=\s*'?([^',\s]+)'?", raw_val) if not like_m else None
                    if like_m:
                        vals = [like_m.group(1).replace('%', '')]
                    elif eq_m:
                        vals = [eq_m.group(1)]
                    else:
                        vals = [v.strip().strip("'\"") for v in raw_val.split(",")]
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

    def _patch_grounding_with_discoveries(self, grounding: str, discoveries: str) -> str:
        """Rewrite FILTER VALUES lines in grounding to use discovered corrections.

        Instead of appending a separate section (which the LLM can ignore),
        patch the original lines with USE → directives.
        """
        if "FILTER VALUES" not in grounding:
            return grounding + f"\n\nDISCOVERED VALUES (use these exact values in WHERE):\n{discoveries}"

        for line in discoveries.split("\n"):
            # Parse: "'term' not found exactly in Table.Col, but LIKE matches: ['actual1', ...]"
            m = re.match(
                r"\s*'([^']+)' not found exactly in (\w+)\.(\w+), but LIKE matches: \[(.+)\]",
                line,
            )
            if not m:
                continue
            _, table, col, matches_str = m.group(1), m.group(2), m.group(3), m.group(4)
            # Extract the first match value
            match_vals = re.findall(r"'([^']+)'", matches_str)
            if not match_vals:
                continue
            best_match = match_vals[0]

            # Find the FILTER VALUES line for this column and rewrite it
            col_pattern = re.compile(
                rf'(\s*"?{re.escape(table)}"?\."?{re.escape(col)}"?:\s*)(.+)',
                re.IGNORECASE,
            )
            new_lines = []
            patched = False
            for gline in grounding.split("\n"):
                if not patched and col_pattern.search(gline):
                    # Rewrite with USE → LIKE directive
                    prefix = col_pattern.search(gline).group(1)
                    new_lines.append(
                        f'{prefix}USE → WHERE "{col}" LIKE \'%{best_match}%\' COLLATE NOCASE'
                    )
                    patched = True
                else:
                    new_lines.append(gline)
            if patched:
                grounding = "\n".join(new_lines)

        return grounding

    def _infer_thresholds(
        self,
        question: str,
        db_path: Path,
        kg: KnowledgeGraph,
        knowledge_text: str,
        grounding_context: str = "",
    ) -> str:
        """Pre-resolve normal/abnormal thresholds via LLM before SQL generation.

        Makes a dedicated LLM call with data distribution + knowledge text to produce
        concrete filter values. Returns explicit CONDITIONS the SQL LLM can use directly.
        """
        q_lower = question.lower()
        needs_threshold = any(w in q_lower for w in (
            "normal", "abnormal", "elevated", "low level", "high level",
            "healthy", "unhealthy", "within range", "out of range",
        ))
        if not needs_threshold:
            return ""

        if not db_path or not db_path.exists():
            return ""

        # Identify columns mentioned in the grounding that need threshold resolution
        # Look for columns referenced in phrase_mapping or data_requirements
        target_cols: list[tuple[str, str]] = []
        if grounding_context:
            for m in re.finditer(r'"?(\w+)"?\."?(\w+)"?', grounding_context):
                table_name, col_name = m.group(1), m.group(2)
                for t in kg.tables:
                    if t.name == table_name:
                        for c in t.columns:
                            if c.name == col_name:
                                target_cols.append((table_name, col_name))
                                break

        # Deduplicate
        target_cols = list(dict.fromkeys(target_cols))

        # Collect stats for target columns only
        resolved_cols: set[str] = set()
        if grounding_context:
            for m in re.finditer(
                r'"([^"]+)"\."([^"]+)":\s*(?:>=|<=|BETWEEN)\s*\d',
                grounding_context,
            ):
                resolved_cols.add(f"{m.group(1)}.{m.group(2)}".lower())

        conn = sqlite3.connect(str(db_path))
        col_stats: list[str] = []
        try:
            for table_name, col_name in target_cols:
                if f"{table_name}.{col_name}".lower() in resolved_cols:
                    continue
                col_lower = col_name.lower()
                if col_lower in ("id", "_id") or col_lower.endswith("_id") or col_lower.startswith("link_to"):
                    continue

                try:
                    stats = conn.execute(
                        f'SELECT MIN(CAST("{col_name}" AS REAL)), '
                        f'MAX(CAST("{col_name}" AS REAL)), '
                        f'AVG(CAST("{col_name}" AS REAL)), '
                        f'COUNT(*) '
                        f'FROM "{table_name}" WHERE "{col_name}" IS NOT NULL'
                    ).fetchone()
                    if not stats or stats[3] == 0 or stats[0] == stats[1]:
                        continue

                    total = stats[3]
                    p5_row = conn.execute(
                        f'SELECT CAST("{col_name}" AS REAL) FROM "{table_name}" '
                        f'WHERE "{col_name}" IS NOT NULL '
                        f'ORDER BY CAST("{col_name}" AS REAL) '
                        f'LIMIT 1 OFFSET {max(int(total * 0.05) - 1, 0)}'
                    ).fetchone()
                    p95_row = conn.execute(
                        f'SELECT CAST("{col_name}" AS REAL) FROM "{table_name}" '
                        f'WHERE "{col_name}" IS NOT NULL '
                        f'ORDER BY CAST("{col_name}" AS REAL) '
                        f'LIMIT 1 OFFSET {max(int(total * 0.95) - 1, 0)}'
                    ).fetchone()
                    p5 = p5_row[0] if p5_row else stats[0]
                    p95 = p95_row[0] if p95_row else stats[1]

                    col_stats.append(
                        f"  {table_name}.{col_name}: min={stats[0]}, max={stats[1]}, "
                        f"avg={stats[2]:.2f}, P5={p5}, P95={p95}, count={total}"
                    )
                except Exception:
                    continue
        finally:
            conn.close()

        if not col_stats:
            return ""

        # Make LLM call to resolve thresholds
        from data_agent_baseline.agents.model import ModelMessage

        prompt = (
            "Given the following question and data columns, determine the concrete numeric "
            "thresholds that define 'normal' vs 'abnormal' for each relevant column.\n\n"
            f"QUESTION: {question}\n\n"
            f"DOMAIN KNOWLEDGE:\n{knowledge_text[:1500] if knowledge_text else 'None provided'}\n\n"
            f"DATA STATISTICS (showing actual scale of values in the database):\n"
            + "\n".join(col_stats) + "\n\n"
            "INSTRUCTIONS:\n"
            "- Use the DOMAIN KNOWLEDGE above to find threshold definitions.\n"
            "- If the knowledge does not define a threshold for a column, use your general "
            "understanding of what the column measures (based on its name and context).\n"
            "- The statistics show the actual data scale. It is possible that ALL values in "
            "the data are abnormal (e.g., the dataset only contains sick patients). Do NOT "
            "adjust your thresholds to force overlap with the data — use the correct domain "
            "thresholds even if no data falls within the normal range.\n"
            "- For each column that needs a threshold, output: table.column: low-high "
            "(meaning normal = BETWEEN low AND high).\n\n"
            "Respond with ONLY the thresholds, one per line, format:\n"
            "table.column: low-high\n"
            "If a column is not relevant to the normal/abnormal question, skip it."
        )

        # Build a lookup of stats per column for validation
        stats_lookup: dict[str, tuple[float, float]] = {}
        for s in col_stats:
            sm = re.match(r'\s*(\w+)\.(\w+): min=([\d.\-]+), max=([\d.\-]+)', s)
            if sm:
                stats_lookup[f"{sm.group(1)}.{sm.group(2)}"] = (
                    float(sm.group(3)), float(sm.group(4)),
                )

        raw = self._model_call_with_retry(
            [ModelMessage(role="user", content=prompt)], thinking=False,
        )
        if not raw:
            return ""

        # Parse and validate thresholds against actual data range
        parsed: list[tuple[str, str, float, float]] = []
        invalid: list[str] = []
        for line in raw.strip().split("\n"):
            m = re.match(r'(\w+)\.(\w+)\s*:\s*([\d.]+)\s*[-–]\s*([\d.]+)', line.strip())
            if m:
                tbl, col = m.group(1), m.group(2)
                low, high = float(m.group(3)), float(m.group(4))
                key = f"{tbl}.{col}"
                if key in stats_lookup:
                    data_min, data_max = stats_lookup[key]
                    no_overlap = low > data_max or high < data_min
                    col_lower = col.lower()
                    is_normal_col = self._col_in_threshold_context(
                        col_lower, "normal", q_lower,
                    )
                    is_abnormal_col = self._col_in_threshold_context(
                        col_lower, "abnormal", q_lower,
                    )
                    if no_overlap and is_normal_col and not is_abnormal_col:
                        # Column is filtered for "normal" but no data is in range → wrong unit
                        invalid.append(
                            f"{key}: you proposed {low}-{high} but data range is "
                            f"{data_min}-{data_max}. Your threshold doesn't overlap "
                            f"with the data. Adjust to fit the data's unit/scale."
                        )
                        continue
                    # For "abnormal" columns, no-overlap is valid (all data is abnormal)
                parsed.append((tbl, col, low, high))

        # If any thresholds were invalid, retry with correction feedback
        if invalid and not parsed:
            correction = (
                prompt + "\n\nCORRECTION NEEDED — your previous answer was wrong:\n"
                + "\n".join(invalid) + "\n"
                "Re-derive the thresholds using the correct unit/scale for this data."
            )
            raw = self._model_call_with_retry(
                [ModelMessage(role="user", content=correction)], thinking=False,
            )
            if raw:
                for line in raw.strip().split("\n"):
                    m = re.match(r'(\w+)\.(\w+)\s*:\s*([\d.]+)\s*[-–]\s*([\d.]+)', line.strip())
                    if m:
                        tbl, col = m.group(1), m.group(2)
                        low, high = float(m.group(3)), float(m.group(4))
                        parsed.append((tbl, col, low, high))

        # Build conditions from validated thresholds
        conditions: list[str] = []
        col_conditions: dict[str, list[str]] = {}  # table -> list of SQL fragments
        for tbl, col, low, high in parsed:
            col_lower = col.lower()
            is_normal = self._col_in_threshold_context(col_lower, "normal", q_lower)
            is_abnormal = self._col_in_threshold_context(col_lower, "abnormal", q_lower)

            if is_normal and not is_abnormal:
                conditions.append(
                    f'  "{tbl}"."{col}": BETWEEN {low} AND {high} (normal range)'
                )
                col_conditions.setdefault(tbl, []).append(
                    f'"{col}" BETWEEN {low} AND {high}'
                )
            elif is_abnormal and not is_normal:
                conditions.append(
                    f'  "{tbl}"."{col}": < {low} OR > {high} (abnormal = outside normal range)'
                )
                col_conditions.setdefault(tbl, []).append(
                    f'("{col}" < {low} OR "{col}" > {high})'
                )
            else:
                conditions.append(
                    f'  "{tbl}"."{col}": normal range is {low}-{high}'
                )

        if not conditions:
            return ""

        result = "RESOLVED THRESHOLDS (use these exact values in your SQL):\n"
        result += "\n".join(conditions)

        # Cross-row detection: if 2+ threshold conditions on same multi-row table,
        # emit subquery pattern since conditions may apply to different rows
        for tbl, frags in col_conditions.items():
            if len(frags) < 2:
                continue
            # Check if this table has multiple rows per entity
            for t in kg.tables:
                if t.name == tbl and t.columns:
                    pk_col = t.columns[0].name
                    try:
                        conn2 = sqlite3.connect(str(db_path))
                        total = conn2.execute(
                            f'SELECT COUNT(*) FROM "{tbl}"'
                        ).fetchone()[0]
                        distinct = conn2.execute(
                            f'SELECT COUNT(DISTINCT "{pk_col}") FROM "{tbl}"'
                        ).fetchone()[0]
                        conn2.close()
                        if distinct > 0 and total / distinct > 1.5:
                            subqs = [
                                f'"{pk_col}" IN (SELECT "{pk_col}" FROM "{tbl}" WHERE {f})'
                                for f in frags
                            ]
                            result += (
                                f"\n\nCROSS-ROW PATTERN (MANDATORY — conditions apply to "
                                f"different rows of the same entity):\n"
                                f"  SELECT COUNT(DISTINCT \"{tbl}\".\"{pk_col}\") FROM \"{tbl}\"\n"
                                f"  WHERE " + "\n    AND ".join(subqs)
                            )
                    except Exception:
                        pass
                    break

        return result

    def _col_in_threshold_context(self, col_lower: str, keyword: str, question: str) -> bool:
        """Check if a column is associated with a threshold keyword in the question."""
        # Common abbreviation expansions for matching
        expansions: dict[str, list[str]] = {
            "wbc": ["white blood cell", "white blood"],
            "rbc": ["red blood cell", "red blood"],
            "hgb": ["hemoglobin"], "hct": ["hematocrit"],
            "plt": ["platelet"], "fg": ["fibrinogen"],
            "ldh": ["lactate dehydrogenase"],
            "alb": ["albumin"], "ua": ["uric acid"],
            "glu": ["glucose"], "crp": ["c-reactive"],
        }
        # Find where the column concept appears in the question
        col_positions: list[int] = []
        if col_lower in question:
            col_positions.append(question.find(col_lower))
        for expansion in expansions.get(col_lower, []):
            if expansion in question:
                col_positions.append(question.find(expansion))

        if not col_positions:
            return False

        # Check if the keyword is near the column mention
        keyword_pos = question.find(keyword)
        if keyword_pos < 0:
            return False

        for cp in col_positions:
            if abs(cp - keyword_pos) < 40:
                return True
        return False

    def _check_concept_coverage(
        self, sql: str, grounding_context: str, question: str,
    ) -> str:
        """Check that all filter conditions from grounding are reflected in the generated SQL.

        Returns a description of missing concepts, or empty string if all covered.
        """
        if not sql or not grounding_context:
            return ""

        # Parse CONDITIONS from grounding
        conditions_section = ""
        in_conditions = False
        for line in grounding_context.split("\n"):
            if line.startswith("CONDITIONS:"):
                in_conditions = True
                continue
            elif in_conditions:
                if line and not line.startswith(" ") and not line.startswith("\t") and not line.startswith('"'):
                    break
                conditions_section += line + "\n"

        if not conditions_section:
            return ""

        # Extract each condition: "table"."column": operator value
        expected_filters: list[tuple[str, str, str]] = []
        for m in re.finditer(
            r'"(\w+)"\."(\w+)":\s*(=|>=|<=|>|<|LIKE|IS NOT NULL)\s*(.*)',
            conditions_section,
        ):
            table, col, _, val = m.group(1), m.group(2), m.group(3), m.group(4).strip()
            val = re.sub(r'\s+COLLATE\s+NOCASE\s*$', '', val, flags=re.IGNORECASE).strip().strip("'\"")
            expected_filters.append((table, col, val))

        if not expected_filters:
            return ""

        # Check which conditions are missing from the SQL
        sql_upper = sql.upper()
        missing: list[str] = []
        for table, col, val in expected_filters:
            # Check if the column appears in the SQL (case-insensitive)
            col_present = (
                f'"{col}"' in sql
                or f'.{col}' in sql.replace('"', '')
                or col.upper() in sql_upper
            )
            if col_present:
                continue
            # Check if the value appears (for text values)
            if val and len(val) >= 2:
                val_present = val in sql or val.lower() in sql.lower()
                if val_present:
                    continue
            # This condition is entirely missing from the SQL
            missing.append(f'"{table}"."{col}" {val} is not filtered in the SQL')

        if missing:
            return (
                "The SQL is missing these required filters from the grounding:\n"
                + "\n".join(f"  - {m}" for m in missing)
                + "\nAdd these conditions to the WHERE clause."
            )
        return ""

