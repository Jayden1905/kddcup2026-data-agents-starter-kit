"""Diagnostics mixin for QuestionDrivenAgent."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

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
                        diagnostics.append(f"  FIX: This filter excludes all rows. Remove it or use a different column.")
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
            term, table, col, matches_str = m.group(1), m.group(2), m.group(3), m.group(4)
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

