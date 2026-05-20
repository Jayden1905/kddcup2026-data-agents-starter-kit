"""Shared utility methods mixin for QuestionDrivenAgent."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.pipeline.kg_builder import KnowledgeGraph


class UtilsMixin:
    """Shared utility methods for QuestionDrivenAgent."""

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
            err_msg = str(e)
            # Retry with trailing column alias stripped (CAST AS REAL / AS alias ambiguity)
            if conn and "syntax error" in err_msg.lower():
                base = sql.rstrip().rstrip(';')
                fixed = re.sub(r'\s+AS\s+"[^"]*"\s*$', '', base, flags=re.IGNORECASE)
                if fixed == base:
                    fixed = re.sub(r'\s+AS\s+[a-z_]\w*\s*$', '', base, flags=re.IGNORECASE)
                if fixed == base:
                    # Also try stripping alias before trailing ) * N pattern
                    fixed = re.sub(r'\)\s*\*\s*\d+\s+AS\s+\w+\s*$', ') * 100', base, flags=re.IGNORECASE)
                if fixed != base:
                    try:
                        cursor = conn.execute(fixed)
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        rows = cursor.fetchall()
                        conn.close()
                        return {"columns": columns, "rows": [list(r) for r in rows]}
                    except Exception:
                        pass
            self._log("sql_error", f"SQL failed: {e}")
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return None

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

    def _raw_result_to_answer(self, data_result: dict[str, Any]) -> dict[str, Any]:
        """Convert raw SQL result to answer format without LLM call."""
        columns = data_result.get("columns", [])
        rows = data_result.get("rows", [])
        if columns and rows:
            return {"columns": columns, "rows": [[str(v) for v in row] for row in rows]}
        return {}

    def _semantic_validate_result(
        self,
        question: str,
        sql: str,
        data_result: dict[str, Any],
        grounding_context: str,
    ) -> str:
        """Verify SQL result semantically matches the question. Returns diagnosis or empty string."""
        cols = data_result.get("columns", [])
        rows = data_result.get("rows", [])
        if not rows:
            return ""

        preview_lines = [" | ".join(str(c) for c in cols)]
        for row in rows[:5]:
            preview_lines.append(" | ".join(str(v)[:40] for v in row))
        if len(rows) > 5:
            preview_lines.append(f"... ({len(rows)} total rows)")
        result_preview = "\n".join(preview_lines)

        # Extract just USER WANTS + COLUMN WARNING lines from grounding (skip full schema)
        grounding_lines = []
        for line in grounding_context.split("\n"):
            if any(k in line for k in ("USER WANTS:", "COLUMN WARNING:", "COMPUTATION TYPE:", "SELECT COLUMNS")):
                grounding_lines.append(line)
        grounding_compact = "\n".join(grounding_lines) if grounding_lines else grounding_context[:800]

        prompt = f"""QUESTION: {question}
SQL: {sql}
RESULT ({len(rows)} rows):
{result_preview}
CONTEXT: {grounding_compact}

Does the SQL semantically answer the question? Check columns, ordering, aggregation, filters, joins.
Reply EXACTLY: PASS or FAIL: <one sentence fix>
RULES: Only reject for CLEAR semantic errors. Do NOT reject for using a different filter column than context suggests — it may be a better semantic match. Do NOT reject for row count, NULLs, style, or complexity."""

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
                diagnosis = first_line[5:].strip() if len(first_line) > 5 else ""
                if not diagnosis and "\n" in raw:
                    diagnosis = raw.split("\n", 1)[1].strip()[:200]
                return diagnosis or "Semantic mismatch detected"
            if any(kw in raw.lower() for kw in ("correct", "pass", "valid", "matches the question")):
                return ""
            return raw[:200]
        except Exception:
            return ""

    def _model_call_with_retry(self, messages: list[ModelMessage], *, thinking: bool = True) -> str:
        """Call model with timeout. Returns empty string on failure (no retry)."""
        try:
            result = self.model.complete(messages, thinking=thinking)
            return result if result else ""
        except RuntimeError as e:
            self._log("llm_error", f"LLM call failed: {e}")
            return ""

    def _validate_result_stats(
        self, data_result: dict[str, Any], comp_type: str,
        output_nodes: list, kg: KnowledgeGraph | None, db_path: Path | None,
    ) -> str:
        """Layer 5: Deterministic result validation using pre-computed statistics.

        Returns anomaly description string, or empty string if result looks OK.
        """
        if not kg or not data_result:
            return ""
        rows = data_result.get("rows", [])
        cols = data_result.get("columns", [])
        if not rows or not cols:
            return ""

        # Check 1: All-NULL result when we expect data
        if len(rows) == 1 and all(v is None for v in rows[0]):
            if output_nodes and db_path:
                node = output_nodes[0]
                ts = kg.get_table(node.table)
                if ts and ts.row_count > 0:
                    return "Result is NULL — filter likely matched zero rows. Check filter value format."
            return ""

        # Check 2: For single-value aggregations, validate against column stats
        if comp_type in ("avg", "sum", "count") and len(rows) == 1 and len(cols) == 1:
            try:
                val = float(rows[0][0]) if rows[0][0] is not None else None
            except (TypeError, ValueError):
                return ""
            if val is None:
                return ""

            if output_nodes:
                node = output_nodes[0]
                ts = kg.get_table(node.table)
                if ts:
                    stats = ts.col_stats.get(node.column, {})
                    col_min = stats.get("min")
                    col_max = stats.get("max")

                    if comp_type == "avg" and col_min is not None and col_max is not None:
                        if val > col_max * 2 or val < col_min * 0.5:
                            return (
                                f"AVG={val} is outside column range [{col_min}, {col_max}] — "
                                f"likely JOIN duplication. Add DISTINCT or fix JOIN."
                            )

                    if comp_type == "sum" and col_max is not None and ts.row_count:
                        theoretical_max = col_max * ts.row_count
                        if val > theoretical_max * 1.5:
                            return (
                                f"SUM={val} exceeds theoretical max ({col_max}×{ts.row_count}) — "
                                f"likely JOIN duplication."
                            )

        # Check 3: min/max computation returning suspiciously many tied results
        if comp_type == "min_max" and len(rows) > 10:
            # A superlative question ("lowest/highest") should usually return 1-5 results.
            # If we get >10 rows tied at the min/max, the column is probably wrong
            # (e.g., many rows have 0 for a budget column that isn't the right metric).
            if output_nodes and kg:
                main_node = output_nodes[0]
                ts = kg.get_table(main_node.table)
                if ts and ts.row_count > 0 and len(rows) > ts.row_count * 0.3:
                    return (
                        f"min_max query returned {len(rows)} rows ({len(rows)}/{ts.row_count} "
                        f"= {len(rows)/ts.row_count:.0%} of table) — too many ties. "
                        f"The metric column is likely wrong. Look for a more specific "
                        f"cost/amount/value column in a related table."
                    )

        return ""

    def _detect_vacuous_filter(
        self, sql: str, data_result: dict[str, Any],
        db_path: Path | None, kg: KnowledgeGraph | None,
        grounding_context: str,
    ) -> str:
        """Detect filters that don't actually constrain results.

        Returns anomaly description if a filter is vacuous, empty string otherwise.
        Two checks:
        1. A numeric comparison that is always satisfied for the entity being queried
           (e.g. number=0, filter is number < 20 — always true).
        2. Result count equals total table rows (filter didn't reduce the result set).
        """
        if not db_path or not kg or not data_result:
            return ""
        rows = data_result.get("rows", [])
        if not rows:
            return ""

        conn = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)

            # Extract WHERE conditions from SQL
            where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)',
                                    sql, re.IGNORECASE | re.DOTALL)
            if not where_match:
                return ""
            where_clause = where_match.group(1)

            # Find numeric comparisons: column < N, column > N, column <= N, column >= N
            num_conds = re.findall(
                r'(\w+)\.(\w+)\s*([<>]=?)\s*(\d+(?:\.\d+)?)',
                where_clause, re.IGNORECASE,
            )
            # Resolve table aliases from SQL
            alias_map: dict[str, str] = {}
            for m in re.finditer(
                r'\b(?:FROM|JOIN)\s+"?(\w+)"?\s+(?:AS\s+)?(\w+)\b',
                sql, re.IGNORECASE,
            ):
                alias_map[m.group(2).lower()] = m.group(1)

            for alias_or_tbl, col, op, val_str in num_conds:
                tbl = alias_map.get(alias_or_tbl.lower(), alias_or_tbl)
                threshold = float(val_str)

                # Check if this filter is an entity-level attribute (not per-row varying)
                # by testing: does the entity have only one distinct value for this column?
                # Find what the entity filter is (e.g. WHERE name = 'X' AND col < 20)
                entity_filters = re.findall(
                    r"(\w+)\.(\w+)\s*=\s*'([^']+)'",
                    where_clause, re.IGNORECASE,
                )
                if not entity_filters:
                    continue

                # Build a query to check the actual value of the filtered column for this entity
                for e_alias, e_col, e_val in entity_filters:
                    e_tbl = alias_map.get(e_alias.lower(), e_alias)
                    if e_tbl.lower() == tbl.lower():
                        # Same table — check if the value is always satisfying the condition
                        try:
                            check_sql = (
                                f'SELECT DISTINCT "{col}" FROM "{tbl}" '
                                f'WHERE "{e_col}" = ? COLLATE NOCASE'
                            )
                            distinct_vals = conn.execute(check_sql, (e_val,)).fetchall()
                            if len(distinct_vals) == 1 and distinct_vals[0][0] is not None:
                                actual = float(distinct_vals[0][0])
                                satisfied = (
                                    (op == "<" and actual < threshold)
                                    or (op == "<=" and actual <= threshold)
                                    or (op == ">" and actual > threshold)
                                    or (op == ">=" and actual >= threshold)
                                )
                                if satisfied:
                                    return (
                                        f"Filter {tbl}.{col} {op} {val_str} is vacuous — "
                                        f"entity '{e_val}' always has {col}={actual}, which always "
                                        f"satisfies the condition. The question likely refers to a "
                                        f"different column. Check if a related table has a numeric "
                                        f"column that actually varies per row."
                                    )
                        except (sqlite3.Error, ValueError, TypeError):
                            continue

            # Check 2: Result returns nearly all rows from the main output table
            # (indicates the filter didn't actually constrain anything)
            from_match = re.search(r'\bFROM\s+"?(\w+)"?', sql, re.IGNORECASE)
            if from_match and len(rows) > 3:
                main_table = from_match.group(1)
                ts = kg.get_table(main_table)
                if not ts:
                    for t in kg.tables:
                        if t.name.lower() == main_table.lower():
                            ts = t
                            break
                if ts and ts.row_count > 0:
                    # If result count is >= 80% of table rows and question has filtering language
                    ratio = len(rows) / ts.row_count
                    if ratio >= 0.8 and "DISTINCT" in sql.upper():
                        # Check if the grounding expected fewer rows
                        has_filter_hint = bool(re.search(
                            r'(?:less than|greater than|more than|fewer than|under|over|below|above|between)',
                            grounding_context.lower() if grounding_context else "",
                        ))
                        if has_filter_hint:
                            return (
                                f"Result has {len(rows)} rows out of {ts.row_count} total in "
                                f"'{main_table}' ({ratio:.0%}) — the filter appears ineffective. "
                                f"Check whether the filtering column is correct."
                            )

        except sqlite3.Error:
            pass
        finally:
            if conn:
                conn.close()

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

