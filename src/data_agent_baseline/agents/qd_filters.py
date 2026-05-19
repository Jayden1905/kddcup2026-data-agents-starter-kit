"""Filter processing mixin for QuestionDrivenAgent."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.pipeline.kg_builder import KnowledgeGraph
from data_agent_baseline.pipeline.kg_path_planner import QueryNode


class FilterProcessingMixin:
    """Filter probing, resolution, and correction methods."""

    def _probe_filter_values(
        self, filter_nodes: list[QueryNode], db_path: Path | None,
        kg: KnowledgeGraph | None = None, question: str = "",
    ) -> list[QueryNode]:
        """Probe actual DB values for TEXT filter columns to fix format mismatches."""
        if not db_path or not filter_nodes:
            return filter_nodes

        probed: list[QueryNode] = []
        try:
            conn = sqlite3.connect(str(db_path))
            for node in filter_nodes:
                if node.column.startswith("_expr:") or re.match(
                    r'^(COUNT|SUM|AVG|MIN|MAX)\s*\(', node.column, re.IGNORECASE
                ):
                    probed.append(node)
                    continue
                if node.operator not in ("=", "LIKE"):
                    probed.append(node)
                    continue

                # --- FK value resolution: ID on FK column may be wrong ---
                _col_lower = node.column.lower()
                _val_str = str(node.value)
                _is_fk_col = (
                    _col_lower.endswith("_id") or _col_lower.startswith("link_to")
                    or _col_lower.endswith("id")
                )
                _is_opaque_id = (
                    _val_str.isdigit()
                    or re.match(r'^rec[A-Za-z0-9]{10,}$', _val_str)
                )
                if question and kg and _is_fk_col and _is_opaque_id:
                    resolved = self._resolve_fk_value(node, conn, kg, question)
                    if resolved:
                        probed.append(resolved)
                        continue

                # Check if exact value exists
                try:
                    if node.operator.upper() == "LIKE":
                        row = conn.execute(
                            f'SELECT 1 FROM "{node.table}" WHERE "{node.column}" LIKE ? LIMIT 1',
                            (node.value,),
                        ).fetchone()
                    else:
                        row = conn.execute(
                            f'SELECT 1 FROM "{node.table}" WHERE "{node.column}" = ? LIMIT 1',
                            (node.value,),
                        ).fetchone()
                    if row:
                        # Exact match exists, but check if there are also prefix
                        # variants (e.g., "Advertisement" exact + "Advertisement for X").
                        # If so, use LIKE prefix to capture all related rows.
                        # Guard: value must be 4+ chars and prefix additions must start
                        # with a word boundary (space/punctuation after the value).
                        if node.operator == "=" and len(node.value) >= 4:
                            exact_cnt = conn.execute(
                                f'SELECT COUNT(*) FROM "{node.table}" WHERE "{node.column}" = ?',
                                (node.value,),
                            ).fetchone()[0]
                            # LIKE 'Value %' — require space after value for word-boundary safety
                            prefix_cnt = conn.execute(
                                f'SELECT COUNT(*) FROM "{node.table}" WHERE "{node.column}" LIKE ? OR "{node.column}" = ?',
                                (f"{node.value} %", node.value),
                            ).fetchone()[0]
                            if prefix_cnt > exact_cnt:
                                # Don't expand if multiple distinct values share this prefix
                                # (e.g. "VYBER" and "VYBER KARTOU" are separate categories)
                                distinct_prefixed = conn.execute(
                                    f'SELECT COUNT(DISTINCT "{node.column}") FROM "{node.table}" '
                                    f'WHERE "{node.column}" LIKE ? OR "{node.column}" = ?',
                                    (f"{node.value} %", node.value),
                                ).fetchone()[0]
                                if distinct_prefixed <= 1:
                                    probed.append(QueryNode(
                                        table=node.table, column=node.column, role=node.role,
                                        operator="LIKE", value=f"{node.value}%",
                                    ))
                                    self._log("value_probe", f"{node.table}.{node.column}: '{node.value}' → LIKE '{node.value}%' ({prefix_cnt} vs {exact_cnt} exact)")
                                    continue
                        probed.append(node)
                        continue
                    # Value doesn't match — try FK resolution (name→ID lookup)
                    if question and kg and not str(node.value).isdigit():
                        resolved = self._resolve_fk_value(node, conn, kg, question)
                        if resolved:
                            probed.append(resolved)
                            continue
                    # Year→FK resolution: numeric value looks like a year but the column
                    # holds small FK IDs. Convert to subquery via the referenced table.
                    if (question and kg and str(node.value).isdigit()
                            and 1900 <= int(node.value) <= 2100
                            and node.column.lower().endswith("id")):
                        # Find FK target table with a 'year' column
                        col_id = f"{node.table}.{node.column}"
                        fk_edges = kg.graph.fk_from.get(col_id, []) if kg.graph else []
                        for edge in fk_edges:
                            dst_parts = edge.dst.split(".")
                            if len(dst_parts) == 2:
                                ref_table = dst_parts[0]
                                # Check if ref_table has a 'year' column
                                ref_schema = kg.get_table(ref_table)
                                if ref_schema and any(c.name.lower() == "year" for c in ref_schema.columns):
                                    # Build subquery filter using year + any name mention in question
                                    name_filter = ""
                                    if ref_schema and any(c.name.lower() == "name" for c in ref_schema.columns):
                                        # Extract potential race/event name from question
                                        q_lower = question.lower()
                                        name_samples = conn.execute(
                                            f'SELECT DISTINCT "name" FROM "{ref_table}" LIMIT 50'
                                        ).fetchall()
                                        for (ns,) in name_samples:
                                            if ns and ns.lower() in q_lower:
                                                name_filter = f' AND "name" = \'{ns}\''
                                                break
                                            # partial match: "Chinese Grand Prix" in question
                                            if ns and all(w.lower() in q_lower for w in ns.split() if len(w) > 2):
                                                name_filter = f' AND "name" = \'{ns}\''
                                                break
                                    subquery = (
                                        f'(SELECT "{dst_parts[1]}" FROM "{ref_table}" '
                                        f'WHERE "year" = {node.value}{name_filter})'
                                    )
                                    probed.append(QueryNode(
                                        table=node.table, column=node.column, role=node.role,
                                        operator="IN", value=subquery,
                                    ))
                                    self._log("value_probe",
                                              f"{node.table}.{node.column}: '{node.value}' is year → "
                                              f"subquery via {ref_table} (year={node.value}{name_filter})")
                                    break
                        else:
                            # No FK edge found — keep as-is, fall through to other checks
                            pass
                        if probed and probed[-1].table == node.table and probed[-1].column == node.column and probed[-1].operator == "IN":
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
                    # Try prefix match: value is a prefix of stored values (e.g.,
                    # "Advertisement" matches "Advertisement for event promotion")
                    if len(val_lower) >= 3:
                        prefix_matches = [
                            v for v in sample_vals
                            if v.lower().startswith(val_lower)
                        ]
                        if not prefix_matches:
                            # Also check DB beyond sample with LIKE
                            like_row = conn.execute(
                                f'SELECT 1 FROM "{node.table}" WHERE "{node.column}" LIKE ? LIMIT 1',
                                (f"{node.value}%",),
                            ).fetchone()
                            if like_row:
                                prefix_matches = [node.value]
                        if prefix_matches:
                            probed.append(QueryNode(
                                table=node.table, column=node.column, role=node.role,
                                operator="LIKE", value=f"{node.value}%",
                            ))
                            self._log("value_probe", f"{node.table}.{node.column}: '{node.value}' → LIKE '{node.value}%' (prefix match)")
                            continue
                    # Abbreviation match: filter value is a long word but column stores
                    # short codes that are prefixes of the value (e.g. "iodine" → "i")
                    _non_empty_samples = [v for v in sample_vals if v]
                    if len(val_lower) >= 4 and _non_empty_samples and max(len(v) for v in _non_empty_samples) <= 3:
                        abbrev_matches = [
                            v for v in sample_vals
                            if v and val_lower.startswith(v.lower()) and len(v) <= len(val_lower) // 2
                        ]
                        if abbrev_matches:
                            best = max(abbrev_matches, key=len)
                            probed.append(QueryNode(
                                table=node.table, column=node.column, role=node.role,
                                operator=node.operator, value=best,
                            ))
                            self._log("value_probe", f"{node.table}.{node.column}: '{node.value}' → '{best}' (abbreviation match)")
                            continue

                    # Try date-format reconciliation: if the filter value looks like
                    # a date prefix and samples show a different date format, rewrite
                    rewritten = self._reconcile_date_format(node, sample_vals)
                    if rewritten:
                        # Verify rewritten filter actually produces results
                        verify_op = "LIKE" if rewritten.operator == "LIKE" else "="
                        verify_row = conn.execute(
                            f'SELECT 1 FROM "{rewritten.table}" WHERE "{rewritten.column}" {verify_op} ? LIMIT 1',
                            (rewritten.value,),
                        ).fetchone()
                        if verify_row:
                            probed.append(rewritten)
                            self._log("value_probe", f"{node.table}.{node.column}: '{node.value}' → {rewritten.operator} '{rewritten.value}'")
                            continue
                        # Rewritten format correct but column has no matching data —
                        # look for semantic sibling (same column name in another table)
                        sibling = self._find_sibling_column(node, conn, kg)
                        if sibling:
                            probed.append(sibling)
                            self._log("value_probe", f"{node.table}.{node.column}: no data for '{node.value}' → moved to {sibling.table}.{sibling.column} {sibling.operator} '{sibling.value}'")
                            continue
                    else:
                        # No date reconciliation possible — still try sibling column
                        sibling = self._find_sibling_column(node, conn, kg)
                        if sibling:
                            probed.append(sibling)
                            self._log("value_probe", f"{node.table}.{node.column}: no data for '{node.value}' → moved to {sibling.table}.{sibling.column} {sibling.operator} '{sibling.value}'")
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

                    # Same-table column search: value not found in assigned column,
                    # check other TEXT columns in the same table
                    if kg:
                        table_schema = kg.get_table(node.table)
                        if table_schema:
                            found_col = False
                            for col in table_schema.columns:
                                if col.name == node.column:
                                    continue
                                if col.sql_type.upper() not in ("TEXT", "VARCHAR", "CHAR", ""):
                                    continue
                                try:
                                    row = conn.execute(
                                        f'SELECT 1 FROM "{node.table}" WHERE "{col.name}" = ? LIMIT 1',
                                        (node.value,),
                                    ).fetchone()
                                    if row:
                                        probed.append(QueryNode(
                                            table=node.table, column=col.name, role=node.role,
                                            operator=node.operator, value=node.value,
                                        ))
                                        self._log("value_probe", f"{node.table}.{node.column}: '{node.value}' not found → moved to {node.table}.{col.name}")
                                        found_col = True
                                        break
                                except Exception:
                                    continue
                            if found_col:
                                continue
                            # Doc-backed resolution: search raw doc text for the value
                            # near an ID, then filter by _id directly
                            if hasattr(self, '_doc_paths') and self._doc_paths:
                                resolved_id = self._resolve_value_from_doc(
                                    node.value, node.table
                                )
                                if resolved_id:
                                    probed.append(QueryNode(
                                        table=node.table, column="_id", role=node.role,
                                        operator="=", value=resolved_id,
                                    ))
                                    self._log("value_probe", f"{node.table}.{node.column}: '{node.value}' resolved from doc → _id='{resolved_id}'")
                                    continue
                            probed.append(node)
                            self._log("value_probe", f"{node.table}.{node.column}: '{node.value}' not found in DB. Samples: {sample_vals[:5]}")
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

    def _strip_spurious_filters(
        self, question: str, filter_nodes: list[QueryNode], user_intent: str,
        db_path: Path | None = None,
    ) -> list[QueryNode]:
        """Deterministically remove filters whose values are not grounded in the question."""
        q_lower = question.lower()
        # Extract only semantic parts of intent
        intent_population = ""
        intent_metric = ""
        for line in (user_intent or "").split("\n"):
            if "Population (WHERE):" in line:
                intent_population = line.split("Population (WHERE):")[1].strip().lower()
            elif "Metric (SELECT):" in line:
                intent_metric = line.split("Metric (SELECT):")[1].strip().lower()

        q_and_intent = q_lower + " " + intent_population + " " + intent_metric
        q_words = set(re.findall(r'[a-z]{2,}', q_and_intent))
        symbol_map = {"#": "triple", "=": "double", "-": "single", "+": "carcinogenic"}

        # Stem question words for matching FK column names
        q_words_stemmed = {w.rstrip("s") if len(w) > 3 else w for w in q_words}

        kept: list[QueryNode] = []
        for node in filter_nodes:
            if node.column.startswith("_expr:") or node.operator != "=":
                kept.append(node)
                continue
            val_lower = str(node.value).lower()
            col_lower = node.column.lower()

            # Keep subquery-style filters — picker intentionally constructed them
            if val_lower.lstrip().startswith("(select") or val_lower.lstrip().startswith("select"):
                kept.append(node)
                continue

            # FK-resolved numeric values: keep if column name words overlap with question
            if val_lower.isdigit() and col_lower.endswith("_id"):
                col_name_words = set(re.findall(r'[a-z]{3,}', col_lower.replace("_id", "")))
                col_words_stemmed = {w.rstrip("s") if len(w) > 3 else w for w in col_name_words}
                if col_words_stemmed & q_words_stemmed:
                    kept.append(node)
                    continue

            if len(val_lower) <= 1:
                domain_word = symbol_map.get(val_lower, "")
                if domain_word and domain_word in q_and_intent:
                    kept.append(node)
                elif col_lower in q_words:
                    kept.append(node)
                else:
                    # Last resort: check if value exists in DB column
                    if self._value_exists_in_db(db_path, node):
                        kept.append(node)
                    else:
                        self._log("kg_strip_spurious", f'Removed: "{node.table}"."{node.column}" = \'{node.value}\' (not in question)')
            else:
                val_in_q = val_lower in q_and_intent or any(val_lower in w for w in q_words)
                col_in_q = col_lower in q_words
                if val_in_q or col_in_q:
                    kept.append(node)
                else:
                    # Last resort: check if value exists in DB column
                    if self._value_exists_in_db(db_path, node):
                        kept.append(node)
                    else:
                        self._log("kg_strip_spurious", f'Removed: "{node.table}"."{node.column}" = \'{node.value}\' (not in question)')
        return kept

    def _value_exists_in_db(self, db_path: Path | None, node: QueryNode) -> bool:
        """Check if a filter value actually exists in its column."""
        if not db_path or not db_path.exists():
            return False
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            row = conn.execute(
                f'SELECT 1 FROM "{node.table}" WHERE "{node.column}" = ? LIMIT 1',
                (node.value,),
            ).fetchone()
            conn.close()
            return row is not None
        except Exception:
            return False

    def _db_check(self, db_path: Path | None, table: str, column: str,
                  operator: str = "IS NOT NULL", value: Any = None) -> bool:
        """Single utility for all DB existence checks.

        Returns True if at least one row matches, False if no match or error.
        Handles subquery values (returns True — can't evaluate, assume valid).
        """
        if not db_path or not db_path.exists():
            return True
        if value is not None:
            val_str = str(value).strip()
            if val_str.lstrip("(").lower().startswith("select"):
                return True
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            if operator == "IS NOT NULL" or value is None:
                row = conn.execute(
                    f'SELECT 1 FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT 1'
                ).fetchone()
            else:
                row = conn.execute(
                    f'SELECT 1 FROM "{table}" WHERE "{column}" {operator} ? LIMIT 1',
                    (value,),
                ).fetchone()
            conn.close()
            return row is not None
        except Exception:
            return True

    def _detect_per_unit_filters(
        self, question: str, filter_nodes: list[QueryNode], kg: KnowledgeGraph | None,
    ) -> list[QueryNode]:
        """Detect 'per unit' language and transform numeric filters into computed expressions.

        When question says "X per unit/per item/per piece" and there's a numeric filter
        on a value column (price, cost, rate) in the same table as a quantity column
        (amount, quantity, count, units), transform the filter to value/quantity.
        """
        if not kg or not kg.graph:
            return filter_nodes

        # Check if question contains "per unit" type language
        per_match = re.search(
            r'\bper\s+(unit|item|piece|product|transaction|order|kg|liter|litre|gallon)\b',
            question.lower(),
        )
        if not per_match:
            return filter_nodes

        # Find numeric comparison filters (>, <, >=, <=) that might be "per unit" values
        value_col_patterns = {"price", "cost", "total", "amount", "value", "revenue", "fee", "charge", "rate"}
        quantity_col_patterns = {"amount", "quantity", "qty", "count", "units", "volume", "num", "number"}

        new_nodes: list[QueryNode] = []
        used_as_denominator: set[tuple[str, str]] = set()
        for node in filter_nodes:
            if node.operator not in (">", "<", ">=", "<="):
                new_nodes.append(node)
                continue

            # Is this a value/price column?
            col_lower = node.column.lower()
            if col_lower not in value_col_patterns and not any(p in col_lower for p in value_col_patterns):
                new_nodes.append(node)
                continue

            # Find a quantity column in the same table
            table_schema = kg.get_table(node.table)
            if not table_schema:
                new_nodes.append(node)
                continue

            quantity_col = None
            for c in table_schema.columns:
                cn = c.name.lower()
                if cn == col_lower:
                    continue
                if cn in quantity_col_patterns or any(p in cn for p in quantity_col_patterns):
                    if c.sql_type.upper() in ("INT", "INTEGER", "REAL", "FLOAT", "NUMERIC", "NUM"):
                        quantity_col = c.name
                        break

            if not quantity_col:
                new_nodes.append(node)
                continue

            # Transform: Price > 29 → CAST(Price AS REAL) / Amount > 29
            expr_sql = f'CAST("{node.table}"."{node.column}" AS REAL) / "{node.table}"."{quantity_col}"'
            new_nodes.append(QueryNode(
                table=node.table, column=f"_expr:{expr_sql}", role="filter",
                operator=node.operator, value=node.value,
            ))
            used_as_denominator.add((node.table, quantity_col.lower()))

        # Remove trivial filters on columns now used as denominators (e.g. Amount > 0)
        if used_as_denominator:
            new_nodes = [
                n for n in new_nodes
                if not (
                    (n.table, n.column.lower()) in used_as_denominator
                    and n.operator in (">", ">=")
                    and str(n.value) in ("0", "0.0", "0.00")
                )
            ]

        return new_nodes

    def _resolve_fk_value(
        self, node: QueryNode, conn: sqlite3.Connection, kg: KnowledgeGraph, question: str,
    ) -> QueryNode | None:
        """Resolve FK ID values by looking up the referenced display column.

        When a filter targets an _id column with a numeric value, the picker may have
        guessed the wrong ID. Look up the FK's display column (e.g., colour.colour for
        eye_colour_id) and find the value matching the question, then return the correct ID.
        """
        g = kg.graph
        if not g:
            return None
        col_id = f"{node.table}.{node.column}"
        display_col_id = g.fk_display_map.get(col_id)
        if not display_col_id:
            # Try to find FK target from all FKs (declared + inferred)
            ref_table = None
            ref_col = None
            for src_tbl, fk in kg.all_foreign_keys():
                if src_tbl == node.table and fk.column == node.column:
                    ref_table = fk.ref_table
                    ref_col = fk.ref_column
                    break
            if not ref_table:
                return None
            # Find a text column in referenced table to use as display
            ref_schema = kg.get_table(ref_table)
            if not ref_schema:
                return None
            display_col_name = None
            for col in ref_schema.columns:
                if col.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", "") and col.name != ref_col:
                    display_col_name = col.name
                    break
            if not display_col_name:
                return None
            display_col_id = f"{ref_table}.{display_col_name}"

        # Parse display_col_id
        if "." not in display_col_id:
            return None
        disp_table, disp_col = display_col_id.split(".", 1)

        # Get FK target (the PK column in the referenced table)
        fk_target_col = None
        for src_tbl, fk in kg.all_foreign_keys():
            if src_tbl == node.table and fk.column == node.column:
                fk_target_col = fk.ref_column
                break
        if not fk_target_col:
            fk_target_col = "id"

        # Extract question words for matching
        q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))

        # Query display values and find the one matching a question word
        # Also try multi-column name matching (first_name + last_name)
        try:
            ref_schema = kg.get_table(disp_table)
            text_cols = [
                c.name for c in (ref_schema.columns if ref_schema else [])
                if c.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", "")
                and c.name != fk_target_col
            ]
            # Build concatenated name query if multiple name-like columns exist
            name_cols = [c for c in text_cols if any(
                k in c.lower() for k in ("name", "first", "last", "surname", "forename")
            )]
            if len(name_cols) >= 2:
                concat_expr = " || ' ' || ".join(f'"{c}"' for c in name_cols)
                rows = conn.execute(
                    f'SELECT "{fk_target_col}", {concat_expr} FROM "{disp_table}"'
                ).fetchall()
            else:
                rows = conn.execute(
                    f'SELECT "{fk_target_col}", "{disp_col}" FROM "{disp_table}" '
                    f'WHERE "{disp_col}" IS NOT NULL'
                ).fetchall()
        except Exception:
            return None

        best_id = None
        best_match = ""
        q_lower = question.lower()
        candidates = []
        for row_id, display_val in rows:
            dv_lower = str(display_val).lower().strip()
            if dv_lower in q_words:
                candidates.append((len(dv_lower), str(row_id), str(display_val)))
            elif len(dv_lower) >= 3 and re.search(r'\b' + re.escape(dv_lower) + r'\b', q_lower):
                candidates.append((len(dv_lower), str(row_id), str(display_val)))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            # Disambiguate ties: when multiple IDs share the same display value,
            # check other columns (e.g. year) against question context
            top_len = candidates[0][0]
            tied = [c for c in candidates if c[0] == top_len]
            if len(tied) > 1:
                q_numbers = re.findall(r'\b((?:19|20)\d{2})\b', question)
                if q_numbers and ref_schema:
                    year_cols = [c.name for c in ref_schema.columns
                                 if any(k in c.name.lower() for k in ("year", "date"))]
                    if year_cols:
                        year_col = year_cols[0]
                        tied_ids = [c[1] for c in tied]
                        placeholders = ",".join("?" * len(tied_ids))
                        try:
                            disambig_rows = conn.execute(
                                f'SELECT "{fk_target_col}", "{year_col}" FROM "{disp_table}" '
                                f'WHERE "{fk_target_col}" IN ({placeholders})',
                                tuple(tied_ids),
                            ).fetchall()
                            for row_id, year_val in disambig_rows:
                                year_str = str(year_val).strip()
                                if any(y in year_str for y in q_numbers):
                                    candidates = [(top_len, str(row_id), tied[0][2])]
                                    break
                        except Exception:
                            pass
            best_id = candidates[0][1]
            best_match = candidates[0][2]

        if best_id and best_id != str(node.value):
            self._log("fk_resolve", f"{node.table}.{node.column}: '{node.value}' → '{best_id}' (matched '{best_match}' in {disp_table}.{disp_col})")
            return QueryNode(
                table=node.table, column=node.column, role=node.role,
                operator=node.operator, value=best_id,
            )
        return None

    def _find_sibling_column(
        self, node: QueryNode, conn: sqlite3.Connection, kg: KnowledgeGraph | None,
    ) -> QueryNode | None:
        """When a filter value has no matches in the picked column, look for a
        same-named or semantically-linked column in another table that does contain
        matching values. Returns a rewritten QueryNode if found.
        """
        if not kg or not kg.graph:
            return None

        # Candidates: (1) same column name in other tables, (2) semantic-edge siblings
        candidates: list[tuple[str, str]] = []

        # Same column name in other tables
        col_lower = node.column.lower()
        for col_id, col_node in kg.graph.columns.items():
            if col_node.name.lower() == col_lower and col_node.table_id != node.table:
                candidates.append((col_node.table_id, col_node.name))

        # Semantic-edge siblings (high similarity columns in other tables)
        if kg.graph.semantic_edges:
            src_id = f"{node.table}.{node.column}"
            for edge in kg.graph.semantic_edges:
                if edge.src == src_id:
                    parts = edge.dst.split(".", 1)
                    if len(parts) == 2 and parts[0] != node.table:
                        candidates.append((parts[0], parts[1]))
                elif edge.dst == src_id:
                    parts = edge.src.split(".", 1)
                    if len(parts) == 2 and parts[0] != node.table:
                        candidates.append((parts[0], parts[1]))

        # Try each candidate — check if value (or reconciled format) produces results
        for tbl, col in candidates:
            # Try exact value first
            try:
                row = conn.execute(
                    f'SELECT 1 FROM "{tbl}" WHERE "{col}" = ? LIMIT 1',
                    (node.value,),
                ).fetchone()
                if row:
                    return QueryNode(
                        table=tbl, column=col, role=node.role,
                        operator="=", value=node.value,
                    )
                # Try LIKE if original was LIKE
                if node.operator == "LIKE":
                    row = conn.execute(
                        f'SELECT 1 FROM "{tbl}" WHERE "{col}" LIKE ? LIMIT 1',
                        (node.value,),
                    ).fetchone()
                    if row:
                        return QueryNode(
                            table=tbl, column=col, role=node.role,
                            operator="LIKE", value=node.value,
                        )
                # Try reconciled date format against this sibling's values
                samples = conn.execute(
                    f'SELECT DISTINCT "{col}" FROM "{tbl}" WHERE "{col}" IS NOT NULL LIMIT 20'
                ).fetchall()
                sibling_samples = [str(r[0]) for r in samples]
                rewritten = self._reconcile_date_format(
                    QueryNode(table=tbl, column=col, role=node.role, operator=node.operator, value=node.value),
                    sibling_samples,
                )
                if rewritten:
                    verify_row = conn.execute(
                        f'SELECT 1 FROM "{tbl}" WHERE "{col}" {rewritten.operator} ? LIMIT 1',
                        (rewritten.value,),
                    ).fetchone()
                    if verify_row:
                        return rewritten
            except Exception:
                continue
        return None

    def _resolve_value_from_doc(self, value: str, table: str) -> str | None:
        """Search raw doc text for a filter value near an ID pattern.

        When extracted doc data has gaps (e.g., race name not populated for a record),
        fall back to scanning the original doc text for the value near an ID reference.
        Returns the ID string if found, None otherwise.
        """
        val_lower = value.lower()
        val_words = set(val_lower.split())
        table_lower = table.lower()

        id_patterns = [
            r'\(\w+\s+ID\s*[:.]?\s*(\d+)\)',
            r'(?:\w+\s+)?ID\s*[:.]?\s*(\d+)',
            r'(?:identifier|registered\s+under)\s*[:.]?\s*(\d+)',
        ]
        # Two passes: exact match first, then fuzzy — prevents "Grand Prix" in
        # every paragraph from resolving to the wrong race
        for require_exact in (True, False):
            for doc_path in self._doc_paths:
                doc_stem = doc_path.stem.lower()
                if table_lower not in doc_stem and doc_stem not in table_lower:
                    continue
                try:
                    doc_text = doc_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
                for para in paragraphs:
                    para_lower = para.lower()
                    if require_exact:
                        if val_lower not in para_lower:
                            continue
                    else:
                        if val_lower in para_lower:
                            continue  # already tried in exact pass
                        if not (
                            len(val_words) > 1
                            and sum(1 for w in val_words if w in para_lower) >= len(val_words) - 1
                        ):
                            continue
                    for pat in id_patterns:
                        m = re.search(pat, para, re.IGNORECASE)
                        if m:
                            return m.group(1)
        return None

    def _reconcile_date_format(self, node: QueryNode, sample_vals: list[str]) -> QueryNode | None:
        """When a filter value doesn't match DB values, try to reconcile date formats.

        Extracts year/month/day components from the filter value, detects the
        actual date format from sample values, and rewrites the filter to match.
        Works for any combination of separators (-, /, .), compacted formats (YYYYMMDD),
        and partial prefixes (YYYYMM%).
        """
        val = node.value.rstrip("%")
        # Extract numeric segments from the filter value
        segments = re.findall(r'\d+', val)
        if not segments:
            return None

        # Determine year/month from segments
        year = month = day = None
        if len(segments) == 1 and len(segments[0]) >= 6:
            # Compacted: YYYYMM or YYYYMMDD
            s = segments[0]
            year, month = s[:4], s[4:6]
            if len(s) >= 8:
                day = s[6:8]
        elif len(segments) >= 2:
            # Separated: YYYY-MM or YYYY-MM-DD or MM/DD/YYYY etc.
            if len(segments[0]) == 4:
                year, month = segments[0], segments[1]
                if len(segments) >= 3:
                    day = segments[2]
            elif len(segments[-1]) == 4:
                year = segments[-1]
                month = segments[0]
                if len(segments) >= 3:
                    day = segments[1]
            else:
                return None
        else:
            return None

        if not year or not month:
            return None

        # Detect actual format from samples
        sample_sep = None
        sample_fmt = None  # "separated" or "compacted"
        for sv in sample_vals:
            sv_str = str(sv)
            if re.match(r'^\d{4}[-/\.]\d{2}[-/\.]\d{2}', sv_str):
                sample_sep = sv_str[4]
                sample_fmt = "separated"
                break
            elif re.match(r'^\d{8}$', sv_str):
                sample_fmt = "compacted"
                break
            elif re.match(r'^\d{6}$', sv_str):
                sample_fmt = "compacted_ym"
                break

        if not sample_fmt:
            return None

        # Build the rewritten filter value in the actual DB format
        if sample_fmt == "separated":
            if day:
                new_val = f"{year}{sample_sep}{month}{sample_sep}{day}"
                return QueryNode(
                    table=node.table, column=node.column, role=node.role,
                    operator="=", value=new_val,
                )
            else:
                # Prefix match for year-month
                new_val = f"{year}{sample_sep}{month}%"
                return QueryNode(
                    table=node.table, column=node.column, role=node.role,
                    operator="LIKE", value=new_val,
                )
        elif sample_fmt == "compacted":
            if day:
                new_val = f"{year}{month}{day}"
                return QueryNode(
                    table=node.table, column=node.column, role=node.role,
                    operator="=", value=new_val,
                )
            else:
                new_val = f"{year}{month}%"
                return QueryNode(
                    table=node.table, column=node.column, role=node.role,
                    operator="LIKE", value=new_val,
                )
        elif sample_fmt == "compacted_ym":
            new_val = f"{year}{month}"
            return QueryNode(
                table=node.table, column=node.column, role=node.role,
                operator="=", value=new_val,
            )
        return None

    def _apply_domain_column_fixes(
        self,
        question: str,
        filter_nodes: list[QueryNode],
        kg: KnowledgeGraph,
        anchor_text: str,
        db_path: Path | None = None,
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
            q_words_dc = re.findall(r'\b[a-z]+', q_lower)
            if not any(w.startswith(defined_col) for w in q_words_dc) or defined_col in used_cols:
                continue
            # Find which table has this column
            for t in kg.tables:
                for c in t.columns:
                    if c.name.lower() == defined_col:
                        domain_cols[defined_col] = (t.name, m.group(2))
                        break

        # Swap: if a filter uses a synonym column (position↔rank, round↔number),
        # replace it with the domain-defined column — validate against DB
        synonym_groups = [
            {"position", "rank", "positionorder"},
            {"round", "number"},
        ]
        fixed = []
        for node in filter_nodes:
            swapped = False
            for defined_col, (col_table, definition) in domain_cols.items():
                for group in synonym_groups:
                    if node.column.lower() in group and defined_col in group:
                        val_str = str(node.value).strip()
                        is_subquery = val_str.lstrip("(").lower().startswith("select")

                        if not db_path or is_subquery:
                            # Can't validate — trust domain knowledge
                            fixed.append(QueryNode(
                                table=col_table, column=defined_col, role=node.role,
                                operator=node.operator, value=node.value,
                            ))
                            self._log("domain_fix",
                                f'Swapped {node.table}.{node.column} → {col_table}.{defined_col} '
                                f'(domain: "{definition[:60]}")')
                            swapped = True
                            break

                        # Try domain column with original value, then type-cast alternatives
                        best_col = None
                        best_val = node.value
                        candidates = [(defined_col, node.value)]
                        try:
                            int_val = int(float(str(node.value)))
                            if str(int_val) != str(node.value):
                                candidates.append((defined_col, int_val))
                        except (ValueError, TypeError):
                            pass
                        for alt_col in group:
                            if alt_col != defined_col and alt_col != node.column.lower():
                                ts = kg.get_table(col_table)
                                if ts and any(c.name.lower() == alt_col for c in ts.columns):
                                    candidates.append((alt_col, node.value))

                        for cand_col, cand_val in candidates:
                            if self._db_check(db_path, col_table, cand_col, node.operator, cand_val):
                                best_col = cand_col
                                best_val = cand_val
                                break

                        if best_col:
                            fixed.append(QueryNode(
                                table=col_table, column=best_col, role=node.role,
                                operator=node.operator, value=best_val,
                            ))
                            self._log("domain_fix",
                                f'Swapped {node.table}.{node.column} → {col_table}.{best_col} '
                                f'(domain: "{definition[:60]}", validated)')
                            swapped = True
                        else:
                            self._log("domain_fix",
                                f'No valid column in group {group} for value {node.value} — keeping {node.table}.{node.column}')
                        break
                if swapped:
                    break
            if not swapped:
                fixed.append(node)

        # Phase 2: Explicit "Filter using Column containing Value" from domain anchors
        # Handles patterns like: "Filter using `District` containing 'Riverside'"
        filter_instructions: list[tuple[str, str, str]] = []  # (column_fragment, operator, value)
        for m in re.finditer(
            r'[Ff]ilter\s+using\s+[`\']?(\w[\w\s]*?\w)[`\']?\s+containing\s+[\'"](\w+)[\'"]',
            anchor_text,
        ):
            filter_instructions.append((m.group(1).lower(), "LIKE", m.group(2)))
        for m in re.finditer(
            r'[Ff]ilter\s+using\s+[`\']?(\w[\w\s]*?\w)[`\']?\s*=\s*[\'"]([^"\']+)[\'"]',
            anchor_text,
        ):
            filter_instructions.append((m.group(1).lower(), "=", m.group(2)))

        if filter_instructions:
            fixed2 = []
            for node in fixed:
                replaced = False
                for anchor_col_frag, anchor_op, anchor_val in filter_instructions:
                    if str(node.value).lower() != anchor_val.lower():
                        continue
                    # The anchor says to use a specific column for this value
                    # Check if node is using a DIFFERENT column
                    if anchor_col_frag in node.column.lower():
                        break  # already using the right column
                    # Find the correct column (prefer TEXT columns for LIKE operations)
                    for t in kg.tables:
                        candidates = [
                            c.name for c in t.columns
                            if anchor_col_frag in c.name.lower()
                            and c.sql_type.upper() in ("TEXT", "VARCHAR", "NVARCHAR", "CHAR", "")
                        ]
                        for match_col in candidates:
                            if db_path and not self._db_check(db_path, t.name, match_col, "LIKE", f"%{anchor_val}%"):
                                continue
                            self._log("domain_fix",
                                f'Anchor override: {node.table}.{node.column} → {t.name}.{match_col} '
                                f'(anchor: "Filter using {anchor_col_frag} containing {anchor_val}")')
                            op = "LIKE" if anchor_op == "LIKE" else node.operator
                            val = f"%{anchor_val}%" if anchor_op == "LIKE" else anchor_val
                            fixed2.append(QueryNode(
                                table=t.name, column=match_col, role=node.role,
                                operator=op, value=val,
                            ))
                            replaced = True
                            break
                        if replaced:
                            break
                    if replaced:
                        break
                if not replaced:
                    fixed2.append(node)
            fixed = fixed2

        # Phase 3: Question-word column override (no anchor dependency)
        # If question says "X-related districts" and filter is on "County Name" LIKE 'X',
        # but a "District Name" column exists with the same value, prefer it.
        q_words_set = set(re.findall(r'\b[a-z]{3,}\b', q_lower))

        def _stem_match_score(col_words: set[str], q_words: set[str]) -> int:
            """Count how many column words have a stem match in the question."""
            score = 0
            for cw in col_words:
                stem = cw[:4] if len(cw) > 4 else cw
                if any(qw.startswith(stem) or cw.startswith(qw[:4]) for qw in q_words):
                    score += 1
            return score

        fixed3 = []
        for node in fixed:
            if node.operator not in ("LIKE", "="):
                fixed3.append(node)
                continue
            node_col_words = set(re.findall(r'[a-z]+', node.column.lower()))
            current_score = _stem_match_score(node_col_words, q_words_set)
            # Don't override if value validates in current column — unless
            # an alternative has a much stronger question-word match (score >= 2)
            _value_validates = db_path and self._db_check(
                db_path, node.table, node.column, node.operator, node.value
            )
            _require_strong_alt = _value_validates

            best_alt = None
            best_alt_score = current_score
            node_table = next((t for t in kg.tables if t.name == node.table), None)
            if not node_table:
                fixed3.append(node)
                continue

            for c in node_table.columns:
                if c.name == node.column:
                    continue
                if c.sql_type.upper() not in ("TEXT", "VARCHAR", "NVARCHAR", "CHAR", ""):
                    continue
                alt_col_words = set(re.findall(r'[a-z]+', c.name.lower()))
                alt_score = _stem_match_score(alt_col_words, q_words_set)
                if alt_score <= best_alt_score:
                    continue
                # This column matches more question words — validate value exists
                val_to_check = str(node.value)
                if "%" not in val_to_check:
                    val_to_check = f"%{val_to_check}%"
                if db_path and not self._db_check(db_path, node.table, c.name, "LIKE", val_to_check):
                    continue
                best_alt = c.name
                best_alt_score = alt_score

            if best_alt and best_alt_score > current_score:
                # When value already validates in current column, require strong alt match
                if _require_strong_alt and best_alt_score < 2:
                    fixed3.append(node)
                    continue
                val = str(node.value)
                if node.operator == "LIKE" and "%" not in val:
                    val = f"%{val}%"
                self._log("domain_fix",
                    f'Question-word override: {node.table}.{node.column} → {node.table}.{best_alt} '
                    f'(question words match "{best_alt}" better)')
                fixed3.append(QueryNode(
                    table=node.table, column=best_alt, role=node.role,
                    operator="LIKE", value=val,
                ))
            else:
                fixed3.append(node)
        fixed = fixed3

        return fixed

    def _apply_knowledge_value_mappings(
        self,
        filter_nodes: list[QueryNode],
        anchor_text: str,
        kg: KnowledgeGraph,
        db_path: Path | None,
    ) -> list[QueryNode]:
        """Override filters using explicit knowledge mappings like 'column = value for keyword'.

        Parses anchor patterns such as:
          "Use label = '+' for carcinogenic"
          "'M' for male"
          "label = '+' for carcinogenic molecules"
        When a filter's value/column name matches a keyword, replace with the canonical form.
        """
        if not anchor_text or not filter_nodes:
            return filter_nodes

        # Parse patterns: column = 'value' for keyword
        # e.g. "label = '+' for carcinogenic" or "Use label = '+' for carcinogenic"
        mappings: list[tuple[str, str, str]] = []  # (column, value, keyword)
        for m in re.finditer(
            r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]\s+for\s+(\w+)",
            anchor_text,
        ):
            col, val, keyword = m.group(1), m.group(2), m.group(3).lower()
            mappings.append((col, val, keyword))

        # Also parse descriptive patterns:
        # "column: ... keyword ('value') or keyword2 ('value2')"
        # e.g. "label: Indicates whether the molecule is carcinogenic ('+') or non-carcinogenic ('-')"
        for m in re.finditer(
            r"- (\w+):\s+[^-\n]*?(\w+)\s*\(['\"]([^'\"]+)['\"]\)",
            anchor_text,
        ):
            col, keyword, val = m.group(1), m.group(2).lower(), m.group(3)
            if keyword not in ("e", "i", "a", "is", "the", "to", "of"):
                mappings.append((col, val, keyword))

        if not mappings:
            return filter_nodes

        fixed: list[QueryNode] = []
        for node in filter_nodes:
            replaced = False
            node_val_lower = str(node.value).lower()
            node_col_lower = node.column.lower()

            for col_name, canon_val, keyword in mappings:
                # Match if: filter value IS the keyword, OR filter column name contains the keyword
                if node_val_lower != keyword and keyword not in node_col_lower:
                    continue

                # Find the table that has this column with this value
                for t in kg.tables:
                    real_col = next(
                        (c.name for c in t.columns if c.name.lower() == col_name.lower()),
                        None,
                    )
                    if not real_col:
                        continue
                    if db_path and self._db_check(db_path, t.name, real_col, "=", canon_val):
                        # Don't replace if already using the correct column+value
                        if node.table == t.name and node.column == real_col and str(node.value) == canon_val:
                            break
                        fixed.append(QueryNode(
                            table=t.name, column=real_col, role=node.role,
                            operator="=", value=canon_val,
                        ))
                        self._log("knowledge_value_map",
                            f"{node.table}.{node.column}='{node.value}' → {t.name}.{real_col}='{canon_val}' "
                            f"(knowledge: '{canon_val}' for {keyword})")
                        replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                fixed.append(node)
        return fixed

    def _check_filter_selectivity(
        self, filter_nodes: list[QueryNode], db_path: Path | None,
    ) -> str:
        """Check if filters are discriminating. A filter that keeps >90% of rows is likely wrong."""
        if not db_path or not filter_nodes:
            return ""
        # If there are highly selective equality filters on OTHER tables (e.g. name='Alex'),
        # a broad range filter on a different table is fine — the JOIN handles selectivity.
        has_selective_eq = False
        _selective_eq_table = ""
        try:
            _sel_conn = sqlite3.connect(str(db_path))
            filter_tables = {n.table for n in filter_nodes}
            for node in filter_nodes:
                if node.operator != "=" or node.column.startswith("_expr:") or re.match(
                    r'^(COUNT|SUM|AVG|MIN|MAX)\s*\(', node.column, re.IGNORECASE
                ):
                    continue
                _sel_total = _sel_conn.execute(
                    f'SELECT COUNT(*) FROM "{node.table}"'
                ).fetchone()[0]
                if _sel_total == 0:
                    continue
                _sel_match = _sel_conn.execute(
                    f'SELECT COUNT(*) FROM "{node.table}" WHERE "{node.column}" = ?',
                    (node.value,),
                ).fetchone()[0]
                if _sel_match / _sel_total < 0.05:
                    has_selective_eq = True
                    _selective_eq_table = node.table
                    break
            _sel_conn.close()
        except Exception:
            pass

        # Identify columns that have paired range filters (>= and <=) — skip individual checks
        range_paired_cols: set[str] = set()
        col_ops: dict[str, set[str]] = {}
        for node in filter_nodes:
            key = f"{node.table}.{node.column}"
            col_ops.setdefault(key, set()).add(node.operator)
        for key, ops in col_ops.items():
            if (">=" in ops or ">" in ops) and ("<=" in ops or "<" in ops):
                range_paired_cols.add(key)

        issues: list[str] = []
        try:
            conn = sqlite3.connect(str(db_path))
            for node in filter_nodes:
                if node.column.startswith("_expr:") or re.match(
                    r'^(COUNT|SUM|AVG|MIN|MAX)\s*\(', node.column, re.IGNORECASE
                ):
                    continue
                if node.operator in ("LIKE",):
                    continue
                # Skip null-guard filters — they protect ORDER BY, not meant to discriminate
                if node.operator in ("IS NOT", "IS NOT NULL") or (
                    node.operator == "!=" and node.value in ("", "NULL", None)
                ):
                    continue
                # Skip individual bounds of paired range filters (the pair is selective together)
                col_key = f"{node.table}.{node.column}"
                if col_key in range_paired_cols and node.operator in (">=", ">", "<=", "<"):
                    continue
                # Skip range filters when another filter on a DIFFERENT table is highly selective
                # (the JOIN with the selective filter handles the actual row reduction).
                # But DON'T skip if the range filter is nearly vacuous (>95% pass) — that
                # means it's likely the wrong column entirely, not just a broad-but-useful filter.
                if has_selective_eq and node.operator in (">=", ">", "<=", "<"):
                    if node.table != _selective_eq_table:
                        try:
                            _pre_total = conn.execute(
                                f'SELECT COUNT(*) FROM "{node.table}"'
                            ).fetchone()[0]
                            _pre_match = conn.execute(
                                f'SELECT COUNT(*) FROM "{node.table}" WHERE "{node.column}" {node.operator} ?',
                                (node.value,),
                            ).fetchone()[0]
                            if _pre_total > 0 and _pre_match / _pre_total <= 0.95:
                                continue
                        except Exception:
                            continue
                try:
                    total = conn.execute(
                        f'SELECT COUNT(*) FROM "{node.table}"'
                    ).fetchone()[0]
                    if total == 0:
                        continue
                    # Skip boolean/status columns (≤3 distinct values) — high pass-through is expected
                    if node.operator == "=":
                        distinct_count = conn.execute(
                            f'SELECT COUNT(DISTINCT "{node.column}") FROM "{node.table}" '
                            f'WHERE "{node.column}" IS NOT NULL'
                        ).fetchone()[0]
                        if distinct_count <= 3:
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

    def _auto_fix_vacuous_filters(
        self, filter_nodes: list[QueryNode], db_path: Path | None,
    ) -> list[QueryNode]:
        """Replace vacuous range filters (>95% pass) with the best discriminating alternative."""
        if not db_path or not filter_nodes:
            return filter_nodes

        try:
            conn = sqlite3.connect(str(db_path))
        except Exception:
            return filter_nodes

        fixed = []
        for node in filter_nodes:
            if node.column.startswith("_expr:") or node.operator in ("=", "LIKE", "IS NOT NULL", "IN"):
                fixed.append(node)
                continue

            try:
                total = conn.execute(f'SELECT COUNT(*) FROM "{node.table}"').fetchone()[0]
                if total == 0:
                    fixed.append(node)
                    continue
                matching = conn.execute(
                    f'SELECT COUNT(*) FROM "{node.table}" WHERE "{node.column}" {node.operator} ?',
                    (node.value,),
                ).fetchone()[0]
                ratio = matching / total
            except Exception:
                fixed.append(node)
                continue

            if ratio <= 0.95:
                fixed.append(node)
                continue

            # This filter is vacuous — find best alternative (exclude ID/key columns)
            best_col = None
            best_table = None
            best_ratio = 1.0
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            for tbl in tables:
                cols = conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()
                pk_col = cols[0][1] if cols else ""
                for c in cols:
                    col_name = c[1]
                    col_type = c[2].upper()
                    if col_name == node.column and tbl == node.table:
                        continue
                    if node.operator in ("<", ">", "<=", ">=") and col_type not in (
                        "INTEGER", "REAL", "NUMERIC", "FLOAT"
                    ):
                        continue
                    col_lower = col_name.lower()
                    if col_lower == pk_col.lower():
                        continue
                    if col_lower.endswith("id") or col_lower.startswith("link_to"):
                        continue
                    try:
                        alt_match = conn.execute(
                            f'SELECT COUNT(*) FROM "{tbl}" WHERE "{col_name}" {node.operator} ?',
                            (node.value,),
                        ).fetchone()[0]
                        alt_total = conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
                        if alt_total > 0:
                            alt_ratio = alt_match / alt_total
                            if 0.01 < alt_ratio < 0.8 and alt_ratio < best_ratio:
                                best_ratio = alt_ratio
                                best_col = col_name
                                best_table = tbl
                    except Exception:
                        pass

            if best_col and best_table:
                self._log("auto_fix_vacuous",
                    f"Replaced {node.table}.{node.column} ({ratio:.0%} pass) → "
                    f"{best_table}.{best_col} ({best_ratio:.0%} pass)")
                fixed.append(QueryNode(
                    table=best_table, column=best_col, role=node.role,
                    operator=node.operator, value=node.value,
                ))
            else:
                fixed.append(node)

        conn.close()
        return fixed


