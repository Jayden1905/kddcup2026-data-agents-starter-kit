"""KG path planning mixin for QuestionDrivenAgent."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelMessage
from data_agent_baseline.pipeline.kg_builder import KGQueryService, KnowledgeGraph
from data_agent_baseline.pipeline.kg_path_planner import (
    QueryNode,
    QueryPath,
    QueryPlan,
    build_adjacency,
    build_query_path,
    find_shortest_path,
)


class KGPlanningMixin:
    """KG-based path planning methods."""

    def _kg_path_plan_grounding(
        self,
        question: str,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
        db_path: Path | None = None,
        kg: KnowledgeGraph | None = None,
        kg_query: KGQueryService | None = None,
    ) -> str:
        """LLM picks nodes from property graph → validate → format as grounding for SQL LLM."""
        if not db_path or not kg:
            return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        # --- Step 1: User intent (primary) + domain anchors (supporting context) ---
        anchor_text = self._extract_domain_anchors(question, knowledge_text, db_path=db_path)
        # Use lightweight table overview for intent detection (only needs table/column awareness)
        intent_schema = kg_query.table_overview() if kg_query else kg_context
        user_intent = self._detect_user_intent_only(question, kg_context=intent_schema, anchor_text=anchor_text)

        # Deterministic answer_shape correction based on question keywords
        _q_lower = question.lower()
        _has_plural_hint = bool(re.search(r'\b(?:each|every|per|all the|for each)\b', _q_lower))
        if user_intent and not _has_plural_hint:
            if re.search(r'\b(?:how many|what is the (?:total|average|number|percentage|ratio|count))\b', _q_lower):
                if "Answer shape: list" in user_intent or "Answer shape: grouped_table" in user_intent:
                    user_intent = re.sub(r'Answer shape: \w+', 'Answer shape: single_value', user_intent)
        if user_intent:
            if re.search(r'\b(?:list all|list the|what are the|which are the|name all|give me all)\b', _q_lower):
                if "Answer shape: single_value" in user_intent:
                    user_intent = re.sub(r'Answer shape: \w+', 'Answer shape: list', user_intent)

        # --- Step 1a: Resolve Columns needed from KG (single authoritative resolver) ---
        if user_intent and kg:
            resolved_cols = self._resolve_columns_from_kg(question, user_intent, kg, anchor_text)
            if resolved_cols:
                user_intent += f"\nColumns needed: {', '.join(resolved_cols)}"

        if user_intent:
            self._log("user_intent", user_intent)

        # --- Step 1c: Early formula extraction for multi-table ratio hints ---
        early_formula = self._extract_domain_formula(question, anchor_text)
        if early_formula:
            formula_tables = set(re.findall(r'\b([a-z]\w*)\.[A-Z]\w*', early_formula))
            if not formula_tables:
                formula_tables = set(re.findall(r'\b([a-z]\w*)\.\w+', early_formula))
            kg_table_names = {t.name.lower() for t in kg.tables}
            # Match formula table refs to actual KG tables (handle singular/plural)
            matched_tables: set[str] = set()
            for ft in formula_tables:
                if ft in kg_table_names:
                    matched_tables.add(ft)
                elif ft + "s" in kg_table_names:
                    matched_tables.add(ft + "s")
                elif ft.rstrip("s") in kg_table_names:
                    matched_tables.add(ft.rstrip("s"))
            if len(matched_tables) >= 2:
                user_intent = (user_intent or "") + (
                    f"\n\nFORMULA HINT: The domain defines this metric as: {early_formula}\n"
                    f"This formula references tables: {', '.join(sorted(matched_tables))}. "
                    f"Include filter_conditions for EACH table so the ratio can be computed independently per table."
                )

        # --- Step 1d: Use user_intent Operation as authoritative computation_type ---
        _det_comp_type = ""
        for _line in user_intent.split("\n"):
            if "Operation:" in _line:
                _op = _line.split("Operation:")[1].strip().lower()
                if _op in ("count", "sum", "avg", "min_max", "ratio", "percentage", "count_distinct"):
                    _det_comp_type = _op
                break

        # ===================================================================
        # PHASE 1: PICK + VALIDATE (one LLM call, then deterministic checks)
        # ===================================================================
        _t_pick = time.monotonic()
        picked = self._pick_graph_nodes(question, kg, anchor_text, user_intent)
        self._log("pick_nodes_time", f"{time.monotonic() - _t_pick:.1f}s")
        if not picked:
            # LLM pick failed — try deterministic graph query before full fallback
            det_picked = self._deterministic_kg_query(question, kg, user_intent, db_path=db_path)
            if det_picked:
                self._log("kg_det_pick", json.dumps(det_picked, default=str))
                picked = det_picked
            else:
                self._log("kg_path", "Node picking failed, falling back")
                return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        if _det_comp_type:
            picked["computation_type"] = _det_comp_type

        self._log("kg_picked", json.dumps(picked, default=str))

        # Validate picks against actual schema
        output_nodes, filter_nodes, errors = self._validate_picked_nodes(picked, kg, db_path, question)
        if errors:
            self._log("kg_validation_errors", "; ".join(errors))
        if not output_nodes:
            # For aggregate operations (count/avg/sum), we can infer the output from the
            # computation type + filter tables — no need to fall back entirely.
            _comp_type = picked.get("computation_type", "")
            if _comp_type in ("count", "avg", "sum", "count_distinct") and filter_nodes:
                # Find a table connected to the filter table to count/aggregate
                filter_table = filter_nodes[0].table
                # Look for tables that have FK to filter_table (the "many" side)
                if kg and kg.graph:
                    for edge in kg.graph.fk_edges:
                        src_col = kg.graph.columns.get(edge.src)
                        dst_col = kg.graph.columns.get(edge.dst)
                        if src_col and dst_col:
                            if dst_col.table_id == filter_table and src_col.table_id != filter_table:
                                # src table references filter table — use src's PK as count target
                                many_table = src_col.table_id
                                ts = kg.get_table(many_table)
                                if ts and ts.columns:
                                    output_nodes.append(QueryNode(
                                        table=many_table, column=ts.columns[0].name, role="output",
                                    ))
                                    break
                            elif src_col.table_id == filter_table and dst_col.table_id != filter_table:
                                many_table = dst_col.table_id
                                ts = kg.get_table(many_table)
                                if ts and ts.columns:
                                    output_nodes.append(QueryNode(
                                        table=many_table, column=ts.columns[0].name, role="output",
                                    ))
                                    break
            if not output_nodes:
                self._log("kg_path", "No valid output nodes after validation, falling back")
                return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        # Remove spurious filters where value matches a table name that the filter is ON
        # (entity self-reference: "members" interpreted as position='Member' on the member table)
        _table_names_lower = {t.name.lower() for t in kg.tables} if kg else set()
        _quoted_in_q = {v.lower() for tup in re.findall(r"'([^']+)'|\"([^\"]+)\"", question) for v in tup if v}
        _cleaned_filters: list[QueryNode] = []
        for fn in filter_nodes:
            if fn.column.startswith("_expr:"):
                _cleaned_filters.append(fn)
                continue
            val_lower = str(fn.value).lower().rstrip("s")
            if (val_lower == fn.table.lower().rstrip("s")
                    and str(fn.value).lower() not in _quoted_in_q
                    and fn.operator == "="):
                self._log("kg_drop_filter", f"Dropped {fn.table}.{fn.column}={fn.value} (self-referential entity filter)")
                continue
            _cleaned_filters.append(fn)
        filter_nodes = _cleaned_filters

        # Override output columns with KG-resolved Columns needed (authoritative)
        _cols_overridden = False
        _picker_output = list(output_nodes)  # preserve LLM picker output before override
        _cols_needed_m = re.search(r'Columns needed:\s*(.+)', user_intent) if user_intent else None
        if _cols_needed_m:
            _resolved_output: list[QueryNode] = []
            for ref in _cols_needed_m.group(1).split(","):
                ref = ref.strip()
                if "." in ref:
                    _tbl, _col = ref.split(".", 1)
                    _t = kg.get_table(_tbl)
                    if _t:
                        _actual_col = next((c.name for c in _t.columns if c.name.lower() == _col.lower()), None)
                        if _actual_col:
                            _resolved_output.append(QueryNode(table=_t.name, column=_actual_col, role="output"))
            if _resolved_output:
                # Prefer picker's choice when it has an equivalent column on the filtered table
                # (e.g. picker chose posts.LastEditorDisplayName, intent says postHistory.UserDisplayName)
                _filter_tbls = {n.table.lower() for n in filter_nodes}
                _upgraded: list[QueryNode] = []
                for rn in _resolved_output:
                    if rn.table.lower() not in _filter_tbls:
                        _rn_split = re.sub(r'([a-z])([A-Z])', r'\1_\2', rn.column).lower()
                        _rn_words = set(re.findall(r'[a-z]+', _rn_split))
                        if "name" in _rn_words or "display" in _rn_words:
                            # Check if picker has equivalent on filtered table
                            _picker_equiv = None
                            for pn in _picker_output:
                                if pn.table.lower() in _filter_tbls:
                                    _pn_split = re.sub(r'([a-z])([A-Z])', r'\1_\2', pn.column).lower()
                                    _pn_words = set(re.findall(r'[a-z]+', _pn_split))
                                    if ("name" in _pn_words or "display" in _pn_words) and _rn_words & _pn_words:
                                        _picker_equiv = pn
                                        break
                            if _picker_equiv:
                                _upgraded.append(_picker_equiv)
                                continue
                    _upgraded.append(rn)
                _resolved_output = _upgraded
                # Merge picker columns that complement intent columns (e.g. last_name for "full name")
                _resolved_ids = {f"{n.table}.{n.column}".lower() for n in _resolved_output}
                _resolved_tables = {n.table.lower() for n in _resolved_output}
                _resolved_suffixes = set()
                for n in _resolved_output:
                    parts = re.split(r'[_.]', n.column.lower())
                    if len(parts) >= 2:
                        _resolved_suffixes.add(parts[-1])
                for pn in _picker_output:
                    pn_id = f"{pn.table}.{pn.column}".lower()
                    if pn_id not in _resolved_ids and pn.table.lower() in _resolved_tables:
                        pn_parts = re.split(r'[_.]', pn.column.lower())
                        pn_suffix = pn_parts[-1] if pn_parts else ""
                        if pn_suffix in _resolved_suffixes:
                            _resolved_output.append(pn)
                            _resolved_ids.add(pn_id)
                # Apply NULL-column FK replacement to resolved output too
                if db_path and db_path.exists() and kg:
                    _null_replacements: dict[int, QueryNode] = {}
                    try:
                        _nc = sqlite3.connect(str(db_path), timeout=5)
                        for _ri, _rn in enumerate(_resolved_output):
                            _total = _nc.execute(f'SELECT COUNT(*) FROM "{_rn.table}"').fetchone()[0]
                            if not _total:
                                continue
                            _nulls = _nc.execute(
                                f'SELECT COUNT(*) FROM "{_rn.table}" WHERE "{_rn.column}" IS NULL'
                            ).fetchone()[0]
                            if _nulls / _total < 0.9:
                                continue
                            # Check if the validated output has a replacement for this column
                            for _vn in output_nodes:
                                if _vn.table.lower() != _rn.table.lower() or _vn.column.lower() != _rn.column.lower():
                                    _null_replacements[_ri] = _vn
                                    break
                        _nc.close()
                    except Exception:
                        pass
                    for _ri, _repl in _null_replacements.items():
                        _resolved_output[_ri] = _repl

                output_nodes = _resolved_output
                _cols_overridden = True

        # Remove filter columns from output (they're criteria, not answer values)
        filter_col_ids = {f"{n.table}.{n.column}".lower() for n in filter_nodes if not n.column.startswith("_expr:")}
        output_nodes = [n for n in output_nodes if f"{n.table}.{n.column}".lower() not in filter_col_ids]
        if not output_nodes:
            output_nodes, _, _ = self._validate_picked_nodes(picked, kg, db_path, question)

        # Disambiguate output: prefer entity-of-interest table (only when single entity)
        _entity_pref_match = re.search(r'Entity of interest:\s*(.+?)(?:\s*\(|$)', user_intent) if user_intent else None
        _entity_tables: list[str] = []
        if _entity_pref_match:
            _entity_raw = _entity_pref_match.group(1).strip()
            _entity_tables = [w.strip() for w in re.split(r'\s+and\s+|,\s*', _entity_raw) if w.strip()]
        _pref_schema = None
        if len(_entity_tables) == 1 and kg:
            _pref_table = _entity_tables[0]
            _pref_schema = kg.get_table(_pref_table)
        if _pref_schema and kg:
            _pref_col_names = {c.name.lower() for c in _pref_schema.columns}
            _new_output = []
            for node in output_nodes:
                if node.table.lower() != _pref_table.lower() and node.column.lower() in _pref_col_names:
                    if self._db_check(db_path, _pref_table, node.column):
                        _new_output.append(QueryNode(table=_pref_table, column=node.column, role="output"))
                    else:
                        alt_found = False
                        for t in kg.tables:
                            if t.name.lower() in (_pref_table.lower(), node.table.lower()):
                                continue
                            if any(c.name.lower() == node.column.lower() for c in t.columns):
                                if self._db_check(db_path, t.name, node.column):
                                    _new_output.append(QueryNode(table=t.name, column=node.column, role="output"))
                                    alt_found = True
                                    break
                        if not alt_found:
                            _new_output.append(node)
                else:
                    _new_output.append(node)
            output_nodes = _new_output

        # Sanity check: detect computation_type mismatch + missing entities
        sanity_issues, entity_col_map = self._sanity_check_picks(
            question, picked, output_nodes, filter_nodes, user_intent, kg, anchor_text, db_path,
        )
        if sanity_issues:
            self._log("kg_sanity", sanity_issues)
            # Fix computation_type deterministically
            comp_type_fix = re.search(r'Change computation_type to "(\w+)"', sanity_issues)
            if comp_type_fix:
                picked["computation_type"] = comp_type_fix.group(1)
                sanity_issues = re.sub(
                    r'Intent says operation is "[^"]+" but computation_type is "[^"]+"\. '
                    r'Change computation_type to "\w+"\.\n?',
                    "", sanity_issues,
                ).strip()
            # Inject missing entity filters deterministically (from DB probe)
            if entity_col_map:
                existing_filter_vals = {str(n.value).lower() for n in filter_nodes}
                for entity_val, tbl_col in entity_col_map.items():
                    if entity_val.lower() in existing_filter_vals:
                        continue
                    if "." in tbl_col:
                        tbl, col = tbl_col.split(".", 1)
                        filter_nodes.append(QueryNode(
                            table=tbl, column=col, role="filter",
                            operator="LIKE", value=f"%{entity_val}%",
                        ))
                        self._log("kg_inject_filter", f"Added {tbl}.{col} LIKE '%{entity_val}%' (from DB probe)")

            # Doc-based injection for entities found in documents
            if hasattr(self, '_doc_paths') and self._doc_paths and db_path:
                flagged = re.findall(r'The question mentions "([^"]+)" but it is not', sanity_issues)
                current_filter_text = " ".join(str(n.value).lower() for n in filter_nodes)
                for entity_val in flagged:
                    if entity_val.lower() in current_filter_text:
                        continue
                    entity_words = set(entity_val.lower().split())
                    injected = False
                    for doc_path in self._doc_paths:
                        try:
                            doc_text = doc_path.read_text(encoding="utf-8", errors="replace")
                        except OSError:
                            continue
                        paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
                        best_para = ""
                        best_score = 0
                        for para in paragraphs:
                            para_lower = para.lower()
                            score = sum(1 for w in entity_words if w in para_lower)
                            if score > best_score:
                                best_score = score
                                best_para = para
                        if best_score < len(entity_words) - 1 or not best_para:
                            continue
                        candidate_ids = re.findall(r'\b(\d+)\b', best_para)
                        if not candidate_ids:
                            continue
                        entity_table = output_nodes[0].table if output_nodes else ""
                        doc_stem = doc_path.stem.lower()
                        fk_col = None
                        if kg:
                            for t in kg.tables:
                                if t.name == entity_table:
                                    for c in t.columns:
                                        if doc_stem in c.name.lower() and "id" in c.name.lower():
                                            fk_col = c.name
                                            break
                                    break
                        if not fk_col:
                            continue
                        try:
                            conn = sqlite3.connect(str(db_path), timeout=5)
                            for cid in candidate_ids:
                                row = conn.execute(
                                    f'SELECT 1 FROM "{entity_table}" WHERE "{fk_col}" = ? LIMIT 1',
                                    (int(cid),),
                                ).fetchone()
                                if row:
                                    filter_nodes.append(QueryNode(
                                        table=entity_table, column=fk_col, role="filter",
                                        operator="=", value=cid,
                                    ))
                                    self._log("kg_inject_filter", f"Added {entity_table}.{fk_col} = {cid} (from doc)")
                                    injected = True
                                    break
                            conn.close()
                        except Exception:
                            pass
                        if injected:
                            break

            # Knowledge-based keyword → value injection
            if anchor_text and db_path:
                pop_match = re.search(r'population filter is "([^"]+)"', sanity_issues)
                if pop_match:
                    pop_words = set(re.findall(r'[a-z]+', pop_match.group(1).lower()))
                    existing_filter_vals = {str(n.value).lower() for n in filter_nodes}
                    for line in anchor_text.split("\n"):
                        field_match = re.match(r"^-\s+(\w[\w\-]*)\s*(?:\([^)]*\))?\s*:", line)
                        if not field_match:
                            continue
                        field_name = field_match.group(1)
                        found = False
                        for m in re.finditer(r"'([^']+)'\s+for\s+(\w+)", line):
                            val, keyword = m.group(1), m.group(2).lower()
                            if keyword not in pop_words or val.lower() in existing_filter_vals:
                                continue
                            try:
                                conn = sqlite3.connect(str(db_path), timeout=5)
                                field_lower = field_name.lower()
                                best_match: tuple[str, str, int] | None = None
                                for t in kg.tables:
                                    for c in t.columns:
                                        if c.name.lower() != field_lower:
                                            continue
                                        row = conn.execute(
                                            f'SELECT 1 FROM "{t.name}" WHERE "{c.name}" = ? LIMIT 1',
                                            (val,),
                                        ).fetchone()
                                        if row:
                                            row_count = t.row_count or 0
                                            if best_match is None or row_count > best_match[2]:
                                                best_match = (t.name, c.name, row_count)
                                if best_match:
                                    filter_nodes.append(QueryNode(
                                        table=best_match[0], column=best_match[1], role="filter",
                                        operator="=", value=val,
                                    ))
                                    self._log("kg_inject_filter", f"Added {best_match[0]}.{best_match[1]} = '{val}' (from knowledge)")
                                    found = True
                                conn.close()
                            except Exception:
                                pass
                            if found:
                                break
                        if found:
                            break

        # For ratio/percentage: if output is on a dimension table but entity is a fact table,
        # move output to entity's PK so the ratio denominator uses the correct population
        _comp = picked.get("computation_type", "")
        if _comp in ("percentage", "ratio") and _entity_tables and kg and not _cols_overridden:
            _output_tables = {n.table.lower() for n in output_nodes}
            for _et in _entity_tables:
                _et_schema = kg.get_table(_et)
                if not _et_schema:
                    continue
                if _et.lower() in _output_tables:
                    break
                # Entity table not in output — check if it's larger (fact table)
                _out_schema = kg.get_table(output_nodes[0].table) if output_nodes else None
                if _out_schema and _et_schema.row_count and _out_schema.row_count:
                    if _et_schema.row_count > _out_schema.row_count:
                        _pk_col = _et_schema.columns[0].name if _et_schema.columns else "Id"
                        output_nodes = [QueryNode(table=_et_schema.name, column=_pk_col, role="output")]
                        self._log("kg_ratio_rebase", f"Moved output to {_et_schema.name}.{_pk_col} (fact table for ratio base)")
                        break

        # Ensure entity_of_interest is in path for aggregate queries
        # Skip when authoritative columns were resolved — they're already correct
        if _comp in ("count", "sum", "avg", "count_distinct") and _entity_tables and kg and not _cols_overridden:
            _all_tables = {n.table.lower() for n in output_nodes} | {n.table.lower() for n in filter_nodes}
            _eoi_table = next((t for t in _entity_tables if t.lower() not in _all_tables), None)
            if _eoi_table:
                _eoi_schema = kg.get_table(_eoi_table)
                if _eoi_schema and _eoi_schema.columns:
                    _eoi_col = _eoi_schema.columns[0].name
                    output_nodes.append(QueryNode(table=_eoi_table, column=_eoi_col, role="output"))
                    self._log("kg_inject_eoi", f"Added {_eoi_table}.{_eoi_col} to ensure path reaches entity")


        # Prefer normalized FK display columns over denormalized copies.
        # Denormalized columns (e.g. postHistory.UserDisplayName) can be stale/empty;
        # the FK path (postHistory.UserId → users.DisplayName) is always authoritative.
        # Exception: skip substitution when the column is a denormalized name on a filtered table
        # AND is not itself a filter column (co-located output avoids unnecessary join).
        _filter_tables_lower = {n.table.lower() for n in filter_nodes}
        _filter_col_ids_lower = {f"{n.table}.{n.column}".lower() for n in filter_nodes if not n.column.startswith("_expr:")}
        if kg and kg.graph:
            _new_output: list[QueryNode] = []
            for node in output_nodes:
                _substituted = False
                _col_lower = node.column.lower()
                _node_id_lower = f"{node.table}.{node.column}".lower()
                # Only attempt for columns that look like denormalized names
                # Skip if on filtered table but NOT itself a filter column (avoids unnecessary join)
                _skip_subst = (
                    node.table.lower() in _filter_tables_lower
                    and _node_id_lower not in _filter_col_ids_lower
                )
                if ("name" in _col_lower or "display" in _col_lower) and not _skip_subst:
                    _col_split = re.sub(r'([a-z])([A-Z])', r'\1_\2', node.column).lower()
                    _col_name_words = set(re.findall(r'[a-z]+', _col_split))
                    # Strategy 1: check fk_display_map
                    for fk_col_id, display_col_id in kg.graph.fk_display_map.items():
                        if not fk_col_id.lower().startswith(f"{node.table.lower()}."):
                            continue
                        if "." not in display_col_id:
                            continue
                        _disp_tbl, _disp_col = display_col_id.split(".", 1)
                        if _disp_tbl.lower() == node.table.lower():
                            continue
                        _disp_split = re.sub(r'([a-z])([A-Z])', r'\1_\2', _disp_col).lower()
                        _disp_words = set(re.findall(r'[a-z]+', _disp_split))
                        if _col_name_words & _disp_words:
                            _disp_schema = kg.get_table(_disp_tbl)
                            if _disp_schema:
                                _new_output.append(QueryNode(table=_disp_tbl, column=_disp_col, role="output"))
                                self._log("kg_fk_display", f"Replaced {node.table}.{node.column} with {display_col_id} (FK display map)")
                                _substituted = True
                                break
                    # Strategy 2: check FK edges directly for a target table with label column
                    if not _substituted:
                        for edge in kg.graph.fk_edges:
                            src_col = kg.graph.columns.get(edge.src)
                            if not src_col or src_col.table_id.lower() != node.table.lower():
                                continue
                            dst_col = kg.graph.columns.get(edge.dst)
                            if not dst_col or dst_col.table_id.lower() == node.table.lower():
                                continue
                            # FK column must share a word with the output column
                            # (e.g. UserId shares "user" with UserDisplayName)
                            _fk_split = re.sub(r'([a-z])([A-Z])', r'\1_\2', src_col.name).lower()
                            _fk_words = set(re.findall(r'[a-z]+', _fk_split)) - {"id"}
                            if not (_fk_words & _col_name_words):
                                continue
                            # Check if target table has a label column with name overlap
                            _target_tbl = kg.get_table(dst_col.table_id)
                            if not _target_tbl:
                                continue
                            for tc in _target_tbl.columns:
                                _tc_split = re.sub(r'([a-z])([A-Z])', r'\1_\2', tc.name).lower()
                                _tc_words = set(re.findall(r'[a-z]+', _tc_split))
                                if _col_name_words & _tc_words and "name" in _tc_words:
                                    _new_output.append(QueryNode(table=_target_tbl.name, column=tc.name, role="output"))
                                    self._log("kg_fk_display", f"Replaced {node.table}.{node.column} with {_target_tbl.name}.{tc.name} (FK to dimension)")
                                    _substituted = True
                                    break
                            if _substituted:
                                break
                if not _substituted:
                    _new_output.append(node)
            # Deduplicate: same table.column
            _seen_ids: set[str] = set()
            _deduped: list[QueryNode] = []
            for n in _new_output:
                _nid = f"{n.table}.{n.column}"
                if _nid not in _seen_ids:
                    _seen_ids.add(_nid)
                    _deduped.append(n)
            # Semantic dedup: if a "name" column on a non-filtered table duplicates
            # a "name" column already on the filtered table, drop the non-filtered one
            _filtered_name_words: set[str] = set()
            for n in _deduped:
                if n.table.lower() in _filter_tables_lower:
                    _ns = re.sub(r'([a-z])([A-Z])', r'\1_\2', n.column).lower()
                    _nw = set(re.findall(r'[a-z]+', _ns))
                    if "name" in _nw or "display" in _nw:
                        _filtered_name_words.update(_nw)
            if _filtered_name_words:
                _final: list[QueryNode] = []
                for n in _deduped:
                    if n.table.lower() not in _filter_tables_lower:
                        _ns2 = re.sub(r'([a-z])([A-Z])', r'\1_\2', n.column).lower()
                        _nw2 = set(re.findall(r'[a-z]+', _ns2))
                        if ("name" in _nw2 or "display" in _nw2) and _nw2 & _filtered_name_words:
                            continue
                    _final.append(n)
                _deduped = _final
            output_nodes = _deduped

        self._log("kg_output_nodes", ", ".join(f"{n.table}.{n.column}" for n in output_nodes))
        self._log("kg_filter_nodes", ", ".join(
            f"{n.table}.{n.column}{n.operator}{n.value}" for n in filter_nodes
        ))

        # ===================================================================
        # PHASE 2: TRANSFORM (deterministic, no plausibility probes)
        # Each transform fixes a known structural pattern — they don't conflict
        # because each operates on a different aspect (format, domain, structure).
        # ===================================================================

        # 2a. Fix value formats (case, prefix, LIKE patterns)
        filter_nodes = self._probe_filter_values(filter_nodes, db_path, kg, question)

        # 2b. Domain column fixes (swap column based on knowledge definitions)
        _pre_domain_cols = {f"{n.table}.{n.column}" for n in filter_nodes}
        filter_nodes = self._apply_domain_column_fixes(question, filter_nodes, kg, anchor_text, db_path)
        _post_domain_cols = {f"{n.table}.{n.column}" for n in filter_nodes}
        domain_locked_columns = _post_domain_cols - _pre_domain_cols

        # 2c. Knowledge-defined value mappings (canonical forms)
        filter_nodes = self._apply_knowledge_value_mappings(filter_nodes, anchor_text, kg, db_path)

        # 2d. Per-unit computed expressions (price/quantity patterns)
        filter_nodes = self._detect_per_unit_filters(question, filter_nodes, kg)

        # 2e. Intent signal enforcement (group HAVING, temporal, ordinal)
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

        filter_nodes, order_nodes = self._enforce_intent_signals(
            user_intent, filter_nodes, order_nodes, output_nodes, picked, kg, db_path,
        )

        # For superlative questions (rank/min_max), the ORDER BY column is the criterion,
        # not an output. Remove it from output_nodes to avoid extra columns in the answer.
        _intent_op = ""
        if user_intent:
            _op_match = re.search(r'Operation:\s*(\w+)', user_intent)
            if _op_match:
                _intent_op = _op_match.group(1).lower()
        _comp_type_for_resolve = picked.get("computation_type", "simple_lookup")
        if _intent_op in ("rank", "min_max") and order_nodes and len(output_nodes) > 1:
            order_cols = {(n.table.lower(), n.column.lower()) for n in order_nodes}
            _trimmed = [n for n in output_nodes if (n.table.lower(), n.column.lower()) not in order_cols]
            if _trimmed:
                output_nodes = _trimmed
        if _comp_type_for_resolve not in ("avg", "count", "count_distinct", "sum", "ratio", "percentage"):
            output_nodes = self._resolve_fk_output_columns(output_nodes, kg, db_path)

        # List+lookup → PK when question doesn't name a specific attribute
        if output_nodes and kg and user_intent:
            _shape = ""
            _operation = ""
            for _line in user_intent.split("\n"):
                if "Answer shape:" in _line:
                    _shape = _line.split("Answer shape:")[1].strip().lower()
                elif "Operation:" in _line:
                    _operation = _line.split("Operation:")[1].strip().lower()
            _q_lower_pk = question.lower().strip()
            _is_enumeration = (
                _q_lower_pk.startswith("list ")
                or _q_lower_pk.startswith("give ")
                or _q_lower_pk.startswith("show ")
                or _q_lower_pk.startswith("find ")
                or "list all" in _q_lower_pk
                or "list the" in _q_lower_pk
            )
            if _shape == "list" and _operation == "lookup" and _is_enumeration:
                entity_table = output_nodes[0].table
                table_schema = kg.get_table(entity_table)
                if table_schema:
                    q_tokens_pk = set(re.findall(r'[a-z]{3,}', question.lower()))
                    pk_names = {c.name.lower() for c in table_schema.columns if c.is_pk}
                    if not pk_names and table_schema.primary_keys:
                        pk_names = {p.lower() for p in table_schema.primary_keys}
                    non_pk_cols = [c for c in table_schema.columns if c.name.lower() not in pk_names]
                    question_names_attribute = False
                    for col in non_pk_cols:
                        col_lower = col.name.lower()
                        parts = re.findall(r'[A-Z][a-z]+|[a-z]+', col.name)
                        if len(parts) < 2 and "_" in col.name:
                            parts = [p for p in col_lower.split("_") if p]
                        col_words = {p.lower() for p in parts} | {col_lower}
                        col_words -= {"id", "the", "all"}
                        if col_words & q_tokens_pk:
                            question_names_attribute = True
                            break
                    if not question_names_attribute:
                        pk_col = None
                        if table_schema.primary_keys:
                            pk_col = table_schema.primary_keys[0]
                        else:
                            for col in table_schema.columns:
                                if col.name.lower().endswith("_id") or col.name.lower() == "id":
                                    pk_col = col.name
                                    break
                        if not pk_col:
                            pk_col = table_schema.columns[0].name
                        output_nodes = [QueryNode(table=entity_table, column=pk_col, role="output")]
                        self._log("kg_list_pk_override", f"Overrode to {entity_table}.{pk_col}")

        # ===================================================================
        # PHASE 3: VALIDATE-AND-FIX LOOP (probe DB, diagnose, fix)
        # One unified loop replaces scattered repick + adaptive + weak-filter.
        # Exit condition: filter set is plausible against the DB.
        # ===================================================================

        # Drop filters on unreachable tables before probing
        _output_tables = {n.table for n in output_nodes}
        _filter_tables = {n.table for n in filter_nodes if not n.column.startswith("_expr:")}
        _foreign_filter_tables = _filter_tables - _output_tables
        if _foreign_filter_tables and kg:
            _adj = build_adjacency(kg)
            _anchor_table = output_nodes[0].table if output_nodes else ""
            _unreachable: set[str] = set()
            for ft in _foreign_filter_tables:
                if find_shortest_path(_adj, _anchor_table, ft) is None:
                    _unreachable.add(ft)
            if _unreachable:
                self._log("kg_path_unreachable",
                          f"Tables {_unreachable} have no FK path to '{_anchor_table}'")
                filter_nodes = [
                    n for n in filter_nodes
                    if n.table not in _unreachable or n.column.startswith("_expr:")
                ]

        # Run rules engine (threshold inference, ratio structure, decomposition)
        from data_agent_baseline.pipeline.rules_engine import run_rules_engine
        comp_type = picked.get("computation_type", "simple_lookup")
        engine_out = run_rules_engine(
            question=question,
            user_intent=user_intent,
            comp_type=comp_type,
            filter_nodes=filter_nodes,
            output_nodes=output_nodes,
            kg=kg,
            db_path=db_path,
            knowledge_text=knowledge_text,
            anchor_text=anchor_text,
            model_call=self._model_call_with_retry,
            domain_locked_columns=domain_locked_columns,
        )
        filter_nodes = engine_out.filter_nodes
        for tag, msg in engine_out.log_entries:
            self._log(tag, msg)
        if engine_out.decomposition:
            self._decomposition_steps = engine_out.decomposition

        # --- Step 4: Graph Traversal (BFS through KG edges) ---
        path = build_query_path(output_nodes, filter_nodes, kg, order_nodes=order_nodes)
        if not path:
            self._log("kg_path", "No path found, falling back")
            return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        # Inject route tables missing from path (needed for computation but no nodes reference them)
        _route_tables = getattr(self, '_route_tables', None) or []
        _path_tables = set(path.tables_in_path)
        _missing_route = [t for t in _route_tables if t not in _path_tables]
        if _missing_route and kg and kg.graph:
            _derived = picked.get("derived_logic", "") or ""
            _extra_tables = list(path.tables_in_path)
            _extra_edges = list(path.edges)
            for mt in _missing_route:
                if mt.lower() not in _derived.lower():
                    continue
                for edge in kg.graph.fk_edges:
                    src_col = kg.graph.columns.get(edge.src)
                    dst_col = kg.graph.columns.get(edge.dst)
                    if not src_col or not dst_col:
                        continue
                    if src_col.table_id == mt and dst_col.table_id in _path_tables:
                        from data_agent_baseline.pipeline.kg_path_planner import GraphEdge
                        _extra_edges.append(GraphEdge(
                            src_table=src_col.table_id, src_column=src_col.name,
                            dst_table=dst_col.table_id, dst_column=dst_col.name,
                            weight=edge.overlap_ratio,
                        ))
                        _extra_tables.append(mt)
                        break
                    elif dst_col.table_id == mt and src_col.table_id in _path_tables:
                        from data_agent_baseline.pipeline.kg_path_planner import GraphEdge
                        _extra_edges.append(GraphEdge(
                            src_table=src_col.table_id, src_column=src_col.name,
                            dst_table=dst_col.table_id, dst_column=dst_col.name,
                            weight=edge.overlap_ratio,
                        ))
                        _extra_tables.append(mt)
                        break
            if len(_extra_tables) > len(path.tables_in_path):
                path = QueryPath(
                    edges=tuple(_extra_edges),
                    output_nodes=path.output_nodes,
                    filter_nodes=path.filter_nodes,
                    tables_in_path=tuple(_extra_tables),
                )

        self._log("kg_path_edges", " → ".join(
            f"{e.src_table}.{e.src_column}={e.dst_table}.{e.dst_column}" for e in path.edges
        ) if path.edges else "(single table)")

        # --- Step 5: Format as grounding context for SQL LLM ---
        # Expose for post-execution validation (Layer 5)
        self._last_output_nodes = output_nodes
        self._last_comp_type = picked.get("computation_type", "simple_lookup")

        _eoi_match = re.search(r'Entity of interest:\s*(\w+)', user_intent or "")
        if _eoi_match:
            picked["_entity_of_interest"] = _eoi_match.group(1)

        # Fix comp_type vs answer shape conflict: "count" + "list" shape means
        # "produce a grouped list", not "aggregate into one number"
        # Exception: if filters target a single entity (= on ID column), keep the aggregation
        _shape_match = re.search(r'Answer shape:\s*(\w+)', user_intent or "")
        _grain_match = re.search(r'Grain:\s*(.+)', user_intent or "")
        if _shape_match and _shape_match.group(1).lower() == "list":
            _ct = picked.get("computation_type", "")
            if _ct in ("count", "sum", "avg"):
                _has_single_entity_filter = False
                for fc in picked.get("filter_conditions", []):
                    col_name = fc.get("column", "").lower()
                    if fc.get("operator") == "=" and ("_id" in col_name or col_name.endswith("id")):
                        _has_single_entity_filter = True
                        break
                if not _has_single_entity_filter:
                    picked["computation_type"] = "grouped_list"

        grounding = self._format_kg_plan_as_grounding(path, None, picked, "", kg=kg, db_path=db_path, user_intent=user_intent)

        # Inject rules engine directives into grounding
        # Suppress RATIO PATTERN when grounding already has an independent/cross-table ratio directive
        if engine_out.sql_directives:
            directives = engine_out.sql_directives
            if "CROSS-TABLE RATIO" in grounding or "INDEPENDENT RATIO" in grounding:
                directives = [d for d in directives if "RATIO PATTERN" not in d]
            if directives:
                grounding += "\n" + "\n".join(directives)

        # --- Step 5b: Positional (Nth) with PREFIX_N ID pattern → numeric ordering ---
        ordinal_match = re.search(
            r'\b(\d+)(?:st|nd|rd|th)\b', question.lower(),
        )
        if ordinal_match and db_path and db_path.exists():
            entity_table = output_nodes[0].table if output_nodes else ""
            if entity_table:
                try:
                    _pos_conn = sqlite3.connect(str(db_path), timeout=5)
                    # Find the primary ordering column (first TEXT column ending in _id or named id)
                    cols_info = _pos_conn.execute(f'PRAGMA table_info("{entity_table}")').fetchall()
                    id_col = None
                    for ci in cols_info:
                        cn = ci[1].lower()
                        if cn.endswith("_id") or cn == "id":
                            id_col = ci[1]
                            break
                    if id_col:
                        sample_vals = [
                            r[0] for r in _pos_conn.execute(
                                f'SELECT DISTINCT "{id_col}" FROM "{entity_table}" LIMIT 10'
                            ).fetchall() if r[0]
                        ]
                        # Detect PREFIX_N pattern (e.g. TR001_1, TR001_10)
                        prefix_n_count = sum(
                            1 for v in sample_vals
                            if re.match(r'^.+[_\-]\d+$', str(v))
                        )
                        if prefix_n_count >= len(sample_vals) * 0.8 and sample_vals:
                            sep = "_" if "_" in str(sample_vals[0]) else "-"
                            nth = ordinal_match.group(1)
                            # Check if there's a group-by column (partition)
                            partition_col = ""
                            for ci in cols_info:
                                cn = ci[1].lower()
                                if cn.endswith("_id") and cn != id_col.lower():
                                    partition_col = ci[1]
                                    break
                            if partition_col:
                                grounding += (
                                    f'\nNUMERIC ORDERING (MANDATORY — use this exact subquery pattern):\n'
                                    f'  The "{id_col}" column encodes position as PREFIX{sep}N (e.g. {sample_vals[0]}).\n'
                                    f'  To get the {nth}th item per group, use:\n'
                                    f'  WHERE "{id_col}" IN (SELECT "{id_col}" FROM '
                                    f'(SELECT "{id_col}", "{partition_col}", '
                                    f'ROW_NUMBER() OVER (PARTITION BY "{partition_col}" '
                                    f'ORDER BY CAST(SUBSTR("{id_col}", INSTR("{id_col}",\'{sep}\')+1) AS INTEGER)) AS rn '
                                    f'FROM "{entity_table}") WHERE rn = {nth})'
                                )
                            else:
                                grounding += (
                                    f'\nNUMERIC ORDERING (MANDATORY): The "{id_col}" column has format '
                                    f'PREFIX{sep}NUMBER (e.g. {sample_vals[0]}). Text sorting gives wrong '
                                    f'positional order (_10 before _2). In ROW_NUMBER() or ORDER BY, use: '
                                    f'ORDER BY CAST(SUBSTR("{id_col}", INSTR("{id_col}",\'{sep}\')+1) AS INTEGER)'
                                )
                    _pos_conn.close()
                except Exception:
                    pass

        # --- Step 5a: Surface domain formulas relevant to this question ---
        # Skip if independent ratio already produced the SQL pattern
        # Skip for percentage/count — those have proper CASE WHEN/COUNT patterns
        comp_type_lower = (picked.get("computation_type") or "").lower()
        if "INDEPENDENT RATIO" not in grounding and comp_type_lower not in ("percentage", "count"):
            domain_formula = self._extract_domain_formula(question, anchor_text)
            if domain_formula:
                sql_hint = self._formula_to_sql_hint(domain_formula, output_nodes, picked)
                if sql_hint:
                    grounding += f"\nDOMAIN FORMULA (MANDATORY — apply exactly): {sql_hint}"
                else:
                    grounding += f"\nDOMAIN FORMULA: {domain_formula}"

        self._log("kg_grounding", grounding)
        return grounding

    def _deterministic_kg_query(
        self,
        question: str,
        kg: KnowledgeGraph,
        user_intent: str,
        db_path: Path | None = None,
    ) -> dict[str, Any] | None:
        """Deterministic graph traversal: question phrases → DB probe → FK paths → pick.

        Generates all n-grams from the question, probes the actual DB for exact matches,
        then uses FK edges to verify connectivity. No stopwords, no LLM — the DB is the filter.

        Returns a picked dict (same format as LLM pick) when confident, else None.
        """
        g = kg.graph
        if not g or not db_path or not db_path.exists():
            return None

        q_lower = question.lower()

        # --- 1. Generate all n-grams from question ---
        tokens = [w.strip("?.,!\"'()") for w in question.split()]
        tokens_clean = [t for t in tokens if t and len(t) > 1]

        # --- 2. Probe DB for exact value matches (longest first) ---
        filters: list[dict[str, str]] = []
        filter_tables: set[str] = set()
        matched_spans: set[int] = set()

        # Collect TEXT columns to probe (skip IDs and numeric-only columns)
        text_cols: list[tuple[str, str]] = []  # (table, column)
        for table in kg.tables:
            for col in table.columns:
                cn = col.name.lower()
                if cn == "id" or cn.endswith("_id"):
                    continue
                if col.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", "NVARCHAR", ""):
                    text_cols.append((table.name, col.name))

        if not text_cols:
            return None

        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            for width in range(min(5, len(tokens_clean)), 0, -1):
                for i in range(len(tokens_clean) - width + 1):
                    span = set(range(i, i + width))
                    if span & matched_spans:
                        continue
                    phrase = " ".join(tokens_clean[i:i + width])
                    if len(phrase) < 3:
                        continue

                    # Probe each text column for this exact phrase
                    for tbl, col in text_cols:
                        try:
                            row = conn.execute(
                                f'SELECT "{col}" FROM "{tbl}" WHERE "{col}" = ? COLLATE NOCASE LIMIT 1',
                                (phrase,),
                            ).fetchone()
                        except Exception:
                            continue
                        if row:
                            actual_value = row[0]
                            filters.append({"column": f"{tbl}.{col}", "operator": "=", "value": actual_value})
                            filter_tables.add(tbl)
                            matched_spans |= span
                            break  # first table match wins for this phrase
            conn.close()
        except Exception:
            return None

        if not filters:
            return None

        # --- 3. Identify output table ---
        output_table = None
        for tname in (t.name for t in kg.tables):
            tname_lower = tname.lower()
            if (tname_lower in q_lower
                    or tname_lower + "s" in q_lower
                    or tname_lower + "es" in q_lower
                    or (tname_lower.endswith("s") and tname_lower[:-1] in q_lower)):
                if tname not in filter_tables:
                    output_table = tname
                    break
                elif output_table is None:
                    output_table = tname

        if not output_table:
            non_filter = [t for t in kg.tables if t.name not in filter_tables]
            if non_filter:
                output_table = max(non_filter, key=lambda t: t.row_count or 0).name
            elif kg.tables:
                output_table = max(kg.tables, key=lambda t: t.row_count or 0).name

        if not output_table:
            return None

        # --- 4. Determine output column from computation type ---
        output_schema = kg.get_table(output_table)
        if not output_schema:
            return None

        comp_type = "count"
        if user_intent:
            for line in user_intent.split("\n"):
                if "Operation:" in line:
                    op = line.split("Operation:")[1].strip().lower()
                    if op in ("count", "sum", "avg", "min_max", "ratio", "percentage", "count_distinct"):
                        comp_type = op
                    break

        select_columns: list[str] = []
        if comp_type in ("count", "count_distinct"):
            pk_col = next(
                (c.name for c in output_schema.columns if c.is_pk or c.name.lower() == "id"),
                output_schema.columns[0].name if output_schema.columns else None,
            )
            if pk_col:
                select_columns.append(f"{output_table}.{pk_col}")
        elif comp_type in ("sum", "avg"):
            for c in output_schema.columns:
                if c.sql_type.upper() in ("INTEGER", "REAL", "FLOAT", "NUMERIC", "DECIMAL"):
                    if not c.name.lower().endswith("_id") and c.name.lower() != "id":
                        select_columns.append(f"{output_table}.{c.name}")
                        break
        else:
            for c in output_schema.columns:
                cn = c.name.lower()
                if any(w in cn for w in ("name", "title", "label", "description")):
                    select_columns.append(f"{output_table}.{c.name}")
                    break
            if not select_columns and output_schema.columns:
                for c in output_schema.columns:
                    if c.name.lower() != "id" and not c.name.lower().endswith("_id"):
                        select_columns.append(f"{output_table}.{c.name}")
                        break

        if not select_columns:
            return None

        # --- 5. Verify FK connectivity ---
        reachable: set[str] = {output_table}
        frontier = [output_table]
        for _ in range(5):
            next_frontier = []
            for tbl in frontier:
                for edge in g.fk_edges:
                    src_col = g.columns.get(edge.src)
                    dst_col = g.columns.get(edge.dst)
                    if not src_col or not dst_col:
                        continue
                    if src_col.table_id == tbl and dst_col.table_id not in reachable:
                        reachable.add(dst_col.table_id)
                        next_frontier.append(dst_col.table_id)
                    elif dst_col.table_id == tbl and src_col.table_id not in reachable:
                        reachable.add(src_col.table_id)
                        next_frontier.append(src_col.table_id)
            frontier = next_frontier
            if not frontier:
                break

        if not filter_tables.issubset(reachable):
            return None

        # --- 6. Build result ---
        return {
            "what_user_wants": f"{comp_type} from {output_table} filtered by {', '.join(f['column'] + '=' + f['value'] for f in filters)}",
            "select_columns": select_columns,
            "filter_conditions": filters,
            "order_by": None,
            "computation_type": comp_type,
        }

    def _pick_graph_nodes(
        self,
        question: str,
        kg: KnowledgeGraph,
        anchor_text: str,
        user_intent: str,
    ) -> dict[str, Any] | None:
        """LLM picks columns via query-based graph traversal.

        Step 1: LLM sees table overview + relationships, decides which tables to explore.
        Step 2: LLM sees full column details for selected tables, makes final pick.
        """
        g = kg.graph
        q_lower = question.lower()

        # --- Build table overview (names, roles, row counts, FK links) ---
        overview_lines: list[str] = []
        for table in kg.tables:
            role = table.role or "table"
            col_names = [c.name for c in table.columns]
            overview_lines.append(
                f'  "{table.name}" [{role}, {table.row_count} rows]: columns={col_names}'
            )

        # FK relationships
        fk_lines: list[str] = []
        fk_targets: dict[str, tuple[str, str]] = {}
        if g:
            for edge in g.fk_edges:
                src_col = g.columns.get(edge.src)
                dst_col = g.columns.get(edge.dst)
                if src_col and dst_col:
                    fk_targets[edge.src] = (dst_col.table_id, dst_col.name)
                    fk_lines.append(
                        f'  "{src_col.table_id}"."{src_col.name}" → "{dst_col.table_id}"."{dst_col.name}"'
                    )

        overview = "TABLES:\n" + "\n".join(overview_lines)
        if fk_lines:
            overview += "\n\nRELATIONSHIPS:\n" + "\n".join(fk_lines)

        intent_section = f"\nUSER INTENT:\n{user_intent}" if user_intent else ""
        anchor_section = f"\nDOMAIN KNOWLEDGE:\n{anchor_text[:2000]}" if anchor_text else ""

        # --- Step 1: Route — which tables matter? ---
        route_prompt = f"""QUESTION: {question}
{intent_section}{anchor_section}

DATABASE OVERVIEW:
{overview}

Think step by step about what the question needs:
1. What entity/value does the question ask ABOUT? → which table has it?
2. What filters does the question mention? → which tables have those columns?
3. What metric (sum/count/value) is needed? → which table stores it?
4. If a qualifier word (like "approved", "active", "completed") matches a column name, that table likely holds the metric too.

Return JSON:
{{"tables": ["table1", "table2", ...], "reasoning": "one sentence explaining the data path"}}"""

        messages = [ModelMessage(role="user", content=route_prompt)]
        try:
            raw = self._model_call_with_retry(messages, thinking=False)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("tables"):
                selected_tables = parsed["tables"]
                self._log("kg_route", f"Tables: {selected_tables} — {parsed.get('reasoning', '')}")
                self._route_tables = selected_tables
            else:
                selected_tables = [t.name for t in kg.tables]
                self._route_tables = selected_tables
        except Exception:
            selected_tables = [t.name for t in kg.tables]

        # --- Build detailed view of selected tables ---
        detail_parts: list[str] = []
        for table in kg.tables:
            if table.name not in selected_tables:
                continue
            cols_desc: list[str] = []
            for col in table.columns:
                parts = [f'"{col.name}" ({col.sql_type})']
                if col.description:
                    parts.append(f"-- {col.description}")
                col_id = f"{table.name}.{col.name}"
                is_fk_col = col_id in fk_targets
                if is_fk_col:
                    ref_t, ref_c = fk_targets[col_id]
                    display_col = g.fk_display_map.get(col_id, "") if g else ""
                    if display_col:
                        parts.append(f'→ references "{ref_t}"."{ref_c}" (display: "{display_col}")')
                    else:
                        parts.append(f'→ references "{ref_t}"."{ref_c}"')
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
                    if col.sql_type.upper() in ("REAL", "FLOAT", "INTEGER", "INT", "NUMERIC"):
                        if col.name.lower() not in ("id",) and not col.name.lower().endswith("_id"):
                            if hasattr(table, 'col_stats') and col.name in table.col_stats:
                                s = table.col_stats[col.name]
                                if "min" in s and "max" in s:
                                    parts.append(f"range: [{s['min']}, {s['max']}]")
                cols_desc.append("  " + " ".join(parts))
            detail_parts.append(f'TABLE "{table.name}" ({table.row_count} rows):\n' + "\n".join(cols_desc))

        if fk_lines:
            detail_parts.append("JOIN RELATIONSHIPS:\n" + "\n".join(fk_lines))

        graph_detail = "\n\n".join(detail_parts)

        # Value pre-grounding
        value_section = ""
        if g:
            _stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
                     "to", "for", "and", "or", "by", "with", "from", "that", "this",
                     "it", "its", "be", "has", "had", "have", "do", "does", "did",
                     "how", "many", "much", "what", "which", "who", "where", "when",
                     "than", "more", "less", "most", "least", "not", "no", "all",
                     "each", "every", "any", "some", "their", "them", "they", "we",
                     "our", "he", "she", "his", "her", "among", "between", "into",
                     "about", "over", "under", "above", "below", "after", "before"}
            _q_tokens = [w.strip("?.,!\"'()") for w in question.split()]
            _q_tokens_lower = [t.lower() for t in _q_tokens if t]
            _candidates: list[str] = []
            for i in range(len(_q_tokens_lower) - 1):
                a, b = _q_tokens_lower[i], _q_tokens_lower[i + 1]
                if a not in _stop and b not in _stop and len(a) > 1 and len(b) > 1:
                    _candidates.append(f"{_q_tokens[i]} {_q_tokens[i + 1]}")
            for i, t in enumerate(_q_tokens_lower):
                if t not in _stop and len(t) > 2:
                    _candidates.append(_q_tokens[i])
            _seen_matches: set[str] = set()
            _value_lines: list[str] = []
            for phrase in _candidates:
                if len(_value_lines) >= 10:
                    break
                hits = g.find_value(phrase)
                for tbl, col, cnt in hits:
                    key = f"{tbl}.{col}={phrase.lower()}"
                    if key in _seen_matches:
                        continue
                    _seen_matches.add(key)
                    _value_lines.append(f'  "{phrase}" → {tbl}.{col}')
                    if len(_value_lines) >= 10:
                        break
            if _value_lines:
                value_section = (
                    "\nVALUE MATCHES (these values EXIST in the DB — use these columns for filters):\n"
                    + "\n".join(_value_lines)
                )

        # Concept map
        concept_section = ""
        q_words = set(question.lower().split())
        if hasattr(kg, 'concept_map') and kg.concept_map:
            relevant = [
                (k, v) for k, v in kg.concept_map.items()
                if k in q_lower or any(w in q_words for w in k.split())
            ]
            if relevant:
                cmap_lines = [f'  "{k}" → {v}' for k, v in relevant]
                concept_section = "\nCONCEPT MAP:\n" + "\n".join(cmap_lines)

        # Ontology
        ontology_section = ""
        if hasattr(kg, 'ontology') and kg.ontology:
            ont_lines = []
            for col_ref, entry in kg.ontology.items():
                vocab = entry.get("value_vocab")
                if vocab and isinstance(vocab, dict):
                    matched = any(
                        str(k).lower() in q_lower or str(v).lower() in q_lower
                        for k, v in vocab.items()
                    )
                    if not matched:
                        continue
                    purpose = entry.get("purpose", "")
                    purpose_prefix = f"({purpose}) " if purpose else ""
                    if len(vocab) <= 12:
                        vocab_str = ", ".join(f"{k}={v}" for k, v in vocab.items())
                    else:
                        vocab_str = ", ".join(f"{k}={v}" for k, v in list(vocab.items())[:10]) + ", ..."
                    ont_lines.append(f'  {col_ref}: {purpose_prefix}{{{vocab_str}}}')
                elif entry.get("hierarchy"):
                    h = entry["hierarchy"]
                    level = h.get("level", "")
                    if level.lower() in q_lower:
                        ont_lines.append(f'  {col_ref}: hierarchy level={level}')
            if ont_lines:
                ontology_section = (
                    "\nVALUE MEANINGS:\n" + "\n".join(ont_lines[:10])
                )

        # --- Step 2: Pick — select exact columns from the explored tables ---
        pick_prompt = f"""QUESTION: {question}
{intent_section}{anchor_section}

SELECTED TABLES (detailed schema):
{graph_detail[:5000]}
{concept_section}{ontology_section}{value_section}

GOAL: Pick the exact columns that produce the final answer.
Think: "What does the answer table look like?" The select_columns are what appears IN the answer. The filter_conditions narrow which rows.

RULES:
- select_columns = columns whose VALUES appear in the final answer CSV.
- filter_conditions = WHERE clause constraints (these do NOT appear in output).
- Use EXACT values from "values:" lists or DOMAIN KNOWLEDGE for filters.
- FK columns (marked →): JOIN to referenced table for human-readable names.
- DOMAIN KNOWLEDGE overrides intent when they conflict on column meaning.

Return ONLY JSON:
{{
  "what_user_wants": "one sentence: what the answer table contains",
  "select_columns": ["Table.Column", ...],
  "filter_conditions": [
    {{"column": "Table.Column", "operator": "= | > | < | >= | <= | LIKE | !=", "value": "..."}}
  ],
  "order_by": {{"column": "Table.Column", "direction": "ASC | DESC"}} or null,
  "computation_type": "simple_lookup | count | sum | avg | min_max | ratio | percentage | derived",
  "derived_logic": "natural language description of the computation (only if computation_type is 'derived')"
}}"""

        messages = [ModelMessage(role="user", content=pick_prompt)]
        try:
            raw = self._model_call_with_retry(messages, thinking=False)
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
        question: str = "",
    ) -> tuple[list[QueryNode], list[QueryNode], list[str]]:
        """Validate LLM's picks against the actual graph. Returns (output_nodes, filter_nodes, errors)."""
        errors: list[str] = []
        output_nodes: list[QueryNode] = []
        filter_nodes: list[QueryNode] = []

        g = kg.graph

        # Validate select_columns
        for col_ref in picked.get("select_columns", []):
            if "." not in col_ref:
                # Handle bare aggregate expressions like "AVG(bond_count)", "COUNT(*)"
                agg_bare = re.match(r'(COUNT|SUM|AVG|MIN|MAX)\(([^)]*)\)', col_ref, re.IGNORECASE)
                if agg_bare:
                    # This is a computed expression — not an error, just skip as output node.
                    # The computation_type + filter_nodes provide enough info for SQL generation.
                    continue
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
                    # Try to extract Table.Column from aggregate expressions like "SUM(table.col) AS alias"
                    agg_match = re.search(r'(\w+)\((\w+)\.(\w+)\)', col_ref)
                    if agg_match:
                        agg_table = agg_match.group(2)
                        agg_col = agg_match.group(3)
                        agg_id = f"{agg_table}.{agg_col}"
                        if g and agg_id in g.columns:
                            output_nodes.append(QueryNode(table=agg_table, column=agg_col, role="output"))
                        else:
                            agg_matched = self._fuzzy_match_column(agg_table, agg_col, kg)
                            if agg_matched:
                                output_nodes.append(QueryNode(table=agg_matched[0], column=agg_matched[1], role="output"))
                            else:
                                errors.append(f"Column not in graph: {col_ref}")
                    else:
                        errors.append(f"Column not in graph: {col_ref}")

        # Fix 1: Detect SUM/AVG on non-numeric columns — these should be filters, not metrics.
        # E.g. "total approved expenses" → SUM(expense.cost) WHERE approved='true', NOT SUM(approved)
        # Only check when there's exactly one output node (the aggregation target).
        # With multiple nodes, one is likely a GROUP BY column (non-numeric is expected).
        _comp = picked.get("computation_type", "")
        if _comp in ("sum", "avg") and len(output_nodes) == 1 and db_path and db_path.exists():
            _nodes_to_demote: list[int] = []
            try:
                _tc_conn = sqlite3.connect(str(db_path), timeout=5)
                for idx, node in enumerate(output_nodes):
                    # Try casting a sample to REAL — if all fail, the column isn't numeric
                    try:
                        numeric_check = _tc_conn.execute(
                            f'SELECT COUNT(*), SUM(CASE WHEN TYPEOF(CAST("{node.column}" AS REAL)) '
                            f'= \'real\' AND CAST("{node.column}" AS REAL) != 0.0 THEN 1 ELSE 0 END) '
                            f'FROM (SELECT "{node.column}" FROM "{node.table}" '
                            f'WHERE "{node.column}" IS NOT NULL LIMIT 20)',
                        ).fetchone()
                        total, numeric_count = numeric_check or (0, 0)
                        if total and total > 0 and (numeric_count or 0) / total < 0.5:
                            _nodes_to_demote.append(idx)
                            # Get distinct values for the error message
                            sample = _tc_conn.execute(
                                f'SELECT DISTINCT "{node.column}" FROM "{node.table}" '
                                f'WHERE "{node.column}" IS NOT NULL LIMIT 5'
                            ).fetchall()
                            vals = [str(r[0]) for r in sample if r[0]]
                            errors.append(
                                f'Column "{node.table}.{node.column}" is non-numeric '
                                f'(sample values: {vals}) — cannot {_comp.upper()} it. '
                                f'This column should be a WHERE filter, not the aggregation target. '
                                f'Pick a NUMERIC column from the same table as the metric to {_comp.upper()}.'
                            )
                    except Exception:
                        pass
                _tc_conn.close()
            except Exception:
                pass
            if _nodes_to_demote:
                _demoted_tables = [output_nodes[i].table for i in _nodes_to_demote]
                output_nodes = [n for i, n in enumerate(output_nodes) if i not in _nodes_to_demote]
                if not output_nodes:
                    demoted_table = _demoted_tables[0] if _demoted_tables else ""
                    if not demoted_table and filter_nodes:
                        demoted_table = filter_nodes[-1].table
                    if demoted_table:
                        try:
                            _tc_conn2 = sqlite3.connect(str(db_path), timeout=5)
                            col_info2 = _tc_conn2.execute(
                                f'PRAGMA table_info("{demoted_table}")'
                            ).fetchall()
                            for ci in col_info2:
                                ct = ci[2].upper()
                                cn = ci[1].lower()
                                if ct in ("INTEGER", "REAL", "NUMERIC", "FLOAT", "DECIMAL"):
                                    if not cn.endswith("_id") and cn != "id":
                                        output_nodes.append(QueryNode(
                                            table=demoted_table, column=ci[1], role="output",
                                        ))
                                        break
                            _tc_conn2.close()
                        except Exception:
                            pass

        # Replace output columns that are ≥90% NULL with FK-resolved alternatives.
        # E.g. posts.LastEditorDisplayName (99% NULL) → JOIN users ON LastEditorUserId = users.Id → users.DisplayName
        if output_nodes and db_path and db_path.exists() and kg:
            _replacements: dict[int, QueryNode] = {}
            try:
                _null_conn = sqlite3.connect(str(db_path), timeout=5)
                for idx, node in enumerate(output_nodes):
                    total_r = _null_conn.execute(
                        f'SELECT COUNT(*) FROM "{node.table}"'
                    ).fetchone()[0]
                    if not total_r:
                        continue
                    null_r = _null_conn.execute(
                        f'SELECT COUNT(*) FROM "{node.table}" WHERE "{node.column}" IS NULL'
                    ).fetchone()[0]
                    if null_r / total_r < 0.9:
                        continue
                    # Column is ≥90% NULL — look for FK-based alternative
                    # Find a UserId/EditorId column with similar name that references another table
                    col_lower = node.column.lower()
                    # Derive candidate FK column: "LastEditorDisplayName" → "LastEditorUserId"
                    # or just scan for columns ending in Id/UserId in same table
                    table_schema = kg.get_table(node.table)
                    if not table_schema:
                        continue
                    fk_candidates = []
                    for c in table_schema.columns:
                        cn = c.name.lower()
                        if cn.endswith("id") and cn != "id":
                            # Check if column name shares prefix with the NULL column
                            # e.g. LastEditor in LastEditorDisplayName and LastEditorUserId
                            prefix = col_lower.replace("displayname", "").replace("name", "")
                            fk_prefix = cn.replace("userid", "").replace("id", "")
                            if prefix and fk_prefix and (prefix.startswith(fk_prefix) or fk_prefix.startswith(prefix)):
                                fk_candidates.append(c.name)
                    for fk_col in fk_candidates:
                        # Find the table this FK references
                        col_id = f"{node.table}.{fk_col}"
                        if kg.graph and col_id in kg.graph.fk_from:
                            for edge in kg.graph.fk_from[col_id]:
                                dst_node = kg.graph.columns.get(edge.dst)
                                if dst_node:
                                    ref_table = dst_node.table_id
                                    ref_schema = kg.get_table(ref_table)
                                    if ref_schema:
                                        # Find a display name column in the referenced table
                                        for rc in ref_schema.columns:
                                            rcn = rc.name.lower()
                                            if any(x in rcn for x in ("name", "display", "title", "label")):
                                                _replacements[idx] = QueryNode(
                                                    table=ref_table, column=rc.name, role="output",
                                                )
                                                break
                                    if idx in _replacements:
                                        break
                        if idx in _replacements:
                            break
                _null_conn.close()
            except Exception:
                pass
            if _replacements:
                for idx, replacement in _replacements.items():
                    old = output_nodes[idx]
                    output_nodes[idx] = replacement
                    errors.append(
                        f'"{old.table}.{old.column}" is ≥90% NULL — replaced with '
                        f'"{replacement.table}.{replacement.column}" via FK join.'
                    )

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

            # Detect aggregate filter: "COUNT(table.col)", "SUM(table.col)", etc.
            agg_match = re.match(
                r'(COUNT|SUM|AVG|MIN|MAX)\((\w+)\.(\w+)\)', col_ref, re.IGNORECASE
            )
            if agg_match:
                agg_func, agg_table, agg_col = agg_match.groups()
                filter_nodes.append(QueryNode(
                    table=agg_table, column=f"_expr:{agg_func}(\"{agg_table}\".\"{agg_col}\")",
                    role="filter", operator=operator, value=value,
                ))
                continue

            # Detect computed expressions: "table.col1/table.col2" or "table.col1/col2"
            expr_match = re.match(r'(\w+)\.(\w+)\s*([/*])\s*(?:(\w+)\.)?(\w+)$', col_ref)
            if expr_match:
                tbl1, col1, op_char, tbl2, col2 = expr_match.groups()
                tbl2 = tbl2 or tbl1
                # Verify both columns exist
                id1 = f"{tbl1}.{col1}"
                id2 = f"{tbl2}.{col2}"
                if g and id1 in g.columns and id2 in g.columns:
                    # Store as expression filter — column holds the SQL expression
                    expr_sql = f'CAST("{tbl1}"."{col1}" AS REAL) {op_char} "{tbl2}"."{col2}"'
                    filter_nodes.append(QueryNode(
                        table=tbl1, column=f"_expr:{expr_sql}", role="filter",
                        operator=operator, value=value,
                    ))
                    continue

            table, col = col_ref.split(".", 1)
            table = table.strip('"').strip("'")
            col = col.strip('"').strip("'")
            col_id = f"{table}.{col}"
            # Skip if value is a table.column reference (join condition, not a filter)
            val_str = str(value or "")
            if re.match(r'\w+\.\w+$', val_str):
                val_parts = val_str.split(".", 1)
                if g and f"{val_parts[0]}.{val_parts[1]}" in g.columns:
                    continue
            if g and col_id in g.columns:
                filter_nodes.append(QueryNode(
                    table=table, column=col, role="filter",
                    operator=operator, value=value,
                ))
            else:
                matched = self._fuzzy_match_column(table, col, kg)
                if not matched:
                    # Column not on specified table — check ALL tables
                    for t in kg.tables:
                        if t.name.lower() != table.lower():
                            for c in t.columns:
                                if c.name.lower() == col.lower():
                                    matched = (t.name, c.name)
                                    break
                            if matched:
                                break
                if matched:
                    filter_nodes.append(QueryNode(
                        table=matched[0], column=matched[1], role="filter",
                        operator=operator, value=value,
                    ))
                else:
                    errors.append(f"Filter column not in graph: {col_ref}")

        # Fix: "No.X" / "#X" patterns — if LLM chose a text column but value is numeric,
        # check if the numeric value exists in an ID column instead.
        if db_path and db_path.exists() and question:
            _id_num_match = re.search(r'(?:No\.|#|user\s+)(\d+)', question, re.IGNORECASE)
            if _id_num_match:
                _id_num = _id_num_match.group(1)
                _corrected: list[tuple[int, QueryNode]] = []
                try:
                    _id_conn = sqlite3.connect(str(db_path), timeout=5)
                    for idx, node in enumerate(filter_nodes):
                        if node.column.startswith("_expr:"):
                            continue
                        val_str = str(node.value)
                        # Only fix if the value contains the numeric ID but isn't purely the number
                        # (e.g. 'user24' contains '24' but is not just '24')
                        if _id_num not in val_str and val_str != _id_num:
                            continue
                        # Check if current column actually has this value
                        try:
                            has_val = _id_conn.execute(
                                f'SELECT 1 FROM "{node.table}" WHERE "{node.column}" = ? LIMIT 1',
                                (node.value,)
                            ).fetchone()
                        except Exception:
                            has_val = None
                        if has_val:
                            continue  # current filter works fine
                        # Look for ID columns (Id, UserId, etc.) that contain the numeric value
                        t_schema = kg.get_table(node.table)
                        if not t_schema:
                            continue
                        for col in t_schema.columns:
                            cn = col.name.lower()
                            if cn == "id" or cn.endswith("id") or cn.endswith("_id"):
                                try:
                                    has_id = _id_conn.execute(
                                        f'SELECT 1 FROM "{node.table}" WHERE "{col.name}" = ? LIMIT 1',
                                        (int(_id_num),)
                                    ).fetchone()
                                except Exception:
                                    has_id = None
                                if has_id:
                                    _corrected.append((idx, QueryNode(
                                        table=node.table, column=col.name,
                                        role="filter", operator="=", value=int(_id_num),
                                    )))
                                    break
                    _id_conn.close()
                except Exception:
                    pass
                for idx, new_node in _corrected:
                    old = filter_nodes[idx]
                    filter_nodes[idx] = new_node
                    errors.append(
                        f'Corrected filter: "{old.table}.{old.column}"=\'{old.value}\' → '
                        f'"{new_node.table}.{new_node.column}"={new_node.value} (numeric ID match)'
                    )

        # Deduplicate: same value used on multiple columns in the same table is nonsensical.
        # Probe DB to find which column actually contains the value, or use LLM to resolve.
        value_to_nodes: dict[tuple[str, str], list[QueryNode]] = {}
        for node in filter_nodes:
            if node.column.startswith("_expr:"):
                continue
            key = (node.table.lower(), str(node.value).lower())
            value_to_nodes.setdefault(key, []).append(node)

        dupes_to_remove: set[int] = set()
        for (tbl_lower, val_lower), nodes in value_to_nodes.items():
            if len(nodes) < 2:
                continue
            # Probe DB: which column(s) actually contain this value?
            valid_cols: list[QueryNode] = []
            if db_path and db_path.exists():
                try:
                    conn = sqlite3.connect(str(db_path), timeout=5)
                    for node in nodes:
                        try:
                            cnt = conn.execute(
                                f'SELECT COUNT(*) FROM "{node.table}" WHERE "{node.column}" = ? COLLATE NOCASE',
                                (node.value,)
                            ).fetchone()[0]
                            if cnt > 0:
                                valid_cols.append(node)
                        except Exception:
                            pass
                    conn.close()
                except Exception:
                    pass

            if len(valid_cols) == 1:
                # Only one column has this value — keep that one, remove others
                keeper = valid_cols[0]
                for i, node in enumerate(filter_nodes):
                    if node in nodes and node is not keeper:
                        dupes_to_remove.add(i)
            elif len(valid_cols) == 0:
                # None have it — keep just the first (LLM's primary pick)
                for i, node in enumerate(filter_nodes):
                    if node in nodes[1:]:
                        dupes_to_remove.add(i)
            else:
                # Multiple columns have the value — pick the column whose OTHER
                # values relate to words in the question. This signals the column
                # provides the semantic distinction the question is asking about.
                # E.g. question="cash withdrawals", operation has VYBER/VYBER KARTOU
                # — "cash" vs "card" is the distinction operation provides.
                best_node = valid_cols[0]
                best_score = -1
                q_words = set(re.findall(r'[a-z]{3,}', question.lower())) if question else set()
                try:
                    conn2 = sqlite3.connect(str(db_path), timeout=5)
                    for node in valid_cols:
                        try:
                            # Get all distinct values in this column
                            all_vals = conn2.execute(
                                f'SELECT DISTINCT "{node.column}" FROM "{node.table}" '
                                f'WHERE "{node.column}" IS NOT NULL LIMIT 20'
                            ).fetchall()
                            col_vals = [str(r[0]).lower() for r in all_vals if r[0]]

                            # Score 1: question words that appear in OTHER column values
                            # (indicates the column's vocabulary matches the question's concepts)
                            question_overlap = 0
                            for cv in col_vals:
                                cv_words = set(re.findall(r'[a-z]{3,}', cv))
                                question_overlap += len(cv_words & q_words)

                            # Score 2: prefix-siblings (sub-type granularity)
                            val_lower = str(node.value).lower()
                            siblings = sum(
                                1 for cv in col_vals
                                if cv.startswith(val_lower) and cv != val_lower
                            )

                            # Score 3: total distinct values (finer categorization)
                            total_distinct = len(col_vals)

                            score = question_overlap * 100 + siblings * 10 + total_distinct
                            if score > best_score:
                                best_score = score
                                best_node = node
                        except Exception:
                            pass
                    conn2.close()
                except Exception:
                    pass
                for i, node in enumerate(filter_nodes):
                    if node in nodes and node is not best_node:
                        dupes_to_remove.add(i)

        if dupes_to_remove:
            filter_nodes = [n for i, n in enumerate(filter_nodes) if i not in dupes_to_remove]

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


    def _resolve_fk_output_columns(
        self, output_nodes: list[QueryNode], kg: KnowledgeGraph, db_path: Path | None,
    ) -> list[QueryNode]:
        """Replace FK output columns with the human-readable display column from the referenced table.

        When the picker selects an FK column (link_to_major, category_id, etc.) as output,
        users want the resolved value (e.g., "Business"), not the opaque ID (e.g., "recXXX").
        Only resolves columns that are clearly reference/link columns, not shared PKs like "ID".
        """
        if not kg or not kg.graph:
            return output_nodes
        g = kg.graph
        new_output: list[QueryNode] = []
        for node in output_nodes:
            col_lower = node.column.lower()
            # Only resolve columns that look like references (link_to_X, X_id, ref_X, fk_X)
            is_ref_col = (
                col_lower.startswith("link_to_")
                or col_lower.startswith("ref_")
                or col_lower.startswith("fk_")
                or (col_lower.endswith("_id") and col_lower != "id" and col_lower != "_id")
                or (col_lower.endswith("_key") and col_lower != "key")
                or (col_lower.endswith("_code") and col_lower != "code")
            )
            if not is_ref_col:
                new_output.append(node)
                continue
            col_id = f"{node.table}.{node.column}"
            fk_edges_from = g.fk_from.get(col_id, [])
            if not fk_edges_from:
                new_output.append(node)
                continue
            edge = fk_edges_from[0]
            dst_col_node = g.columns.get(edge.dst)
            if not dst_col_node:
                new_output.append(node)
                continue
            ref_table = dst_col_node.table_id
            ref_col = dst_col_node.name
            # Look up display column from fk_display_map
            display_col_id = g.fk_display_map.get(col_id)
            if display_col_id and "." in display_col_id:
                disp_table, disp_col = display_col_id.split(".", 1)
                if self._db_check(db_path, disp_table, disp_col):
                    self._log("fk_output_resolve", f"{col_id} → {display_col_id}")
                    new_output.append(QueryNode(table=disp_table, column=disp_col, role="output"))
                    continue
            # Fallback: find first non-PK text column in referenced table
            ref_schema = kg.get_table(ref_table)
            if ref_schema:
                for col in ref_schema.columns:
                    if col.name == ref_col:
                        continue
                    if col.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", ""):
                        if self._db_check(db_path, ref_table, col.name):
                            self._log("fk_output_resolve", f"{col_id} → {ref_table}.{col.name}")
                            new_output.append(QueryNode(table=ref_table, column=col.name, role="output"))
                            break
                else:
                    new_output.append(node)
            else:
                new_output.append(node)
        return new_output

    def _resolve_columns_from_kg(
        self, question: str, user_intent: str, kg: KnowledgeGraph, anchor_text: str,
    ) -> list[str]:
        """Resolve Columns needed by querying the KG.

        Hybrid approach:
        1. Deterministic word-matching scores columns against question terms.
        2. If top score is high (≥3), trust it — column name directly in question.
        3. If low confidence, LLM picks from the full KG column list with bias
           toward columns whose names appear in the question.
        """
        metric_m = re.search(r'Metric \(SELECT\):\s*(.+)', user_intent)
        metric_text = metric_m.group(1) if metric_m else ""
        entity_m = re.search(r'Entity of interest:\s*(\w+)', user_intent)
        entity_table = entity_m.group(1).lower() if entity_m else ""

        metric_tokens = set(re.findall(r'[a-z]+', metric_text.lower())) if metric_text else set()
        question_tokens = set(re.findall(r'[a-z]+', question.lower()))
        pop_m = re.search(r'Population \(WHERE\):\s*(.+)', user_intent)
        pop_tokens = set(re.findall(r'[a-z]+', pop_m.group(1).lower())) if pop_m else set()

        def _s(w: str) -> str:
            return w.rstrip("s") if len(w) > 3 else w

        metric_stems = {_s(t) for t in metric_tokens if len(t) >= 2}
        question_stems = {_s(t) for t in question_tokens if len(t) >= 2}
        pop_stems = {_s(t) for t in pop_tokens if len(t) >= 2}

        _q_mentions_id = bool(re.search(r'\bid\b', question.lower())) or "id" in metric_stems
        _col_null_ratio: dict[str, float] = {}
        col_candidates: list[tuple[str, str, str, set[str]]] = []
        _q_lower_resolve = question.lower()
        for table in kg.tables:
            for col in table.columns:
                cn = col.name
                cn_lower = cn.lower()
                if "link" in cn_lower:
                    continue
                if "ref" in cn_lower and cn_lower not in _q_lower_resolve and "reference" not in _q_lower_resolve:
                    continue
                if cn_lower == "id" and not _q_mentions_id:
                    continue
                if cn_lower != "id" and cn_lower.endswith("id"):
                    continue
                camel_parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', cn)
                space_parts = re.findall(r'[a-z]+', cn_lower.replace("_", " "))
                words = {w.lower() for w in camel_parts} | set(space_parts)
                words = {w for w in words if len(w) >= 2}
                if words:
                    col_candidates.append((table.name, col.name, cn_lower, words))
                    if hasattr(col, 'null_ratio') and col.null_ratio and col.null_ratio >= 0.95:
                        _col_null_ratio[f"{table.name}.{col.name}"] = col.null_ratio

        q_stem_list = [_s(t) for t in re.findall(r'[a-z]+', question.lower()) if len(t) >= 2]
        q_bigrams = {(q_stem_list[i], q_stem_list[i+1]) for i in range(len(q_stem_list)-1)}

        scored: list[tuple[float, str, str, set[str]]] = []
        for table_name, col_name, col_lower, words in col_candidates:
            col_stems = {_s(w) for w in words}
            metric_match = col_stems & metric_stems
            question_match = col_stems & question_stems
            pop_only = (col_stems & pop_stems) - metric_stems
            score = len(metric_match) * 2 + len(question_match - metric_stems)
            if not metric_match and pop_only and len(pop_only) >= len(question_match):
                score *= 0.5
            col_stem_list = list(col_stems)
            for a in col_stem_list:
                for b in col_stem_list:
                    if a != b and (a, b) in q_bigrams:
                        score += 2
                        break
                else:
                    continue
                break
            if score >= 1:
                if table_name.lower() == entity_table:
                    score += 1
                if f"{table_name}.{col_name}" in _col_null_ratio:
                    score *= 0.3
                scored.append((score, table_name, col_name, words))

        scored.sort(key=lambda x: -x[0])

        # Build FK adjacency for joinability check
        connected: dict[str, set[str]] = {}
        if kg.graph:
            for edge in kg.graph.fk_edges:
                src_tbl = edge.src.split(".")[0] if "." in edge.src else ""
                dst_tbl = edge.dst.split(".")[0] if "." in edge.dst else ""
                if src_tbl and dst_tbl:
                    connected.setdefault(src_tbl.lower(), set()).add(dst_tbl.lower())
                    connected.setdefault(dst_tbl.lower(), set()).add(src_tbl.lower())

        # Confidence gate: top score ≥ 3 means direct name match is strong
        top_score = scored[0][0] if scored else 0
        if top_score >= 3:
            resolved = self._select_from_scored(scored, connected, entity_table, question_stems, metric_stems)
            self._log("kg_col_resolve", f"deterministic={resolved}, top={top_score}")
            if resolved:
                # Structural gap check: how many output values does the metric expect?
                metric_phrases = re.split(r'\band\b|,', metric_text)
                expected_count = sum(1 for p in metric_phrases if len(p.strip().split()) >= 2)
                if len(resolved) >= expected_count or expected_count <= 1:
                    return resolved
                # Found fewer columns than expected — LLM fills gaps
                self._log("kg_col_resolve", f"gap: have {len(resolved)}, expect {expected_count}")
                llm_cols = self._resolve_columns_llm(
                    question, user_intent, kg, entity_table, connected,
                )
                seen = {r.lower() for r in resolved}
                for lc in llm_cols:
                    if lc.lower() not in seen:
                        resolved.append(lc)
                        seen.add(lc.lower())
                return resolved[:5]

        # Low confidence — full LLM resolution
        self._log("kg_col_resolve", f"low confidence, top={top_score}, falling to LLM")
        return self._resolve_columns_llm(
            question, user_intent, kg, entity_table, connected,
        )

    def _select_from_scored(
        self,
        scored: list[tuple[float, str, str, set[str]]],
        connected: dict[str, set[str]],
        entity_table: str,
        question_stems: set[str] | None = None,
        metric_stems: set[str] | None = None,
    ) -> list[str]:
        """Select columns from scored list with joinability + coverage filtering."""
        def _s(w: str) -> str:
            return w.rstrip("s") if len(w) > 3 else w

        all_relevant_stems = (question_stems or set()) | (metric_stems or set())
        resolved: list[str] = []
        seen_cols: set[str] = set()
        selected_tables: set[str] = set()
        covered_stems: set[str] = set()
        min_score = 2
        for score, table_name, col_name, words in scored:
            if score < min_score:
                break
            col_key = col_name.lower()
            if col_key in seen_cols:
                continue
            col_stems = {_s(w) for w in words}
            # Coverage: does this column contribute a NEW relevant stem?
            if resolved and all_relevant_stems:
                relevant_new = (col_stems & all_relevant_stems) - covered_stems
                if not relevant_new:
                    continue
            tbl_lower = table_name.lower()
            if selected_tables and tbl_lower not in selected_tables:
                reachable = any(
                    tbl_lower in connected.get(st, set())
                    for st in selected_tables
                )
                if not reachable:
                    continue
            seen_cols.add(col_key)
            selected_tables.add(tbl_lower)
            covered_stems |= col_stems
            resolved.append(f"{table_name}.{col_name}")
            if len(resolved) >= 5:
                break
        return resolved

    def _resolve_columns_llm(
        self,
        question: str,
        user_intent: str,
        kg: KnowledgeGraph,
        entity_table: str,
        connected: dict[str, set[str]],
    ) -> list[str]:
        """LLM picks output columns from KG schema when deterministic match is weak."""
        # Build column list grouped by table
        table_cols: list[str] = []
        for table in kg.tables:
            cols = []
            for col in table.columns:
                cn_lower = col.name.lower()
                if cn_lower == "id" or cn_lower.endswith("id"):
                    continue
                desc = f" -- {col.description}" if col.description else ""
                cols.append(f'    "{col.name}" ({col.sql_type}){desc}')
            if cols:
                table_cols.append(f'  TABLE "{table.name}":\n' + "\n".join(cols))

        schema_text = "\n".join(table_cols)

        prompt = f"""QUESTION: {question}

{user_intent}

AVAILABLE COLUMNS:
{schema_text}

Pick the OUTPUT columns needed to answer this question (the columns whose values appear in the final answer).
Do NOT pick filter/WHERE columns — only columns whose values the user wants to SEE.

RULES:
- Strongly prefer columns whose names directly match words in the question
- Only pick columns from tables that can be joined together
- Pick 1-4 columns maximum
- If the question mentions a concept that maps to a specific column (e.g. "earnings" → Salary, "popularity" → ViewCount), pick that column

Return JSON: {{"columns": ["table.column", ...]}}"""

        messages = [ModelMessage(role="user", content=prompt)]
        try:
            raw = self._model_call_with_retry(messages, thinking=False)
            parsed = self._parse_json(raw)
            if not isinstance(parsed, dict) or not parsed.get("columns"):
                return []
        except Exception:
            return []

        # Validate against actual KG schema + joinability
        resolved: list[str] = []
        seen_cols: set[str] = set()
        selected_tables: set[str] = set()
        for ref in parsed["columns"]:
            if "." not in ref:
                continue
            tbl, col = ref.split(".", 1)
            tbl = tbl.strip('"').strip("'")
            col = col.strip('"').strip("'")
            table_schema = kg.get_table(tbl)
            if not table_schema:
                # Try case-insensitive match
                for t in kg.tables:
                    if t.name.lower() == tbl.lower():
                        table_schema = t
                        tbl = t.name
                        break
            if not table_schema:
                continue
            actual_col = next(
                (c.name for c in table_schema.columns if c.name.lower() == col.lower()), None
            )
            if not actual_col:
                continue
            col_key = actual_col.lower()
            if col_key in seen_cols:
                continue
            # Joinability check
            tbl_lower = tbl.lower()
            if selected_tables and tbl_lower not in selected_tables:
                reachable = any(
                    tbl_lower in connected.get(st, set())
                    for st in selected_tables
                )
                if not reachable:
                    continue
            seen_cols.add(col_key)
            selected_tables.add(tbl_lower)
            resolved.append(f"{tbl}.{actual_col}")
            if len(resolved) >= 5:
                break

        return resolved


    def _sanity_check_picks(
        self,
        question: str,
        picked: dict[str, Any],
        output_nodes: list[QueryNode],
        filter_nodes: list[QueryNode],
        user_intent: str,
        kg: KnowledgeGraph,
        anchor_text: str = "",
        db_path: Path | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Check if LLM picks are consistent with the question and user intent.
        Returns (issues_text, entity_found_in) where entity_found_in maps entity values to table.column."""
        issues: list[str] = []
        q_lower = question.lower()

        # Parse structured fields from user_intent
        intent_entity = ""
        intent_metric = ""
        intent_operation = ""
        intent_population = ""
        intent_shape = ""
        for line in user_intent.split("\n"):
            if "Entity of interest:" in line:
                intent_entity = line.split("Entity of interest:")[1].split("(")[0].strip().lower()
            elif "Metric (SELECT):" in line:
                intent_metric = line.split("Metric (SELECT):")[1].strip().lower()
            elif "Operation:" in line:
                intent_operation = line.split("Operation:")[1].strip().lower()
            elif "Population (WHERE):" in line:
                intent_population = line.split("Population (WHERE):")[1].strip().lower()
            elif "Answer shape:" in line:
                intent_shape = line.split("Answer shape:")[1].strip().lower()

        # --- Check 1: Named entities in question should appear in filters ---
        # Only match deliberate quoting — skip apostrophes in contractions/possessives
        quoted = re.findall(r'"([^"]+)"', question)
        named_entities = list(quoted)
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b', question)
        named_entities.extend(proper_nouns)
        # Also catch single capitalized words that match known column values in the graph
        single_caps = re.findall(r'\b([A-Z][a-z]{2,})\b', question)
        # Identify sentence-start positions (pos 0 or after ./?/! + space)
        _sentence_starts: set[int] = {0}
        for m in re.finditer(r'[.?!]\s+', question):
            _sentence_starts.add(m.end())
        # Track which DB column each named entity was found in (for better sanity messages)
        entity_found_in: dict[str, str] = {}
        table_names_lower = {t.name.lower() for t in kg.tables} if kg else set()
        if kg and kg.graph:
            for word in single_caps:
                # Skip words that are capitalized only because they start a sentence
                _word_pos = question.find(word)
                if _word_pos in _sentence_starts:
                    continue
                if word in named_entities:
                    continue
                if any(word in ne for ne in named_entities):
                    continue
                word_lower = word.lower()
                if word_lower in table_names_lower:
                    continue
                if kg.graph.value_index and word_lower in kg.graph.value_index:
                    named_entities.append(word)
                elif len(word_lower) >= 4:
                    # Substring match against indexed values
                    found = False
                    if kg.graph.value_index:
                        found = any(word_lower in idx_val for idx_val in kg.graph.value_index)
                    # Probe actual DB: check TEXT columns for LIKE match
                    # Prefer columns whose name relates to surrounding context in question
                    if not found and db_path:
                        try:
                            _conn = sqlite3.connect(str(db_path))
                            # Get context words around the entity in the question
                            word_pos = question.find(word)
                            context_window = question[max(0, word_pos - 30):word_pos + len(word) + 50].lower()
                            context_words = set(re.findall(r'[a-z]{3,}', context_window))
                            matches: list[tuple[str, str, int]] = []
                            for t in kg.tables:
                                for col in t.columns:
                                    if col.sql_type.upper() not in ("TEXT", "VARCHAR", "CHAR", ""):
                                        continue
                                    # Skip free-text columns (avg length > 100 chars) —
                                    # they match any English word via LIKE
                                    avg_len_row = _conn.execute(
                                        f'SELECT AVG(LENGTH("{col.name}")) FROM (SELECT "{col.name}" FROM "{t.name}" LIMIT 200)',
                                    ).fetchone()
                                    if avg_len_row and avg_len_row[0] and avg_len_row[0] > 100:
                                        continue
                                    row = _conn.execute(
                                        f'SELECT 1 FROM "{t.name}" WHERE "{col.name}" LIKE ? LIMIT 1',
                                        (f"%{word}%",),
                                    ).fetchone()
                                    if row:
                                        # Score: how well does column name relate to context?
                                        col_lower = col.name.lower()
                                        col_name_words = set(re.findall(r'[a-z]{3,}', col_lower))
                                        ctx_stemmed = {w.rstrip("s") if len(w) > 3 else w for w in context_words}
                                        col_stemmed = {w.rstrip("s") if len(w) > 3 else w for w in col_name_words}
                                        overlap = ctx_stemmed & col_stemmed
                                        # Base score from overlap count
                                        score = len(overlap) * 10
                                        # Tiebreak: prefer columns where the matching context word
                                        # appears later in the noun phrase (head noun > modifier)
                                        ctx_list = re.findall(r'[a-z]{3,}', context_window)
                                        for ow in overlap:
                                            for i, cw in enumerate(ctx_list):
                                                cw_stem = cw.rstrip("s") if len(cw) > 3 else cw
                                                if cw_stem == ow:
                                                    score += i
                                                    break
                                        matches.append((t.name, col.name, score))
                            _conn.close()
                            if matches:
                                found = True
                                matches.sort(key=lambda x: -x[2])
                                entity_found_in[word] = f'{matches[0][0]}.{matches[0][1]}'
                        except Exception:
                            pass
                    if found:
                        named_entities.append(word)

        if named_entities:
            filter_values = [str(n.value).lower() for n in filter_nodes]
            filter_text = " ".join(filter_values)
            filter_cols = " ".join(f"{n.column}" for n in filter_nodes).lower()
            output_text = " ".join(f"{n.table}.{n.column}" for n in output_nodes).lower()
            for entity in named_entities:
                entity_lower = entity.lower()
                if (entity_lower not in filter_text and entity_lower not in output_text
                        and entity_lower not in filter_cols):
                    col_hint = ""
                    if entity in entity_found_in:
                        col_hint = f' (found in column {entity_found_in[entity]} — use LIKE \'%{entity}%\')'
                    issues.append(
                        f'The question mentions "{entity}" but it is not in any filter or output. '
                        f'Add a filter condition for it.{col_hint}'
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
        # Skip for "list" shape with lookup — metric describes the entity being listed, not a column
        # Skip when the metric is a concept (no matching column exists in the schema at all)
        comp_type = picked.get("computation_type", "")
        is_aggregate = comp_type in ("count", "sum", "avg", "min_max", "ratio", "percentage")
        is_superlative = intent_operation in ("min_max", "count", "sum", "avg", "ratio", "percentage")
        is_list_lookup = intent_shape == "list" and intent_operation == "lookup"
        if intent_metric and output_nodes and not is_aggregate and not is_superlative and not is_list_lookup:
            metric_words = set(intent_metric.replace(",", " ").split())
            output_cols = {n.column.lower() for n in output_nodes}
            metric_found = any(
                any(mw in col or col in mw for mw in metric_words)
                for col in output_cols
            )
            if not metric_found and intent_metric not in ("all", "all columns requested"):
                # Check if the metric corresponds to any column in the schema
                all_schema_cols = set()
                for t in kg.tables:
                    for c in t.columns:
                        all_schema_cols.add(c.name.lower())
                metric_has_column = any(
                    any(mw in col or col in mw for mw in metric_words)
                    for col in all_schema_cols
                )
                if metric_has_column:
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
                hint = ""
                if "normal" in pop_words:
                    hint = (
                        ' Remember: "normal level" means value is WITHIN the healthy range '
                        '(use >= lower AND <= upper). Check that your filters keep values '
                        'inside normal bounds, not outside.'
                    )
                elif "abnormal" in pop_words:
                    hint = (
                        ' Remember: "abnormal level" means value is OUTSIDE the healthy range. '
                        'If the data range is entirely outside normal, use IS NOT NULL.'
                    )
                issues.append(
                    f'Intent says population filter is "{intent_population}" '
                    f'but no matching filter condition was found.{hint}'
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

        # --- Check 9: Question words matching column names that imply a filter ---
        # e.g., "approved" in the question when there's an "approved" column with boolean values
        # Inject the filter directly (not just warn) so the table enters the query path
        if kg and kg.graph:
            filter_col_names = {n.column.lower() for n in filter_nodes}
            output_col_names = {n.column.lower() for n in output_nodes}
            q_words = set(re.findall(r'[a-z]+', q_lower))
            for col_id, col_node in kg.graph.columns.items():
                col_lower = col_node.name.lower()
                if col_lower in q_words and col_lower not in filter_col_names and col_lower not in output_col_names:
                    val_nodes = kg.graph.get_column_values(col_id)
                    if val_nodes:
                        vals = [str(v.value).lower() for v in val_nodes]
                        is_boolean = set(vals) <= {"true", "false", "yes", "no", "0", "1", "t", "f"}
                        if is_boolean and len(vals) <= 4:
                            positive_val = "true" if "true" in vals else vals[0]
                            filter_nodes.append(QueryNode(
                                table=col_node.table_id, column=col_node.name,
                                role="filter", operator="=", value=positive_val,
                            ))
                            filter_col_names.add(col_lower)
                            issues.append(
                                f'Injected filter: "{col_node.table_id}"."{col_node.name}" = \'{positive_val}\' '
                                f'(question word "{col_lower}" matches boolean column).'
                            )

        # --- Check 10: Spurious filters — value not grounded in question or intent ---
        if filter_nodes:
            # Only use question + semantic intent fields (not column resolution hints)
            q_and_intent = q_lower + " " + intent_population + " " + intent_metric
            q_words = set(re.findall(r'[a-z]{2,}', q_and_intent))
            q_words_stemmed_10 = {w.rstrip("s") if len(w) > 3 else w for w in q_words}
            for node in filter_nodes:
                if node.column.startswith("_expr:"):
                    continue
                if node.operator != "=":
                    continue
                val_lower = str(node.value).lower()
                col_lower = node.column.lower()
                # Skip subquery-style values — picker intentionally constructed them
                if val_lower.lstrip().startswith("(select") or val_lower.lstrip().startswith("select"):
                    continue
                # FK-resolved numeric IDs: keep if column name semantically matches question
                if val_lower.isdigit() and col_lower.endswith("_id"):
                    col_name_words = set(re.findall(r'[a-z]{3,}', col_lower.replace("_id", "")))
                    col_stemmed = {w.rstrip("s") if len(w) > 3 else w for w in col_name_words}
                    if col_stemmed & q_words_stemmed_10:
                        continue
                # For single-char values, use symbol mapping — substring match is unreliable
                if len(val_lower) <= 1:
                    symbol_map = {"#": "triple", "=": "double", "-": "single", "+": "carcinogenic"}
                    domain_word = symbol_map.get(val_lower, "")
                    if domain_word and domain_word in q_and_intent:
                        continue
                    # Not matched symbolically — check DB
                    if col_lower not in q_words:
                        if not self._value_exists_in_db(db_path, node):
                            issues.append(
                                f'Filter "{node.table}"."{node.column}" = \'{node.value}\' is not grounded in the question. '
                                f'Remove it unless the question explicitly asks for this filter.'
                            )
                    continue
                # For multi-char values: check if value or column name appears as a word
                val_in_q = val_lower in q_and_intent or any(val_lower in w for w in q_words)
                col_in_q = col_lower in q_words
                if val_in_q or col_in_q:
                    continue
                # Last resort: if value exists in the DB column, it's valid
                if not self._value_exists_in_db(db_path, node):
                    issues.append(
                        f'Filter "{node.table}"."{node.column}" = \'{node.value}\' is not grounded in the question. '
                        f'Remove it unless the question explicitly asks for this filter.'
                    )

        return "\n".join(issues), entity_found_in

    # ------------------------------------------------------------------
    # Deterministic intent signal enforcement
    # ------------------------------------------------------------------

    def _enforce_intent_signals(
        self,
        user_intent: str,
        filter_nodes: list[QueryNode],
        order_nodes: list[QueryNode],
        output_nodes: list[QueryNode],
        picked: dict[str, Any],
        kg: KnowledgeGraph,
        db_path: Path | None,
    ) -> tuple[list[QueryNode], list[QueryNode]]:
        """Deterministically enforce intent signals that the picker LLM may have missed.

        Consumes: group_condition, temporal_filter, ordinal, sort_direction.
        Returns updated (filter_nodes, order_nodes).
        """
        if not user_intent:
            return filter_nodes, order_nodes

        # Parse intent signals
        group_condition = ""
        temporal_filter = ""
        ordinal = ""
        sort_direction = ""
        for line in user_intent.split("\n"):
            if "Group condition (HAVING):" in line:
                group_condition = line.split("Group condition (HAVING):")[1].strip()
            elif "Temporal filter:" in line:
                temporal_filter = line.split("Temporal filter:")[1].strip()
            elif "Ordinal:" in line:
                ordinal = line.split("Ordinal:")[1].strip()
            elif "Sort:" in line:
                sort_direction = line.split("Sort:")[1].strip()

        # --- 1. Group condition → HAVING filter ---
        # If intent detected a group condition but no _expr: filter exists, inject one.
        if group_condition:
            has_expr_filter = any(n.column.startswith("_expr:") for n in filter_nodes)
            if not has_expr_filter:
                injected = self._inject_group_condition(
                    group_condition, filter_nodes, output_nodes, kg, db_path,
                )
                if injected:
                    filter_nodes = injected
                    self._log("intent_enforce_having",
                              f"Injected HAVING from group_condition: {group_condition}")

        # --- 2. Temporal filter → date/year condition ---
        # If intent detected a time constraint but no date filter exists, inject one.
        if temporal_filter:
            has_temporal = any(
                "date" in n.column.lower() or "year" in n.column.lower()
                or "time" in n.column.lower() or "season" in n.column.lower()
                for n in filter_nodes
            )
            if not has_temporal:
                injected = self._inject_temporal_filter(
                    temporal_filter, filter_nodes, output_nodes, kg, db_path,
                )
                if injected:
                    filter_nodes = injected
                    self._log("intent_enforce_temporal",
                              f"Injected temporal filter: {temporal_filter}")

        # --- 3. Ordinal + sort_direction → ORDER BY ---
        # If intent detected a positional selector but picker has no order_by, inject one.
        if ordinal and not order_nodes:
            new_order = self._inject_ordinal_order(
                ordinal, sort_direction, output_nodes, filter_nodes, kg, db_path,
            )
            if new_order:
                order_nodes = new_order
                self._log("intent_enforce_ordinal",
                          f"Injected ORDER BY from ordinal={ordinal}, dir={sort_direction}")
        elif sort_direction and sort_direction.upper() in ("ASC", "DESC") and not order_nodes:
            new_order = self._inject_ordinal_order(
                "", sort_direction, output_nodes, filter_nodes, kg, db_path,
            )
            if new_order:
                order_nodes = new_order
                self._log("intent_enforce_sort",
                          f"Injected ORDER BY from sort_direction={sort_direction}")

        return filter_nodes, order_nodes

    def _inject_group_condition(
        self,
        group_condition: str,
        filter_nodes: list[QueryNode],
        output_nodes: list[QueryNode],
        kg: KnowledgeGraph,
        db_path: Path | None,
    ) -> list[QueryNode] | None:
        """Parse a group condition string and inject a HAVING _expr: filter."""
        # Extract the aggregate pattern: "more than N", "at least N", "total X > N"
        cond_lower = group_condition.lower()

        # Parse threshold
        num_match = re.search(r'(\d+(?:\.\d+)?)', group_condition)
        if not num_match:
            return None
        threshold = num_match.group(1)

        # Parse operator
        if "more than" in cond_lower or "greater than" in cond_lower or "above" in cond_lower:
            operator = ">"
        elif "at least" in cond_lower or "no less than" in cond_lower:
            operator = ">="
        elif "fewer than" in cond_lower or "less than" in cond_lower or "below" in cond_lower:
            operator = "<"
        elif "at most" in cond_lower or "no more than" in cond_lower:
            operator = "<="
        else:
            operator = ">"

        # Determine aggregate function
        if re.search(r'\b(?:total|sum)\b', cond_lower):
            agg_func = "SUM"
        elif re.search(r'\b(?:average|avg|mean)\b', cond_lower):
            agg_func = "AVG"
        else:
            agg_func = "COUNT"

        # Find the table to count from. Look for a detail/child table that
        # connects to the output table via FK.
        entity_table = output_nodes[0].table if output_nodes else ""
        if not entity_table:
            return None

        # Find a table that references the entity table (child → parent FK)
        count_table = ""
        count_col = ""
        if kg and kg.graph:
            for edge in kg.graph.fk_edges:
                src_col_node = kg.graph.columns.get(edge.src)
                dst_col_node = kg.graph.columns.get(edge.dst)
                if not src_col_node or not dst_col_node:
                    continue
                # Child table's FK → parent table's PK
                if dst_col_node.table_id == entity_table and src_col_node.table_id != entity_table:
                    count_table = src_col_node.table_id
                    count_col = src_col_node.name
                    break
                # Or: entity is the child, parent is different
                if src_col_node.table_id == entity_table and dst_col_node.table_id != entity_table:
                    count_table = dst_col_node.table_id
                    count_col = dst_col_node.name
                    break

        if not count_table or not count_col:
            # Fallback: use entity table itself with its PK
            ts = kg.get_table(entity_table)
            if ts and ts.columns:
                count_table = entity_table
                count_col = ts.columns[0].name
            else:
                return None

        # Build the _expr: filter
        expr_sql = f'{agg_func}("{count_table}"."{count_col}")'
        new_filter = QueryNode(
            table=count_table, column=f"_expr:{expr_sql}",
            role="filter", operator=operator, value=threshold,
        )
        return list(filter_nodes) + [new_filter]

    def _inject_temporal_filter(
        self,
        temporal_filter: str,
        filter_nodes: list[QueryNode],
        output_nodes: list[QueryNode],
        kg: KnowledgeGraph,
        db_path: Path | None,
    ) -> list[QueryNode] | None:
        """Inject a year/date filter from the temporal_filter intent signal."""
        if not db_path or not db_path.exists():
            return None

        # Extract year(s) from temporal_filter
        year_match = re.search(r'\b((?:19|20)\d{2})\b', temporal_filter)
        if not year_match:
            return None
        year = year_match.group(1)

        # Find date/year columns in tables involved in the query + 1-hop neighbors
        involved_tables = {n.table for n in output_nodes} | {n.table for n in filter_nodes}
        if not involved_tables:
            return None

        best_col: tuple[str, str] | None = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            for tname in involved_tables:
                ts = kg.get_table(tname)
                if not ts:
                    continue
                for col in ts.columns:
                    cl = col.name.lower()
                    if "date" in cl or "year" in cl or cl == "season" or "time" in cl:
                        # Verify the year exists in this column
                        row = conn.execute(
                            f'SELECT 1 FROM "{tname}" WHERE CAST("{col.name}" AS TEXT) LIKE ? LIMIT 1',
                            (f"%{year}%",),
                        ).fetchone()
                        if row:
                            best_col = (tname, col.name)
                            break
                if best_col:
                    break
            conn.close()
        except Exception:
            return None

        if not best_col:
            return None

        tname, col_name = best_col
        # Decide filter format based on column type
        ts = kg.get_table(tname)
        col_type = "TEXT"
        if ts:
            for c in ts.columns:
                if c.name == col_name:
                    col_type = c.sql_type.upper()
                    break

        if "year" in col_name.lower() or col_type in ("INTEGER", "INT"):
            new_filter = QueryNode(
                table=tname, column=col_name,
                role="filter", operator="=", value=year,
            )
        else:
            new_filter = QueryNode(
                table=tname, column=col_name,
                role="filter", operator="LIKE", value=f"%{year}%",
            )

        return list(filter_nodes) + [new_filter]

    def _inject_ordinal_order(
        self,
        ordinal: str,
        sort_direction: str,
        output_nodes: list[QueryNode],
        filter_nodes: list[QueryNode],
        kg: KnowledgeGraph,
        db_path: Path | None,
    ) -> list[QueryNode] | None:
        """Inject an ORDER BY node from ordinal/sort_direction intent signals."""
        # Determine direction
        direction = "ASC"
        if sort_direction and sort_direction.upper() in ("ASC", "DESC"):
            direction = sort_direction.upper()
        elif ordinal:
            ord_lower = ordinal.lower()
            if any(w in ord_lower for w in ("last", "latest", "newest", "highest", "most", "best", "top", "champion", "winner")):
                direction = "DESC"
            elif any(w in ord_lower for w in ("first", "earliest", "oldest", "lowest", "least", "worst", "bottom")):
                direction = "ASC"

        # Find the best ordering column from the output/filter tables
        entity_table = output_nodes[0].table if output_nodes else ""
        if not entity_table:
            return None

        ts = kg.get_table(entity_table)
        if not ts:
            return None

        # Look for a numeric column that makes sense for ordering
        # Priority: date columns > numeric non-PK columns > PK
        order_col = None
        for col in ts.columns:
            cl = col.name.lower()
            if "date" in cl or "time" in cl or "year" in cl:
                order_col = col.name
                break
        if not order_col:
            for col in ts.columns:
                cl = col.name.lower()
                if col.sql_type.upper() in ("REAL", "FLOAT", "INTEGER", "INT", "NUMERIC"):
                    if not (cl.endswith("id") or cl.startswith("link")):
                        order_col = col.name
                        break

        if not order_col:
            return None

        return [QueryNode(table=entity_table, column=order_col, role="order")]

