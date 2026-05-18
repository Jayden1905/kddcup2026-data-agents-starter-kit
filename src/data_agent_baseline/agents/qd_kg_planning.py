"""KG path planning mixin for QuestionDrivenAgent."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelMessage
from data_agent_baseline.pipeline.kg_builder import KnowledgeGraph
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
    ) -> str:
        """LLM picks nodes from property graph → validate → format as grounding for SQL LLM."""
        if not db_path or not kg:
            return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        # --- Step 1: User intent (primary) + domain anchors (supporting context) ---
        anchor_text = self._extract_domain_anchors(question, knowledge_text, db_path=db_path)
        user_intent = self._detect_user_intent_only(question, kg_context=kg_context, anchor_text=anchor_text)

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

        if user_intent:
            self._log("user_intent", user_intent)

        # --- Step 1b: Deterministic column resolution from question words ---
        col_hints = self._resolve_question_columns(question, kg, anchor_text)
        if col_hints:
            user_intent = (user_intent or "") + f"\n\nCOLUMN RESOLUTION (use these exact columns):\n{col_hints}"

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
            self._log("kg_path", "Node picking failed, falling back")
            return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        if _det_comp_type:
            picked["computation_type"] = _det_comp_type

        self._log("kg_picked", json.dumps(picked, default=str))

        # Validate picks against actual schema
        output_nodes, filter_nodes, errors = self._validate_picked_nodes(picked, kg, db_path)
        if errors:
            self._log("kg_validation_errors", "; ".join(errors))
        if not output_nodes:
            self._log("kg_path", "No valid output nodes after validation, falling back")
            return self._call_semantic_grounding(question, kg_context, sample_data, knowledge_text, db_path, kg)

        # Disambiguate output: prefer entity-of-interest table
        _entity_pref_match = re.search(r'Entity of interest:\s*(\w+)', user_intent)
        if _entity_pref_match and kg:
            _pref_table = _entity_pref_match.group(1)
            _pref_schema = kg.get_table(_pref_table)
            if _pref_schema:
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

        # Ensure entity_of_interest is in path for aggregate queries
        _comp = picked.get("computation_type", "")
        if _comp in ("count", "sum", "avg", "count_distinct") and _entity_pref_match and kg:
            _eoi_table = _entity_pref_match.group(1)
            _all_tables = {n.table.lower() for n in output_nodes} | {n.table.lower() for n in filter_nodes}
            if _eoi_table.lower() not in _all_tables:
                _eoi_schema = kg.get_table(_eoi_table)
                if _eoi_schema and _eoi_schema.columns:
                    _eoi_col = _eoi_schema.columns[0].name
                    output_nodes.append(QueryNode(table=_eoi_table, column=_eoi_col, role="output"))
                    self._log("kg_inject_eoi", f"Added {_eoi_table}.{_eoi_col} to ensure path reaches entity")

        # Enforce multi-column output from intent's "Columns needed:" line
        if user_intent and output_nodes and kg:
            _cols_needed_match = re.search(r'Columns needed:\s*(.+)', user_intent)
            if _cols_needed_match:
                _needed_names = [
                    c.strip().lower() for c in _cols_needed_match.group(1).split(",")
                ]
                _existing_cols = {n.column.lower() for n in output_nodes}
                _out_table = output_nodes[0].table
                for needed in _needed_names:
                    if needed in _existing_cols:
                        continue
                    # Search for column in output table first, then all path tables
                    _found = False
                    for t in [kg.get_table(_out_table)] + [kg.get_table(n.table) for n in filter_nodes]:
                        if not t:
                            continue
                        for c in t.columns:
                            if c.name.lower() == needed or needed in c.name.lower():
                                if c.name.lower() not in _existing_cols:
                                    output_nodes.append(QueryNode(table=t.name, column=c.name, role="output"))
                                    _existing_cols.add(c.name.lower())
                                    self._log("kg_inject_col", f"Added {t.name}.{c.name} (from intent 'Columns needed')")
                                    _found = True
                                    break
                        if _found:
                            break

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

        # 2f. Output column refinements
        output_nodes = self._override_columns_from_question(question, output_nodes, filter_nodes, kg, db_path)

        _agg_comp = picked.get("computation_type", "simple_lookup")
        if _agg_comp in ("sum", "avg") and output_nodes and filter_nodes and db_path:
            _q_words_agg = set(re.findall(r'[a-z_]+', question.lower()))
            filter_tables = {n.table for n in filter_nodes}
            output_tables = {n.table for n in output_nodes}
            detail_tables = filter_tables - output_tables
            if detail_tables:
                for i, node in enumerate(output_nodes):
                    if node.column.lower() in _q_words_agg:
                        continue
                    ts = kg.get_table(node.table) if kg else None
                    if not ts:
                        continue
                    col_type = ""
                    for c in ts.columns:
                        if c.name == node.column:
                            col_type = c.sql_type.upper()
                            break
                    if col_type not in ("REAL", "FLOAT", "INTEGER", "INT", "NUMERIC"):
                        continue
                    if node.table in detail_tables:
                        continue
                    for dt in detail_tables:
                        dt_schema = kg.get_table(dt)
                        if not dt_schema:
                            continue
                        for c in dt_schema.columns:
                            if c.sql_type.upper() in ("REAL", "FLOAT", "INTEGER", "INT", "NUMERIC"):
                                cl = c.name.lower()
                                if cl.endswith("id") or cl.startswith("link"):
                                    continue
                                output_nodes[i] = QueryNode(table=dt, column=c.name, role="output")
                                self._log("agg_detail_fix",
                                    f"SUM/AVG target: {node.table}.{node.column} → {dt}.{c.name}")
                                break
                        break

        _comp_type_for_resolve = picked.get("computation_type", "simple_lookup")
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
        _shape_match = re.search(r'Answer shape:\s*(\w+)', user_intent or "")
        _grain_match = re.search(r'Grain:\s*(.+)', user_intent or "")
        if _shape_match and _shape_match.group(1).lower() == "list":
            _ct = picked.get("computation_type", "")
            if _ct in ("count", "sum", "avg"):
                picked["computation_type"] = "grouped_list"

        grounding = self._format_kg_plan_as_grounding(path, None, picked, "", kg=kg, db_path=db_path, user_intent=user_intent)

        # Inject rules engine directives into grounding
        if engine_out.sql_directives:
            grounding += "\n" + "\n".join(engine_out.sql_directives)

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
                    # For numeric columns: always show range (helps LLM reason about normal/abnormal)
                    if col.sql_type.upper() in ("REAL", "FLOAT", "INTEGER", "INT", "NUMERIC"):
                        if col.name.lower() not in ("id",) and not col.name.lower().endswith("_id"):
                            if hasattr(table, 'col_stats') and col.name in table.col_stats:
                                s = table.col_stats[col.name]
                                if "min" in s and "max" in s:
                                    parts.append(f"range: [{s['min']}, {s['max']}]")
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

        intent_section = f"\nUSER INTENT (primary — this is what the user wants):\n{user_intent}" if user_intent else ""
        anchor_section = f"\nDOMAIN KNOWLEDGE (supports the intent — defines terms and valid values):\n{anchor_text[:2000]}" if anchor_text else ""

        # Question-filtered context: only surface entries relevant to THIS question
        q_words = set(question.lower().split())
        q_lower = question.lower()

        # Concept map: only entries whose concept word appears in the question
        concept_section = ""
        if hasattr(kg, 'concept_map') and kg.concept_map:
            relevant = [
                (k, v) for k, v in kg.concept_map.items()
                if k in q_lower or any(w in q_words for w in k.split())
            ]
            if relevant:
                cmap_lines = [f'  "{k}" → {v}' for k, v in relevant]
                concept_section = "\nCONCEPT MAP:\n" + "\n".join(cmap_lines)

        # Ontology: show vocab for columns whose meanings match the question
        # Show FULL vocab (all values) so the LLM can distinguish between columns
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
                    # Show purpose + all values so LLM can distinguish columns
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
                    "\nVALUE MEANINGS (when multiple columns match the same concept, "
                    "pick the column whose PURPOSE best fits — use only ONE column per concept):\n"
                    + "\n".join(ont_lines[:10])
                )

        # Value pre-grounding: look up question terms in the KG value index
        # so the LLM knows exactly WHERE each filter value lives
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
            # Build candidate phrases: bigrams first (more specific), then unigrams
            _candidates: list[str] = []
            for i in range(len(_q_tokens_lower) - 1):
                a, b = _q_tokens_lower[i], _q_tokens_lower[i + 1]
                if a not in _stop and b not in _stop and len(a) > 1 and len(b) > 1:
                    _candidates.append(f"{_q_tokens[i]} {_q_tokens[i + 1]}")
            for i, t in enumerate(_q_tokens_lower):
                if t not in _stop and len(t) > 2:
                    _candidates.append(_q_tokens[i])
            # Look up each candidate, deduplicate by (table, column, value)
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
                    "\nVALUE MATCHES (these values EXIST in the DB — use these exact columns for filters):\n"
                    + "\n".join(_value_lines)
                )

        prompt = f"""QUESTION: {question}

PROPERTY GRAPH:
{graph_dump[:6000]}
{intent_section}{anchor_section}{concept_section}{ontology_section}{value_section}

HOW TO READ THE GRAPH:
- Each TABLE has columns with their SQL types. Some columns show known "values:" — these are categorical values in that column.
- Columns marked "→ references Table.Column" are FOREIGN KEYS that store IDs pointing to another table. To get human-readable values (names, labels), you must SELECT from the referenced table, not the FK column itself.
- JOIN RELATIONSHIPS show which columns link tables together. SEMANTIC LINKS show columns with matching ID patterns (useful for bridge/junction tables).
- When you need to connect tables that have no direct FK, look for a BRIDGE TABLE that has FKs to both.

YOUR TASK: Pick exact columns from this graph to answer the question.

PRIORITY (resolve conflicts in this order):
1. USER INTENT — this tells you what the user wants returned and how to compute it. Follow it.
2. DOMAIN KNOWLEDGE — defines what terms mean in this database. When a question term (like "track number") is defined in domain knowledge, use the column it maps to — even if another column literally contains the word.
3. VALUE MATCHES — if a value is confirmed to exist in a specific column, filter on that column.
4. QUESTION — the user's exact words. Only use literal column name matching if 1-3 don't resolve it.
5. CONCEPT MAP — if a question term doesn't match any column name, check the concept map for the best column mapping.
6. Your own inference — only if 1-5 don't apply.

Return ONLY JSON:
{{
  "what_user_wants": "one sentence restating the expected output",
  "select_columns": ["Table.Column", ...],
  "filter_conditions": [
    {{"column": "Table.Column", "operator": "= | > | < | >= | <= | LIKE | !=", "value": "..."}}
  ],
  "order_by": {{"column": "Table.Column", "direction": "ASC | DESC"}} or null,
  "computation_type": "simple_lookup | count | sum | avg | min_max | ratio | percentage | derived",
  "derived_logic": "natural language description of the computation (only if computation_type is 'derived')"
}}

CONSTRAINTS:
- You may ONLY use Table.Column names that appear in the PROPERTY GRAPH above.
- You may ONLY use values from "values:" lists, DOMAIN KNOWLEDGE, or the QUESTION itself.
- If a column is a FK reference (marked with →), do NOT select it directly — instead JOIN to the referenced table and select the human-readable column there (e.g. the name/label column, not the ID).
- For "best/lowest/highest/fastest" questions, use order_by instead of equality filters on position/rank columns.
- For filter values, use the EXACT format shown in "values:" or DOMAIN KNOWLEDGE. If unsure of the format, use LIKE with a partial match.
- "List all X" / "What are the X" without specifying WHICH columns to show → select ONLY the primary key or identifier column of the entity (e.g. trans_id, Id, order_id). Do NOT guess descriptive columns unless the question explicitly asks for them (e.g. "list the names", "show dates and amounts").
- ONE filter per concept: if VALUE MATCHES shows which column holds a value, use THAT column. Otherwise, if two columns in the same table both contain a matching value, pick the column whose overall vocabulary best describes the concept. Do not filter on both — redundant filters risk returning 0 rows.
- "Normal level" means the value falls WITHIN the healthy range — use TWO conditions: >= lower_bound AND <= upper_bound (e.g. WBC >= 3.5 AND WBC <= 9.0 keeps only normal values). "Abnormal level" means the value falls OUTSIDE the healthy range — use a SINGLE condition on whichever side the data actually falls (e.g. Column < lower OR Column > upper), OR use IS NOT NULL if the column's data range (shown above) is entirely outside the standard normal range."""

        messages = [ModelMessage(role="user", content=prompt)]
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
                # Multiple columns have the value — check which is more specific
                # The column with FEWER matching rows is more likely the correct filter
                # (e.g., operation='VYBER' is more specific than type='VYBER')
                best_node = valid_cols[0]
                best_count = float("inf")
                try:
                    conn2 = sqlite3.connect(str(db_path), timeout=5)
                    for node in valid_cols:
                        try:
                            cnt = conn2.execute(
                                f'SELECT COUNT(*) FROM "{node.table}" WHERE "{node.column}" = ? COLLATE NOCASE',
                                (node.value,)
                            ).fetchone()[0]
                            if cnt < best_count:
                                best_count = cnt
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

    def _override_columns_from_question(
        self, question: str, output_nodes: list[QueryNode],
        filter_nodes: list[QueryNode], kg: KnowledgeGraph,
        db_path: Path | None = None,
    ) -> list[QueryNode]:
        """Deterministically replace output columns when the question contains an exact column name
        that exists in the graph but the picker chose a different column with the same structural role.

        E.g., question says "type" and there's a column named "type" in the graph,
        but picker chose "category" — swap to "type".
        """
        if not kg or not kg.graph or not output_nodes:
            return output_nodes
        q_lower = question.lower()
        q_tokens = set(re.findall(r'[a-z_]+', q_lower))

        # Columns already used as filters should NOT be considered as output replacements
        filter_col_names = {n.column.lower() for n in filter_nodes}

        # Build map: col_name_lower → [(table, col_name, sql_type)]
        exact_cols: dict[str, list[tuple[str, str, str]]] = {}
        for col_id, col_node in kg.graph.columns.items():
            cn = col_node.name.lower()
            if cn.endswith("id") or "link" in cn or "ref" in cn:
                continue
            table_schema = kg.get_table(col_node.table_id)
            sql_type = ""
            if table_schema:
                for c in table_schema.columns:
                    if c.name == col_node.name:
                        sql_type = c.sql_type.upper()
                        break
            exact_cols.setdefault(cn, []).append((col_node.table_id, col_node.name, sql_type))

        # Detect tokens near a numeric value — likely filter context, not output
        # e.g. "number less than 20", "age over 30", "round 5"
        _q_words = q_lower.split()
        _filter_context_tokens: set[str] = set()
        for i, w in enumerate(_q_words):
            if w in exact_cols:
                # Check if any word within 3 positions contains digits
                window = _q_words[max(0, i - 2):i + 4]
                if any(re.search(r'\d', tok) for tok in window):
                    _filter_context_tokens.add(w)

        # Find question tokens that exactly match column names (exclude filter columns)
        matched_cols: dict[str, list[tuple[str, str, str]]] = {}
        for token in q_tokens:
            if len(token) < 3:
                continue
            if token in filter_col_names:
                continue
            if token in _filter_context_tokens:
                continue
            if token in exact_cols:
                matched_cols[token] = exact_cols[token]

        # Also match multi-word column names (CamelCase or snake_case) against question words
        # e.g., "UpVotes" → ["up", "votes"], "up_votes" → ["up", "votes"]
        for col_lower, candidates in exact_cols.items():
            if col_lower in matched_cols:
                continue
            original_name = candidates[0][1]
            # Split CamelCase: "UpVotes" → ["Up", "Votes"]
            parts = re.findall(r'[A-Z][a-z]+|[a-z]+', original_name)
            # Split snake_case: "up_votes" → ["up", "votes"]
            if len(parts) < 2 and "_" in original_name:
                parts = [p for p in original_name.lower().split("_") if p]
            if len(parts) >= 2:
                all_in_q = all(p.lower() in q_tokens for p in parts)
                if all_in_q and col_lower not in filter_col_names:
                    matched_cols[col_lower] = candidates

        if not matched_cols:
            return output_nodes

        # For each output node, check if there's an exact-match column
        # that the question directly references
        new_output: list[QueryNode] = []
        for node in output_nodes:
            node_type = ""
            table_schema = kg.get_table(node.table)
            if table_schema:
                for c in table_schema.columns:
                    if c.name == node.column:
                        node_type = c.sql_type.upper()
                        break
            is_text = node_type in ("TEXT", "VARCHAR", "CHAR", "STRING")
            is_numeric = node_type in ("REAL", "FLOAT", "NUMERIC", "INTEGER", "INT")

            # Check if this node's column is already directly referenced in the question
            node_col_lower = node.column.lower()
            node_mentioned = node_col_lower in q_tokens
            if not node_mentioned:
                # Also check CamelCase or snake_case split
                node_parts = re.findall(r'[A-Z][a-z]+|[a-z]+', node.column)
                if len(node_parts) < 2 and "_" in node.column:
                    node_parts = [p for p in node.column.lower().split("_") if p]
                if len(node_parts) >= 2:
                    node_mentioned = all(p.lower() in q_tokens for p in node_parts)
                # Partial match: require ALL substantive parts to appear in question
                # (skip parts that are table names — they're structural, not semantic)
                if not node_mentioned and node_parts:
                    table_names = {t.name.lower() for t in kg.tables} if kg else set()
                    semantic_parts = [p.lower() for p in node_parts if p.lower() not in table_names and len(p) >= 4]
                    if semantic_parts and all(
                        pl in q_tokens or pl + "s" in q_tokens or pl.rstrip("s") in q_tokens
                        for pl in semantic_parts
                    ):
                        node_mentioned = True

            if not node_mentioned:
                # This output column is NOT directly referenced in the question.
                # Check if there's a matched column with compatible type.
                # Skip columns already in another output node.
                existing_cols = {(n.table, n.column) for n in output_nodes}
                # Tables already in the query (reachable without extra joins)
                query_tables = {n.table.lower() for n in output_nodes} | {n.table.lower() for n in filter_nodes}
                # Tables with direct FK edges to query tables
                reachable_tables = set(query_tables)
                if kg.graph and kg.graph.fk_edges:
                    for edge in kg.graph.fk_edges:
                        src_tbl = edge.src.split(".")[0].lower() if "." in edge.src else ""
                        dst_tbl = edge.dst.split(".")[0].lower() if "." in edge.dst else ""
                        if src_tbl in query_tables:
                            reachable_tables.add(dst_tbl)
                        if dst_tbl in query_tables:
                            reachable_tables.add(src_tbl)

                replacement = None
                type_set = (
                    ("TEXT", "VARCHAR", "CHAR", "STRING") if is_text
                    else ("REAL", "FLOAT", "NUMERIC", "INTEGER", "INT") if is_numeric
                    else ()
                )
                if type_set:
                    # Collect all reachable, type-compatible candidates
                    all_candidates = []
                    for token, candidates in matched_cols.items():
                        for tbl, col_name, sql_type in candidates:
                            if sql_type in type_set and (tbl, col_name) not in existing_cols:
                                if tbl.lower() in reachable_tables:
                                    all_candidates.append((tbl, col_name))
                    # Try each until one has data
                    for cand in all_candidates:
                        if self._db_check(db_path, cand[0], cand[1]):
                            replacement = cand
                            break
                if replacement:
                    new_output.append(QueryNode(
                        table=replacement[0], column=replacement[1], role="output",
                    ))
                    continue
            new_output.append(node)

        return new_output

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

    def _resolve_question_columns(
        self, question: str, kg: KnowledgeGraph, anchor_text: str,
    ) -> str:
        """Deterministically resolve question words to exact columns in the graph.

        Two strategies:
        1. Exact match: question word == column name → hard hint
        2. Table mention: question mentions table → surface its structural columns
        """
        if not kg or not kg.graph:
            return ""
        q_lower = question.lower()
        q_tokens = set(re.findall(r'[a-z_]+', q_lower))
        hints: list[str] = []

        # --- Strategy 1: Exact column name matches ---
        # If a question word exactly matches a column name, that's a direct reference
        col_by_name: dict[str, list[str]] = {}  # col_name_lower → [table.col, ...]
        for col_id, col_node in kg.graph.columns.items():
            cn = col_node.name.lower()
            # Skip FK/PK columns — they're structural, not meaningful references
            if cn.endswith("id") or "link" in cn or "ref" in cn:
                continue
            col_by_name.setdefault(cn, []).append(f'"{col_node.table_id}"."{col_node.name}"')

        for token in q_tokens:
            if len(token) < 3:
                continue
            if token in col_by_name:
                cols = col_by_name[token]
                hints.append(f'"{token}" in question matches column: {", ".join(cols)}')

        # --- Strategy 1b: Multi-word column name matching ---
        # Column names like "home_team_goal" or "Phone" should match "home team goal" or "phone"
        for col_lower, col_refs in col_by_name.items():
            col_words = set(col_lower.replace("_", " ").split())
            if len(col_words) >= 2 and col_words.issubset(q_tokens):
                if not any(col_lower in h for h in hints):
                    hints.append(f'question words match multi-word column "{col_lower}": {", ".join(col_refs)}')

        # --- Strategy 2: Table mention → structural columns ---
        for table in kg.tables:
            table_lower = table.name.lower()
            table_stem = table_lower.rstrip("s").rstrip("e")
            mentioned = any(
                (len(table_stem) >= 4 and table_stem in tok) or
                (len(tok) >= 4 and tok in table_lower)
                for tok in q_tokens
            )
            if not mentioned:
                continue

            descriptor_col = None
            measure_col = None
            for col in table.columns:
                cn = col.name.lower()
                is_fk_or_pk = ("id" in cn or "link" in cn or "ref" in cn or "key" in cn)
                is_text = col.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", "STRING")
                is_numeric = col.sql_type.upper() in ("REAL", "FLOAT", "NUMERIC", "INTEGER", "INT")

                if is_text and not is_fk_or_pk and not descriptor_col:
                    descriptor_col = col.name
                if is_numeric and not is_fk_or_pk and not measure_col:
                    measure_col = col.name

            if descriptor_col or measure_col:
                parts = [f'The question references "{table.name}" table:']
                if descriptor_col:
                    parts.append(f'  descriptor column → "{table.name}"."{descriptor_col}"')
                if measure_col:
                    parts.append(f'  measure column → "{table.name}"."{measure_col}"')
                hints.append("\n".join(parts))

        return "\n".join(hints)

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
        stop_words = {"the", "which", "what", "how", "many", "list", "give",
                      "find", "show", "all", "are", "for", "from", "with",
                      "calculate", "identify", "please", "among", "total"}
        # Track which DB column each named entity was found in (for better sanity messages)
        entity_found_in: dict[str, str] = {}
        table_names_lower = {t.name.lower() for t in kg.tables} if kg else set()
        if kg and kg.graph:
            for word in single_caps:
                if word.lower() in stop_words:
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
                            context_words = set(re.findall(r'[a-z]{3,}', context_window)) - stop_words
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
        if kg and kg.graph:
            filter_col_names = {n.column.lower() for n in filter_nodes}
            output_col_names = {n.column.lower() for n in output_nodes}
            q_words = set(re.findall(r'[a-z]+', q_lower))
            for col_id, col_node in kg.graph.columns.items():
                col_lower = col_node.name.lower()
                if col_lower in q_words and col_lower not in filter_col_names and col_lower not in output_col_names:
                    # Check if this column has boolean/status-like values
                    val_nodes = kg.graph.get_column_values(col_id)
                    if val_nodes:
                        vals = [str(v.value).lower() for v in val_nodes]
                        is_boolean = set(vals) <= {"true", "false", "yes", "no", "0", "1", "t", "f"}
                        if is_boolean and len(vals) <= 4:
                            issues.append(
                                f'The question mentions "{col_lower}" which is a column in '
                                f'"{col_node.table_id}" with values {vals}. '
                                f'Add a filter: "{col_node.table_id}"."{col_node.name}" = \'true\' (or appropriate value).'
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

