"""Semantic grounding mixin for QuestionDrivenAgent."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelMessage
from data_agent_baseline.agents.qd_prompts import _build_semantic_prompt, _format_grounding_for_sql
from data_agent_baseline.pipeline.kg_builder import KnowledgeGraph, build_kg_from_sqlite


class SemanticGroundingMixin:
    """Semantic grounding LLM call and validation methods."""

    def _call_semantic_grounding(
        self,
        question: str,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
        db_path: Path | None = None,
        kg: KnowledgeGraph | None = None,
    ) -> str:
        """Validate-first-then-ground: deterministic pre-validation → one-shot constrained grounding."""

        # Extract domain anchors
        anchor_text = self._extract_domain_anchors(question, knowledge_text, db_path=db_path)

        # --- Step 1: Intent Detection (stable — never redone) ---
        user_intent = self._detect_user_intent_only(question, kg_context=kg_context, anchor_text=anchor_text)
        if user_intent:
            self._log("user_intent", user_intent)

        # Pre-grounding: extract formula + column mappings from knowledge, scoped to user intent
        knowledge_guidance = ""
        if knowledge_text:
            knowledge_guidance = self._extract_knowledge_guidance(question, knowledge_text, user_intent, kg_context=kg_context)
            if knowledge_guidance:
                self._log("knowledge_guidance", knowledge_guidance)

        self._log("grounding_iter", "--- Grounding ---")

        effective_anchor = anchor_text

        # --- Step 2: Table Selection ---
        selected_tables = self._grounding_select_tables(
            question, kg_context, effective_anchor, db_path, kg=kg, user_intent=user_intent
        )

        # Deterministic: if question words derive from column names, surface the column definition
        if knowledge_guidance and db_path and selected_tables:
            col_notes = self._match_question_words_to_columns(question, db_path, selected_tables, anchor_text)
            if col_notes:
                knowledge_guidance = f"{col_notes}\n\n{knowledge_guidance}"

        # --- Step 3: Build focused schema ---
        focused_schema = ""
        if selected_tables and db_path:
            focused_schema = self._build_focused_schema_for_grounding(db_path, selected_tables, question, kg=kg)
        grounding_schema = focused_schema if focused_schema else kg_context

        # Extract matching SQL from domain knowledge
        domain_sql_ref = self._extract_matching_domain_sql(question, anchor_text)
        if domain_sql_ref:
            effective_anchor = (
                f"⚠️ REFERENCE SQL FROM DOMAIN KNOWLEDGE (use this exact pattern — do NOT add extra conversion or transformation):\n"
                f"  {domain_sql_ref}\n\n{effective_anchor}"
            )
            self._log("domain_sql_match", domain_sql_ref)

        # --- Step 4: DETERMINISTIC PRE-VALIDATION ---
        # Build validated context BEFORE the grounding LLM call
        validated = self._build_validated_context(
            question, db_path, selected_tables, kg, user_intent, effective_anchor
        )

        # --- PRE-GROUNDING EVIDENCE ---
        ambiguous_evidence = ""
        if db_path and selected_tables and len(selected_tables) > 1:
            ambiguous_evidence = self._scan_ambiguous_columns(db_path, selected_tables, question)
            if ambiguous_evidence:
                self._log("ambiguous_cols_evidence", ambiguous_evidence)

        literal_matches = self._scan_literal_column_matches(db_path, selected_tables, question)
        if literal_matches:
            ambiguous_evidence = f"{ambiguous_evidence}\n\n{literal_matches}" if ambiguous_evidence else literal_matches

        # --- Step 5: ONE constrained grounding LLM call ---
        # Inject validated facts as hard constraints so the LLM doesn't need to guess
        validated_section = self._format_validated_context(validated)

        prompt = _build_semantic_prompt(
            question=question,
            kg_context=grounding_schema,
            sample_data=sample_data if not focused_schema else "",
            anchor_text=effective_anchor,
            previous_attempt="",
            ambiguous_columns=ambiguous_evidence,
            user_intent=user_intent,
            knowledge_guidance=knowledge_guidance,
            feedback=validated_section if validated_section else "",
        )
        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages, thinking=False)
        grounding = self._parse_json(raw)

        if not isinstance(grounding, dict) or not grounding:
            self._log("semantic_grounding", "(failed to parse, retrying)")
            raw = self._model_call_with_retry(messages, thinking=False)
            grounding = self._parse_json(raw)

        if not isinstance(grounding, dict) or not grounding:
            self._log("semantic_grounding", "(failed to parse after retry)")
            return ""

        self._log("grounding_v1", json.dumps(grounding, default=str))

        # --- Step 6: Merge validated facts into grounding (deterministic, no LLM) ---
        grounding = self._merge_validated_into_grounding(grounding, validated, db_path, selected_tables, kg)

        # Verify all question entities appear in formula/known_values
        if grounding:
            grounding = self._verify_filter_completeness(question, grounding)

        # Deterministic enrichment: add any schema columns whose name matches a question word
        if db_path:
            grounding = self._enrich_data_requirements(db_path, question, grounding)

        formatted = _format_grounding_for_sql(grounding)

        self._log("semantic_grounding_final", formatted if formatted else "(empty)")

        return formatted

    def _build_validated_context(
        self,
        question: str,
        db_path: Path | None,
        selected_tables: list[str],
        kg: KnowledgeGraph | None,
        user_intent: str,
        anchor_text: str,
    ) -> dict[str, Any]:
        """Deterministic pre-validation using KG as ground truth for schema facts,
        DB probes only for filter value existence. Returns validated facts."""
        validated: dict[str, Any] = {
            "join_paths": [],
            "filter_probes": {},
            "filter_overrides": {},
            "preagg_columns": [],
            "format_notes": [],
        }
        if not db_path or not db_path.exists() or not selected_tables:
            return validated

        if kg is None:
            kg = build_kg_from_sqlite(db_path)

        # --- 1. Join paths from KG (already validated by value overlap during KG build) ---
        tables_lower = {t.lower() for t in selected_tables}
        for src_table, fk in kg.all_foreign_keys():
            if src_table.lower() in tables_lower and fk.ref_table.lower() in tables_lower:
                clause = f"{src_table}.{fk.column} = {fk.ref_table}.{fk.ref_column}"
                if clause not in validated["join_paths"]:
                    validated["join_paths"].append(clause)

        if validated["join_paths"]:
            self._log("grounding_join_paths", str(validated["join_paths"]))

        # --- 2. Detect pre-aggregated columns from KG column names + stats ---
        for tname in selected_tables:
            table_schema = kg.get_table(tname)
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
                        validated["preagg_columns"].append(f"{tname}.{col.name}")

        # --- 3. Probe filter values against DB ---
        # Extract candidates from question: quoted strings + proper nouns
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
        proper_nouns = re.findall(r'(?:the |in |for |of |from )([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)', question)
        filter_candidates = quoted + proper_nouns

        # Extract domain-anchor filter conditions (e.g., "Filter condition: Thrombosis = 2")
        anchor_filters: list[tuple[str, str]] = []
        if anchor_text:
            for m in re.finditer(r'Filter condition:\s*(\w+)\s*=\s*(\S+)', anchor_text):
                anchor_filters.append((m.group(1), m.group(2)))

        conn = sqlite3.connect(str(db_path))
        try:
            # 3a. Verify domain-anchor filters using KG to find the right table
            for col_name, val in anchor_filters:
                for tname in selected_tables:
                    table_schema = kg.get_table(tname)
                    if not table_schema:
                        continue
                    real_col = next((c.name for c in table_schema.columns if c.name.lower() == col_name.lower()), None)
                    if not real_col:
                        continue
                    try:
                        cnt = conn.execute(
                            f'SELECT COUNT(*) FROM "{tname}" WHERE "{real_col}" = ?', (val,)
                        ).fetchone()[0]
                        if cnt > 0:
                            key = f"{tname}.{real_col}"
                            validated["filter_probes"][key] = {
                                "value": val, "count": cnt, "method": "exact"
                            }
                            self._log("filter_verified", f"{key}='{val}' exists ({cnt} rows)")
                    except Exception:
                        pass
                    break

            # 3b. Probe question filter candidates against text columns (from KG)
            for candidate in filter_candidates:
                if len(candidate) < 2:
                    continue
                found_exact = False
                for tname in selected_tables:
                    if found_exact:
                        break
                    table_schema = kg.get_table(tname)
                    if not table_schema:
                        continue
                    text_cols = [c.name for c in table_schema.columns
                                 if c.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", "")]
                    for col in text_cols:
                        try:
                            cnt = conn.execute(
                                f'SELECT COUNT(*) FROM "{tname}" WHERE "{col}" = ? COLLATE NOCASE',
                                (candidate,)
                            ).fetchone()[0]
                            if cnt > 0:
                                key = f"{tname}.{col}"
                                if key not in validated["filter_probes"]:
                                    validated["filter_probes"][key] = {
                                        "value": candidate, "count": cnt, "method": "exact"
                                    }
                                found_exact = True
                                break
                        except Exception:
                            continue
                # LIKE fallback
                if not found_exact:
                    for tname in selected_tables:
                        table_schema = kg.get_table(tname)
                        if not table_schema:
                            continue
                        text_cols = [c.name for c in table_schema.columns
                                     if c.sql_type.upper() in ("TEXT", "VARCHAR", "CHAR", "")]
                        for col in text_cols:
                            try:
                                like_cnt = conn.execute(
                                    f'SELECT COUNT(*) FROM "{tname}" WHERE "{col}" LIKE ? COLLATE NOCASE',
                                    (f'%{candidate}%',)
                                ).fetchone()[0]
                                if like_cnt > 0:
                                    key = f"{tname}.{col}"
                                    if key not in validated["filter_probes"]:
                                        validated["filter_probes"][key] = {
                                            "value": candidate, "count": like_cnt, "method": "like",
                                            "override": f'"{col}" LIKE \'%{candidate}%\' COLLATE NOCASE',
                                        }
                                        validated["filter_overrides"][key] = (
                                            f'"{col}" LIKE \'%{candidate}%\' COLLATE NOCASE'
                                        )
                                        validated["format_notes"].append(
                                            f"For {tname}.{col}: use WHERE \"{col}\" LIKE '%{candidate}%' COLLATE NOCASE"
                                        )
                                    break
                            except Exception:
                                continue
        finally:
            conn.close()

        return validated

    def _format_validated_context(self, validated: dict[str, Any]) -> str:
        """Format validated facts as constraints for the grounding LLM."""
        parts: list[str] = []

        if validated["join_paths"]:
            parts.append(
                "VALIDATED JOIN PATHS (verified to produce rows — use these exactly):\n"
                + "\n".join(f"  {jp}" for jp in validated["join_paths"])
            )

        if validated["filter_probes"]:
            lines: list[str] = []
            for key, info in validated["filter_probes"].items():
                if info["method"] == "exact":
                    lines.append(f"  {key} = '{info['value']}' ({info['count']} rows)")
                else:
                    lines.append(f"  {key} LIKE '%{info['value']}%' ({info['count']} rows) — use LIKE, not =")
            if lines:
                parts.append(
                    "VALIDATED FILTER VALUES (verified against DB — use these in known_values):\n"
                    + "\n".join(lines)
                )

        if validated["preagg_columns"]:
            parts.append(
                "PRE-AGGREGATED COLUMNS (already store per-row aggregates — use WHERE, not AVG()):\n"
                + "\n".join(f"  {col}" for col in validated["preagg_columns"])
            )

        if not parts:
            return ""
        return "⚠️ VALIDATED FACTS (verified against actual database — these override assumptions):\n" + "\n".join(parts)

    def _merge_validated_into_grounding(
        self,
        grounding: dict[str, Any],
        validated: dict[str, Any],
        db_path: Path | None,
        selected_tables: list[str],
        kg: KnowledgeGraph | None,
    ) -> dict[str, Any]:
        """Deterministically merge validated DB facts into grounding output."""
        # Merge join paths (validated ones take priority)
        if validated["join_paths"]:
            grounding["join_paths"] = validated["join_paths"]

        # Detect COUNT(DISTINCT) need: when a table has many rows per entity
        if (
            db_path
            and db_path.exists()
            and kg
            and selected_tables
            and grounding.get("computation_type", "").lower()
            in ("count", "ratio", "percentage")
        ):
            self._inject_distinct_hint(grounding, db_path, selected_tables, kg)

        # Merge filter overrides for LIKE patterns
        if validated["filter_overrides"]:
            existing_overrides = grounding.get("filter_overrides", {})
            known_values = grounding.get("known_values", {})
            for key, override in validated["filter_overrides"].items():
                if key in known_values and key not in existing_overrides:
                    existing_overrides[key] = override
                    self._log("filter_override_applied", f"{key}: {override}")
            grounding["filter_overrides"] = existing_overrides

        # Merge format notes
        if validated["format_notes"]:
            existing_notes = grounding.get("data_format_notes", [])
            for note in validated["format_notes"]:
                if note not in existing_notes:
                    existing_notes.append(note)
            grounding["data_format_notes"] = existing_notes

        # Validate known_values against DB — fix format mismatches
        if db_path and grounding.get("known_values"):
            grounding = self._validate_known_values_against_db(db_path, grounding, validated)

        return grounding

    def _inject_distinct_hint(
        self,
        grounding: dict[str, Any],
        db_path: Path,
        selected_tables: list[str],
        kg: KnowledgeGraph,
    ) -> None:
        """When counting entities via a many-rows-per-entity table, inject COUNT(DISTINCT) guidance."""
        tables_lower = {t.lower(): t for t in selected_tables}
        conn = sqlite3.connect(str(db_path))
        try:
            for src_table, fk in kg.all_foreign_keys():
                if src_table.lower() not in tables_lower:
                    continue
                if fk.ref_table.lower() not in tables_lower:
                    continue
                if fk.column.lower() != fk.ref_column.lower():
                    continue
                src_schema = kg.get_table(src_table)
                if not src_schema or src_schema.row_count < 10:
                    continue
                try:
                    distinct = conn.execute(
                        f'SELECT COUNT(DISTINCT "{fk.column}") FROM "{src_table}"'
                    ).fetchone()[0]
                except Exception:
                    continue
                if distinct < 1:
                    continue
                rows_per_entity = src_schema.row_count / distinct
                if rows_per_entity > 10:
                    overrides = grounding.get("_semantic_overrides", [])
                    hint = (
                        f'ENTITY COUNTING: Table "{src_table}" has {src_schema.row_count} rows but only '
                        f'{distinct} distinct "{fk.column}" values ({rows_per_entity:.0f} rows per entity). '
                        f'If the question counts "{fk.ref_table}" entities (how many {fk.ref_table}s), '
                        f'use COUNT(DISTINCT "{src_table}"."{fk.column}") — NOT COUNT(*). '
                        f'But if it counts "{src_table}" rows themselves, use COUNT(*).'
                    )
                    if not any("COUNT(DISTINCT" in o for o in overrides):
                        overrides.append(hint)
                        grounding["_semantic_overrides"] = overrides
                        self._log("distinct_hint", hint)
                    return
        finally:
            conn.close()

    def _validate_known_values_against_db(
        self,
        db_path: Path,
        grounding: dict[str, Any],
        validated: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify that grounding's known_values actually exist in DB. Fix format mismatches."""
        known_values = grounding.get("known_values", {})
        filter_overrides = grounding.get("filter_overrides", {})
        format_notes = grounding.get("data_format_notes", [])

        try:
            conn = sqlite3.connect(str(db_path))
        except Exception:
            return grounding

        try:
            for col_key, values in list(known_values.items()):
                if "." not in col_key or not values:
                    continue
                table_name, col_name = col_key.split(".", 1)
                # Skip comparison operators, SQL expressions, numerics
                if all(re.match(r'^[<>!=]', str(v).strip()) for v in values if v):
                    continue
                if any(any(kw in str(v).upper() for kw in ('COUNT', 'SUM', 'AVG', 'MIN', 'MAX')) for v in values if v):
                    continue
                if all(re.match(r'^-?\d+\.?\d*$', str(v).strip()) for v in values if v):
                    continue
                if len(values) > 5:
                    continue

                # Already validated via pre-validation
                if col_key in validated["filter_overrides"]:
                    if col_key not in filter_overrides:
                        filter_overrides[col_key] = validated["filter_overrides"][col_key]
                    continue

                corrected_values = []
                for val in values:
                    val_str = str(val)
                    try:
                        # Check LIKE wildcards first
                        if '%' in val_str:
                            like_cnt = conn.execute(
                                f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" LIKE ? COLLATE NOCASE',
                                (val_str,)
                            ).fetchone()[0]
                            if like_cnt > 0:
                                corrected_values.append(val)
                                filter_overrides[col_key] = f'"{col_name}" LIKE \'{val_str}\' COLLATE NOCASE'
                                format_notes.append(
                                    f"For {table_name}.{col_name}: use WHERE \"{col_name}\" LIKE '{val_str}' COLLATE NOCASE"
                                )
                                self._log("filter_verified", f"{col_key}='{val}' matches {like_cnt} rows via LIKE")
                                continue

                        # Exact match
                        cnt = conn.execute(
                            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" = ?',
                            (val_str,)
                        ).fetchone()[0]
                        if cnt > 0:
                            corrected_values.append(val)
                            self._log("filter_verified", f"{col_key}='{val}' exists ({cnt} rows)")
                            continue

                        # Case-insensitive check
                        ci_row = conn.execute(
                            f'SELECT "{col_name}" FROM "{table_name}" WHERE "{col_name}" = ? COLLATE NOCASE LIMIT 1',
                            (val_str,)
                        ).fetchone()
                        if ci_row:
                            corrected_values.append(str(ci_row[0]))
                            self._log("filter_verified", f"{col_key}='{val}' → case-corrected to '{ci_row[0]}'")
                            continue

                        # LIKE fallback — try prefix match
                        like_cnt = conn.execute(
                            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" LIKE ? COLLATE NOCASE',
                            (f'%{val_str}%',)
                        ).fetchone()[0]
                        if like_cnt > 0:
                            corrected_values.append(val)
                            filter_overrides[col_key] = f'"{col_name}" LIKE \'%{val_str}%\' COLLATE NOCASE'
                            format_notes.append(
                                f"For {table_name}.{col_name}: use WHERE \"{col_name}\" LIKE '%{val_str}%' COLLATE NOCASE"
                            )
                            self._log("filter_corrected", f"{col_key}: '{val}' → LIKE '%{val_str}%'")
                            continue

                        # Value not found — check if any distinct value is a prefix/abbreviation
                        # of the queried concept (e.g. element "iodine" → stored as "i")
                        prefix_match = conn.execute(
                            f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                            f'WHERE "{col_name}" IS NOT NULL AND ? LIKE "{col_name}" || \'%\' '
                            f'COLLATE NOCASE LIMIT 5',
                            (val_str,)
                        ).fetchall()
                        if prefix_match and len(prefix_match) == 1:
                            corrected_values.append(str(prefix_match[0][0]))
                            self._log("filter_verified", f"{col_key}='{val}' → prefix-matched to '{prefix_match[0][0]}'")
                            continue
                        # Keep original but note
                        corrected_values.append(val)
                        self._log("filter_verified", f"{col_key}='{val}' NOT found in DB")
                    except Exception:
                        corrected_values.append(val)

                known_values[col_key] = corrected_values

            grounding["known_values"] = known_values
            grounding["filter_overrides"] = filter_overrides
            grounding["data_format_notes"] = format_notes
        finally:
            conn.close()

        return grounding

