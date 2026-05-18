"""Question-driven agent: semantic grounding + closed-loop SQL.

Pipeline:
  1. [Code] Scan context, consolidate structured data -> SQLite
  2. [Code] Deterministic doc extraction -> additional tables in SQLite
  3. [Code] Build KG from full SQLite (structured + extracted)
  4. [LLM] Semantic grounding: question + schema -> structured decomposition
  5. [LLM] Closed loop: SQL generation -> execute -> evaluate -> iterate
  6. [LLM] Answer formatting
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.pipeline.context_scanner import scan_context
from data_agent_baseline.pipeline.kg_builder import (
    KnowledgeGraph,
    build_kg_from_sqlite,
    build_ontology,
    classify_columns_with_llm,
    discover_joins_with_llm,
    format_kg_for_llm,
    profile_schema,
)
from data_agent_baseline.tools.knowledge_graph import consolidate_to_sqlite

# Mixin imports
from data_agent_baseline.agents.qd_utils import UtilsMixin
from data_agent_baseline.agents.qd_knowledge import KnowledgeMixin
from data_agent_baseline.agents.qd_schema_selection import SchemaSelectionMixin
from data_agent_baseline.agents.qd_filters import FilterProcessingMixin
from data_agent_baseline.agents.qd_kg_planning import KGPlanningMixin
from data_agent_baseline.agents.qd_grounding_format import GroundingFormatMixin
from data_agent_baseline.agents.qd_semantic_grounding import SemanticGroundingMixin
from data_agent_baseline.agents.qd_diagnostics import DiagnosticsMixin

# Module-level function imports
from data_agent_baseline.agents.qd_prompts import (
    CONSOLIDATED_DB_NAME,
    _build_sql_prompt,
)
from data_agent_baseline.agents.qd_sql_utils import (
    _fix_unescaped_apostrophes,
    _sanitize_sql,
    _enforce_grounding_filters,
    _apply_null_guard,
)

logger = logging.getLogger(__name__)


# Agent
# ---------------------------------------------------------------------------

class QuestionDrivenAgent(
    KGPlanningMixin,
    GroundingFormatMixin,
    SemanticGroundingMixin,
    KnowledgeMixin,
    SchemaSelectionMixin,
    FilterProcessingMixin,
    DiagnosticsMixin,
    UtilsMixin,
):
    def __init__(
        self,
        *,
        model: ModelAdapter,
        config: dict[str, Any] | None = None,
        log_callback: Any = None,
        **kwargs: Any,
    ):
        self.model = model
        self.config = config or {}
        self.log_callback = log_callback
        self.steps: list[dict[str, Any]] = []

    def run(self, task: PublicTask) -> AgentRunResult:
        """Execute the question-driven pipeline."""
        self._start_time = time.monotonic()
        self._decomposition_steps: list = []
        context_dir = task.context_dir
        question = task.question
        self._log_file: Path | None = None
        try:
            log_path = context_dir / "_agent.log"
            log_path.write_text(f"=== {task.task_id} ===\nQ: {question}\n\n")
            self._log_file = log_path
        except OSError:
            pass

        # Clean up stale DB
        stale_db = context_dir / CONSOLIDATED_DB_NAME
        if stale_db.exists():
            try:
                stale_db.unlink()
            except OSError:
                pass

        try:
            # Step 1: Scan context (deterministic, instant)
            ctx = scan_context(context_dir)
            self._log("scan", f"Scanned: {ctx.task_type}, "
                      f"{len(ctx.structured_sources)} structured, "
                      f"{len(ctx.doc_sources)} docs")

            # Step 2: Consolidate structured data → SQLite (deterministic)
            for stale in context_dir.glob("_consolidated*.db"):
                try:
                    stale.unlink()
                except OSError:
                    pass
            db_path = consolidate_to_sqlite(context_dir)
            if not db_path or not db_path.exists():
                # Fallback: try context_dir, then /tmp
                import tempfile as _tf
                db_path = context_dir / CONSOLIDATED_DB_NAME
                try:
                    sqlite3.connect(str(db_path)).close()
                except OSError:
                    db_path = Path(_tf.gettempdir()) / f"_consolidated_{context_dir.name}.db"
                    sqlite3.connect(str(db_path)).close()

            # Track which tables came from structured data (CSV/JSON)
            structured_tables: list[str] = []
            if db_path.exists():
                try:
                    _conn = sqlite3.connect(str(db_path))
                    structured_tables = [
                        r[0] for r in _conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall() if not r[0].startswith("_")
                    ]
                    _conn.close()
                except Exception:
                    pass

            # Store doc paths for downstream access (entity resolution from docs)
            self._doc_paths = [doc.path for doc in ctx.doc_sources] if ctx.doc_sources else []

            # Step 3: Doc extraction — evidence-based decision.
            # A doc should be extracted if:
            #   (a) doc name overlaps an existing table (additional records for it), OR
            #   (b) DB probe: a column contains opaque IDs that appear in the doc text
            #       (the doc is a lookup table for that FK column), OR
            #   (c) No structured tables exist (doc is the only data source)
            if ctx.doc_sources:
                docs_to_extract: list[Path] = []

                # (a) Name overlap with existing tables
                if structured_tables:
                    tables_lower = {t.lower() for t in structured_tables}
                    for doc in ctx.doc_sources:
                        doc_stem = doc.path.stem.lower()
                        if any(doc_stem in tbl or tbl in doc_stem for tbl in tables_lower):
                            docs_to_extract.append(doc.path)

                # (b) Evidence-based: probe DB columns for opaque IDs, check if they
                #     appear in doc text. No naming conventions assumed.
                if structured_tables and db_path.exists():
                    doc_texts: dict[Path, str] = {}
                    for doc in ctx.doc_sources:
                        if doc.path in docs_to_extract:
                            continue
                        try:
                            doc_texts[doc.path] = doc.path.read_text(
                                encoding="utf-8", errors="replace"
                            )
                        except OSError:
                            pass

                    if doc_texts:
                        try:
                            _conn = sqlite3.connect(str(db_path))
                            for tbl in structured_tables:
                                cols = _conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()
                                for col_info in cols:
                                    col_name = col_info[1]
                                    col_type = (col_info[2] or "").upper()
                                    # Skip numeric/primary columns
                                    if col_type in ("INTEGER", "REAL", "NUMERIC"):
                                        continue
                                    # Sample values from this column
                                    sample_rows = _conn.execute(
                                        f'SELECT DISTINCT "{col_name}" FROM "{tbl}" '
                                        f'WHERE "{col_name}" IS NOT NULL LIMIT 20'
                                    ).fetchall()
                                    if not sample_rows:
                                        continue
                                    sample_vals = [str(r[0]) for r in sample_rows if r[0]]
                                    if not sample_vals:
                                        continue
                                    # Signal: values look like opaque IDs (not readable text)
                                    # — short-ish, no spaces, mixed alphanumeric
                                    is_opaque = all(
                                        len(v) >= 5 and len(v) <= 50
                                        and " " not in v and "@" not in v
                                        and not v.replace(".", "").replace("-", "").isdigit()
                                        and any(c.isalpha() for c in v)
                                        and any(c.isdigit() for c in v)
                                        for v in sample_vals
                                    )
                                    if not is_opaque:
                                        continue
                                    # Probe: do these IDs appear in any doc?
                                    for doc_path, doc_text in doc_texts.items():
                                        matches = sum(1 for v in sample_vals if v in doc_text)
                                        if matches >= 2 or (matches >= 1 and len(sample_vals) <= 2):
                                            if doc_path not in docs_to_extract:
                                                docs_to_extract.append(doc_path)
                                                self._log(
                                                    "doc_fk_probe",
                                                    f"{tbl}.{col_name} IDs found in {doc_path.name} "
                                                    f"({matches}/{len(sample_vals)} matched)",
                                                )
                                            break
                            _conn.close()
                        except Exception:
                            pass

                # (c) Fallback: if no structured tables exist, extract all docs
                if not structured_tables and ctx.doc_sources:
                    docs_to_extract = [doc.path for doc in ctx.doc_sources]

                # (c2) FK-referenced docs: if structured tables have FK columns
                # pointing to tables not in DB, check if a doc matches that table name
                if structured_tables and db_path.exists():
                    try:
                        _fk_conn = sqlite3.connect(str(db_path))
                        existing_tables = {
                            r[0].lower()
                            for r in _fk_conn.execute(
                                "SELECT name FROM sqlite_master WHERE type='table'"
                            ).fetchall()
                        }
                        for tbl in structured_tables:
                            cols = _fk_conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()
                            for col_info in cols:
                                col_name = col_info[1]
                                # Detect FK naming: ends with Id/_id and references missing table
                                ref_table = ""
                                cn_lower = col_name.lower()
                                if cn_lower.endswith("id") and cn_lower != "id":
                                    ref_table = cn_lower[:-2].rstrip("_")
                                elif cn_lower.endswith("_id"):
                                    ref_table = cn_lower[:-3]
                                if ref_table and ref_table not in existing_tables:
                                    for doc in ctx.doc_sources:
                                        if doc.path in docs_to_extract:
                                            continue
                                        doc_stem = doc.path.stem.lower()
                                        if ref_table in doc_stem or doc_stem in ref_table:
                                            docs_to_extract.append(doc.path)
                                            self._log(
                                                "doc_fk_ref",
                                                f"'{doc.path.name}' matches FK column {tbl}.{col_name} (missing table '{ref_table}')",
                                            )
                        _fk_conn.close()
                    except Exception:
                        pass

                # (d) Question-referenced: if the question mentions a term that
                # matches a doc file stem, that doc likely holds the primary data
                if structured_tables:
                    q_lower = question.lower()
                    for doc in ctx.doc_sources:
                        if doc.path in docs_to_extract:
                            continue
                        stem = doc.path.stem.lower()
                        if len(stem) < 4:
                            continue
                        # Check if the doc stem (or its plural/singular) appears in the question
                        variants = [stem, stem + "s", stem + "es"]
                        if stem.endswith("s"):
                            variants.append(stem[:-1])
                        if any(re.search(r'\b' + re.escape(v) + r'\b', q_lower) for v in variants):
                            docs_to_extract.append(doc.path)
                            self._log(
                                "doc_question_match",
                                f"'{doc.path.name}' matches question term",
                            )

                # (e) Knowledge-referenced: if knowledge mentions columns (format, status)
                # that appear in the question but NOT in any loaded table, extract docs
                # whose stem matches a table name from knowledge SQL examples
                if structured_tables and ctx.knowledge_text:
                    q_lower = question.lower()
                    # Get all column names from loaded tables
                    loaded_cols: set[str] = set()
                    if db_path.exists():
                        try:
                            _kconn = sqlite3.connect(str(db_path), timeout=5)
                            for tbl in structured_tables:
                                for col in _kconn.execute(f'PRAGMA table_info("{tbl}")').fetchall():
                                    loaded_cols.add(col[1].lower())
                            _kconn.close()
                        except Exception:
                            pass
                    # Find knowledge-defined field names that are in the question but not loaded
                    missing_fields: list[str] = []
                    for km in re.finditer(
                        r"[-*]\s+\*{0,2}(\w[\w\s\-]*?)\s*(?:\((\w+)\))?\s*\*{0,2}\s*:",
                        ctx.knowledge_text,
                    ):
                        kf = km.group(1).strip().lower().replace(" ", "_")
                        if kf and kf not in loaded_cols and kf in q_lower:
                            missing_fields.append(kf)
                    # Also check knowledge SQL for table names matching doc stems
                    if missing_fields:
                        knowledge_tables: set[str] = set()
                        for tm in re.finditer(
                            r'\bFROM\s+(\w+)|JOIN\s+(\w+)',
                            ctx.knowledge_text, re.IGNORECASE,
                        ):
                            t = (tm.group(1) or tm.group(2)).lower()
                            if t not in {s.lower() for s in structured_tables}:
                                knowledge_tables.add(t)
                        for doc in ctx.doc_sources:
                            if doc.path in docs_to_extract:
                                continue
                            doc_stem = doc.path.stem.lower()
                            if doc_stem in knowledge_tables or any(
                                doc_stem.startswith(t) or t.startswith(doc_stem)
                                for t in knowledge_tables
                            ):
                                docs_to_extract.append(doc.path)
                                self._log(
                                    "doc_knowledge_ref",
                                    f"'{doc.path.name}' matches knowledge table "
                                    f"(missing fields: {missing_fields})",
                                )

                if docs_to_extract:
                    from data_agent_baseline.pipeline.regex_extractor import (
                        regex_extract_docs,
                    )
                    regex_extract_docs(
                        doc_paths=docs_to_extract,
                        db_path=db_path,
                        model=self.model,
                        question=question,
                        knowledge_text=ctx.knowledge_text,
                        log_fn=self._log,
                        structured_tables=structured_tables,
                    )

            # Step 4: Build KG
            self._log("kg_step", "building property graph from SQLite...")
            t0 = time.time()
            kg = build_kg_from_sqlite(db_path)
            self._log("kg_step", f"property graph built: {time.time()-t0:.1f}s ({len(kg.tables)} tables)")

            t0 = time.time()
            kg = discover_joins_with_llm(kg, model=self.model, log_fn=self._log)
            self._log("kg_step", f"joins discovered: {time.time()-t0:.1f}s")

            kg = classify_columns_with_llm(kg, model=self.model, log_fn=self._log)

            # Sequential small LLM calls: classify → decode vocab → concepts
            t0 = time.time()
            kg.ontology = build_ontology(kg, model=self.model, db_path=db_path, log_fn=self._log)
            self._log("kg_step", f"ontology complete: {time.time()-t0:.1f}s")

            profile_schema(kg)
            kg_context = format_kg_for_llm(kg)
            g = kg.graph
            roles_str = ", ".join(f"{t.name}={t.role}" for t in kg.tables if t.role)
            self._log("kg_built", (
                f"KG: {len(kg.tables)} tables, {len(kg.inferred_fks)} inferred FKs\n"
                f"  Graph: {len(g.columns)} columns, {len(g.values)} value nodes, "
                f"{len(g.fk_edges)} FK edges, {len(g.semantic_edges)} semantic edges\n"
                f"  Value index: {len(g.value_index)} unique values indexed\n"
                f"  Roles: {roles_str}"
            ))
            if g.fk_edges:
                fk_summary = "; ".join(
                    f"{e.src}→{e.dst} ({e.overlap_ratio:.0%})"
                    for e in g.fk_edges[:5]
                )
                self._log("kg_fk_edges", fk_summary)
            if g.semantic_edges:
                sem_summary = "; ".join(
                    f"{e.src}~{e.dst} ({e.similarity_score:.2f})"
                    for e in g.semantic_edges[:5]
                )
                self._log("kg_semantic_edges", sem_summary)

            # Get sample data for each table (question-aware probing)
            sample_data = self._get_sample_data(db_path, kg, question)

            # Step 5: KG path planning — graph-based reasoning to reach the goal
            grounding_context = self._kg_path_plan_grounding(
                question, kg_context, sample_data, ctx.knowledge_text,
                db_path=db_path, kg=kg,
            )

            # Step 5b: Value Discovery — probe DB for actual filter values
            # Patch grounding in-place so corrections override original FILTER VALUES
            value_discovery = self._discover_filter_values(
                question, db_path, kg, grounding_context, ctx.knowledge_text,
            )
            if value_discovery:
                self._log("value_discovery", value_discovery)
                grounding_context = self._patch_grounding_with_discoveries(
                    grounding_context, value_discovery,
                )

            # Step 5c: Threshold inference — infer normal/abnormal ranges if needed
            threshold_context = self._infer_thresholds(
                question, db_path, kg, ctx.knowledge_text,
            )
            if threshold_context:
                self._log("threshold_inference", threshold_context)
                grounding_context += f"\n\n{threshold_context}"

            # ----------------------------------------------------------
            # Multi-step decomposition (if rule engine emitted steps)
            # ----------------------------------------------------------
            data_result = None
            if self._decomposition_steps:
                data_result = self._execute_decomposition(
                    question, db_path, grounding_context, kg_context, sample_data, ctx.knowledge_text,
                )
                if data_result and data_result.get("rows"):
                    self._log("decomposition_success", f"rows={len(data_result['rows'])}")

            # ----------------------------------------------------------
            # SQL Generation: LLM writes SQL, close-loop on failure
            # ----------------------------------------------------------
            max_sql_attempts = 4
            sql = ""
            failed_sqls: list[str] = []
            gaps = ""

            if data_result and data_result.get("rows"):
                pass  # decomposition already produced result, skip single-shot
            else:
                data_result = None

            for attempt in range(max_sql_attempts):
                if data_result and data_result.get("rows"):
                    break
                sql = self._call_sql(
                    question,
                    grounding_context=grounding_context,
                    gaps=gaps,
                )
                if not sql:
                    break

                sql = _sanitize_sql(sql, db_path)
                sql = _apply_null_guard(sql)
                sql = _enforce_grounding_filters(sql, grounding_context, db_path)

                # Break if model repeats a previous SQL (no variation = no point retrying)
                if sql in failed_sqls:
                    self._log("sql_repeat", "Model repeated same SQL, stopping retries")
                    break

                self._log("sql_generated" if attempt == 0 else f"sql_retry_{attempt}", sql)
                data_result = self._try_sql(db_path, sql)

                if data_result and data_result.get("rows"):
                    # Layer 5: Deterministic result validation
                    _l5_comp = getattr(self, '_last_comp_type', '')
                    _l5_nodes = getattr(self, '_last_output_nodes', [])
                    anomaly = self._validate_result_stats(
                        data_result, _l5_comp, _l5_nodes, kg, db_path,
                    )
                    if anomaly and attempt < max_sql_attempts - 1:
                        self._log("result_anomaly", anomaly)
                        gaps = f"RESULT ANOMALY: {anomaly}\nFailed SQL: {sql}"
                        failed_sqls.append(sql)
                        data_result = None
                        continue
                    break

                # Diagnose failure for next iteration
                failed_sqls.append(sql)
                last_error = next(
                    (s.get("detail", "") for s in reversed(self.steps) if s.get("action") == "sql_error"),
                    "",
                )
                if last_error:
                    gaps = f"- SQL ERROR: {last_error}\n- Failed SQL: {sql}"
                elif data_result is not None:
                    # Executed OK but 0 rows — try deterministic blocker removal
                    fixed_sql = self._try_remove_blocker_filter(db_path, sql)
                    if fixed_sql and fixed_sql != sql:
                        data_result = self._try_sql(db_path, fixed_sql)
                        if data_result and data_result.get("rows"):
                            sql = fixed_sql
                            self._log(f"sql_retry_{attempt}", f"(blocker removed) {fixed_sql}")
                            break
                    diagnosis = self._diagnose_empty_result(db_path, sql) if sql else ""
                    gaps = f"- ZERO ROWS returned.\n- Failed SQL: {sql}"
                    if diagnosis:
                        gaps += f"\n- DIAGNOSIS: {diagnosis}"
                else:
                    break

            # Last resort: if close-loop exhausted, try multi-hypothesis approach
            if not (data_result and data_result.get("rows")):
                self._log("hypothesis_trigger", "SQL close-loop exhausted — trying multi-hypothesis fallback")
                # Diagnose the first valid (non-erroring) SQL that returned 0 rows
                diag_sql = sql
                for fs in failed_sqls:
                    test_r = self._try_sql(db_path, fs)
                    if test_r is not None:
                        diag_sql = fs
                        break
                diagnosis = self._diagnose_empty_result(db_path, diag_sql) if diag_sql else ""
                hyp_result, hyp_sql = self._try_multi_hypothesis(
                    question=question,
                    db_path=db_path,
                    kg_context=kg_context,
                    sample_data=sample_data,
                    knowledge_text=ctx.knowledge_text,
                    failed_sqls=failed_sqls,
                    diagnosis=diagnosis,
                )
                if hyp_result and hyp_result.get("rows"):
                    data_result = hyp_result
                    self._log("hypothesis_success", f"Hypothesis produced {len(hyp_result['rows'])} rows")


            # Shape validation before formatting
            if data_result and data_result.get("rows"):
                data_result = self._validate_result_shape(
                    question, data_result, db_path, kg_context,
                    sample_data, ctx.knowledge_text,
                    grounding_context=grounding_context,
                    column_hints="",
                )

            # Format answer
            raw_row_count = len(data_result.get("rows", [])) if data_result else 0
            self._log("pre_answer", f"cols={data_result.get('columns') if data_result else None}, rows={raw_row_count}")
            if data_result and data_result.get("rows"):
                answer = self._call_answer_with_schema(
                    question, data_result, ctx.knowledge_text,
                    grounding_context=grounding_context,
                )
                if not answer or not answer.get("rows"):
                    answer = self._raw_result_to_answer(data_result)
            else:
                answer = {"columns": [], "rows": []}

            return self._build_result(answer, task)

        except Exception as e:
            logger.exception("Pipeline failed")
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=[],
                failure_reason=str(e),
            )


    # ------------------------------------------------------------------
    # LLM Call 1: SQL Generation
    # ------------------------------------------------------------------

    def _call_sql(
        self, question: str, kg_context: str = "",
        gaps: str = "", column_hints: str = "",
        grounding_context: str = "",
    ) -> str:
        prompt = _build_sql_prompt(
            question=question,
            kg_context=kg_context,
            column_hints=column_hints,
            gaps=gaps,
            grounding_context=grounding_context,
        )
        self._log("sql_prompt_size", f"chars={len(prompt)}, grounding={len(grounding_context)}, kg={len(kg_context)}, gaps={len(gaps)}")

        messages = [ModelMessage(role="user", content=prompt)]
        t0 = time.monotonic()
        raw = self._model_call_with_retry(messages, thinking=False)
        self._log("sql_llm_time", f"{time.monotonic() - t0:.1f}s, response_len={len(raw) if raw else 0}")
        if not raw:
            self._log("sql_call_empty", "LLM returned empty response")
            return ""
        parsed = self._parse_json(raw)

        if isinstance(parsed, dict):
            sql = parsed.get("sql") or parsed.get("query") or ""
            if not sql:
                for v in parsed.values():
                    if isinstance(v, str) and v.strip().upper().startswith("SELECT"):
                        sql = v
                        break
            if sql:
                return sql
        # Try extracting SQL directly from raw text
        # First try: extract from "sql": "..." pattern (handles properly escaped JSON)
        sql_val_match = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if sql_val_match:
            sql_candidate = sql_val_match.group(1).replace('\\"', '"').replace("\\n", "\n")
            if sql_candidate.strip().upper().startswith("SELECT") and len(sql_candidate.strip()) > 20:
                return sql_candidate.strip()
        # Second try: find SELECT...up to end or code fence (handles unescaped quotes)
        select_match = re.search(r'(SELECT\s.+?)(?:```|\Z)', raw, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql_candidate = select_match.group(1).strip()
            # Clean trailing JSON artifacts
            sql_candidate = re.sub(r'"\s*\}?\s*$', '', sql_candidate).strip()
            sql_candidate = sql_candidate.replace('\\"', '"').replace("\\n", "\n")
            if sql_candidate.upper().startswith("SELECT"):
                return sql_candidate
        self._log("sql_parse_failed", f"raw={raw}")
        return ""

    # Multi-step decomposition executor
    # ------------------------------------------------------------------

    def _execute_decomposition(
        self,
        question: str,
        db_path: Path,
        grounding_context: str,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
    ) -> dict[str, Any] | None:
        """Execute decomposed steps: get values separately, compute final answer."""
        steps = self._decomposition_steps
        if not steps or len(steps) < 3:
            return None

        # Steps: [get_value_a, get_value_b, compute]
        values: dict[str, float] = {}
        descriptions: dict[str, str] = {}

        for step in steps:
            if step.sql_template == "__compute__":
                if len(values) < 2:
                    return None
                # Use SQL for full-precision arithmetic instead of LLM
                compute_result = self._decomposition_compute_sql(
                    question, db_path, values, descriptions,
                )
                if compute_result is None:
                    # Fallback to LLM compute
                    compute_result = self._decomposition_compute(question, values, descriptions)
                if compute_result is not None:
                    self._log("decomposition_compute", f"result={compute_result}")
                    return {"columns": ["result"], "rows": [[compute_result]]}
                return None

            # Close-loop per step: retry up to 3 times on failure
            max_step_attempts = 3
            step_gaps = ""
            step_value = None

            for attempt in range(max_step_attempts):
                sql = self._decomposition_step_sql(
                    question, step.description, step.output_var,
                    grounding_context, values, gaps=step_gaps,
                )
                if not sql:
                    break

                sql = _sanitize_sql(sql, db_path)
                self._log("decomposition_step", f"{step.output_var}[{attempt}]: {sql}")
                result = self._try_sql(db_path, sql)

                if not result:
                    last_err = next(
                        (s.get("detail", "") for s in reversed(self.steps) if s.get("action") == "sql_error"), ""
                    )
                    step_gaps = f"SQL ERROR: {last_err}\nFailed SQL: {sql}"
                    continue

                if not result.get("rows") or result["rows"][0][0] is None:
                    hint = ""
                    if "JOIN" in sql.upper():
                        hint = " Try using WHERE \"raceId\" IN (SELECT \"_id\" FROM \"races\" WHERE ...) instead of JOIN."
                    step_gaps = f"Returned NULL or 0 rows.{hint}\nFailed SQL: {sql}"
                    continue

                try:
                    step_value = float(result["rows"][0][0])
                    break
                except (TypeError, ValueError):
                    step_gaps = f"Non-numeric result: {result['rows'][0][0]}\nFailed SQL: {sql}"
                    continue

            if step_value is None:
                self._log("decomposition_step_fail", f"{step.output_var}: exhausted retries")
                return None

            values[step.output_var] = step_value
            descriptions[step.output_var] = step.description

        return None

    def _decomposition_step_sql(
        self,
        question: str,
        step_description: str,
        output_var: str,
        grounding_context: str,
        prior_values: dict[str, float],
        gaps: str = "",
    ) -> str:
        """Generate a simple SQL for one decomposition step."""
        prior_info = ""
        if prior_values:
            prior_info = "\nALREADY RETRIEVED:\n" + "\n".join(
                f"  {k} = {v}" for k, v in prior_values.items()
            )

        error_section = ""
        if gaps:
            error_section = f"\nPREVIOUS ATTEMPT FAILED:\n{gaps}\nFix the issue."

        # Strip grounding to factual sections only —
        # remove USER WANTS, COMPUTATION TYPE, VALUE COMPARISON, RATIO PATTERN, etc.
        # that cause the LLM to solve the whole question instead of one step.
        keep_headers = ("SCHEMA", "JOIN PATHS", "FILTER VALUES", "ORDER BY")
        keep_sections: list[str] = []
        current_section: list[str] = []
        current_header = ""
        for line in grounding_context.split("\n"):
            if any(line.startswith(h) for h in keep_headers):
                if current_header and current_section:
                    keep_sections.extend(current_section)
                current_header = line.split(":")[0]
                current_section = [line]
            elif current_header:
                if line and not line.startswith(" ") and not line.startswith("\t") and not line.startswith('"') and not line.startswith("⚠"):
                    if current_section:
                        keep_sections.extend(current_section)
                    current_header = ""
                    current_section = []
                else:
                    current_section.append(line)
        if current_header and current_section:
            keep_sections.extend(current_section)
        step_grounding = "\n".join(keep_sections).strip()

        prompt = f"""Step: {step_description}

{step_grounding}
{prior_info}
{error_section}

Write ONE SQL that returns exactly one row with one numeric value.
- SELECT one column only. No subqueries in SELECT.
- Simple: SELECT col FROM table WHERE ... LIMIT 1
- Quote identifiers with double-quotes.
- For non-aggregate: WHERE col IS NOT NULL. For AVG/SUM/COUNT: do NOT add IS NOT NULL on aggregated columns.
- If a JOIN returns 0 rows, use WHERE "col" IN (SELECT ...) instead.

Return ONLY: {{"sql": "SELECT ..."}}"""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages, thinking=False)
        if not raw:
            return ""
        parsed = self._parse_json(raw)
        if isinstance(parsed, dict):
            sql = parsed.get("sql", "")
            if sql:
                return sql
        select_match = re.search(r'(SELECT\s.+?)(?:```|"|\Z)', raw, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip().rstrip('"').rstrip("'")
        return ""

    def _decomposition_compute_sql(
        self, question: str, db_path: Path, values: dict[str, float],
        descriptions: dict[str, str] | None = None,
    ) -> float | None:
        """Ask LLM for formula, execute in SQLite for full precision."""
        lines = []
        for k, v in values.items():
            label = (descriptions or {}).get(k, k)
            lines.append(f"  {k} ({label}) = {v}")
        values_text = "\n".join(lines)

        prompt = f"""QUESTION: {question}

RETRIEVED VALUES:
{values_text}

Write a SQL SELECT that computes the final answer using the literal numeric values above.
The answer should be positive when the question asks "how much more/faster/higher".

Return ONLY: {{"sql": "SELECT ..."}}"""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages, thinking=False)
        if not raw:
            return None
        parsed = self._parse_json(raw)
        sql = ""
        if isinstance(parsed, dict):
            sql = parsed.get("sql", "")
        if not sql:
            m = re.search(r'(SELECT\s.+?)(?:```|"|\Z)', raw, re.DOTALL | re.IGNORECASE)
            if m:
                sql = m.group(1).strip().rstrip('"').rstrip("'")
        if not sql:
            return None
        result = self._try_sql(db_path, sql)
        if result and result.get("rows") and result["rows"][0][0] is not None:
            try:
                return float(result["rows"][0][0])
            except (TypeError, ValueError):
                pass
        return None

    def _decomposition_compute(
        self, question: str, values: dict[str, float], descriptions: dict[str, str] | None = None,
    ) -> float | None:
        """Compute final answer from retrieved values using LLM."""
        # Label values with their step descriptions for semantic clarity
        lines = []
        for k, v in values.items():
            label = (descriptions or {}).get(k, k)
            lines.append(f"  {label} = {v}")
        values_text = "\n".join(lines)

        prompt = f"""QUESTION: {question}

RETRIEVED VALUES:
{values_text}

Compute the final numeric answer. Return FULL precision (no rounding).
The answer should be a positive number when the question asks "how much more/faster/higher".

Return ONLY: {{"result": <number>}}"""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages)
        if not raw:
            return None
        parsed = self._parse_json(raw)
        if isinstance(parsed, dict):
            result = parsed.get("result")
            if result is not None:
                try:
                    return float(result)
                except (TypeError, ValueError):
                    pass
        return None

    def _call_answer_with_schema(
        self,
        question: str,
        data_result: dict[str, Any],
        knowledge_text: str,
        grounding_context: str = "",
    ) -> dict[str, Any]:
        """Two-phase answer: LLM picks columns from schema, code applies to full data."""
        columns = data_result.get("columns", [])
        rows = data_result.get("rows", [])

        if not columns or not rows:
            return {}

        # Single column — no need to ask LLM
        if len(columns) == 1:
            return self._raw_result_to_answer(data_result)

        # Extract user intent from grounding context
        user_wants = ""
        if grounding_context:
            match = re.search(r"USER WANTS:\s*(.+)", grounding_context)
            if match:
                user_wants = match.group(1).strip()

        col_list = "\n".join(f"  {i}: {c}" for i, c in enumerate(columns))

        # Detect superlative pattern to deterministically drop criterion column
        sup_match = re.search(
            r'\b(?:which|what|who)\b.+?\b(?:has|have|with|had)\b.+?\b(?:the\s+)?(?:lowest|highest|most|least|best|worst|fastest|slowest|largest|smallest|longest|shortest)\b\s+(\w+)',
            question.lower(),
        )
        criterion_col = sup_match.group(1) if sup_match else ""

        # If we can deterministically identify the criterion column, just drop it
        if criterion_col and len(columns) == 2:
            criterion_idx = next(
                (i for i, c in enumerate(columns) if criterion_col in c.lower()),
                None,
            )
            if criterion_idx is not None:
                keep_idx = 1 - criterion_idx
                self._log("answer_schema", f"Kept columns [{keep_idx}] → ['{columns[keep_idx]}'] ({len(rows)} rows)")
                return {
                    "columns": [columns[keep_idx]],
                    "rows": [[str(row[keep_idx])] for row in rows],
                }

        # Detect grouped_list with aggregate column: when SQL returns [col, COUNT/SUM/...]
        # and the question asks to "tally/list/identify" rather than "how many of each",
        # keep only the descriptive column (drop the aggregate).
        if len(columns) == 2:
            agg_pattern = re.compile(
                r'^(COUNT|SUM|AVG|MIN|MAX)\s*\(', re.IGNORECASE,
            )
            agg_idx = next(
                (i for i, c in enumerate(columns) if agg_pattern.match(c)), None,
            )
            if agg_idx is not None:
                q_lower = question.lower()
                # Only drop if the question does NOT explicitly ask for counts/breakdown
                asks_for_count = bool(re.search(
                    r'\bhow many (?:of each|per|for each|times each)\b'
                    r'|\bcount (?:of|for) each\b'
                    r'|\bbreakdown\b|\bfrequency\b|\bdistribution\b',
                    q_lower,
                ))
                if not asks_for_count:
                    keep_idx = 1 - agg_idx
                    self._log("answer_schema", f"Kept columns [{keep_idx}] → ['{columns[keep_idx]}'] ({len(rows)} rows)")
                    return {
                        "columns": [columns[keep_idx]],
                        "rows": [[str(row[keep_idx])] for row in rows],
                    }

        prompt = f"""The user asked a question. The SQL returned these columns. Which columns should appear in the final answer?

QUESTION: {question}
USER INTENT: {user_wants or question}

SQL RESULT COLUMNS:
{col_list}

Return ONLY: {{"keep_columns": [0, 2]}}

RULES:
- Keep columns the user explicitly asked to SEE in the answer.
- "full name" = keep BOTH first_name AND last_name (or all name-related columns).
- If the question asks for multiple attributes ("list X and Y"), keep ALL of them.
- REMOVE criterion/sorting columns used only to find the answer. "Which X has the lowest Y?" → keep X, drop Y. "What is the Y of X?" → keep Y, drop X's ID.
- REMOVE internal IDs and filter-echo columns (constant values from WHERE clause).
- NEVER merge columns. Just pick indices to keep."""
        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages, thinking=False)
        parsed = self._parse_json(raw)

        if not isinstance(parsed, dict) or "keep_columns" not in parsed:
            self._log("answer_schema", "(failed to parse, using raw)")
            return self._raw_result_to_answer(data_result)

        keep_indices = parsed.get("keep_columns", [])

        # Validate indices
        if not keep_indices or not all(isinstance(i, int) and 0 <= i < len(columns) for i in keep_indices):
            self._log("answer_schema", f"Invalid indices {keep_indices}, using raw")
            return self._raw_result_to_answer(data_result)

        # Drop columns whose values look like raw FK IDs (alphanumeric hashes not asked for)
        if len(keep_indices) > 1 and rows:
            cleaned = []
            for i in keep_indices:
                col_name = columns[i].lower()
                # Check if column name suggests FK/ID and values look like hashes
                if ("link_to" in col_name or col_name.endswith("_id")) and col_name not in question.lower():
                    sample_vals = [str(row[i]) for row in rows[:5] if i < len(row)]
                    if sample_vals and all(re.match(r'^rec[A-Za-z0-9]{10,}$', v) for v in sample_vals):
                        continue
                cleaned.append(i)
            if cleaned:
                keep_indices = cleaned

        output_names = [columns[i] for i in keep_indices]

        # Apply column selection to ALL rows (no LLM, no truncation)
        filtered_rows = [
            [str(row[i]) for i in keep_indices]
            for row in rows if len(row) > max(keep_indices)
        ]
        if not filtered_rows:
            return self._raw_result_to_answer(data_result)

        self._log("answer_schema", f"Kept columns {keep_indices} → {output_names} ({len(filtered_rows)} rows)")
        return {"columns": output_names, "rows": filtered_rows}

    # Component: Multi-Hypothesis SQL
    # ------------------------------------------------------------------

    def _try_multi_hypothesis(
        self,
        question: str,
        db_path: Path,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
        failed_sqls: list[str] | None = None,
        diagnosis: str = "",
    ) -> tuple[dict[str, Any] | None, str]:
        """Generate multiple SQL interpretations and execute first that returns data. Kept lean to avoid timeout."""
        if not db_path or not db_path.exists():
            return None, ""

        failed_section = ""
        if failed_sqls:
            failed_section = "\nFAILED SQL (do NOT repeat):\n" + failed_sqls[-1][:300]

        diag_section = ""
        if diagnosis:
            diag_section = f"\nDIAGNOSIS:\n{diagnosis}"

        # Detect if all failed SQLs use JOINs — suggest single-table approach
        join_hint = ""
        if failed_sqls and all("JOIN" in s.upper() for s in failed_sqls if s):
            join_hint = (
                "\n⚠️ ALL previous attempts used JOIN and returned 0 rows. "
                "The JOIN key may not match between tables. "
                "At least one hypothesis MUST query the main/larger table WITHOUT any JOIN — "
                "use only columns available directly in that table."
            )

        prompt = f"""Previous SQL returned 0 rows.

QUESTION: {question}

SCHEMA:
{kg_context[:3000]}

SAMPLES:
{sample_data[:500]}

{f"DOMAIN: {knowledge_text[:500]}" if knowledge_text else ""}
{failed_section}
{diag_section}
{join_hint}

Generate 3 DIFFERENT SQL queries. Each must try a DIFFERENT column, join, filter value, or format.

Return ONLY: {{"hypotheses": [{{"sql": "SELECT ..."}}, ...]}}

RULES:
- Each must be materially different
- Use LIKE for text matching when unsure of format
- Use actual DB values from DIAGNOSIS/SAMPLES if available
- NEVER use AS to rename columns
- ALWAYS double-quote column names that contain spaces (e.g. "School Name", "District Type")
- ONLY use columns that appear in the SCHEMA above. NEVER invent or guess column names."""

        messages = [ModelMessage(role="user", content=prompt)]
        raw = self._model_call_with_retry(messages, thinking=False)
        parsed = self._parse_json(raw)

        if not isinstance(parsed, dict):
            return None, ""

        hypotheses = parsed.get("hypotheses", [])
        if not hypotheses:
            return None, ""

        valid_hyps = [h for h in hypotheses[:3] if h.get("sql", "").strip()]
        if not valid_hyps:
            return None, ""

        # Build set of valid column names from all tables in the DB
        valid_columns: set[str] = set()
        try:
            conn = sqlite3.connect(str(db_path))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            for t in tables:
                for r in conn.execute(f'PRAGMA table_info("{t}")').fetchall():
                    valid_columns.add(r[1])
            conn.close()
        except Exception:
            pass

        # Execute in order, return first that produces rows
        for i, hyp in enumerate(valid_hyps):
            sql = _fix_unescaped_apostrophes(hyp["sql"])
            # Reject SQL that references non-existent columns (SQLite treats them as string literals)
            if valid_columns:
                quoted_refs = re.findall(r'"([^"]+)"', sql)
                invalid = [r for r in quoted_refs if r not in valid_columns and r not in tables]
                if invalid:
                    self._log("hypothesis_try", f"Option {i}: REJECTED (invalid columns: {invalid})")
                    continue
            self._log("hypothesis_try", f"Option {i}: {sql}")
            result = self._try_sql(db_path, sql)
            if result and result.get("rows"):
                all_null = all(
                    all(v is None or str(v).strip().lower() in ("none", "null", "") for v in row)
                    for row in result["rows"]
                )
                if not all_null:
                    self._log("hypothesis_success",
                              f"Option {i}: cols={result['columns']}, rows={len(result['rows'])}")
                    return result, hyp["sql"]

        return None, ""

    # Post-execution result shape validation
    # ------------------------------------------------------------------

    def _validate_result_shape(
        self,
        question: str,
        data_result: dict[str, Any],
        db_path: Path,
        kg_context: str,
        sample_data: str,
        knowledge_text: str,
        grounding_context: str,
        column_hints: str,
    ) -> dict[str, Any]:
        """Validate result shape matches question expectations and fix if needed."""
        rows = data_result.get("rows", [])
        cols = data_result.get("columns", [])
        if not rows:
            return data_result

        q_lower = question.lower()


        # Fix 4: Detect error strings or None/NULL values in result and retry
        first_row_str = " ".join(str(v) for v in rows[0]) if rows else ""
        has_none_values = any(str(v).lower() in ("none", "null", "") for v in rows[0]) if rows else False
        # Only flag None as suspicious if question asks for names/descriptions (not counts/aggregations)
        name_indicators = ["name", "who", "full name", "surname", "title", "display"]
        none_is_suspicious = has_none_values and any(w in q_lower for w in name_indicators)

        if "error" in first_row_str.lower() or first_row_str.strip() in ("0", "0.0", "0.00") or none_is_suspicious:
            # Check if result is suspicious (error text or zero when expecting real data)
            has_error = "error" in first_row_str.lower()
            if has_error or none_is_suspicious:
                issue_desc = "error/null values" if none_is_suspicious else "error string"
                self._log("shape_fix_error", f"Result contains {issue_desc}: {first_row_str}")
                if none_is_suspicious:
                    fix_prompt = f"""The SQL returned NULL/None values for columns that should have real data (names, descriptions, etc.).

QUESTION: {question}
CURRENT RESULT: columns={cols}, values={rows[0]}

The NULL values likely mean: a column name is WRONG (e.g., 'name' doesn't exist but 'first_name'+'last_name' do), or the JOIN failed.
Check the DATABASE SCHEMA carefully for the ACTUAL column names and fix the query.

DATABASE SCHEMA:
{kg_context[:2000]}

{grounding_context[:1000]}

Write a corrected SQL using the EXACT column names from the schema.
Return ONLY: {{"sql": "SELECT ..."}}"""
                else:
                    fix_prompt = f"""The SQL returned an error value instead of real data.

QUESTION: {question}
CURRENT RESULT: columns={cols}, values={rows[0]}

This is wrong. The result should be a meaningful number or value, not an error.
Possible issues: division by zero, NULL in computation, wrong column type.

DATABASE SCHEMA:
{kg_context[:2000]}

{grounding_context[:1000]}

Write a SIMPLER SQL that avoids the computation error. Use NULLIF for division, COALESCE for NULLs.
Return ONLY: {{"sql": "SELECT ..."}}"""
                messages = [ModelMessage(role="user", content=fix_prompt)]
                raw = self._model_call_with_retry(messages, thinking=False)
                parsed = self._parse_json(raw)
                if isinstance(parsed, dict) and parsed.get("sql"):
                    self._log("shape_fix_sql", parsed["sql"])
                    fix_result = self._try_sql(db_path, parsed["sql"])
                    if fix_result and fix_result.get("rows"):
                        fix_str = " ".join(str(v) for v in fix_result["rows"][0])
                        fix_nones = sum(1 for v in fix_result["rows"][0] if str(v).lower() in ("none", "null", ""))
                        orig_nones = sum(1 for v in rows[0] if str(v).lower() in ("none", "null", ""))
                        if "error" not in fix_str.lower() and fix_nones < orig_nones:
                            self._log("shape_fixed_error", f"Fixed: {fix_result['rows'][0]}")
                            return fix_result

        # Fix 2: Detect raw FK IDs in output and re-query with JOIN for human-readable values
        # Skip when the output column name appears in the question (user asked for it)
        _q_lower_shape = question.lower()
        _output_col_requested = any(
            col.lower().replace("_", " ") in _q_lower_shape
            or col.lower().replace("_", "") in _q_lower_shape.replace(" ", "")
            or col.lower() in _q_lower_shape
            for col in cols
        )
        if rows and cols and not _output_col_requested:
            has_raw_id = False
            raw_id_cols = []
            for i, col in enumerate(cols):
                sample_vals = [str(rows[r][i]) for r in range(min(len(rows), 3))]
                col_lower = col.lower()
                is_id_col = col_lower == "id" or col_lower.endswith("_id")
                all_int = all(v.lstrip("-").isdigit() for v in sample_vals if v and v.lstrip("-").isdigit())
                all_opaque = all(
                    len(v) >= 5 and not v.replace(".", "").replace("-", "").isdigit()
                    and " " not in v
                    for v in sample_vals if v
                )
                if sample_vals and ((is_id_col and all_int and len(cols) == 1) or (all_opaque and not all_int)):
                    has_raw_id = True
                    raw_id_cols.append((i, col, sample_vals[0]))

            # Skip if there's already a human-readable column alongside the ID
            if has_raw_id and len(raw_id_cols) > 0 and len(cols) > len(raw_id_cols):
                non_id_cols = [c for i, c in enumerate(cols) if i not in {idx for idx, _, _ in raw_id_cols}]
                has_readable = any(
                    any(w in c.lower() for w in ("name", "title", "label", "description", "forename", "surname"))
                    for c in non_id_cols
                )
                if has_readable:
                    # Just drop the ID columns instead of re-querying
                    keep_indices = [i for i in range(len(cols)) if i not in {idx for idx, _, _ in raw_id_cols}]
                    data_result = {
                        "columns": [cols[i] for i in keep_indices],
                        "rows": [[row[i] for i in keep_indices] for row in rows],
                    }
                    self._log("shape_fix_fk", f"Dropped raw ID columns: {[c for _, c, _ in raw_id_cols]}")
                    cols = data_result["columns"]
                    rows = data_result["rows"]
                    has_raw_id = False
                    raw_id_cols = []

            if has_raw_id and len(raw_id_cols) > 0:
                id_desc = ", ".join(f"'{c}' has values like '{v}'" for _, c, v in raw_id_cols)
                self._log("shape_fix_fk", f"Raw IDs detected: {id_desc}")
                fix_prompt = f"""The SQL result contains raw foreign key IDs instead of human-readable names.

QUESTION: {question}
CURRENT RESULT: columns={cols}, sample row={rows[0]}
RAW ID COLUMNS: {id_desc}

The user expects human-readable names/descriptions, not internal IDs. Add a JOIN to resolve these IDs to their display values.

DATABASE SCHEMA:
{kg_context[:2000]}

Return ONLY: {{"sql": "SELECT ..."}}"""
                messages = [ModelMessage(role="user", content=fix_prompt)]
                raw = self._model_call_with_retry(messages, thinking=False)
                parsed = self._parse_json(raw)
                if isinstance(parsed, dict) and parsed.get("sql"):
                    self._log("shape_fix_sql", parsed["sql"])
                    fix_result = self._try_sql(db_path, parsed["sql"])
                    if fix_result and fix_result.get("rows"):
                        new_vals = [str(v) for v in fix_result["rows"][0]]
                        # Verify at least one raw ID was resolved
                        old_vals = [str(rows[0][idx]) for idx, _, _ in raw_id_cols]
                        if any(nv != ov for nv, ov in zip(new_vals, old_vals)):
                            self._log("shape_fixed_fk", f"Resolved IDs: {fix_result['rows'][0]}")
                            return fix_result

        # Check: "X and Y" pattern expects 2+ columns but we got 1
        and_pattern = re.search(
            r'(?:what is|identify|find)\s+(?:the\s+)?(\w+).+?\band\b\s+(?:the\s+)?(\w+)',
            q_lower,
        )
        if and_pattern and len(cols) == 1 and len(rows) == 1:
            self._log("shape_fix", f"Question asks for two values ('{and_pattern.group(1)}' and '{and_pattern.group(2)}') but got 1 column — re-querying")
            fix_prompt = f"""The SQL returned a SINGLE combined value but the question asks for TWO SEPARATE values.

QUESTION: {question}
CURRENT RESULT: {cols[0]} = {rows[0][0]}

The question asks for '{and_pattern.group(1)}' AND '{and_pattern.group(2)}' as SEPARATE values.

DATABASE SCHEMA:
{kg_context[:2000]}

Write a corrected SQL that returns TWO columns (one for each value).
Return ONLY: {{"sql": "SELECT ..."}}"""
            messages = [ModelMessage(role="user", content=fix_prompt)]
            raw = self._model_call_with_retry(messages, thinking=False)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("sql"):
                self._log("shape_fix_sql", parsed["sql"])
                fix_result = self._try_sql(db_path, parsed["sql"])
                if fix_result and fix_result.get("rows") and len(fix_result.get("columns", [])) >= 2:
                    self._log("shape_fixed", f"Now has {len(fix_result['columns'])} columns")
                    return fix_result

        # Check: "how many" question expects a single count but got multiple rows
        count_patterns = [r"how many (?:of them|of these|of those)?\s*(?:are|is|were|was|have|had)\b",
                          r"how many .+? (?:are|is|were|was)\b"]
        expects_count = any(re.search(p, q_lower) for p in count_patterns)
        list_indicators = ["list", "what are", "identify", "name the", "which"]
        has_list = any(p in q_lower for p in list_indicators)
        if expects_count and not has_list and len(rows) > 1:
            self._log("shape_fix_count", f"'how many' question returned {len(rows)} rows — re-querying as COUNT")
            fix_prompt = f"""The SQL returned {len(rows)} rows but the question asks "how many" — it expects a SINGLE COUNT number.

QUESTION: {question}
CURRENT RESULT: {len(rows)} rows, columns={cols}
FIRST ROWS: {rows[:3]}

Rewrite the query to return COUNT(*) — the number of items matching the criteria, not the items themselves.

DATABASE SCHEMA:
{kg_context[:2000]}

{grounding_context[:1000]}

Return ONLY: {{"sql": "SELECT COUNT(...) ..."}}"""
            messages = [ModelMessage(role="user", content=fix_prompt)]
            raw = self._model_call_with_retry(messages, thinking=False)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("sql"):
                fix_result = self._try_sql(db_path, parsed["sql"])
                if fix_result and fix_result.get("rows") and len(fix_result["rows"]) == 1:
                    self._log("shape_fixed_count", f"COUNT result: {fix_result['rows'][0]}")
                    return fix_result

        # Check: singular question ("what was THE score") but got many rows
        singular_patterns = [
            r"what (?:is|was|were) the .+? (?:for|of|in) the ",
            r"what is the .+? of the ",
            r"identify the .+? (?:for|of) the ",
        ]
        expects_singular = any(re.search(p, q_lower) for p in singular_patterns)
        plural_indicators = ["list", "all", "each", "every", "which", "how many",
                             "lowest", "highest", "most", "least", "best", "worst"]
        has_plural = any(p in q_lower for p in plural_indicators)

        if expects_singular and not has_plural and len(rows) > 5:
            self._log("shape_fix_singular", f"Singular question but got {len(rows)} rows — attempting fix")
            fix_prompt = f"""The SQL returned {len(rows)} rows but the question expects a SINGLE result (it uses "the" indicating one specific item).

QUESTION: {question}
CURRENT SQL RETURNED: {len(rows)} rows with columns {cols}
FIRST FEW ROWS: {rows[:3]}

The question likely needs additional filters from its context that were missed. Re-read the question and add the missing WHERE conditions to narrow to exactly 1 row.

DATABASE SCHEMA:
{kg_context[:1500]}

Return ONLY: {{"sql": "SELECT ..."}}"""
            messages = [ModelMessage(role="user", content=fix_prompt)]
            raw = self._model_call_with_retry(messages, thinking=False)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("sql"):
                self._log("shape_fix_sql", parsed["sql"])
                fix_result = self._try_sql(db_path, parsed["sql"])
                if fix_result and fix_result.get("rows") and 0 < len(fix_result["rows"]) < len(rows):
                    self._log("shape_fixed_singular", f"Narrowed from {len(rows)} to {len(fix_result['rows'])} rows")
                    return fix_result

        return data_result

