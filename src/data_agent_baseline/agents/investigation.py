from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.prompt import (
    build_investigation_evaluator_prompt,
    build_investigation_planner_prompt,
    build_investigation_synthesizer_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.anomaly import detect_anomalies, format_anomaly_flags
from data_agent_baseline.tools.filesystem import list_context_tree
from data_agent_baseline.tools.graph_extract import (
    _entity_schemas_from_grounding,
    extract_multiple_docs_to_sqlite,
)
from data_agent_baseline.tools.grounding import (
    get_entity_schemas_for_task,
    get_full_knowledge_text,
)
from data_agent_baseline.tools.knowledge_graph import (
    CONSOLIDATED_DB_NAME,
    build_knowledge_graph,
    consolidate_to_sqlite,
    get_consolidated_schema,
    render_knowledge_graph,
)
from data_agent_baseline.tools.registry import ToolRegistry
from data_agent_baseline.tools.schema_graph import slice_schema_for_task


@dataclass(frozen=True, slots=True)
class InvestigationAgentConfig:
    max_steps: int = 16
    max_iterations: int = 5


MAX_EVIDENCE_ENTRY_CHARS = 8000
LARGE_DOC_THRESHOLD_BYTES = 20_000


def _truncate_content(content: Any, max_chars: int = MAX_EVIDENCE_ENTRY_CHARS) -> Any:
    if isinstance(content, dict):
        rows = content.get("rows")
        if isinstance(rows, list) and len(rows) > 30:
            content = {**content, "rows": rows[:30], "_truncated_rows": len(rows)}
        preview = content.get("preview")
        if isinstance(preview, str) and len(preview) > max_chars:
            content = {**content, "preview": preview[:max_chars] + "...(truncated)"}
    if isinstance(content, str) and len(content) > max_chars:
        return content[:max_chars] + "...(truncated)"
    return content


@dataclass(slots=True)
class _EvidenceStore:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, step_id: str, description: str, result: dict[str, Any]) -> None:
        truncated = {**result}
        if "content" in truncated:
            truncated["content"] = _truncate_content(truncated["content"])
        self.entries[step_id] = {"description": description, **truncated}

    def render(self) -> str:
        if not self.entries:
            return "(none yet)"
        return json.dumps(self.entries, ensure_ascii=False, indent=2)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence is not None:
        return fence.group(1).strip()
    generic = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic is not None:
        return generic.group(1).strip()
    return text


def _repair_json_string_concat(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        i = 0
        while i < len(text):
            q = text.find('"', i)
            if q < 0:
                break
            j = q + 1
            while j < len(text) and text[j] in " \t\n\r":
                j += 1
            if j < len(text) and text[j] == "+":
                k = j + 1
                while k < len(text) and text[k] in " \t\n\r":
                    k += 1
                if k < len(text) and text[k] == '"':
                    text = text[:q] + text[k + 1 :]
                    i = q
                    continue
            i = q + 1
    return text


def _parse_json(text: str) -> Any:
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return json.loads(_repair_json_string_concat(cleaned))


def _context_listing_text(task: PublicTask) -> str:
    tree = list_context_tree(task, max_depth=3)
    lines = []
    for entry in tree.get("entries", []):
        kind = entry.get("kind", "file")
        path = entry.get("path", "")
        size = entry.get("size")
        suffix = f" ({size} bytes)" if size is not None else ""
        lines.append(f"  [{kind}] {path}{suffix}")
    return "\n".join(lines) if lines else "(empty)"


def _normalize_gap_key(gap: str) -> str:
    return re.sub(r"\s+", " ", gap.strip().lower())


def _scan_json_keys(text: str, start: int, end: int, max_keys: int = 20) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    i = start
    while i < end and len(keys) < max_keys:
        q1 = text.find('"', i)
        if q1 < 0 or q1 >= end:
            break
        q2 = text.find('"', q1 + 1)
        if q2 < 0 or q2 >= end:
            break
        j = q2 + 1
        while j < end and text[j] in " \t\n\r":
            j += 1
        if j < end and text[j] == ":":
            key = text[q1 + 1 : q2]
            if 0 < len(key) <= 60 and key not in seen:
                keys.append(key)
                seen.add(key)
        i = q2 + 1
    return keys


def _scan_json_keys_typed(
    text: str, start: int, end: int, max_keys: int = 20
) -> list[tuple[str, str, str]]:
    """Like _scan_json_keys but also infers type and sample value."""
    keys: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    i = start
    while i < end and len(keys) < max_keys:
        q1 = text.find('"', i)
        if q1 < 0 or q1 >= end:
            break
        q2 = text.find('"', q1 + 1)
        if q2 < 0 or q2 >= end:
            break
        j = q2 + 1
        while j < end and text[j] in " \t\n\r":
            j += 1
        if j < end and text[j] == ":":
            key = text[q1 + 1 : q2]
            if 0 < len(key) <= 60 and key not in seen:
                v = j + 1
                while v < end and text[v] in " \t\n\r":
                    v += 1
                typ = "(text)"
                sample = ""
                if v < end:
                    ch = text[v]
                    if ch == '"':
                        vq = text.find('"', v + 1)
                        if vq > 0:
                            sample = text[v + 1 : vq]
                            typ = _infer_type(sample)
                    elif ch in "0123456789-":
                        num_end = v
                        has_dot = False
                        while num_end < end and text[num_end] in "0123456789.-+eE":
                            if text[num_end] == ".":
                                has_dot = True
                            num_end += 1
                        typ = "(real)" if has_dot else "(integer)"
                    elif text[v : v + 4] in ("true", "fals"):
                        typ = "(boolean)"
                    elif text[v : v + 4] == "null":
                        typ = "(null)"
                    elif ch == "[":
                        typ = "(array)"
                    elif ch == "{":
                        typ = "(object)"
                keys.append((key, typ, sample))
                seen.add(key)
        i = q2 + 1
    return keys


def _extract_first_object_keys(
    head: str,
) -> tuple[list[str], list[str], list[tuple[str, str, str]]] | None:
    """Returns (top_keys, record_key_names, record_keys_typed_with_sample)."""
    bracket_pos = head.find("[")
    first_brace = head.find("{")
    if first_brace < 0:
        return None

    if bracket_pos < 0 or first_brace < bracket_pos:
        scan_end = bracket_pos if bracket_pos > first_brace else first_brace + 200
        top_keys = _scan_json_keys(head, first_brace, scan_end, max_keys=10)

        record_keys_typed: list[tuple[str, str, str]] = []
        for container_key in ("records", "data"):
            pos = head.find(f'"{container_key}"', first_brace)
            if pos < 0:
                continue
            arr_start = head.find("[", pos)
            if arr_start < 0:
                continue
            rec_brace = head.find("{", arr_start)
            if rec_brace < 0:
                continue
            record_keys_typed = _scan_json_keys_typed(head, rec_brace, rec_brace + 2000)
            break

        record_names = [k for k, _, _ in record_keys_typed]
        return (top_keys, record_names, record_keys_typed) if top_keys else None

    rec_brace = head.find("{", bracket_pos)
    if rec_brace < 0:
        return None
    typed = _scan_json_keys_typed(head, rec_brace, rec_brace + 2000)
    names = [k for k, _, _ in typed]
    return ([], names, typed) if typed else None


def _infer_type(val: str) -> str:
    if not val:
        return "(text)"
    try:
        int(val)
        return "(integer)"
    except ValueError:
        pass
    try:
        float(val)
        return "(real)"
    except ValueError:
        pass
    if len(val) == 10 and val[4:5] == "-" and val[7:8] == "-":
        return "(date)"
    return "(text)"


def _infer_type_from_json(val: Any) -> str:
    if isinstance(val, bool):
        return "(boolean)"
    if isinstance(val, int):
        return "(integer)"
    if isinstance(val, float):
        return "(real)"
    if isinstance(val, str):
        if len(val) == 10 and val[4:5] == "-" and val[7:8] == "-":
            return "(date)"
        return "(text)"
    if isinstance(val, list):
        return "(array)"
    if isinstance(val, dict):
        return "(object)"
    if val is None:
        return "(null)"
    return "(text)"


def _json_field_hint(key: str, val: Any) -> str:
    typ = _infer_type_from_json(val)
    hint = f"{key} {typ}"
    if isinstance(val, str) and typ == "(text)" and 0 < len(val) <= 40:
        hint += f' e.g. "{val}"'
    return hint


LABEL_CLASSIFICATION_PROMPT = """
You are a data domain expert. Classify each status label into exactly one category:
"abnormal" (clearly problematic/pathological), "borderline" (uncertain/edge case),
or "normal" (healthy/expected/unremarkable).

QUESTION: {question}

DOMAIN CONTEXT:
{knowledge_text}

STATUS LABELS TO CLASSIFY:
{status_labels}

Return ONLY a JSON object mapping each label to its category (no markdown fences):
{{
  "<label_1>": "abnormal",
  "<label_2>": "normal",
  "<label_3>": "borderline",
  ...
}}

RULES:
- "abnormal" = the label clearly indicates a confirmed problem (with severity words
  like "significantly", "markedly", "severely", or words like "impaired", "compromised",
  "damage", "dysfunction")
- "borderline" = the label indicates mild or unqualified deviation (e.g. just "elevated"
  or "high" without severity modifier)
- "normal" = the label indicates a healthy or expected state
- Labels WITH a severity modifier (e.g. "significantly elevated") → abnormal.
  Labels WITHOUT a severity modifier (e.g. just "elevated") → borderline.
""".strip()


def _collect_status_columns(db_path: Path) -> dict[str, dict[str, list[str]]]:
    """Return {table: {status_col: [distinct values]}} from the consolidated DB."""
    import sqlite3

    result: dict[str, dict[str, list[str]]] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=DELETE")
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]

        for table in tables:
            cursor = conn.execute(f"PRAGMA table_info('{table}')")
            columns = [r[1] for r in cursor.fetchall()]
            status_cols = [c for c in columns if c.endswith("_status")]
            for col in status_cols:
                cursor = conn.execute(
                    f"SELECT DISTINCT \"{col}\" FROM \"{table}\" WHERE \"{col}\" IS NOT NULL LIMIT 30"
                )
                values = [r[0] for r in cursor.fetchall()]
                if values:
                    result.setdefault(table, {})[col] = values
        conn.close()
    except Exception:
        pass
    return result


def _classify_and_materialize(
    *,
    model: ModelAdapter,
    db_path: Path,
    question: str,
    knowledge_text: str,
) -> str:
    """Classify status labels via LLM and add _abnormal boolean columns to the DB.

    Returns a short domain guide string for the planner prompt.
    """
    import sqlite3

    status_map = _collect_status_columns(db_path)
    if not status_map:
        return ""

    # Flatten all labels for classification
    all_labels: set[str] = set()
    for cols in status_map.values():
        for values in cols.values():
            all_labels.update(values)

    if not all_labels:
        return ""

    # Rule-based classification: deterministic, no LLM dependency
    _ABNORMAL_MARKERS = (
        "significantly", "markedly", "severely", "impaired", "compromised",
        "dysfunction", "damage", "critical", "acute", "deficient",
        "active disease", "reduced function",
    )
    _NORMAL_MARKERS = (
        "normal", "healthy", "unremarkable", "adequate", "stable",
        "within range", "negative", "clear", "quiescent",
    )

    abnormal_labels: set[str] = set()
    normal_labels: set[str] = set()
    borderline_labels: set[str] = set()

    for label in all_labels:
        low = label.lower()
        if any(m in low for m in _ABNORMAL_MARKERS):
            abnormal_labels.add(low)
        elif any(m in low for m in _NORMAL_MARKERS):
            normal_labels.add(low)
        else:
            borderline_labels.add(low)

    # Materialize boolean _abnormal columns in DB
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=DELETE")

        for table, cols in status_map.items():
            for status_col, _ in cols.items():
                # Derive the abnormal column name: CRE_status → CRE_abnormal
                base = status_col.removesuffix("_status")
                abnormal_col = f"{base}_abnormal"

                conn.execute(
                    f"ALTER TABLE \"{table}\" ADD COLUMN \"{abnormal_col}\" INTEGER"
                )
                # Set 1 for abnormal, 0 for normal/borderline, NULL if no status
                for label in abnormal_labels:
                    conn.execute(
                        f"UPDATE \"{table}\" SET \"{abnormal_col}\" = 1 "
                        f"WHERE LOWER(\"{status_col}\") = ?",
                        (label,),
                    )
                for label in normal_labels | borderline_labels:
                    conn.execute(
                        f"UPDATE \"{table}\" SET \"{abnormal_col}\" = 0 "
                        f"WHERE LOWER(\"{status_col}\") = ?",
                        (label,),
                    )
        conn.commit()
        conn.close()
    except Exception:
        return ""

    # Build a short guide text for the planner
    guide_parts = [
        "IMPORTANT — Pre-computed boolean columns for abnormality filtering:",
    ]
    abnormal_cols: list[str] = []
    for table, cols in status_map.items():
        for status_col in cols:
            base = status_col.removesuffix("_status")
            col_ref = f"{table}.{base}_abnormal"
            abnormal_cols.append(col_ref)
            guide_parts.append(
                f"  - {col_ref}: 1 = truly abnormal, 0 = normal/borderline, NULL = no data"
            )
    guide_parts.append("")
    guide_parts.append("Label classification used:")
    if abnormal_labels:
        guide_parts.append(f"  ABNORMAL (= 1): {sorted(abnormal_labels)}")
    if borderline_labels:
        guide_parts.append(f"  BORDERLINE (= 0, NOT abnormal): {sorted(borderline_labels)}")
    if normal_labels:
        guide_parts.append(f"  NORMAL (= 0): {sorted(normal_labels)}")
    guide_parts.append("")
    guide_parts.append(
        "CRITICAL: When the question asks about 'abnormal' values, you MUST use "
        "the _abnormal column (e.g. WHERE CRE_abnormal = 1). Do NOT use numeric "
        "thresholds or _status text columns — the _abnormal column already encodes "
        "the correct domain-specific classification."
    )

    return "\n".join(guide_parts)


def _find_db_paths(task: PublicTask) -> list[Path]:
    context_dir = task.context_dir
    return sorted(context_dir.rglob("*.db")) + sorted(context_dir.rglob("*.sqlite"))


def _find_large_docs(task: PublicTask) -> list[Path]:
    doc_exts = {".md", ".txt", ".text"}
    large: list[Path] = []
    for p in sorted(task.context_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in doc_exts and p.name != "knowledge.md":
            if p.stat().st_size > LARGE_DOC_THRESHOLD_BYTES:
                large.append(p)
    return large


LogCallback = Callable[[str], None]


class InvestigationAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: InvestigationAgentConfig | None = None,
        fast_model: ModelAdapter | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.model = model
        self.fast_model = fast_model
        self.tools = tools
        self.config = config or InvestigationAgentConfig()
        self._log = log_callback or (lambda _msg: None)

    def run(self, task: PublicTask) -> AgentRunResult:
        self._log(f"Starting investigation for {task.task_id}")

        # Clean up stale _consolidated.db from prior timed-out runs
        stale_db = task.context_dir / CONSOLIDATED_DB_NAME
        if stale_db.exists():
            try:
                stale_db.unlink()
            except OSError:
                pass

        evidence = _EvidenceStore()
        all_steps: list[StepRecord] = []
        step_counter = 0
        gaps_text = ""
        context_listing = _context_listing_text(task)
        tool_descriptions = "\n".join(
            line
            for line in self.tools.describe_for_prompt().splitlines()
            if not line.startswith("- answer:")
        )

        schema_hint = self._build_schema_hint(task)
        if schema_hint:
            self._log("Found data schema")
            context_listing = f"{context_listing}\n\nDATA SCHEMA:\n{schema_hint}"

        kg_model = self.fast_model or self.model
        self._log("Building grounding pipeline...")

        knowledge_text = get_full_knowledge_text(task.context_dir)

        # Phase 1: Consolidate structured data (CSV/JSON/DB) into SQLite
        consolidated_db = consolidate_to_sqlite(task.context_dir)
        if consolidated_db:
            self._log(f"Consolidated structured data into {CONSOLIDATED_DB_NAME}")

        # Phase 2: Graph-based extraction from unstructured docs → write directly to SQLite
        large_docs = _find_large_docs(task)
        if large_docs:
            self._log(f"Extracting graph from {len(large_docs)} document(s)...")
            if not consolidated_db:
                consolidated_db = task.context_dir / CONSOLIDATED_DB_NAME

            # Try to get entity schemas from knowledge.md to skip discovery
            extraction_model = self.fast_model or self.model
            grounding_schemas = get_entity_schemas_for_task(
                task.context_dir, model=extraction_model
            )
            known_schemas = _entity_schemas_from_grounding(grounding_schemas) if grounding_schemas else None

            extracted_tables = extract_multiple_docs_to_sqlite(
                extraction_model, large_docs, task.question, consolidated_db,
                known_schemas=known_schemas,
            )
            if extracted_tables:
                self._log(f"Extracted tables: {', '.join(extracted_tables)}")
                step_counter += 1
                all_steps.append(
                    StepRecord(
                        step_index=step_counter,
                        thought=f"graph-extracted {len(extracted_tables)} entity tables from docs",
                        action="__extract_graph__",
                        action_input={"tables": extracted_tables},
                        raw_response="",
                        observation={"ok": True, "tables": extracted_tables},
                        ok=True,
                    )
                )
                evidence.add(
                    "graph_extraction",
                    f"Extracted {len(extracted_tables)} entity tables from unstructured docs",
                    {"ok": True, "content": f"Tables created: {', '.join(extracted_tables)}. Use SQL to query."},
                )

        # Phase 3: Classify status labels and materialize _abnormal boolean columns
        # (must happen before schema read so _abnormal columns are visible)
        domain_guide = ""
        if consolidated_db and consolidated_db.exists():
            domain_guide = _classify_and_materialize(
                model=kg_model,
                db_path=consolidated_db,
                question=task.question,
                knowledge_text=knowledge_text[:4000] if knowledge_text else "",
            )
            if domain_guide:
                self._log("Classified status labels and materialized _abnormal columns")

        # Phase 4: Build consolidated schema (includes _abnormal columns)
        consolidated_schema = ""
        if consolidated_db and consolidated_db.exists():
            consolidated_schema = get_consolidated_schema(consolidated_db)
            context_listing = (
                f"{context_listing}\n\n"
                f"CONSOLIDATED DATABASE ({CONSOLIDATED_DB_NAME}):\n"
                f"All data (structured + extracted from docs) in one SQLite.\n"
                f"{consolidated_schema}"
            )
            if domain_guide:
                context_listing = f"{context_listing}\n\nDOMAIN INTERPRETATION GUIDE:\n{domain_guide}"

        # Phase 5: Knowledge graph with validation against consolidated DB
        kg_schema = consolidated_schema if consolidated_schema else schema_hint
        self._log("Building knowledge graph (with validation loop)...")
        kg_result = build_knowledge_graph(
            model=kg_model,
            file_tree=context_listing,
            schema_hint=kg_schema,
            knowledge_text=knowledge_text[:6000] if knowledge_text else "",
            question=task.question,
            db_path=consolidated_db,
        )

        kg_text = render_knowledge_graph(kg_result)
        if kg_text:
            self._log("Knowledge graph built and validated")
            context_listing = f"{context_listing}\n\nKNOWLEDGE GRAPH:\n{kg_text}"
            evidence.add(
                "knowledge_graph",
                "Grounding: computation steps, joins, filters, domain rules",
                {"ok": True, "content": kg_text},
            )

        # Build full schema for evaluator/synthesizer
        full_schema = schema_hint
        if consolidated_schema:
            full_schema = f"{full_schema}\n\n{consolidated_schema}"
        if kg_text:
            full_schema = f"{full_schema}\n\n{kg_text}"
        if domain_guide:
            full_schema = f"{full_schema}\n\nDOMAIN INTERPRETATION GUIDE:\n{domain_guide}"

        seen_gap_keys: list[frozenset[str]] = []
        prev_evidence_count = len(evidence.entries)

        for iteration in range(1, self.config.max_iterations + 1):
            self._log(f"--- Iteration {iteration}/{self.config.max_iterations} ---")
            self._log("Planning next steps...")
            plan_steps, plan_raw = self._plan(
                task=task,
                tool_descriptions=tool_descriptions,
                context_listing=context_listing,
                evidence=evidence,
                gaps=gaps_text,
            )
            self._log(f"Planner produced {len(plan_steps)} steps")
            step_counter += 1
            all_steps.append(
                StepRecord(
                    step_index=step_counter,
                    thought=f"iteration {iteration}: planning {len(plan_steps)} steps",
                    action="__plan__",
                    action_input={},
                    raw_response=plan_raw,
                    observation={"plan_step_count": len(plan_steps)},
                    ok=True,
                )
            )

            if not plan_steps:
                self._log("No steps planned, ending loop")
                break

            for planned in plan_steps:
                if step_counter >= self.config.max_steps:
                    break
                step_counter += 1
                step_id = planned.get("id", f"step_{step_counter}")
                description = planned.get("description", "")
                tool_name = planned.get("tool", "")
                action_input = planned.get("action_input", {})

                if tool_name == "answer":
                    continue

                self._log(f"[{step_id}] {tool_name}: {description}")

                try:
                    result = self.tools.execute(task, tool_name, action_input)
                    ok_str = "OK" if result.ok else "FAIL"
                    self._log(f"  -> {ok_str}")
                    observation = {
                        "ok": result.ok,
                        "tool": tool_name,
                        "content": result.content,
                    }
                    evidence.add(
                        step_id,
                        description,
                        {
                            "ok": result.ok,
                            "content": result.content,
                        },
                    )
                except Exception as exc:
                    self._log(f"  -> ERROR: {exc}")
                    observation = {"ok": False, "error": str(exc)}
                    evidence.add(
                        step_id,
                        description,
                        {
                            "ok": False,
                            "error": str(exc),
                        },
                    )

                all_steps.append(
                    StepRecord(
                        step_index=step_counter,
                        thought=description,
                        action=tool_name,
                        action_input=action_input,
                        raw_response="",
                        observation=observation,
                        ok=observation.get("ok", False),
                    )
                )

            if step_counter >= self.config.max_steps:
                self._log("Max steps reached")
                break

            anomaly_flags = detect_anomalies(evidence.entries)
            anomaly_text = format_anomaly_flags(anomaly_flags)

            self._log("Evaluating evidence completeness...")
            eval_result, eval_raw = self._evaluate(
                task=task,
                evidence=evidence,
                anomaly_text=anomaly_text,
                schema=full_schema,
            )
            step_counter += 1
            all_steps.append(
                StepRecord(
                    step_index=step_counter,
                    thought=f"iteration {iteration}: evaluating evidence",
                    action="__evaluate__",
                    action_input={},
                    raw_response=eval_raw,
                    observation=eval_result,
                    ok=True,
                )
            )

            verdict = eval_result.get("verdict", "complete")
            self._log(f"Evaluator verdict: {verdict}")
            if verdict == "complete":
                break

            gaps = eval_result.get("gaps", [])
            if not gaps:
                break

            for g in gaps:
                self._log(f"  Gap: {g}")

            error_sigs = frozenset(
                _normalize_gap_key(e.get("error", ""))
                for e in evidence.entries.values()
                if not e.get("ok", True) and e.get("error")
            )
            current_gap_key = frozenset(_normalize_gap_key(g) for g in gaps) | error_sigs
            current_evidence_count = len(evidence.entries)
            no_new_evidence = current_evidence_count == prev_evidence_count
            is_repeat = current_gap_key in seen_gap_keys or any(
                current_gap_key <= prev for prev in seen_gap_keys
            )
            if is_repeat or no_new_evidence:
                if no_new_evidence:
                    self._log("No new evidence collected, stopping")
                else:
                    self._log("Repeated gaps detected, stopping")
                break
            seen_gap_keys.append(current_gap_key)
            prev_evidence_count = current_evidence_count

            gaps_text = "\n".join(f"- {g}" for g in gaps)

        self._log("Synthesizing final answer...")
        answer = self._synthesize(task=task, evidence=evidence, schema=full_schema)
        step_counter += 1
        all_steps.append(
            StepRecord(
                step_index=step_counter,
                thought="synthesizing final answer",
                action="__synthesize__",
                action_input={},
                raw_response="",
                observation={"answer": answer.to_dict() if answer else None},
                ok=answer is not None,
            )
        )

        consolidated_path = task.context_dir / CONSOLIDATED_DB_NAME
        if consolidated_path.exists():
            try:
                consolidated_path.unlink()
            except OSError:
                pass

        if answer is not None:
            self._log(f"Answer: {len(answer.columns)} columns, {len(answer.rows)} rows")
        else:
            self._log("Synthesis failed — no answer produced")

        failure_reason = None
        if answer is None:
            failure_reason = "Investigation could not synthesize an answer from collected evidence."

        return AgentRunResult(
            task_id=task.task_id,
            answer=answer,
            steps=all_steps,
            failure_reason=failure_reason,
        )

    def _build_schema_hint(self, task: PublicTask) -> str:
        parts: list[str] = []

        db_paths = _find_db_paths(task)
        if db_paths:
            try:
                _, _, rendered = slice_schema_for_task(db_paths)
                if rendered:
                    parts.append(rendered)
            except Exception:
                pass

        for csv_path in sorted(task.context_dir.rglob("*.csv")):
            if csv_path.name.startswith("_extracted_"):
                continue
            rel = csv_path.relative_to(task.context_dir).as_posix()
            try:
                with csv_path.open(encoding="utf-8", errors="replace") as f:
                    header = f.readline().strip()
                    sample = f.readline().strip()
                if header:
                    cols = header.split(",")
                    vals = sample.split(",") if sample else []
                    typed_cols = []
                    for ci, col in enumerate(cols):
                        val = vals[ci].strip().strip('"') if ci < len(vals) else ""
                        hint = f"{col} {_infer_type(val)}"
                        if val and _infer_type(val) == "(text)" and len(val) <= 40:
                            hint += f' e.g. "{val}"'
                        typed_cols.append(hint)
                    parts.append(f"CSV {rel} ({', '.join(typed_cols)})")
            except Exception:
                pass

        for json_path in sorted(task.context_dir.rglob("*.json")):
            if json_path.name == "task.json":
                continue
            rel = json_path.relative_to(task.context_dir).as_posix()
            file_size = json_path.stat().st_size
            try:
                parts.append(self._json_schema_hint(json_path, rel, file_size))
            except Exception:
                pass

        return "\n".join(parts)

    @staticmethod
    def _json_schema_hint(json_path: Path, rel: str, file_size: int) -> str:
        max_read = 32_768
        if file_size <= max_read:
            raw = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                typed = [_json_field_hint(k, v) for k, v in raw[0].items()]
                return f"JSON {rel} (array of {len(raw)} objects, fields: {typed})"
            if isinstance(raw, dict):
                top_keys = list(raw.keys())[:10]
                records = raw.get("records", raw.get("data"))
                if isinstance(records, list) and records and isinstance(records[0], dict):
                    typed = [_json_field_hint(k, v) for k, v in records[0].items()]
                    return (
                        f"JSON {rel} (top keys: {top_keys}, "
                        f"records: {len(records)} objects, fields: {typed})"
                    )
                return f"JSON {rel} (keys: {top_keys})"
            return f"JSON {rel}"

        with json_path.open(encoding="utf-8", errors="replace") as f:
            head = f.read(max_read)
        first_obj_keys = _extract_first_object_keys(head)
        size_mb = file_size / 1_048_576
        if first_obj_keys is not None:
            top_level, _record_names, record_keys_typed = first_obj_keys
            if record_keys_typed:
                typed_strs = []
                for k, t, s in record_keys_typed:
                    hint = f"{k} {t}"
                    if s and t == "(text)" and len(s) <= 40:
                        hint += f' e.g. "{s}"'
                    typed_strs.append(hint)
                return (
                    f"JSON {rel} ({size_mb:.1f}MB, top keys: {top_level}, "
                    f"record fields: {typed_strs})"
                )
            return f"JSON {rel} ({size_mb:.1f}MB, keys: {top_level})"
        return f"JSON {rel} ({size_mb:.1f}MB — use execute_python to inspect)"


    def _plan(
        self,
        *,
        task: PublicTask,
        tool_descriptions: str,
        context_listing: str,
        evidence: _EvidenceStore,
        gaps: str,
    ) -> tuple[list[dict[str, Any]], str]:
        prompt = build_investigation_planner_prompt(
            question=task.question,
            tool_descriptions=tool_descriptions,
            context_listing=context_listing,
            evidence=evidence.render(),
            gaps=gaps,
        )
        messages = [
            ModelMessage(role="system", content="You are a data investigation planner."),
            ModelMessage(role="user", content=prompt),
        ]
        adapter = self.fast_model or self.model
        raw = adapter.complete(messages)
        try:
            steps = _parse_json(raw)
            if not isinstance(steps, list):
                return [], raw
            return steps, raw
        except (json.JSONDecodeError, ValueError):
            return [], raw

    def _evaluate(
        self,
        *,
        task: PublicTask,
        evidence: _EvidenceStore,
        anomaly_text: str = "",
        schema: str = "",
    ) -> tuple[dict[str, Any], str]:
        evidence_text = evidence.render()
        if anomaly_text:
            evidence_text = f"{evidence_text}\n\n{anomaly_text}"
        prompt = build_investigation_evaluator_prompt(
            question=task.question,
            evidence=evidence_text,
            schema=schema,
        )
        messages = [
            ModelMessage(role="system", content="You are an evidence evaluator."),
            ModelMessage(role="user", content=prompt),
        ]
        adapter = self.fast_model or self.model
        raw = adapter.complete(messages)
        try:
            result = _parse_json(raw)
            if isinstance(result, dict):
                return result, raw
        except (json.JSONDecodeError, ValueError):
            pass
        return {"verdict": "complete", "reasoning": "Could not parse evaluation.", "gaps": []}, raw

    def _synthesize(
        self, *, task: PublicTask, evidence: _EvidenceStore, schema: str = ""
    ) -> AnswerTable | None:
        prompt = build_investigation_synthesizer_prompt(
            question=task.question,
            evidence=evidence.render(),
            schema=schema,
        )
        messages = [
            ModelMessage(role="system", content="You are a data investigation synthesizer."),
            ModelMessage(role="user", content=prompt),
        ]
        raw = self.model.complete(messages)
        try:
            result = _parse_json(raw)
            if not isinstance(result, dict):
                return None
            columns = result.get("columns")
            rows = result.get("rows")
            if not isinstance(columns, list) or not columns or not isinstance(rows, list):
                return None
            normalized_rows = []
            for row in rows:
                if not isinstance(row, list) or len(row) != len(columns):
                    continue
                normalized_rows.append([str(v) for v in row])
            return AnswerTable(
                columns=[str(c) for c in columns],
                rows=normalized_rows,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return None
