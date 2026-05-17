from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


@dataclass(frozen=True, slots=True)
class MetricDef:
    name: str
    formula: str
    description: str


@dataclass(frozen=True, slots=True)
class ExemplarCase:
    title: str
    sql: str
    explanation: str


@dataclass(frozen=True, slots=True)
class EntityField:
    name: str
    field_type: str
    description: str


@dataclass(frozen=True, slots=True)
class EntitySchema:
    entity_name: str
    fields: list[EntityField]


@dataclass(slots=True)
class GroundingContext:
    database_name: str = ""
    entities: list[EntitySchema] = field(default_factory=list)
    metrics: list[MetricDef] = field(default_factory=list)
    constraints_text: str = ""
    exemplars: list[ExemplarCase] = field(default_factory=list)
    ambiguity_text: str = ""
    raw_text: str = ""


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in text.lower().split():
        w = raw.strip(".,;:!?()\"'`*_[]{}#/")
        if len(w) >= 3:
            tokens.add(w)
    return tokens


_PARSE_GROUNDING_PROMPT = """
Parse the following knowledge document into structured JSON.

Return ONLY a JSON object (no markdown fences, no explanation) with these fields:
{{
  "database_name": "string or empty",
  "entities": [
    {{
      "entity_name": "string",
      "fields": [
        {{"name": "string", "field_type": "string (e.g. text, integer, real, date)", "description": "string"}}
      ]
    }}
  ],
  "metrics": [
    {{"name": "string", "formula": "string", "description": "string"}}
  ],
  "constraints_text": "string - full text of constraints/conventions section",
  "exemplars": [
    {{"title": "string", "sql": "string - the SQL query if present", "explanation": "string"}}
  ],
  "ambiguity_text": "string - full text of ambiguity resolution section"
}}

RULES:
- Extract ALL entities, metrics, and exemplars.
- For field_type, extract from parentheses if present (e.g. "ID (integer)" -> type "integer").
  Otherwise use "text".
- For SQL, extract the actual query from code blocks or inline backticks.
- For constraints_text and ambiguity_text, preserve the original text.
- If a section is missing, use "" or [].

DOCUMENT:
{text}
""".strip()


def _strip_json_fence(raw: str) -> str:
    raw = raw.strip()
    fence = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    generic = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return raw


def _build_context_from_json(data: dict[str, Any], raw_text: str) -> GroundingContext:
    ctx = GroundingContext(raw_text=raw_text)
    ctx.database_name = data.get("database_name", "")
    ctx.constraints_text = data.get("constraints_text", "")
    ctx.ambiguity_text = data.get("ambiguity_text", "")

    for m in data.get("metrics", []):
        if not isinstance(m, dict):
            continue
        ctx.metrics.append(
            MetricDef(
                name=m.get("name", ""),
                formula=m.get("formula", ""),
                description=m.get("description", ""),
            )
        )

    for ex in data.get("exemplars", []):
        if not isinstance(ex, dict):
            continue
        ctx.exemplars.append(
            ExemplarCase(
                title=ex.get("title", ""),
                sql=ex.get("sql", ""),
                explanation=ex.get("explanation", ""),
            )
        )

    for e in data.get("entities", []):
        if not isinstance(e, dict):
            continue
        fields: list[EntityField] = []
        for f in e.get("fields", []):
            if not isinstance(f, dict):
                continue
            fields.append(
                EntityField(
                    name=f.get("name", ""),
                    field_type=f.get("field_type", "text"),
                    description=f.get("description", ""),
                )
            )
        if fields:
            ctx.entities.append(
                EntitySchema(entity_name=e.get("entity_name", ""), fields=fields)
            )

    return ctx


def parse_grounding(text: str, *, model: ModelAdapter) -> GroundingContext:
    messages = [
        ModelMessage(
            role="system",
            content="You are a structured document parser. Extract information exactly as requested.",
        ),
        ModelMessage(role="user", content=_PARSE_GROUNDING_PROMPT.format(text=text)),
    ]
    try:
        raw = model.complete(messages, thinking=False)
        data = json.loads(_strip_json_fence(raw))
        if isinstance(data, dict):
            return _build_context_from_json(data, text)
    except (json.JSONDecodeError, ValueError, Exception):
        pass
    return GroundingContext(raw_text=text)


def extract_entity_schemas(ctx: GroundingContext) -> list[EntitySchema]:
    return ctx.entities


def render_extraction_schema(schemas: list[EntitySchema]) -> str:
    lines: list[str] = []
    for s in schemas:
        lines.append(f"Entity: {s.entity_name}")
        for f in s.fields:
            lines.append(f"  - {f.name} ({f.field_type}): {f.description}")
    return "\n".join(lines)


def extract_relevant_grounding(ctx: GroundingContext, question: str) -> str:
    q_words = _tokenize(question)

    lines: list[str] = []

    lines.append(f"DATABASE: {ctx.database_name}" if ctx.database_name else "DATABASE: unknown")

    if ctx.constraints_text:
        lines.append("\nCONSTRAINTS & CONVENTIONS:")
        lines.append(ctx.constraints_text[:1500])

    if ctx.ambiguity_text:
        lines.append("\nAMBIGUITY NOTES:")
        lines.append(ctx.ambiguity_text[:1000])

    if ctx.metrics:
        scored = []
        for m in ctx.metrics:
            m_words = _tokenize(f"{m.name} {m.description}")
            overlap = len(q_words & m_words)
            scored.append((overlap, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [m for score, m in scored[:5] if score > 0]
        if not top:
            top = [m for _, m in scored[:3]]
        if top:
            lines.append("\nRELEVANT METRICS:")
            for m in top:
                lines.append(f"  - {m.name}: {m.formula}")
                if m.description:
                    lines.append(f"    {m.description}")

    if ctx.exemplars:
        scored_ex = []
        for ex in ctx.exemplars:
            ex_words = _tokenize(f"{ex.title} {ex.explanation}")
            overlap = len(q_words & ex_words)
            scored_ex.append((overlap, ex))
        scored_ex.sort(key=lambda x: x[0], reverse=True)
        top_ex = [ex for score, ex in scored_ex[:3] if score > 0]
        if top_ex:
            lines.append("\nRELEVANT EXAMPLES:")
            for ex in top_ex:
                lines.append(f"  [{ex.title}]")
                if ex.sql:
                    lines.append(f"    SQL: {ex.sql}")

    return "\n".join(lines)


_grounding_cache: dict[str, GroundingContext] = {}


def get_grounding_for_task(
    task_context_dir: Path, question: str, *, model: ModelAdapter
) -> str:
    knowledge_path = _find_knowledge_md(task_context_dir)
    if knowledge_path is None:
        return ""

    cache_key = str(knowledge_path.resolve())
    if cache_key in _grounding_cache:
        ctx = _grounding_cache[cache_key]
    else:
        text = knowledge_path.read_text(encoding="utf-8", errors="replace")
        ctx = parse_grounding(text, model=model)
        _grounding_cache[cache_key] = ctx

    return extract_relevant_grounding(ctx, question)


def get_full_knowledge_text(task_context_dir: Path) -> str:
    knowledge_path = _find_knowledge_md(task_context_dir)
    if knowledge_path is None:
        return ""
    return knowledge_path.read_text(encoding="utf-8", errors="replace")


def get_entity_schemas_for_task(
    task_context_dir: Path, *, model: ModelAdapter
) -> list[EntitySchema]:
    knowledge_path = _find_knowledge_md(task_context_dir)
    if knowledge_path is None:
        return []
    cache_key = str(knowledge_path.resolve())
    if cache_key in _grounding_cache:
        ctx = _grounding_cache[cache_key]
    else:
        text = knowledge_path.read_text(encoding="utf-8", errors="replace")
        ctx = parse_grounding(text, model=model)
        _grounding_cache[cache_key] = ctx
    return extract_entity_schemas(ctx)


def _find_knowledge_md(context_dir: Path) -> Path | None:
    direct = context_dir / "knowledge.md"
    if direct.is_file():
        return direct
    for p in context_dir.rglob("knowledge.md"):
        return p
    return None
