from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)
_NUMBER_RE = re.compile(r"\d+\.?\d*")
_ID_PATTERN_RE = re.compile(r"(?:\d{5,}|rec[A-Za-z0-9]{3,})")


@dataclass(frozen=True, slots=True)
class Section:
    heading: str
    level: int
    content: str
    paragraphs: list[str]


@dataclass(frozen=True, slots=True)
class ExtractionUnit:
    unit_id: int
    doc_path: Path
    section_heading: str
    text: str
    token_estimate: int


def _is_data_bearing(paragraph: str) -> bool:
    if len(paragraph) < 30:
        return False
    if _ID_PATTERN_RE.search(paragraph):
        return True
    return len(_NUMBER_RE.findall(paragraph)) >= 2


def parse_md_structure(text: str) -> list[Section]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [Section(heading="", level=0, content=text, paragraphs=paragraphs)]

    sections: list[Section] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        sections.append(Section(heading=heading, level=level, content=content, paragraphs=paragraphs))
    return sections


def filter_data_sections(sections: list[Section], min_density: float = 0.3) -> list[Section]:
    result: list[Section] = []
    for section in sections:
        meaningful = [p for p in section.paragraphs if len(p) >= 30]
        if not meaningful:
            continue
        data_count = sum(1 for p in meaningful if _is_data_bearing(p))
        density = data_count / len(meaningful)
        if density >= min_density:
            result.append(section)
    return result


def build_extraction_units(
    doc_path: Path, max_tokens_per_unit: int = 3000
) -> list[ExtractionUnit]:
    text = doc_path.read_text(encoding="utf-8")
    sections = parse_md_structure(text)
    filtered = filter_data_sections(sections)

    if not filtered:
        return []

    units: list[ExtractionUnit] = []
    unit_id = 0

    # Collect all data-bearing paragraphs across sections, pack into units
    buffer: list[str] = []
    buffer_tokens = 0
    current_heading = filtered[0].heading if filtered else ""

    for section in filtered:
        for para in section.paragraphs:
            if len(para) < 30:
                continue
            if not _is_data_bearing(para):
                continue
            para_tokens = len(para) // 4
            if buffer and buffer_tokens + para_tokens > max_tokens_per_unit:
                combined = "\n\n".join(buffer)
                units.append(ExtractionUnit(
                    unit_id=unit_id,
                    doc_path=doc_path,
                    section_heading=current_heading,
                    text=combined,
                    token_estimate=buffer_tokens,
                ))
                unit_id += 1
                buffer = []
                buffer_tokens = 0
            if not buffer:
                current_heading = section.heading
            buffer.append(para)
            buffer_tokens += para_tokens

    if buffer:
        combined = "\n\n".join(buffer)
        units.append(ExtractionUnit(
            unit_id=unit_id,
            doc_path=doc_path,
            section_heading=current_heading,
            text=combined,
            token_estimate=buffer_tokens,
        ))
        unit_id += 1

    return units
