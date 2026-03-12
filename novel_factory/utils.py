"""Utility helpers shared across the repository."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from novel_factory.schemas import ChapterPlan, Outline, SceneCard

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "project"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)


def write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


def json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True)


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(stripped) if chunk.strip()]


def split_paragraphs(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [chunk.strip() for chunk in PARAGRAPH_SPLIT_RE.split(stripped) if chunk.strip()]


def first_token(text: str) -> str:
    match = WORD_RE.search(text)
    return match.group(0).lower() if match else ""


def plain_text_from_markdown(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    converted = []
    for line in lines:
        if line.startswith("#"):
            converted.append(line.lstrip("#").strip().upper())
        else:
            converted.append(line)
    return "\n".join(converted).strip() + "\n"


def format_scene_number(scene_number: int) -> str:
    return f"{scene_number:02d}"


def format_chapter_number(chapter_number: int) -> str:
    return f"{chapter_number:02d}"


def serialise_model(model: object) -> str:
    if hasattr(model, "model_dump"):
        data = model.model_dump()
    else:
        data = model
    return json_dumps(data)


def truncate_text(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3].rstrip() + "..."


def get_chapter_plan(outline: Outline, chapter_number: int) -> ChapterPlan:
    for chapter in outline.chapters:
        if chapter.chapter_number == chapter_number:
            return chapter
    raise KeyError(f"Chapter {chapter_number} was not found in the outline.")


def get_scene_card(scene_cards: Sequence[SceneCard], scene_number: int) -> SceneCard:
    for scene_card in scene_cards:
        if scene_card.scene_number == scene_number:
            return scene_card
    raise KeyError(f"Scene {scene_number} was not found in the scene cards.")


def chapter_scene_numbers(scene_cards: Iterable[SceneCard], chapter_number: int) -> list[int]:
    return sorted(
        [scene.scene_number for scene in scene_cards if scene.chapter_number == chapter_number]
    )


def compute_sentence_length_stats(text: str) -> dict:
    """Computes sentence length statistics for prose rhythm analysis."""
    sentences = split_sentences(text)
    if not sentences:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "variance_ratio": 0}
    lengths = [count_words(s) for s in sentences]
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((l - mean) ** 2 for l in lengths) / n
    std = variance ** 0.5
    return {
        "count": n,
        "mean": round(mean, 1),
        "std": round(std, 1),
        "min": min(lengths),
        "max": max(lengths),
        "variance_ratio": round(std / mean, 2) if mean > 0 else 0,
    }


def extract_dialogue_lines(text: str) -> list[str]:
    """Extracts all quoted dialogue lines from prose text."""
    pattern = re.compile(r'["\u201c]([^"\u201d]+)["\u201d]')
    return pattern.findall(text)


def build_ngrams(tokens: list[str], n: int) -> dict[tuple, int]:
    """Builds n-gram frequency counts from a token list."""
    counts: dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts
