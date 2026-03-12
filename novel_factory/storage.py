"""Filesystem-based checkpoint and artifact storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from novel_factory.config import AppConfig
from novel_factory.utils import (
    ensure_directory,
    format_chapter_number,
    format_scene_number,
    json_dumps,
    serialise_model,
    timestamp_utc,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class RunStorage:
    """Persists all project artifacts under runs/<project_slug>/."""

    def __init__(self, config: AppConfig, project_slug: str) -> None:
        self.root = ensure_directory(config.run_root / project_slug)
        self.config = config
        # Create subdirectories
        self.scenes_dir = ensure_directory(self.root / "scenes")
        self.qa_dir = ensure_directory(self.root / "qa")
        self.rewrites_dir = ensure_directory(self.root / "rewrites")
        self.chapters_dir = ensure_directory(self.root / "chapters")
        # NEW: additional artifact directories
        self.voice_dir = ensure_directory(self.root / "voice")
        self.passes_dir = ensure_directory(self.root / "passes")
        self.candidates_dir = ensure_directory(self.root / "candidates")

    # ── Path helpers ──────────────────────────────────────────────────

    @property
    def synopsis_path(self) -> Path:
        return self.root / "synopsis.md"

    @property
    def story_spec_path(self) -> Path:
        return self.root / "story_spec.json"

    @property
    def beat_sheet_path(self) -> Path:
        return self.root / "beat_sheet.json"

    @property
    def plant_payoff_path(self) -> Path:
        return self.root / "plant_payoff_map.json"

    @property
    def subplot_weave_path(self) -> Path:
        return self.root / "subplot_weave_map.json"

    @property
    def voice_dna_path(self) -> Path:
        return self.voice_dir / "voice_dna.json"

    @property
    def outline_path(self) -> Path:
        return self.root / "outline.json"

    @property
    def scene_cards_path(self) -> Path:
        return self.root / "scene_cards.json"

    @property
    def continuity_path(self) -> Path:
        return self.root / "continuity_state.json"

    @property
    def manuscript_md_path(self) -> Path:
        return self.root / "manuscript.md"

    @property
    def manuscript_txt_path(self) -> Path:
        return self.root / "manuscript.txt"

    @property
    def run_log_path(self) -> Path:
        return self.root / "run_log.jsonl"

    @property
    def cold_reader_path(self) -> Path:
        return self.qa_dir / "cold_reader_report.json"

    @property
    def pacing_analysis_path(self) -> Path:
        return self.qa_dir / "pacing_analysis.json"

    @property
    def anti_ai_report_path(self) -> Path:
        return self.passes_dir / "anti_ai_report.json"

    @property
    def editorial_blueprint_path(self) -> Path:
        return self.root / "editorial_blueprint.json"

    @property
    def global_qa_path(self) -> Path:
        return self.qa_dir / "global_qa.json"

    def scene_path(self, scene_number: int) -> Path:
        return self.scenes_dir / f"scene_{format_scene_number(scene_number)}.md"

    def scene_qa_path(self, scene_number: int) -> Path:
        return self.qa_dir / f"scene_{format_scene_number(scene_number)}_qa.json"

    def scene_validation_path(self, scene_number: int) -> Path:
        return self.qa_dir / f"scene_{format_scene_number(scene_number)}_validation.json"

    def chapter_path(self, chapter_number: int) -> Path:
        return self.chapters_dir / f"chapter_{format_chapter_number(chapter_number)}.md"

    def chapter_qa_path(self, chapter_number: int) -> Path:
        return self.qa_dir / f"chapter_{format_chapter_number(chapter_number)}_qa.json"

    def arc_qa_path(self, arc_name: str) -> Path:
        safe_name = arc_name.lower().replace(" ", "_").replace("/", "_")
        return self.qa_dir / f"arc_qa_{safe_name}.json"

    def candidate_path(self, scene_number: int, candidate_index: int) -> Path:
        return self.candidates_dir / f"scene_{format_scene_number(scene_number)}_candidate_{candidate_index}.md"

    def character_voice_path(self, character_name: str) -> Path:
        safe_name = character_name.lower().replace(" ", "_")
        return self.voice_dir / f"voice_{safe_name}.json"

    def transition_report_path(self, scene_a: int, scene_b: int) -> Path:
        return self.passes_dir / f"transition_{format_scene_number(scene_a)}_to_{format_scene_number(scene_b)}.json"

    def dialogue_audit_path(self, character_name: str) -> Path:
        safe_name = character_name.lower().replace(" ", "_")
        return self.passes_dir / f"dialogue_audit_{safe_name}.json"

    def prose_rhythm_path(self, scene_number: int) -> Path:
        return self.passes_dir / f"prose_rhythm_{format_scene_number(scene_number)}.json"

    def rewrite_path(self, scene_number: int, attempt: int) -> Path:
        return self.rewrites_dir / f"scene_{format_scene_number(scene_number)}_rewrite_{attempt}.md"

    # ── I/O helpers ───────────────────────────────────────────────────

    def save_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=self.config.synopsis_encoding)

    def load_text(self, path: Path) -> str:
        return path.read_text(encoding=self.config.synopsis_encoding)

    def save_model(self, path: Path, model: BaseModel) -> None:
        self.save_text(path, serialise_model(model))

    def load_model(self, path: Path, schema: type[ModelT]) -> ModelT:
        raw = self.load_text(path)
        return schema.model_validate_json(raw)

    def save_model_list(self, path: Path, models: list[BaseModel]) -> None:
        data = [m.model_dump() for m in models]
        self.save_text(path, json_dumps(data))

    def load_model_list(self, path: Path, schema: type[ModelT]) -> list[ModelT]:
        raw = json.loads(self.load_text(path))
        return [schema.model_validate(item) for item in raw]

    def append_log(self, event_type: str, details: str = "", **kwargs) -> None:
        from novel_factory.schemas import RunLogEvent
        event = RunLogEvent(
            timestamp=timestamp_utc(),
            event_type=event_type,
            details=details,
            **kwargs,
        )
        with open(self.run_log_path, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    def has_approved_scene(self, scene_number: int) -> bool:
        scene_file = self.scene_path(scene_number)
        qa_file = self.scene_qa_path(scene_number)
        if not scene_file.exists() or not qa_file.exists():
            return False
        from novel_factory.schemas import SceneQaReport
        try:
            report = self.load_model(qa_file, SceneQaReport)
            return report.passed
        except Exception:
            return False

    def exists(self, path: Path) -> bool:
        return path.exists()
