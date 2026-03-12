"""Environment-backed application configuration with improved defaults."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from rich.logging import RichHandler


class ReasoningProfiles(BaseModel):
    """Reasoning effort profiles used across the pipeline."""

    planning: str = "high"
    drafting: str = "medium"
    rewriting: str = "high"
    qa: str = "high"
    global_qa: str = "high"
    repair: str = "high"
    polish: str = "high"
    voice_calibration: str = "high"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str
    model: str = "gpt-5.4"
    drafting_model: str = ""  # optional separate model for prose generation
    qa_model: str = ""  # optional separate model for QA/judging
    run_root: Path = Path("runs")
    synopsis_encoding: str = "utf-8"

    # Story defaults
    default_audience: str = "Adult"
    default_rating_ceiling: str = "R"
    default_market_position: str = "adult thriller"
    target_words: int = 40_000
    target_chapters: int = 14
    target_scenes: int = 28

    # Context windows
    recent_scene_summaries: int = 5  # increased from 3 for better continuity
    lookahead_scenes: int = 3  # NEW: how many future scene cards to include
    max_recent_scene_summary_chars: int = 2400  # increased from 1800
    max_synopsis_context_chars: int = 18_000
    max_scene_context_chars: int = 12_000  # increased from 10_000

    # Retry / reliability
    retry_attempts: int = 4
    retry_base_delay_seconds: float = 1.5
    request_timeout_seconds: float = 300.0  # increased from 240

    # Rewrite budgets
    max_scene_rewrites: int = 4  # increased from 2
    max_repair_cycles: int = 3  # NEW: how many global QA -> repair loops
    repair_improvement_threshold: float = 0.02  # stop if improvement < 2%

    # Best-of-N drafting
    best_of_n_candidates: int = 3  # NEW: generate N drafts, pick best
    best_of_n_enabled: bool = True  # NEW: toggle for cost control

    # Temperature profiles (scene-type adaptive)
    planning_temperature: float = 0.2
    drafting_temperature: float = 0.85
    drafting_temperature_action: float = 0.7  # lower for precision
    drafting_temperature_dialogue: float = 0.8
    drafting_temperature_introspective: float = 0.9  # higher for creativity
    drafting_temperature_climax: float = 0.65  # lowest for critical scenes
    rewriting_temperature: float = 0.7
    qa_temperature: float = 0.1
    global_qa_temperature: float = 0.1
    polish_temperature: float = 0.6  # NEW: for post-draft passes

    # Premium rewrite budgets
    opening_chapter_rewrite_budget: int = 6  # NEW: extra rewrites for ch 1
    closing_chapter_rewrite_budget: int = 5  # NEW: extra rewrites for final ch

    # Reasoning profiles
    reasoning: ReasoningProfiles = Field(default_factory=ReasoningProfiles)

    def get_drafting_model(self) -> str:
        return self.drafting_model or self.model

    def get_qa_model(self) -> str:
        return self.qa_model or self.model

    def get_scene_temperature(self, scene_type: str) -> float:
        """Returns adaptive temperature based on scene type."""
        temp_map = {
            "action": self.drafting_temperature_action,
            "confrontation": self.drafting_temperature_action,
            "chase": self.drafting_temperature_action,
            "dialogue": self.drafting_temperature_dialogue,
            "negotiation": self.drafting_temperature_dialogue,
            "interrogation": self.drafting_temperature_dialogue,
            "introspective": self.drafting_temperature_introspective,
            "reflective": self.drafting_temperature_introspective,
            "quiet": self.drafting_temperature_introspective,
            "climax": self.drafting_temperature_climax,
            "revelation": self.drafting_temperature_climax,
            "crisis": self.drafting_temperature_climax,
        }
        return temp_map.get(scene_type.lower(), self.drafting_temperature)


def load_config(require_api_key: bool = True) -> AppConfig:
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env.")

    return AppConfig(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-5.4").strip() or "gpt-5.4",
        drafting_model=os.getenv("OPENAI_DRAFTING_MODEL", "").strip(),
        qa_model=os.getenv("OPENAI_QA_MODEL", "").strip(),
        run_root=Path(os.getenv("NOVEL_FACTORY_RUN_ROOT", "runs")),
        default_audience=os.getenv("NOVEL_FACTORY_DEFAULT_AUDIENCE", "Adult").strip() or "Adult",
        default_rating_ceiling=os.getenv("NOVEL_FACTORY_DEFAULT_RATING_CEILING", "R").strip() or "R",
        default_market_position=(
            os.getenv("NOVEL_FACTORY_DEFAULT_MARKET_POSITION", "adult thriller").strip()
            or "adult thriller"
        ),
        best_of_n_candidates=int(os.getenv("NOVEL_FACTORY_BEST_OF_N", "3")),
        best_of_n_enabled=os.getenv("NOVEL_FACTORY_BEST_OF_N_ENABLED", "true").lower() == "true",
        max_scene_rewrites=int(os.getenv("NOVEL_FACTORY_MAX_REWRITES", "4")),
        max_repair_cycles=int(os.getenv("NOVEL_FACTORY_MAX_REPAIR_CYCLES", "3")),
    )


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
