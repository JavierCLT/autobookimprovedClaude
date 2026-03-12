"""Model-based QA judges: scene, chapter, arc, global, cold reader, pacing."""

from __future__ import annotations

from novel_factory.config import AppConfig
from novel_factory.intake import build_drafting_guidance, build_planning_guidance
from novel_factory.llm import OpenAIResponsesClient
from novel_factory.prompts import (
    arc_qa_system_prompt,
    arc_qa_user_prompt,
    chapter_qa_system_prompt,
    chapter_qa_user_prompt,
    cold_reader_system_prompt,
    cold_reader_user_prompt,
    global_qa_system_prompt,
    global_qa_user_prompt,
    pacing_analysis_system_prompt,
    pacing_analysis_user_prompt,
    scene_qa_system_prompt,
    scene_qa_user_prompt,
)
from novel_factory.schemas import (
    ArcQaReport,
    BookIntake,
    ChapterQaReport,
    ColdReaderReport,
    ContinuityState,
    DeterministicValidationReport,
    GlobalQaReport,
    Outline,
    PacingAnalysis,
    SceneCard,
    SceneQaReport,
    StorySpec,
)


class SceneJudge:
    """Runs the scene-level model judge."""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def judge(
        self,
        *,
        story_spec: StorySpec,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        validation_report: DeterministicValidationReport,
        scene_text: str,
        book_intake: BookIntake | None = None,
    ) -> SceneQaReport:
        return self.llm.structured(
            system_prompt=scene_qa_system_prompt(story_spec),
            user_prompt=scene_qa_user_prompt(
                story_spec=story_spec,
                scene_card=scene_card,
                continuity_state=continuity_state,
                validation_report=validation_report,
                scene_text=scene_text,
                intake_guidance=build_drafting_guidance(book_intake),
            ),
            schema=SceneQaReport,
            task_name=f"scene_qa_{scene_card.scene_number:02d}",
            reasoning_effort=self.config.reasoning.qa,
            temperature=self.config.qa_temperature,
            max_output_tokens=2_500,
            verbosity="low",
            model_override=self.config.get_qa_model(),
        )


class GlobalJudge:
    """Runs global, chapter, and arc-level judges."""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def judge(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        manuscript_text: str,
        book_intake: BookIntake | None = None,
    ) -> GlobalQaReport:
        return self.llm.structured(
            system_prompt=global_qa_system_prompt(story_spec),
            user_prompt=global_qa_user_prompt(
                story_spec=story_spec,
                outline=outline,
                manuscript_text=manuscript_text,
                intake_guidance=build_planning_guidance(book_intake, max_chars=8_000),
            ),
            schema=GlobalQaReport,
            task_name="global_qa",
            reasoning_effort=self.config.reasoning.global_qa,
            temperature=self.config.global_qa_temperature,
            max_output_tokens=4_000,
            verbosity="low",
            model_override=self.config.get_qa_model(),
        )

    def judge_chapter(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        chapter_number: int,
        chapter_text: str,
        scene_cards: list[SceneCard],
    ) -> ChapterQaReport:
        return self.llm.structured(
            system_prompt=chapter_qa_system_prompt(story_spec),
            user_prompt=chapter_qa_user_prompt(
                story_spec=story_spec,
                outline=outline,
                chapter_number=chapter_number,
                chapter_text=chapter_text,
                scene_cards=scene_cards,
            ),
            schema=ChapterQaReport,
            task_name=f"chapter_qa_{chapter_number:02d}",
            reasoning_effort=self.config.reasoning.qa,
            temperature=self.config.qa_temperature,
            max_output_tokens=2_000,
            verbosity="low",
            model_override=self.config.get_qa_model(),
        )

    def judge_arc(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        arc_name: str,
        arc_focus: str,
        scene_numbers: list[int],
        arc_text: str,
    ) -> ArcQaReport:
        return self.llm.structured(
            system_prompt=arc_qa_system_prompt(story_spec),
            user_prompt=arc_qa_user_prompt(
                story_spec=story_spec,
                outline=outline,
                arc_name=arc_name,
                arc_focus=arc_focus,
                scene_numbers=scene_numbers,
                arc_text=arc_text,
            ),
            schema=ArcQaReport,
            task_name=f"arc_qa_{arc_name}",
            reasoning_effort=self.config.reasoning.qa,
            temperature=self.config.qa_temperature,
            max_output_tokens=2_000,
            verbosity="low",
            model_override=self.config.get_qa_model(),
        )


class ColdReaderJudge:
    """Reads the manuscript with ZERO planning context — simulates a real reader. (NEW)"""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def judge(self, *, manuscript_text: str) -> ColdReaderReport:
        return self.llm.structured(
            system_prompt=cold_reader_system_prompt(),
            user_prompt=cold_reader_user_prompt(manuscript_text=manuscript_text),
            schema=ColdReaderReport,
            task_name="cold_reader",
            reasoning_effort="high",
            temperature=0.3,
            max_output_tokens=4_000,
            model_override=self.config.get_qa_model(),
        )


class PacingAnalyzer:
    """Maps tension curves across the full manuscript. (NEW)"""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def analyze(self, *, manuscript_text: str, scene_count: int) -> PacingAnalysis:
        return self.llm.structured(
            system_prompt=pacing_analysis_system_prompt(),
            user_prompt=pacing_analysis_user_prompt(
                manuscript_text=manuscript_text,
                scene_count=scene_count,
            ),
            schema=PacingAnalysis,
            task_name="pacing_analysis",
            reasoning_effort="high",
            temperature=0.2,
            max_output_tokens=4_000,
            model_override=self.config.get_qa_model(),
        )
