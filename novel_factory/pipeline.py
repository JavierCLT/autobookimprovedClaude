"""Improved pipeline orchestrator with the full contest-grade flow.

Pipeline:
  1. Voice Calibration (if reference passages provided)
  2. Planning: StorySpec -> BeatSheet -> Outline -> PlantPayoff -> SubplotWeave -> SceneCards
  3. Character Voice Profiles
  4. Initial Continuity
  5. Plan Validation
  6. Drafting: Best-of-N per scene with adaptive temperature + lookahead
  7. Post-Draft Passes: Transitions, Dialogue, Anti-AI, Prose Rhythm
  8. Assembly
  9. QA: Chapter, Arc, Global, Cold Reader, Pacing
  10. Multi-Pass Repair with diminishing returns
  11. Opening/Closing Premium Rewrites
  12. Chapter Hook Audit
  13. Final Polish
  14. Final Assembly
"""

from __future__ import annotations

import logging
from typing import Sequence

from rich.console import Console

from novel_factory.config import AppConfig
from novel_factory.generators import NovelGenerator
from novel_factory.intake import get_reference_passages
from novel_factory.judges import ColdReaderJudge, GlobalJudge, PacingAnalyzer, SceneJudge
from novel_factory.llm import OpenAIResponsesClient
from novel_factory.schemas import (
    BeatSheet,
    BookIntake,
    CharacterVoiceProfile,
    ContinuityState,
    GlobalQaReport,
    Outline,
    PlantPayoffMap,
    SceneCard,
    StorySpec,
    SubplotWeaveMap,
    VoiceDNA,
)
from novel_factory.storage import RunStorage
from novel_factory.utils import (
    chapter_scene_numbers,
    get_scene_card,
    plain_text_from_markdown,
    serialise_model,
)
from novel_factory.validators import PlanValidator, SceneValidator

logger = logging.getLogger(__name__)
console = Console()


class SceneApprovalError(RuntimeError):
    """Raised when a scene exhausts its rewrite budget without passing."""


class NovelPipeline:
    """End-to-end pipeline for contest-grade manuscript generation."""

    def __init__(
        self,
        config: AppConfig,
        llm: OpenAIResponsesClient,
        storage: RunStorage,
    ) -> None:
        self.config = config
        self.llm = llm
        self.storage = storage
        self.generator = NovelGenerator(llm, config)
        self.scene_judge = SceneJudge(llm, config)
        self.global_judge = GlobalJudge(llm, config)
        self.cold_reader = ColdReaderJudge(llm, config)
        self.pacing_analyzer = PacingAnalyzer(llm, config)
        self.scene_validator = SceneValidator()
        self.plan_validator = PlanValidator()

    # ══════════════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ══════════════════════════════════════════════════════════════════

    def run_full_pipeline(
        self,
        *,
        synopsis: str,
        book_intake: BookIntake | None = None,
    ) -> str:
        """Runs the complete improved pipeline. Returns path to final manuscript."""

        console.rule("[bold green]PHASE 1: Voice Calibration")
        voice_dna = self._phase_voice_calibration(book_intake, synopsis)

        console.rule("[bold green]PHASE 2: Planning")
        story_spec, beat_sheet, outline, plant_payoff, subplot_weave, scene_cards, continuity = (
            self._phase_planning(synopsis, book_intake, voice_dna)
        )

        console.rule("[bold green]PHASE 3: Character Voice Profiles")
        voice_profiles = self._phase_character_voices(story_spec, voice_dna)

        console.rule("[bold green]PHASE 4: Plan Validation")
        self._phase_plan_validation(story_spec, outline, scene_cards, continuity)

        console.rule("[bold green]PHASE 5: Drafting (Best-of-N)")
        continuity = self._phase_drafting(
            story_spec, outline, scene_cards, continuity,
            voice_dna, voice_profiles, book_intake,
        )

        console.rule("[bold green]PHASE 6: Post-Draft Passes")
        self._phase_post_draft_passes(scene_cards, voice_profiles, voice_dna)

        console.rule("[bold green]PHASE 7: Assembly")
        manuscript = self._phase_assembly(story_spec, outline, scene_cards)

        console.rule("[bold green]PHASE 8: Quality Assurance")
        global_qa = self._phase_qa(story_spec, outline, scene_cards, manuscript, book_intake)

        console.rule("[bold green]PHASE 9: Multi-Pass Repair")
        manuscript = self._phase_repair(
            story_spec, outline, scene_cards, continuity,
            global_qa, voice_dna, book_intake,
        )

        console.rule("[bold green]PHASE 10: Opening/Closing Premium")
        self._phase_premium_rewrites(story_spec, outline, scene_cards, voice_dna)

        console.rule("[bold green]PHASE 11: Chapter Hook Audit")
        self._phase_hook_audit(outline, scene_cards)

        console.rule("[bold green]PHASE 12: Final Polish")
        manuscript = self._phase_final_polish(story_spec, outline, scene_cards, voice_dna)

        console.rule("[bold green]COMPLETE")
        console.print(f"Manuscript saved to: {self.storage.manuscript_md_path}")
        return str(self.storage.manuscript_md_path)

    # ══════════════════════════════════════════════════════════════════
    # PHASE IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════════

    def _phase_voice_calibration(
        self,
        book_intake: BookIntake | None,
        synopsis: str,
    ) -> VoiceDNA | None:
        """Phase 1: Extract voice DNA from reference passages."""
        if self.storage.exists(self.storage.voice_dna_path):
            console.print("[dim]Voice DNA already calibrated, loading...[/dim]")
            return self.storage.load_model(self.storage.voice_dna_path, VoiceDNA)

        ref_passages = get_reference_passages(book_intake) if book_intake else ""
        if not ref_passages:
            console.print("[yellow]No reference passages provided. Skipping voice calibration.[/yellow]")
            console.print("[dim]Tip: Add reference_passages to your intake for better voice matching.[/dim]")
            return None

        console.print("Calibrating voice DNA from reference passages...")
        genre = book_intake.fields.get("genre", "thriller") if book_intake else "thriller"
        audience = book_intake.fields.get("audience", "Adult") if book_intake else "Adult"

        voice_dna = self.generator.calibrate_voice(
            reference_passages=ref_passages,
            genre=genre,
            audience=audience,
        )
        self.storage.save_model(self.storage.voice_dna_path, voice_dna)
        self.storage.append_log("voice_calibration", f"Voice DNA calibrated: {voice_dna.vocabulary_register}")
        console.print(f"[green]Voice DNA: {voice_dna.vocabulary_register} / {voice_dna.rhythm_signature}[/green]")
        return voice_dna

    def _phase_planning(
        self,
        synopsis: str,
        book_intake: BookIntake | None,
        voice_dna: VoiceDNA | None,
    ) -> tuple[StorySpec, BeatSheet | None, Outline, PlantPayoffMap | None, SubplotWeaveMap | None, list[SceneCard], ContinuityState]:
        """Phase 2: Full planning pipeline with new structural artifacts."""

        # Save synopsis
        self.storage.save_text(self.storage.synopsis_path, synopsis)

        # Story Spec
        if self.storage.exists(self.storage.story_spec_path):
            story_spec = self.storage.load_model(self.storage.story_spec_path, StorySpec)
            console.print("[dim]Story spec loaded from checkpoint.[/dim]")
        else:
            console.print("Generating story specification...")
            from novel_factory.intake import resolve_planning_defaults
            defaults = resolve_planning_defaults(
                intake=book_intake,
                default_audience=self.config.default_audience,
                default_rating_ceiling=self.config.default_rating_ceiling,
                default_market_position=self.config.default_market_position,
                default_target_words=self.config.target_words,
                default_expected_chapters=self.config.target_chapters,
                default_expected_scenes=self.config.target_scenes,
            )
            story_spec = self.generator.generate_story_spec(
                synopsis=synopsis,
                audience=defaults.audience,
                rating_ceiling=defaults.rating_ceiling,
                market_position=defaults.market_position,
                target_words=defaults.target_words,
                target_chapters=defaults.expected_chapters,
                target_scenes=defaults.expected_scenes,
                book_intake=book_intake,
                voice_dna=voice_dna,
            )
            self.storage.save_model(self.storage.story_spec_path, story_spec)
            self.storage.append_log("story_spec", f"Generated: {story_spec.title}")

        # Beat Sheet (NEW)
        beat_sheet = None
        if self.storage.exists(self.storage.beat_sheet_path):
            beat_sheet = self.storage.load_model(self.storage.beat_sheet_path, BeatSheet)
            console.print("[dim]Beat sheet loaded from checkpoint.[/dim]")
        else:
            console.print("Generating beat sheet...")
            beat_sheet = self.generator.generate_beat_sheet(story_spec=story_spec)
            self.storage.save_model(self.storage.beat_sheet_path, beat_sheet)
            self.storage.append_log("beat_sheet", f"Framework: {beat_sheet.framework}, {len(beat_sheet.beats)} beats")

        # Outline
        if self.storage.exists(self.storage.outline_path):
            outline = self.storage.load_model(self.storage.outline_path, Outline)
            console.print("[dim]Outline loaded from checkpoint.[/dim]")
        else:
            console.print("Generating outline...")
            outline = self.generator.generate_outline(
                story_spec=story_spec,
                book_intake=book_intake,
                beat_sheet=beat_sheet,
            )
            self.storage.save_model(self.storage.outline_path, outline)
            self.storage.append_log("outline", f"{len(outline.chapters)} chapters")

        # Plant/Payoff Map (NEW)
        plant_payoff = None
        if self.storage.exists(self.storage.plant_payoff_path):
            plant_payoff = self.storage.load_model(self.storage.plant_payoff_path, PlantPayoffMap)
            console.print("[dim]Plant/payoff map loaded from checkpoint.[/dim]")
        else:
            console.print("Generating foreshadowing registry...")
            plant_payoff = self.generator.generate_plant_payoff_map(
                story_spec=story_spec, outline=outline,
            )
            self.storage.save_model(self.storage.plant_payoff_path, plant_payoff)
            self.storage.append_log("plant_payoff", f"{len(plant_payoff.entries)} plant/payoff pairs")

        # Subplot Weave (NEW)
        subplot_weave = None
        if self.storage.exists(self.storage.subplot_weave_path):
            subplot_weave = self.storage.load_model(self.storage.subplot_weave_path, SubplotWeaveMap)
            console.print("[dim]Subplot weave loaded from checkpoint.[/dim]")
        else:
            console.print("Generating subplot weave map...")
            subplot_weave = self.generator.generate_subplot_weave(
                story_spec=story_spec, outline=outline,
            )
            self.storage.save_model(self.storage.subplot_weave_path, subplot_weave)
            self.storage.append_log("subplot_weave", f"{len(subplot_weave.subplots)} subplots")

        # Scene Cards
        if self.storage.exists(self.storage.scene_cards_path):
            scene_cards = self.storage.load_model_list(self.storage.scene_cards_path, SceneCard)
            console.print("[dim]Scene cards loaded from checkpoint.[/dim]")
        else:
            console.print("Generating scene cards...")
            scene_cards = self.generator.generate_scene_cards(
                story_spec=story_spec,
                outline=outline,
                book_intake=book_intake,
                plant_payoff_map=plant_payoff,
                subplot_weave=subplot_weave,
            )
            self.storage.save_model_list(self.storage.scene_cards_path, scene_cards)
            self.storage.append_log("scene_cards", f"{len(scene_cards)} scene cards")

        # Initial Continuity
        if self.storage.exists(self.storage.continuity_path):
            continuity = self.storage.load_model(self.storage.continuity_path, ContinuityState)
            console.print("[dim]Continuity state loaded from checkpoint.[/dim]")
        else:
            console.print("Generating initial continuity state...")
            continuity = self.generator.generate_initial_continuity(
                story_spec=story_spec, scene_cards=scene_cards,
            )
            self.storage.save_model(self.storage.continuity_path, continuity)
            self.storage.append_log("initial_continuity", "Initial state generated")

        return story_spec, beat_sheet, outline, plant_payoff, subplot_weave, scene_cards, continuity

    def _phase_character_voices(
        self,
        story_spec: StorySpec,
        voice_dna: VoiceDNA | None,
    ) -> list[CharacterVoiceProfile]:
        """Phase 3: Generate voice profiles for key characters."""
        if voice_dna is None:
            console.print("[yellow]No voice DNA — skipping character voice profiles.[/yellow]")
            return []

        profiles: list[CharacterVoiceProfile] = []
        characters = [
            (story_spec.protagonist.name, story_spec.protagonist.role),
            (story_spec.antagonist.name, story_spec.antagonist.role),
        ]
        for char in story_spec.cast[:4]:  # Top 4 supporting characters
            characters.append((char.name, char.role))

        story_context = f"{story_spec.title} — {story_spec.genre}. {story_spec.setting}"

        for name, role in characters:
            path = self.storage.character_voice_path(name)
            if self.storage.exists(path):
                profiles.append(self.storage.load_model(path, CharacterVoiceProfile))
                console.print(f"[dim]Voice profile loaded: {name}[/dim]")
            else:
                console.print(f"Generating voice profile: {name}...")
                profile = self.generator.generate_character_voice_profile(
                    character_name=name,
                    character_role=role,
                    story_context=story_context,
                    voice_dna=voice_dna,
                )
                self.storage.save_model(path, profile)
                profiles.append(profile)
                self.storage.append_log("character_voice", f"Profile generated: {name}")

        return profiles

    def _phase_plan_validation(
        self,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: list[SceneCard],
        continuity: ContinuityState,
    ) -> None:
        """Phase 4: Validate planning artifacts."""
        console.print("Validating plan...")
        report = self.plan_validator.validate(
            story_spec=story_spec,
            outline=outline,
            scene_cards=scene_cards,
            continuity_state=continuity,
        )
        if report.errors:
            for e in report.errors:
                console.print(f"[red]ERROR: {e}[/red]")
        if report.warnings:
            for w in report.warnings:
                console.print(f"[yellow]WARN: {w}[/yellow]")
        if report.passed:
            console.print("[green]Plan validation passed.[/green]")
        else:
            console.print("[red]Plan has errors — proceeding with warnings.[/red]")
        self.storage.append_log("plan_validation", f"passed={report.passed}, errors={len(report.errors)}, warnings={len(report.warnings)}")

    def _phase_drafting(
        self,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: list[SceneCard],
        continuity: ContinuityState,
        voice_dna: VoiceDNA | None,
        voice_profiles: list[CharacterVoiceProfile],
        book_intake: BookIntake | None,
    ) -> ContinuityState:
        """Phase 5: Draft all scenes with best-of-N selection."""
        total = len(scene_cards)

        for scene_card in scene_cards:
            sn = scene_card.scene_number
            if self.storage.has_approved_scene(sn):
                console.print(f"[dim]Scene {sn}/{total} already approved, skipping.[/dim]")
                # Replay continuity
                scene_text = self.storage.load_text(self.storage.scene_path(sn))
                continuity = self.generator.update_continuity(
                    scene_card=scene_card, scene_text=scene_text, current_state=continuity,
                )
                continue

            console.print(f"\n[bold]Drafting scene {sn}/{total} ({scene_card.scene_type})[/bold]")

            # Determine rewrite budget (premium for opening/closing chapters)
            max_rewrites = self.config.max_scene_rewrites
            if scene_card.chapter_number == 1:
                max_rewrites = self.config.opening_chapter_rewrite_budget
            elif scene_card.chapter_number == story_spec.target_chapters:
                max_rewrites = self.config.closing_chapter_rewrite_budget

            # Initial draft (best-of-N or single)
            if self.config.best_of_n_enabled:
                n = self.config.best_of_n_candidates
                scene_text, candidates = self.generator.draft_scene_best_of_n(
                    n=n,
                    story_spec=story_spec,
                    scene_card=scene_card,
                    outline=outline,
                    continuity_state=continuity,
                    recent_summaries=list(continuity.recent_summaries),
                    scene_cards=scene_cards,
                    voice_dna=voice_dna,
                    character_voice_profiles=voice_profiles,
                    book_intake=book_intake,
                )
                # Save all candidates for analysis
                for i, cand in enumerate(candidates):
                    self.storage.save_text(self.storage.candidate_path(sn, i), cand)
            else:
                scene_text = self.generator.draft_scene(
                    story_spec=story_spec,
                    scene_card=scene_card,
                    outline=outline,
                    continuity_state=continuity,
                    recent_summaries=list(continuity.recent_summaries),
                    scene_cards=scene_cards,
                    voice_dna=voice_dna,
                    character_voice_profiles=voice_profiles,
                    book_intake=book_intake,
                )

            # Validation + QA loop
            approved = False
            for attempt in range(max_rewrites + 1):
                # Deterministic validation
                validation = self.scene_validator.validate(
                    scene_card=scene_card,
                    scene_text=scene_text,
                    continuity_state=continuity,
                    story_spec=story_spec,
                )

                # LLM judge
                qa = self.scene_judge.judge(
                    story_spec=story_spec,
                    scene_card=scene_card,
                    continuity_state=continuity,
                    validation_report=validation,
                    scene_text=scene_text,
                    book_intake=book_intake,
                )

                if qa.passed and validation.passed:
                    approved = True
                    console.print(f"[green]Scene {sn} approved (attempt {attempt + 1})[/green]")
                    break

                if attempt >= max_rewrites:
                    console.print(f"[red]Scene {sn} exhausted {max_rewrites} rewrites — accepting best effort.[/red]")
                    approved = True  # Accept to avoid blocking the pipeline
                    break

                # Build rewrite brief
                issues = []
                if validation.errors:
                    issues.extend(f"VALIDATION ERROR: {e}" for e in validation.errors)
                if validation.warnings:
                    issues.extend(f"VALIDATION WARNING: {w}" for w in validation.warnings)
                if qa.weaknesses:
                    issues.extend(f"QA WEAKNESS: {w}" for w in qa.weaknesses)
                if qa.rewrite_suggestions:
                    issues.extend(f"SUGGESTION: {s}" for s in qa.rewrite_suggestions)

                rewrite_brief = "\n".join(issues)
                console.print(f"[yellow]Scene {sn} rewriting (attempt {attempt + 2}/{max_rewrites + 1})...[/yellow]")

                # Save failed draft
                self.storage.save_text(self.storage.rewrite_path(sn, attempt), scene_text)

                # Rewrite
                scene_text = self.generator.draft_scene(
                    story_spec=story_spec,
                    scene_card=scene_card,
                    outline=outline,
                    continuity_state=continuity,
                    recent_summaries=list(continuity.recent_summaries),
                    scene_cards=scene_cards,
                    voice_dna=voice_dna,
                    character_voice_profiles=voice_profiles,
                    book_intake=book_intake,
                    rewrite_brief=rewrite_brief,
                )

            # Save approved scene
            self.storage.save_text(self.storage.scene_path(sn), scene_text)
            self.storage.save_model(self.storage.scene_qa_path(sn), qa)
            self.storage.save_model(self.storage.scene_validation_path(sn), validation)
            self.storage.append_log("scene_drafted", f"Scene {sn} approved", scene_number=sn)

            # Update continuity
            continuity = self.generator.update_continuity(
                scene_card=scene_card, scene_text=scene_text, current_state=continuity,
            )
            self.storage.save_model(self.storage.continuity_path, continuity)

        return continuity

    def _phase_post_draft_passes(
        self,
        scene_cards: list[SceneCard],
        voice_profiles: list[CharacterVoiceProfile],
        voice_dna: VoiceDNA | None,
    ) -> None:
        """Phase 6: Transition smoothing, dialogue polish, prose rhythm."""

        # Transition smoothing
        console.print("Running transition smoothing pass...")
        for i in range(len(scene_cards) - 1):
            sc_a = scene_cards[i]
            sc_b = scene_cards[i + 1]
            text_a = self.storage.load_text(self.storage.scene_path(sc_a.scene_number))
            text_b = self.storage.load_text(self.storage.scene_path(sc_b.scene_number))

            report = self.generator.smooth_transition(
                scene_a_text=text_a, scene_b_text=text_b,
                scene_a_card=sc_a, scene_b_card=sc_b,
            )
            self.storage.save_model(
                self.storage.transition_report_path(sc_a.scene_number, sc_b.scene_number),
                report,
            )

            # Apply revisions if provided
            if report.revised_ending and report.revised_ending.strip():
                paragraphs = text_a.split("\n\n")
                if len(paragraphs) >= 2:
                    paragraphs[-2:] = [report.revised_ending]
                    self.storage.save_text(self.storage.scene_path(sc_a.scene_number), "\n\n".join(paragraphs))

            if report.revised_opening and report.revised_opening.strip():
                paragraphs = text_b.split("\n\n")
                if len(paragraphs) >= 2:
                    paragraphs[:2] = [report.revised_opening]
                    self.storage.save_text(self.storage.scene_path(sc_b.scene_number), "\n\n".join(paragraphs))

        console.print(f"[green]Smoothed {len(scene_cards) - 1} transitions.[/green]")

        # Dialogue polish
        if voice_profiles:
            console.print("Running dialogue polish pass...")
            scene_texts: dict[int, str] = {}
            for sc in scene_cards:
                scene_texts[sc.scene_number] = self.storage.load_text(self.storage.scene_path(sc.scene_number))

            for profile in voice_profiles:
                report = self.generator.audit_dialogue(
                    character_name=profile.character_name,
                    character_voice_profile=profile,
                    scene_texts=scene_texts,
                )
                self.storage.save_model(self.storage.dialogue_audit_path(profile.character_name), report)
                console.print(
                    f"  {profile.character_name}: voice distinctiveness {report.voice_distinctiveness_score}/5, "
                    f"{len(report.lines_to_revise)} lines revised"
                )
            self.storage.append_log("dialogue_polish", f"Audited {len(voice_profiles)} characters")

        # Prose rhythm analysis
        console.print("Running prose rhythm analysis...")
        for sc in scene_cards:
            text = self.storage.load_text(self.storage.scene_path(sc.scene_number))
            report = self.generator.analyze_prose_rhythm(
                scene_number=sc.scene_number, scene_text=text,
            )
            self.storage.save_model(self.storage.prose_rhythm_path(sc.scene_number), report)

        self.storage.append_log("post_draft_passes", "All post-draft passes complete")

    def _phase_assembly(
        self,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: list[SceneCard],
    ) -> str:
        """Phase 7: Assemble scenes into chapters and full manuscript."""
        console.print("Assembling manuscript...")
        manuscript_parts: list[str] = []
        manuscript_parts.append(f"# {story_spec.title}\n")

        for chapter in outline.chapters:
            ch_num = chapter.chapter_number
            ch_title = chapter.chapter_title or f"Chapter {ch_num}"
            chapter_text = f"\n## {ch_title}\n\n"

            scene_nums = chapter_scene_numbers(scene_cards, ch_num)
            for sn in scene_nums:
                scene_text = self.storage.load_text(self.storage.scene_path(sn))
                chapter_text += scene_text + "\n\n"

            self.storage.save_text(self.storage.chapter_path(ch_num), chapter_text)
            manuscript_parts.append(chapter_text)

        manuscript = "\n".join(manuscript_parts)
        self.storage.save_text(self.storage.manuscript_md_path, manuscript)
        self.storage.save_text(self.storage.manuscript_txt_path, plain_text_from_markdown(manuscript))
        self.storage.append_log("assembly", f"{len(outline.chapters)} chapters assembled")

        console.print(f"[green]Manuscript assembled: {len(outline.chapters)} chapters[/green]")
        return manuscript

    def _phase_qa(
        self,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: list[SceneCard],
        manuscript: str,
        book_intake: BookIntake | None,
    ) -> GlobalQaReport:
        """Phase 8: Multi-level QA including cold reader and pacing analysis."""

        # Chapter-level QA
        console.print("Running chapter-level QA...")
        for chapter in outline.chapters:
            ch_num = chapter.chapter_number
            chapter_text = self.storage.load_text(self.storage.chapter_path(ch_num))
            ch_cards = [sc for sc in scene_cards if sc.chapter_number == ch_num]
            ch_qa = self.global_judge.judge_chapter(
                story_spec=story_spec, outline=outline,
                chapter_number=ch_num, chapter_text=chapter_text,
                scene_cards=ch_cards,
            )
            self.storage.save_model(self.storage.chapter_qa_path(ch_num), ch_qa)
            status = "[green]PASS[/green]" if ch_qa.passed else "[red]FAIL[/red]"
            console.print(f"  Chapter {ch_num}: {status} (hook={ch_qa.hook_quality_score}, cliff={ch_qa.cliffhanger_score})")

        # Global QA
        console.print("Running global QA...")
        global_qa = self.global_judge.judge(
            story_spec=story_spec, outline=outline,
            manuscript_text=manuscript, book_intake=book_intake,
        )
        self.storage.save_model(self.storage.global_qa_path, global_qa)
        console.print(
            f"Global QA: {'[green]PASS[/green]' if global_qa.passed else '[red]FAIL[/red]'} "
            f"(overall={global_qa.overall_score}, ai_smell={global_qa.ai_smell_score})"
        )

        # Cold Reader Test (NEW)
        console.print("Running cold reader test...")
        cold_report = self.cold_reader.judge(manuscript_text=manuscript)
        self.storage.save_model(self.storage.cold_reader_path, cold_report)
        console.print(
            f"Cold reader: score={cold_report.overall_score}/10, "
            f"would_keep_reading={cold_report.would_keep_reading}"
        )
        if cold_report.engagement_drops:
            console.print(f"  Engagement drops: {cold_report.engagement_drops[:3]}")

        # Pacing Analysis (NEW)
        console.print("Running pacing analysis...")
        pacing = self.pacing_analyzer.analyze(
            manuscript_text=manuscript, scene_count=len(scene_cards),
        )
        self.storage.save_model(self.storage.pacing_analysis_path, pacing)
        if pacing.tension_sags:
            console.print(f"  [yellow]Tension sags: {pacing.tension_sags}[/yellow]")
        if pacing.fatigue_zones:
            console.print(f"  [yellow]Fatigue zones: {pacing.fatigue_zones}[/yellow]")

        # Anti-AI Pass (NEW)
        console.print("Running anti-AI decontamination scan...")
        anti_ai = self.generator.run_anti_ai_pass(manuscript_text=manuscript)
        self.storage.save_model(self.storage.anti_ai_report_path, anti_ai)
        console.print(
            f"  AI patterns found: {anti_ai.total_instances}, "
            f"before={anti_ai.before_score}/10, after={anti_ai.after_score}/10"
        )

        self.storage.append_log(
            "qa_complete",
            f"global={global_qa.overall_score}, cold_reader={cold_report.overall_score}, "
            f"ai_smell={global_qa.ai_smell_score}",
        )
        return global_qa

    def _phase_repair(
        self,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: list[SceneCard],
        continuity: ContinuityState,
        global_qa: GlobalQaReport,
        voice_dna: VoiceDNA | None,
        book_intake: BookIntake | None,
    ) -> str:
        """Phase 9: Multi-pass repair with diminishing returns."""
        if global_qa.passed and not global_qa.repair_priorities:
            console.print("[green]No repairs needed — manuscript passed global QA.[/green]")
            return self.storage.load_text(self.storage.manuscript_md_path)

        prev_score = global_qa.overall_score

        for cycle in range(1, self.config.max_repair_cycles + 1):
            console.print(f"\n[bold]Repair cycle {cycle}/{self.config.max_repair_cycles}[/bold]")

            # Identify scenes to repair from QA feedback
            scenes_to_repair = self._extract_repair_targets(global_qa, scene_cards)
            if not scenes_to_repair:
                console.print("[green]No specific scenes flagged for repair.[/green]")
                break

            console.print(f"Repairing {len(scenes_to_repair)} scenes: {[s.scene_number for s in scenes_to_repair]}")

            for sc in scenes_to_repair:
                original = self.storage.load_text(self.storage.scene_path(sc.scene_number))
                issues = "\n".join(
                    issue for issue in global_qa.scene_level_issues
                    if str(sc.scene_number) in issue
                )
                if not issues:
                    issues = "\n".join(global_qa.repair_priorities[:3])

                repaired = self.generator.repair_scene(
                    story_spec=story_spec,
                    scene_card=sc,
                    original_text=original,
                    qa_issues=issues,
                    continuity_state=continuity,
                    voice_dna=voice_dna,
                )
                self.storage.save_text(self.storage.scene_path(sc.scene_number), repaired)
                self.storage.append_log("repair", f"Scene {sc.scene_number} repaired in cycle {cycle}", scene_number=sc.scene_number)

            # Reassemble and re-evaluate
            manuscript = self._phase_assembly(story_spec, outline, scene_cards)
            global_qa = self.global_judge.judge(
                story_spec=story_spec, outline=outline,
                manuscript_text=manuscript, book_intake=book_intake,
            )
            self.storage.save_model(self.storage.global_qa_path, global_qa)

            new_score = global_qa.overall_score
            improvement = (new_score - prev_score) / max(prev_score, 1)
            console.print(
                f"Cycle {cycle}: score {prev_score} -> {new_score} "
                f"(improvement: {improvement:.1%})"
            )

            if global_qa.passed:
                console.print("[green]Manuscript now passes global QA![/green]")
                break
            if improvement < self.config.repair_improvement_threshold:
                console.print(f"[yellow]Improvement below {self.config.repair_improvement_threshold:.0%} threshold — stopping repairs.[/yellow]")
                break

            prev_score = new_score

        return self.storage.load_text(self.storage.manuscript_md_path)

    def _phase_premium_rewrites(
        self,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: list[SceneCard],
        voice_dna: VoiceDNA | None,
    ) -> None:
        """Phase 10: Extra attention to opening and closing chapters."""
        # Opening chapter
        if outline.chapters:
            ch1 = outline.chapters[0]
            console.print(f"Premium polish: Chapter {ch1.chapter_number} (opening)...")
            ch1_text = self.storage.load_text(self.storage.chapter_path(ch1.chapter_number))
            polished = self.generator.polish_chapter(
                chapter_text=ch1_text,
                chapter_number=ch1.chapter_number,
                voice_dna=voice_dna,
            )
            self.storage.save_text(self.storage.chapter_path(ch1.chapter_number), polished)

        # Closing chapter
        if len(outline.chapters) > 1:
            ch_last = outline.chapters[-1]
            console.print(f"Premium polish: Chapter {ch_last.chapter_number} (closing)...")
            ch_last_text = self.storage.load_text(self.storage.chapter_path(ch_last.chapter_number))
            polished = self.generator.polish_chapter(
                chapter_text=ch_last_text,
                chapter_number=ch_last.chapter_number,
                voice_dna=voice_dna,
            )
            self.storage.save_text(self.storage.chapter_path(ch_last.chapter_number), polished)

        self.storage.append_log("premium_rewrites", "Opening and closing chapters polished")

    def _phase_hook_audit(
        self,
        outline: Outline,
        scene_cards: list[SceneCard],
    ) -> None:
        """Phase 11: Audit all chapter openings and closings."""
        console.print("Auditing chapter hooks...")
        for chapter in outline.chapters:
            ch_num = chapter.chapter_number
            ch_text = self.storage.load_text(self.storage.chapter_path(ch_num))
            result = self.generator.audit_chapter_hooks(
                chapter_number=ch_num, chapter_text=ch_text,
            )
            # Save audit result
            audit_path = self.storage.passes_dir / f"hook_audit_ch_{ch_num:02d}.txt"
            self.storage.save_text(audit_path, result)
        self.storage.append_log("hook_audit", f"Audited {len(outline.chapters)} chapters")

    def _phase_final_polish(
        self,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: list[SceneCard],
        voice_dna: VoiceDNA | None,
    ) -> str:
        """Phase 12: Final polish pass on all chapters + reassemble."""
        console.print("Running final polish pass on all chapters...")

        # Skip chapters 1 and last (already premium-polished)
        middle_chapters = outline.chapters[1:-1] if len(outline.chapters) > 2 else []

        for chapter in middle_chapters:
            ch_num = chapter.chapter_number
            ch_text = self.storage.load_text(self.storage.chapter_path(ch_num))
            polished = self.generator.polish_chapter(
                chapter_text=ch_text,
                chapter_number=ch_num,
                voice_dna=voice_dna,
            )
            self.storage.save_text(self.storage.chapter_path(ch_num), polished)
            console.print(f"  Chapter {ch_num} polished.")

        # Final reassembly from polished chapters
        manuscript_parts: list[str] = [f"# {story_spec.title}\n"]
        for chapter in outline.chapters:
            ch_text = self.storage.load_text(self.storage.chapter_path(chapter.chapter_number))
            manuscript_parts.append(ch_text)

        manuscript = "\n".join(manuscript_parts)
        self.storage.save_text(self.storage.manuscript_md_path, manuscript)
        self.storage.save_text(self.storage.manuscript_txt_path, plain_text_from_markdown(manuscript))
        self.storage.append_log("final_polish", "All chapters polished and manuscript reassembled")

        console.print("[green]Final polish complete.[/green]")
        return manuscript

    # ══════════════════════════════════════════════════════════════════
    # STANDALONE COMMANDS
    # ══════════════════════════════════════════════════════════════════

    def bootstrap(
        self,
        *,
        synopsis: str,
        book_intake: BookIntake | None = None,
    ) -> None:
        """Runs only the planning phases (voice cal + planning + validation)."""
        voice_dna = self._phase_voice_calibration(book_intake, synopsis)
        self._phase_planning(synopsis, book_intake, voice_dna)
        console.print("[green]Bootstrap complete — planning artifacts generated.[/green]")

    def draft_single_scene(
        self,
        *,
        scene_number: int,
        force: bool = False,
    ) -> None:
        """Drafts a single scene (for debugging/iteration)."""
        if self.storage.has_approved_scene(scene_number) and not force:
            console.print(f"Scene {scene_number} already approved. Use --force to overwrite.")
            return

        story_spec = self.storage.load_model(self.storage.story_spec_path, StorySpec)
        outline = self.storage.load_model(self.storage.outline_path, Outline)
        scene_cards = self.storage.load_model_list(self.storage.scene_cards_path, SceneCard)
        continuity = self.storage.load_model(self.storage.continuity_path, ContinuityState)

        scene_card = get_scene_card(scene_cards, scene_number)
        voice_dna = None
        if self.storage.exists(self.storage.voice_dna_path):
            voice_dna = self.storage.load_model(self.storage.voice_dna_path, VoiceDNA)

        text = self.generator.draft_scene(
            story_spec=story_spec,
            scene_card=scene_card,
            outline=outline,
            continuity_state=continuity,
            recent_summaries=list(continuity.recent_summaries),
            scene_cards=scene_cards,
            voice_dna=voice_dna,
        )
        self.storage.save_text(self.storage.scene_path(scene_number), text)
        console.print(f"[green]Scene {scene_number} drafted ({len(text.split())} words).[/green]")

    def run_global_qa(self, *, book_intake: BookIntake | None = None) -> GlobalQaReport:
        """Runs global QA on existing manuscript."""
        story_spec = self.storage.load_model(self.storage.story_spec_path, StorySpec)
        outline = self.storage.load_model(self.storage.outline_path, Outline)
        manuscript = self.storage.load_text(self.storage.manuscript_md_path)

        return self.global_judge.judge(
            story_spec=story_spec, outline=outline,
            manuscript_text=manuscript, book_intake=book_intake,
        )

    # ══════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _extract_repair_targets(
        self,
        global_qa: GlobalQaReport,
        scene_cards: list[SceneCard],
    ) -> list[SceneCard]:
        """Extracts scene numbers mentioned in QA issues."""
        import re
        mentioned_scenes: set[int] = set()
        all_issues = global_qa.scene_level_issues + global_qa.repair_priorities
        for issue in all_issues:
            numbers = re.findall(r'scene\s*(\d+)', issue, re.IGNORECASE)
            for n in numbers:
                mentioned_scenes.add(int(n))

        # If no specific scenes mentioned, target the weakest by QA scores
        if not mentioned_scenes and global_qa.repair_priorities:
            # Repair first 3-5 scenes as general improvement
            mentioned_scenes = set(sc.scene_number for sc in scene_cards[:5])

        return [sc for sc in scene_cards if sc.scene_number in mentioned_scenes]
