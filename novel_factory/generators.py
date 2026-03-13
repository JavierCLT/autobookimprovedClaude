"""Content generators: planning, voice calibration, drafting (best-of-N), post-draft passes."""

from __future__ import annotations

import json
import logging
import re
from typing import Sequence

from novel_factory.config import AppConfig
from novel_factory.intake import build_drafting_guidance, build_planning_guidance
from novel_factory.llm import AnthropicClient
from novel_factory.prompts import (
    anti_ai_system_prompt,
    anti_ai_user_prompt,
    beat_sheet_system_prompt,
    beat_sheet_user_prompt,
    best_of_n_selection_system_prompt,
    best_of_n_selection_user_prompt,
    character_voice_profile_prompt,
    chapter_hook_audit_system_prompt,
    chapter_hook_audit_user_prompt,
    continuity_system_prompt,
    continuity_update_user_prompt,
    dialogue_polish_system_prompt,
    dialogue_polish_user_prompt,
    editorial_blueprint_system_prompt,
    editorial_blueprint_user_prompt,
    final_polish_system_prompt,
    final_polish_user_prompt,
    initial_continuity_user_prompt,
    outline_user_prompt,
    planning_system_prompt,
    plant_payoff_system_prompt,
    plant_payoff_user_prompt,
    prose_rhythm_system_prompt,
    prose_rhythm_user_prompt,
    scene_cards_user_prompt,
    scene_draft_system_prompt,
    scene_draft_user_prompt,
    story_spec_user_prompt,
    subplot_weave_system_prompt,
    subplot_weave_user_prompt,
    transition_smoothing_system_prompt,
    transition_smoothing_user_prompt,
    voice_calibration_system_prompt,
    voice_calibration_user_prompt,
)
from novel_factory.schemas import (
    AntiAiPassReport,
    BeatSheet,
    BookIntake,
    CharacterVoiceProfile,
    ContinuityState,
    ContinuityUpdate,
    DialogueAuditReport,
    EditorialBlueprint,
    Outline,
    PlantPayoffMap,
    ProseRhythmReport,
    SceneCard,
    StorySpec,
    SubplotWeaveMap,
    TransitionReport,
    VoiceDNA,
)
from novel_factory.utils import (
    count_words,
    extract_dialogue_lines,
    get_chapter_plan,
    serialise_model,
    truncate_text,
)

logger = logging.getLogger(__name__)


class NovelGenerator:
    """Orchestrates all LLM-powered generation steps."""

    def __init__(self, llm: AnthropicClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    # ══════════════════════════════════════════════════════════════════
    # PRE-PLANNING: Voice Calibration (NEW)
    # ══════════════════════════════════════════════════════════════════

    def calibrate_voice(
        self,
        *,
        reference_passages: str,
        genre: str,
        audience: str,
    ) -> VoiceDNA:
        """Extracts voice DNA from reference passages."""
        return self.llm.structured(
            system_prompt=voice_calibration_system_prompt(),
            user_prompt=voice_calibration_user_prompt(
                reference_passages=reference_passages,
                genre=genre,
                audience=audience,
            ),
            schema=VoiceDNA,
            task_name="voice_calibration",
            reasoning_effort=self.config.reasoning.voice_calibration,
            temperature=0.3,
            max_output_tokens=3_000,
        )

    def generate_character_voice_profile(
        self,
        *,
        character_name: str,
        character_role: str,
        story_context: str,
        voice_dna: VoiceDNA,
    ) -> CharacterVoiceProfile:
        """Generates a detailed voice profile for a specific character."""
        return self.llm.structured(
            system_prompt=voice_calibration_system_prompt(),
            user_prompt=character_voice_profile_prompt(
                character=character_name,
                role=character_role,
                story_context=story_context,
                voice_dna=serialise_model(voice_dna),
            ),
            schema=CharacterVoiceProfile,
            task_name=f"voice_profile_{character_name.lower().replace(' ', '_')}",
            reasoning_effort=self.config.reasoning.voice_calibration,
            temperature=0.4,
            max_output_tokens=2_500,
        )

    # ══════════════════════════════════════════════════════════════════
    # PLANNING
    # ══════════════════════════════════════════════════════════════════

    def generate_story_spec(
        self,
        *,
        synopsis: str,
        audience: str,
        rating_ceiling: str,
        market_position: str,
        target_words: int,
        target_chapters: int,
        target_scenes: int,
        book_intake: BookIntake | None = None,
        voice_dna: VoiceDNA | None = None,
    ) -> StorySpec:
        voice_summary = ""
        if voice_dna:
            voice_summary = (
                f"Register: {voice_dna.vocabulary_register}\n"
                f"Rhythm: {voice_dna.rhythm_signature}\n"
                f"Techniques: {', '.join(voice_dna.characteristic_techniques)}\n"
                f"Avoid: {', '.join(voice_dna.avoid_patterns)}"
            )
        return self.llm.structured(
            system_prompt=planning_system_prompt(),
            user_prompt=story_spec_user_prompt(
                synopsis=synopsis,
                audience=audience,
                rating_ceiling=rating_ceiling,
                market_position=market_position,
                target_words=target_words,
                target_chapters=target_chapters,
                target_scenes=target_scenes,
                intake_guidance=build_planning_guidance(book_intake),
                voice_dna_summary=voice_summary,
            ),
            schema=StorySpec,
            task_name="story_spec",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=8_000,
        )

    def generate_beat_sheet(
        self,
        *,
        story_spec: StorySpec,
        framework: str = "Save the Cat",
    ) -> BeatSheet:
        """Generates a structural beat sheet. (NEW)"""
        return self.llm.structured(
            system_prompt=beat_sheet_system_prompt(),
            user_prompt=beat_sheet_user_prompt(
                story_spec_json=serialise_model(story_spec),
                framework=framework,
            ),
            schema=BeatSheet,
            task_name="beat_sheet",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=4_000,
        )

    def generate_outline(
        self,
        *,
        story_spec: StorySpec,
        book_intake: BookIntake | None = None,
        beat_sheet: BeatSheet | None = None,
    ) -> Outline:
        beat_json = serialise_model(beat_sheet) if beat_sheet else ""
        return self.llm.structured(
            system_prompt=planning_system_prompt(),
            user_prompt=outline_user_prompt(
                story_spec_json=serialise_model(story_spec),
                intake_guidance=build_planning_guidance(book_intake),
                beat_sheet_json=beat_json,
            ),
            schema=Outline,
            task_name="outline",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=12_000,
        )

    def generate_plant_payoff_map(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
    ) -> PlantPayoffMap:
        """Generates a foreshadowing plant/payoff registry. (NEW)"""
        return self.llm.structured(
            system_prompt=plant_payoff_system_prompt(),
            user_prompt=plant_payoff_user_prompt(
                story_spec_json=serialise_model(story_spec),
                outline_json=serialise_model(outline),
            ),
            schema=PlantPayoffMap,
            task_name="plant_payoff_map",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=5_000,
        )

    def generate_subplot_weave(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
    ) -> SubplotWeaveMap:
        """Generates a subplot weave map. (NEW)"""
        return self.llm.structured(
            system_prompt=subplot_weave_system_prompt(),
            user_prompt=subplot_weave_user_prompt(
                story_spec_json=serialise_model(story_spec),
                outline_json=serialise_model(outline),
            ),
            schema=SubplotWeaveMap,
            task_name="subplot_weave",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=4_000,
        )

    def generate_scene_cards(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        book_intake: BookIntake | None = None,
        plant_payoff_map: PlantPayoffMap | None = None,
        subplot_weave: SubplotWeaveMap | None = None,
    ) -> list[SceneCard]:
        plant_json = serialise_model(plant_payoff_map) if plant_payoff_map else ""
        subplot_json = serialise_model(subplot_weave) if subplot_weave else ""
        raw_cards = self.llm.structured(
            system_prompt=planning_system_prompt(),
            user_prompt=scene_cards_user_prompt(
                story_spec_json=serialise_model(story_spec),
                outline_json=serialise_model(outline),
                plant_payoff_json=plant_json,
                subplot_weave_json=subplot_json,
                intake_guidance=build_planning_guidance(book_intake),
            ),
            schema=list[SceneCard],
            task_name="scene_cards",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=24_000,
        )
        return self._normalize_scene_cards(raw_cards)

    def generate_initial_continuity(
        self,
        *,
        story_spec: StorySpec,
        scene_cards: Sequence[SceneCard],
    ) -> ContinuityState:
        return self.llm.structured(
            system_prompt=continuity_system_prompt(),
            user_prompt=initial_continuity_user_prompt(
                story_spec_json=serialise_model(story_spec),
                scene_cards_json=serialise_model(list(scene_cards)),
            ),
            schema=ContinuityState,
            task_name="initial_continuity",
            reasoning_effort=self.config.reasoning.planning,
            temperature=0.1,
            max_output_tokens=3_000,
        )

    def generate_editorial_blueprint(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: Sequence[SceneCard],
    ) -> EditorialBlueprint:
        """Generates the editorial blueprint with escalation ladders and chapter missions."""
        return self.llm.structured(
            system_prompt=editorial_blueprint_system_prompt(),
            user_prompt=editorial_blueprint_user_prompt(
                story_spec_json=serialise_model(story_spec),
                outline_json=serialise_model(outline),
                scene_cards_json=serialise_model(list(scene_cards)),
            ),
            schema=EditorialBlueprint,
            task_name="editorial_blueprint",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=12_000,
        )

    # ══════════════════════════════════════════════════════════════════
    # DRAFTING (with best-of-N)
    # ══════════════════════════════════════════════════════════════════

    def draft_scene(
        self,
        *,
        story_spec: StorySpec,
        scene_card: SceneCard,
        outline: Outline,
        continuity_state: ContinuityState,
        recent_summaries: list[str],
        scene_cards: Sequence[SceneCard],
        voice_dna: VoiceDNA | None = None,
        character_voice_profiles: list[CharacterVoiceProfile] | None = None,
        book_intake: BookIntake | None = None,
        rewrite_brief: str = "",
    ) -> str:
        """Drafts a single scene with full context."""
        chapter_plan = get_chapter_plan(outline, scene_card.chapter_number)

        # Build voice context
        voice_summary = ""
        if voice_dna:
            voice_summary = (
                f"Register: {voice_dna.vocabulary_register} | "
                f"Rhythm: {voice_dna.rhythm_signature}\n"
                f"Techniques: {', '.join(voice_dna.characteristic_techniques[:5])}\n"
                f"Avoid: {', '.join(voice_dna.avoid_patterns[:5])}"
            )

        char_profiles_text = ""
        if character_voice_profiles:
            pov_profiles = [p for p in character_voice_profiles if p.character_name == scene_card.pov_character]
            other_profiles = [p for p in character_voice_profiles if p.character_name != scene_card.pov_character]
            relevant = pov_profiles + other_profiles[:3]
            for p in relevant:
                char_profiles_text += (
                    f"\n{p.character_name} ({p.vocabulary_range}): "
                    f"Speech: {', '.join(p.speech_patterns[:3])}. "
                    f"Tics: {', '.join(p.verbal_tics[:3])}. "
                    f"Style: {p.sentence_style}"
                )

        # Build lookahead context (NEW)
        lookahead = ""
        current_idx = None
        for i, sc in enumerate(scene_cards):
            if sc.scene_number == scene_card.scene_number:
                current_idx = i
                break
        if current_idx is not None:
            future_cards = list(scene_cards)[current_idx + 1 : current_idx + 1 + self.config.lookahead_scenes]
            if future_cards:
                lookahead_parts = []
                for fc in future_cards:
                    lookahead_parts.append(
                        f"Scene {fc.scene_number}: {fc.dramatic_purpose} "
                        f"(opening: {fc.opening_disturbance})"
                    )
                lookahead = "\n".join(lookahead_parts)

        # Build recent summaries text
        summaries_text = "\n".join(
            f"Scene {continuity_state.after_scene - len(recent_summaries) + i + 1}: {s}"
            for i, s in enumerate(recent_summaries)
        ) if recent_summaries else "(First scene — no prior context)"

        # Adaptive temperature based on scene type
        temperature = self.config.get_scene_temperature(scene_card.scene_type)

        return self.llm.text(
            system_prompt=scene_draft_system_prompt(story_spec),
            user_prompt=scene_draft_user_prompt(
                story_spec=story_spec,
                scene_card=scene_card,
                chapter_plan_json=serialise_model(chapter_plan),
                continuity_state_json=serialise_model(continuity_state),
                recent_summaries=summaries_text,
                voice_dna_summary=voice_summary,
                character_voice_profiles=char_profiles_text,
                lookahead_cards=lookahead,
                intake_guidance=build_drafting_guidance(book_intake),
                rewrite_brief=rewrite_brief,
            ),
            task_name=f"draft_scene_{scene_card.scene_number:02d}",
            reasoning_effort=self.config.reasoning.drafting if not rewrite_brief else self.config.reasoning.rewriting,
            temperature=temperature,
            max_output_tokens=7_000,
            model_override=self.config.get_drafting_model(),
        )

    def draft_scene_best_of_n(
        self,
        *,
        n: int,
        story_spec: StorySpec,
        scene_card: SceneCard,
        outline: Outline,
        continuity_state: ContinuityState,
        recent_summaries: list[str],
        scene_cards: Sequence[SceneCard],
        voice_dna: VoiceDNA | None = None,
        character_voice_profiles: list[CharacterVoiceProfile] | None = None,
        book_intake: BookIntake | None = None,
    ) -> tuple[str, list[str]]:
        """Generates N candidate drafts and selects the best one. (NEW)

        Returns (winning_text, all_candidate_texts).
        """
        candidates: list[str] = []
        for i in range(n):
            logger.info("Generating candidate %d/%d for scene %d", i + 1, n, scene_card.scene_number)
            text = self.draft_scene(
                story_spec=story_spec,
                scene_card=scene_card,
                outline=outline,
                continuity_state=continuity_state,
                recent_summaries=recent_summaries,
                scene_cards=scene_cards,
                voice_dna=voice_dna,
                character_voice_profiles=character_voice_profiles,
                book_intake=book_intake,
            )
            candidates.append(text)

        if n == 1:
            return candidates[0], candidates

        # Select the best candidate via LLM judge
        labeled = [(f"CANDIDATE_{chr(65 + i)}", text) for i, text in enumerate(candidates)]
        selection = self.llm.text(
            system_prompt=best_of_n_selection_system_prompt(),
            user_prompt=best_of_n_selection_user_prompt(
                scene_card_json=serialise_model(scene_card),
                candidates=labeled,
            ),
            task_name=f"best_of_n_select_{scene_card.scene_number:02d}",
            reasoning_effort="high",
            temperature=0.1,
            max_output_tokens=500,
            model_override=self.config.get_qa_model(),
        )

        # Parse the winner
        winner_idx = 0
        for i, (label, _) in enumerate(labeled):
            if label in selection:
                winner_idx = i
                break

        logger.info(
            "Scene %d: selected %s from %d candidates",
            scene_card.scene_number, labeled[winner_idx][0], n,
        )
        return candidates[winner_idx], candidates

    # ══════════════════════════════════════════════════════════════════
    # CONTINUITY TRACKING
    # ══════════════════════════════════════════════════════════════════

    def update_continuity(
        self,
        *,
        scene_card: SceneCard,
        scene_text: str,
        current_state: ContinuityState,
    ) -> ContinuityState:
        """Updates continuity state after a scene is approved."""
        update = self.llm.structured(
            system_prompt=continuity_system_prompt(),
            user_prompt=continuity_update_user_prompt(
                scene_card_json=serialise_model(scene_card),
                scene_text=scene_text,
                current_state_json=serialise_model(current_state),
            ),
            schema=ContinuityUpdate,
            task_name=f"continuity_update_{scene_card.scene_number:02d}",
            reasoning_effort=self.config.reasoning.planning,
            temperature=0.1,
            max_output_tokens=2_500,
        )
        return self._apply_continuity_update(current_state, update)

    def _apply_continuity_update(
        self,
        state: ContinuityState,
        update: ContinuityUpdate,
    ) -> ContinuityState:
        """Merges a continuity update into the current state."""
        new_facts = [f for f in state.known_facts if f not in update.facts_to_remove]
        new_facts.extend(update.facts_to_add)

        new_threads = [t for t in state.open_threads if t not in update.threads_closed]
        new_threads.extend(update.threads_opened)

        new_relationships = list(state.relationship_states)
        new_relationships.extend(update.relationship_updates)

        new_suspicion = list(state.suspicion_levels)
        new_suspicion.extend(update.suspicion_updates)

        new_evidence = list(state.evidence_items)
        new_evidence.extend(update.evidence_updates)

        new_moral = list(state.moral_lines_crossed)
        new_moral.extend(update.moral_lines_crossed)

        new_locations = dict(state.character_locations)
        new_locations.update(update.location_updates)

        new_knowledge = dict(state.character_knowledge)
        for char, items in update.knowledge_updates.items():
            existing = new_knowledge.get(char, [])
            new_knowledge[char] = existing + items

        new_emotional = dict(state.emotional_states)
        new_emotional.update(update.emotional_updates)

        new_promises = [p for p in state.active_promises if p not in update.promises_fulfilled]
        new_promises.extend(update.promises_made)

        # Rolling summaries
        new_summaries = list(state.recent_summaries)
        if update.summary:
            new_summaries.append(
                truncate_text(update.summary, self.config.max_recent_scene_summary_chars // self.config.recent_scene_summaries)
            )
        if len(new_summaries) > self.config.recent_scene_summaries:
            new_summaries = new_summaries[-self.config.recent_scene_summaries:]

        # Cap lists to prevent unbounded growth
        return ContinuityState(
            after_scene=update.scene_number,
            character_locations=new_locations,
            known_facts=new_facts[-50:],
            open_threads=new_threads[-30:],
            relationship_states=new_relationships[-30:],
            suspicion_levels=new_suspicion[-20:],
            evidence_items=new_evidence[-30:],
            moral_lines_crossed=new_moral[-20:],
            recent_summaries=new_summaries,
            character_knowledge=new_knowledge,
            emotional_states=new_emotional,
            active_promises=new_promises[-20:],
        )

    # ══════════════════════════════════════════════════════════════════
    # POST-DRAFT PASSES (ALL NEW)
    # ══════════════════════════════════════════════════════════════════

    def smooth_transition(
        self,
        *,
        scene_a_text: str,
        scene_b_text: str,
        scene_a_card: SceneCard,
        scene_b_card: SceneCard,
    ) -> TransitionReport:
        """Smooths the transition between consecutive scenes."""
        # Extract endings and openings
        a_paragraphs = scene_a_text.split("\n\n")
        b_paragraphs = scene_b_text.split("\n\n")
        scene_a_ending = "\n\n".join(a_paragraphs[-3:]) if len(a_paragraphs) >= 3 else scene_a_text[-800:]
        scene_b_opening = "\n\n".join(b_paragraphs[:3]) if len(b_paragraphs) >= 3 else scene_b_text[:800:]

        return self.llm.structured(
            system_prompt=transition_smoothing_system_prompt(),
            user_prompt=transition_smoothing_user_prompt(
                scene_a_ending=scene_a_ending,
                scene_b_opening=scene_b_opening,
                scene_a_card_json=serialise_model(scene_a_card),
                scene_b_card_json=serialise_model(scene_b_card),
            ),
            schema=TransitionReport,
            task_name=f"transition_{scene_a_card.scene_number:02d}_to_{scene_b_card.scene_number:02d}",
            reasoning_effort=self.config.reasoning.polish,
            temperature=self.config.polish_temperature,
            max_output_tokens=2_500,
        )

    def audit_dialogue(
        self,
        *,
        character_name: str,
        character_voice_profile: CharacterVoiceProfile,
        scene_texts: dict[int, str],  # scene_number -> text
    ) -> DialogueAuditReport:
        """Audits all dialogue for a character across the manuscript. (NEW)"""
        all_lines: list[str] = []
        all_contexts: list[str] = []
        for scene_num in sorted(scene_texts.keys()):
            text = scene_texts[scene_num]
            lines = extract_dialogue_lines(text)
            # Simple attribution: lines near character name mentions
            for line in lines:
                line_idx = text.find(line)
                context_window = text[max(0, line_idx - 200) : line_idx + len(line) + 200]
                if character_name.lower() in context_window.lower():
                    all_lines.append(line)
                    all_contexts.append(f"Scene {scene_num}")

        if not all_lines:
            return DialogueAuditReport(
                character_name=character_name,
                voice_distinctiveness_score=3,
                issues=["No dialogue found for this character"],
                lines_to_revise=[],
                revised_lines=[],
            )

        # Limit to avoid token overflow
        max_lines = 40
        sampled_lines = all_lines[:max_lines]
        sampled_contexts = all_contexts[:max_lines]

        return self.llm.structured(
            system_prompt=dialogue_polish_system_prompt(),
            user_prompt=dialogue_polish_user_prompt(
                character_name=character_name,
                character_voice_profile=serialise_model(character_voice_profile),
                dialogue_lines=sampled_lines,
                scene_contexts=sampled_contexts,
            ),
            schema=DialogueAuditReport,
            task_name=f"dialogue_audit_{character_name.lower().replace(' ', '_')}",
            reasoning_effort=self.config.reasoning.polish,
            temperature=self.config.polish_temperature,
            max_output_tokens=4_000,
        )

    def run_anti_ai_pass(self, *, manuscript_text: str) -> AntiAiPassReport:
        """Scans manuscript for AI-tell patterns and generates fixes. (NEW)"""
        # Truncate if too long for a single call
        text = truncate_text(manuscript_text, 80_000)
        return self.llm.structured(
            system_prompt=anti_ai_system_prompt(),
            user_prompt=anti_ai_user_prompt(manuscript_text=text),
            schema=AntiAiPassReport,
            task_name="anti_ai_pass",
            reasoning_effort=self.config.reasoning.polish,
            temperature=0.3,
            max_output_tokens=8_000,
        )

    def analyze_prose_rhythm(
        self,
        *,
        scene_number: int,
        scene_text: str,
    ) -> ProseRhythmReport:
        """Analyzes prose rhythm for a single scene. (NEW)"""
        return self.llm.structured(
            system_prompt=prose_rhythm_system_prompt(),
            user_prompt=prose_rhythm_user_prompt(
                scene_number=scene_number,
                scene_text=scene_text,
            ),
            schema=ProseRhythmReport,
            task_name=f"prose_rhythm_{scene_number:02d}",
            reasoning_effort=self.config.reasoning.polish,
            temperature=self.config.polish_temperature,
            max_output_tokens=3_000,
        )

    def polish_chapter(
        self,
        *,
        chapter_text: str,
        chapter_number: int,
        voice_dna: VoiceDNA | None = None,
    ) -> str:
        """Final polish pass on a single chapter. (NEW)"""
        voice_summary = ""
        if voice_dna:
            voice_summary = (
                f"Register: {voice_dna.vocabulary_register}\n"
                f"Rhythm: {voice_dna.rhythm_signature}\n"
                f"Techniques: {', '.join(voice_dna.characteristic_techniques)}\n"
                f"Avoid: {', '.join(voice_dna.avoid_patterns)}"
            )
        return self.llm.text(
            system_prompt=final_polish_system_prompt(),
            user_prompt=final_polish_user_prompt(
                chapter_text=chapter_text,
                chapter_number=chapter_number,
                voice_dna_summary=voice_summary,
            ),
            task_name=f"final_polish_ch_{chapter_number:02d}",
            reasoning_effort=self.config.reasoning.polish,
            temperature=self.config.polish_temperature,
            max_output_tokens=12_000,
            model_override=self.config.get_drafting_model(),
        )

    def audit_chapter_hooks(
        self,
        *,
        chapter_number: int,
        chapter_text: str,
    ) -> str:
        """Audits and optionally rewrites chapter opening/closing hooks. (NEW)"""
        paragraphs = chapter_text.split("\n\n")
        opening = "\n\n".join(paragraphs[:3]) if len(paragraphs) >= 3 else chapter_text[:800]
        closing = "\n\n".join(paragraphs[-3:]) if len(paragraphs) >= 3 else chapter_text[-800:]

        return self.llm.text(
            system_prompt=chapter_hook_audit_system_prompt(),
            user_prompt=chapter_hook_audit_user_prompt(
                chapter_number=chapter_number,
                chapter_opening=opening,
                chapter_closing=closing,
            ),
            task_name=f"hook_audit_ch_{chapter_number:02d}",
            reasoning_effort=self.config.reasoning.polish,
            temperature=self.config.polish_temperature,
            max_output_tokens=3_000,
        )

    # ══════════════════════════════════════════════════════════════════
    # REPAIR
    # ══════════════════════════════════════════════════════════════════

    def repair_scene(
        self,
        *,
        story_spec: StorySpec,
        scene_card: SceneCard,
        original_text: str,
        qa_issues: str,
        continuity_state: ContinuityState,
        voice_dna: VoiceDNA | None = None,
    ) -> str:
        """Targeted rewrite of a scene based on QA feedback."""
        from novel_factory.prompts import repair_scene_system_prompt, repair_scene_user_prompt

        voice_summary = ""
        if voice_dna:
            voice_summary = (
                f"Register: {voice_dna.vocabulary_register}\n"
                f"Rhythm: {voice_dna.rhythm_signature}\n"
                f"Techniques: {', '.join(voice_dna.characteristic_techniques)}\n"
                f"Avoid: {', '.join(voice_dna.avoid_patterns)}"
            )

        return self.llm.text(
            system_prompt=repair_scene_system_prompt(story_spec),
            user_prompt=repair_scene_user_prompt(
                scene_card_json=serialise_model(scene_card),
                original_text=original_text,
                qa_issues=qa_issues,
                continuity_state_json=serialise_model(continuity_state),
                voice_dna_summary=voice_summary,
            ),
            task_name=f"repair_scene_{scene_card.scene_number:02d}",
            reasoning_effort=self.config.reasoning.repair,
            temperature=self.config.rewriting_temperature,
            max_output_tokens=7_000,
            model_override=self.config.get_drafting_model(),
        )

    # ══════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _normalize_scene_cards(self, cards: list[SceneCard]) -> list[SceneCard]:
        """Ensures sequential numbering and reasonable word targets."""
        cards.sort(key=lambda c: (c.chapter_number, c.scene_number))
        normalized = []
        for i, card in enumerate(cards, start=1):
            card.scene_number = i
            card.word_target = max(900, min(2200, card.word_target))
            if not card.scene_type:
                card.scene_type = "general"
            normalized.append(card)
        return normalized
