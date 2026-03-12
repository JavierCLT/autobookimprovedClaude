"""Pydantic schemas for all pipeline artifacts — expanded for contest-grade output."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Voice & Style Calibration (NEW)
# ---------------------------------------------------------------------------

class VoiceDNA(ArtifactModel):
    """Extracted voice fingerprint from reference passages."""
    avg_sentence_length: float = Field(description="Average words per sentence")
    sentence_length_variance: float = Field(description="Std dev of sentence lengths")
    dialogue_to_narration_ratio: float = Field(description="Ratio of dialogue words to total")
    sensory_density: float = Field(description="Sensory words per 100 words")
    vocabulary_register: str = Field(description="e.g. 'literary-accessible', 'sparse-hardboiled'")
    rhythm_signature: str = Field(description="Description of prose rhythm patterns")
    characteristic_techniques: list[str] = Field(description="Key stylistic techniques observed")
    avoid_patterns: list[str] = Field(description="Patterns to avoid from reference analysis")
    sample_paragraph: str = Field(description="A generated exemplar paragraph in the target voice")


class CharacterVoiceProfile(ArtifactModel):
    """Speech and thought patterns for a specific character."""
    character_name: str
    education_level: str
    vocabulary_range: str = Field(description="e.g. 'technical-precise', 'colloquial-warm'")
    speech_patterns: list[str] = Field(description="Distinctive verbal habits")
    verbal_tics: list[str] = Field(description="Filler words, catchphrases, habits")
    sentence_style: str = Field(description="e.g. 'clipped fragments', 'long subordinate clauses'")
    topics_they_gravitate_toward: list[str]
    topics_they_avoid: list[str]
    emotional_expression_style: str = Field(description="How they show vs hide emotion")
    internal_monologue_style: str = Field(description="How their thoughts read on page")
    sample_dialogue: str = Field(description="3-5 lines of characteristic dialogue")


# ---------------------------------------------------------------------------
# Planning Models (enhanced)
# ---------------------------------------------------------------------------

class StyleGuide(ArtifactModel):
    prose_traits: list[str]
    banned_language_patterns: list[str]
    dialogue_rules: list[str]
    narration_rules: list[str]
    sensory_preferences: list[str]
    anti_ai_rules: list[str] = Field(default_factory=list, description="Specific AI-smell patterns to avoid")


class CharacterCard(ArtifactModel):
    name: str
    role: str
    age: str = ""
    public_persona: str
    private_need: str
    fear: str
    contradiction: str
    external_goal: str = ""
    inner_wound: str = ""
    secret_pressure: str = ""
    what_they_hide: str = ""
    relationships: list[str] = Field(default_factory=list)


class StorySpec(ArtifactModel):
    title: str
    genre: str
    subgenre: str = ""
    audience: str
    rating_ceiling: str = "R"
    pov: str
    tense: str = "past"
    themes: list[str]
    setting: str
    timeline_window: str = ""
    hook: str
    midpoint: str
    climax: str
    resolution: str
    protagonist: CharacterCard
    antagonist: CharacterCard
    cast: list[CharacterCard] = Field(default_factory=list)
    style_guide: StyleGuide
    emotional_engine: str = ""
    adversarial_engine: str = ""
    moral_fault_line: str = ""
    target_word_count: int = 40_000
    target_chapters: int = 14
    target_scenes: int = 28


# ---------------------------------------------------------------------------
# Plot Structure (NEW)
# ---------------------------------------------------------------------------

class BeatSheetEntry(ArtifactModel):
    """A single beat in a structural beat sheet."""
    beat_name: str = Field(description="e.g. 'Opening Image', 'Catalyst', 'Midpoint'")
    beat_description: str
    target_percentage: float = Field(description="Where in the story this beat should land (0-100)")
    mapped_scene_numbers: list[int] = Field(default_factory=list)
    notes: str = ""


class BeatSheet(ArtifactModel):
    """Structural beat sheet mapping story to proven framework."""
    framework: str = Field(description="e.g. 'Save the Cat', 'Story Grid', 'Three Act'")
    beats: list[BeatSheetEntry]
    structural_notes: str = ""


class PlantPayoff(ArtifactModel):
    """A single plant-payoff pair for foreshadowing."""
    element: str = Field(description="What is being planted/paid off")
    plant_scene: int = Field(description="Scene number where the seed is planted")
    payoff_scene: int = Field(description="Scene number where it pays off")
    subtlety_level: str = Field(description="'subtle', 'moderate', 'overt'")
    plant_method: str = Field(description="How to naturally embed the plant")
    payoff_method: str = Field(description="How the payoff manifests")


class PlantPayoffMap(ArtifactModel):
    """Registry of all foreshadowing plants and their payoffs."""
    entries: list[PlantPayoff]


class SubplotArc(ArtifactModel):
    """A single subplot tracked across the manuscript."""
    subplot_name: str
    subplot_type: str = Field(description="e.g. 'romantic', 'professional', 'internal', 'mystery'")
    description: str
    scene_appearances: list[int] = Field(description="Scene numbers where this subplot advances")
    arc_shape: str = Field(description="Brief description of the subplot's trajectory")
    intersection_with_main_plot: str = Field(description="How/where it connects to the main story")


class SubplotWeaveMap(ArtifactModel):
    """Map of all subplots and their threading through the story."""
    subplots: list[SubplotArc]


# ---------------------------------------------------------------------------
# Outline & Scene Cards (enhanced)
# ---------------------------------------------------------------------------

class ChapterSceneBeat(ArtifactModel):
    slot_number: int
    scene_type: str = Field(default="", description="action/dialogue/introspective/confrontation/revelation/quiet/climax")
    dramatic_purpose: str = ""
    word_target: int = 1500


class ChapterPlan(ArtifactModel):
    chapter_number: int
    chapter_title: str = ""
    dramatic_purpose: str = ""
    opening_hook: str = ""
    closing_hook: str = ""
    scenes: list[ChapterSceneBeat] = Field(default_factory=list)


class Outline(ArtifactModel):
    chapters: list[ChapterPlan]
    total_planned_scenes: int = 0


class SceneCard(ArtifactModel):
    scene_number: int
    chapter_number: int
    scene_type: str = Field(default="", description="action/dialogue/introspective/confrontation/revelation/quiet/climax")
    pov_character: str = ""
    location: str = ""
    time_of_day: str = ""
    dramatic_purpose: str = ""
    opening_disturbance: str = ""
    mid_scene_reversal: str = ""
    closing_choice: str = ""
    power_shift: str = ""
    suspicion_delta: str = ""
    emotional_arc: str = ""
    sensory_anchor: str = ""
    counterforce_trace: str = ""
    continuity_inputs: list[str] = Field(default_factory=list)
    continuity_outputs: list[str] = Field(default_factory=list)
    required_entities: list[str] = Field(default_factory=list)
    forbidden_entities: list[str] = Field(default_factory=list)
    plants_in_this_scene: list[str] = Field(default_factory=list, description="Foreshadowing elements to plant here")
    payoffs_in_this_scene: list[str] = Field(default_factory=list, description="Earlier plants that pay off here")
    word_target: int = 1500
    notes: str = ""


# ---------------------------------------------------------------------------
# Continuity Tracking
# ---------------------------------------------------------------------------

class ContinuityState(ArtifactModel):
    after_scene: int = 0
    character_locations: dict[str, str] = Field(default_factory=dict)
    known_facts: list[str] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)
    relationship_states: list[str] = Field(default_factory=list)
    suspicion_levels: list[str] = Field(default_factory=list)
    evidence_items: list[str] = Field(default_factory=list)
    moral_lines_crossed: list[str] = Field(default_factory=list)
    recent_summaries: list[str] = Field(default_factory=list)
    character_knowledge: dict[str, list[str]] = Field(
        default_factory=dict,
        description="NEW: tracks what each character knows and doesn't know"
    )
    emotional_states: dict[str, str] = Field(
        default_factory=dict,
        description="NEW: current emotional state per character"
    )
    active_promises: list[str] = Field(
        default_factory=list,
        description="NEW: narrative promises made to the reader"
    )


class ContinuityUpdate(ArtifactModel):
    scene_number: int
    facts_to_add: list[str] = Field(default_factory=list)
    facts_to_remove: list[str] = Field(default_factory=list)
    threads_opened: list[str] = Field(default_factory=list)
    threads_closed: list[str] = Field(default_factory=list)
    relationship_updates: list[str] = Field(default_factory=list)
    suspicion_updates: list[str] = Field(default_factory=list)
    evidence_updates: list[str] = Field(default_factory=list)
    moral_lines_crossed: list[str] = Field(default_factory=list)
    location_updates: dict[str, str] = Field(default_factory=dict)
    knowledge_updates: dict[str, list[str]] = Field(default_factory=dict)
    emotional_updates: dict[str, str] = Field(default_factory=dict)
    promises_made: list[str] = Field(default_factory=list)
    promises_fulfilled: list[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# QA Reports (enhanced)
# ---------------------------------------------------------------------------

class DeterministicValidationReport(ArtifactModel):
    scene_number: int = 0
    passed: bool = True
    word_count: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SceneQaReport(ArtifactModel):
    scene_number: int = 0
    passed: bool = True
    continuity_score: int = Field(default=3, ge=1, le=5)
    engagement_score: int = Field(default=3, ge=1, le=5)
    voice_score: int = Field(default=3, ge=1, le=5)
    pacing_score: int = Field(default=3, ge=1, le=5)
    emotional_movement_score: int = Field(default=3, ge=1, le=5)
    dialogue_quality_score: int = Field(default=3, ge=1, le=5)
    sensory_detail_score: int = Field(default=3, ge=1, le=5)
    subtext_score: int = Field(default=3, ge=1, le=5)
    tension_score: int = Field(default=3, ge=1, le=5)
    character_consistency_score: int = Field(default=3, ge=1, le=5)
    prose_rhythm_score: int = Field(default=3, ge=1, le=5)
    originality_score: int = Field(default=3, ge=1, le=5)
    ai_smell_score: int = Field(default=3, ge=1, le=5, description="5=fully human, 1=obviously AI")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    rewrite_suggestions: list[str] = Field(default_factory=list)
    overall_notes: str = ""


class ChapterQaReport(ArtifactModel):
    chapter_number: int = 0
    passed: bool = True
    coherence_score: int = Field(default=3, ge=1, le=5)
    pacing_score: int = Field(default=3, ge=1, le=5)
    arc_progression_score: int = Field(default=3, ge=1, le=5)
    hook_quality_score: int = Field(default=3, ge=1, le=5, description="NEW: chapter opening hook")
    cliffhanger_score: int = Field(default=3, ge=1, le=5, description="NEW: chapter ending pull")
    transition_quality_score: int = Field(default=3, ge=1, le=5, description="NEW: scene-to-scene flow")
    issues: list[str] = Field(default_factory=list)
    notes: str = ""


class ArcQaReport(ArtifactModel):
    arc_name: str = ""
    passed: bool = True
    arc_coherence_score: int = Field(default=3, ge=1, le=5)
    escalation_score: int = Field(default=3, ge=1, le=5)
    payoff_score: int = Field(default=3, ge=1, le=5)
    issues: list[str] = Field(default_factory=list)
    notes: str = ""


class GlobalQaReport(ArtifactModel):
    passed: bool = True
    overall_score: int = Field(default=3, ge=1, le=5)
    narrative_coherence_score: int = Field(default=3, ge=1, le=5)
    character_development_score: int = Field(default=3, ge=1, le=5)
    pacing_score: int = Field(default=3, ge=1, le=5)
    voice_consistency_score: int = Field(default=3, ge=1, le=5)
    emotional_impact_score: int = Field(default=3, ge=1, le=5)
    thematic_depth_score: int = Field(default=3, ge=1, le=5)
    dialogue_quality_score: int = Field(default=3, ge=1, le=5)
    prose_quality_score: int = Field(default=3, ge=1, le=5)
    originality_score: int = Field(default=3, ge=1, le=5)
    ai_smell_score: int = Field(default=3, ge=1, le=5)
    commercial_viability_score: int = Field(default=3, ge=1, le=5)
    scene_level_issues: list[str] = Field(default_factory=list)
    structural_issues: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    repair_priorities: list[str] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# NEW: Cold Reader Report
# ---------------------------------------------------------------------------

class ColdReaderReport(ArtifactModel):
    """Report from a judge reading the manuscript without any planning context."""
    confusion_points: list[str] = Field(description="Where the reader got confused")
    predictable_moments: list[str] = Field(description="Where the reader predicted what was coming")
    engagement_drops: list[str] = Field(description="Where the reader lost interest")
    character_tracking_issues: list[str] = Field(description="Where characters felt inconsistent or forgettable")
    emotional_peaks: list[str] = Field(description="Where the reader felt most engaged/moved")
    unanswered_questions: list[str] = Field(description="Loose threads or unresolved elements")
    overall_impression: str
    would_keep_reading: bool
    standout_scenes: list[int] = Field(description="Scene numbers that were strongest")
    weakest_scenes: list[int] = Field(description="Scene numbers that need most work")
    overall_score: int = Field(ge=1, le=10)


# ---------------------------------------------------------------------------
# NEW: Pacing Analysis
# ---------------------------------------------------------------------------

class ScenePacingData(ArtifactModel):
    scene_number: int
    tension_level: int = Field(ge=1, le=10)
    stakes_level: int = Field(ge=1, le=10)
    action_density: int = Field(ge=1, le=10)
    emotional_intensity: int = Field(ge=1, le=10)


class PacingAnalysis(ArtifactModel):
    """Tension curve analysis across the full manuscript."""
    scene_data: list[ScenePacingData]
    tension_sags: list[str] = Field(description="Sections where tension drops too long")
    fatigue_zones: list[str] = Field(description="Sections of sustained high intensity")
    pacing_verdict: str
    recommendations: list[str]


# ---------------------------------------------------------------------------
# NEW: Post-Draft Pass Reports
# ---------------------------------------------------------------------------

class TransitionReport(ArtifactModel):
    """Report on scene-to-scene transition quality."""
    scene_pair: str = Field(description="e.g. 'scene_05_to_06'")
    original_transition_quality: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)
    revised_ending: str = Field(default="", description="Revised last 2 paragraphs of prior scene")
    revised_opening: str = Field(default="", description="Revised first 2 paragraphs of next scene")


class DialogueAuditReport(ArtifactModel):
    """Report on dialogue quality and character voice distinctiveness."""
    character_name: str
    voice_distinctiveness_score: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)
    lines_to_revise: list[str] = Field(default_factory=list)
    revised_lines: list[str] = Field(default_factory=list)


class AntiAiPassReport(ArtifactModel):
    """Report from the anti-AI decontamination pass."""
    patterns_found: list[str]
    total_instances: int
    revisions_made: list[str]
    before_score: int = Field(ge=1, le=10)
    after_score: int = Field(ge=1, le=10)


class ProseRhythmReport(ArtifactModel):
    """Report on prose rhythm and sentence variety analysis."""
    scene_number: int
    monotonous_passages: list[str] = Field(description="Passages with repetitive sentence length")
    rhythm_score: int = Field(ge=1, le=5)
    revisions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Intake & Logging
# ---------------------------------------------------------------------------

class BookIntake(ArtifactModel):
    raw_markdown: str = ""
    fields: dict[str, str] = Field(default_factory=dict)


class RunLogEvent(ArtifactModel):
    timestamp: str
    event_type: str
    details: str = ""
    scene_number: int | None = None
    score: float | None = None
