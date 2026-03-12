"""Centralized prompt system for the improved novel pipeline.

All prompts emphasize operational specificity over thematic abstraction,
with new prompts for voice calibration, beat sheet validation, best-of-N
selection, post-draft passes, and cold reader testing.
"""

from __future__ import annotations

from novel_factory.schemas import (
    ContinuityState,
    DeterministicValidationReport,
    Outline,
    SceneCard,
    StorySpec,
)


# ---------------------------------------------------------------------------
# Global prose policy
# ---------------------------------------------------------------------------

def global_prose_policy(story_spec: StorySpec | None = None) -> str:
    audience = (story_spec.audience if story_spec else "Adult").lower()
    if audience in ("ya", "young adult"):
        content_rules = (
            "Content ceiling: YA-appropriate. No explicit sex, graphic violence, gore, or strong profanity. "
            "Tension comes from psychological pressure, moral dilemmas, and relationship stakes."
        )
    else:
        content_rules = (
            "Content ceiling: Adult trade fiction. Mature institutions, careers, moral ambiguity, "
            "adult relationships are permitted. Avoid gratuitous shock — every dark beat must serve the story."
        )

    return f"""{content_rules}

Style priorities:
- Precision with pressure: every sentence earns its place
- Controlled heat: dangerous, intimate, specific — never melodramatic
- Show internal states through behavior, body language, rhythm, omission — never explain emotions
- Vary sentence length deliberately: short punches after long flowing sentences, fragments for impact
- Ground every scene in at least two senses beyond sight
- Dialogue must sound like real speech: interrupted, elided, subtext-laden, never on-the-nose
- Each character's voice must be distinctive and recognizable without dialogue tags
- Avoid: adverb clusters, "a sense of", "couldn't help but", "the weight of", "a wave of",
  "found himself", "let out a breath", "something shifted", "the silence stretched",
  explaining emotions after showing them, identical sentence structures in sequence,
  purple prose, over-qualifying nouns, rhetorical questions as paragraph openers
"""


# ---------------------------------------------------------------------------
# Voice calibration (NEW)
# ---------------------------------------------------------------------------

def voice_calibration_system_prompt() -> str:
    return (
        "You are a literary analyst specializing in prose style forensics. "
        "You extract precise, measurable voice characteristics from reference passages — "
        "not vague impressions, but concrete patterns a writer could replicate."
    )


def voice_calibration_user_prompt(
    *,
    reference_passages: str,
    genre: str,
    audience: str,
) -> str:
    return f"""Analyze these reference passages and extract a detailed voice DNA profile.

Genre: {genre}
Audience: {audience}

Reference passages:
{reference_passages}

Extract:
1. avg_sentence_length: Count words per sentence, compute average
2. sentence_length_variance: Standard deviation of sentence lengths
3. dialogue_to_narration_ratio: Proportion of dialogue vs narration
4. sensory_density: Sensory words per 100 words (touch, smell, taste, sound, texture)
5. vocabulary_register: Precise description (e.g. "literary-accessible", "sparse-hardboiled")
6. rhythm_signature: How does the prose breathe? Short bursts then long? Steady mid-length?
7. characteristic_techniques: What makes this voice unique? List 5-8 concrete techniques
8. avoid_patterns: What does this voice NOT do? List patterns to avoid
9. sample_paragraph: Write a NEW 150-word paragraph in this exact voice about a character entering a room

Be concrete and operational — a writer should be able to replicate this voice from your analysis.
"""


def character_voice_profile_prompt(
    *,
    character: str,
    role: str,
    story_context: str,
    voice_dna: str,
) -> str:
    return f"""Create a detailed speech and thought profile for this character.

Character: {character}
Role: {role}
Story context: {story_context}
Overall prose voice DNA: {voice_dna}

Define:
1. education_level and vocabulary_range — what words would they use and avoid?
2. speech_patterns — 3-5 distinctive verbal habits (e.g. "starts sentences with 'Look,'", "uses technical jargon when nervous")
3. verbal_tics — filler words, catchphrases, habitual constructions
4. sentence_style — how do their spoken sentences differ from other characters?
5. topics_they_gravitate_toward vs topics_they_avoid
6. emotional_expression_style — do they deflect with humor? Go quiet? Get precise?
7. internal_monologue_style — how their POV thoughts read (fragmented? analytical? sensory?)
8. sample_dialogue — write 5 lines of dialogue ONLY this character would say, in a tense moment

The goal: a reader should identify this character by voice alone, without dialogue tags.
"""


# ---------------------------------------------------------------------------
# Beat sheet validation (NEW)
# ---------------------------------------------------------------------------

def beat_sheet_system_prompt() -> str:
    return (
        "You are a structural story analyst. Map stories to proven beat sheet frameworks "
        "with precision. Identify structural weaknesses before they become prose problems."
    )


def beat_sheet_user_prompt(*, story_spec_json: str, framework: str = "Save the Cat") -> str:
    return f"""Map this story specification to a {framework} beat sheet.

Story specification:
{story_spec_json}

For each beat in the {framework} framework:
1. Name the beat
2. Describe what happens at this beat in THIS specific story
3. Assign the target percentage (where it falls in the manuscript, 0-100)
4. Note any structural concerns (e.g. "catalyst comes too late", "midpoint lacks reversal")

Also provide:
- structural_notes: Overall assessment of structural health
- Flag any missing beats, weak connections between acts, or pacing risks

Be specific to THIS story — don't give generic framework descriptions.
"""


# ---------------------------------------------------------------------------
# Plant/Payoff registry (NEW)
# ---------------------------------------------------------------------------

def plant_payoff_system_prompt() -> str:
    return (
        "You are a narrative craftsperson specializing in foreshadowing and payoff design. "
        "Every plant must feel natural in context. Every payoff must feel inevitable in retrospect."
    )


def plant_payoff_user_prompt(*, story_spec_json: str, outline_json: str) -> str:
    return f"""Design a foreshadowing registry for this story.

Story specification:
{story_spec_json}

Outline:
{outline_json}

Create 8-15 plant/payoff pairs. For each:
1. element: What is being planted (an object, a line of dialogue, a detail, a behavior pattern)
2. plant_scene: Which scene number plants the seed
3. payoff_scene: Which scene number delivers the payoff
4. subtlety_level: "subtle" (reader won't notice first time), "moderate" (attentive reader catches it), "overt" (clear setup)
5. plant_method: How to embed it naturally — it must serve the scene's immediate dramatic purpose too
6. payoff_method: How the payoff manifests — callback, revelation, irony, or consequence

Rules:
- At least 3 plants must span more than 5 scenes between plant and payoff
- At least 2 must involve character behavior (not objects)
- No plant should feel like it's only there for the payoff — it must work in context
- The climax and resolution scenes should pay off at least 3 earlier plants each
"""


# ---------------------------------------------------------------------------
# Subplot weave (NEW)
# ---------------------------------------------------------------------------

def subplot_weave_system_prompt() -> str:
    return (
        "You are a narrative architect. Design subplot structures that enhance the main plot "
        "without competing with it. Every subplot must pressure the protagonist's central dilemma."
    )


def subplot_weave_user_prompt(*, story_spec_json: str, outline_json: str) -> str:
    return f"""Design a subplot weave map for this story.

Story specification:
{story_spec_json}

Outline:
{outline_json}

Identify 2-4 subplots. For each:
1. subplot_name and subplot_type
2. description: What is this subplot about?
3. scene_appearances: Which scenes advance this subplot? (at least 4 scenes each)
4. arc_shape: How does this subplot develop? (e.g. "slow burn to crisis in scene 20")
5. intersection_with_main_plot: Where and how does this subplot create pressure on the main story?

Rules:
- No subplot should go dormant for more than 5 consecutive scenes
- Each subplot must intersect with the main plot at least twice
- Subplots should create competing demands on the protagonist
- At least one subplot must complicate the climax
"""


# ---------------------------------------------------------------------------
# Planning prompts (enhanced from original)
# ---------------------------------------------------------------------------

def planning_system_prompt() -> str:
    return (
        "You generate strict planning artifacts for thriller manuscripts. "
        "Lock choices. Make concrete operational decisions — not aspirational themes. "
        "Every field must contain specific, dramatizable content."
    )


def story_spec_user_prompt(
    *,
    synopsis: str,
    audience: str,
    rating_ceiling: str,
    market_position: str,
    target_words: int,
    target_chapters: int,
    target_scenes: int,
    intake_guidance: str = "",
    voice_dna_summary: str = "",
) -> str:
    voice_section = ""
    if voice_dna_summary:
        voice_section = f"\n\nTarget voice profile:\n{voice_dna_summary}\nThe style guide must align with this voice DNA."

    intake_section = ""
    if intake_guidance:
        intake_section = f"\n\nIntake guidance (highest priority — override defaults where specified):\n{intake_guidance}"

    return f"""Lock the story specification from this synopsis.

Synopsis:
{synopsis}

Parameters:
- Audience: {audience}
- Rating ceiling: {rating_ceiling}
- Market position: {market_position}
- Target: ~{target_words:,} words, {target_chapters} chapters, {target_scenes} scenes
{voice_section}{intake_section}

Requirements:
- The protagonist must WANT something concrete, FEAR something specific, and HIDE something dangerous
- The antagonist/counterforce must have a coherent private goal that creates structural opposition
- The emotional engine must create pressure that compounds — not episodic, cumulative
- The adversarial engine must escalate through the story, not repeat at the same level
- The moral fault line must force choices where every option costs something
- The style guide must include at least 5 banned AI-tell patterns specific to this genre
- Cast: include all named characters with concrete roles and relationships

{global_prose_policy()}
"""


def outline_user_prompt(
    *,
    story_spec_json: str,
    intake_guidance: str = "",
    beat_sheet_json: str = "",
) -> str:
    beat_section = ""
    if beat_sheet_json:
        beat_section = f"\n\nBeat sheet (structure must align with these beats):\n{beat_sheet_json}"

    intake_section = ""
    if intake_guidance:
        intake_section = f"\n\nIntake guidance:\n{intake_guidance}"

    return f"""Expand this story spec into a detailed chapter outline.

Story specification:
{story_spec_json}
{beat_section}{intake_section}

Requirements:
- Every chapter must have a distinct dramatic purpose — no "setup" chapters that only establish
- Chapters must advance pressure, relationship deterioration, or moral compromise every 2-3 scenes
- Each chapter needs an opening_hook (why start reading) and closing_hook (why keep reading)
- Scene types must vary within chapters: don't stack 3 dialogue scenes or 3 action scenes
- The midpoint must genuinely change the story's direction, not just raise stakes
- The final third must accelerate — shorter chapters, faster scene changes
- Assign scene_type to every scene beat: action/dialogue/introspective/confrontation/revelation/quiet/climax
"""


def scene_cards_user_prompt(
    *,
    story_spec_json: str,
    outline_json: str,
    plant_payoff_json: str = "",
    subplot_weave_json: str = "",
    intake_guidance: str = "",
) -> str:
    extras = ""
    if plant_payoff_json:
        extras += f"\n\nForeshadowing registry (embed these plants/payoffs in the specified scenes):\n{plant_payoff_json}"
    if subplot_weave_json:
        extras += f"\n\nSubplot weave map (advance subplots in the specified scenes):\n{subplot_weave_json}"
    if intake_guidance:
        extras += f"\n\nIntake guidance:\n{intake_guidance}"

    return f"""Expand the outline into detailed scene cards.

Story specification:
{story_spec_json}

Outline:
{outline_json}
{extras}

Each scene card MUST include:
- scene_number, chapter_number, scene_type
- pov_character, location, time_of_day
- dramatic_purpose: what this scene accomplishes in the story (not theme — action)
- opening_disturbance: the first thing that creates tension (not scene-setting)
- mid_scene_reversal: what shifts power, knowledge, or emotional direction mid-scene
- closing_choice: the visible decision the POV character makes that ends the scene
- power_shift: who gains/loses power and how
- suspicion_delta: how suspicion/trust levels change
- emotional_arc: character's emotional journey within this scene (start → end state)
- sensory_anchor: one dominant sensory detail that grounds the scene
- counterforce_trace: how the antagonist's pressure is felt (even if offscreen)
- continuity_inputs: facts/state this scene MUST acknowledge from prior scenes
- continuity_outputs: facts/state this scene establishes for future scenes
- required_entities: characters/objects that MUST appear
- forbidden_entities: characters/objects that MUST NOT appear
- plants_in_this_scene: foreshadowing elements to embed (from the registry)
- payoffs_in_this_scene: earlier plants that pay off here
- word_target: 900-2200 words, with variation (not all scenes the same length)

Rules:
- At least one meaningful deterioration per scene (relationship, trust, safety, moral standing)
- No scene exists just to convey information — information must come through conflict
- Scene types must match the outline: don't make an "action" beat into a dialogue scene
- Word targets should vary: quiet scenes shorter (900-1200), climactic scenes longer (1800-2200)
"""


# ---------------------------------------------------------------------------
# Drafting prompts (enhanced)
# ---------------------------------------------------------------------------

def scene_draft_system_prompt(story_spec: StorySpec) -> str:
    return f"""You are writing a scene for a {story_spec.genre} novel.

{global_prose_policy(story_spec)}

Core rules:
- Write ONLY the prose. No scene headers, no meta-commentary, no author notes.
- Dramatize pressure through behavior, rhythm, omission, and action — never explain directly.
- Every paragraph must either advance conflict, reveal character, or shift power. Delete anything else.
- Dialogue must drip with subtext. Characters rarely say what they mean.
- Ground every scene in sensory reality: temperature, texture, sound, smell.
- Vary your sentence rhythm: short punches, long flowing observations, fragments, one-word paragraphs.
- The scene must feel like it was torn from the middle of a life, not constructed for a reader.
"""


def scene_draft_user_prompt(
    *,
    story_spec: StorySpec,
    scene_card: SceneCard,
    chapter_plan_json: str,
    continuity_state_json: str,
    recent_summaries: str,
    voice_dna_summary: str = "",
    character_voice_profiles: str = "",
    lookahead_cards: str = "",
    intake_guidance: str = "",
    rewrite_brief: str = "",
) -> str:
    voice_section = ""
    if voice_dna_summary:
        voice_section = f"\n\nVoice DNA (match this style):\n{voice_dna_summary}"
    if character_voice_profiles:
        voice_section += f"\n\nCharacter voice profiles (each character must sound distinct):\n{character_voice_profiles}"

    lookahead_section = ""
    if lookahead_cards:
        lookahead_section = f"\n\nUpcoming scenes (plant seeds for these — do NOT spoil them):\n{lookahead_cards}"

    intake_section = ""
    if intake_guidance:
        intake_section = f"\n\nIntake guidance (highest priority):\n{intake_guidance}"

    rewrite_section = ""
    if rewrite_brief:
        rewrite_section = f"\n\nREWRITE BRIEF — fix these specific issues while preserving what works:\n{rewrite_brief}"

    plants_section = ""
    if scene_card.plants_in_this_scene:
        plants_section = "\n\nFORESHADOWING TO PLANT (embed naturally — must serve the scene's immediate drama too):\n"
        for plant in scene_card.plants_in_this_scene:
            plants_section += f"- {plant}\n"

    payoffs_section = ""
    if scene_card.payoffs_in_this_scene:
        payoffs_section = "\n\nPAYOFFS TO DELIVER (earlier seeds paying off — make them feel inevitable):\n"
        for payoff in scene_card.payoffs_in_this_scene:
            payoffs_section += f"- {payoff}\n"

    return f"""Draft scene {scene_card.scene_number} ({scene_card.scene_type}).

Scene card:
- POV: {scene_card.pov_character}
- Location: {scene_card.location} | Time: {scene_card.time_of_day}
- Dramatic purpose: {scene_card.dramatic_purpose}
- Opening disturbance: {scene_card.opening_disturbance}
- Mid-scene reversal: {scene_card.mid_scene_reversal}
- Closing choice: {scene_card.closing_choice}
- Power shift: {scene_card.power_shift}
- Emotional arc: {scene_card.emotional_arc}
- Sensory anchor: {scene_card.sensory_anchor}
- Counterforce trace: {scene_card.counterforce_trace}

Continuity inputs (MUST acknowledge): {', '.join(scene_card.continuity_inputs) or 'none'}
Continuity outputs (MUST establish): {', '.join(scene_card.continuity_outputs) or 'none'}
Required entities: {', '.join(scene_card.required_entities) or 'none'}
Forbidden entities: {', '.join(scene_card.forbidden_entities) or 'none'}

Target: ~{scene_card.word_target} words
{plants_section}{payoffs_section}

Chapter plan:
{chapter_plan_json}

Current story state:
{continuity_state_json}

Recent scene summaries:
{recent_summaries}
{voice_section}{lookahead_section}{intake_section}{rewrite_section}

CRITICAL RULES:
1. Start with action or tension — never scene-setting or weather
2. The opening disturbance must land in the first 3 paragraphs
3. Continuity inputs are CONSTRAINTS — reference them naturally, never dump them
4. Continuity outputs must be ESTABLISHED through drama, not exposition
5. The closing choice must be a visible decision with consequences
6. If this is a rewrite, address every issue in the rewrite brief
"""


# ---------------------------------------------------------------------------
# Best-of-N selection prompt (NEW)
# ---------------------------------------------------------------------------

def best_of_n_selection_system_prompt() -> str:
    return (
        "You are a senior literary editor selecting the strongest draft from multiple candidates. "
        "Judge by prose quality, emotional impact, narrative momentum, and human-sounding voice — "
        "not by adherence to instructions (all candidates follow the same brief)."
    )


def best_of_n_selection_user_prompt(
    *,
    scene_card_json: str,
    candidates: list[tuple[str, str]],  # list of (label, text) pairs
) -> str:
    candidates_text = ""
    for label, text in candidates:
        candidates_text += f"\n\n--- {label} ---\n{text}\n--- END {label} ---"

    return f"""Select the best draft for this scene.

Scene card:
{scene_card_json}

Candidates:{candidates_text}

Evaluate each candidate on:
1. Prose quality (rhythm, word choice, sentence variety, sensory grounding)
2. Emotional impact (does it make you feel something?)
3. Narrative momentum (does it pull you forward?)
4. Character voice (do characters sound like distinct people?)
5. Subtext (is meaning conveyed under the surface?)
6. AI-smell (does it read as human-written?)

Return ONLY the label of the best candidate (e.g. "CANDIDATE_A") and a brief explanation of why.
Do not suggest combining candidates. Pick one winner.
"""


# ---------------------------------------------------------------------------
# Continuity prompts (enhanced)
# ---------------------------------------------------------------------------

def continuity_system_prompt() -> str:
    return (
        "You track narrative state with forensic precision. Record only what is concretely "
        "established in the scene text — not implications, not themes, not what 'might' follow."
    )


def initial_continuity_user_prompt(*, story_spec_json: str, scene_cards_json: str) -> str:
    return f"""Create the initial continuity state (before scene 1).

Story specification:
{story_spec_json}

Scene cards:
{scene_cards_json}

Set:
- after_scene: 0
- character_locations: where each named character is at story start
- known_facts: established backstory facts (from the story spec, not scene cards)
- open_threads: narrative questions the reader has after the hook/premise
- relationship_states: starting relationship dynamics
- suspicion_levels: initial trust/suspicion between key characters
- evidence_items: empty
- moral_lines_crossed: empty
- recent_summaries: empty
- character_knowledge: what each character knows at story start (critical for dramatic irony)
- emotional_states: starting emotional state for each character
- active_promises: narrative promises the premise makes to the reader
"""


def continuity_update_user_prompt(
    *,
    scene_card_json: str,
    scene_text: str,
    current_state_json: str,
) -> str:
    return f"""Record the continuity delta from this scene.

Scene card:
{scene_card_json}

Scene text:
{scene_text}

Current state before this scene:
{current_state_json}

Record ONLY what is concretely established in the scene text:
- facts_to_add / facts_to_remove: new information established or contradicted
- threads_opened / threads_closed: new questions raised or answered
- relationship_updates: how relationships changed (with evidence from the text)
- suspicion_updates: trust/suspicion shifts
- evidence_updates: physical evidence introduced, moved, or revealed
- moral_lines_crossed: ethical boundaries crossed in this scene
- location_updates: where characters are at scene end
- knowledge_updates: what each character now knows (critical: don't let characters know things they shouldn't)
- emotional_updates: emotional state changes
- promises_made / promises_fulfilled: narrative promises to the reader
- summary: 2-3 sentence summary of the scene's key events

Do NOT infer or predict — record only what happened on page.
"""


# ---------------------------------------------------------------------------
# Post-draft pass prompts (ALL NEW)
# ---------------------------------------------------------------------------

def transition_smoothing_system_prompt() -> str:
    return (
        "You are a prose editor specializing in scene transitions. Your job is to make the seam "
        "between consecutive scenes feel natural and propulsive — never jarring, never redundant."
    )


def transition_smoothing_user_prompt(
    *,
    scene_a_ending: str,
    scene_b_opening: str,
    scene_a_card_json: str,
    scene_b_card_json: str,
) -> str:
    return f"""Smooth the transition between these consecutive scenes.

SCENE A (ending — last ~300 words):
{scene_a_ending}

SCENE B (opening — first ~300 words):
{scene_b_opening}

Scene A card: {scene_a_card_json}
Scene B card: {scene_b_card_json}

Evaluate:
1. Does Scene A end with momentum that pulls into Scene B?
2. Does Scene B open with enough orientation without redundant scene-setting?
3. Is there an emotional bridge — does the reader's feeling at A's end connect to B's opening?
4. Is there unintentional repetition of information, mood, or imagery?

If the transition scores below 4/5, provide:
- revised_ending: rewritten last 2 paragraphs of Scene A
- revised_opening: rewritten first 2 paragraphs of Scene B

If it scores 4-5/5, return empty revised fields and note why it works.
"""


def dialogue_polish_system_prompt() -> str:
    return (
        "You are a dialogue specialist. Your ear is tuned to how real people talk — "
        "interrupted, elided, loaded with subtext, shaped by power dynamics and emotional state. "
        "You can identify AI-generated dialogue by its completeness, politeness, and on-the-nose quality."
    )


def dialogue_polish_user_prompt(
    *,
    character_name: str,
    character_voice_profile: str,
    dialogue_lines: list[str],
    scene_contexts: list[str],
) -> str:
    lines_text = "\n".join(f"  {i+1}. \"{line}\"" for i, line in enumerate(dialogue_lines))
    contexts_text = "\n".join(f"  {i+1}. {ctx}" for i, ctx in enumerate(scene_contexts))

    return f"""Audit and polish dialogue for {character_name}.

Voice profile:
{character_voice_profile}

Dialogue lines across the manuscript:
{lines_text}

Scene contexts for each line:
{contexts_text}

For each line, evaluate:
1. Does it sound like THIS specific character?
2. Is there subtext — is the character saying less than they mean?
3. Does it sound like natural speech (contractions, interruptions, false starts)?
4. Is it free of AI-tells (too complete, too articulate, too on-the-nose)?

Return:
- voice_distinctiveness_score (1-5)
- issues found
- lines_to_revise: the original lines that need work
- revised_lines: the improved versions (same order)
"""


def anti_ai_system_prompt() -> str:
    return (
        "You are an AI-prose decontamination specialist. You have an encyclopedic knowledge of "
        "patterns that mark AI-generated fiction: overuse of certain phrases, rhythmic monotony, "
        "emotional over-explaining, purple tendencies, and structural tells. Your job is to hunt "
        "these patterns and replace them with human-sounding alternatives."
    )


def anti_ai_user_prompt(*, manuscript_text: str) -> str:
    return f"""Scan this manuscript for AI-generated prose patterns and fix them.

Manuscript:
{manuscript_text}

Hunt for these pattern categories:

PHRASE-LEVEL:
- "a sense of [emotion]", "couldn't help but", "the weight of", "a wave of"
- "found himself/herself", "let out a breath", "something shifted"
- "the silence stretched/hung", "a flicker of", "the air felt"
- "despite himself", "involuntarily", "a part of him/her"
- Any emotion named after being shown (double-dipping)

STRUCTURAL:
- Identical sentence openings (He/She starting 3+ consecutive sentences)
- All sentences within a paragraph being similar length
- Paragraphs that follow the same internal structure
- Over-use of em dashes for parenthetical asides
- Semicolons used for rhythmic effect more than 2x per scene

DIALOGUE:
- Characters speaking in complete, grammatically perfect sentences
- Dialogue that perfectly articulates emotional states
- Characters who never interrupt, trail off, or change direction mid-sentence
- Exposition delivered through dialogue ("As you know, Bob...")

EMOTIONAL:
- Naming emotions directly ("he felt angry") instead of showing behavior
- Characters who process emotions too neatly within a single scene
- Emotional reactions that are too proportionate to stimuli
- Internal monologue that over-analyzes in real-time

For each pattern found, provide the original text and a revised version.
Return the total count and before/after AI-smell scores (1-10 scale, 10 = fully human).
"""


def prose_rhythm_system_prompt() -> str:
    return (
        "You are a prose rhythm analyst. You read for cadence, music, and breath — "
        "the way sentences interact, the way paragraph lengths create visual and emotional pacing."
    )


def prose_rhythm_user_prompt(*, scene_number: int, scene_text: str) -> str:
    return f"""Analyze and fix prose rhythm issues in scene {scene_number}.

Scene text:
{scene_text}

Analyze:
1. Sentence length variety within each paragraph (flag if std dev < 3 words)
2. Paragraph length variety (flag if all paragraphs within 20% of each other)
3. Opening word variety (flag if >30% of sentences start with the same word)
4. Rhythmic monotony: sections where the prose falls into a repetitive cadence
5. Missing impact: moments that should hit hard but are buried in long sentences
6. Missing breath: dense sequences with no short sentences or fragments for relief

Return:
- monotonous_passages: quote the specific passages
- rhythm_score: 1-5
- revisions: specific before/after replacements for the worst offenders
"""


# ---------------------------------------------------------------------------
# QA prompts (enhanced)
# ---------------------------------------------------------------------------

def scene_qa_system_prompt(story_spec: StorySpec) -> str:
    return f"""You are a senior editor evaluating a scene from a {story_spec.genre} manuscript.
Judge as a professional who has read thousands of published novels in this genre.
Score honestly — a 3 means competent but unremarkable. A 5 means publishable by a top house.
A 1-2 means the scene needs fundamental rework.

{global_prose_policy(story_spec)}
"""


def scene_qa_user_prompt(
    *,
    story_spec: StorySpec,
    scene_card: SceneCard,
    continuity_state: ContinuityState,
    validation_report: DeterministicValidationReport,
    scene_text: str,
    intake_guidance: str = "",
) -> str:
    intake_section = ""
    if intake_guidance:
        intake_section = f"\n\nIntake guidance:\n{intake_guidance}"

    return f"""Evaluate this scene draft.

Scene card:
- Scene {scene_card.scene_number} (Ch {scene_card.chapter_number}), Type: {scene_card.scene_type}
- POV: {scene_card.pov_character}
- Dramatic purpose: {scene_card.dramatic_purpose}
- Opening disturbance: {scene_card.opening_disturbance}
- Mid-scene reversal: {scene_card.mid_scene_reversal}
- Closing choice: {scene_card.closing_choice}
- Required entities: {scene_card.required_entities}
- Continuity inputs: {scene_card.continuity_inputs}
- Continuity outputs: {scene_card.continuity_outputs}

Validation report:
- Passed: {validation_report.passed}
- Errors: {validation_report.errors}
- Warnings: {validation_report.warnings}

Current continuity state (after scene {continuity_state.after_scene}):
- Character knowledge: {continuity_state.character_knowledge}
- Emotional states: {continuity_state.emotional_states}

Scene text:
{scene_text}
{intake_section}

Score each dimension 1-5:
- continuity_score: Does it respect and advance the story state?
- engagement_score: Would a reader be compelled to continue?
- voice_score: Does the prose have a distinctive, consistent voice?
- pacing_score: Does the scene breathe — tension and release, fast and slow?
- emotional_movement_score: Does the POV character's emotional state change?
- dialogue_quality_score: Does dialogue sound human, distinct, subtext-laden?
- sensory_detail_score: Is the scene grounded in physical reality?
- subtext_score: Is meaning conveyed beneath the surface?
- tension_score: Is there genuine dramatic tension throughout?
- character_consistency_score: Do characters behave consistently with their profiles?
- prose_rhythm_score: Does the prose have varied, musical sentence patterns?
- originality_score: Does the scene avoid cliche and surprise the reader?
- ai_smell_score: 5 = reads fully human, 1 = obviously AI-generated

Set passed=true only if ALL scores >= 3 AND average >= 3.5 AND ai_smell_score >= 3.
List specific strengths, weaknesses, and concrete rewrite suggestions.
"""


def chapter_qa_system_prompt(story_spec: StorySpec) -> str:
    return f"""You are evaluating a full chapter from a {story_spec.genre} manuscript.
Judge chapter-level qualities: coherence across scenes, pacing, arc progression,
and critically — how the chapter opens and closes.
"""


def chapter_qa_user_prompt(
    *,
    story_spec: StorySpec,
    outline: Outline,
    chapter_number: int,
    chapter_text: str,
    scene_cards: list[SceneCard],
) -> str:
    return f"""Evaluate chapter {chapter_number}.

Story: {story_spec.title} ({story_spec.genre})
Chapter scenes: {[sc.scene_number for sc in scene_cards]}

Chapter text:
{chapter_text}

Score 1-5:
- coherence_score: Do scenes connect into a unified chapter experience?
- pacing_score: Does the chapter move well — no sags, no rushing?
- arc_progression_score: Does the chapter advance the story meaningfully?
- hook_quality_score: Does the chapter open with something that demands attention?
- cliffhanger_score: Does the chapter end making the reader NEED to continue?
- transition_quality_score: Do scenes flow into each other naturally?

List specific issues and notes.
"""


def arc_qa_system_prompt(story_spec: StorySpec) -> str:
    return f"You evaluate narrative arcs in {story_spec.genre} manuscripts for coherence, escalation, and payoff."


def arc_qa_user_prompt(
    *,
    story_spec: StorySpec,
    outline: Outline,
    arc_name: str,
    arc_focus: str,
    scene_numbers: list[int],
    arc_text: str,
) -> str:
    return f"""Evaluate the "{arc_name}" arc.

Focus: {arc_focus}
Scenes involved: {scene_numbers}

Arc text (extracted scenes):
{arc_text}

Score 1-5:
- arc_coherence_score: Does this arc tell a complete, connected sub-story?
- escalation_score: Does tension/stakes increase across the arc's appearances?
- payoff_score: Does the arc resolve satisfyingly (or set up a clear future payoff)?

List issues and notes.
"""


def global_qa_system_prompt(story_spec: StorySpec) -> str:
    return f"""You are the final editorial judge on a complete {story_spec.genre} manuscript.
This is a contest submission. Judge it against the best published novels in the genre.
Be ruthlessly honest. A 3 is "competent debut." A 5 is "award-worthy."
"""


def global_qa_user_prompt(
    *,
    story_spec: StorySpec,
    outline: Outline,
    manuscript_text: str,
    intake_guidance: str = "",
) -> str:
    intake_section = ""
    if intake_guidance:
        intake_section = f"\n\nIntake guidance:\n{intake_guidance}"

    return f"""Evaluate the complete manuscript.

Title: {story_spec.title}
Genre: {story_spec.genre}
Target audience: {story_spec.audience}
{intake_section}

Manuscript:
{manuscript_text}

Score 1-5:
- overall_score: Holistic quality assessment
- narrative_coherence_score: Does the story make sense from beginning to end?
- character_development_score: Do characters change believably?
- pacing_score: Does the manuscript sustain momentum across its full length?
- voice_consistency_score: Is the prose voice consistent throughout?
- emotional_impact_score: Does the story move the reader?
- thematic_depth_score: Does the story say something meaningful?
- dialogue_quality_score: Is dialogue consistently strong across the manuscript?
- prose_quality_score: Is the writing itself excellent — sentence-level craft?
- originality_score: Does the story feel fresh within its genre?
- ai_smell_score: 5 = reads as a skilled human author, 1 = obviously AI
- commercial_viability_score: Would a publisher acquire this?

List:
- scene_level_issues: specific scenes that need work (with scene numbers)
- structural_issues: bigger-picture problems
- strengths: what works well
- repair_priorities: ordered list of what to fix first
"""


# ---------------------------------------------------------------------------
# Cold reader prompt (NEW)
# ---------------------------------------------------------------------------

def cold_reader_system_prompt() -> str:
    return (
        "You are a voracious reader picking up a new novel with NO prior knowledge of the story. "
        "You have not seen any outlines, character sheets, or planning documents. "
        "Read as a reader, not an editor. Report your genuine experience — confusion, boredom, "
        "excitement, predictions, questions. Be brutally honest about where you'd put the book down."
    )


def cold_reader_user_prompt(*, manuscript_text: str) -> str:
    return f"""Read this manuscript cold — as if you picked it up in a bookstore.

Manuscript:
{manuscript_text}

Report your genuine reading experience:

1. confusion_points: Where did you get lost? Couldn't follow who was who, what was happening, or why?
2. predictable_moments: Where did you predict what was coming? (This suggests the story is telegraphing)
3. engagement_drops: Where did you lose interest? Where would you have put the book down?
4. character_tracking_issues: Which characters felt inconsistent, forgettable, or interchangeable?
5. emotional_peaks: Where did you feel most engaged, tense, moved, or surprised?
6. unanswered_questions: What questions does the story raise but never answer?
7. overall_impression: Your honest gut reaction in 2-3 sentences
8. would_keep_reading: If this were the first 50 pages, would you buy the book?
9. standout_scenes: Scene numbers (roughly — by position) that were strongest
10. weakest_scenes: Scene numbers that need the most work
11. overall_score: 1-10 (1=unreadable, 5=competent debut, 7=strong commercial, 10=masterpiece)

Be honest. Don't be kind. A 7 is genuinely good.
"""


# ---------------------------------------------------------------------------
# Pacing analysis prompt (NEW)
# ---------------------------------------------------------------------------

def pacing_analysis_system_prompt() -> str:
    return (
        "You analyze narrative pacing by mapping tension, stakes, action density, and emotional "
        "intensity across a full manuscript. You identify sags, fatigue zones, and pacing imbalances."
    )


def pacing_analysis_user_prompt(*, manuscript_text: str, scene_count: int) -> str:
    return f"""Map the pacing curve of this {scene_count}-scene manuscript.

Manuscript:
{manuscript_text}

For each scene (1 through {scene_count}), rate:
- tension_level (1-10): How tense is this scene?
- stakes_level (1-10): How much is at risk?
- action_density (1-10): How much happens? (1=contemplative, 10=constant action)
- emotional_intensity (1-10): How emotionally charged?

Then analyze the overall curve:
- tension_sags: Identify any run of 3+ scenes below tension level 4
- fatigue_zones: Identify any run of 4+ scenes above tension level 8
- pacing_verdict: Overall assessment of the pacing architecture
- recommendations: Specific actionable suggestions for improvement
"""


# ---------------------------------------------------------------------------
# Repair prompt (enhanced)
# ---------------------------------------------------------------------------

def repair_scene_system_prompt(story_spec: StorySpec) -> str:
    return f"""You are rewriting a scene that failed quality review in a {story_spec.genre} manuscript.
Preserve the scene's role and approximate length. Fix the identified issues.
Do not introduce new continuity violations or change the scene's dramatic purpose.

{global_prose_policy(story_spec)}
"""


def repair_scene_user_prompt(
    *,
    scene_card_json: str,
    original_text: str,
    qa_issues: str,
    continuity_state_json: str,
    voice_dna_summary: str = "",
) -> str:
    voice_section = ""
    if voice_dna_summary:
        voice_section = f"\n\nVoice DNA (match this style):\n{voice_dna_summary}"

    return f"""Rewrite this scene to fix the identified issues.

Scene card:
{scene_card_json}

Issues to fix:
{qa_issues}

Current continuity state:
{continuity_state_json}

Original text:
{original_text}
{voice_section}

Rules:
1. Fix EVERY listed issue
2. Preserve the scene's dramatic purpose and approximate word count
3. Do not introduce new characters or plot elements not in the scene card
4. Do not violate continuity — characters can only know what they've learned on-page
5. Maintain the overall voice and style of the manuscript
"""


# ---------------------------------------------------------------------------
# Chapter hook audit (NEW)
# ---------------------------------------------------------------------------

def chapter_hook_audit_system_prompt() -> str:
    return (
        "You audit chapter openings and closings for propulsive power. "
        "A great opening creates an immediate question or tension. "
        "A great closing makes putting the book down feel impossible."
    )


def chapter_hook_audit_user_prompt(
    *,
    chapter_number: int,
    chapter_opening: str,
    chapter_closing: str,
) -> str:
    return f"""Audit the hooks for chapter {chapter_number}.

Chapter opening (first ~300 words):
{chapter_opening}

Chapter closing (last ~300 words):
{chapter_closing}

Evaluate:
1. Opening hook strength (1-5): Does the first paragraph create immediate tension or a question?
2. Closing hook strength (1-5): Does the last paragraph make the reader NEED to turn the page?
3. If either scores below 4, provide a rewritten version that scores higher.

Rules for rewrites:
- Opening: Start mid-action, mid-thought, or mid-conflict — never with scene-setting or weather
- Closing: End on a decision, a revelation, a threat, or an unanswered question — never on resolution
"""


# ---------------------------------------------------------------------------
# Final polish prompt (NEW)
# ---------------------------------------------------------------------------

def final_polish_system_prompt() -> str:
    return (
        "You are performing the final editorial pass on a complete manuscript before submission. "
        "This is line-level work: consistency in voice, elimination of remaining AI-tells, "
        "tightening of flabby prose, and ensuring every chapter ending pulls the reader forward. "
        "Do NOT change plot, character, or structure. Only polish the prose."
    )


def final_polish_user_prompt(*, chapter_text: str, chapter_number: int, voice_dna_summary: str = "") -> str:
    voice_section = ""
    if voice_dna_summary:
        voice_section = f"\n\nVoice DNA target:\n{voice_dna_summary}"

    return f"""Final polish pass on chapter {chapter_number}.

{chapter_text}
{voice_section}

Make line-level improvements ONLY:
1. Eliminate any remaining AI-tell phrases
2. Tighten flabby sentences (cut unnecessary words)
3. Fix any inconsistencies in character voice
4. Improve sentence rhythm where it feels monotonous
5. Strengthen sensory details that are vague
6. Ensure chapter ends with maximum forward pull

Return the complete polished chapter text. Changes should be surgical, not wholesale.
"""
