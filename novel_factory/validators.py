"""Deterministic validation checks for scenes and planning artifacts."""

from __future__ import annotations

import re
from typing import Sequence

from novel_factory.schemas import (
    ContinuityState,
    DeterministicValidationReport,
    Outline,
    SceneCard,
    StorySpec,
)
from novel_factory.utils import (
    build_ngrams,
    count_words,
    first_token,
    split_paragraphs,
    split_sentences,
    compute_sentence_length_stats,
    WORD_RE,
)

# AI-tell cliches to flag
AI_CLICHES = [
    "heart pounded", "heart hammered", "heart raced",
    "deafening silence", "palpable tension", "thick with tension",
    "a sense of", "a wave of", "a flicker of",
    "couldn't help but", "found himself", "found herself",
    "let out a breath", "released a breath", "exhaled a breath",
    "the weight of", "the silence stretched", "the air felt",
    "something shifted", "despite himself", "despite herself",
    "involuntarily", "a part of him", "a part of her",
    "time seemed to", "the world seemed to",
    "knuckles whitened", "jaw clenched", "jaw tightened",
    "piercing gaze", "steely gaze", "searching gaze",
    "swallowed hard", "throat tightened",
    "exchanged a glance", "exchanged a look",
    "the gravity of", "the enormity of",
    "a chill ran down", "sent a shiver",
    "blood ran cold", "stomach dropped",
    "eyes widened", "brow furrowed",
]

RHETORICAL_PATTERNS = [
    r"\bas if\b", r"\bit felt like\b", r"\bit was as though\b",
    r"\bsomething about\b", r"\bthere was something\b",
]


class SceneValidator:
    """Runs deterministic pre-submission checks on a drafted scene."""

    def validate(
        self,
        *,
        scene_card: SceneCard,
        scene_text: str,
        continuity_state: ContinuityState,
        story_spec: StorySpec | None = None,
    ) -> DeterministicValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        word_count = count_words(scene_text)
        target = scene_card.word_target

        # Word count checks
        floor = min(750, int(target * 0.70))
        ceiling = max(2200, int(target * 1.35))
        if word_count < floor:
            errors.append(f"Word count {word_count} below floor {floor}")
        elif word_count > ceiling:
            warnings.append(f"Word count {word_count} above ceiling {ceiling}")

        # Sentence opening repetition
        sentences = split_sentences(scene_text)
        if sentences:
            openers = [first_token(s) for s in sentences]
            opener_counts: dict[str, int] = {}
            for op in openers:
                opener_counts[op] = opener_counts.get(op, 0) + 1
            for token, count in opener_counts.items():
                ratio = count / len(openers)
                if count >= 4 and ratio >= 0.28:
                    warnings.append(
                        f"Sentence opener '{token}' appears {count} times ({ratio:.0%})"
                    )

        # N-gram repetition
        tokens = [m.group(0).lower() for m in WORD_RE.finditer(scene_text)]
        bigrams = build_ngrams(tokens, 2)
        for gram, count in bigrams.items():
            if count >= 5:
                warnings.append(f"Bigram '{' '.join(gram)}' repeated {count} times")
        trigrams = build_ngrams(tokens, 3)
        for gram, count in trigrams.items():
            if count >= 4:
                warnings.append(f"Trigram '{' '.join(gram)}' repeated {count} times")

        # AI cliche detection
        text_lower = scene_text.lower()
        cliches_found = []
        for cliche in AI_CLICHES:
            occurrences = text_lower.count(cliche)
            if occurrences > 0:
                cliches_found.append(f"'{cliche}' x{occurrences}")
        if cliches_found:
            warnings.append(f"AI-tell cliches found: {', '.join(cliches_found)}")
        if len(cliches_found) >= 3:
            errors.append(f"Too many AI-tell cliches ({len(cliches_found)}): scene needs decontamination")

        # Paragraph metrics
        paragraphs = split_paragraphs(scene_text)
        for i, para in enumerate(paragraphs):
            para_words = count_words(para)
            if para_words > 220:
                warnings.append(f"Paragraph {i+1} is {para_words} words (max 220)")

        # Paragraph opening diversity
        if paragraphs:
            para_openers = [first_token(p) for p in paragraphs if p.strip()]
            if para_openers:
                opener_counts_p: dict[str, int] = {}
                for op in para_openers:
                    opener_counts_p[op] = opener_counts_p.get(op, 0) + 1
                for token, count in opener_counts_p.items():
                    if count / len(para_openers) >= 0.55:
                        warnings.append(
                            f"Paragraph opener '{token}' dominates ({count}/{len(para_openers)})"
                        )

        # Dialogue ratio
        dialogue_chars = sum(len(m.group(0)) for m in re.finditer(r'["\u201c][^"\u201d]*["\u201d]', scene_text))
        total_chars = len(scene_text)
        if total_chars > 0:
            dialogue_ratio = dialogue_chars / total_chars
            if dialogue_ratio < 0.02:
                warnings.append(f"Very low dialogue ratio: {dialogue_ratio:.1%}")
            elif dialogue_ratio > 0.78:
                warnings.append(f"Very high dialogue ratio: {dialogue_ratio:.1%}")

        # Rhetorical pattern overuse
        for pattern in RHETORICAL_PATTERNS:
            matches = re.findall(pattern, scene_text, re.IGNORECASE)
            if len(matches) >= 4:
                warnings.append(f"Rhetorical pattern '{pattern}' appears {len(matches)} times")

        # Punctuation overuse
        em_dashes = scene_text.count("\u2014") + scene_text.count("--")
        if word_count > 0 and em_dashes / word_count > 0.015:
            warnings.append(f"Excessive em dashes: {em_dashes} in {word_count} words")
        semicolons = scene_text.count(";")
        if word_count > 0 and semicolons / word_count > 0.008:
            warnings.append(f"Excessive semicolons: {semicolons} in {word_count} words")

        # Prose rhythm check (NEW)
        stats = compute_sentence_length_stats(scene_text)
        if stats["count"] > 10 and stats["variance_ratio"] < 0.25:
            warnings.append(
                f"Low sentence length variety (variance ratio {stats['variance_ratio']}). "
                f"Mean: {stats['mean']} words, std: {stats['std']}"
            )

        # Entity validation
        for entity in scene_card.required_entities:
            if not self._entity_present(entity, scene_text):
                errors.append(f"Required entity missing: '{entity}'")
        for entity in scene_card.forbidden_entities:
            if self._entity_present(entity, scene_text):
                errors.append(f"Forbidden entity present: '{entity}'")

        passed = len(errors) == 0
        return DeterministicValidationReport(
            scene_number=scene_card.scene_number,
            passed=passed,
            word_count=word_count,
            errors=errors,
            warnings=warnings,
        )

    def _entity_present(self, entity: str, text: str) -> bool:
        """Flexible entity matching with multiple strategies."""
        text_lower = text.lower()
        entity_lower = entity.lower().strip()

        # Skip abstract beat instructions
        abstract_keywords = [
            "conversation", "realization", "gesture", "moment", "feeling",
            "decision", "tension", "conflict", "reveal",
        ]
        if any(kw in entity_lower for kw in abstract_keywords):
            return True  # Don't enforce abstract concepts

        # Direct match
        if entity_lower in text_lower:
            return True

        # Token-based match for multi-word entities
        entity_tokens = entity_lower.split()
        if len(entity_tokens) > 1:
            if all(token in text_lower for token in entity_tokens):
                return True

        # Handle "or"-delimited alternatives
        if " or " in entity_lower:
            alternatives = entity_lower.split(" or ")
            return any(alt.strip() in text_lower for alt in alternatives)

        # Quoted term extraction
        if '"' in entity or '\u201c' in entity:
            quoted = re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', entity)
            return any(q.lower() in text_lower for q in quoted)

        return False


class PlanValidator:
    """Validates planning artifacts for structural integrity."""

    def validate(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: Sequence[SceneCard],
        continuity_state: ContinuityState,
    ) -> DeterministicValidationReport:
        errors: list[str] = []
        warnings: list[str] = []

        # Chapter and scene count alignment
        if len(outline.chapters) != story_spec.target_chapters:
            warnings.append(
                f"Outline has {len(outline.chapters)} chapters, "
                f"spec targets {story_spec.target_chapters}"
            )
        if len(scene_cards) != story_spec.target_scenes:
            warnings.append(
                f"{len(scene_cards)} scene cards, spec targets {story_spec.target_scenes}"
            )

        # Scene numbering contiguity
        scene_numbers = sorted(sc.scene_number for sc in scene_cards)
        expected = list(range(1, len(scene_cards) + 1))
        if scene_numbers != expected:
            errors.append(f"Scene numbering not contiguous: {scene_numbers}")

        # Scene card completeness
        for sc in scene_cards:
            if not sc.dramatic_purpose or sc.dramatic_purpose.lower() in ("tbd", "placeholder", ""):
                warnings.append(f"Scene {sc.scene_number}: missing dramatic_purpose")
            if not sc.opening_disturbance:
                warnings.append(f"Scene {sc.scene_number}: missing opening_disturbance")
            if not sc.closing_choice:
                warnings.append(f"Scene {sc.scene_number}: missing closing_choice")
            if not sc.scene_type:
                warnings.append(f"Scene {sc.scene_number}: missing scene_type")

        # Scene type variety within chapters
        for chapter in outline.chapters:
            chapter_cards = [sc for sc in scene_cards if sc.chapter_number == chapter.chapter_number]
            types = [sc.scene_type for sc in chapter_cards if sc.scene_type]
            if len(types) >= 3 and len(set(types)) == 1:
                warnings.append(
                    f"Chapter {chapter.chapter_number}: all scenes are type '{types[0]}' — needs variety"
                )

        # Rolling intensity check: every 3-scene window should have at least one intensifier
        for i in range(0, len(scene_cards) - 2):
            window = scene_cards[i : i + 3]
            has_intensifier = any(
                sc.power_shift or sc.suspicion_delta or sc.mid_scene_reversal
                for sc in window
            )
            if not has_intensifier:
                warnings.append(
                    f"Scenes {window[0].scene_number}-{window[-1].scene_number}: "
                    f"no intensifier in 3-scene window"
                )

        # Initial continuity state validation
        if continuity_state.after_scene != 0:
            errors.append("Initial continuity state should have after_scene=0")
        if continuity_state.recent_summaries:
            errors.append("Initial continuity state should have empty recent_summaries")

        # Plant/payoff validation (if cards have them)
        scenes_with_payoffs = [sc for sc in scene_cards if sc.payoffs_in_this_scene]
        for sc in scenes_with_payoffs:
            for payoff in sc.payoffs_in_this_scene:
                # Check that a corresponding plant exists in an earlier scene
                plant_found = any(
                    payoff in earlier_sc.plants_in_this_scene
                    for earlier_sc in scene_cards
                    if earlier_sc.scene_number < sc.scene_number
                )
                if not plant_found:
                    warnings.append(
                        f"Scene {sc.scene_number}: payoff '{payoff}' has no matching plant in earlier scenes"
                    )

        passed = len(errors) == 0
        return DeterministicValidationReport(
            passed=passed,
            word_count=0,
            errors=errors,
            warnings=warnings,
        )
