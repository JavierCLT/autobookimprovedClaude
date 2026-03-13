"""Novel Factory CLI — improved pipeline for contest-grade manuscript generation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


def _build_pipeline(project_slug: str):
    """Configures and returns a NovelPipeline instance."""
    from novel_factory.config import configure_logging, load_config
    from novel_factory.llm import AnthropicClient
    from novel_factory.pipeline import NovelPipeline
    from novel_factory.storage import RunStorage

    configure_logging()
    config = load_config()
    llm = AnthropicClient(config)
    storage = RunStorage(config, project_slug)
    return NovelPipeline(config, llm, storage)


def _require_story_input(
    synopsis: Path | None,
    intake: Path | None,
) -> tuple[str, "BookIntake | None"]:
    """Validates and loads story input files."""
    from novel_factory.intake import parse_book_intake
    from novel_factory.schemas import BookIntake

    book_intake: BookIntake | None = None
    synopsis_text: str = ""

    if intake:
        raw = intake.read_text(encoding="utf-8")
        book_intake = parse_book_intake(raw)
        synopsis_text = book_intake.fields.get("synopsis", "")

    if synopsis:
        synopsis_text = synopsis.read_text(encoding="utf-8")

    if not synopsis_text.strip():
        console.print("[red]Error: No synopsis provided. Use --synopsis or include synopsis in --intake.[/red]")
        raise typer.Exit(1)

    return synopsis_text, book_intake


@app.command()
def run_project(
    project_slug: str = typer.Argument(..., help="Project identifier (used as directory name)"),
    synopsis: Path | None = typer.Option(None, "--synopsis", "-s", help="Path to synopsis file"),
    intake: Path | None = typer.Option(None, "--intake", "-i", help="Path to filled intake template"),
) -> None:
    """Run the complete improved pipeline: planning through final polish."""
    synopsis_text, book_intake = _require_story_input(synopsis, intake)
    pipeline = _build_pipeline(project_slug)
    pipeline.run_full_pipeline(synopsis=synopsis_text, book_intake=book_intake)


@app.command()
def bootstrap(
    project_slug: str = typer.Argument(..., help="Project identifier"),
    synopsis: Path | None = typer.Option(None, "--synopsis", "-s"),
    intake: Path | None = typer.Option(None, "--intake", "-i"),
) -> None:
    """Generate only planning artifacts (voice, spec, outline, scene cards)."""
    synopsis_text, book_intake = _require_story_input(synopsis, intake)
    pipeline = _build_pipeline(project_slug)
    pipeline.bootstrap(synopsis=synopsis_text, book_intake=book_intake)


@app.command()
def draft_scene(
    project_slug: str = typer.Argument(..., help="Project identifier"),
    scene: int = typer.Argument(..., help="Scene number (1-based)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite approved scene"),
) -> None:
    """Draft a single scene from existing planning artifacts."""
    pipeline = _build_pipeline(project_slug)
    pipeline.draft_single_scene(scene_number=scene, force=force)


@app.command()
def global_qa(
    project_slug: str = typer.Argument(..., help="Project identifier"),
    intake: Path | None = typer.Option(None, "--intake", "-i"),
) -> None:
    """Run global QA on an existing manuscript."""
    from novel_factory.intake import parse_book_intake
    book_intake = None
    if intake:
        book_intake = parse_book_intake(intake.read_text(encoding="utf-8"))

    pipeline = _build_pipeline(project_slug)
    report = pipeline.run_global_qa(book_intake=book_intake)
    status = "PASS" if report.passed else "FAIL"
    console.print(f"\nGlobal QA: {status}")
    console.print(f"  Overall: {report.overall_score}/5")
    console.print(f"  AI smell: {report.ai_smell_score}/5")
    console.print(f"  Prose: {report.prose_quality_score}/5")
    if report.repair_priorities:
        console.print("\nRepair priorities:")
        for p in report.repair_priorities:
            console.print(f"  - {p}")


@app.command()
def repair_project(
    project_slug: str = typer.Argument(..., help="Project identifier"),
    intake: Path | None = typer.Option(None, "--intake", "-i"),
) -> None:
    """Run repair cycles on an existing manuscript based on QA feedback."""
    from novel_factory.intake import parse_book_intake
    from novel_factory.schemas import GlobalQaReport

    book_intake = None
    if intake:
        book_intake = parse_book_intake(intake.read_text(encoding="utf-8"))

    pipeline = _build_pipeline(project_slug)

    # Load existing QA
    from novel_factory.schemas import Outline, StorySpec, ContinuityState, SceneCard, VoiceDNA
    story_spec = pipeline.storage.load_model(pipeline.storage.story_spec_path, StorySpec)
    outline = pipeline.storage.load_model(pipeline.storage.outline_path, Outline)
    scene_cards = pipeline.storage.load_model_list(pipeline.storage.scene_cards_path, SceneCard)
    continuity = pipeline.storage.load_model(pipeline.storage.continuity_path, ContinuityState)
    global_qa = pipeline.storage.load_model(pipeline.storage.global_qa_path, GlobalQaReport)

    voice_dna = None
    if pipeline.storage.exists(pipeline.storage.voice_dna_path):
        voice_dna = pipeline.storage.load_model(pipeline.storage.voice_dna_path, VoiceDNA)

    pipeline._phase_repair(
        story_spec, outline, scene_cards, continuity,
        global_qa, voice_dna, book_intake,
    )
    console.print("[green]Repair complete.[/green]")


if __name__ == "__main__":
    app()
