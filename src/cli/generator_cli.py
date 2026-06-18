# FILE: src/cli/generator_cli.py

import asyncio
from pathlib import Path
import traceback
from datetime import datetime

import click
from rich.progress import Progress

from src.extractor.async_frame_extractor import extract_frames_async

# --- Model + pipeline imports ---
from src.models.ball_detector import BallDetector
from src.tracking.ball.ball_tracker import BallTracker
from src.tracking.ball.ball_tracking_pipeline import BallTrackingPipeline

from src.video.clip_extractor import extract_clip
from src.video.reel_assembler import ReelAssembler

from src.rally.scoring.highlight_scoring_engine import HighlightScoringEngine


def debug(msg: str):
    """Simple debug print helper."""
    print(f"[DEBUG] {msg}", flush=True)


@click.command()
@click.option("--video", "-v", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="artifacts/highlight_reel.mp4")
@click.option("--max", "max_highlights", default=10, show_default=True)
def generate(video, output, max_highlights):
    """
    Full Story 4 → Story 6 highlight generator:
    video → frames → ball tracking → rallies → scoring → selection → clips → reel.
    """
    debug(f"===== STARTING HIGHLIGHT GENERATOR ===== --> [{datetime.now().isoformat()}]")
    debug(f"Input video: {video}")
    debug(f"Output path: {output}")
    debug(f"Max highlights: {max_highlights}")

    video_path = Path(video)
    output_path = Path(output)
    clips_dir = output_path.parent / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # --- configure model + tracker ---
    debug("Initializing BallDetector...")
    detector = BallDetector(model_path="runs/detect/train-4/weights/best.pt")
    debug("BallDetector initialized.")

    debug("Initializing BallTracker...")
    tracker = BallTracker()
    debug("BallTracker initialized.")

    # Simple dummy court definition
    dummy_court = type("Court", (), {
        "left": 0, "right": 1920, "top": 0, "bottom": 1080
    })()

    debug(f"Initializing BallTrackingPipeline... --> [{datetime.now().isoformat()}]")
    pipeline = BallTrackingPipeline(
        detector=detector,
        tracker=tracker,
        fps=30,
        court=dummy_court,
    )
    debug(f"BallTrackingPipeline initialized. --> [{datetime.now().isoformat()}]")

    scoring_engine = HighlightScoringEngine()

    with Progress() as progress:
        # 1. Extract frames (async) – used by downstream tools if needed
        debug(f"STEP 1: Extracting frames (async)... --> [{datetime.now().isoformat()}]")
        t1 = progress.add_task("Extracting frames…", total=1)
        try:
            frame_info = asyncio.run(extract_frames_async(video_path))
            debug(f"Frames extracted: {frame_info}")
        except Exception as e:
            debug("ERROR during frame extraction!")
            debug(traceback.format_exc())
            raise e
        progress.update(t1, advance=1)

        # 2. Full ball tracking pipeline
        debug(f"STEP 2: Running ball tracking pipeline... --> [{datetime.now().isoformat()}]")
        t2 = progress.add_task("Running ball tracking…", total=1)
        try:
            results = pipeline.process_video(video_path)
            debug(f"Ball tracking pipeline completed. --> [{datetime.now().isoformat()}]")
            debug(f"Raw track points: {len(results['raw_track'])}")
            debug(f"Trajectory points: {len(results['trajectory'])}")
            debug(f"Start events: {len(results['start_events'])}")
            debug(f"End events: {len(results['end_events'])}")
            debug(f"Rally segments: {len(results['rally_segments'])}")
            debug(f"Rally metadata entries: {len(results['rally_metadata'])}")
        except Exception as e:
            debug(f"ERROR during ball tracking pipeline! --> [{datetime.now().isoformat()}]")
            debug(traceback.format_exc())
            raise e
        progress.update(t2, advance=1)

        rally_metadata = results["rally_metadata"]

        # 3. Score ALL rallies
        debug(f"STEP 3: Scoring rallies... --> [{datetime.now().isoformat()}]")
        t3 = progress.add_task("Scoring rallies…", total=1)

        scored_rallies = []
        for r in rally_metadata:
            result = scoring_engine.score_rally(r)
            r.highlight_score = result["highlight_score"]
            r.highlight_reasons = result["reasons"]
            scored_rallies.append(r)

            debug(
                f"  Rally {r.rally_id}: score={r.highlight_score}, "
                f"duration={r.duration_seconds:.2f}s, hits={r.hit_count}, "
                f"reasons={r.highlight_reasons}"
            )

        scored_sorted = sorted(scored_rallies, key=lambda r: r.highlight_score, reverse=True)
        selected = scored_sorted[:max_highlights]

        debug(f"Selected {len(selected)} highlights (top {max_highlights}).")
        for i, r in enumerate(selected):
            debug(
                f"  Highlight {i+1}: score={r.highlight_score}, "
                f"start_frame={r.start_frame}, end_frame={r.end_frame}, "
                f"reasons={r.highlight_reasons}"
            )

        progress.update(t3, advance=1)

        # 4. Clip extraction
        debug(f"STEP 4: Extracting clips... --> [{datetime.now().isoformat()}]")
        t4 = progress.add_task("Extracting clips…", total=len(selected))
        clip_paths = []

        fps = pipeline.fps

        for i, r in enumerate(selected):
            try:
                clip_path = clips_dir / f"clip_{i+1}.mp4"
                debug(f"Extracting clip {i+1} → {clip_path}")

                start_time = r.start_frame / fps
                end_time = r.end_frame / fps

                extract_clip(
                    video_path,
                    start_time,
                    end_time,
                    clip_path,
                )
                clip_paths.append(clip_path)
            except Exception as e:
                debug(f"ERROR extracting clip {i+1}!")
                debug(traceback.format_exc())
                raise e
            progress.update(t4, advance=1)

        # 5. Reel assembly (no extra tracking here)
        debug(f"STEP 5: Assembling final highlight reel... --> [{datetime.now().isoformat()}]")
        t5 = progress.add_task("Assembling final reel…", total=1)
        try:
            assembler = ReelAssembler()
            assembler.assemble_reel(clip_paths, output_path)
            debug(f"Reel assembled successfully at: {output_path}")
        except Exception as e:
            debug("ERROR during reel assembly!")
            debug(traceback.format_exc())
            raise e
        progress.update(t5, advance=1)

    debug(f"===== HIGHLIGHT GENERATION COMPLETE ===== --> [{datetime.now().isoformat()}]")
    click.echo(f"🎉 Highlight reel created → {output_path}")


if __name__ == "__main__":
    generate()
